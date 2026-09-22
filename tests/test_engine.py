import numpy as np
import pytest

from safe.engine import MIN_HISTORY, PageHinkley, SensorDrift

RNG = np.random.default_rng(7)
SENSOR = "TEST_SENSOR"


def make_engine(**kwargs):
    """Engine with a shared alert-capture list; alerts also land in engine.alerts."""
    kwargs.setdefault("cooldown_seconds", 0)  # tests want every alert
    return SensorDrift(**kwargs)


def feed(engine, values, metric="pm1_0", start_ts=1_700_000_000, step=60):
    for i, v in enumerate(values):
        engine.data_processing(SENSOR, {
            "unix_timestamp": start_ts + i * step,
            "str_timestamp": f"t{i}",
            metric: v,
        })


def alert_types(engine):
    return [a["alert"] for _, a, _ in engine.alerts]


class TestHardBounds:
    def test_impossible_value_alerts_and_is_excluded(self):
        engine = make_engine()
        feed(engine, [10.0] * 5 + [-5.0] + [10.0] * 5)  # pm1_0 cannot be negative
        assert "hard-bounds-violation" in alert_types(engine)
        state = engine._states[SENSOR]["pm1_0"]
        assert -5.0 not in state.buffer
        assert len(state.buffer) == 10

    @pytest.mark.parametrize("pm_bin", ["pm0_1", "pm0_3", "pm0_5", "pm2_5", "pm5_0", "pm10_0"])
    def test_all_pm_bins_have_bounds(self, pm_bin):
        # The 1-second dataset streams every IPS7100 bin, not just pm1_0
        engine = make_engine()
        feed(engine, [10.0, -5.0], metric=pm_bin)
        assert "hard-bounds-violation" in alert_types(engine)

    def test_unknown_metric_has_no_bounds(self):
        engine = make_engine()
        feed(engine, [1e9], metric="someUnknownMetric")
        assert engine.alerts == []


class TestRobustZScore:
    def test_single_spike_flagged_and_excluded(self):
        engine = make_engine(enable_page_hinkley=False)
        baseline = list(RNG.normal(10, 0.5, MIN_HISTORY + 20))
        feed(engine, baseline + [50.0])
        assert "z-score-outlier" in alert_types(engine)
        assert 50.0 not in engine._states[SENSOR]["pm1_0"].buffer

    def test_normal_variation_not_flagged(self):
        engine = make_engine(enable_page_hinkley=False)
        feed(engine, RNG.normal(10, 1.0, 150))
        assert "z-score-outlier" not in alert_types(engine)

    def test_robust_to_outlier_burst_masking(self):
        # A mean/std z-score inflates its own std after a burst of outliers and
        # then misses the next one (masking). The median/MAD baseline must not:
        # after 9 spikes (below the step-change run of 10), a 10th spike-sized
        # value must STILL be flagged.
        engine = make_engine(enable_page_hinkley=False)
        baseline = list(RNG.normal(10, 0.5, 60))
        feed(engine, baseline)
        engine.alerts.clear()
        # 9 spikes interleaved with good values so the consecutive counter resets
        burst = []
        for _ in range(9):
            burst += [60.0, 10.0]
        feed(engine, burst + [60.0], start_ts=1_800_000_000)
        spikes_flagged = alert_types(engine).count("z-score-outlier")
        assert spikes_flagged == 10  # every spike caught, none masked


class TestStepChange:
    def test_persistent_level_shift_reseeds_buffer(self):
        engine = make_engine(enable_page_hinkley=False)
        feed(engine, list(RNG.normal(10, 0.5, 60)) + [100.0] * 10)
        assert "Step-Change Detected" in alert_types(engine)
        state = engine._states[SENSOR]["pm1_0"]
        # Buffer reseeded at the new level
        assert list(state.buffer) == [100.0] * 10
        assert state.eval_count == 10

    def test_tracking_resumes_after_step(self):
        engine = make_engine(enable_page_hinkley=False)
        # Step values carry realistic noise (identical values would be a
        # quantization artifact, which refresh_baseline guards separately)
        step = list(RNG.normal(100, 0.5, 10))
        feed(engine, list(RNG.normal(10, 0.5, 60)) + step)
        engine.alerts.clear()
        # New readings around the new level are normal, not outliers
        feed(engine, list(RNG.normal(100, 0.5, 40)), start_ts=1_900_000_000)
        assert "z-score-outlier" not in alert_types(engine)


class TestPageHinkley:
    def test_detector_alarms_on_sustained_shift(self):
        ph = PageHinkley(delta=0.25, lam=18.0)
        assert all(ph.update(r) is None for r in RNG.normal(0, 1, 500))
        # sustained 1-sigma upward shift
        results = [ph.update(r) for r in RNG.normal(1.0, 1.0, 100)]
        assert "up" in results

    def test_detector_direction_down(self):
        ph = PageHinkley(delta=0.25, lam=18.0)
        results = [ph.update(r) for r in RNG.normal(-1.0, 1.0, 100)]
        assert "down" in results

    def test_engine_catches_subthreshold_drift(self):
        # A 2-sigma level shift never trips the 3.5-sigma z-score, but is a
        # real drift; Page-Hinkley must catch it well within one window.
        engine = make_engine(window_size=500, enable_page_hinkley=True)
        feed(engine, list(RNG.normal(10, 1.0, 60)) + list(RNG.normal(12, 1.0, 60)))
        assert "Sustained Mean Shift (Page-Hinkley)" in alert_types(engine)

    def test_no_false_alarm_on_stationary_noise(self):
        engine = make_engine(window_size=10_000, enable_page_hinkley=True)
        feed(engine, RNG.normal(10, 1.0, 2000))
        assert "Sustained Mean Shift (Page-Hinkley)" not in alert_types(engine)


class TestWindowedDrift:
    def test_drift_detected_across_window_halves(self):
        engine = make_engine(window_size=200, enable_page_hinkley=False,
                             z_threshold=100.0)  # disable outlier layer
        # First half at 10, second half at 11 (1-sigma shift, iid noise)
        feed(engine, list(RNG.normal(10, 1.0, 100)) + list(RNG.normal(11, 1.0, 100)))
        assert "Sensor Drift Detected" in alert_types(engine)

    def test_no_drift_on_stationary_data(self):
        engine = make_engine(window_size=200, enable_page_hinkley=False)
        feed(engine, RNG.normal(10, 1.0, 600))
        assert "Sensor Drift Detected" not in alert_types(engine)

    def test_flat_step_change_between_windows(self):
        # 100 identical readings would otherwise be flagged as a frozen sensor
        engine = make_engine(window_size=200, enable_page_hinkley=False,
                             z_threshold=1e9, freeze_detection=False)
        feed(engine, [5.0] * 100 + [7.0] * 100)
        assert "Sensor Drift Detected Step-Change" in alert_types(engine)


class TestScaleFloor:
    def test_small_move_off_flat_baseline_is_not_outlier(self):
        # A held temperature collapses MAD/IQR/std to 0; a 0.05 °C move must
        # not score as an enormous z-score.
        engine = make_engine(freeze_detection=False)
        feed(engine, [20.0] * 40 + [20.05] * 3, metric="temperature")
        assert "z-score-outlier" not in alert_types(engine)

    def test_large_move_off_flat_baseline_still_flagged(self):
        engine = make_engine(freeze_detection=False)
        feed(engine, [20.0] * 40 + [22.0], metric="temperature")
        assert "z-score-outlier" in alert_types(engine)

    def test_ordinary_readings_after_flat_stretch_are_not_step_change(self):
        engine = make_engine(freeze_detection=False)
        feed(engine, [20.0] * 40 + list(20.0 + RNG.normal(0, 0.05, 20)), metric="temperature")
        assert "Step-Change Detected" not in alert_types(engine)


class TestFreeze:
    def test_repeated_value_for_an_hour_is_frozen(self):
        engine = make_engine()
        noisy = list(RNG.normal(20, 0.5, 40))
        feed(engine, noisy + [21.3] * 20, metric="temperature", step=300)
        assert alert_types(engine).count("Frozen Sensor Detected") == 1
        state = engine._states[SENSOR]["temperature"]
        # At 5-min cadence the hour is reached on the 13th reading; only the
        # 12 before confirmation reached history
        assert list(state.buffer).count(21.3) == 12

    def test_short_repeat_is_not_frozen(self):
        engine = make_engine()
        # 20 readings but only 19 minutes: under the one-hour minimum
        feed(engine, list(RNG.normal(20, 0.5, 40)) + [21.3] * 20, metric="temperature", step=60)
        assert "Frozen Sensor Detected" not in alert_types(engine)

    def test_recovery_restarts_tracking_without_step_change(self):
        engine = make_engine()
        feed(engine, list(RNG.normal(20, 0.5, 40)) + [21.3] * 30, metric="temperature", step=300)
        engine.alerts.clear()
        # Live readings return far from the stale pre-freeze level
        feed(engine, list(RNG.normal(30, 0.5, 20)), metric="temperature",
             start_ts=1_800_000_000, step=300)
        types = alert_types(engine)
        assert types.count("Frozen Sensor Recovered") == 1
        assert "Step-Change Detected" not in types
        assert "z-score-outlier" not in types
        state = engine._states[SENSOR]["temperature"]
        assert len(state.buffer) == 20 and not state.frozen

    def test_zero_count_bins_are_not_frozen(self):
        engine = make_engine()
        feed(engine, [0.0] * 30, metric="pc10_0", step=300)
        assert "Frozen Sensor Detected" not in alert_types(engine)

    def test_can_be_disabled(self):
        engine = make_engine(freeze_detection=False)
        feed(engine, [21.3] * 30, metric="temperature", step=300)
        assert "Frozen Sensor Detected" not in alert_types(engine)
        assert len(engine._states[SENSOR]["temperature"].buffer) == 30


class TestCooldown:
    def test_repeat_alerts_suppressed_within_cooldown(self):
        engine = make_engine(cooldown_seconds=1800, enable_page_hinkley=False)
        baseline = list(RNG.normal(10, 0.5, 60))
        # Two spikes 60 s apart -> second alert suppressed
        feed(engine, baseline + [50.0, 10.0, 50.0])
        assert alert_types(engine).count("z-score-outlier") == 1

    def test_alert_fires_again_after_cooldown(self):
        engine = make_engine(cooldown_seconds=1800, enable_page_hinkley=False)
        baseline = list(RNG.normal(10, 0.5, 60))
        feed(engine, baseline + [50.0], step=60)
        # next spike a day later, far past the 1800 s cooldown
        engine.data_processing(SENSOR, {
            "unix_timestamp": 1_700_000_000 + 3600 * 24, "str_timestamp": "later", "pm1_0": 50.0})
        assert alert_types(engine).count("z-score-outlier") == 2


class TestPlumbing:
    def test_on_alert_callback_receives_alerts(self):
        received = []
        engine = SensorDrift(cooldown_seconds=0,
                             on_alert=lambda s, a, t: received.append((s, a, t)))
        feed(engine, [10.0] * 5 + [-5.0])
        assert len(received) == 1
        assert received[0][0] == SENSOR
        assert received[0][1]["alert"] == "hard-bounds-violation"

    def test_non_numeric_and_nan_values_skipped(self):
        engine = make_engine()
        engine.data_processing(SENSOR, {
            "unix_timestamp": 1_700_000_000, "str_timestamp": "t",
            "pm1_0": "not-a-number", "temperature": float("nan")})
        assert engine._states.get(SENSOR, {}) == {}

    def test_datetime_fallback_path(self):
        engine = make_engine()
        engine.data_processing(SENSOR, {"dateTime": "2026-01-01 00:00:00", "pm1_0": 10.0})
        assert len(engine._states[SENSOR]["pm1_0"].buffer) == 1

    def test_multiple_sensors_isolated(self):
        engine = make_engine()
        engine.data_processing("A", {"unix_timestamp": 1, "str_timestamp": "t", "pm1_0": 10.0})
        engine.data_processing("B", {"unix_timestamp": 1, "str_timestamp": "t", "pm1_0": 99.0})
        assert list(engine._states["A"]["pm1_0"].buffer) == [10.0]
        assert list(engine._states["B"]["pm1_0"].buffer) == [99.0]
