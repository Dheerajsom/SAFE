from dataclasses import replace
import json
import math

import numpy as np
import pytest

from safe import MetricProfile, PageHinkley, ProfileRegistry, SensorHealth, SensorRules
from safe.baseline import SeasonalBaseline
from safe.incidents import IncidentManager
from safe.profiles import metric_profile


def short_profile(**changes):
    return replace(metric_profile("temperature"), expected_interval_seconds=60,
                   warmup_duration_seconds=3600, seasonal_period_seconds=3600,
                   seasonal_bins=12, minimum_samples=30, window_seconds=3600,
                   completeness_window_seconds=3600, evaluation_interval_seconds=600,
                   recovery_seconds=180, freeze_duration_seconds=600, **changes)


def engine_for(profile=None, **kwargs):
    profile = profile or short_profile()
    return SensorHealth(profiles=ProfileRegistry(overrides=[dict(match={"metric": "temperature"},
                                                                  profile=profile.to_dict())]), **kwargs)


def feed(engine, values, start=0, dt=60, sensor="s", metric="temperature"):
    for index, value in enumerate(values):
        engine.data_processing(sensor, {"unix_timestamp": start + index * dt, metric: value})


def warm(engine, n=120):
    feed(engine, np.random.default_rng(99).normal(20, .15, n))
    assert engine._states[("s", "temperature")].baseline.ready


def test_offset_correlates_outliers_escalates_once_and_recovers():
    notices, lifecycle = [], []
    engine = engine_for(on_notification=notices.append, on_event=lambda a, e: lifecycle.append((a, e)))
    warm(engine)
    feed(engine, np.random.default_rng(2).normal(28, .1, 40), start=7200)
    change = [e for e in engine.events if e["family"] == "change"]
    assert len(change) == 1
    assert change[0]["category"] == "abrupt_shift"
    assert change[0]["detection_count"] >= 40
    assert len(notices) == 1
    assert {a for a, _ in lifecycle} >= {"opened", "escalated", "updated"}
    feed(engine, [20.01, 19.99], start=9600)
    assert engine.incidents.active[("s", "temperature", "change")].status == "recovering"
    feed(engine, [20.01, 19.99, 20.02, 19.98, 20], start=9720)
    assert not engine.incidents.active
    assert engine.events[0]["status"] == "closed"
    assert engine._states[("s", "temperature")].mode == "MONITORING"


def test_tick_detects_never_started_sensor_and_silence_does_not_recover():
    engine = engine_for()
    engine.register("s", "temperature", 0)
    engine.tick(180)
    event = engine.events[0]
    assert event["category"] == "missing_data"
    engine.tick(10000)
    assert len(engine.events) == 1 and engine.events[0]["status"] == "open"
    with pytest.raises(ValueError, match="monotonic"):
        engine.tick(9999)


def test_old_data_after_clock_tick_is_delayed_and_cannot_recover_silence():
    engine = engine_for()
    feed(engine, [20])
    engine.tick(10000)
    feed(engine, [20.1], start=60)
    assert engine._states[("s", "temperature")].last_at == 0
    assert engine.events[0]["category"] == "delayed_data"


def test_freeze_from_startup_quantization_and_legitimate_zero_pm():
    engine = engine_for()
    feed(engine, [20 + .0001 * (-1)**i for i in range(20)])
    assert any(e["category"] == "sensor_freeze" for e in engine.events)
    assert not engine._states[("s", "temperature")].baseline.ready
    pm = SensorHealth()
    feed(pm, [0] * 400, metric="pm2_5", dt=300)
    assert not any(e["category"] == "sensor_freeze" for e in pm.events)


def test_zero_particle_counts_are_not_frozen_but_stuck_counts_are():
    # Large-particle count bins legitimately read exactly zero for hours in clean air.
    clean = SensorHealth()
    feed(clean, [3, 0, 5, 0] + [0] * 400, metric="pc5_0", dt=300)
    assert not any(e["category"] == "sensor_freeze" for e in clean.events)
    stuck = SensorHealth()
    feed(stuck, [3, 0, 5, 0] + [7] * 400, metric="pc5_0", dt=300)
    assert any(e["category"] == "sensor_freeze" for e in stuck.events)


def test_page_hinkley_alarms_on_sustained_shift_in_either_direction():
    rng = np.random.default_rng(0)
    up = PageHinkley(delta=0.25, lam=18.0)
    assert all(up.update(r) is None for r in rng.normal(0, 1, 500))
    assert "up" in [up.update(r) for r in rng.normal(1.0, 1.0, 100)]
    down = PageHinkley(delta=0.25, lam=18.0)
    assert "down" in [down.update(r) for r in rng.normal(-1.0, 1.0, 100)]
    with pytest.raises(ValueError):
        PageHinkley(lam=0)
    with pytest.raises(ValueError):
        down.update(float("nan"))


def test_freeze_run_is_anchored_not_a_chain_of_small_moves():
    engine = engine_for(short_profile(freeze_tolerance=.01))
    feed(engine, [20 + .009 * i for i in range(80)])
    assert not any(e["category"] == "sensor_freeze" for e in engine.events)


def test_gap_breaks_freeze_and_step_runs():
    engine = engine_for()
    feed(engine, [20] * 10)
    feed(engine, [20] * 5, start=10000)
    assert not any(e["category"] == "sensor_freeze" for e in engine.events)
    assert engine._states[("s", "temperature")].freeze_count == 5


def test_duplicate_disorder_and_partial_fields_do_not_contaminate_history():
    engine = engine_for()
    feed(engine, [20], start=1000)
    engine.data_processing("s", {"unix_timestamp": 1000, "humidity": 50})
    feed(engine, [30], start=1000)
    feed(engine, [40], start=999)
    state = engine._states[("s", "temperature")]
    assert list(state.warmup) == [(1000, 20)]
    assert state.last_value == 20
    assert len(engine.events) == 1
    assert {"duplicate_timestamp", "timestamp_disorder"} <= engine.events[0]["evidence"].keys()


@pytest.mark.parametrize("value", [150, -100, float("nan"), float("inf")])
def test_invalid_values_emit_finite_json_and_never_enter_model(value):
    engine = engine_for()
    feed(engine, [value])
    assert engine.events[0]["category"] == "invalid_measurement"
    assert not engine._states[("s", "temperature")].warmup
    json.dumps(engine.snapshot(), allow_nan=False)


def test_warmup_requires_coverage_and_rejects_initial_contamination():
    engine = engine_for()
    feed(engine, [20 + .01 * (i % 3) for i in range(30)], dt=.01)
    assert not engine._states[("s", "temperature")].baseline.ready
    contaminated = engine_for()
    feed(contaminated, [20 + .1 * (i % 3) for i in range(10)] + [60] * 8)
    state = contaminated._states[("s", "temperature")]
    assert state.warmup_restarts == 1
    assert not state.warmup
    assert not any(e["category"] in {"abrupt_shift", "gradual_degradation"} for e in contaminated.events)


def test_bad_initial_samples_restart_warmup():
    engine = engine_for()
    feed(engine, [20, 20.1, 20.2, 150, 150, 150])
    assert engine._states[("s", "temperature")].warmup_restarts == 1
    assert not engine._states[("s", "temperature")].warmup


def test_cadence_and_completeness_count_slots_not_bursts():
    engine = engine_for()
    feed(engine, [20 + .1 * (i % 4) for i in range(40)], dt=120)
    assert any("cadence_degradation" in e["evidence"] for e in engine.events)
    assert any("completeness_loss" in e["evidence"] for e in engine.events)


def test_fast_bursts_cannot_evict_completeness_slots():
    engine = engine_for(short_profile(max_samples=100))
    feed(engine, [20 + .1 * (i % 4) for i in range(61)])
    feed(engine, [20 + .1 * (i % 4) for i in range(200)], start=3601, dt=.01)
    state = engine._states[("s", "temperature")]
    assert len(state.arrivals) >= 60
    assert not any("completeness_loss" in e["evidence"] for e in engine.events)


def test_pm_order_status_dewpoint_and_simultaneous_jump_rules():
    rules = SensorRules(ordered_metrics=("pm1_0", "pm2_5", "pm10_0"), status_field="status",
                        dewpoint_field="dewpoint")
    engine = SensorHealth(sensors={"s": {"model": "test"}}, rules={"test": rules})
    engine.data_processing("s", {"unix_timestamp": 0, "temperature": 20, "dewpoint": 25,
                                "pm1_0": 10, "pm10_0": 5, "status": 3})
    assert any("pm_ordering" in e["evidence"] for e in engine.events)
    assert any("sensor_status" in e["evidence"] for e in engine.events)
    assert any("physical_relationship" in e["evidence"] for e in engine.events)
    other = SensorHealth()
    other.data_processing("s", {"unix_timestamp": 0, "temperature": 20, "humidity": 50, "pressure": 1000})
    other.data_processing("s", {"unix_timestamp": 300, "temperature": 30, "humidity": 60, "pressure": 1010})
    assert sum(e["category"] == "possible_restart" for e in other.events) == 3


def test_baseline_models_daily_shape_and_is_bounded():
    p = short_profile()
    baseline = SeasonalBaseline(p)
    sample = [(i * 60, 20 + 8 * math.sin(i * 2 * math.pi / 60)) for i in range(61)]
    assert baseline.fit(sample)
    assert abs(baseline.predict(4500)[0] - 28) < 1
    for i in range(61, 5000):
        baseline.learn(i * 60, 20 + 8 * math.sin(i * 2 * math.pi / 60))
    assert all(len(b) <= p.seasonal_cycles for b in baseline.bins)


def test_peer_consensus_separates_shared_episode_and_isolated_offset():
    notices = []
    engine = engine_for(on_notification=notices.append)
    for i in range(120):
        t = i * 60
        refs = [{"sensor": s, "value": 20, "timestamp": t} for s in ("a", "b")]
        engine.data_processing("s", {"unix_timestamp": t, "temperature": 20 + .05 * (-1)**i},
                               references={"temperature": refs})
    for i in range(20):
        t = 7200 + 60 * i
        refs = [{"sensor": s, "value": 28 + j * .01, "timestamp": t} for j, s in enumerate(("a", "b"))]
        engine.data_processing("s", {"unix_timestamp": t, "temperature": 28 + .05 * (-1)**i}, references={"temperature": refs})
    assert not notices
    for i in range(20, 40):
        t = 7200 + 60 * i
        refs = [{"sensor": s, "value": 28, "timestamp": t} for s in ("a", "b")]
        engine.data_processing("s", {"unix_timestamp": t, "temperature": 36 + .05 * (-1)**i}, references={"temperature": refs})
    assert len(notices) == 1 and notices[0]["category"] == "abrupt_shift"


def _feed_with_reference(engine, sensor_values, reference_values, start=0, dt=60):
    for index, (value, reference) in enumerate(zip(sensor_values, reference_values)):
        t = start + index * dt
        engine.data_processing("s", {"unix_timestamp": t, "temperature": value},
                               references={"temperature": [{"sensor": "ref", "timestamp": t,
                                                              "value": reference, "trusted": True}]})


def test_reference_drift_below_step_threshold_is_flagged():
    notices = []
    engine = engine_for(on_notification=notices.append)
    rng = np.random.default_rng(5)
    warm(engine)
    reference = 20 + rng.normal(0, .1, 600)
    # Offset learned while healthy, then a 1.5 C ramp: under the 2 C step
    # threshold, so only the reference-drift check can see it.
    drift = np.concatenate([np.zeros(100), np.linspace(0, 1.5, 500)])
    _feed_with_reference(engine, reference + 0.3 + drift + rng.normal(0, .1, 600), reference,
                         start=120 * 60)
    assert [n["category"] for n in notices] == ["gradual_degradation"]
    evidence = notices[0]["evidence"]["gradual_degradation"]
    assert evidence["evidence_source"] == "reference"
    assert abs(evidence["median_reference_residual"]) >= 0.5
    assert abs(evidence["reference_offset"] - 0.3) < 0.1


def test_sensor_tracking_reference_through_trend_is_not_drift():
    notices = []
    engine = engine_for(on_notification=notices.append)
    rng = np.random.default_rng(6)
    warm(engine)
    # A genuine 10 C ambient trend: a single sensor could not tell this from
    # drift, but agreement with the reference clears it.
    reference = 20 + np.concatenate([np.zeros(100), np.linspace(0, 10, 500)]) + rng.normal(0, .1, 600)
    _feed_with_reference(engine, reference + 0.3 + rng.normal(0, .1, 600), reference, start=120 * 60)
    assert notices == []
    assert engine._states[("s", "temperature")].reference_drift_evidence is None


def _pm_gain_run(sensor_gain, scatter=0.0, **changes):
    """A PM sensor reading `sensor_gain` x a trusted reference through a 6x episode.

    `scatter` adds independent percentage noise, like co-located sensors show.
    """
    profile = replace(short_profile(), hard_bounds=(0.0, 10000.0), residual_scale_floor=1,
                      step_min_effect=5, freeze_at_startup=False, freeze_tolerance=0,
                      reference_drift_tolerance=2, **{**dict(
                          reference_ratio_floor=metric_profile("pm2_5").reference_ratio_floor,
                          reference_drift_relative_tolerance=metric_profile(
                              "pm2_5").reference_drift_relative_tolerance), **changes})
    notices = []
    engine = SensorHealth(profiles=ProfileRegistry(overrides=[dict(match={"metric": "pm2_5"},
                                                                    profile=profile.to_dict())]),
                          on_notification=notices.append)
    rng = np.random.default_rng(11)
    reference = 10 * np.concatenate([np.ones(200), np.linspace(1, 6, 200), 6 * np.ones(200)])
    reference = reference * (1 + rng.normal(0, .02, reference.size))
    gain = np.concatenate([np.full(200, 1.06), np.asarray(sensor_gain, dtype=float) * np.ones(400)])
    gain = gain * (1 + rng.normal(0, scatter, gain.size))
    for index, (ref, g) in enumerate(zip(reference, gain)):
        t = index * 60
        engine.data_processing("s", {"unix_timestamp": t, "pm2_5": ref * g},
                               references={"pm2_5": [{"sensor": "ref", "timestamp": t,
                                                      "value": ref, "trusted": True}]})
    return [n["category"] for n in notices]


def test_pm_gain_difference_is_not_drift_during_an_episode():
    # A healthy sensor 6% above its co-located reference stays 6% above it at 6x
    # concentration; a learned constant difference would call that growing gap drift.
    assert _pm_gain_run(1.06, reference_ratio_floor=0, reference_drift_relative_tolerance=0) \
        == ["gradual_degradation"]
    assert _pm_gain_run(1.06) == []


def test_pm_gain_drift_against_reference_is_still_flagged():
    # At 6x concentration a gain drifting to 1.7 grows fast enough to read as a shift.
    categories = _pm_gain_run(np.linspace(1.06, 1.7, 400))
    assert categories and set(categories) <= {"gradual_degradation", "abrupt_shift"}


def test_pm_percentage_scatter_scales_the_residual_yardstick():
    # 5% independent scatter is ~3 ug/m3 at 60 ug/m3; judged by a scale learned at
    # 10 ug/m3 it looks like repeated outliers, judged in percent it is ordinary.
    assert _pm_gain_run(1.06, scatter=0.05, reference_ratio_floor=0,
                        reference_drift_relative_tolerance=0)
    assert _pm_gain_run(1.06, scatter=0.05) == []


def test_pm_step_against_reference_is_flagged_despite_scatter():
    categories = _pm_gain_run(1.06 * 1.5, scatter=0.05)
    assert categories and set(categories) <= {"gradual_degradation", "abrupt_shift"}


def test_stuck_sensor_raises_freeze_only_not_a_duplicate_shift_or_drift():
    notices = []
    engine = engine_for(on_notification=notices.append)
    rng = np.random.default_rng(8)
    warm(engine)
    # Learn the reference relationship, then the sensor sticks while the
    # reference climbs 3 C, then it recovers and tracks the reference again.
    reference = 20 + np.concatenate([np.zeros(100), np.linspace(0, 3, 120), np.full(180, 3)])
    reference = reference + rng.normal(0, .1, reference.size)
    sensor = reference + 0.3 + rng.normal(0, .1, reference.size)
    sensor[100:220] = 20.3
    _feed_with_reference(engine, sensor, reference, start=120 * 60)
    assert [n["category"] for n in notices] == ["sensor_freeze"]
    assert not any(e["family"] == "change" and e["severity"] != "info" for e in engine.events)


def test_reference_drift_state_survives_snapshot_and_old_snapshots_load():
    engine = engine_for()
    warm(engine)
    snapshot = engine.snapshot()
    restored = SensorHealth.from_snapshot(snapshot)
    assert restored._states[("s", "temperature")].reference_drift_confirmations == 0
    for item in snapshot["states"]:
        del item["state"]["reference_drift_confirmations"]
        del item["state"]["reference_drift_evidence"]
    legacy = SensorHealth.from_snapshot(snapshot)
    assert legacy._states[("s", "temperature")].reference_drift_evidence is None


def test_stale_duplicate_self_and_future_peers_are_excluded():
    engine = engine_for()
    p = short_profile()
    refs = [{"sensor": "s", "timestamp": 100, "value": 20},
            {"sensor": "a", "timestamp": 101, "value": 20},
            {"sensor": "b", "timestamp": 0, "value": 20}]
    assert engine._peer("s", p, 100, refs) is None
    one = {"sensor": "a", "timestamp": 100, "value": 20}
    assert engine._peer("s", p, 100, [one, one]) is None
    assert engine._peer("s", p, 100, [{**one, "trusted": True}]) == 20


def test_page_hinkley_requires_stationarity_and_resets_on_reference_switch():
    with pytest.raises(ValueError, match="stationary"):
        MetricProfile(enable_page_hinkley=True)
    engine = engine_for(short_profile(stationary_residuals=True, enable_page_hinkley=True,
                                     seasonal_rate_per_day=0))
    warm(engine)
    engine._states[("s", "temperature")].ph._m_up = 10
    engine.data_processing("s", {"unix_timestamp": 7200, "temperature": 20},
                           references={"temperature": [{"sensor": "ref", "timestamp": 7200,
                                                          "value": 20, "trusted": True}]})
    assert abs(engine._states[("s", "temperature")].ph._m_up) < 2


def test_restart_equivalence_and_no_duplicate_notification(tmp_path):
    continuous, restarted = engine_for(), engine_for()
    rng = np.random.default_rng(4)
    values = np.r_[rng.normal(20, .1, 120), rng.normal(28, .1, 50), rng.normal(20, .1, 20)]
    for i, value in enumerate(values):
        if i in {121, 135, 165, 172}:
            restarted.save_state(tmp_path / "state.json")
            restarted = SensorHealth.load_state(tmp_path / "state.json")
        feed(continuous, [value], start=i * 60)
        feed(restarted, [value], start=i * 60)
    assert continuous.snapshot() == restarted.snapshot()
    assert continuous.incidents.total_notifications == 1
    detached = continuous.snapshot()
    detached["states"][0]["state"]["last_value"] = 999
    assert continuous._states[("s", "temperature")].last_value != 999


def test_checksum_schema_and_failed_save_preserve_previous_state(tmp_path, monkeypatch):
    engine = engine_for()
    feed(engine, [20])
    path = tmp_path / "state.json"
    engine.save_state(path)
    original = path.read_bytes()
    monkeypatch.setattr("safe.health.os.replace", lambda *_: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError):
        engine.save_state(path)
    assert path.read_bytes() == original
    data = json.loads(original)
    data["payload"]["clock"] = 42
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        SensorHealth.load_state(path)
    snapshot = engine.snapshot()
    snapshot["schema_version"] = 999
    with pytest.raises(ValueError, match="schema"):
        SensorHealth.from_snapshot(snapshot)


def test_retention_capacity_and_retiring_series():
    engine = engine_for(history_limit=2, max_series=1)
    with pytest.raises(ValueError, match="max_series"):
        engine.data_processing("s", {"unix_timestamp": 0, "temperature": 20, "humidity": 50})
    assert not engine._states
    feed(engine, [150])
    with pytest.raises(ValueError, match="active"):
        engine.retire("s", "temperature")
    feed(engine, [20.01, 19.99, 20.01, 19.99, 20.01], start=60)
    engine.retire("s", "temperature")
    assert not engine._states
    manager = IncidentManager(history_limit=2, notification_interval_seconds=0)
    for i in range(10):
        manager.observe("s", "x", "invalid_measurement", "validity", i * 10, "critical", 1, {}, {})
        manager.recover("s", "x", i * 10 + 1, set(), 1, 2)
        manager.recover("s", "x", i * 10 + 2, set(), 1, 2)
    assert len(manager.history) == 2 and manager.total_opened == 10


def test_notification_throttle_and_callback_failure_isolation():
    notices = []
    manager = IncidentManager(notification_interval_seconds=100, on_notification=notices.append,
                              on_event=lambda *_: (_ for _ in ()).throw(RuntimeError("handler")))
    manager.observe("s", "x", "a", "first", 0, "warning", .5, {}, {})
    manager.observe("s", "x", "b", "second", 10, "critical", 1, {}, {})
    assert len(notices) == 1
    manager.observe("s", "x", "b", "second", 101, "critical", 1, {}, {})
    assert len(notices) == 2
    manager.observe("s", "x", "b", "second", 1000, "critical", 1, {}, {})
    assert len(notices) == 2


@pytest.mark.parametrize("change", [dict(expected_interval_seconds=0), dict(expected_interval_seconds=float("nan")),
    dict(minimum_samples=True), dict(freeze_tolerance=-1), dict(hard_bounds=[20, 10]),
    dict(family_alpha=1), dict(max_samples=50), dict(unknown=1)])
def test_invalid_profiles_fail(change):
    with pytest.raises((ValueError, TypeError)):
        MetricProfile(**change)


def test_profile_specificity_and_cadence_memory_budget():
    custom = short_profile().to_dict()
    registry = ProfileRegistry(overrides=[dict(match={"metric": "temperature"}, profile=custom)])
    assert registry.resolve("s", "temperature").expected_interval_seconds == 60
    assert registry.resolve("s", "humidity").expected_interval_seconds == 300
    assert metric_profile("temperature", expected_interval_seconds=1).max_samples >= 86402
    with pytest.raises(ValueError):
        ProfileRegistry(overrides=[dict(match={"typo": "s"}, profile=custom)])


def test_future_timestamp_cannot_advance_other_series_clock():
    engine = engine_for()
    feed(engine, [20])
    engine.data_processing("s", {"unix_timestamp": 100000, "temperature": 20.1}, received_at=60)
    assert engine._clock == 60
    assert engine._states[("s", "temperature")].last_at == 0
    assert engine.events[0]["category"] == "future_timestamp"
    with pytest.raises(ValueError):
        engine.data_processing("s", {"unix_timestamp": True, "temperature": 20})


def test_forty_day_daily_restarts_match_uninterrupted_state_and_bound_retention():
    profile = replace(metric_profile("temperature"), expected_interval_seconds=900, max_samples=512)
    continuous, restarted = engine_for(profile, history_limit=3), engine_for(profile, history_limit=3)
    rng = np.random.default_rng(987)
    sizes = []
    for i in range(40 * 96):
        value = 20 + 4 * math.sin(2 * math.pi * i / 96) + float(rng.normal(0, .15))
        if 5 * 96 <= i < 10 * 96:
            value += 8
        if i % 96 == 0 and i:
            restarted = SensorHealth.from_snapshot(restarted.snapshot())
        feed(continuous, [value], start=i * 900)
        feed(restarted, [value], start=i * 900)
        if i % 96 == 95 and i > 20 * 96:
            sizes.append(len(json.dumps(restarted.snapshot())))
    assert restarted.snapshot() == continuous.snapshot()
    assert len(restarted.incidents.history) <= 3
    state = restarted._states[("s", "temperature")]
    assert len(state.arrivals) <= 98 and len(state.residuals) <= 26
    assert all(len(b) <= 14 for b in state.baseline.bins)
    assert max(sizes) < 150000
    assert max(sizes[-10:]) - min(sizes[-10:]) < 10000


def test_healthy_pressure_diurnal_and_full_year_temperature_have_no_notifications():
    pressure = SensorHealth()
    rng = np.random.default_rng(4921)
    for i in range(7 * 288):
        pressure.data_processing("s", {"unix_timestamp": i * 300,
            "pressure": 1010 + 4 * math.sin(2 * math.pi * i / 288) + float(rng.normal(0, .2))})
    assert pressure.incidents.total_notifications == 0
    profile = replace(metric_profile("temperature"), expected_interval_seconds=3600,
                      seasonal_bins=24, minimum_samples=24, window_seconds=86400)
    annual = engine_for(profile)
    for i in range(365 * 24):
        value = (20 + 12 * math.sin(2 * math.pi * i / (365 * 24))
                 + 6 * math.sin(2 * math.pi * i / 24) + float(rng.normal(0, .1)))
        feed(annual, [value], start=i * 3600)
    assert annual._states[("s", "temperature")].baseline.ready
    assert annual.incidents.total_notifications == 0


def _pm_bins(rng, n, total):
    """Cumulative IPS7100 bins (pm0_1 ... pm10_0) around a pm10_0 level of `total`."""
    shares = np.array([.04, .22, .42, .58, .78, .91, 1.0])
    level = total * (1 + rng.normal(0, .05, (n, 1)))
    return np.round(np.maximum(level * shares * (1 + rng.normal(0, .03, (n, 7))), 0), 3)


def _stuck_zero_run(bins, dead=None, peers=None):
    """Replay bins at 5-minute cadence for an IPS7100; zero `dead` for the last 36 readings."""
    engine = SensorHealth(sensors={"s": {"model": "IPS7100"}})
    metrics = ("pm0_1", "pm0_3", "pm0_5", "pm1_0", "pm2_5", "pm5_0", "pm10_0")
    for index, row in enumerate(bins):
        reading = dict(zip(metrics, row))
        if dead is not None and index >= len(bins) - 36:
            reading[dead] = 0.0
        refs = None
        if peers is not None:
            refs = {m: [{"sensor": f"p{k}", "timestamp": index * 300, "value": peers[m]} for k in (1, 2)]
                    for m in metrics}
        engine.data_processing("s", {"unix_timestamp": index * 300, **reading}, references=refs)
    return [e for e in engine.events if e["category"] == "sensor_freeze"]


def test_dead_pm_channel_at_zero_is_stuck_when_a_smaller_bin_reads_high():
    freezes = _stuck_zero_run(_pm_bins(np.random.default_rng(1), 150, 20), dead="pm2_5")
    assert [e["metric"] for e in freezes] == ["pm2_5"]
    evidence = freezes[0]["evidence"]["sensor_freeze"]
    assert evidence["stuck_at_zero"] and "smaller_bin" in evidence["contradiction_evidence"]


def test_smallest_bin_dead_at_zero_is_stuck_from_its_learned_share_of_the_next_bin():
    # pm0_1 has no smaller bin; its usual share of pm0_3 predicts about 4 ug/m3.
    freezes = _stuck_zero_run(_pm_bins(np.random.default_rng(2), 150, 100), dead="pm0_1")
    assert [e["metric"] for e in freezes] == ["pm0_1"]
    assert "larger_bin_share" in freezes[0]["evidence"]["sensor_freeze"]["contradiction_evidence"]


def test_clean_air_zeros_are_not_stuck():
    rng = np.random.default_rng(3)
    # Hours of genuinely clean air: every bin near zero, small bins exactly zero.
    bins = np.vstack([_pm_bins(rng, 100, 20), _pm_bins(rng, 60, 0.3)])
    bins[100:, :2] = 0.0
    assert _stuck_zero_run(bins) == []
    # Clean neighbors agree with a zero reading, so it is not contradicted either.
    single = SensorHealth()
    for index, value in enumerate(np.r_[rng.uniform(5, 15, 100), np.zeros(40)]):
        single.data_processing("s", {"unix_timestamp": index * 300, "pm2_5": value},
                               references={"pm2_5": [{"sensor": f"p{k}", "timestamp": index * 300,
                                                      "value": 0.4} for k in (1, 2)]})
    assert not any(e["category"] == "sensor_freeze" for e in single.events)


def test_zero_contradicted_by_neighbors_is_stuck():
    rng = np.random.default_rng(4)
    engine = SensorHealth()
    for index, value in enumerate(np.r_[rng.uniform(5, 15, 100), np.zeros(36)]):
        engine.data_processing("s", {"unix_timestamp": index * 300, "pm2_5": value},
                               references={"pm2_5": [{"sensor": f"p{k}", "timestamp": index * 300,
                                                      "value": 9.0} for k in (1, 2)]})
    freezes = [e for e in engine.events if e["category"] == "sensor_freeze"]
    assert len(freezes) == 1
    assert "reference" in freezes[0]["evidence"]["sensor_freeze"]["contradiction_evidence"]
    assert not any(e["family"] == "change" and e["severity"] != "info" for e in engine.events)
