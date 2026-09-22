# ***************************************************************************
#  SAFE — streaming sensor drift & failure engine
#  --------------------------------------------------------------------------
#  Layered per-reading detection, fastest to slowest:
#
#    1. Hard bounds        : physically impossible values (instant)
#       Frozen value       : the same reading repeated for >= 1 h is a stuck
#                            sensor; frozen readings never enter history and
#                            tracking restarts when the value moves again
#    2. Robust z-score     : per-reading outliers via a median/MAD modified
#                            z-score — robust to the very outliers it hunts,
#                            unlike a mean/std z-score (masking/swamping)
#    3. Step-change        : >= 10 consecutive outliers = regime shift;
#                            the buffer is reseeded at the new level
#    4. Page-Hinkley       : sequential CUSUM-style test on standardized
#                            residuals; catches sustained small/medium mean
#                            shifts within tens of samples instead of waiting
#                            a full evaluation window
#    5. Welch + Levene     : windowed mean / variance shift tests with
#                            effect-size gates and autocorrelation-corrected
#                            sample sizes (see safe.stats)
# ***************************************************************************

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from safe.config import (
    DEFAULT_FREEZE_TOLERANCE,
    DEFAULT_ROBUST_SCALE_FLOOR,
    DEFAULT_Z_THRESHOLD,
    FREEZE_EXEMPT_ZERO,
    FREEZE_MIN_READINGS,
    FREEZE_MIN_SECONDS,
    FREEZE_TOLERANCES,
    HARD_BOUNDS,
    ROBUST_SCALE_FLOORS,
)
from safe.stats import sample_comparison, validate_alpha

logger = logging.getLogger(__name__)

# Recompute the robust baseline (median / MAD) every this many accepted
# readings. The baseline moves slowly, so refreshing periodically keeps the
# per-reading cost near O(1) while staying robust.
BASELINE_REFRESH_EVERY = 20

# Minimum buffered readings before outlier / Page-Hinkley checks run
MIN_HISTORY = 30

# Consecutive robust-z outliers that count as a step change
STEP_CHANGE_RUN = 10

# 1.4826 * MAD estimates the standard deviation of a normal distribution
MAD_TO_SIGMA = 1.4826


def default_alert_handler(sensor_name, alert_dict, data_time="N/A"):
    """Log an alert. Swap in an MQTT publisher via SensorDrift(on_alert=...)."""
    lines = [f"\n[ALERT] Sensor: {sensor_name} | Data Time: {data_time}"]
    for key, value in alert_dict.items():
        lines.append(f"  - {key}: {value}")
    lines.append("-" * 30)
    logger.warning("\n".join(lines))


class PageHinkley:
    """Two-sided Page-Hinkley test on standardized residuals.

    Accumulates m_t = sum(r_i - delta) and alarms when m_t rises more than
    `lam` above its running minimum (and symmetrically for downward shifts).
    With delta = 0.25 and lam = 18, a sustained 1-sigma mean shift alarms
    after ~24 samples; pure noise drifts downward and almost never alarms.
    """

    def __init__(self, delta=0.25, lam=18.0):
        if not np.isfinite(delta) or delta < 0 or not np.isfinite(lam) or lam <= 0:
            raise ValueError("delta must be finite and nonnegative; lam must be finite and positive")
        self.delta = delta
        self.lam = lam
        self.reset()

    def reset(self):
        self._m_up = 0.0
        self._min_up = 0.0
        self._m_down = 0.0
        self._min_down = 0.0
        self.samples = 0

    def update(self, residual):
        """Feed one standardized residual; return 'up' / 'down' on alarm, else None."""
        if not np.isfinite(residual):
            raise ValueError("residual must be finite")
        self.samples += 1

        self._m_up += residual - self.delta
        self._min_up = min(self._min_up, self._m_up)

        self._m_down += -residual - self.delta
        self._min_down = min(self._min_down, self._m_down)

        if self._m_up - self._min_up > self.lam:
            self.reset()
            return "up"
        if self._m_down - self._min_down > self.lam:
            self.reset()
            return "down"
        return None


@dataclass
class _MetricState:
    """All per-(sensor, metric) streaming state in one place."""
    buffer: deque
    outliers: deque                       # values of the current consecutive-outlier run
    ph: PageHinkley
    eval_count: int = 0                   # accepted readings since last drift eval
    run_sum: float = 0.0                  # running sum over buffer (mean fallback)
    run_sumsq: float = 0.0                # running sum of squares over buffer
    baseline_median: float = 0.0
    baseline_scale: float = 1.0           # robust sigma estimate (1.4826 * MAD)
    baseline_ready: bool = False
    since_refresh: int = 0                # accepted readings since baseline refresh
    scale_floor: float = DEFAULT_ROBUST_SCALE_FLOOR
    freeze_value: float = None            # value of the current repeated-value run
    freeze_count: int = 0                 # readings in that run
    freeze_start: float = 0.0             # timestamp of the run's first reading
    frozen: bool = False                  # run confirmed as a freeze

    def mean_std(self):
        n = len(self.buffer)
        if n == 0:
            return 0.0, 0.0
        mean = self.run_sum / n
        var = max(self.run_sumsq / n - mean * mean, 0.0)
        return mean, var ** 0.5

    def refresh_baseline(self):
        arr = np.fromiter(self.buffer, dtype=float, count=len(self.buffer))
        med = float(np.median(arr))
        # Robust sigma = max of the MAD and IQR estimators. MAD alone collapses
        # when many readings are identical (quantized PM values, or the buffer
        # right after a step-change reseed); the IQR estimator survives up to
        # 25% duplicates-at-median while both stay robust to outlier bursts.
        mad_sigma = MAD_TO_SIGMA * float(np.median(np.abs(arr - med)))
        q75, q25 = np.percentile(arr, [75.0, 25.0])
        iqr_sigma = float(q75 - q25) / 1.349
        scale = max(mad_sigma, iqr_sigma)
        if scale < self.scale_floor:
            # Flat baseline: fall back to the classic std, floored at the
            # metric's resolution so a one-step move off a flat stretch is not
            # scored as an enormous z.
            scale = max(float(arr.std()), self.scale_floor)
        self.baseline_median = med
        self.baseline_scale = scale
        self.baseline_ready = True
        self.since_refresh = 0

    def rebuild_accumulators(self):
        self.run_sum = float(sum(self.buffer))
        self.run_sumsq = float(sum(v * v for v in self.buffer))

    def restart(self):
        """Drop all history so tracking warms up afresh."""
        self.buffer.clear()
        self.outliers.clear()
        self.eval_count = 0
        self.rebuild_accumulators()
        self.baseline_ready = False
        self.ph.reset()


class SensorDrift:
    """Streaming drift/failure detector. Feed readings via data_processing().

    Parameters
    ----------
    window_size : evaluation window length (drift tests compare its two halves)
    z_threshold : modified z-score cutoff (3.5 = Iglewicz-Hoaglin convention)
    p_alpha     : significance level for the windowed Welch/Levene tests
    cooldown_seconds : minimum spacing between repeat alerts of the same type
    on_alert    : callable(sensor_name, alert_dict, data_time_str); defaults to
                  logging. Alerts are also appended to self.alerts.
    enable_page_hinkley : toggle the sequential mean-shift layer. Off by
                  default: ambient outdoor signals have genuine diurnal mean
                  shifts (weather), so on 5-min air-quality data this layer
                  alarms daily on real atmospheric changes rather than sensor
                  faults. Enable it for stationary streams (e.g. shuntVoltage)
                  or short high-rate windows where the baseline is stable.
    autocorr_correction : use effective sample sizes in the windowed tests
    freeze_detection : flag a metric that repeats one value for
                  FREEZE_MIN_READINGS readings spanning FREEZE_MIN_SECONDS as a
                  frozen sensor, and keep its frozen readings out of history
    """

    def __init__(self, window_size=200, z_threshold=DEFAULT_Z_THRESHOLD,
                 p_alpha=0.01, cooldown_seconds=1800, on_alert=None,
                 enable_page_hinkley=False, autocorr_correction=True,
                 freeze_detection=True):
        if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < MIN_HISTORY:
            raise ValueError(f"window_size must be an integer >= {MIN_HISTORY}")
        validate_alpha(p_alpha)
        if not np.isfinite(z_threshold) or z_threshold <= 0:
            raise ValueError("z_threshold must be finite and positive")
        if not np.isfinite(cooldown_seconds) or cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be finite and nonnegative")
        if on_alert is not None and not callable(on_alert):
            raise ValueError("on_alert must be callable")
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.p_alpha = p_alpha
        self.cooldown_seconds = cooldown_seconds
        self.on_alert = on_alert or default_alert_handler
        self.enable_page_hinkley = enable_page_hinkley
        self.autocorr_correction = autocorr_correction
        self.freeze_detection = freeze_detection

        self.hard_bounds = HARD_BOUNDS.copy()
        self.alerts = []                  # (sensor, alert_dict, data_time) history
        self._states = {}                 # {sensor: {metric: _MetricState}}
        self._last_alert_time = {}
        self._last_reading_time = {}

    # ------------------------------------------------------------------
    # alert plumbing
    # ------------------------------------------------------------------
    def _emit(self, sensor_name, alert_dict, data_time):
        self.alerts.append((sensor_name, alert_dict, data_time))
        try:
            self.on_alert(sensor_name, alert_dict, data_time)
        except Exception:
            logger.exception(f"Alert handler failed for {sensor_name}")

    def _alert_cooldown(self, sensor_name, metric, alert_type, current_timestamp):
        """True when an alert of this type may fire (and record that it did)."""
        key = (sensor_name, metric, alert_type)
        last_time = self._last_alert_time.get(key)
        if last_time is not None and current_timestamp - last_time < self.cooldown_seconds:
            return False
        self._last_alert_time[key] = current_timestamp
        return True

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def _state_for(self, sensor_name, metric):
        sensor_states = self._states.setdefault(sensor_name, {})
        state = sensor_states.get(metric)
        if state is None:
            state = _MetricState(
                buffer=deque(maxlen=self.window_size),
                outliers=deque(maxlen=STEP_CHANGE_RUN),
                ph=PageHinkley(),
                scale_floor=ROBUST_SCALE_FLOORS.get(metric, DEFAULT_ROBUST_SCALE_FLOOR),
            )
            sensor_states[metric] = state
        return state

    def data_processing(self, sensor_name, sensor_dict):
        """Process one reading dict for one sensor.

        The dict may carry pre-parsed 'unix_timestamp' / 'str_timestamp' keys
        (fast path used by the CSV loader) or a raw 'dateTime' value; every
        other numeric key is treated as a metric.
        """
        current_timestamp = sensor_dict.get('unix_timestamp')
        data_time_str = sensor_dict.get('str_timestamp')

        if current_timestamp is None:
            dt = sensor_dict.get('dateTime')
            if dt is None:
                return
            parsed = pd.to_datetime(dt, utc=True)
            if pd.isna(parsed):
                raise ValueError("dateTime must be a valid timestamp")
            current_timestamp = parsed.timestamp()
            data_time_str = str(dt)

        current_timestamp = float(current_timestamp)
        if not np.isfinite(current_timestamp):
            raise ValueError("unix_timestamp must be finite")

        previous = self._last_reading_time.get(sensor_name)
        if previous is not None and current_timestamp < previous:
            raise ValueError(f"out-of-order reading for {sensor_name}")
        self._last_reading_time[sensor_name] = current_timestamp

        for key, val in sensor_dict.items():
            if key in ("dateTime", "unix_timestamp", "str_timestamp"):
                continue
            try:
                value = float(val)
                if not np.isfinite(value):
                    continue
            except (ValueError, TypeError):
                continue  # skip non-numeric values

            self._process_value(sensor_name, key, value, current_timestamp, data_time_str)

    def _process_value(self, sensor_name, metric, value, current_timestamp, data_time_str):
        # ---- layer 1: hard physical bounds -----------------------------
        bounds = self.hard_bounds.get(metric)
        if bounds and (value < bounds[0] or value > bounds[1]):
            state = self._states.get(sensor_name, {}).get(metric)
            if state is not None:
                state.outliers.clear()
            if self._alert_cooldown(sensor_name, metric, "hard-bounds", current_timestamp):
                self._emit(sensor_name, {
                    "alert": "hard-bounds-violation",
                    "metric": metric,
                    "value": round(value, 3),
                    "bounds": bounds,
                }, data_time_str)
            return  # never let impossible values into history

        state = self._state_for(sensor_name, metric)
        buffer = state.buffer

        if self.freeze_detection and self._track_freeze(sensor_name, metric, state, value,
                                                        current_timestamp, data_time_str):
            return  # a frozen reading is not data

        if len(buffer) >= MIN_HISTORY:
            if not state.baseline_ready or state.since_refresh >= BASELINE_REFRESH_EVERY:
                state.refresh_baseline()

            residual = (value - state.baseline_median) / state.baseline_scale

            # ---- layer 4: Page-Hinkley sequential mean-shift test ------
            # Fed BEFORE outlier rejection so a sustained shift whose values
            # get rejected as outliers is still seen.
            if self.enable_page_hinkley:
                direction = state.ph.update(residual)
                if direction and self._alert_cooldown(sensor_name, metric, "page-hinkley", current_timestamp):
                    self._emit(sensor_name, {
                        "alert": "Sustained Mean Shift (Page-Hinkley)",
                        "metric": metric,
                        "direction": direction,
                        "value": round(value, 3),
                        "baseline_median": round(state.baseline_median, 3),
                        "baseline_scale": round(state.baseline_scale, 4),
                    }, data_time_str)

            # ---- layer 2: robust (median/MAD) z-score outlier check ----
            if abs(residual) > self.z_threshold:
                # Alternating high/low spikes are not a sustained level shift.
                if state.outliers and (state.outliers[-1] - state.baseline_median) * residual < 0:
                    state.outliers.clear()
                state.outliers.append(value)

                # ---- layer 3: consecutive outliers = step change -------
                if len(state.outliers) >= STEP_CHANGE_RUN:
                    self._handle_step_change(sensor_name, metric, state, value,
                                             current_timestamp, data_time_str)
                elif self._alert_cooldown(sensor_name, metric, "z-score", current_timestamp):
                    self._emit(sensor_name, {
                        "alert": "z-score-outlier",
                        "metric": metric,
                        "value": round(value, 3),
                        "modified_z": round(abs(residual), 3),
                    }, data_time_str)

                # outliers never enter the buffer or the eval counter
                return

        # ---- value accepted: maintain buffer + accumulators -------------
        state.outliers.clear()

        if len(buffer) == self.window_size:
            evicted = buffer[0]
            state.run_sum -= evicted
            state.run_sumsq -= evicted * evicted

        buffer.append(value)
        state.run_sum += value
        state.run_sumsq += value * value
        state.since_refresh += 1
        state.eval_count += 1

        # ---- layer 5: windowed drift evaluation --------------------------
        if state.eval_count >= self.window_size:
            if len(buffer) == self.window_size:
                self._evaluate_drift(sensor_name, metric, list(buffer),
                                     current_timestamp, data_time_str)

            # Rebuild the accumulators to clear incremental floating-point drift
            state.rebuild_accumulators()

            # Reset to half the window so evaluations overlap (samples 101-200
            # get compared against 201-300 — no blind spots between windows).
            state.eval_count = self.window_size // 2

    def _track_freeze(self, sensor_name, metric, state, value,
                      current_timestamp, data_time_str):
        """Track repeated-value runs; return True while the metric is frozen.

        When a confirmed freeze ends, history is dropped: the buffer holds
        pre-freeze readings that may be hours stale, so the first live reading
        would otherwise be misread as an outlier or step change.
        """
        tolerance = FREEZE_TOLERANCES.get(metric, DEFAULT_FREEZE_TOLERANCE)
        if state.freeze_value is not None and abs(value - state.freeze_value) <= tolerance:
            state.freeze_count += 1
        else:
            if state.frozen:
                if self._alert_cooldown(sensor_name, metric, "freeze-recovered", current_timestamp):
                    self._emit(sensor_name, {
                        "alert": "Frozen Sensor Recovered",
                        "metric": metric,
                        "frozen_value": round(state.freeze_value, 3),
                        "frozen_readings": state.freeze_count,
                        "frozen_seconds": round(current_timestamp - state.freeze_start),
                        "value": round(value, 3),
                    }, data_time_str)
                state.restart()
            state.freeze_value = value
            state.freeze_count = 1
            state.freeze_start = current_timestamp
            state.frozen = False
            return False

        if state.frozen:
            return True
        if value == 0.0 and metric in FREEZE_EXEMPT_ZERO:
            return False
        if (state.freeze_count >= FREEZE_MIN_READINGS
                and current_timestamp - state.freeze_start >= FREEZE_MIN_SECONDS):
            state.frozen = True
            if self._alert_cooldown(sensor_name, metric, "freeze", current_timestamp):
                self._emit(sensor_name, {
                    "alert": "Frozen Sensor Detected",
                    "metric": metric,
                    "value": round(value, 3),
                    "repeated_readings": state.freeze_count,
                    "frozen_seconds": round(current_timestamp - state.freeze_start),
                }, data_time_str)
            return True
        return False

    def _handle_step_change(self, sensor_name, metric, state, value,
                            current_timestamp, data_time_str):
        """A run of consecutive outliers is a regime shift, not noise: alert,
        then reseed the buffer at the new level so tracking resumes immediately."""
        saved_values = list(state.outliers)
        old_mean, _ = state.mean_std()

        if self._alert_cooldown(sensor_name, metric, "step-change", current_timestamp):
            self._emit(sensor_name, {
                "alert": "Step-Change Detected",
                "metric": metric,
                "value": round(value, 3),
                "consecutive_outliers": len(saved_values),
                "old_mean": round(old_mean, 3),
            }, data_time_str)

        state.buffer.clear()
        state.buffer.extend(saved_values)
        state.outliers.clear()
        state.eval_count = len(saved_values)
        state.rebuild_accumulators()
        state.baseline_ready = False   # force a baseline refresh at the new level
        state.ph.reset()

    # ------------------------------------------------------------------
    # windowed evaluation
    # ------------------------------------------------------------------
    def _evaluate_drift(self, sensor_name, metric, data, current_timestamp, data_time_str):
        # Compare the older half of the window against the newer half
        mid = len(data) // 2
        result = sample_comparison(data[:mid], data[mid:], self.p_alpha, metric=metric,
                                   autocorr_correction=self.autocorr_correction)

        old_mean = result['old_mean']
        new_mean = result['new_mean']

        # Both halves flat: a step-change only matters if the level moved.
        if result['both_flat']:
            if result['mean_val_changed']:
                if self._alert_cooldown(sensor_name, metric, "drift", current_timestamp):
                    self._emit(sensor_name, {
                        "alert": "Sensor Drift Detected Step-Change",
                        "metric": metric,
                        "old_mean": round(old_mean, 3),
                        "new_mean": round(new_mean, 3),
                    }, data_time_str)
            return

        # One half flat, the other not: variance regime change. Publish, but
        # do NOT return — the mean/variance tests below still apply.
        if result['half_flat']:
            if self._alert_cooldown(sensor_name, metric, "variance-regime", current_timestamp):
                self._emit(sensor_name, {
                    "alert": "Variance Regime Change",
                    "metric": metric,
                    "detail": "One half is flat while the other has variance",
                    "old_mean": round(old_mean, 3),
                    "new_mean": round(new_mean, 3),
                    "old_variance": round(float(result['old_var']), 6),
                    "new_variance": round(float(result['new_var']), 6),
                }, data_time_str)

        if result['mean_shift'] or result['variance_shift']:
            if not self._alert_cooldown(sensor_name, metric, "drift", current_timestamp):
                return
            self._emit(sensor_name, {
                "alert": "Sensor Drift Detected",
                "metric": metric,
                "mean_shift": result['mean_shift'],
                "variance_shift": result['variance_shift'],
                "cohens_d": round(result['cohens_d'], 3),
                "std_ratio": round(result['std_ratio'], 3),
                "n_eff": f"{result['old_n_eff']}/{result['new_n_eff']}",
                "p_welch": format(result['p_welch'], ".2e"),
                "p_levene": format(result['p_levene'], ".2e"),
            }, data_time_str)
