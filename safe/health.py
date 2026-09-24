"""Incident-based sensor health engine: SAFE's single streaming detector."""

from collections import deque
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from safe.baseline import SeasonalBaseline, robust_scale
from safe.config import PC_METRICS, PM_METRICS
from safe.incidents import IncidentManager
from safe.profiles import MetricProfile, ProfileRegistry, SensorRules
from safe.stats import sample_comparison

STATE_SCHEMA_VERSION = 1
ENGINE_VERSION = "3.0.0"


def utc_seconds(value):
    if isinstance(value, bool):
        raise ValueError("timestamp cannot be boolean")
    if isinstance(value, (int, float, np.number)):
        result = float(value)
    else:
        parsed = pd.to_datetime(value, utc=True)
        result = parsed.timestamp() if not pd.isna(parsed) else float("nan")
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite and valid")
    return result


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


def _peer_sample(p, value, peer):
    """One reading's target/reference relationship: a difference, or a log ratio."""
    c = p.reference_ratio_floor
    if not c:
        return value - peer
    # Clamp so a value below -floor (only possible without hard bounds) stays finite.
    return math.log(max(value + c, 1e-3 * c) / max(peer + c, 1e-3 * c))


def _peer_expected(p, peer, offset):
    """Target value predicted from the reference and the learned relationship."""
    c = p.reference_ratio_floor
    return (peer + c) * math.exp(offset) - c if c else peer + offset


class _HealthState:
    def __init__(self, profile):
        self.profile = profile
        self.baseline = SeasonalBaseline(profile)
        self.ph = PageHinkley()
        self.warmup = deque(maxlen=profile.max_samples)
        self.residuals = deque(maxlen=profile.max_samples)
        self.reference_levels = deque(maxlen=profile.max_samples)  # (t, reference-predicted value)
        self.recent_anomalies = deque(maxlen=profile.max_samples)
        self.peer_samples = deque(maxlen=profile.max_samples)
        self.arrivals = deque(maxlen=profile.max_samples)
        self.cadences = deque(maxlen=12)
        self.first_at = None
        self.last_at = None
        self.last_value = None
        self.freeze_anchor = None
        self.freeze_started = None
        self.freeze_count = 0
        self.has_variability = False
        self.run_sign = 0
        self.run_count = 0
        self.run_started = None
        self.last_evaluation = None
        self.evaluations = 0
        self.mean_confirmations = 0
        self.variance_confirmations = 0
        self.invalid_run = 0
        self.startup_run = 0
        self.warmup_restarts = 0
        self.mode = "WARMING_UP"
        self.peer_mode = False
        self.peer_offset = None
        self.window_category = None
        self.window_evidence = {}
        self.reference_drift_confirmations = 0
        self.reference_drift_evidence = None

    def to_dict(self):
        output = {}
        for key, value in vars(self).items():
            if key == "profile":
                output[key] = value.to_dict()
            elif key == "baseline":
                output[key] = value.to_dict()
            elif key == "ph":
                output[key] = vars(value).copy()
            elif isinstance(value, deque):
                output[key] = list(value)
            else:
                output[key] = value
        return output

    @classmethod
    def from_dict(cls, data):
        obj = cls(MetricProfile(**data["profile"]))
        for key, value in data.items():
            if key == "profile":
                continue
            if key == "baseline":
                obj.baseline = SeasonalBaseline.from_dict(obj.profile, value)
            elif key == "ph":
                obj.ph = PageHinkley(value["delta"], value["lam"])
                vars(obj.ph).update(value)
            elif key in {"warmup", "residuals", "reference_levels", "arrivals", "recent_anomalies",
                         "peer_samples"}:
                if len(value) > obj.profile.max_samples:
                    raise ValueError("state history exceeds configured bound")
                setattr(obj, key, deque(value, maxlen=obj.profile.max_samples))
            elif key == "cadences":
                if len(value) > 12:
                    raise ValueError("state cadence history exceeds bound")
                obj.cadences = deque(value, maxlen=12)
            elif key not in vars(obj):
                raise ValueError("unknown health state field")
            else:
                setattr(obj, key, value)
        return obj


class SensorHealth:
    """Feed data_processing(sensor, reading), and tick(clock) during silence.

    Per-metric timestamps must increase; equal sensor timestamps may carry disjoint
    fields. Invalid values become evidence and never enter the environmental model.
    References are explicit, co-located/compatible measurements supplied by callers;
    the engine never guesses geographic neighbors or treats label data as inputs.
    """

    def __init__(self, profiles=None, sensors=None, rules=None, history_limit=1000,
                 max_series=1000, notification_interval_seconds=1800,
                 on_event=None, on_notification=None):
        if isinstance(max_series, bool) or not isinstance(max_series, int) or max_series < 1:
            raise ValueError("max_series must be a positive integer")
        self.profiles = profiles if isinstance(profiles, ProfileRegistry) else ProfileRegistry(**(profiles or {}))
        self.sensors = deepcopy(sensors or {})
        if any(not set(v) <= {"model", "site"} for v in self.sensors.values()):
            raise ValueError("sensor metadata only supports model and site")
        self.rules = {k: (v if isinstance(v, SensorRules) else SensorRules(**v))
                      for k, v in (rules or {"IPS7100": SensorRules(ordered_metrics=PM_METRICS)}).items()}
        self.max_series = max_series
        self.incidents = IncidentManager(history_limit, notification_interval_seconds,
                                         on_event, on_notification)
        self._states = {}
        self._clock = None

    def configuration(self):
        return dict(profiles=self.profiles.to_dict(), sensors=self.sensors,
                    rules={k: asdict(v) for k, v in self.rules.items()},
                    max_series=self.max_series, history_limit=self.incidents.history.maxlen,
                    notification_interval_seconds=self.incidents.notification_interval_seconds)

    @property
    def events(self):
        return [e.to_dict() for e in self.incidents.events()]

    def register(self, sensor, metric, started_at=None):
        if not isinstance(sensor, str) or not sensor.strip() or not isinstance(metric, str) or not metric.strip():
            raise ValueError("sensor and metric must be nonempty strings")
        key = (sensor, metric)
        if key not in self._states:
            if len(self._states) >= self.max_series:
                raise ValueError("max_series exceeded; explicitly retire inactive series")
            metadata = self.sensors.get(sensor, {})
            state = _HealthState(self.profiles.resolve(sensor, metric, **metadata))
            if started_at is not None:
                state.first_at = utc_seconds(started_at)
            self._states[key] = state
        return self._states[key]

    def retire(self, sensor, metric):
        """Explicit deployment removal; refuses to discard active incidents."""
        if any(k[:2] == (sensor, metric) for k in self.incidents.active):
            raise ValueError("cannot retire a series with active incidents")
        self._states.pop((sensor, metric), None)
        self.incidents.last_notification.pop((sensor, metric), None)

    def _observe(self, sensor, metric, category, family, t, severity="warning", confidence=0.8, **evidence):
        state = self._states[(sensor, metric)]
        if severity != "info":
            state.mode = "INCIDENT"
        elif state.mode == "MONITORING":
            state.mode = "SUSPECTED"
        configuration = dict(profile=state.profile.to_dict(), **self.sensors.get(sensor, {}))
        return self.incidents.observe(sensor, metric, category, family, t, severity,
                                      confidence, evidence, configuration)

    def tick(self, current_time):
        t = utc_seconds(current_time)
        if self._clock is not None and t < self._clock:
            raise ValueError("tick clock must be monotonic")
        self._clock = t
        for (sensor, metric), state in self._states.items():
            anchor = state.last_at if state.last_at is not None else state.first_at
            if anchor is None:
                continue
            p = state.profile
            if t - anchor >= p.expected_interval_seconds * p.gap_factor:
                self._observe(sensor, metric, "missing_data", "availability", t,
                              confidence=0.99, last_reading_at=state.last_at,
                              silence_seconds=t - anchor, expected_interval=p.expected_interval_seconds)
                state.mode = "INCIDENT"

    def data_processing(self, sensor_name, sensor_dict, references=None, received_at=None):
        """references: metric -> [{value, timestamp, sensor}, ...], >=2 peers.

        One trusted reference is supported with trusted=True. Reference timestamps
        must be no later than this reading and no older than one nominal interval.
        """
        stamp = sensor_dict.get("unix_timestamp", sensor_dict.get("dateTime"))
        if stamp is None:
            raise ValueError("reading requires dateTime or unix_timestamp")
        t = utc_seconds(stamp)
        arrival = utc_seconds(received_at) if received_at is not None else None
        metadata = {"dateTime", "unix_timestamp", "str_timestamp"}
        values = {}
        for metric, value in sensor_dict.items():
            if metric in metadata or value is None:
                continue
            try:
                values[metric] = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"nonnumeric measurement: {metric}") from None
        new_keys = {(sensor_name, metric) for metric in values} - self._states.keys()
        if len(self._states) + len(new_keys) > self.max_series:
            raise ValueError("max_series exceeded")
        if arrival is not None:
            self.tick(arrival)
            accepted = {}
            for metric, value in values.items():
                state = self.register(sensor_name, metric, arrival)
                if t > arrival + state.profile.expected_interval_seconds:
                    self._observe(sensor_name, metric, "future_timestamp", "timestamp", arrival,
                                  confidence=0.99, rejected_timestamp=t, received_at=arrival)
                else:
                    accepted[metric] = value
            values = accepted
            if not values:
                return
        model = self.sensors.get(sensor_name, {}).get("model")
        rule = self.rules.get(model, SensorRules())
        rule_findings = self._plausibility(values, rule)
        jumps = []
        for metric, value in values.items():
            state = self.register(sensor_name, metric)
            if state.last_value is not None and math.isfinite(value):
                if abs(value - state.last_value) >= state.profile.step_min_effect:
                    jumps.append(metric)
        if len(jumps) >= rule.simultaneous_jump_metrics:
            for metric in jumps:
                rule_findings.setdefault(metric, []).append(("possible_restart", {"jumped_metrics": jumps}))
        # Normal ingestion advances a clock so a later resumed reading exposes a gap.
        # Explicit tick calls are still needed to detect a sensor that never resumes.
        if arrival is None and (self._clock is None or t > self._clock):
            self.tick(t)
        for metric, value in values.items():
            self._process(sensor_name, metric, value, t, rule_findings.get(metric, []),
                          (references or {}).get(metric))

    def _plausibility(self, values, rule):
        findings = {}
        ordered = [m for m in rule.ordered_metrics if m in values and math.isfinite(values[m])]
        for small, large in zip(ordered, ordered[1:]):
            if values[small] > values[large] + rule.ordering_tolerance:
                for metric in (small, large):
                    findings.setdefault(metric, []).append(("pm_ordering", {"smaller": small, "larger": large,
                        "smaller_value": values[small], "larger_value": values[large]}))
        if rule.status_field in values and values[rule.status_field] not in rule.healthy_status_values:
            for metric in values:
                findings.setdefault(metric, []).append(("sensor_status", {"field": rule.status_field,
                    "value": values[rule.status_field] if math.isfinite(values[rule.status_field]) else None}))
        if (rule.dewpoint_field in values and "temperature" in values
                and values[rule.dewpoint_field] > values["temperature"] + rule.dewpoint_tolerance):
            findings.setdefault("temperature", []).append(("physical_relationship", {
                "detail": "dewpoint exceeds temperature", "dewpoint": values[rule.dewpoint_field]}))
        return findings

    def _peer(self, sensor, p, t, references):
        valid = {}
        trusted = []
        for reference in references or ():
            stamp = utc_seconds(reference["timestamp"])
            value = float(reference["value"])
            peer = reference["sensor"]
            if peer != sensor and math.isfinite(value) and 0 <= t - stamp <= p.expected_interval_seconds:
                valid[peer] = value
                if reference.get("trusted") is True:
                    trusted.append(value)
        if trusted:
            return float(np.median(trusted))
        return float(np.median(list(valid.values()))) if len(valid) >= 2 else None

    def _process(self, sensor, metric, value, t, findings, references):
        state = self._states[(sensor, metric)]
        p = state.profile
        if state.last_at is not None and t <= state.last_at:
            self._observe(sensor, metric, "duplicate_timestamp" if t == state.last_at else "timestamp_disorder",
                          "timestamp", max(t, state.last_at), confidence=0.99, rejected_timestamp=t)
            return
        if self._clock is not None and self._clock - t >= p.gap_factor * p.expected_interval_seconds:
            self._observe(sensor, metric, "delayed_data", "availability", self._clock,
                          confidence=0.99, rejected_timestamp=t, delay_seconds=self._clock - t)
            return
        seen = set()
        if state.first_at is None:
            state.first_at = t
        previous = state.last_at
        state.last_at = t
        slot = math.floor((t - state.first_at) / p.expected_interval_seconds)
        if not state.arrivals or slot != math.floor((state.arrivals[-1] - state.first_at) / p.expected_interval_seconds):
            state.arrivals.append(t)
        if previous is not None:
            state.cadences.append(t - previous)
        while state.arrivals and t - state.arrivals[0] > p.completeness_window_seconds:
            state.arrivals.popleft()
        if previous is not None and t - previous >= p.expected_interval_seconds * p.gap_factor:
            seen.add("availability")
            state.freeze_count = 0
            state.freeze_started = None
            state.run_count = 0
            state.residuals.clear()
            state.reference_levels.clear()
            state.recent_anomalies.clear()
            state.window_category = None
            state.ph.reset()
        duration = min(t - state.first_at, p.completeness_window_seconds)
        # Bucket occupancy counts at most one reading per expected cadence interval.
        if duration >= p.completeness_window_seconds:
            slots = {int((a - state.first_at) // p.expected_interval_seconds) for a in state.arrivals}
            expected = math.floor(duration / p.expected_interval_seconds) + 1
            completeness = min(1.0, len(slots) / expected)
            if completeness < p.minimum_completeness:
                seen.add("availability")
                self._observe(sensor, metric, "completeness_loss", "availability", t,
                              completeness=completeness, expected_slots=expected, occupied_slots=len(slots))
        if len(state.cadences) >= 11:
            cadence = float(np.median(state.cadences))
            if cadence > 1.5 * p.expected_interval_seconds:
                seen.add("availability")
                self._observe(sensor, metric, "cadence_degradation", "availability", t,
                              observed_interval=cadence, expected_interval=p.expected_interval_seconds)
        for category, evidence in findings:
            family = "change" if category == "possible_restart" else "plausibility"
            seen.add(family)
            self._observe(sensor, metric, category, family, t,
                          severity="info" if category == "possible_restart" else "warning",
                          confidence=0.6 if category == "possible_restart" else 0.95, **evidence)
        if (not math.isfinite(value) or p.hard_bounds is not None
                and not p.hard_bounds[0] <= value <= p.hard_bounds[1]):
            seen.add("validity")
            self._observe(sensor, metric, "invalid_measurement", "validity", t, "critical", 1.0,
                          value=value if math.isfinite(value) else None, bounds=p.hard_bounds)
            state.invalid_run += 1
            state.freeze_started = None
            state.run_count = 0
            if state.invalid_run >= 3 and not state.baseline.ready:
                state.warmup.clear()
                state.warmup_restarts += 1
            state.mode = "INCIDENT"
            return
        state.invalid_run = 0
        if state.freeze_started is None or abs(value - state.freeze_anchor) > p.freeze_tolerance:
            if state.freeze_anchor is not None and abs(value - state.freeze_anchor) > 4 * p.freeze_tolerance:
                state.has_variability = True
            state.freeze_anchor = value
            state.freeze_started = t
            state.freeze_count = 1
        else:
            state.freeze_count += 1
        if (state.freeze_count >= p.freeze_min_readings and t - state.freeze_started >= p.freeze_duration_seconds
                and (p.freeze_at_startup or state.has_variability)
                # Clean air legitimately holds PM and particle-count bins at zero for hours.
                and not ((metric in PM_METRICS or metric in PC_METRICS) and value <= p.freeze_tolerance)):
            seen.add("freeze")
            self._observe(sensor, metric, "sensor_freeze", "freeze", t, confidence=0.95,
                          run_length=state.freeze_count, duration_seconds=t - state.freeze_started,
                          tolerance=p.freeze_tolerance, value=value)
        state.last_value = value
        peer = self._peer(sensor, p, t, references)
        if not state.baseline.ready:
            if peer is not None and not ({"freeze", "plausibility"} & seen):
                state.peer_samples.append(_peer_sample(p, value, peer))
            self._warmup(sensor, metric, state, value, t, seen)
        else:
            self._detect(sensor, metric, state, value, t, peer, seen)
        change = self.incidents.active.get((sensor, metric, "change"))
        self.incidents.recover(sensor, metric, t, seen,
                               p.window_seconds if change and change.category == "noise_change" else p.recovery_seconds,
                               p.recovery_readings, eligible_families={"change"})
        self.incidents.recover(sensor, metric, t, seen, p.recovery_seconds, p.recovery_readings,
                               eligible_families={"validity", "availability", "timestamp", "plausibility",
                                                  "freeze", "startup", "environment"})
        active = [e for k, e in self.incidents.active.items() if k[:2] == (sensor, metric)]
        state.mode = ("RECOVERING" if active and all(e.status == "recovering" for e in active)
                      else "INCIDENT" if any(e.severity != "info" for e in active)
                      else "SUSPECTED" if active else "MONITORING" if state.baseline.ready else "WARMING_UP")

    def _warmup(self, sensor, metric, state, value, t, seen):
        p = state.profile
        if len(state.warmup) >= 8:
            center, scale = robust_scale([v for _, v in list(state.warmup)[-24:]], p.residual_scale_floor)
            # A provisional gate only catches extreme contamination; ordinary daily
            # movement must still populate a full cycle before residual detection.
            if abs(value - center) > max(12 * scale, 4 * p.step_min_effect):
                state.startup_run += 1
                seen.add("startup")
                self._observe(sensor, metric, "startup_contamination", "startup", t, "info", 0.6,
                              value=value, provisional_center=center)
                if state.startup_run >= p.step_readings:
                    state.warmup.clear()
                    state.warmup_restarts += 1
                return
        state.startup_run = 0
        if not ({"freeze", "plausibility"} & seen):
            state.warmup.append((t, value))
        if len(state.warmup) < p.minimum_samples:
            return
        while len(state.warmup) > p.minimum_samples and t - state.warmup[0][0] > 2 * p.warmup_duration_seconds:
            state.warmup.popleft()
        elapsed = t - state.warmup[0][0]
        slots = {int(a // p.expected_interval_seconds) for a, _ in state.warmup}
        expected = math.floor(elapsed / p.expected_interval_seconds) + 1
        if (elapsed >= p.warmup_duration_seconds and len(slots) / expected >= p.minimum_coverage
                and state.baseline.fit(list(state.warmup))):
            state.warmup.clear()
            state.ph.reset()
            state.last_evaluation = t

    def _detect(self, sensor, metric, state, value, t, peer, seen):
        p = state.profile
        expected, scale = state.baseline.predict(t)
        if peer is not None and state.peer_offset is None:
            if len(state.peer_samples) >= p.minimum_samples:
                state.peer_offset = float(np.median(state.peer_samples))
                state.peer_samples.clear()
            elif abs(value - expected) < p.step_min_effect:
                state.peer_samples.append(_peer_sample(p, value, peer))
            if state.peer_offset is None:
                peer = None  # no peer diagnosis until target/reference offset is learned
        if (peer is not None) != state.peer_mode:
            state.peer_mode = peer is not None
            state.residuals.clear()
            state.reference_levels.clear()
            state.ph.reset()
            state.run_count = 0
            state.reference_drift_confirmations = 0
            state.reference_drift_evidence = None
        if peer is not None:
            expected = _peer_expected(p, peer, state.peer_offset)
        residual = value - expected
        z = residual / scale
        anomalous = abs(z) >= p.outlier_threshold and abs(residual) >= p.step_min_effect
        environmental = metric in PM_METRICS and peer is None
        if peer is not None and metric in PM_METRICS and not anomalous:
            ambient_expected, _ = state.baseline.predict(t)
            if abs(value - ambient_expected) >= p.step_min_effect:
                seen.add("environment")
                self._observe(sensor, metric, "environmental_event", "environment", t, "info", 0.7,
                              value=value, peer_consensus=peer, residual=residual)
        if anomalous:
            sign = 1 if residual > 0 else -1
            if state.run_sign != sign:
                state.run_count = 0
            if state.run_count == 0:
                state.run_started = t
            state.run_sign = sign
            state.run_count += 1
            persistent = state.run_count >= p.step_readings
            category = ("uncertain_change" if environmental and persistent else
                        "abrupt_shift" if persistent else "isolated_anomaly")
            seen.add("change")
            self._observe(sensor, metric, category, "change", t,
                          "warning" if persistent and not environmental else "info",
                          0.85 if persistent and not environmental else 0.55,
                          value=value, expected=expected, residual=residual, residual_scale=scale,
                          standardized_residual=z, consecutive_readings=state.run_count,
                          peer_reference=peer, interpretation="change evidence; cause unconfirmed")
        else:
            state.run_count = 0
            state.run_sign = 0
        state.recent_anomalies.append((t, (1 if residual > 0 else -1) if anomalous else 0))
        while state.recent_anomalies and t - state.recent_anomalies[0][0] > p.window_seconds:
            state.recent_anomalies.popleft()
        positive = sum(flag > 0 for _, flag in state.recent_anomalies)
        negative = sum(flag < 0 for _, flag in state.recent_anomalies)
        if (len(state.recent_anomalies) >= p.minimum_samples and min(positive, negative) >= 2
                and positive + negative >= max(4, .1 * len(state.recent_anomalies))
                and state.run_count < p.step_readings):
            seen.add("change")
            self._observe(sensor, metric, "uncertain_change" if environmental else "noise_change",
                          "change", t, "info" if environmental else "warning", 0.7,
                          anomalous_readings=positive + negative,
                          window_readings=len(state.recent_anomalies),
                          detail="persistent excess of residual anomalies; cause unconfirmed")
        if p.enable_page_hinkley and state.baseline.ready and not environmental:
            direction = state.ph.update(z)
            if direction:
                seen.add("change")
                self._observe(sensor, metric, "gradual_degradation", "change", t,
                              confidence=0.7, direction=direction, residual=residual,
                              stationary_residuals=True)
        state.residuals.append((t, residual))
        while state.residuals and t - state.residuals[0][0] > p.window_seconds:
            state.residuals.popleft()
        if peer is not None:
            state.reference_levels.append((t, expected))
            while t - state.reference_levels[0][0] > p.window_seconds:
                state.reference_levels.popleft()
        if state.last_evaluation is None or t - state.last_evaluation >= p.evaluation_interval_seconds:
            state.last_evaluation = t
            self._window_test(sensor, metric, state, t, seen, environmental)
            self._reference_drift_test(state)
        if state.window_category is not None:
            seen.add("change")
            self._observe(sensor, metric, state.window_category, "change", t,
                          "info" if environmental else "warning", 0.7, **state.window_evidence)
        if state.reference_drift_evidence is not None:
            seen.add("change")
            self._observe(sensor, metric, "gradual_degradation", "change", t, "warning", 0.8,
                          **state.reference_drift_evidence)
        if not anomalous and not ({"freeze", "plausibility"} & seen) and not p.stationary_residuals:
            before = state.baseline.revision
            state.baseline.learn(t, value)
            if state.baseline.revision != before:
                # Any changing expected baseline invalidates a cumulative test's origin.
                state.ph.reset()

    def _window_test(self, sensor, metric, state, t, seen, environmental):
        p = state.profile
        midpoint = t - p.window_seconds / 2
        old = [v for stamp, v in state.residuals if stamp < midpoint]
        new = [v for stamp, v in state.residuals if stamp >= midpoint]
        if min(len(old), len(new)) < max(8, p.minimum_samples // 2):
            return
        state.evaluations += 1
        # Summable alpha spending: sum_k alpha/[2*k*(k+1)] per test is alpha/2.
        # This controls the budget across both test families if each p-value is valid;
        # the underlying AR(1) approximation does not establish field calibration.
        alpha = p.family_alpha / (2 * state.evaluations * (state.evaluations + 1))
        result = sample_comparison(old, new, alpha, metric, autocorr_correction=True)
        mean = result["mean_shift"] and abs(result["mean_delta"]) >= p.step_min_effect
        variance = result["variance_shift"] and (result["std_ratio"] >= 2 or result["std_ratio"] <= 0.5)
        state.mean_confirmations = state.mean_confirmations + 1 if mean else 0
        state.variance_confirmations = state.variance_confirmations + 1 if variance else 0
        if not mean and not variance:
            state.window_category = None
            state.window_evidence = {}
        if max(state.mean_confirmations, state.variance_confirmations) < p.persistence_evaluations:
            return
        category = ("uncertain_change" if environmental else "gradual_degradation"
                    if state.mean_confirmations >= p.persistence_evaluations else "noise_change")
        seen.add("change")
        evidence = {k: v for k, v in result.items() if isinstance(v, (int, float, bool)) and math.isfinite(v)}
        state.window_category = category
        state.window_evidence = dict(alpha_spent_this_test=alpha, evaluation_number=state.evaluations, **evidence)

    def _reference_drift_test(self, state):
        """Calibration drift: sustained disagreement with a reference.

        Residuals against a reference cancel shared weather, so a persistent
        nonzero median is attributable to this sensor. The median over the
        window is robust to reference glitches; the tolerance is a practical
        calibration limit, not a significance level.
        """
        p = state.profile
        values = [v for _, v in state.residuals]
        if not state.peer_mode or len(values) < max(8, p.minimum_samples // 2):
            state.reference_drift_confirmations = 0
            state.reference_drift_evidence = None
            return
        median = float(np.median(values))
        level = float(np.median([v for _, v in state.reference_levels])) if state.reference_levels else 0.0
        tolerance = max(p.reference_drift_tolerance, p.reference_drift_relative_tolerance * abs(level))
        if abs(median) < tolerance:
            state.reference_drift_confirmations = 0
            state.reference_drift_evidence = None
            return
        state.reference_drift_confirmations += 1
        if state.reference_drift_confirmations >= p.persistence_evaluations:
            state.reference_drift_evidence = dict(
                evidence_source="reference", median_reference_residual=median,
                reference_offset=state.peer_offset,
                reference_model="ratio" if p.reference_ratio_floor else "difference",
                window_readings=len(values),
                tolerance=tolerance, reference_level=level,
                interpretation="sustained disagreement with reference; calibration drift suspected")

    def snapshot(self):
        payload = {"schema_version": STATE_SCHEMA_VERSION, "engine_version": ENGINE_VERSION,
                   "configuration": self.configuration(), "clock": self._clock,
                   "states": [{"sensor": s, "metric": m, "state": st.to_dict()}
                              for (s, m), st in self._states.items()],
                   "incidents": self.incidents.to_dict()}
        # Roundtrip detaches all nested mutable objects and rejects nonfinite state.
        return json.loads(json.dumps(payload, allow_nan=False))

    @classmethod
    def from_snapshot(cls, payload, on_event=None, on_notification=None):
        data = json.loads(json.dumps(payload, allow_nan=False))
        if data.get("schema_version") != STATE_SCHEMA_VERSION or data.get("engine_version") != ENGINE_VERSION:
            raise ValueError("unsupported SAFE state schema or engine version")
        obj = cls(**data["configuration"], on_event=on_event, on_notification=on_notification)
        if len(data["states"]) > obj.max_series:
            raise ValueError("snapshot exceeds max_series")
        for item in data["states"]:
            key = (item["sensor"], item["metric"])
            if key in obj._states:
                raise ValueError("duplicate snapshot series")
            obj.register(*key)
            obj._states[key] = _HealthState.from_dict(item["state"])
        obj._clock = data["clock"]
        obj.incidents = IncidentManager.from_dict(data["incidents"], on_event=on_event,
                                                 on_notification=on_notification)
        return obj

    def save_state(self, path):
        """Atomic replacement with SHA-256 corruption detection; no pickle execution."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        canonical = json.dumps(data, sort_keys=True, allow_nan=False).encode()
        envelope = {"sha256": hashlib.sha256(canonical).hexdigest(), "payload": data}
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".", delete=False) as handle:
                name = handle.name
                json.dump(envelope, handle, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load_state(cls, path, **callbacks):
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        canonical = json.dumps(envelope["payload"], sort_keys=True, allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != envelope["sha256"]:
            raise ValueError("state checksum mismatch")
        return cls.from_snapshot(envelope["payload"], **callbacks)
