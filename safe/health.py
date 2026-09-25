"""Incident-based sensor health engine: SAFE's single streaming detector."""

from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from safe.baseline import SeasonalBaseline, robust_scale
from safe.config import (
    ANOMALY_EXCESS_FRACTION,
    ANOMALY_EXCESS_MIN,
    ANOMALY_EXCESS_MIN_EACH_SIGN,
    CADENCE_DEGRADATION_FACTOR,
    CADENCE_HISTORY,
    CADENCE_MIN_INTERVALS,
    CONFIDENCE_DEFAULT,
    CONFIDENCE_DEFINITE,
    CONFIDENCE_ISOLATED,
    CONFIDENCE_OBSERVED,
    CONFIDENCE_PERSISTENT,
    CONFIDENCE_RULE,
    CONFIDENCE_STATISTICAL,
    CONFIDENCE_WEAK,
    ENGINE_VERSION,
    INVALID_RUN_RESTART,
    NOISE_STD_RATIO,
    PC_METRICS,
    PM_METRICS,
    PROVISIONAL_MIN_SAMPLES,
    PROVISIONAL_OUTLIER_SCALES,
    PROVISIONAL_OUTLIER_STEPS,
    PROVISIONAL_WINDOW,
    STUCK_ZERO_MIN_RUN,
    STUCK_ZERO_SUPPORT,
    VARIABILITY_TOLERANCE_FACTOR,
    WINDOW_MIN_SAMPLES,
)
from safe.incidents import HealthEvent, IncidentManager
from safe.profiles import MetricProfile, ProfileRegistry, SensorRules
from safe.stats import median, sample_comparison

STATE_SCHEMA_VERSION = 1

READING_METADATA = frozenset({"dateTime", "unix_timestamp", "str_timestamp"})
# Families whose presence means a reading must not teach any model.
UNTRUSTED_FAMILIES = frozenset({"freeze", "plausibility"})
# Families that recover on the ordinary schedule; "change" has its own.
ROUTINE_FAMILIES = frozenset({"validity", "availability", "timestamp", "plausibility",
                              "freeze", "startup", "environment"})
_DEFAULT_RULES = SensorRules()

Findings = list[tuple[str, dict[str, Any]]]
BinContext = dict[str, float | None]


def utc_seconds(value: Any) -> float:
    """Elapsed UTC seconds from a number or a timestamp string (naive means UTC)."""
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

    def __init__(self, delta: float = 0.25, lam: float = 18.0) -> None:
        if not np.isfinite(delta) or delta < 0 or not np.isfinite(lam) or lam <= 0:
            raise ValueError("delta must be finite and nonnegative; lam must be finite and positive")
        self.delta = delta
        self.lam = lam
        self.reset()

    def reset(self) -> None:
        self._m_up = 0.0
        self._min_up = 0.0
        self._m_down = 0.0
        self._min_down = 0.0
        self.samples = 0

    def update(self, residual: float) -> str | None:
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


class _CountedDeque(deque):
    """Bounded deque that counts items by key, so window statistics cost O(1).

    It is still a deque, so state serialization is unchanged. Only append, extend,
    pop, popleft, and clear keep the counts; do not use other mutators.
    """

    def __init__(self, key: Callable[[Any], Any], iterable: Iterable = (), maxlen: int | None = None) -> None:
        super().__init__(maxlen=maxlen)
        self.key = key
        self.counts: Counter = Counter()
        self.extend(iterable)

    def _discount(self, item: Any) -> None:
        key = self.key(item)
        self.counts[key] -= 1
        if not self.counts[key]:
            del self.counts[key]

    def append(self, item: Any) -> None:
        if len(self) == self.maxlen:
            self._discount(self[0])  # deque evicts the oldest item
        super().append(item)
        self.counts[self.key(item)] += 1

    def extend(self, items: Iterable) -> None:
        for item in items:
            self.append(item)

    def pop(self) -> Any:
        item = super().pop()
        self._discount(item)
        return item

    def popleft(self) -> Any:
        item = super().popleft()
        self._discount(item)
        return item

    def clear(self) -> None:
        super().clear()
        self.counts.clear()


def _peer_sample(p: MetricProfile, value: float, peer: float) -> float:
    """One reading's target/reference relationship: a difference, or a log ratio."""
    c = p.reference_ratio_floor
    if not c:
        return value - peer
    # Clamp so a value below -floor (only possible without hard bounds) stays finite.
    return math.log(max(value + c, 1e-3 * c) / max(peer + c, 1e-3 * c))


def _peer_expected(p: MetricProfile, peer: float, offset: float) -> float:
    """Target value predicted from the reference and the learned relationship."""
    c = p.reference_ratio_floor
    return (peer + c) * math.exp(offset) - c if c else peer + offset


def _measurements(sensor_dict: dict[str, Any]) -> dict[str, float]:
    """Numeric measurements of a reading; None is an absent field."""
    values = {}
    for metric, value in sensor_dict.items():
        if metric in READING_METADATA or value is None:
            continue
        try:
            values[metric] = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"nonnumeric measurement: {metric}") from None
    return values


def _ordered_present(values: dict[str, float], rule: SensorRules) -> list[str]:
    """The rule's size-ordered metrics present in this reading with finite values."""
    return [m for m in rule.ordered_metrics if m in values and math.isfinite(values[m])]


def _bin_context(values: dict[str, float], ordered: list[str]) -> dict[str, BinContext]:
    """Per ordered bin: the largest smaller bin and the next larger bin in this reading."""
    return {m: dict(smaller_max=max((values[s] for s in ordered[:i]), default=None),
                    larger=values[ordered[i + 1]] if i + 1 < len(ordered) else None)
            for i, m in enumerate(ordered)}


# Histories persisted as bounded lists of the profile's max_samples.
_BOUNDED_HISTORIES = ("warmup", "residuals", "reference_levels", "arrivals", "recent_anomalies",
                      "peer_samples", "bin_shares")


class _HealthState:
    """Everything the engine remembers about one (sensor, metric) series."""

    def __init__(self, profile: MetricProfile) -> None:
        self.profile = profile
        self.baseline = SeasonalBaseline(profile)
        self.ph = PageHinkley()
        interval = profile.expected_interval_seconds
        bound = profile.max_samples
        self.warmup = _CountedDeque(lambda item: int(item[0] // interval), maxlen=bound)  # (t, value)
        self.residuals = deque(maxlen=bound)
        self.reference_levels = deque(maxlen=bound)  # (t, reference-predicted value)
        self.recent_anomalies = _CountedDeque(lambda item: item[1], maxlen=bound)  # (t, sign or 0)
        self.peer_samples = deque(maxlen=bound)
        self.arrivals = _CountedDeque(lambda stamp: int((stamp - self.first_at) // interval),
                                      maxlen=bound)  # first arrival per cadence slot
        self.cadences = deque(maxlen=CADENCE_HISTORY)
        self.first_at: float | None = None
        self.last_at: float | None = None
        self.last_value: float | None = None
        self.freeze_anchor: float | None = None
        self.freeze_started: float | None = None
        self.freeze_count = 0
        self.zero_contradictions = 0
        self.bin_shares = deque(maxlen=bound)  # value / next larger cumulative bin
        self.has_variability = False
        self.run_sign = 0
        self.run_count = 0
        self.run_started: float | None = None
        self.last_evaluation: float | None = None
        self.evaluations = 0
        self.mean_confirmations = 0
        self.variance_confirmations = 0
        self.invalid_run = 0
        self.startup_run = 0
        self.warmup_restarts = 0
        self.mode = "WARMING_UP"
        self.peer_mode = False
        self.peer_offset: float | None = None
        self.peer_spread: float | None = None  # ratio mode: typical |log ratio| scatter around peer_offset
        self.window_category: str | None = None
        self.window_evidence: dict[str, Any] = {}
        self.reference_drift_confirmations = 0
        self.reference_drift_evidence: dict[str, Any] | None = None

    def reset_change_tracking(self) -> None:
        """Forget change evidence that no longer describes the current signal."""
        self.run_count = self.run_sign = 0
        self.ph.reset()
        self.mean_confirmations = self.variance_confirmations = 0
        self.window_category, self.window_evidence = None, {}
        self.reference_drift_confirmations = 0
        self.reference_drift_evidence = None

    def to_dict(self) -> dict[str, Any]:
        output = {}
        for key, value in vars(self).items():
            if key in ("profile", "baseline"):
                output[key] = value.to_dict()
            elif key == "ph":
                output[key] = vars(value).copy()
            elif isinstance(value, deque):
                output[key] = list(value)
            else:
                output[key] = value
        return output

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_HealthState":
        obj = cls(MetricProfile(**data["profile"]))
        histories = {}
        for key, value in data.items():
            if key == "profile":
                continue
            if key == "baseline":
                obj.baseline = SeasonalBaseline.from_dict(obj.profile, value)
            elif key == "ph":
                obj.ph = PageHinkley(value["delta"], value["lam"])
                vars(obj.ph).update(value)
            elif key in _BOUNDED_HISTORIES:
                if len(value) > obj.profile.max_samples:
                    raise ValueError("state history exceeds configured bound")
                histories[key] = value
            elif key == "cadences":
                if len(value) > CADENCE_HISTORY:
                    raise ValueError("state cadence history exceeds bound")
                obj.cadences.extend(value)
            elif key not in vars(obj):
                raise ValueError("unknown health state field")
            else:
                setattr(obj, key, value)
        # Counted histories key on scalar fields such as first_at, so fill them last.
        for key, value in histories.items():
            getattr(obj, key).extend(value)
        return obj


@dataclass(slots=True)
class _Residual:
    """One reading's comparison with its expected value during detection."""

    value: float
    t: float
    expected: float
    scale: float
    residual: float
    z: float
    anomalous: bool
    environmental: bool
    peer: float | None


class SensorHealth:
    """Feed data_processing(sensor, reading), and tick(clock) during silence.

    Per-metric timestamps must increase; equal sensor timestamps may carry disjoint
    fields. Invalid values become evidence and never enter the environmental model.
    References are explicit, co-located/compatible measurements supplied by callers;
    the engine never guesses geographic neighbors or treats label data as inputs.
    """

    def __init__(self, profiles: ProfileRegistry | dict | None = None, sensors: dict | None = None,
                 rules: dict | None = None, history_limit: int = 1000, max_series: int = 1000,
                 notification_interval_seconds: float = 1800,
                 on_event: Callable[[str, dict], Any] | None = None,
                 on_notification: Callable[[dict], Any] | None = None) -> None:
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
        self._states: dict[tuple[str, str], _HealthState] = {}
        self._clock: float | None = None

    def configuration(self) -> dict[str, Any]:
        """Constructor arguments that reproduce this engine (as saved in snapshots)."""
        return dict(profiles=self.profiles.to_dict(), sensors=self.sensors,
                    rules={k: asdict(v) for k, v in self.rules.items()},
                    max_series=self.max_series, history_limit=self.incidents.history.maxlen,
                    notification_interval_seconds=self.incidents.notification_interval_seconds)

    @property
    def events(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.incidents.events()]

    def register(self, sensor: str, metric: str, started_at: Any = None) -> _HealthState:
        """Return the series state, creating it (optionally observed since started_at)."""
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

    def retire(self, sensor: str, metric: str) -> None:
        """Explicit deployment removal; refuses to discard active incidents."""
        if any(k[:2] == (sensor, metric) for k in self.incidents.active):
            raise ValueError("cannot retire a series with active incidents")
        self._states.pop((sensor, metric), None)
        self.incidents.last_notification.pop((sensor, metric), None)

    def _observe(self, sensor: str, metric: str, category: str, family: str, t: float,
                 severity: str = "warning", confidence: float = CONFIDENCE_DEFAULT,
                 **evidence: Any) -> HealthEvent:
        state = self._states[(sensor, metric)]
        if severity != "info":
            state.mode = "INCIDENT"
        elif state.mode == "MONITORING":
            state.mode = "SUSPECTED"
        configuration = dict(profile=state.profile.to_dict(), **self.sensors.get(sensor, {}))
        return self.incidents.observe(sensor, metric, category, family, t, severity,
                                      confidence, evidence, configuration)

    def tick(self, current_time: Any) -> None:
        """Advance the monotonic clock and report every series silent for too long."""
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
                              confidence=CONFIDENCE_OBSERVED, last_reading_at=state.last_at,
                              silence_seconds=t - anchor, expected_interval=p.expected_interval_seconds)
                state.mode = "INCIDENT"

    # ------------------------------------------------------------------
    # Reading ingestion
    # ------------------------------------------------------------------

    def data_processing(self, sensor_name: str, sensor_dict: dict[str, Any],
                        references: dict[str, list[dict]] | None = None, received_at: Any = None) -> None:
        """references: metric -> [{value, timestamp, sensor}, ...], >=2 peers.

        One trusted reference is supported with trusted=True. Reference timestamps
        must be no later than this reading and no older than one nominal interval.
        """
        stamp = sensor_dict.get("unix_timestamp", sensor_dict.get("dateTime"))
        if stamp is None:
            raise ValueError("reading requires dateTime or unix_timestamp")
        t = utc_seconds(stamp)
        arrival = utc_seconds(received_at) if received_at is not None else None
        values = _measurements(sensor_dict)
        new_keys = {(sensor_name, metric) for metric in values} - self._states.keys()
        if len(self._states) + len(new_keys) > self.max_series:
            raise ValueError("max_series exceeded")
        if arrival is not None:
            values = self._accept_arrival(sensor_name, values, t, arrival)
            if not values:
                return
        rule = self.rules.get(self.sensors.get(sensor_name, {}).get("model"), _DEFAULT_RULES)
        ordered = _ordered_present(values, rule)
        rule_findings = self._plausibility(values, rule, ordered)
        self._flag_simultaneous_jumps(sensor_name, values, rule, rule_findings)
        # Normal ingestion advances a clock so a later resumed reading exposes a gap.
        # Explicit tick calls are still needed to detect a sensor that never resumes.
        if arrival is None and (self._clock is None or t > self._clock):
            self.tick(t)
        bins = _bin_context(values, ordered)
        references = references or {}
        for metric, value in values.items():
            self._process(sensor_name, metric, value, t, rule_findings.get(metric, []),
                          references.get(metric), bins.get(metric))

    def _accept_arrival(self, sensor: str, values: dict[str, float], t: float,
                        arrival: float) -> dict[str, float]:
        """Advance the clock to the arrival time and drop readings stamped in the future."""
        self.tick(arrival)
        accepted = {}
        for metric, value in values.items():
            state = self.register(sensor, metric, arrival)
            if t > arrival + state.profile.expected_interval_seconds:
                self._observe(sensor, metric, "future_timestamp", "timestamp", arrival,
                              confidence=CONFIDENCE_OBSERVED, rejected_timestamp=t, received_at=arrival)
            else:
                accepted[metric] = value
        return accepted

    def _flag_simultaneous_jumps(self, sensor: str, values: dict[str, float], rule: SensorRules,
                                 findings: dict[str, Findings]) -> None:
        """Registers every metric; many metrics stepping at once suggests a restart."""
        jumps = []
        for metric, value in values.items():
            state = self.register(sensor, metric)
            if (state.last_value is not None and math.isfinite(value)
                    and abs(value - state.last_value) >= state.profile.step_min_effect):
                jumps.append(metric)
        if len(jumps) >= rule.simultaneous_jump_metrics:
            for metric in jumps:
                findings.setdefault(metric, []).append(("possible_restart", {"jumped_metrics": jumps}))

    def _plausibility(self, values: dict[str, float], rule: SensorRules,
                      ordered: list[str] | None = None) -> dict[str, Findings]:
        """Cross-metric rule violations in one reading, per metric."""
        findings: dict[str, Findings] = {}
        if ordered is None:
            ordered = _ordered_present(values, rule)
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

    def _peer(self, sensor: str, p: MetricProfile, t: float, references: list[dict] | None) -> float | None:
        """Consensus of timely references: a trusted median, else a median of >= 2 peers."""
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
            return median(trusted)
        return median(valid.values()) if len(valid) >= 2 else None

    def _zero_contradictions(self, state: _HealthState, value: float, peer: float | None,
                             bins: BinContext | None) -> dict[str, float]:
        """Evidence that a zero reading is not clean air, as {source: predicted value}."""
        p = state.profile
        level = p.stuck_zero_min_expected
        if not level or value > p.freeze_tolerance:
            return {}
        evidence = {}
        if peer is not None and peer >= level:
            evidence["reference"] = peer
        if bins and bins["smaller_max"] is not None and bins["smaller_max"] >= level:
            # Cumulative bins: this bin can never read below a smaller one.
            evidence["smaller_bin"] = bins["smaller_max"]
        if bins and bins["larger"] is not None and len(state.bin_shares) >= p.minimum_samples:
            predicted = float(np.median(state.bin_shares)) * bins["larger"]
            if predicted >= level:
                evidence["larger_bin_share"] = predicted
        return evidence

    # ------------------------------------------------------------------
    # Per-series processing
    # ------------------------------------------------------------------

    def _process(self, sensor: str, metric: str, value: float, t: float, findings: Findings,
                 references: list[dict] | None, bins: BinContext | None = None) -> None:
        state = self._states[(sensor, metric)]
        if self._rejected_timestamp(sensor, metric, state, t):
            return
        seen: set[str] = set()
        self._track_availability(sensor, metric, state, t, seen)
        for category, evidence in findings:
            restart = category == "possible_restart"
            family = "change" if restart else "plausibility"
            seen.add(family)
            self._observe(sensor, metric, category, family, t,
                          severity="info" if restart else "warning",
                          confidence=CONFIDENCE_WEAK if restart else CONFIDENCE_RULE, **evidence)
        if self._invalid(sensor, metric, state, value, t, seen):
            return
        state.invalid_run = 0
        peer = self._peer(sensor, state.profile, t, references)
        self._check_freeze(sensor, metric, state, value, t, peer, bins, seen)
        self._learn_bin_share(state, value, bins, seen)
        state.last_value = value
        if not state.baseline.ready:
            if peer is not None and not (UNTRUSTED_FAMILIES & seen):
                state.peer_samples.append(_peer_sample(state.profile, value, peer))
            self._warmup(sensor, metric, state, value, t, seen)
        else:
            self._detect(sensor, metric, state, value, t, peer, seen)
        self._recover(sensor, metric, state, t, seen)

    def _rejected_timestamp(self, sensor: str, metric: str, state: _HealthState, t: float) -> bool:
        """Report and reject a reading that is not newer than the last or arrives too late."""
        p = state.profile
        if state.last_at is not None and t <= state.last_at:
            self._observe(sensor, metric, "duplicate_timestamp" if t == state.last_at else "timestamp_disorder",
                          "timestamp", max(t, state.last_at), confidence=CONFIDENCE_OBSERVED,
                          rejected_timestamp=t)
            return True
        if self._clock is not None and self._clock - t >= p.gap_factor * p.expected_interval_seconds:
            self._observe(sensor, metric, "delayed_data", "availability", self._clock,
                          confidence=CONFIDENCE_OBSERVED, rejected_timestamp=t, delay_seconds=self._clock - t)
            return True
        return False

    def _track_availability(self, sensor: str, metric: str, state: _HealthState, t: float,
                            seen: set[str]) -> None:
        """Record the arrival; handle a resumed gap, completeness loss, and slow cadence."""
        p = state.profile
        interval = p.expected_interval_seconds
        if state.first_at is None:
            state.first_at = t
        previous = state.last_at
        state.last_at = t
        slot = math.floor((t - state.first_at) / interval)
        if not state.arrivals or slot != math.floor((state.arrivals[-1] - state.first_at) / interval):
            state.arrivals.append(t)
        if previous is not None:
            state.cadences.append(t - previous)
        while state.arrivals and t - state.arrivals[0] > p.completeness_window_seconds:
            state.arrivals.popleft()
        if previous is not None and t - previous >= interval * p.gap_factor:
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
            occupied = len(state.arrivals.counts)
            expected = math.floor(duration / interval) + 1
            completeness = min(1.0, occupied / expected)
            if completeness < p.minimum_completeness:
                seen.add("availability")
                self._observe(sensor, metric, "completeness_loss", "availability", t,
                              completeness=completeness, expected_slots=expected, occupied_slots=occupied)
        if len(state.cadences) >= CADENCE_MIN_INTERVALS:
            cadence = median(state.cadences)
            if cadence > CADENCE_DEGRADATION_FACTOR * interval:
                seen.add("availability")
                self._observe(sensor, metric, "cadence_degradation", "availability", t,
                              observed_interval=cadence, expected_interval=interval)

    def _invalid(self, sensor: str, metric: str, state: _HealthState, value: float, t: float,
                 seen: set[str]) -> bool:
        """Report a nonfinite or out-of-bounds value; repeated ones restart an unfinished warmup."""
        p = state.profile
        if math.isfinite(value) and (p.hard_bounds is None or p.hard_bounds[0] <= value <= p.hard_bounds[1]):
            return False
        seen.add("validity")
        self._observe(sensor, metric, "invalid_measurement", "validity", t, "critical", CONFIDENCE_DEFINITE,
                      value=value if math.isfinite(value) else None, bounds=p.hard_bounds)
        state.invalid_run += 1
        state.freeze_started = None
        state.run_count = 0
        if state.invalid_run >= INVALID_RUN_RESTART and not state.baseline.ready:
            state.warmup.clear()
            state.warmup_restarts += 1
        state.mode = "INCIDENT"
        return True

    def _check_freeze(self, sensor: str, metric: str, state: _HealthState, value: float, t: float,
                      peer: float | None, bins: BinContext | None, seen: set[str]) -> None:
        """Track runs of a constant value and report a confirmed freeze."""
        p = state.profile
        if state.freeze_started is None or abs(value - state.freeze_anchor) > p.freeze_tolerance:
            if (state.freeze_anchor is not None
                    and abs(value - state.freeze_anchor) > VARIABILITY_TOLERANCE_FACTOR * p.freeze_tolerance):
                state.has_variability = True
            state.freeze_anchor = value
            state.freeze_started = t
            state.freeze_count = 1
            state.zero_contradictions = 0
        else:
            state.freeze_count += 1
        contradictions = self._zero_contradictions(state, value, peer, bins)
        state.zero_contradictions += bool(contradictions)
        # Clean air legitimately holds PM and particle-count bins at zero for hours;
        # a zero run is stuck only when most of it contradicts clean air.
        zero = (metric in PM_METRICS or metric in PC_METRICS) and value <= p.freeze_tolerance
        stuck_zero = zero and state.zero_contradictions >= STUCK_ZERO_SUPPORT * state.freeze_count
        if (state.freeze_count >= p.freeze_min_readings and t - state.freeze_started >= p.freeze_duration_seconds
                and (p.freeze_at_startup or state.has_variability) and (not zero or stuck_zero)):
            seen.add("freeze")
            extra = dict(stuck_at_zero=True, contradicted_readings=state.zero_contradictions,
                         contradiction_evidence=contradictions) if zero else {}
            self._observe(sensor, metric, "sensor_freeze", "freeze", t, confidence=CONFIDENCE_RULE,
                          run_length=state.freeze_count, duration_seconds=t - state.freeze_started,
                          tolerance=p.freeze_tolerance, value=value, **extra)

    @staticmethod
    def _learn_bin_share(state: _HealthState, value: float, bins: BinContext | None, seen: set[str]) -> None:
        """Learn this bin's share of the next larger bin from trusted nonzero readings."""
        level = state.profile.stuck_zero_min_expected
        if (bins and bins["larger"] is not None and level and bins["larger"] >= level
                and value > state.profile.freeze_tolerance and not (UNTRUSTED_FAMILIES & seen)):
            # Learn only from nonzero readings so a dead channel cannot teach a zero share.
            state.bin_shares.append(value / bins["larger"])

    def _recover(self, sensor: str, metric: str, state: _HealthState, t: float, seen: set[str]) -> None:
        """Advance recovery of incidents not seen in this reading, then update the series mode."""
        p = state.profile
        change = self.incidents.active.get((sensor, metric, "change"))
        self.incidents.recover(sensor, metric, t, seen,
                               p.window_seconds if change and change.category == "noise_change" else p.recovery_seconds,
                               p.recovery_readings, eligible_families={"change"})
        self.incidents.recover(sensor, metric, t, seen, p.recovery_seconds, p.recovery_readings,
                               eligible_families=ROUTINE_FAMILIES)
        active = [e for k, e in self.incidents.active.items() if k[:2] == (sensor, metric)]
        state.mode = ("RECOVERING" if active and all(e.status == "recovering" for e in active)
                      else "INCIDENT" if any(e.severity != "info" for e in active)
                      else "SUSPECTED" if active else "MONITORING" if state.baseline.ready else "WARMING_UP")

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self, sensor: str, metric: str, state: _HealthState, value: float, t: float,
                seen: set[str]) -> None:
        p = state.profile
        if self._startup_contaminated(sensor, metric, state, value, t, seen):
            return
        state.startup_run = 0
        if not (UNTRUSTED_FAMILIES & seen):
            state.warmup.append((t, value))
        if len(state.warmup) < p.minimum_samples:
            return
        while len(state.warmup) > p.minimum_samples and t - state.warmup[0][0] > 2 * p.warmup_duration_seconds:
            state.warmup.popleft()
        elapsed = t - state.warmup[0][0]
        slots = len(state.warmup.counts)
        expected = math.floor(elapsed / p.expected_interval_seconds) + 1
        if (elapsed >= p.warmup_duration_seconds and slots / expected >= p.minimum_coverage
                and state.baseline.fit(list(state.warmup))):
            state.warmup.clear()
            state.ph.reset()
            state.last_evaluation = t

    def _startup_contaminated(self, sensor: str, metric: str, state: _HealthState, value: float,
                              t: float, seen: set[str]) -> bool:
        """Provisional gate for extreme startup values; a persistent run restarts warmup.

        It only catches extreme contamination; ordinary daily movement must still
        populate a full cycle before residual detection.
        """
        p = state.profile
        if len(state.warmup) < PROVISIONAL_MIN_SAMPLES:
            return False
        recent = [state.warmup[i][1] for i in range(-min(PROVISIONAL_WINDOW, len(state.warmup)), 0)]
        center, scale = robust_scale(recent, p.residual_scale_floor)
        if abs(value - center) <= max(PROVISIONAL_OUTLIER_SCALES * scale,
                                      PROVISIONAL_OUTLIER_STEPS * p.step_min_effect):
            return False
        state.startup_run += 1
        seen.add("startup")
        self._observe(sensor, metric, "startup_contamination", "startup", t, "info", CONFIDENCE_WEAK,
                      value=value, provisional_center=center)
        if state.startup_run >= p.step_readings:
            state.warmup.clear()
            state.warmup_restarts += 1
        return True

    # ------------------------------------------------------------------
    # Detection after warmup
    # ------------------------------------------------------------------

    def _detect(self, sensor: str, metric: str, state: _HealthState, value: float, t: float,
                peer: float | None, seen: set[str]) -> None:
        p = state.profile
        ambient_expected, scale = state.baseline.predict(t)
        expected = ambient_expected
        peer = self._calibrated_peer(state, value, expected, peer)
        if peer is not None:
            expected = _peer_expected(p, peer, state.peer_offset)
            if state.peer_spread is not None:
                # Healthy gain-type sensors disagree with a reference by a roughly
                # constant percentage, so the residual scale grows with the level.
                # The ambient scale, learned mostly at lower levels, stays the floor.
                scale = max(scale, state.peer_spread * (expected + p.reference_ratio_floor))
        if self._held_for_freeze(sensor, metric, state, value, seen):
            return
        residual = value - expected
        z = residual / scale
        r = _Residual(value, t, expected, scale, residual, z,
                      anomalous=abs(z) >= p.outlier_threshold and abs(residual) >= p.step_min_effect,
                      environmental=metric in PM_METRICS and peer is None, peer=peer)
        if peer is not None and metric in PM_METRICS and not r.anomalous:
            if abs(value - ambient_expected) >= p.step_min_effect:
                seen.add("environment")
                self._observe(sensor, metric, "environmental_event", "environment", t, "info",
                              CONFIDENCE_STATISTICAL, value=value, peer_consensus=peer, residual=residual)
        self._track_anomaly_run(sensor, metric, state, r, seen)
        self._check_anomaly_excess(sensor, metric, state, r, seen)
        if p.enable_page_hinkley and state.baseline.ready and not r.environmental:
            direction = state.ph.update(z)
            if direction:
                seen.add("change")
                self._observe(sensor, metric, "gradual_degradation", "change", t,
                              confidence=CONFIDENCE_STATISTICAL, direction=direction, residual=residual,
                              stationary_residuals=True)
        state.residuals.append((t, residual))
        while state.residuals and t - state.residuals[0][0] > p.window_seconds:
            state.residuals.popleft()
        if peer is not None:
            state.reference_levels.append((t, expected))
            while t - state.reference_levels[0][0] > p.window_seconds:
                state.reference_levels.popleft()
        self._report_window_evidence(sensor, metric, state, t, r.environmental, seen)
        if not r.anomalous and not (UNTRUSTED_FAMILIES & seen) and not p.stationary_residuals:
            before = state.baseline.revision
            state.baseline.learn(t, value)
            if state.baseline.revision != before:
                # Any changing expected baseline invalidates a cumulative test's origin.
                state.ph.reset()

    def _calibrated_peer(self, state: _HealthState, value: float, expected: float,
                         peer: float | None) -> float | None:
        """Learn the target/reference relationship; return the peer only once it is known.

        Switching between reference and ambient models restarts residual history.
        """
        p = state.profile
        if peer is not None and state.peer_offset is None:
            if len(state.peer_samples) >= p.minimum_samples:
                state.peer_offset = float(np.median(state.peer_samples))
                if p.reference_ratio_floor:
                    state.peer_spread = robust_scale(list(state.peer_samples), 0.0)[1]
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
        return peer

    def _held_for_freeze(self, sensor: str, metric: str, state: _HealthState, value: float,
                         seen: set[str]) -> bool:
        """Skip change detection for stuck readings, which are the freeze incident's evidence."""
        p = state.profile
        frozen = "freeze" in seen or (sensor, metric, "freeze") in self.incidents.active
        if not frozen:
            # A contradicted zero run is a pending stuck-at-zero: hold shift/drift until
            # the freeze check confirms it (or the run breaks) instead of misdiagnosing.
            # Skip the reading without discarding evidence gathered before the run.
            return bool(p.stuck_zero_min_expected and value <= p.freeze_tolerance
                        and state.freeze_count >= STUCK_ZERO_MIN_RUN
                        and state.zero_contradictions >= STUCK_ZERO_SUPPORT * state.freeze_count)
        # A stuck value disagrees with the ambient model and any reference, but that is
        # the freeze incident's evidence, not a second shift or drift. Drop the stuck
        # readings gathered before the freeze was confirmed so they cannot raise drift
        # once the sensor recovers.
        if "freeze" in seen and state.freeze_started is not None:
            for history in (state.residuals, state.reference_levels, state.recent_anomalies):
                while history and history[-1][0] >= state.freeze_started:
                    history.pop()
        state.reset_change_tracking()
        return True

    def _track_anomaly_run(self, sensor: str, metric: str, state: _HealthState, r: _Residual,
                           seen: set[str]) -> None:
        """Report residual anomalies; a same-sign run of step_readings is a persistent shift."""
        p = state.profile
        if not r.anomalous:
            state.run_count = 0
            state.run_sign = 0
            return
        sign = 1 if r.residual > 0 else -1
        if state.run_sign != sign:
            state.run_count = 0
        if state.run_count == 0:
            state.run_started = r.t
        state.run_sign = sign
        state.run_count += 1
        persistent = state.run_count >= p.step_readings
        actionable = persistent and not r.environmental
        category = ("uncertain_change" if r.environmental and persistent else
                    "abrupt_shift" if persistent else "isolated_anomaly")
        seen.add("change")
        self._observe(sensor, metric, category, "change", r.t,
                      "warning" if actionable else "info",
                      CONFIDENCE_PERSISTENT if actionable else CONFIDENCE_ISOLATED,
                      value=r.value, expected=r.expected, residual=r.residual, residual_scale=r.scale,
                      standardized_residual=r.z, consecutive_readings=state.run_count,
                      peer_reference=r.peer, interpretation="change evidence; cause unconfirmed")

    def _check_anomaly_excess(self, sensor: str, metric: str, state: _HealthState, r: _Residual,
                              seen: set[str]) -> None:
        """Report a window with persistently many two-sided anomalies (a noise change)."""
        p = state.profile
        recent = state.recent_anomalies
        recent.append((r.t, (1 if r.residual > 0 else -1) if r.anomalous else 0))
        while recent and r.t - recent[0][0] > p.window_seconds:
            recent.popleft()
        positive, negative = recent.counts[1], recent.counts[-1]
        if (len(recent) >= p.minimum_samples and min(positive, negative) >= ANOMALY_EXCESS_MIN_EACH_SIGN
                and positive + negative >= max(ANOMALY_EXCESS_MIN, ANOMALY_EXCESS_FRACTION * len(recent))
                and state.run_count < p.step_readings):
            seen.add("change")
            self._observe(sensor, metric, "uncertain_change" if r.environmental else "noise_change",
                          "change", r.t, "info" if r.environmental else "warning", CONFIDENCE_STATISTICAL,
                          anomalous_readings=positive + negative,
                          window_readings=len(recent),
                          detail="persistent excess of residual anomalies; cause unconfirmed")

    def _report_window_evidence(self, sensor: str, metric: str, state: _HealthState, t: float,
                                environmental: bool, seen: set[str]) -> None:
        """Run the periodic window tests when due, and re-report any standing result."""
        p = state.profile
        if state.last_evaluation is None or t - state.last_evaluation >= p.evaluation_interval_seconds:
            state.last_evaluation = t
            self._window_test(metric, state, t, seen, environmental)
            self._reference_drift_test(state)
        if state.window_category is not None:
            seen.add("change")
            self._observe(sensor, metric, state.window_category, "change", t,
                          "info" if environmental else "warning", CONFIDENCE_STATISTICAL,
                          **state.window_evidence)
        if state.reference_drift_evidence is not None:
            seen.add("change")
            self._observe(sensor, metric, "gradual_degradation", "change", t, "warning", CONFIDENCE_DEFAULT,
                          **state.reference_drift_evidence)

    def _window_test(self, metric: str, state: _HealthState, t: float, seen: set[str],
                     environmental: bool) -> None:
        """Compare the older and newer halves of the residual window for mean/variance shifts."""
        p = state.profile
        midpoint = t - p.window_seconds / 2
        old = [v for stamp, v in state.residuals if stamp < midpoint]
        new = [v for stamp, v in state.residuals if stamp >= midpoint]
        if min(len(old), len(new)) < max(WINDOW_MIN_SAMPLES, p.minimum_samples // 2):
            return
        state.evaluations += 1
        # Summable alpha spending: sum_k alpha/[2*k*(k+1)] per test is alpha/2.
        # This controls the budget across both test families if each p-value is valid;
        # the underlying AR(1) approximation does not establish field calibration.
        alpha = p.family_alpha / (2 * state.evaluations * (state.evaluations + 1))
        result = sample_comparison(old, new, alpha, metric, autocorr_correction=True)
        mean = result["mean_shift"] and abs(result["mean_delta"]) >= p.step_min_effect
        variance = result["variance_shift"] and (result["std_ratio"] >= NOISE_STD_RATIO
                                                 or result["std_ratio"] <= 1 / NOISE_STD_RATIO)
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

    def _reference_drift_test(self, state: _HealthState) -> None:
        """Calibration drift: sustained disagreement with a reference.

        Residuals against a reference cancel shared weather, so a persistent
        nonzero median is attributable to this sensor. The median over the
        window is robust to reference glitches; the tolerance is a practical
        calibration limit, not a significance level.
        """
        p = state.profile
        values = [v for _, v in state.residuals]
        if not state.peer_mode or len(values) < max(WINDOW_MIN_SAMPLES, p.minimum_samples // 2):
            state.reference_drift_confirmations = 0
            state.reference_drift_evidence = None
            return
        drift = float(np.median(values))
        level = float(np.median([v for _, v in state.reference_levels])) if state.reference_levels else 0.0
        tolerance = max(p.reference_drift_tolerance, p.reference_drift_relative_tolerance * abs(level))
        if abs(drift) < tolerance:
            state.reference_drift_confirmations = 0
            state.reference_drift_evidence = None
            return
        state.reference_drift_confirmations += 1
        if state.reference_drift_confirmations >= p.persistence_evaluations:
            state.reference_drift_evidence = dict(
                evidence_source="reference", median_reference_residual=drift,
                reference_offset=state.peer_offset,
                reference_model="ratio" if p.reference_ratio_floor else "difference",
                window_readings=len(values),
                tolerance=tolerance, reference_level=level,
                interpretation="sustained disagreement with reference; calibration drift suspected")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe copy of the full engine state (configuration, series, incidents)."""
        payload = {"schema_version": STATE_SCHEMA_VERSION, "engine_version": ENGINE_VERSION,
                   "configuration": self.configuration(), "clock": self._clock,
                   "states": [{"sensor": s, "metric": m, "state": st.to_dict()}
                              for (s, m), st in self._states.items()],
                   "incidents": self.incidents.to_dict()}
        # Roundtrip detaches all nested mutable objects and rejects nonfinite state.
        return json.loads(json.dumps(payload, allow_nan=False))

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any], on_event: Callable[[str, dict], Any] | None = None,
                      on_notification: Callable[[dict], Any] | None = None) -> "SensorHealth":
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

    def save_state(self, path: str | os.PathLike) -> None:
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
    def load_state(cls, path: str | os.PathLike, **callbacks: Any) -> "SensorHealth":
        """Load a save_state file after verifying its checksum and engine version."""
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        canonical = json.dumps(envelope["payload"], sort_keys=True, allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != envelope["sha256"]:
            raise ValueError("state checksum mismatch")
        return cls.from_snapshot(envelope["payload"], **callbacks)
