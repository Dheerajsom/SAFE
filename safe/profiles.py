"""Validated, serializable health profiles. Durations are elapsed UTC seconds."""

from dataclasses import asdict, dataclass, fields
import math

from safe.config import HARD_BOUNDS, PM_METRICS


@dataclass(frozen=True)
class MetricProfile:
    expected_interval_seconds: float = 300
    warmup_duration_seconds: float = 86400
    minimum_samples: int = 48
    minimum_coverage: float = 0.75
    window_seconds: float = 21600
    evaluation_interval_seconds: float = 3600
    outlier_threshold: float = 6
    residual_scale_floor: float = 0.3
    step_min_effect: float = 2
    step_readings: int = 8
    # Sustained disagreement with a reference beyond this calibration tolerance
    # is reported as drift. Only evaluated when a reference is supplied: a single
    # sensor cannot separate slow drift from a genuine ambient trend.
    reference_drift_tolerance: float = 0.5
    # 0 compares against a reference by a learned constant difference. A positive
    # value compares by a learned ratio of (value + floor) / (reference + floor),
    # for metrics whose co-located sensors differ by gain (PM): a fixed-percentage
    # difference then stays constant as concentrations rise. The floor keeps the
    # ratio stable near zero, where the comparison becomes nearly additive.
    reference_ratio_floor: float = 0
    # Drift must also exceed this fraction of the typical reference-predicted
    # level, like an accuracy spec of "± tolerance or ± percent, whichever is
    # larger": co-located PM sensors wander a few percent for hours, which is
    # several µg/m³ during pollution episodes. 0 keeps the absolute limit only.
    reference_drift_relative_tolerance: float = 0
    freeze_tolerance: float = 0.001
    freeze_duration_seconds: float = 3600
    freeze_min_readings: int = 12
    freeze_at_startup: bool = True
    gap_factor: float = 3
    completeness_window_seconds: float = 86400
    minimum_completeness: float = 0.8
    recovery_seconds: float = 1800
    recovery_readings: int = 3
    seasonal_period_seconds: float = 86400
    seasonal_bins: int = 48
    seasonal_cycles: int = 14
    seasonal_rate_per_day: float = 2
    family_alpha: float = 0.01
    persistence_evaluations: int = 2
    enable_page_hinkley: bool = False
    stationary_residuals: bool = False
    max_samples: int = 4096
    hard_bounds: tuple[float, float] | None = None

    def __post_init__(self):
        integer_names = {"minimum_samples", "step_readings", "freeze_min_readings",
                         "recovery_readings", "seasonal_bins", "seasonal_cycles",
                         "persistence_evaluations", "max_samples"}
        boolean_names = {"freeze_at_startup", "enable_page_hinkley", "stationary_residuals"}
        nonnegative = {"freeze_tolerance", "seasonal_rate_per_day", "reference_ratio_floor",
                       "reference_drift_relative_tolerance"}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "hard_bounds":
                if value is not None:
                    if (len(value) != 2 or not all(math.isfinite(x) for x in value)
                            or value[0] >= value[1]):
                        raise ValueError("hard_bounds must be two ordered finite numbers")
                    object.__setattr__(self, f.name, tuple(value))
            elif f.name in boolean_names:
                if not isinstance(value, bool):
                    raise ValueError(f"{f.name} must be boolean")
            elif f.name in integer_names:
                if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                    raise ValueError(f"{f.name} must be an integer >= 2")
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value) or value < 0
                  or (value == 0 and f.name not in nonnegative)):
                raise ValueError(f"{f.name} must be finite and positive")
        for name in ("minimum_coverage", "minimum_completeness", "family_alpha"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.gap_factor <= 1:
            raise ValueError("gap_factor must exceed one")
        if self.max_samples < self.minimum_samples:
            raise ValueError("max_samples must accommodate minimum_samples")
        required = math.ceil(max(self.warmup_duration_seconds, self.window_seconds,
                                self.completeness_window_seconds) / self.expected_interval_seconds) + 2
        if self.max_samples < required:
            raise ValueError("max_samples cannot cover the configured elapsed-time windows at this cadence")
        if self.enable_page_hinkley and not self.stationary_residuals:
            raise ValueError("Page-Hinkley requires explicitly stationary residuals")

    def to_dict(self):
        return asdict(self)


def metric_profile(metric, deployment="outdoor", expected_interval_seconds=300):
    if deployment not in {"outdoor", "indoor", "mobile", "laboratory"}:
        raise ValueError("unknown deployment type")
    options = dict(expected_interval_seconds=expected_interval_seconds,
                   hard_bounds=HARD_BOUNDS.get(metric),
                   max_samples=max(4096, math.ceil(86400 / expected_interval_seconds) + 2)
                   if isinstance(expected_interval_seconds, (int, float)) and expected_interval_seconds > 0 else 4096)
    if metric == "humidity":
        options.update(residual_scale_floor=1, step_min_effect=5, freeze_tolerance=0.01,
                       seasonal_rate_per_day=5, reference_drift_tolerance=3)
    elif metric == "pressure":
        options.update(residual_scale_floor=0.5, step_min_effect=3,
                       seasonal_rate_per_day=5, reference_drift_tolerance=1)
    elif metric in PM_METRICS or metric.startswith("pc"):
        options.update(residual_scale_floor=1, step_min_effect=5,
                       freeze_at_startup=False, freeze_tolerance=0,
                       seasonal_rate_per_day=10, reference_drift_tolerance=2,
                       reference_ratio_floor=1, reference_drift_relative_tolerance=0.15)
    elif metric == "shuntVoltage":
        options.update(residual_scale_floor=0.001, step_min_effect=0.005,
                       freeze_tolerance=0.000001, seasonal_rate_per_day=0.001,
                       reference_drift_tolerance=0.002)
    elif metric != "temperature":
        options.update(freeze_at_startup=False)
    if deployment == "laboratory":
        options.update(stationary_residuals=True, seasonal_rate_per_day=0)
    return MetricProfile(**options)


class ProfileRegistry:
    """Most-specific matching override wins; all overrides contain full profiles.

    A selector may include sensor, model, metric, deployment, site, and cadence.
    Equal-specificity ties use the last entry, enabling explicit local overrides.
    """

    def __init__(self, deployment="outdoor", expected_interval_seconds=300, overrides=()):
        metric_profile("temperature", deployment, expected_interval_seconds)
        self.deployment = deployment
        self.expected_interval_seconds = expected_interval_seconds
        self.overrides = []
        for item in overrides:
            if set(item) != {"match", "profile"}:
                raise ValueError("profile override requires match and profile")
            if not set(item["match"]) <= {"sensor", "model", "metric", "deployment", "site", "cadence"}:
                raise ValueError("unknown profile selector")
            self.overrides.append({"match": dict(item["match"]),
                                   "profile": MetricProfile(**item["profile"]).to_dict()})

    def resolve(self, sensor, metric, model=None, site=None):
        context = dict(sensor=sensor, metric=metric, model=model, site=site,
                       deployment=self.deployment, cadence=self.expected_interval_seconds)
        matches = [(len(item["match"]), i, item) for i, item in enumerate(self.overrides)
                   if all(context[k] == v for k, v in item["match"].items())]
        if matches:
            return MetricProfile(**max(matches, key=lambda x: (x[0], x[1]))[2]["profile"])
        return metric_profile(metric, self.deployment, self.expected_interval_seconds)

    def to_dict(self):
        return dict(deployment=self.deployment, expected_interval_seconds=self.expected_interval_seconds,
                    overrides=self.overrides)


@dataclass(frozen=True)
class SensorRules:
    """Opt-in model relationships. Particle counts are differential, not cumulative."""

    ordered_metrics: tuple[str, ...] = ()
    ordering_tolerance: float = 0.1
    status_field: str | None = None
    healthy_status_values: tuple[float, ...] = (0,)
    dewpoint_field: str | None = None
    dewpoint_tolerance: float = 0.5
    simultaneous_jump_metrics: int = 3

    def __post_init__(self):
        object.__setattr__(self, "ordered_metrics", tuple(self.ordered_metrics))
        object.__setattr__(self, "healthy_status_values", tuple(self.healthy_status_values))
        if len(set(self.ordered_metrics)) != len(self.ordered_metrics):
            raise ValueError("ordered_metrics must be unique")
        for name in ("ordering_tolerance", "dewpoint_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if (isinstance(self.simultaneous_jump_metrics, bool)
                or not isinstance(self.simultaneous_jump_metrics, int)
                or self.simultaneous_jump_metrics < 2):
            raise ValueError("simultaneous_jump_metrics must be an integer >= 2")
        if not self.healthy_status_values or not all(math.isfinite(x) for x in self.healthy_status_values):
            raise ValueError("healthy_status_values must be finite and nonempty")
