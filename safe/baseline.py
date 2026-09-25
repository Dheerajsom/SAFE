"""Bounded robust UTC time-of-day profile with a rate-limited seasonal level.

No raw ambient Page-Hinkley inputs. Predictions must precede learning a reading.
"""

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from safe.stats import median

# Converts a MAD or an interquartile range to a normal standard deviation.
MAD_TO_SIGMA = 1.4826
IQR_TO_SIGMA = 1.349
# Seconds over which a level innovation is fully absorbed (before rate limiting).
LEVEL_RESPONSE_SECONDS = 21600
# Fraction of a within-cycle innovation that updates a phase's representative.
PHASE_LEARNING_RATE = 0.1


def robust_scale(values: Sequence[float], floor: float) -> tuple[float, float]:
    """Return (median, robust sigma), with sigma = max(MAD, IQR) scaled to normal, and >= floor."""
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center))) * MAD_TO_SIGMA
    q25, q75 = np.percentile(values, [25, 75])
    return center, max(mad, float(q75 - q25) / IQR_TO_SIGMA, floor)


class SeasonalBaseline:
    """Per-phase robust centers and spreads over bounded cycles, plus a slow level."""

    def __init__(self, profile: Any) -> None:
        self.profile = profile
        # Each phase bin holds [cycle, center, spread] entries, newest last.
        self.bins: list[list[list[float]]] = [[] for _ in range(profile.seasonal_bins)]
        self.level = 0.0
        self.last_learned: float | None = None
        self.ready = False
        self.revision = 0  # bumped by every change to bins, level, or readiness
        self._prediction: tuple[tuple[int, float], tuple[float, float]] | None = None

    def _position(self, timestamp: float) -> float:
        p = self.profile
        return (timestamp % p.seasonal_period_seconds) / p.seasonal_period_seconds * p.seasonal_bins

    def fit(self, samples: Sequence[tuple[float, float]]) -> bool:
        """Fit every phase from (timestamp, value) warmup samples; False if coverage is too low."""
        p = self.profile
        buckets: list[list[float]] = [[] for _ in self.bins]
        for timestamp, value in samples:
            buckets[int(self._position(timestamp))].append(value)
        coverage = sum(bool(b) for b in buckets) / len(buckets)
        if coverage < p.minimum_coverage:
            return False
        overall, scale = robust_scale([v for _, v in samples], p.residual_scale_floor)
        for index, bucket in enumerate(buckets):
            if bucket:
                center, spread = robust_scale(bucket, p.residual_scale_floor)
            else:
                center, spread = overall, scale
            self.bins[index] = [[int(samples[-1][0] // p.seasonal_period_seconds), center, spread]]
        self.ready = True
        self.last_learned = samples[-1][0]
        self.revision += 1
        return True

    def predict(self, timestamp: float) -> tuple[float, float] | None:
        """Expected value and residual scale at a timestamp, interpolated between phases.

        Detection predicts before learning the same reading, so the latest result is
        reused until the model changes (tracked by revision).
        """
        if not self.ready:
            return None
        memo = (self.revision, timestamp)
        if self._prediction is not None and self._prediction[0] == memo:
            return self._prediction[1]
        position = self._position(timestamp) - 0.5
        left = math.floor(position) % len(self.bins)
        right = (left + 1) % len(self.bins)
        fraction = position - math.floor(position)
        centers, scales = [], []
        for index in (left, right):
            centers.append(median(b[1] for b in self.bins[index]))
            scales.append(median(b[2] for b in self.bins[index]))
        result = (centers[0] * (1 - fraction) + centers[1] * fraction + self.level,
                  max(scales[0] * (1 - fraction) + scales[1] * fraction,
                      self.profile.residual_scale_floor))
        self._prediction = (memo, result)
        return result

    def learn(self, timestamp: float, value: float) -> None:
        """Absorb one reading: rate-limited level update, then one clipped phase update."""
        prediction = self.predict(timestamp)
        if prediction is None:
            return
        expected, scale = prediction
        p = self.profile
        elapsed = max(0, timestamp - self.last_learned)
        # Large gaps must not cause a one-reading jump in the expected level.
        elapsed = min(elapsed, 2 * p.expected_interval_seconds)
        rate_limit = p.seasonal_rate_per_day * elapsed / 86400
        innovation = value - expected
        # Scalar min(max()) matches np.clip exactly (including signed zeros) at a fraction of the cost.
        delta = float(min(max(innovation * min(1, elapsed / LEVEL_RESPONSE_SECONDS), -rate_limit), rate_limit))
        self.level += delta
        self.last_learned = timestamp
        # One robust, clipped representative per phase per cycle, bounded in cycles.
        bucket = self.bins[int(self._position(timestamp))]
        cycle = int(timestamp // p.seasonal_period_seconds)
        corrected = value - self.level
        if bucket[-1][0] == cycle:
            old = bucket[-1][1]
            bucket[-1][1] = old + PHASE_LEARNING_RATE * float(min(max(corrected - old, -scale), scale))
        else:
            bucket.append([cycle, corrected, scale])
            del bucket[:-p.seasonal_cycles]
        self.revision += 1

    def to_dict(self) -> dict[str, Any]:
        return dict(bins=self.bins, level=self.level, last_learned=self.last_learned,
                    ready=self.ready, revision=self.revision)

    @classmethod
    def from_dict(cls, profile: Any, data: dict[str, Any]) -> "SeasonalBaseline":
        obj = cls(profile)
        if len(data["bins"]) != profile.seasonal_bins:
            raise ValueError("state seasonal bin count differs from profile")
        for key in obj.to_dict():
            setattr(obj, key, data[key])
        return obj
