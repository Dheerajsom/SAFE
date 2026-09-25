# ***************************************************************************
#  SAFE — drift-test statistics
#  --------------------------------------------------------------------------
#  sample_comparison() is the single source of truth for "did this metric
#  drift between two samples". It layers three defenses against the classic
#  failure modes of naive two-sample testing on sensor data:
#
#    1. Effect-size gates (Cohen's d, std ratio): p-values saturate toward 0
#       on large samples; a shift must also be practically meaningful.
#    2. Autocorrelation-corrected sample sizes (n_eff): sensor readings are
#       serially correlated, so the nominal n wildly overstates the amount of
#       independent evidence. Welch's test is computed with the AR(1)
#       effective sample size n_eff = n * (1 - rho) / (1 + rho); Levene's test
#       runs on a thinned (approximately independent) subsample.
#    3. Flat-signal handling: two constant arrays make the tests meaningless,
#       so a flat-vs-flat comparison reduces to "did the level move more than
#       a per-metric threshold".
# ***************************************************************************

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import stats

from safe.config import (
    DEFAULT_FLAT_MEAN_SHIFT,
    FLAT_MEAN_SHIFT_THRESHOLDS,
    FLAT_VAR_THRESHOLD,
    MIN_COHENS_D,
    MIN_STD_RATIO,
)

# p-values are floored here so -log10(p) plots stay bounded
P_FLOOR = 1e-15

# Below this many effective observations per side the tests carry essentially
# no evidence; we report p = 1.0 rather than pretend otherwise.
MIN_EFFECTIVE_N = 3.0

# Cap on the AR(1) coefficient so a pathological rho ≈ 1 estimate cannot
# drive n_eff to zero.
MAX_RHO = 0.98


def median(values: Iterable[float]) -> float:
    """Median of finite values, bit-identical to ``float(np.median(values))``.

    Streaming detectors take medians of a few dozen values per reading, where
    numpy's per-call overhead dominates. Like numpy, the middle value(s) are
    summed from +0.0, so a median of negative zeros is +0.0.
    """
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return 0.0 + ordered[middle]
    return (0.0 + ordered[middle - 1] + ordered[middle]) / 2


def validate_alpha(p_alpha: float) -> None:
    """Reject invalid significance levels before any data or output work."""
    if not np.isfinite(p_alpha) or not 0 < p_alpha < 1:
        raise ValueError("p_alpha must be finite and strictly between 0 and 1")


def _sample(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("samples must be one-dimensional and contain only finite values")
    return values


def lag1_autocorrelation(x: Any) -> float:
    """Lag-1 autocorrelation of a 1-D array, clipped to [0, MAX_RHO].

    Negative estimates are clipped to 0: anti-correlated noise would *inflate*
    n_eff above n, and we only ever want the conservative correction.
    Returns 0.0 when the series is too short or flat for a stable estimate.
    """
    return _lag1(_sample(x))


def _lag1(x: np.ndarray) -> float:
    """lag1_autocorrelation for an already validated sample."""
    if x.size < 10:
        return 0.0
    a, b = x[:-1], x[1:]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    rho = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    return float(np.clip(rho, 0.0, MAX_RHO))


def effective_sample_size(x: Any) -> tuple[float, float]:
    """AR(1) effective sample size: n_eff = n * (1 - rho) / (1 + rho). Returns (n_eff, rho)."""
    return _effective_sample_size(_sample(x))


def _effective_sample_size(x: np.ndarray) -> tuple[float, float]:
    rho = _lag1(x)
    return x.size * (1.0 - rho) / (1.0 + rho), rho


def _welch_test(old: np.ndarray, new: np.ndarray, mean_delta: float,
                n_eff_old: float, n_eff_new: float) -> float:
    """Welch's t-test using effective sample sizes.

    With n_eff = n this reproduces scipy.stats.ttest_ind(equal_var=False).
    Returns p = 1.0 when either side has too little effective evidence.
    """
    if n_eff_old < MIN_EFFECTIVE_N or n_eff_new < MIN_EFFECTIVE_N:
        return 1.0

    s1 = float(np.var(old, ddof=1))
    s2 = float(np.var(new, ddof=1))
    se_sq = s1 / n_eff_old + s2 / n_eff_new
    if se_sq <= 0.0:
        return 1.0

    t = mean_delta / se_sq ** 0.5
    df = se_sq ** 2 / (
        (s1 / n_eff_old) ** 2 / (n_eff_old - 1.0)
        + (s2 / n_eff_new) ** 2 / (n_eff_new - 1.0)
    )
    return 2.0 * float(stats.t.sf(abs(t), df))


def _thin(x: np.ndarray, n_eff: float) -> np.ndarray:
    """Evenly-spaced subsample of ~n_eff approximately independent readings."""
    x = np.asarray(x, dtype=float)
    stride = max(int(np.ceil(x.size / max(n_eff, 1.0))), 1)
    return x[::stride]


def _levene_test(old: np.ndarray, new: np.ndarray, n_eff_old: float, n_eff_new: float) -> float:
    """Brown-Forsythe (median-centered) Levene test on thinned samples.

    Thinning to ~n_eff readings per side reduces serial correlation under an
    AR(1) approximation. It does not guarantee independent observations for
    seasonal, irregularly sampled, or higher-order processes.
    """
    if n_eff_old < MIN_EFFECTIVE_N or n_eff_new < MIN_EFFECTIVE_N:
        return 1.0
    a = _thin(old, n_eff_old)
    b = _thin(new, n_eff_new)
    if a.size < 3 or b.size < 3:
        return 1.0
    _, p = stats.levene(a, b, center="median")
    return float(p)


def sample_comparison(old: Any, new: Any, p_alpha: float = 0.01, metric: str | None = None,
                      autocorr_correction: bool = True) -> dict[str, Any]:
    """Compare two samples; return descriptive stats + drift-test results.

    Parameters
    ----------
    old, new : array-like
        The two samples (older period first).
    p_alpha : float
        Significance level for the Welch / Levene tests.
    metric : str, optional
        Metric name; selects the per-metric flat-step threshold.
    autocorr_correction : bool
        When True (default) the tests use AR(1) effective sample sizes so
        serially-correlated data does not produce fake certainty. Set False
        to reproduce the classic (nominal-n) tests.

    A shift is only flagged when it is BOTH statistically significant at the
    (corrected) p_alpha AND large enough to matter (effect-size gates).
    """
    validate_alpha(p_alpha)
    old = _sample(old)
    new = _sample(new)
    if old.size < 2 or new.size < 2:
        raise ValueError("each sample must contain at least two readings")

    n_old = int(old.size)
    n_new = int(new.size)

    old_var = float(np.var(old))
    new_var = float(np.var(new))
    old_std = old_var ** 0.5
    new_std = new_var ** 0.5

    old_mean = float(np.mean(old))
    new_mean = float(np.mean(new))
    mean_delta = new_mean - old_mean

    old_flat = old_var < FLAT_VAR_THRESHOLD
    new_flat = new_var < FLAT_VAR_THRESHOLD
    both_flat = old_flat and new_flat
    half_flat = old_flat != new_flat

    # Per-metric "did the constant level move" threshold (flat-vs-flat step change)
    flat_threshold = FLAT_MEAN_SHIFT_THRESHOLDS.get(metric, DEFAULT_FLAT_MEAN_SHIFT)
    mean_val_changed = abs(mean_delta) > flat_threshold

    # Cohen's d (pooled SD): scale-free practical magnitude of the mean shift.
    denom = n_old + n_new - 2
    pooled_std = ((n_old * old_var + n_new * new_var) / denom) ** 0.5 if denom > 0 else 0.0
    cohens_d = mean_delta / pooled_std if pooled_std > 1e-12 else 0.0

    # Directional fold-change in spread; inf when only one side is flat.
    if old_std > 1e-12:
        std_ratio = new_std / old_std
    elif new_std > 1e-12:
        std_ratio = float('inf')
    else:
        std_ratio = 1.0

    if autocorr_correction:
        n_eff_old, rho_old = _effective_sample_size(old)
        n_eff_new, rho_new = _effective_sample_size(new)
    else:
        n_eff_old, rho_old = float(n_old), 0.0
        n_eff_new, rho_new = float(n_new), 0.0

    if both_flat:
        # Two constant arrays --> the tests are meaningless; drift = did the level move
        p_welch = 1.0
        p_levene = 1.0
        mean_shift = mean_val_changed
        variance_shift = False
    else:
        p_welch = _welch_test(old, new, mean_delta, n_eff_old, n_eff_new)
        p_levene = _levene_test(old, new, n_eff_old, n_eff_new)

        p_welch = 1.0 if np.isnan(p_welch) else max(p_welch, P_FLOOR)
        p_levene = 1.0 if np.isnan(p_levene) else max(p_levene, P_FLOOR)

        # Two-gate rule: statistically significant AND practically meaningful.
        mean_shift = (p_welch < p_alpha) and (abs(cohens_d) >= MIN_COHENS_D)
        variance_shift = (p_levene < p_alpha) and (
            std_ratio >= MIN_STD_RATIO or std_ratio <= 1.0 / MIN_STD_RATIO
        )

    return {
        'old_n': n_old,
        'new_n': n_new,
        'old_n_eff': round(float(n_eff_old), 1),
        'new_n_eff': round(float(n_eff_new), 1),
        'rho_old': round(float(rho_old), 4),
        'rho_new': round(float(rho_new), 4),
        'old_mean': old_mean,
        'new_mean': new_mean,
        'mean_delta': mean_delta,
        'old_std': old_std,
        'new_std': new_std,
        'old_var': old_var,
        'new_var': new_var,
        'both_flat': both_flat,
        'half_flat': half_flat,
        'mean_val_changed': mean_val_changed,
        'cohens_d': float(cohens_d),
        'std_ratio': float(std_ratio),
        'p_welch': float(p_welch),
        'p_levene': float(p_levene),
        'mean_shift': bool(mean_shift),
        'variance_shift': bool(variance_shift),
    }
