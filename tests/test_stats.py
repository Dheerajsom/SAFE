import numpy as np
import pytest
from scipy import stats as sstats

from safe.stats import (
    effective_sample_size,
    lag1_autocorrelation,
    sample_comparison,
)

RNG = np.random.default_rng(42)


def ar1(n, rho, sigma=1.0, mean=0.0, rng=RNG):
    """Generate an AR(1) series with the given lag-1 autocorrelation."""
    x = np.empty(n)
    x[0] = rng.normal()
    innovation = sigma * np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(scale=innovation)
    return mean + x


class TestAutocorrelation:
    def test_white_noise_rho_near_zero(self):
        rho = lag1_autocorrelation(RNG.normal(size=5000))
        assert rho < 0.05

    def test_ar1_rho_recovered(self):
        rho = lag1_autocorrelation(ar1(20000, 0.8))
        assert 0.75 < rho < 0.85

    def test_negative_rho_clipped_to_zero(self):
        # Alternating series is anti-correlated; we never inflate n_eff
        x = np.tile([1.0, -1.0], 500) + RNG.normal(scale=0.1, size=1000)
        assert lag1_autocorrelation(x) == 0.0

    def test_short_or_flat_series(self):
        assert lag1_autocorrelation(np.arange(5)) == 0.0
        assert lag1_autocorrelation(np.ones(100)) == 0.0

    def test_n_eff_less_than_n_for_correlated(self):
        x = ar1(2000, 0.9)
        n_eff, rho = effective_sample_size(x)
        assert n_eff < 400  # 2000 * (1-0.9)/(1+0.9) ~ 105
        assert rho > 0.8


class TestSampleComparison:
    def test_matches_scipy_without_correction(self):
        old = RNG.normal(10, 2, 300)
        new = RNG.normal(10.5, 2, 300)
        res = sample_comparison(old, new, autocorr_correction=False)
        _, p_ref = sstats.ttest_ind(old, new, equal_var=False)
        assert res["p_welch"] == pytest.approx(p_ref, rel=1e-6)

    def test_detects_clear_mean_shift(self):
        old = RNG.normal(10, 1, 500)
        new = RNG.normal(12, 1, 500)  # 2-sigma shift
        res = sample_comparison(old, new)
        assert res["mean_shift"] is True
        assert res["cohens_d"] > 1.0

    def test_effect_size_gate_blocks_tiny_shift_on_large_n(self):
        # Statistically significant but practically meaningless: d ~ 0.05
        old = RNG.normal(10, 1, 100_000)
        new = RNG.normal(10.05, 1, 100_000)
        res = sample_comparison(old, new)
        assert abs(res["cohens_d"]) < 0.2
        assert res["mean_shift"] is False

    def test_autocorrelation_correction_reduces_significance(self):
        # Same tiny level offset; highly autocorrelated data should NOT be
        # called significant once n_eff is honest.
        old = ar1(3000, 0.95, mean=10.0)
        new = ar1(3000, 0.95, mean=10.1)
        res_corr = sample_comparison(old, new, autocorr_correction=True)
        res_raw = sample_comparison(old, new, autocorr_correction=False)
        assert res_corr["p_welch"] >= res_raw["p_welch"]
        assert res_corr["old_n_eff"] < res_corr["old_n"] / 5

    def test_detects_variance_shift(self):
        old = RNG.normal(10, 1, 1000)
        new = RNG.normal(10, 3, 1000)
        res = sample_comparison(old, new)
        assert res["variance_shift"] is True
        assert res["std_ratio"] > 1.5

    def test_variance_ratio_gate(self):
        # 20% spread change is below the 50% gate, huge n makes it significant
        old = RNG.normal(10, 1.0, 50_000)
        new = RNG.normal(10, 1.2, 50_000)
        res = sample_comparison(old, new)
        assert res["variance_shift"] is False

    def test_both_flat_no_move(self):
        res = sample_comparison(np.full(100, 5.0), np.full(100, 5.0), metric="pm1_0")
        assert res["both_flat"] is True
        assert res["mean_shift"] is False
        assert res["p_welch"] == 1.0

    def test_both_flat_step_change(self):
        res = sample_comparison(np.full(100, 5.0), np.full(100, 7.0), metric="pm1_0")
        assert res["both_flat"] is True
        assert res["mean_shift"] is True

    def test_flat_threshold_is_per_metric(self):
        # 0.05 move: above the pm1_0 noise floor? No (0.1). Above shuntVoltage's? Yes (0.001).
        old, new = np.full(50, 1.0), np.full(50, 1.05)
        assert sample_comparison(old, new, metric="pm1_0")["mean_shift"] is False
        assert sample_comparison(old, new, metric="shuntVoltage")["mean_shift"] is True

    @pytest.mark.parametrize("pc_bin", ["pc0_1", "pc0_3", "pc0_5", "pc1_0", "pc2_5", "pc5_0", "pc10_0"])
    def test_particle_count_bins_have_flat_threshold(self, pc_bin):
        # Counts are particles/L; the generic 0.01 default would flag any move
        old, new = np.full(50, 500.0), np.full(50, 505.0)
        assert sample_comparison(old, new, metric=pc_bin)["mean_shift"] is False
        assert sample_comparison(old, np.full(50, 600.0), metric=pc_bin)["mean_shift"] is True

    def test_half_flat_reported(self):
        res = sample_comparison(np.full(100, 5.0), RNG.normal(5, 1, 100))
        assert res["half_flat"] is True
        assert res["std_ratio"] == float("inf")

    def test_identical_samples_not_flagged(self):
        x = RNG.normal(10, 1, 400)
        res = sample_comparison(x, x)
        assert res["mean_shift"] is False
        assert res["variance_shift"] is False
