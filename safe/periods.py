# ***************************************************************************
#  SAFE — period-over-period drift analysis
#  --------------------------------------------------------------------------
#  Buckets each metric by calendar period (day / week / month / year), then
#  compares consecutive buckets — and the first vs last month — with the
#  shared drift math from safe.stats. Writes one CSV per granularity.
# ***************************************************************************

from collections.abc import Iterable
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from safe.config import HARD_BOUNDS
from safe.loader import load_pivoted_dataframe
from safe.stats import sample_comparison, validate_alpha

logger = logging.getLogger(__name__)

Bucket = tuple[pd.Timestamp, np.ndarray]

# Rolling consecutive-period comparisons: (output stem, pandas offset alias, label format)
#   D  = calendar day,  W = week (ending Sunday),  MS = month start,  YS = year start
GRANULARITIES = [
    ("day_to_day",     "D",  "%Y-%m-%d"),
    ("week_to_week",   "W",  "%Y-%m-%d"),
    ("month_to_month", "MS", "%Y-%m"),
    ("year_to_year",   "YS", "%Y"),
]
FIRST_VS_LAST = "first_vs_last_month"

CSV_COLUMNS = [
    "sensor", "metric", "granularity", "old_period", "new_period",
    "old_n", "new_n", "old_n_eff", "new_n_eff",
    "old_mean", "new_mean", "mean_delta", "old_std", "new_std",
    "old_min", "old_max", "new_min", "new_max",
    "cohens_d", "std_ratio", "p_welch", "p_levene", "mean_shift", "variance_shift",
]


def apply_hard_bounds(df: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    """NaN out nonfinite and physically impossible readings (in place) so they never enter the stats."""
    for metric in metric_cols:
        values = df[metric]
        invalid = ~np.isfinite(values)
        bounds = HARD_BOUNDS.get(metric)
        if bounds:
            lo, hi = bounds
            invalid |= (values < lo) | (values > hi)
        df.loc[invalid, metric] = np.nan
    return df


def comparison_row(sensor: str, metric: str, granularity: str, old_label: str, new_label: str,
                   old_vals: np.ndarray, new_vals: np.ndarray, result: dict[str, Any]) -> dict[str, Any]:
    """Assemble one output row from a sample_comparison() result plus min/max."""
    return {
        "sensor": sensor,
        "metric": metric,
        "granularity": granularity,
        "old_period": old_label,
        "new_period": new_label,
        "old_n": result["old_n"],
        "new_n": result["new_n"],
        "old_n_eff": result["old_n_eff"],
        "new_n_eff": result["new_n_eff"],
        "old_mean": round(result["old_mean"], 4),
        "new_mean": round(result["new_mean"], 4),
        "mean_delta": round(result["mean_delta"], 4),
        "old_std": round(result["old_std"], 4),
        "new_std": round(result["new_std"], 4),
        "old_min": round(float(old_vals.min()), 4),
        "old_max": round(float(old_vals.max()), 4),
        "new_min": round(float(new_vals.min()), 4),
        "new_max": round(float(new_vals.max()), 4),
        "cohens_d": round(result["cohens_d"], 4),
        "std_ratio": round(result["std_ratio"], 4),
        "p_welch": format(result["p_welch"], ".2e"),
        "p_levene": format(result["p_levene"], ".2e"),
        "mean_shift": result["mean_shift"],
        "variance_shift": result["variance_shift"],
    }


def _validate_min_samples(min_samples: Any) -> None:
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)) or min_samples < 2:
        raise ValueError("min_samples must be an integer >= 2")


def _finite_sorted(series: pd.Series) -> pd.Series:
    series = series[np.isfinite(series)]
    if not series.index.is_monotonic_increasing:
        series = series.sort_index(kind='stable')
    return series


def _resampled_buckets(series: pd.Series, freq: str, min_samples: int) -> list[Bucket]:
    return [(period, vals.to_numpy()) for period, vals in series.resample(freq) if len(vals) >= min_samples]


def bucketize(series: pd.Series, freq: str, min_samples: int) -> list[Bucket]:
    """Return [(period_timestamp, values_array), ...] for non-empty calendar
    buckets, in time order, keeping only buckets with >= min_samples readings."""
    _validate_min_samples(min_samples)
    return _resampled_buckets(_finite_sorted(series), freq, min_samples)


def _pair_row(sensor: str, metric: str, granularity: str, label_fmt: str, old: Bucket, new: Bucket,
              p_alpha: float) -> dict[str, Any]:
    (old_period, old_vals), (new_period, new_vals) = old, new
    result = sample_comparison(old_vals, new_vals, p_alpha, metric=metric)
    return comparison_row(sensor, metric, granularity, old_period.strftime(label_fmt),
                          new_period.strftime(label_fmt), old_vals, new_vals, result)


def _period_comparisons(df: pd.DataFrame, metric_cols: list[str], p_alpha: float,
                        min_samples: int) -> dict[str, list[dict[str, Any]]]:
    """Rows per output stem: every granularity in GRANULARITIES plus first_vs_last_month.

    Consecutive comparisons pair each non-empty period bucket with the previous
    one, per sensor and metric, so a pair may span a data gap (the old_period /
    new_period columns make gaps visible). first_vs_last_month compares the first
    calendar month of data with the last.
    """
    rows: dict[str, list[dict[str, Any]]] = {stem: [] for stem, _, _ in GRANULARITIES}
    rows[FIRST_VS_LAST] = []
    for sensor, sensor_df in df.groupby("_sensor_name"):
        for metric in metric_cols:
            series = _finite_sorted(sensor_df[metric])
            for stem, freq, label_fmt in GRANULARITIES:
                buckets = _resampled_buckets(series, freq, min_samples)
                rows[stem].extend(_pair_row(sensor, metric, stem, label_fmt, old, new, p_alpha)
                                  for old, new in zip(buckets, buckets[1:]))
                if freq == "MS" and len(buckets) >= 2:
                    rows[FIRST_VS_LAST].append(
                        _pair_row(sensor, metric, FIRST_VS_LAST, "%Y-%m", buckets[0], buckets[-1], p_alpha))
    return rows


def run_period_analysis(file_path: Any, output_dir: str | os.PathLike, p_alpha: float = 0.01,
                        min_samples: int = 2, make_plots: bool = True) -> bool:
    """Run all period comparisons on a CSV and write period_*.csv into output_dir.

    Returns True on success, False when the input CSV could not be loaded or plotting failed."""
    validate_alpha(p_alpha)
    _validate_min_samples(min_samples)
    df, metric_cols = load_pivoted_dataframe(file_path)
    if df is None:
        return False

    print(f"Metrics found : {metric_cols}")
    print(f"Date range    : {df.index.min()}  ->  {df.index.max()}")
    print("NOTE: first/last day, week, month and the end years are partial; the old_n/new_n "
          "columns show coverage. Year-to-Year therefore is not a like-for-like comparison.")

    df = apply_hard_bounds(df, metric_cols)
    os.makedirs(output_dir, exist_ok=True)

    # One CSV per granularity, then first month vs last month
    for stem, rows in _period_comparisons(df, metric_cols, p_alpha, min_samples).items():
        out_path = os.path.join(output_dir, f"period_{stem}.csv")
        pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(out_path, index=False)
        print(f"Wrote {os.path.basename(out_path)} ({len(rows)} rows)")

    print("Period analysis complete.")

    if make_plots:
        try:
            from safe.plotting import run_all_plotting
            plots_dir = os.path.join(output_dir, "plots")
            print("\nAuto-generating period plots...")
            run_all_plotting(output_dir, plots_dir, p_alpha=p_alpha)
        except Exception:
            logger.exception("Failed to auto-generate plots")
            return False

    return True
