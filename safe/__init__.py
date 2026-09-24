"""SAFE — Sensor Analysis and Failure Evaluation.

Drift and failure detection for MINTS low-cost air-quality sensor nodes.

Public API:
    SensorHealth         streaming incident-based sensor health engine
    sample_comparison    two-sample drift test (effect-size + n_eff gated)
    load_pivoted_dataframe / replay_csv   InfluxDB-export CSV helpers
    run_period_analysis  period-over-period comparisons + CSVs + plots
    HARD_BOUNDS          physical limits per metric
"""

from safe.config import HARD_BOUNDS
from safe.animation import FieldMetadata, field_metadata, pdf_axis_upper_limit
from safe.loader import load_pivoted_dataframe, replay_csv
from safe.periods import run_period_analysis
from safe.stats import effective_sample_size, sample_comparison
from safe.health import PageHinkley, SensorHealth
from safe.incidents import HealthEvent
from safe.profiles import MetricProfile, ProfileRegistry, SensorRules

__version__ = "3.0.0"

__all__ = [
    "HARD_BOUNDS",
    "FieldMetadata",
    "PageHinkley",
    "SensorHealth",
    "HealthEvent",
    "MetricProfile",
    "ProfileRegistry",
    "SensorRules",
    "effective_sample_size",
    "field_metadata",
    "load_pivoted_dataframe",
    "replay_csv",
    "run_period_analysis",
    "pdf_axis_upper_limit",
    "sample_comparison",
    "__version__",
]
