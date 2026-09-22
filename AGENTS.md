# AGENTS.md

Guidance for coding agents working in this repository.

## Project

SAFE, Sensor Analysis and Failure Evaluation, is a Python analysis project for
MINTS low-cost air-quality sensor data. It detects outliers, drift,
distribution changes, and period-over-period changes.

The core lives in the `safe` package at the repo root; `mintsXU4/` holds thin
compatibility shims plus older live-sensor utilities. Bundled source data
lives at `mintsXU4/data/valo_node_01_full_year.csv`. Generated outputs live
under `mintsXU4/output/`.

## Setup

```bash
pip install -e ".[dev]"
```

Core dependencies: `numpy`, `scipy`, `pandas>=2.0`, `matplotlib`. The `video`
extra installs `imageio-ffmpeg`; `dev` includes it because video helper tests
import the rendering modules. The `sensor`
extra adds the legacy live-node packages (`pyserial`, `paho-mqtt`, `pyyaml`, ...).

## Common Commands

```bash
python -m pytest tests/                 # test suite — run this first
safe stream mintsXU4/data/valo_node_01_full_year.csv
safe periods mintsXU4/data/valo_node_01_full_year.csv -o mintsXU4/output
python mintsXU4/mintsDriftAnalysis.py   # legacy entry points still work
python mintsXU4/mintsPeriodAnalysis.py
python dataVisualizer.py
```

## Key Files

- `safe/stats.py`: Source of truth for `sample_comparison()` — Welch/Levene
  drift tests with effect-size gates and AR(1) effective-sample-size (n_eff)
  correction.
- `safe/engine.py`: `SensorDrift` streaming engine — hard bounds, frozen-value
  detection, robust (median/MAD+IQR, per-metric scale floor) modified z-score,
  step-change detection, optional
  Page-Hinkley layer, windowed drift evaluation. `PageHinkley` is off by
  default because ambient diurnal cycles trigger it daily.
- `safe/config.py`: `HARD_BOUNDS`, effect-size gates, flat-step thresholds,
  robust-scale floors, freeze thresholds.
- `safe/loader.py`: InfluxDB-export CSV loading (`load_pivoted_dataframe`) and
  streaming replay (`replay_csv`).
- `safe/periods.py` + `safe/plotting.py`: period-over-period comparisons and
  their plots.
- `mintsXU4/mintsDriftAnalysis.py`, `mintsXU4/mintsPeriodAnalysis.py`,
  `mintsXU4/mintsPeriodPlotter.py`: compatibility shims re-exporting from
  `safe`; keep them in sync when renaming public symbols.
- `mintsXU4/mints1sLoader.py`, `mintsXU4/mintsPmRegen.py`: 1-second PM data
  pipeline (data is git-ignored, ~5 GB local).
- `mintsXU4/mintsDefinitions.py`, `mintsXU4/mintsLatest.py`,
  `mintsXU4/mintsSensorReader.py`: older/live sensor utilities. Avoid changing
  these unless the task is about live sensor behavior.

## Data Notes

The bundled CSV is an InfluxDB-style export. The loader expects `_time`,
`_value`, `_field`, `_measurement`, `device_id`. `load_pivoted_dataframe()`
handles numeric coercion, duplicate removal, pivoting fields into metric
columns, and timestamp normalization.

Normalize ISO timestamps to UTC before deduplication; naive timestamps mean
UTC. Discard nonfinite values and invalid timestamps/identifiers with a warning.
Keep device IDs as strings (leading zeros matter). The historical sensor label
is reserved for its configured device; other devices get an ID suffix.
Replay files in chronological, non-overlapping order. A failed replay may leave
partial state in a supplied engine and returns `None`; do not retry it blindly.
`sample_comparison` expects finite 1-D samples with at least two readings each.
The AR(1) correction is an approximation, not a guarantee of calibrated false
positive rates for irregular sampling, seasonal cycles, or arbitrary processes.

## Output Notes

Treat files under `mintsXU4/output/` as generated. Prefer regenerating outputs
from source scripts instead of manually editing generated CSV/PNG/Markdown
files. Important areas:

- Period CSVs/plots: `mintsXU4/output/period_*.csv`, `mintsXU4/output/plots/`
- Distribution visualizer outputs: `mintsXU4/output/<field>/`

## Coding Guidelines

- Put shared drift math in `safe/` — never duplicate it in scripts.
- Use pandas/numpy vectorized operations for CSV processing.
- Keep plotting headless with the existing matplotlib `Agg` pattern.
- Do not add live MQTT, serial, or credential side effects to offline analysis
  paths.
- Missing local credential YAML files should not crash imports.
- Do not commit `__pycache__`, `.DS_Store`, credentials, or local device data.

## Verification

Before handing off substantial changes:

1. Run `python -m pytest tests/`.
2. Run the relevant analysis or plotting script on the bundled CSV.
3. Confirm outputs land in the expected `mintsXU4/output/` location.
4. Check `git status --short` and call out generated files that changed.

Run the full suite before changes as well as after them. For substantial
analysis changes, run all five offline workflows listed under Common Commands.
Check CSV schemas, finite probabilities/counts, PNG readability, and nonzero
failure exit codes. For animation changes, run a short headless preview; full
year video generation requires optional local archives and substantial memory.
Never execute live sensor or downloader entry points as offline smoke tests.

If tracked source data or generated outputs have pre-existing user changes,
validate in an isolated temporary copy using the committed CSV. Keep the same
`mintsXU4/output/` layout there, and record this deviation and the output path.
Do not restore deleted user files or stage regenerated artifacts by default.
Use a local `.venv`; if `python` is absent from PATH, use its explicit executable.

## Git workflow

- Inspect branch, status, and staged changes before editing; preserve unrelated
  work. Stage explicit in-scope paths rather than `git add .` or `git add -A`.
- Use the exact user-requested branch name; otherwise follow the configured
  `codex/` branch prefix. Do not commit directly to `main`.
- Commit/push when requested, configure upstream tracking, and do not force-push.
- Before deleting a branch, fetch its remote state, verify it is not checked out
  in any worktree, and prove both local and remote tips are ancestors of the
  preserved branch. Inspect unique commits; never force-delete unmerged work.
- Finish with `git status --short`, report the commit and push/deletion outcomes,
  and identify pre-existing changes or remaining generated artifacts.
