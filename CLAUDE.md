# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

AGENTS.md (imported above) is the canonical agent guide: setup, commands, key
files, data/output rules, verification checklist, and git workflow. Keep shared
rules there; this file only adds what AGENTS.md does not cover.

## Additional commands

```bash
python -m pytest tests/test_stats.py                       # one test file
python -m pytest tests/test_health.py -k freeze            # tests matching a name
python scripts/verify_offline_health.py --output <new-dir> # validate HEAD in an isolated copy (dir must not exist)
```

CI (`.github/workflows/`) runs on Python 3.10 and 3.12: `pytest tests/` then
`scripts/evaluate_health.py --require-targets`, uploading `evaluation_output/health/`.
A change that passes tests but misses a health holdout target still fails CI.

## Engine structure

`SensorHealth` is the only streaming engine; the SAFE 2 `SensorDrift` engine and
its alert-level evaluation (`safe/engine.py`, `safe/evaluation.py`) were removed
in SAFE 3. Responsibilities are split across modules: detectors, warmup, and
state in `health.py`; seasonal expectation in `baseline.py`; configuration in
`profiles.py`; and incident lifecycle (open/update/escalate/recover/close) plus
notification policy in `incidents.py`. Detectors emit findings; `incidents.py`
decides what becomes an incident or notification.

## Rules to follow

1. NEVER use Haiku 4.5 for anything in this codebase. Utilize Sonnet 5 Medium effort instead. 
