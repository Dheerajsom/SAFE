# SAFE — terminal cheat sheet

Reference for running this repo's **1-second PM/PC data** workflows on a fresh
machine. All commands assume you're in the repo root unless noted.
PowerShell is called out separately wherever its syntax differs from bash.

## 0. Setup

```bash
git clone <repo-url> SAFE
cd SAFE

python -m venv .venv
source .venv/bin/activate          # PowerShell: .venv\Scripts\Activate.ps1

pip install -e ".[dev]"            # safe package + CLI + pytest
pip install -e ".[sensor]"         # only if you need the live-node serial/MQTT scripts

python -m pytest tests/            # full regression suite
```

## 1. The data

`mintsXU4/data/valo_node_01_1s/` (git-ignored) can hold gzipped daily files
for the seven IPS7100 PM bins and seven corresponding PC bins (`pc0_1` through
`pc10_0`). Archives downloaded with the downloader's default fields remain
PM-only; include PC fields with `--fields` when downloading them. PC values
use the IPS protocol's default `particles/L` unit. No
temperature/pressure/humidity is included at this resolution.

- Filename pattern: `valo_node_01_YYYYMMDD_YYYYMMDD.csv.gz` (one calendar day each)
- Size: ~2.9 GB gzipped per year
- Range as of 2026-07: `2025-06-14` → `2026-06-14`

Neither this directory nor its files are tracked in git — download them
(section 2) or copy them from another machine.

## 2. Downloading data from InfluxDB

`mintsInfluxDownloader.py` needs credentials as environment variables —
never pass the token on the command line:

```bash
export INFLUX_HOST="http://mdash.circ.utdallas.edu:8086"
export INFLUX_TOKEN="<api token>"
# INFLUX_ORG defaults to "MINTS"
```

Pull the full 1-second PM history. It's resumable — safe to re-run, it only
fills in missing day-files. `--start auto` probes the first available point;
`--stop` defaults to now if omitted:

```bash
python mintsInfluxDownloader.py --window 1s --gzip --start auto
```

| Flag | Purpose |
|---|---|
| `--start YYYY-MM-DD` | explicit start date instead of `auto` |
| `--stop YYYY-MM-DD` | explicit end date instead of "now" |
| `--chunk-days N` | days per request (default 1 — keep small at 1s resolution) |
| `--fields ...` | override the 7 default PM fields |
| `--out-dir DIR` | default: `mintsXU4/data/valo_node_01_1s` |
| `--merge` | also stitch everything into one merged CSV |
| `--backend cli` | shell out to the `influx` CLI instead of raw HTTP |

## 3. Sensor-health incidents — `safe health`

Replay 1-second PM data through the health engine, one metric at a time,
across a directory of day-files, as a single continuous stream (state carries
across day boundaries — baselines and open incidents aren't reset at each
file). `safe stream` is an alias of `safe health`.

```bash
python -m safe.cli health mintsXU4/data/valo_node_01_1s --metric pm1_0 \
    --config docs/health-config-1s.json -o mintsXU4/output/health_1s
```

> `--config docs/health-config-1s.json` matters, not just a knob: it sets the
> profile cadence to 1 second. With the default 300-second profiles, gaps,
> completeness, and window sizes are interpreted at the wrong rate.

If `safe` isn't on `PATH` (e.g. Git Bash on some setups), use
`python -m safe.cli` as above instead of the bare `safe` command.

**One week only** — first 7 day-files, chronological by filename:

```powershell
# PowerShell
$files = (Get-ChildItem mintsXU4\data\valo_node_01_1s\*.csv.gz | Sort-Object Name | Select-Object -First 7).FullName
python -m safe.cli health $files --metric pm1_0 --config docs/health-config-1s.json
```

```bash
# bash
python -m safe.cli health $(ls mintsXU4/data/valo_node_01_1s/*.csv.gz | sort | head -7) --metric pm1_0 --config docs/health-config-1s.json
```

Swap `-First 7` / `head -7` for a later slice to shift the window, e.g.
`-Skip 30 -First 7` for the 5th week of data. Use a fresh `-o` directory per
independent replay: the JSONL files append.

| Flag | Purpose |
|---|---|
| `--metric NAME` | repeatable; restrict to one or more metrics |
| `--config FILE` | JSON profiles, model rules, and sensor metadata |
| `--state-in` / `--state-out` | resume from / save a restart-safe checkpoint |
| `--tick-until ISO` | explicit observation end for silence detection |
| `-o DIR` | output directory (default `mintsXU4/output/health`) |

## 4. Daily incident-count summary over the whole 1s dataset

Replays every day-file through **one continuous** engine (1-second cadence
profiles by default) and tabulates the incidents opened per day to a CSV —
good for eyeballing trends across many days at once:

```bash
python scripts/summarize_daily_drift.py
python scripts/summarize_daily_drift.py --limit 10 -o /tmp/drift.csv
python scripts/summarize_daily_drift.py --config docs/health-config-1s.json
```

Default output: `mintsXU4/output/drift_day_summary.csv`

## 5. Regenerating PM plots/histograms from the 1s data

```bash
cd mintsXU4
python mintsPmRegen.py
```

Regenerates **all 7 PM bins'** plots/histograms into `output/<field>/plots/`
and `output/<field>/histograms/` (168 PNGs total; no CSVs written). Reads
and caches a wide float32 frame at
`data/valo_node_01_1s/_wide_pm_cache.pkl` (git-ignored) — pass
`rebuild=True` in `mints1sLoader.load_wide()` if the raw data changed.

## 6. Tests

```bash
python -m pytest tests/                      # full suite
python -m pytest tests/test_engine.py -v     # one module, verbose
```
