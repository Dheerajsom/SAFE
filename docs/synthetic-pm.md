# Synthetic PM test dataset

`mintsXU4/data/synthetic_pm/` holds a seeded, labeled dataset for measuring how
accurately `SensorHealth` finds PM sensor faults. It replaces the full-year field
export for engine testing; it is not field evidence.

| File | Contents |
|---|---|
| `pm0_1.csv.gz` … `pm10_0.csv.gz` | One InfluxDB-style long export per PM bin, all six nodes |
| `labels.csv` | Every injected fault (annotation format, see `safe.annotations`) |
| `context.csv` | Healthy look-alike windows (pollution episodes, clean air, haze trend) |
| `health-config.json` | Engine configuration: IPS7100 model rules for each node |

Six co-located nodes (`IPS7100SYN_node01` … `node06`) report seven cumulative
PM bins every 5 minutes for 90 days from 2025-03-01 UTC. All nodes share one ambient
signal with diurnal and weekly cycles; each adds its own gain, micro-environment,
and measurement noise. Healthy bins are always size-ordered (pm0_1 ≤ … ≤ pm10_0).
`node01` and `node06` are fault-free controls.

## Regenerate, verify, evaluate

```bash
python scripts/generate_synthetic_pm.py            # rewrite the dataset (seed 20250301)
python scripts/generate_synthetic_pm.py --check    # exit 1 if files differ from the generator
python scripts/evaluate_synthetic_pm.py            # accuracy report, both modes (~5 min)
safe health mintsXU4/data/synthetic_pm/*.csv.gz --merge \
    --config mintsXU4/data/synthetic_pm/health-config.json -o mintsXU4/output/health_synthetic
```

`--merge` is required when replaying the per-bin files together: they cover the
same period, so without it the second file is rejected as overlapping. Merging
also lets the IPS7100 bin-ordering rule see every bin at each timestamp.

The report (`evaluation_output/synthetic_pm/accuracy.md` and `.json`) scores two
modes. **single-sensor** matches `safe health`. **with-references** also passes the
same bin from the other five nodes as co-located peers, which lets the engine
separate a sensor fault from a shared ambient change. Scoring reuses
`safe.health_evaluation` (see [evaluation.md](evaluation.md)): diagnostic and
actionable recall per fault, delays, and false actionable incidents, which the
report also groups by the nearest healthy look-alike.

## Embedded faults

| ID | Node | Bins | Fault | Label |
|---|---|---|---|---|
| F1 | node02 | pm2_5 | One negative reading (−3) | invalid |
| F2 | node02 | pm10_0 | 25,000 µg/m³ for 3 readings | invalid |
| F3 | node02 | pm1_0 | Stuck value for 8 h | freeze |
| F4 | node02 | all | Offline for 10 h | offline |
| F5 | node02 | all | Whole sensor frozen for 12 h | freeze |
| F6 | node02 | pm5_0 | Dead channel reading 0 for 24 h | freeze |
| L1 | node02 | all | Conflicting duplicates and out-of-order rows | clock (loader) |
| S1 | node03 | pm2_5 | Six isolated +60 spikes | spike |
| S2 | node03 | pm1_0 | +12 offset for 48 h | offset |
| S3 | node03 | pm0_3 | Subtle +2.5 offset for 48 h | offset |
| S4 | node03 | pm10_0 | Noise sd 8 for 24 h | noise |
| S5 | node03 | pm5_0 | Gain 0.35 for 48 h | sensitivity |
| S6 | node03 | pm2_5 | +25 offset for 12 h | offset |
| D1 | node04 | pm2_5 | Drifts +10 over 10 days, held 5 days | drift |
| D2 | node04 | pm10_0 | Gain drifts to 1.7 over 15 days, held 5 days | drift |
| D3 | node04 | pm0_5 | Drifts +4 over 15 days, during the haze trend | drift |
| C1 | node05 | pm1_0, pm2_5 | Channels swapped for 24 h | ordering |
| C2 | node05 | pm0_1 | Channel outputs garbage for 12 h | noise |
| C3 | node05 | all | Restart: all bins read 0 for 3 readings | restart |
| C4 | node05 | all | All bins read 1.5× for 3 days | offset |
| C5 | node05 | pm0_5 | Reads 0.6 above pm1_0 for 12 h | offset |
| O | any | neighbors | Ordering violation a fault causes in an adjacent bin | ordering |

Healthy look-alikes shared by all nodes: four regional pollution episodes (3–6×,
10–36 h), three clean-air periods where small bins read exact zeros, and a
two-week haze trend (up to 1.8×) that overlaps drift D3.

The loader removes L1's duplicates and restores time order before replay, so L1
is checked by `loader_check`, not scored against the engine. Some faults are
below the engine's PM effect size (5 µg/m³), such as S3, D3, and the small bins
in C3 and C4; they are only caught when they break the bin ordering. Automatic
`O` labels are easy side effects of other faults, so judge recall on the primary
faults. The generator is the exact specification
(`safe/synthetic_pm.py`); bump `GENERATOR_REVISION` when changing it and
regenerate the files.
