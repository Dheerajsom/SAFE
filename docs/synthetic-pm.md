# Synthetic PM and PC test datasets

`mintsXU4/data/synthetic_pm/` and `mintsXU4/data/synthetic_pc/` hold a seeded,
labeled dataset for measuring how accurately `SensorHealth` finds IPS7100 sensor
faults: the same six nodes and 90 days, reported as PM mass bins in one
directory and as particle-count bins in the other. It is not field evidence.

| File | Contents |
|---|---|
| `pm0_1.csv.gz` … `pm10_0.csv.gz` | One InfluxDB-style long export per PM bin (µg/m³), all six nodes |
| `pc0_1.csv.gz` … `pc10_0.csv.gz` | One export per particle-count bin (particles/L), all six nodes |
| `labels.csv` | Every injected fault on that family's bins (annotation format, see `safe.annotations`) |
| `context.csv` | Healthy look-alike windows (pollution episodes, clean-air dips, haze trend); same in both |
| `health-config.json` | Engine configuration: IPS7100 model rules for each node |

## How the data is built

Six co-located nodes (`IPS7100SYN_node01` … `node06`) report every 5 minutes for
90 days from 2025-03-01 UTC. The statistics are fitted to the valo_node_01 1 s
export (2024-08 … 2026-07) averaged to 5 minutes:

- **Counts are the primary signal.** A shared ambient level (UTC diurnal cycle
  peaking mid-morning, a small weekend lift, lognormal variability on 10-minute
  to multi-day scales, Poisson-timed regional pollution episodes and clean-air
  dips, one two-week haze trend) times a slowly varying size composition gives
  the seven differential count bins. Medians, spread, autocorrelation, and bin
  ratios match the field data; large bins are Poisson-sampled, so `pc10_0` is
  almost always 0 and `pc5_0` is often 0, as in the field. Coincidence loss keeps
  healthy `pc0_1` below about 1,000,000/L, like the field data.
- **PM is derived exactly as the IPS7100 derives it**: each PM bin is the running
  sum of count bins times a fixed mass per particle (`MASS_PER_COUNT`, recovered
  from the field data to within 0.5 % for the small bins). Healthy PM is
  therefore always size-ordered (pm0_1 ≤ … ≤ pm10_0), pm2_5 has a median near
  3 µg/m³, and pm5_0 ≈ pm10_0.
- Each node adds its own per-bin gain, micro-environment, and noise.
  `node01` and `node06` are fault-free controls.

## Three kinds of fault

- **Physical** faults change the particle counts (detector, laser, optics,
  airflow, power). They are labeled on the affected count bins and, through the
  PM derivation, on every PM bin they materially change (≥10 % for at least 10 %
  of the window, or any spike) — marked "(via counts)" in the PM labels.
- **PM-output** faults change only the reported PM values (firmware or
  transmission), so PM disagrees with the counts.
- **PC-output** faults change only the reported counts.

Ordering violations that faults cause in neighboring PM bins get automatic `O`
labels; PC bins are differential and have no ordering rule.

| ID | Node | Kind | Bins | Fault | Label |
|---|---|---|---|---|---|
| F1 | node02 | PM output | pm2_5 | One negative reading (−3) | invalid |
| F2 | node02 | PM output | pm10_0 | 25,000 µg/m³ for 3 readings | invalid |
| F3 | node02 | PM output | pm1_0 | Output stuck for 8 h | freeze |
| F6 | node02 | PM output | pm5_0 | Output reads 0 for 24 h | freeze |
| P1 | node02 | PC output | pc0_5 | One negative reading (−40) | invalid |
| P2 | node02 | PC output | pc1_0 | Reads 4,294,967,295 (uint32 overflow) for 3 readings | invalid |
| P3 | node02 | PC output | pc1_0 | Output stuck for 8 h | freeze |
| F4 | node02 | physical | all | Offline for 10 h | offline |
| F5 | node02 | physical | all | Whole sensor frozen for 12 h | freeze |
| P6 | node02 | physical | pc0_3 | Detector channel dead (reads 0) for 24 h; PM bins lose mass | freeze / sensitivity |
| L1 | node02 | loader | all | Conflicting duplicates and out-of-order rows | clock (loader) |
| S1 | node03 | PM output | pm2_5 | Six isolated +60 spikes | spike |
| S2 | node03 | PM output | pm1_0 | +12 offset for 48 h | offset |
| S3 | node03 | PM output | pm0_3 | Subtle +1.5 offset for 48 h | offset |
| S4 | node03 | PM output | pm10_0 | Noise sd 8 for 24 h | noise |
| S5 | node03 | PM output | pm5_0 | Gain 0.35 for 48 h | sensitivity |
| S6 | node03 | PM output | pm2_5 | +25 offset for 12 h | offset |
| P7 | node03 | physical | pc2_5 | Six isolated +3000/L bursts (insect or debris) | spike |
| P8 | node03 | PC output | pc0_5 | +20,000/L offset for 48 h | offset |
| P9 | node03 | physical | pc1_0 | Electrical noise (×lognormal sd 1) for 24 h | noise |
| P10 | node03 | physical | pc0_1 | Sensitivity 0.3 for 48 h (laser power drop) | sensitivity |
| D1 | node04 | PM output | pm2_5 | Drifts +6 over 10 days, held 5 days | drift |
| D2 | node04 | physical | pc2_5 | Gain drifts to 2.5 over 15 days, held 5 days (dirty optics) | drift |
| D3 | node04 | physical | pc0_3 | Gain drifts to 0.4 over 20 days, held 2 days, during the haze trend | drift |
| D4 | node04 | physical | all | Whole-sensor gain drifts to 1.25 over 14 days, held 2 days | drift |
| C1 | node05 | PM output | pm1_0, pm2_5 | Outputs swapped for 24 h | ordering |
| C2 | node05 | PM output | pm0_1 | Output is garbage for 12 h | noise |
| C5 | node05 | PM output | pm0_5 | Reads 0.6 above pm1_0 for 12 h | offset |
| P11 | node05 | PC output | pc0_3, pc0_5 | Outputs swapped for 24 h | offset |
| P12 | node05 | PC output | pc5_0 | Output is garbage for 12 h | noise |
| C3 | node05 | physical | all | Restart: all bins read 0 for 3 readings | restart |
| C6 | node05 | physical | all | Flow failure: all counts ×0.15 for 36 h | sensitivity |
| C4 | node05 | physical | all | All bins read 1.5× for 3 days (reseated inlet) | offset |
| C7 | node05 | physical | all | Intermittent link: 30 % of readings lost for 48 h | offline |
| O | any | PM side effect | neighbors | Ordering violation a fault causes in an adjacent PM bin | ordering |

Some faults are deliberately hard. D4 scales every bin together and cannot be
told apart from the air without a reference; P10 and D2 barely change PM
because the affected count bins carry little mass; S3 and C5 are below the
engine's PM effect size and are only caught when they break the bin ordering.
Automatic `O` labels are easy side effects of other faults, so judge recall on
the primary faults.

## Regenerate, verify, evaluate

```bash
python scripts/generate_synthetic_pm.py            # rewrite both datasets (seed 20250301)
python scripts/generate_synthetic_pm.py --check    # exit 1 if files differ from the generator
python scripts/evaluate_synthetic_pm.py            # accuracy reports, both families and modes (~20 min)
python scripts/evaluate_synthetic_pm.py --family pc --mode single-sensor
safe health mintsXU4/data/synthetic_pm/*.csv.gz --merge \
    --config mintsXU4/data/synthetic_pm/health-config.json -o mintsXU4/output/health_synthetic
```

`--check` and the tests compare gzip files decompressed, because the compressed
bytes differ between zlib builds. `--merge` is required when replaying the
per-bin files together: they cover the same period, so without it the second
file is rejected as overlapping. Merging also lets the IPS7100 bin-ordering
rule see every PM bin at each timestamp.

The reports (`evaluation_output/synthetic_pm/` and `evaluation_output/synthetic_pc/`,
`accuracy.md` and `.json`) score two modes. **single-sensor** matches `safe health`.
**with-references** also passes the same bin from the other five nodes as
co-located peers, which lets the engine separate a sensor fault from a shared
ambient change. Scoring reuses `safe.health_evaluation` (see
[evaluation.md](evaluation.md)): diagnostic and actionable recall per fault,
delays, and false actionable incidents, which the report also groups by the
nearest healthy look-alike.

The generator is the exact specification (`safe/synthetic_pm.py`); bump
`GENERATOR_REVISION` when changing it and regenerate the files.
