# Reproducible SAFE evaluation

```bash
python scripts/evaluate_health.py --require-targets
```

This replays the same fixed synthetic fixtures through `SensorHealth` for three
predeclared seeds (calibration 1729, development 2718, holdout 31415) and writes
strict JSON (`comparison.json`, undefined quantities use null) and a Markdown
summary (`comparison.md`) under the ignored `evaluation_output/health/`. Override
the directory with `--output` and the engine configuration with `--config`.
`--require-targets` exits 2 when any holdout acceptance target fails; CI runs it.
The runner never fits a parameter to these results.

`safe.scenarios` contains the input model and deterministic generators.
`safe.health_evaluation.replay_health` feeds readings (and any `references`) in
arrival order, drives `tick()` over absent readings without synthesizing
measurements, and restarts through `snapshot()`/`from_snapshot()` where a fixture
marks a restart. `score_incidents` attributes incidents to labels;
`evaluate_health` aggregates; `acceptance` checks targets. To add real labeled
data, construct `Scenario`, `Reading`, and `Fault` instances (or use
`safe.annotations`) and call `evaluate_health`. Times are Unix seconds in UTC.
Each declared sensor is assumed observed for the entire scenario interval; split
datasets into scenarios when exposure bounds differ between sensors.

## Attribution policy

An incident is evidence compatible with a label, not confirmation of a sensor
failure. An observation matches a fault when sensor and metric are identical, the
evidence category observed *at that time* is compatible, and the timestamp lies in
`[fault.start, fault.end)`; there is no grace period or early tolerance, and a later
upgrade of the incident's category never re-credits earlier observations. One
incident matches at most one fault; one fault may be matched by several incidents.

| Evidence category | Eligible label categories |
|---|---|
| invalid_measurement | physical_bounds |
| isolated_anomaly | spike, level_offset, calibration_drift, noise_increase |
| abrupt_shift, gradual_degradation | level_offset, calibration_drift |
| noise_change | noise_increase, sensitivity_loss |
| sensor_freeze | freeze |
| missing_data, completeness_loss, cadence_degradation | missing_data |
| timestamp_disorder, duplicate_timestamp | timestamp_order |

## Units

- **Diagnostic recall** counts faults matched by any evidence, including
  informational anomalies. **Actionable recall** requires warning or critical
  evidence during the labeled interval.
- **False actionable incidents** are warning/critical incidents with no attributed
  fault, whether or not rate limiting suppressed their notification. They are
  reported per sensor-day of scored exposure (elapsed time, including gaps).
- Delays are measured from fault onset to the first matching observation, for
  detected faults only; inspect recall alongside them.
- Notification counts verify that each incident notifies at most once.

## Fixtures and limitations

Eighteen independent seeded fixtures each span seven days at five-minute sampling.
They include IID Gaussian noise (sigma 0.3), stationary AR(1) rho 0.95, an
8-degree diurnal sinusoid, a 12-degree slow ramp, and a legitimate Gaussian-shaped
PM episode. Faults include a 150-degree reading, spaced 12-degree spikes, an
8-degree permanent offset, an 8-degree calibration ramp, a constant freeze, a
one-day gap, fivefold noise, sinusoidal sensitivity reduced to 15%, an out-of-order
arrival, a restart during a fault, and a startup freeze. Two `*_with_reference`
fixtures pair a calibration drift with a healthy trend of the same slope, each seen
alongside two co-located peers. Generator code is the exact specification.

These are simplified signals, not validated sensor physics or field labels. A
seven-day ramp is only a seasonal-like segment, and a single-sensor calibration
ramp remains a known miss: without a reference it cannot be separated from a real
ambient trend. PM episodes without peers remain uncertain. Results cover fixed
seeds, cadence, and configuration without confidence intervals or environmental
diversity. Do not infer field accuracy or readiness.
