# SAFE sensor health

`SensorHealth` is SAFE's streaming engine: a sensor-health and data-quality
engine that produces correlated incidents with evidence and recovery. It replaced
the SAFE 2 per-reading `SensorDrift` alert engine in SAFE 3; `safe stream` is now an
alias of `safe health`. No live serial, MQTT, downloader, or credential dependency
is introduced.

## Analyze a deployment

From the repository root with its local environment:

```powershell
.\.venv\Scripts\python.exe -m safe.cli health path/to/export.csv `
  --config docs/health-config.json --state-out mintsXU4/output/health/state.json

.\.venv\Scripts\python.exe -m safe.cli health path/to/next-export.csv `
  --state-in mintsXU4/output/health/state.json `
  --state-out mintsXU4/output/health/state.json
```

Output defaults to `mintsXU4/output/health/`:

- `events.jsonl`: append-only opening, update, escalation, recovery, and closure evidence.
- `notifications.jsonl`: at most one operator-notification attempt per incident.
- `summary.json`: configuration, cumulative counts, and retained incident snapshots.
- `state.json`: only written when requested, and only after successful replay.

`--metric` restricts metrics; `-o` selects an output directory. Input files must
be chronological and non-overlapping with each other and any saved state.
`--tick-until 2026-09-12T12:00:00Z` supplies an explicit observation end for silence
after the final input. SAFE never assumes that an archived file's last reading
means the sensor subsequently went offline. Missing/invalid inputs exit nonzero.
CSV loading normalizes timestamps, sorts and deduplicates, and warns when
discarding invalid data. To audit original arrival order, duplicates, or nonfinite
measurements as incidents, use the streaming API before that cleanup.

The JSONL files append when the directory is reused; use a fresh directory for an
independent replay. They are an evidence archive and require external rotation for
continuous operation. Repeated identical-category updates export at most every
30 minutes; lifecycle changes and new detector evidence export immediately.
`--event-update-interval 0` exports every observation. Counts and latest snapshots
retain every detection regardless of export spacing. Repeated structured log updates
use DEBUG to keep ordinary runtime logs manageable. A failed replay may leave partial evidence; do not replay it
blindly into the same output directory. Resume from an appropriate checkpoint.

## Continuous use

```python
from safe import SensorHealth

engine = SensorHealth(on_notification=lambda event: print(event))
engine.register("node-001", "temperature", started_at="2026-09-12T00:00:00Z")
engine.data_processing("node-001", {
    "dateTime": "2026-09-12T00:05:00Z", "temperature": 24.7,
}, received_at="2026-09-12T00:05:03Z")
engine.tick("2026-09-12T00:25:00Z")
engine.save_state("mintsXU4/output/health/state.json")
engine = SensorHealth.load_state("mintsXU4/output/health/state.json")
```

Call `tick` from your scheduler even when no readings arrive. Register expected
sensor/metric identities to detect devices that never produce an initial reading.
`received_at` enables future-clock and delayed-arrival checks against a real clock;
without it the engine only knows observation timestamps. Naive dates mean UTC.
Do not feed backlog older than the clock's allowed delay into a live engine.
Equal timestamps can carry different fields, but a repeated timestamp for the same
field is rejected from the baseline and recorded as a timestamp incident.

`on_event(action, event)` receives detached JSON-compatible snapshots.
`on_notification(event)` receives one attempt when severity first becomes
actionable, subject to minimum spacing per sensor/metric. Escalations remain in
the incident evidence without issuing a second notification. Callback failures are
logged and ingestion continues. Notification delivery is **not** exactly-once:
an external durable outbox is needed for guaranteed delivery across crashes or
callback failures. Checkpoint/notification side effects are not a distributed
transaction. No callbacks, credentials, or executable objects are serialized.

## What the engine measures

| Evidence | Behavior |
|---|---|
| Invalid/nonfinite/out-of-bounds value | Immediate critical data-quality incident; baseline excludes it |
| Duplicate, backward, future timestamp | Timestamp evidence; quarantines the measurement |
| Silence or excessive gaps | Clock-driven availability incident |
| Slower cadence or missing slots | Median interval and elapsed-time completeness evidence |
| Repeated/nearly identical readings | Time- and run-length freeze detector, including startup |
| Large isolated residual | Informational anomaly, correlated with subsequent change evidence |
| Persistent residual offset | Warning after the configured consecutive-reading requirement |
| Excess residual anomalies | Persistent noise-change evidence with a longer quiet recovery period |
| Repeated mean/variance tests | Effect gates, AR(1) correction, persistence, and per-series alpha spending |
| PM-bin ordering/status/dewpoint rules | Opt-in model-specific plausibility evidence |
| Simultaneous jumps | Possible restart evidence; cause remains uncertain |
| Shared PM movement with calibrated peers | Informational environmental event |
| PM movement without peers | Informational uncertain change; inspect neighboring/reference data |

An incident aggregates related detectors by sensor, metric, and symptom family.
An anomaly followed by a persistent step retains its ID and becomes more severe.
Validity, availability, freeze, timestamp, and plausibility have separate families
so a new independent data-quality problem is not hidden by an old change incident.
Cross-metric restart evidence lists all jumping fields but does not assert that
they share a proven hardware cause. Confidence is a bounded evidence-strength
heuristic, **not** a calibrated probability of failure.

State progresses through `WARMING_UP`, `MONITORING`, `SUSPECTED`, `INCIDENT`, and
`RECOVERING`. Warm-up requires both sample count and elapsed coverage. A provisional
robust gate rejects gross startup contamination and repeated contamination restarts
the provisional sample. Daily environmental modeling waits for sufficient phase
coverage. An unknown stable offset present throughout startup cannot be identified
without a reference. PM and particle-count zeros are allowed; freeze rules do not
call stable clean-air zero values a failure. Customize tolerances to the instrument's quantization.

The expected signal is a robust UTC phase profile, interpolated between bins, with
a rate-limited seasonal level. Each phase retains a bounded number of cycle
representatives and a robust local scale. Suspect outliers, invalid values, freeze,
and plausibility failures cannot train the environmental profile. A daily period
is the default; a justified weekly profile can use a seven-day period and a matching
warm-up duration. Local-time/DST profiles and automatic calendar-model selection
are not inferred. Mobile/indoor profiles select a configuration context; those
deployment types still require calibration against their actual operating data.

Page-Hinkley is off by default. Enabling it requires
`stationary_residuals=True`; its input is a standardized health residual and it
resets when the expected model or reference mode changes. Explicit stationary
profiles keep the fitted expected model fixed. This assertion needs
deployment evidence; it is not an automatic stationarity test. Windowed tests
compare elapsed half-windows, run less frequently, and spend
`family_alpha / (2*k*(k+1))` on each of two tests at evaluation `k`. The sum bounds
the nominal testing budget per series, assuming valid p-values. AR(1) thinning is
an approximation; this is not proof of field false-positive calibration.

## Profiles and references

`MetricProfile` validates units, finite ranges, capacities, and durations. Defaults
target outdoor MINTS at 300-second intervals, with a 24-hour warm-up. Metric-specific
scales, effects, freeze rules, and physical limits differ for temperature, pressure,
humidity, voltage, PM, and particle counts. Override known deployments rather than
assuming these candidate defaults are calibrated for every sensor.

```python
from dataclasses import replace
from safe import ProfileRegistry, SensorHealth
from safe.profiles import metric_profile

p = replace(metric_profile("temperature"), freeze_tolerance=0.01)
profiles = ProfileRegistry(overrides=[{
    "match": {"model": "my-model", "metric": "temperature", "site": "site-a"},
    "profile": p.to_dict(),
}])
engine = SensorHealth(profiles=profiles, sensors={
    "node-001": {"model": "my-model", "site": "site-a"},
})
```

Selectors support sensor, model, metric, deployment, site, and nominal cadence.
The most specific match wins; the last entry wins ties. JSON overrides contain full
profiles to make the configuration reproducible. `max_samples` must accommodate
the configured elapsed-time windows. At faster sampling, memory budgets increase
accordingly; storage remains capped. `max_series` caps tracked identities, and
`retire(sensor, metric)` explicitly removes an inactive series without open incidents.

Reference input is opt-in. Supply the same trusted monitor or compatible peer
cohort during warm-up and operation. At least two distinct peers, or one explicitly
trusted reference, must be fresh and no later than the target observation. The
target-to-consensus offset needs a minimum-history calibration before it is used.
Introducing references into an existing engine also needs that calibration; it
does not silently assume all instruments have zero offset. Use a new calibration
when changing peer cohorts, sensor placement, or instrument units.

```python
engine.data_processing("node-001", {
    "unix_timestamp": 1789218300, "temperature": 24.7,
}, references={"temperature": [
    {"sensor": "reference", "timestamp": 1789218300,
     "value": 24.6, "trusted": True},
]})
```

## Evaluation and field evidence

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_health.py --require-targets
.\.venv\Scripts\python.exe -m pytest tests/
```

The first command evaluates the health engine on fixed calibration, development,
and holdout seeds. It writes strict JSON plus a Markdown table under ignored
`evaluation_output/health/`. There is no automatic threshold tuning against labels.
Fault attribution uses exact sensor/metric, compatible evidence categories, and
evidence observed inside the fault interval. An incident can match one fault;
repeated observations never create extra true positives. Reports distinguish
diagnostic recall, actionable recall, actionable incident precision, false
incidents per sensor-day, notification counts, and delays. See
[evaluation.md](evaluation.md) for the attribution table and units.

Use `docs/fault-labels-template.csv` for operator annotation. Supported labels are
`healthy`, `pollution_event`, `maintenance`, `freeze`, `offline`, `offset`, `spike`,
`invalid`, `drift`, `noise`, `sensitivity`, `clock`, and `unknown`.
`read_annotations` validates identities, UTC intervals, confidence, and overlaps.
`labeled_scenarios` preserves continuous input while scoring only sufficiently
confident known intervals; unlabeled/unknown/maintenance time is excluded from
exposure. Include healthy lead-in time so fault onset has a trained baseline.
`chronological_split` creates distinct real-data partitions and rejects faults
straddling split boundaries. Choose these boundaries before calibration.

The synthetic report is a pilot of the software, not a field reliability claim.
The single-sensor slow calibration-ramp fixture remains a documented miss: its
trajectory is indistinguishable from the accepted seasonal/environmental trend.
Decreased sensitivity can produce general change evidence without a specific
diagnosis. Peer/reference comparisons and independently labeled field experiments
are needed to resolve those ambiguities.

For a field pilot, record exact fault start/recovery UTC times, sensor identity,
metric/units, sampling cadence, maintenance, weather/pollution context, and operator
confidence. Begin with fixed-value, offset, noise, disconnection, and clock injection.
Physical inlet/fan interventions require the instrument owner's approved procedure;
the software does not operate that hardware. Predeclare calibration/development/
holdout dates and inspect both missed faults and false incidents before accepting
deployment-specific thresholds.

## Upgrade and operational limits

Install with `pip install -e ".[dev]"`. The package and engine version is 3.0.0;
health checkpoint schema is 1. Engine 3.0.0 compares PM and particle-count bins with
a reference by ratio rather than difference (profile `reference_ratio_floor`) and
requires reference drift to exceed both `reference_drift_tolerance` and
`reference_drift_relative_tolerance` times the reference-predicted level, so a
co-located sensor a few percent off does not alarm during pollution episodes.
Engine 2.0.0 checkpoints are rejected; start a new engine.
Checkpoints embed complete configuration, detector history, baseline bins, incident
IDs, cooldowns, notification attempts, and engine version. Writes use a same-directory
temporary file and atomic replacement with a checksum; incompatible schemas/versions
and checksum failures are rejected. Start a new engine for configuration changes.
JSON avoids executable pickle state; only load checkpoints from a trusted source.

In-memory histories and identities are bounded. Callback-owned storage, JSONL log
files, and the batch CSV loader are not: CSV replay currently pivots a file in memory.
Use smaller chronological export files for large archives. Live streaming ingestion
does not need that batch loader. Retained events carry engine version and profile
configuration; the stream summary/checkpoint carries the complete registry and rules.
The CI workflow runs Python 3.10/3.12 tests and publishes the synthetic comparison.

For local offline verification with pre-existing source/output changes, run
`scripts/verify_offline_health.py --output evaluation_output/NEW_DIRECTORY`.
It extracts committed source data into that fresh directory, overlays current SAFE
code, runs all five existing offline workflows plus health replay, checks CSV
schemas/probabilities/counts and PNG decoding, and records source/data hashes.
