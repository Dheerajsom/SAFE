"""Seeded synthetic IPS7100 PM dataset with labeled faults and healthy look-alikes.

Six co-located nodes report the seven cumulative PM bins (pm0_1 ... pm10_0) every
five minutes for 90 days. All nodes see one shared ambient signal (diurnal and
weekly cycles, regional pollution episodes, clean-air periods, a slow haze
trend); each node adds its own gain, micro-environment, and measurement noise.
Bins are built as cumulative sums of nonnegative size increments, so healthy
data never violates the size ordering. Faults are injected afterwards and every
one is labeled; ordering violations they cause in neighboring bins are labeled
automatically. Labels never feed the engine.

Generator code is the exact specification; `docs/synthetic-pm.md` summarizes it.
"""

from dataclasses import dataclass
import csv
import gzip
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from safe.annotations import LABEL_CATEGORIES, read_annotations
from safe.config import PM_METRICS
from safe.health_evaluation import evaluate_health
from safe.scenarios import Fault, Reading, Scenario

GENERATOR_REVISION = 1
DEFAULT_SEED = 20250301
START = pd.Timestamp("2025-03-01T00:00:00Z").timestamp()
DAYS = 90
INTERVAL = 300
MEASUREMENT = "IPS7100SYN"
NODES = tuple(f"node{i:02d}" for i in range(1, 7))
# Cumulative share of pm10_0 below each bin's cut size, for the mean aerosol.
BIN_FRACTIONS = (0.04, 0.22, 0.42, 0.58, 0.78, 0.91, 1.0)
ORDERING_TOLERANCE = 0.1  # SensorRules default; violations beyond it are evidence
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "mintsXU4" / "data" / "synthetic_pm"

# Healthy look-alikes shared by every node: (kind, start day, duration hours, factor)
EPISODES = (("pollution_event", 21.3, 30, 4.0), ("pollution_event", 38.6, 10, 6.0),
            ("pollution_event", 58.2, 20, 3.0), ("pollution_event", 80.4, 36, 5.0))
CLEAN_AIR = (("clean_air", 29.0, 18, 0.02), ("clean_air", 49.5, 12, 0.02),
             ("clean_air", 84.5, 24, 0.02))
HAZE = ("haze_trend", 62.0, 14 * 24, 1.8)


def sensor_name(node):
    """Name the loader assigns to a node (measurement + device ID)."""
    return f"{MEASUREMENT}_{node}"


@dataclass(frozen=True)
class Label:
    node: str
    metric: str
    start: float
    end: float
    label: str
    notes: str


@dataclass
class Dataset:
    times: np.ndarray                 # (n,) Unix seconds
    values: dict                      # node -> (n, 7) array; NaN = absent reading
    labels: list                      # fault Labels (engine-scored and loader-level)
    context: list                     # healthy look-alike windows, metric "all"
    loader_rows: dict                 # node -> extra (index, bin, value, position) rows
    seed: int


def _ar1(rng, n, tau_seconds, size=None):
    """Unit-variance AR(1) at the sampling interval with time constant tau."""
    rho = math.exp(-INTERVAL / tau_seconds)
    shape = (n,) if size is None else (size, n)
    noise = rng.normal(0, math.sqrt(1 - rho * rho), shape)
    return lfilter([1.0], [1.0, -rho], noise, axis=-1)


def _index(day, hour=0.0):
    return int(round((day * 86400 + hour * 3600) / INTERVAL))


def _bump(t, center_day, duration_hours):
    sigma = duration_hours * 3600 / 4
    return np.exp(-0.5 * ((t - START - center_day * 86400 - duration_hours * 1800) / sigma) ** 2)


def _ambient(rng, t):
    n = len(t)
    hours = ((t - START) % 86400) / 3600
    weekday = ((t - START) // 86400 + 5) % 7  # 2025-03-01 is a Saturday
    log_level = (math.log(12) + 0.40 * _ar1(rng, n, 8 * 3600) + 0.20 * _ar1(rng, n, 3 * 86400)
                 + 0.25 * np.cos(2 * np.pi * (hours - 8) / 24)
                 + 0.12 * np.cos(4 * np.pi * (hours - 19) / 24)
                 - 0.12 * (weekday >= 5))
    ambient = np.exp(log_level)
    for _, day, duration, factor in EPISODES:
        ambient *= 1 + (factor - 1) * _bump(t, day, duration)
    for _, day, duration, factor in CLEAN_AIR:
        ambient *= 1 - (1 - factor) * np.clip(_bump(t, day, duration) * 2.5, 0, 1)
    _, day, duration, factor = HAZE
    phase = np.clip((t - START - day * 86400) / (duration * 3600), 0, 1)
    ambient *= 1 + (factor - 1) * (1 - np.abs(2 * phase - 1)) * (phase > 0) * (phase < 1)
    # Slowly varying size composition, shared across co-located nodes.
    composition = np.exp(0.15 * _ar1(rng, n, 12 * 3600, len(BIN_FRACTIONS)))
    shares = np.diff((0.0,) + BIN_FRACTIONS)[:, None] * composition
    return ambient[None, :] * shares  # (7, n) size increments


def _node_values(rng, increments):
    bins, n = increments.shape
    gain = 1 + 0.06 * rng.normal(size=(bins, 1))
    micro = np.exp(0.05 * _ar1(rng, n, 3600))
    measured = increments * gain * micro * (1 + 0.04 * rng.normal(size=(bins, n)))
    measured = np.clip(measured + rng.normal(0, 0.03, (bins, n)), 0, None)
    return np.round(np.cumsum(measured, axis=0).T, 3)  # (n, 7), ordered by construction


def _inject(rng, values, t):
    """Apply every fault in place and return its labels."""
    labels = []
    col = {m: k for k, m in enumerate(PM_METRICS)}

    def label(node, metrics, i0, i1, kind, notes):
        for metric in metrics:
            labels.append(Label(node, metric, float(t[i0]), float(t[i0] + (i1 - i0) * INTERVAL), kind, notes))

    def nonzero_start(x, i):
        while not (x[i] > 0.1).all():
            i += 1
        return i

    # node02: data-quality faults ---------------------------------------------
    x = values["node02"]
    i = _index(10, 12)
    x[i, col["pm2_5"]] = -3.0
    label("node02", ["pm2_5"], i, i + 1, "invalid", "F1: pm2_5 negative reading (-3)")
    i = _index(16, 9)
    x[i:i + 3, col["pm10_0"]] = 25000.0
    label("node02", ["pm10_0"], i, i + 3, "invalid", "F2: pm10_0 above 10000 ug/m3 for 3 readings")
    i = nonzero_start(x, _index(24, 6))
    x[i:i + 96, col["pm1_0"]] = x[i, col["pm1_0"]]
    label("node02", ["pm1_0"], i, i + 96, "freeze", "F3: pm1_0 stuck for 8 h")
    i = _index(33, 3)
    x[i:i + 120] = np.nan
    label("node02", PM_METRICS, i, i + 120, "offline", "F4: all bins offline for 10 h")
    i = nonzero_start(x, _index(42, 1))
    x[i:i + 144] = x[i]
    label("node02", PM_METRICS, i, i + 144, "freeze", "F5: whole sensor frozen for 12 h")
    i = _index(60, 0)
    x[i:i + 288, col["pm5_0"]] = 0.0
    label("node02", ["pm5_0"], i, i + 288, "freeze", "F6: pm5_0 channel dead (reads 0) for 24 h")

    # node03: signal faults --------------------------------------------------
    x = values["node03"]
    spikes = [_index(10, h) for h in (2, 6, 10, 14, 18, 22)]
    x[spikes, col["pm2_5"]] += 60
    label("node03", ["pm2_5"], spikes[0], spikes[-1] + 1, "spike", "S1: six isolated +60 pm2_5 spikes")
    i = _index(18)
    x[i:i + 576, col["pm1_0"]] += 12
    label("node03", ["pm1_0"], i, i + 576, "offset", "S2: pm1_0 offset +12 for 48 h")
    i = _index(27)
    x[i:i + 576, col["pm0_3"]] += 2.5
    label("node03", ["pm0_3"], i, i + 576, "offset", "S3: subtle pm0_3 offset +2.5 for 48 h")
    i = _index(36)
    x[i:i + 288, col["pm10_0"]] = np.clip(x[i:i + 288, col["pm10_0"]] + rng.normal(0, 8, 288), 0, None)
    label("node03", ["pm10_0"], i, i + 288, "noise", "S4: pm10_0 noise sd 8 for 24 h")
    i = _index(45)
    x[i:i + 576, col["pm5_0"]] *= 0.35
    label("node03", ["pm5_0"], i, i + 576, "sensitivity", "S5: pm5_0 gain 0.35 for 48 h")
    i = _index(55)
    x[i:i + 144, col["pm2_5"]] += 25
    label("node03", ["pm2_5"], i, i + 144, "offset", "S6: pm2_5 offset +25 for 12 h")

    # node04: calibration drifts ---------------------------------------------
    x = values["node04"]
    i, ramp, hold = _index(15), _index(10), _index(5)
    x[i:i + ramp, col["pm2_5"]] += np.linspace(0, 10, ramp)
    x[i + ramp:i + ramp + hold, col["pm2_5"]] += 10
    label("node04", ["pm2_5"], i, i + ramp + hold, "drift", "D1: pm2_5 drifts +10 over 10 d, held 5 d")
    i, ramp, hold = _index(40), _index(15), _index(5)
    x[i:i + ramp, col["pm10_0"]] *= np.linspace(1, 1.7, ramp)
    x[i + ramp:i + ramp + hold, col["pm10_0"]] *= 1.7
    label("node04", ["pm10_0"], i, i + ramp + hold, "drift", "D2: pm10_0 gain drifts to 1.7 over 15 d, held 5 d")
    i, ramp = _index(65), _index(15)
    x[i:i + ramp, col["pm0_5"]] += np.linspace(0, 4, ramp)
    label("node04", ["pm0_5"], i, i + ramp, "drift", "D3: pm0_5 drifts +4 over 15 d (during haze trend)")

    # node05: cross-bin faults -----------------------------------------------
    x = values["node05"]
    i = _index(12)
    x[i:i + 288, [col["pm1_0"], col["pm2_5"]]] = x[i:i + 288, [col["pm2_5"], col["pm1_0"]]]
    label("node05", ["pm1_0", "pm2_5"], i, i + 288, "ordering", "C1: pm1_0/pm2_5 channels swapped for 24 h")
    i = _index(22)
    x[i:i + 144, col["pm0_1"]] = np.round(rng.uniform(0, 50, 144), 3)
    label("node05", ["pm0_1"], i, i + 144, "noise", "C2: pm0_1 channel outputs garbage for 12 h")
    i = _index(32, 14)
    x[i:i + 3] = 0.0
    label("node05", PM_METRICS, i, i + 3, "restart", "C3: restart, all bins read 0 for 3 readings")
    i = _index(45)
    x[i:i + 864] *= 1.5
    label("node05", PM_METRICS, i, i + 864, "offset", "C4: all bins read 1.5x for 3 d (reseated inlet)")
    i = _index(60)
    x[i:i + 144, col["pm0_5"]] = x[i:i + 144, col["pm1_0"]] + 0.6
    label("node05", ["pm0_5"], i, i + 144, "offset", "C5: pm0_5 reads 0.6 above pm1_0 for 12 h")

    for node in values:
        values[node] = np.round(values[node], 3)
    return labels


def _ordering_labels(values, t, labels, merge_gap=12):
    """Label bins pulled into an ordering violation by a fault in a neighbor bin."""
    result = []
    for node, x in values.items():
        present = ~np.isnan(x)
        bad = np.zeros_like(present)
        with np.errstate(invalid="ignore"):
            violation = (x[:, :-1] > x[:, 1:] + ORDERING_TOLERANCE) & present[:, :-1] & present[:, 1:]
        bad[:, :-1] |= violation
        bad[:, 1:] |= violation
        for k, metric in enumerate(PM_METRICS):
            existing = [(a.start, a.end) for a in labels if a.node == node and a.metric == metric]
            idx = np.flatnonzero(bad[:, k])
            if not len(idx):
                continue
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > merge_gap) + 1)
            for run in runs:
                start, end = float(t[run[0]]), float(t[run[-1]] + INTERVAL)
                if any(s < end and start < e for s, e in existing):
                    continue  # the primary fault label already covers this bin
                result.append(Label(node, metric, start, end, "ordering",
                                    "O: side effect of a neighboring-bin fault"))
    if any(not any(f.node == o.node and f.start <= o.start < f.end + 86400 for f in labels)
           for o in result):
        raise AssertionError("healthy data produced an unexplained ordering violation")
    return result


def _loader_rows(values):
    """Duplicate and out-of-order rows (node02, day 51) that the loader must clean up."""
    x = values["node02"]
    i = _index(51)
    rows = []
    for j in range(i, i + 48):  # 4 h of conflicting duplicates, appended after the originals
        for k in range(len(PM_METRICS)):
            if not np.isnan(x[j, k]):
                rows.append((j, k, round(float(x[j, k]) + 50.0, 3), "append"))
    return rows, (i + 60, i + 84)  # rows i+60..i+83 are written before i (out of order)


def generate(seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    t = START + np.arange(DAYS * 86400 // INTERVAL) * INTERVAL
    increments = _ambient(rng, t)
    values = {node: _node_values(rng, increments) for node in NODES}
    labels = _inject(rng, values, t)
    labels += _ordering_labels(values, t, labels)
    extra, shuffled = _loader_rows(values)
    labels += [Label("node02", m, float(t[_index(51)]), float(t[_index(51)] + 88 * INTERVAL), "clock",
                     "L1: conflicting duplicates and out-of-order rows (loader removes them)")
               for m in PM_METRICS]
    context = []
    for kind, day, duration, _ in EPISODES + CLEAN_AIR + (HAZE,):
        start = START + day * 86400
        for node in NODES:
            context.append(Label(node, "all", start, start + duration * 3600,
                                 "pollution_event" if kind == "pollution_event" else "healthy", kind))
    return Dataset(t, values, labels, context, {"node02": (extra, shuffled)}, seed)


def _iso(seconds):
    return pd.Timestamp(seconds, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _gzip_bytes(text):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=9) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def bin_csv_bytes(dataset, metric):
    """One InfluxDB-style long export for one PM bin, all nodes, gzip-compressed."""
    k = PM_METRICS.index(metric)
    stamps = [_iso(s) for s in dataset.times]
    lines = [",result,table,_time,_value,_field,_measurement,device_id"]
    for node in NODES:
        x = dataset.values[node][:, k]
        extra, shuffled = dataset.loader_rows.get(node, ((), None))
        order = list(range(len(x)))
        if shuffled is not None:
            a, b = shuffled
            late = order[a:b]
            del order[a:b]
            insert_at = a - 84
            order[insert_at:insert_at] = late
        for j in order:
            if not np.isnan(x[j]):
                lines.append(f",_result,0,{stamps[j]},{x[j]:.3f},{metric},{MEASUREMENT},{node}")
        for j, col, value, _ in extra:
            if col == k:
                lines.append(f",_result,0,{stamps[j]},{value:.3f},{metric},{MEASUREMENT},{node}")
    return _gzip_bytes("\n".join(lines) + "\n")


def _labels_csv(rows):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("sensor", "metric", "start", "end", "label", "confidence", "notes"))
    for a in sorted(rows, key=lambda a: (a.node, a.metric, a.start)):
        writer.writerow((sensor_name(a.node), a.metric, _iso(a.start), _iso(a.end), a.label, 1, a.notes))
    return out.getvalue()


def dataset_files(dataset):
    """Published path -> bytes for every file in the dataset directory."""
    files = {f"{m}.csv.gz": bin_csv_bytes(dataset, m) for m in PM_METRICS}
    files["labels.csv"] = _labels_csv(dataset.labels).encode()
    files["context.csv"] = _labels_csv(dataset.context).encode()
    config = {"profiles": {"deployment": "outdoor", "expected_interval_seconds": INTERVAL},
              "sensors": {sensor_name(n): {"model": "IPS7100", "site": "synthetic-colocation"} for n in NODES}}
    files["health-config.json"] = (json.dumps(config, indent=2) + "\n").encode()
    return files


def write_dataset(directory=DEFAULT_DIRECTORY, seed=DEFAULT_SEED):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = dataset_files(generate(seed))
    for name, data in files.items():
        (directory / name).write_bytes(data)
    return sorted(files)


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def load_scenario(directory=DEFAULT_DIRECTORY, references=False):
    """Replay-ready Scenario from the published files, labels as faults.

    Loader-level labels ("clock") are excluded: the loader removes those rows
    before the engine sees them, which `loader_check` verifies separately.
    With references=True every reading carries the same bin from the other
    co-located nodes at the same timestamp, as SensorHealth peers.
    """
    from safe.loader import load_pivoted_dataframe

    directory = Path(directory)
    frame, metrics = load_pivoted_dataframe([directory / f"{m}.csv.gz" for m in PM_METRICS])
    if frame is None:
        raise ValueError(f"cannot load synthetic PM dataset from {directory}")
    labels = read_annotations(directory / "labels.csv")
    faults = tuple(Fault(f"{a.notes.split(':')[0]}|{a.sensor}|{a.metric}|{int(a.start)}", a.sensor, a.metric,
                         LABEL_CATEGORIES[a.label], a.start, a.end)
                   for a in labels if a.label != "clock")
    readings = []
    columns = ["_sensor_name", "_unix_time"] + metrics
    for stamp, group in frame[columns].groupby("_unix_time", sort=True):
        rows = [(sensor, {m: float(v) for m, v in zip(metrics, values) if not math.isnan(v)})
                for sensor, _, *values in group.itertuples(index=False, name=None)]
        for sensor, values in rows:
            if not values:
                continue
            peers = None
            if references:
                peers = {m: [{"sensor": other, "value": v[m], "timestamp": float(stamp)}
                             for other, v in rows if other != sensor and m in v] for m in values}
            readings.append(Reading(float(stamp), sensor, values, references=peers))
    identities = tuple((sensor_name(n), m) for n in NODES for m in PM_METRICS)
    start = float(frame["_unix_time"].min())
    end = start + DAYS * 86400
    return Scenario("synthetic_pm_references" if references else "synthetic_pm_single_sensor",
                    tuple(readings), faults, start, end, identities,
                    sampling_interval_seconds=INTERVAL,
                    description="90-day co-located IPS7100 synthetic PM dataset",
                    parameters={"generator_revision": GENERATOR_REVISION, "references": references})


def loader_check(directory=DEFAULT_DIRECTORY):
    """Confirm the loader dropped injected duplicates and restored time order."""
    from safe.loader import load_pivoted_dataframe

    directory = Path(directory)
    raw = sum(len(pd.read_csv(directory / f"{m}.csv.gz")) for m in PM_METRICS)
    frame, metrics = load_pivoted_dataframe([directory / f"{m}.csv.gz" for m in PM_METRICS])
    kept = int(frame[metrics].notna().sum().sum())
    ordered = bool((frame.groupby("_sensor_name")["_unix_time"].diff().dropna() > 0).all())
    return dict(raw_rows=raw, pivoted_values=kept, removed_rows=raw - kept, time_ordered=ordered)


def evaluate(directory=DEFAULT_DIRECTORY, references=False):
    config = json.loads((Path(directory) / "health-config.json").read_text(encoding="utf-8"))
    scenario = load_scenario(directory, references)
    report = evaluate_health([scenario], config)
    context = read_annotations(Path(directory) / "context.csv")
    row = report["by_scenario"][0]
    matched = {i for f in row["faults"] for i in f["incident_ids"]}
    false = []
    for event in row["events"]:
        if (event["id"] in row["scored_actionable_ids"] and event["id"] not in matched
                and event["id"] not in row["misdiagnosed_ids"]):
            first = event["started_at"]
            windows = sorted({c.notes for c in context if c.sensor == event["sensor"]
                              and c.start - 3600 <= first < c.end + 6 * 3600})
            false.append(dict(sensor=event["sensor"], metric=event["metric"], category=event["category"],
                              severity=event["severity"], opened_at=_iso(first),
                              lookalike=", ".join(windows) or "none"))
    report["false_actionable"] = false
    return report


def _hours(seconds):
    return "—" if seconds is None else f"{seconds / 3600:.1f}"


def write_report(reports, output):
    """Write accuracy.json and accuracy.md for one or more evaluated modes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "accuracy.json").write_text(json.dumps(reports, indent=2, sort_keys=True, allow_nan=False)
                                          + "\n", encoding="utf-8")
    lines = ["# SAFE accuracy on the synthetic PM dataset", "",
             f"Generator revision {GENERATOR_REVISION}; 6 co-located nodes x 7 PM bins, 90 days at 5 min.",
             "Synthetic evidence only; not a field-accuracy claim. Timestamp faults are removed by",
             "the loader before the engine sees them and are checked separately.", "",
             "Misdiagnosed incidents first alarmed inside a labeled fault on the same bin but",
             "with an incompatible category; they earn no detection credit and are not false.", "",
             "| Mode | Faults | Detected | Actionable | Missed | False actionable incidents | per sensor-day "
             "| Misdiagnosed incidents |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, report in reports.items():
        row = report["by_scenario"][0]
        actionable = sum(f["actionable_detection_delay_seconds"] is not None for f in row["faults"])
        lines.append(f"| {mode} | {row['expected_faults']} | {row['detected_faults']} | {actionable} | "
                     f"{row['missed_faults']} | {row['false_actionable_incidents']} | "
                     f"{row['false_actionable_incidents_per_sensor_day']:.3f} | "
                     f"{row['misdiagnosed_actionable_incidents']} |")
    for mode, report in reports.items():
        row = report["by_scenario"][0]
        lines += ["", f"## {mode}", "", "| Fault | Node | Bin | Category | Detected | Actionable delay (h) |",
                  "|---|---|---|---|---|---:|"]
        for f in sorted(row["faults"], key=lambda f: (f["id"].split("|")[0] == "O", f["id"])):
            name, sensor, metric, _ = f["id"].split("|")
            lines.append(f"| {name} | {sensor.split('_')[-1]} | {metric} | {f['category']} | "
                         f"{'yes' if f['incident_ids'] else 'no'} | {_hours(f['actionable_detection_delay_seconds'])} |")
        lines += ["", "| Bin | Faults | Detected | False actionable incidents |", "|---|---:|---:|---:|"]
        for metric, m in report["by_metric"].items():
            lines.append(f"| {metric} | {m['expected']} | {m['detected']} | {m['false_actionable_incidents']} |")
        grouped = {}
        for item in report["false_actionable"]:
            key = (item["lookalike"], item["category"])
            grouped[key] = grouped.get(key, 0) + 1
        lines += ["", "False actionable incidents by nearby healthy look-alike and category:", "",
                  "| Look-alike | Category | Incidents |", "|---|---|---:|"]
        lines += [f"| {k[0]} | {k[1]} | {v} |" for k, v in sorted(grouped.items())] or ["| — | — | 0 |"]
    if "loader" in next(iter(reports.values())):
        check = next(iter(reports.values()))["loader"]
        lines += ["", f"Loader: {check['removed_rows']} duplicate rows removed; time order restored: "
                      f"{check['time_ordered']}."]
    (output / "accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
