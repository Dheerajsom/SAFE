"""Seeded synthetic IPS7100 PM and PC datasets with labeled faults and healthy look-alikes.

Six co-located nodes report the seven differential particle-count bins
(pc0_1 ... pc10_0, particles/L) and the seven cumulative PM bins (pm0_1 ...
pm10_0, ug/m3) every five minutes for 90 days. The statistics are fitted to the
2024-08 .. 2026-07 valo_node_01 IPS7100 1 s export averaged to 5 minutes:

* Counts are the primary signal. Each count bin is a shared ambient level (UTC
  diurnal cycle, weekend lift, multi-scale lognormal variability, regional
  pollution episodes, clean-air dips, a slow haze trend) times a slowly varying
  size composition. Large bins are Poisson-sampled, so pc5_0 and pc10_0 read
  exact zeros as often as in the field data.
* PM is derived exactly as the sensor derives it: each PM bin is the running sum
  of count bins times a fixed mass per particle (MASS_PER_COUNT, recovered from
  the field data to <0.5 %). Healthy PM is therefore always size-ordered.

Faults come in three kinds. Physical faults change the counts, so they reach PM
through the derivation and are labeled on every PM bin they materially change.
PM-output and PC-output faults change only that family's reported values (a
firmware or transmission fault), which is what makes a PM/PC mismatch evidence.
Ordering violations that faults cause in neighboring PM bins are labeled
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
from safe.config import IPS7100_MASS_PER_COUNT, PC_METRICS, PM_METRICS
from safe.health_evaluation import evaluate_health
from safe.scenarios import Fault, Reading, Scenario

GENERATOR_REVISION = 3
DEFAULT_SEED = 20250301
START = pd.Timestamp("2025-03-01T00:00:00Z").timestamp()
DAYS = 90
INTERVAL = 300
MEASUREMENT = "IPS7100SYN"
NODES = tuple(f"node{i:02d}" for i in range(1, 7))
FAMILIES = {"pm": PM_METRICS, "pc": PC_METRICS}
_DATA = Path(__file__).resolve().parents[1] / "mintsXU4" / "data"
DIRECTORIES = {"pm": _DATA / "synthetic_pm", "pc": _DATA / "synthetic_pc"}
DEFAULT_DIRECTORY = DIRECTORIES["pm"]
ORDERING_TOLERANCE = 0.1  # SensorRules default; violations beyond it are evidence

# Fitted to valo_node_01 (5-minute means of the 1 s export, 2024-08 .. 2026-07).
MASS_PER_COUNT = IPS7100_MASS_PER_COUNT
COUNT_MEDIANS = (88485.0, 31047.0, 10315.0, 728.0, 14.3, 0.237, 2e-5)  # median neighbor ratios
COMPOSITION_SD = (0.0, 0.35, 0.50, 0.30, 0.35, 0.60, 0.50)     # log neighbor-ratio variability
COMPOSITION_SLOW = (0.0, 0.65, 0.35, 0.10, 0.10, 0.10, 0.10)   # share of that variance lasting weeks
COMPOSITION_DIURNAL = (0.0, 0.0, 0.04, 0.07, 0.10, 0.10, 0.10)  # large particles peak mid-morning
POISSON_LITERS = 20.0  # effective sampled volume per 5-minute mean; sets large-bin zero rates
SATURATION = 9e5       # coincidence loss keeps healthy pc0_1 near the field maximum

# Healthy look-alikes shared by every node: Poisson-timed regional episodes and
# clean-air dips (rates, durations, and factors from the field data) and one haze trend.
EPISODE_RATE, EPISODE_HOURS, EPISODE_FACTOR = 0.12, 4.0, 4.0
DIP_RATE, DIP_HOURS, DIP_FACTOR = 0.10, 4.0, 0.15
HAZE = ("haze_trend", 58.0, 14 * 24, 1.8)


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
    values: dict                      # family -> node -> (n, 7) array; NaN = absent reading
    labels: dict                      # family -> fault Labels (engine-scored and loader-level)
    context: list                     # healthy look-alike windows, metric "all"
    loader_rows: dict                 # family -> node -> (extra rows, shuffled index range)
    seed: int


def _ar1(rng, n, tau_seconds, size=None):
    """Unit-variance AR(1) at the sampling interval with time constant tau."""
    rho = math.exp(-INTERVAL / tau_seconds)
    shape = (n,) if size is None else (size, n)
    noise = rng.normal(0, math.sqrt(1 - rho * rho), shape)
    start = rng.normal(0, 1, shape[:-1] + (1,))  # stationary from the first reading
    return lfilter([1.0], [1.0, -rho], noise, axis=-1) + start * rho ** np.arange(1, n + 1)


def _index(day, hour=0.0):
    return int(round((day * 86400 + hour * 3600) / INTERVAL))


def _bump(t, start, duration_hours):
    sigma = duration_hours * 3600 / 4
    return np.exp(-0.5 * ((t - start - duration_hours * 1800) / sigma) ** 2)


def _events(rng, rate, hours, factor, factor_sd, low, high):
    """Poisson-timed (start, duration hours, factor) events over the dataset."""
    count = rng.poisson(rate * DAYS)
    starts = np.sort(rng.uniform(0.5, DAYS - 1.5, count)) * 86400 + START
    durations = np.clip(hours * np.exp(0.7 * rng.normal(size=count)), 1, 36)
    factors = np.clip(factor * np.exp(factor_sd * rng.normal(size=count)), low, high)
    return [(float(s), float(d), float(f)) for s, d, f in zip(starts, durations, factors)]


def _ambient(rng, t):
    """Shared true counts (7, n) in particles/L and the look-alike windows."""
    n = len(t)
    hours = ((t - START) % 86400) / 3600
    weekday = ((t - START) // 86400 + 5) % 7  # 2025-03-01 is a Saturday
    level = (0.12 * np.cos(2 * np.pi * (hours - 10.3) / 24) + 0.07 * np.cos(4 * np.pi * (hours - 2.5) / 24)
             + 0.08 * (weekday >= 5) + 0.16 * _ar1(rng, n, 600) + 0.20 * _ar1(rng, n, 1.5 * 3600)
             + 0.45 * _ar1(rng, n, 16 * 3600) + 0.60 * _ar1(rng, n, 2.5 * 86400))
    ambient = np.exp(level - np.median(level))  # COUNT_MEDIANS stay the typical level
    _, haze_day, haze_hours, haze_factor = HAZE
    taken = [(START + haze_day * 86400, START + haze_day * 86400 + haze_hours * 3600)]

    def disjoint(events):  # context labels for one sensor must not overlap
        kept = []
        for start, hours_, f in events:
            end = start + hours_ * 3600
            if all(end + 3600 <= a or b + 3600 <= start for a, b in taken):
                taken.append((start, end))
                kept.append((start, hours_, f))
        return kept

    episodes = disjoint(_events(rng, EPISODE_RATE, EPISODE_HOURS, EPISODE_FACTOR, 0.5, 1.8, 15))
    dips = disjoint(_events(rng, DIP_RATE, DIP_HOURS, DIP_FACTOR, 0.4, 0.03, 0.4))
    for start, duration, factor in episodes:
        ambient *= 1 + (factor - 1) * _bump(t, start, duration)
    for start, duration, factor in dips:
        ambient *= 1 - (1 - factor) * np.clip(_bump(t, start, duration) * 2.5, 0, 1)
    phase = np.clip((t - START - haze_day * 86400) / (haze_hours * 3600), 0, 1)
    ambient *= 1 + (haze_factor - 1) * (1 - np.abs(2 * phase - 1)) * (phase > 0) * (phase < 1)
    sd, slow = np.array(COMPOSITION_SD)[:, None], np.array(COMPOSITION_SLOW)[:, None]
    bins = len(COUNT_MEDIANS)
    steps = sd * (np.sqrt(slow) * _ar1(rng, n, 10 * 86400, bins) + np.sqrt(1 - slow) * _ar1(rng, n, 6 * 3600, bins))
    composition = np.cumsum(steps, axis=0)  # neighboring bins share most of their size variation
    composition -= np.median(composition, axis=1, keepdims=True)
    composition += np.array(COMPOSITION_DIURNAL)[:, None] * np.cos(2 * np.pi * (hours - 13) / 24)
    counts = np.array(COUNT_MEDIANS)[:, None] * ambient[None, :] * np.exp(composition)
    counts /= np.sqrt(1 + (counts[0] / SATURATION) ** 2)
    windows = ([("pollution_event", s, d) for s, d, _ in episodes] + [("clean_air", s, d) for s, d, _ in dips]
               + [("haze_trend", START + haze_day * 86400, haze_hours)])
    return counts, windows


def _node_counts(rng, counts):
    """One node's measured counts (n, 7): gain, micro-environment, noise, counting statistics."""
    bins, n = counts.shape
    gain = 1 + 0.05 * rng.normal(size=(bins, 1))
    micro = np.exp(0.05 * _ar1(rng, n, 3600))
    measured = counts * gain * micro * np.exp(0.03 * rng.normal(size=(bins, n)))
    return (rng.poisson(measured * POISSON_LITERS) / POISSON_LITERS).T


def derive_pm(counts):
    """Cumulative PM (ug/m3) from differential counts, as the IPS7100 computes it."""
    return np.cumsum(counts * np.array(MASS_PER_COUNT), axis=1)


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------

PM_COL = {m: k for k, m in enumerate(PM_METRICS)}
PC_COL = {m: k for k, m in enumerate(PC_METRICS)}
ALL = "all"


def _physical_faults(rng, counts, t):
    """Faults in the counts themselves. Returns (node, bins, i0, i1, pc label, pm label, pm scope, notes)."""
    faults = []

    def add(node, bins, i0, i1, kind, pm_kind, scope, notes):
        faults.append((node, bins, i0, i1, kind, pm_kind, scope, notes))

    def nonzero_start(x, i):
        while not (x[i, :5] > 0).all():
            i += 1
        return i

    # node02: availability and dead channels
    x = counts["node02"]
    i = _index(33, 3)
    x[i:i + 120] = np.nan
    add("node02", ALL, i, i + 120, "offline", "offline", ALL, "F4: all bins offline for 10 h")
    i = nonzero_start(x, _index(42, 1))
    x[i:i + 144] = x[i]
    add("node02", ALL, i, i + 144, "freeze", "freeze", ALL, "F5: whole sensor frozen for 12 h")
    i = _index(66)
    x[i:i + 288, PC_COL["pc0_3"]] = 0.0
    add("node02", ["pc0_3"], i, i + 288, "freeze", "sensitivity", "auto",
        "P6: pc0_3 detector channel dead (reads 0) for 24 h")

    # node03: bursts, noise, laser power
    x = counts["node03"]
    bursts = [_index(12, h) for h in (1, 5, 9, 13, 17, 21)]
    x[bursts, PC_COL["pc2_5"]] += 3000
    add("node03", ["pc2_5"], bursts[0], bursts[-1] + 1, "spike", "spike", "auto",
        "P7: six isolated +3000/L pc2_5 bursts (insect or debris)")
    i = _index(40)
    x[i:i + 288, PC_COL["pc1_0"]] *= np.exp(rng.normal(0, 1.0, 288))
    add("node03", ["pc1_0"], i, i + 288, "noise", "noise", "auto", "P9: pc1_0 electrical noise (x lognormal sd 1) for 24 h")
    i = _index(50)
    x[i:i + 576, PC_COL["pc0_1"]] *= 0.3
    add("node03", ["pc0_1"], i, i + 576, "sensitivity", "sensitivity", "auto",
        "P10: pc0_1 sensitivity 0.3 for 48 h (laser power drop)")

    # node04: calibration drifts
    x = counts["node04"]
    i, ramp, hold = _index(28), _index(15), _index(5)
    x[i:i + ramp, PC_COL["pc2_5"]] *= np.linspace(1, 2.5, ramp)
    x[i + ramp:i + ramp + hold, PC_COL["pc2_5"]] *= 2.5
    add("node04", ["pc2_5"], i, i + ramp + hold, "drift", "drift", "auto",
        "D2: pc2_5 gain drifts to 2.5 over 15 d, held 5 d (dirty optics)")
    i, ramp, hold = _index(50), _index(20), _index(2)
    x[i:i + ramp, PC_COL["pc0_3"]] *= np.linspace(1, 0.4, ramp)
    x[i + ramp:i + ramp + hold, PC_COL["pc0_3"]] *= 0.4
    add("node04", ["pc0_3"], i, i + ramp + hold, "drift", "drift", "auto",
        "D3: pc0_3 gain drifts to 0.4 over 20 d, held 2 d (small-bin sensitivity loss, during haze trend)")
    i, ramp, hold = _index(73), _index(14), _index(2)
    x[i:i + ramp] *= np.linspace(1, 1.25, ramp)[:, None]
    x[i + ramp:i + ramp + hold] *= 1.25
    add("node04", ALL, i, i + ramp + hold, "drift", "drift", "auto",
        "D4: whole-sensor gain drifts to 1.25 over 14 d, held 2 d (needs a reference)")

    # node05: whole-sensor faults
    x = counts["node05"]
    i = _index(32, 14)
    x[i:i + 3] = 0.0
    add("node05", ALL, i, i + 3, "restart", "restart", ALL, "C3: restart, all bins read 0 for 3 readings")
    i = _index(38)
    x[i:i + 432] *= 0.15
    add("node05", ALL, i, i + 432, "sensitivity", "sensitivity", ALL, "C6: flow failure, all counts x0.15 for 36 h")
    i = _index(45)
    x[i:i + 864] *= 1.5
    add("node05", ALL, i, i + 864, "offset", "offset", ALL, "C4: all bins read 1.5x for 3 d (reseated inlet)")
    i = _index(75)
    drop = np.flatnonzero(rng.random(576) < 0.3) + i
    x[drop] = np.nan
    add("node05", ALL, i, i + 576, "offline", "offline", ALL, "C7: intermittent link, 30 % of readings lost for 48 h")
    return faults


def _pm_output_faults(rng, pm):
    """Faults in the reported PM values only. Returns (node, metrics, i0, i1, label, notes)."""
    faults = []
    x = pm["node02"]
    i = _index(10, 12)
    x[i, PM_COL["pm2_5"]] = -3.0
    faults.append(("node02", ["pm2_5"], i, i + 1, "invalid", "F1: pm2_5 negative reading (-3)"))
    i = _index(16, 9)
    x[i:i + 3, PM_COL["pm10_0"]] = 25000.0
    faults.append(("node02", ["pm10_0"], i, i + 3, "invalid", "F2: pm10_0 above 10000 ug/m3 for 3 readings"))
    i = _index(24, 6)
    x[i:i + 96, PM_COL["pm1_0"]] = x[i, PM_COL["pm1_0"]]
    faults.append(("node02", ["pm1_0"], i, i + 96, "freeze", "F3: pm1_0 output stuck for 8 h"))
    i = _index(60)
    x[i:i + 288, PM_COL["pm5_0"]] = 0.0
    faults.append(("node02", ["pm5_0"], i, i + 288, "freeze", "F6: pm5_0 output reads 0 for 24 h"))

    x = pm["node03"]
    spikes = [_index(10, h) for h in (2, 6, 10, 14, 18, 22)]
    x[spikes, PM_COL["pm2_5"]] += 60
    faults.append(("node03", ["pm2_5"], spikes[0], spikes[-1] + 1, "spike", "S1: six isolated +60 pm2_5 spikes"))
    i = _index(18)
    x[i:i + 576, PM_COL["pm1_0"]] += 12
    faults.append(("node03", ["pm1_0"], i, i + 576, "offset", "S2: pm1_0 offset +12 for 48 h"))
    i = _index(27)
    x[i:i + 576, PM_COL["pm0_3"]] += 1.5
    faults.append(("node03", ["pm0_3"], i, i + 576, "offset", "S3: subtle pm0_3 offset +1.5 for 48 h"))
    i = _index(36)
    x[i:i + 288, PM_COL["pm10_0"]] = np.clip(x[i:i + 288, PM_COL["pm10_0"]] + rng.normal(0, 8, 288), 0, None)
    faults.append(("node03", ["pm10_0"], i, i + 288, "noise", "S4: pm10_0 noise sd 8 for 24 h"))
    i = _index(45)
    x[i:i + 576, PM_COL["pm5_0"]] *= 0.35
    faults.append(("node03", ["pm5_0"], i, i + 576, "sensitivity", "S5: pm5_0 gain 0.35 for 48 h"))
    i = _index(56)
    x[i:i + 144, PM_COL["pm2_5"]] += 25
    faults.append(("node03", ["pm2_5"], i, i + 144, "offset", "S6: pm2_5 offset +25 for 12 h"))

    x = pm["node04"]
    i, ramp, hold = _index(10), _index(10), _index(5)
    x[i:i + ramp, PM_COL["pm2_5"]] += np.linspace(0, 6, ramp)
    x[i + ramp:i + ramp + hold, PM_COL["pm2_5"]] += 6
    faults.append(("node04", ["pm2_5"], i, i + ramp + hold, "drift", "D1: pm2_5 output drifts +6 over 10 d, held 5 d"))

    x = pm["node05"]
    i = _index(12)
    x[i:i + 288, [PM_COL["pm1_0"], PM_COL["pm2_5"]]] = x[i:i + 288, [PM_COL["pm2_5"], PM_COL["pm1_0"]]]
    faults.append(("node05", ["pm1_0", "pm2_5"], i, i + 288, "ordering", "C1: pm1_0/pm2_5 outputs swapped for 24 h"))
    i = _index(22)
    x[i:i + 144, PM_COL["pm0_1"]] = rng.uniform(0, 50, 144)
    faults.append(("node05", ["pm0_1"], i, i + 144, "noise", "C2: pm0_1 output is garbage for 12 h"))
    i = _index(60)
    x[i:i + 144, PM_COL["pm0_5"]] = x[i:i + 144, PM_COL["pm1_0"]] + 0.6
    faults.append(("node05", ["pm0_5"], i, i + 144, "offset", "C5: pm0_5 reads 0.6 above pm1_0 for 12 h"))
    return faults


def _pc_output_faults(rng, pc):
    """Faults in the reported counts only. Returns (node, metrics, i0, i1, label, notes)."""
    faults = []
    x = pc["node02"]
    i = _index(11, 12)
    x[i, PC_COL["pc0_5"]] = -40.0
    faults.append(("node02", ["pc0_5"], i, i + 1, "invalid", "P1: pc0_5 negative reading (-40)"))
    i = _index(17, 9)
    x[i:i + 3, PC_COL["pc1_0"]] = 4294967295.0
    faults.append(("node02", ["pc1_0"], i, i + 3, "invalid", "P2: pc1_0 reads 4,294,967,295 (uint32 overflow) for 3 readings"))
    i = _index(26, 6)
    x[i:i + 96, PC_COL["pc1_0"]] = x[i, PC_COL["pc1_0"]]
    faults.append(("node02", ["pc1_0"], i, i + 96, "freeze", "P3: pc1_0 output stuck for 8 h"))

    x = pc["node03"]
    i = _index(30)
    x[i:i + 576, PC_COL["pc0_5"]] += 20000
    faults.append(("node03", ["pc0_5"], i, i + 576, "offset", "P8: pc0_5 output offset +20000/L for 48 h"))

    x = pc["node05"]
    i = _index(14)
    x[i:i + 288, [PC_COL["pc0_3"], PC_COL["pc0_5"]]] = x[i:i + 288, [PC_COL["pc0_5"], PC_COL["pc0_3"]]]
    faults.append(("node05", ["pc0_3", "pc0_5"], i, i + 288, "offset", "P11: pc0_3/pc0_5 outputs swapped for 24 h"))
    i = _index(24)
    x[i:i + 144, PC_COL["pc5_0"]] = rng.uniform(0, 500, 144)
    faults.append(("node05", ["pc5_0"], i, i + 144, "noise", "P12: pc5_0 output is garbage for 12 h"))
    return faults


def _pm_effect(clean, faulty, k, i0, i1, spike):
    """Whether a physical fault materially changed PM bin k over [i0, i1)."""
    a, b = clean[i0:i1, k], faulty[i0:i1, k]
    if (np.isnan(a) != np.isnan(b)).any():
        return True
    with np.errstate(invalid="ignore", divide="ignore"):
        effect = np.abs(b - a) / np.maximum(np.abs(a), 0.01)
    effect = effect[np.isfinite(effect)]
    if not len(effect):
        return False
    return bool(effect.max() >= 0.1) if spike else bool(np.quantile(effect, 0.9) >= 0.1)


def _ordering_labels(values, t, labels, merge_gap=12):
    """Label PM bins pulled into an ordering violation by a fault in a neighbor bin."""
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


def _loader_rows(values, bump):
    """Duplicate and out-of-order rows (node02, day 51) that the loader must clean up."""
    x = values["node02"]
    i = _index(51)
    rows = []
    for j in range(i, i + 48):  # 4 h of conflicting duplicates, appended after the originals
        for k in range(x.shape[1]):
            if not np.isnan(x[j, k]):
                rows.append((j, k, bump(float(x[j, k]))))
    return {"node02": (rows, (i + 60, i + 84))}  # rows i+60..i+83 are written before i (out of order)


def generate(seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    t = START + np.arange(DAYS * 86400 // INTERVAL) * INTERVAL
    ambient, windows = _ambient(rng, t)
    clean = {node: _node_counts(rng, ambient) for node in NODES}
    counts = {node: x.copy() for node, x in clean.items()}
    physical = _physical_faults(rng, counts, t)
    clean_pm = {node: derive_pm(x) for node, x in clean.items()}
    pm = {node: derive_pm(x) for node, x in counts.items()}
    pc = {node: x.copy() for node, x in counts.items()}

    def span(node, metric, i0, i1, kind, notes):
        return Label(node, metric, float(t[i0]), float(t[i0] + (i1 - i0) * INTERVAL), kind, notes)

    labels = {"pm": [], "pc": []}
    for node, bins, i0, i1, kind, pm_kind, scope, notes in physical:
        labels["pc"] += [span(node, m, i0, i1, kind, notes) for m in (PC_METRICS if bins == ALL else bins)]
        for k, metric in enumerate(PM_METRICS):
            if scope == ALL or _pm_effect(clean_pm[node], pm[node], k, i0, i1, pm_kind == "spike"):
                labels["pm"].append(span(node, metric, i0, i1, pm_kind, notes + " (via counts)"))
    for family, injected in (("pm", _pm_output_faults(rng, pm)), ("pc", _pc_output_faults(rng, pc))):
        labels[family] += [span(node, m, i0, i1, kind, notes) for node, ms, i0, i1, kind, notes in injected for m in ms]
    values = {"pm": {n: np.round(x, 5) for n, x in pm.items()}, "pc": {n: np.round(x, 2) for n, x in pc.items()}}
    labels["pm"] += _ordering_labels(values["pm"], t, labels["pm"])
    loader_rows = {"pm": _loader_rows(values["pm"], lambda v: round(v + 50.0, 5)),
                   "pc": _loader_rows(values["pc"], lambda v: round(v * 1.5 + 100.0, 2))}
    for family, metrics in FAMILIES.items():
        labels[family] += [Label("node02", m, float(t[_index(51)]), float(t[_index(51)] + 88 * INTERVAL), "clock",
                                 "L1: conflicting duplicates and out-of-order rows (loader removes them)")
                           for m in metrics]
    context = [Label(node, "all", start, start + duration * 3600,
                     "pollution_event" if kind == "pollution_event" else "healthy", kind)
               for kind, start, duration in windows for node in NODES]
    return Dataset(t, values, labels, context, loader_rows, seed)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def _iso(seconds):
    return pd.Timestamp(seconds, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _gzip_bytes(text):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=9) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def bin_csv_bytes(dataset, metric):
    """One InfluxDB-style long export for one bin, all nodes, gzip-compressed."""
    family = "pm" if metric in PM_METRICS else "pc"
    k = FAMILIES[family].index(metric)
    digits = 5 if family == "pm" else 2
    stamps = [_iso(s) for s in dataset.times]
    lines = [",result,table,_time,_value,_field,_measurement,device_id"]
    for node in NODES:
        x = dataset.values[family][node][:, k]
        extra, shuffled = dataset.loader_rows[family].get(node, ((), None))
        order = list(range(len(x)))
        if shuffled is not None:
            a, b = shuffled
            late = order[a:b]
            del order[a:b]
            insert_at = a - 84
            order[insert_at:insert_at] = late
        for j in order:
            if not np.isnan(x[j]):
                lines.append(f",_result,0,{stamps[j]},{x[j]:.{digits}f},{metric},{MEASUREMENT},{node}")
        for j, col, value in extra:
            if col == k:
                lines.append(f",_result,0,{stamps[j]},{value:.{digits}f},{metric},{MEASUREMENT},{node}")
    return _gzip_bytes("\n".join(lines) + "\n")


def _labels_csv(rows):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("sensor", "metric", "start", "end", "label", "confidence", "notes"))
    for a in sorted(rows, key=lambda a: (a.node, a.metric, a.start)):
        writer.writerow((sensor_name(a.node), a.metric, _iso(a.start), _iso(a.end), a.label, 1, a.notes))
    return out.getvalue()


def dataset_files(dataset, family="pm"):
    """Published path -> bytes for every file in one family's dataset directory."""
    files = {f"{m}.csv.gz": bin_csv_bytes(dataset, m) for m in FAMILIES[family]}
    files["labels.csv"] = _labels_csv(dataset.labels[family]).encode()
    files["context.csv"] = _labels_csv(dataset.context).encode()
    config = {"profiles": {"deployment": "outdoor", "expected_interval_seconds": INTERVAL},
              "sensors": {sensor_name(n): {"model": "IPS7100", "site": "synthetic-colocation"} for n in NODES}}
    files["health-config.json"] = (json.dumps(config, indent=2) + "\n").encode()
    return files


def file_content(name, data):
    """Comparable content of a dataset file: gzip bytes vary with the zlib build, text does not."""
    return gzip.decompress(data) if name.endswith(".gz") else data


def write_dataset(directories=None, seed=DEFAULT_SEED):
    """Write both families; `directories` maps family -> output directory."""
    dataset = generate(seed)
    written = []
    for family, directory in (directories or DIRECTORIES).items():
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in dataset_files(dataset, family).items():
            (directory / name).write_bytes(data)
            written.append(directory / name)
    return written


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def _metrics(family):
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    return FAMILIES[family]


def load_scenario(directory=DEFAULT_DIRECTORY, references=False, family="pm"):
    """Replay-ready Scenario from the published files, labels as faults.

    Loader-level labels ("clock") are excluded: the loader removes those rows
    before the engine sees them, which `loader_check` verifies separately.
    With references=True every reading carries the same bin from the other
    co-located nodes at the same timestamp, as SensorHealth peers.
    """
    from safe.loader import load_pivoted_dataframe

    directory = Path(directory)
    frame, metrics = load_pivoted_dataframe([directory / f"{m}.csv.gz" for m in _metrics(family)])
    if frame is None:
        raise ValueError(f"cannot load synthetic {family.upper()} dataset from {directory}")
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
    identities = tuple((sensor_name(n), m) for n in NODES for m in _metrics(family))
    start = float(frame["_unix_time"].min())
    end = start + DAYS * 86400
    mode = "references" if references else "single_sensor"
    return Scenario(f"synthetic_{family}_{mode}", tuple(readings), faults, start, end, identities,
                    sampling_interval_seconds=INTERVAL,
                    description=f"90-day co-located IPS7100 synthetic {family.upper()} dataset",
                    parameters={"generator_revision": GENERATOR_REVISION, "references": references,
                                "family": family})


def loader_check(directory=DEFAULT_DIRECTORY, family="pm"):
    """Confirm the loader dropped injected duplicates and restored time order."""
    from safe.loader import load_pivoted_dataframe

    directory = Path(directory)
    paths = [directory / f"{m}.csv.gz" for m in _metrics(family)]
    raw = sum(len(pd.read_csv(p)) for p in paths)
    frame, metrics = load_pivoted_dataframe(paths)
    kept = int(frame[metrics].notna().sum().sum())
    ordered = bool((frame.groupby("_sensor_name")["_unix_time"].diff().dropna() > 0).all())
    return dict(raw_rows=raw, pivoted_values=kept, removed_rows=raw - kept, time_ordered=ordered)


def evaluate(directory=DEFAULT_DIRECTORY, references=False, family="pm"):
    config = json.loads((Path(directory) / "health-config.json").read_text(encoding="utf-8"))
    scenario = load_scenario(directory, references, family)
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


def write_report(reports, output, family="pm"):
    """Write accuracy.json and accuracy.md for one or more evaluated modes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "accuracy.json").write_text(json.dumps(reports, indent=2, sort_keys=True, allow_nan=False)
                                          + "\n", encoding="utf-8")
    name = family.upper()
    lines = [f"# SAFE accuracy on the synthetic {name} dataset", "",
             f"Generator revision {GENERATOR_REVISION}; 6 co-located nodes x 7 {name} bins, 90 days at 5 min.",
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
            fault, sensor, metric, _ = f["id"].split("|")
            lines.append(f"| {fault} | {sensor.split('_')[-1]} | {metric} | {f['category']} | "
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
