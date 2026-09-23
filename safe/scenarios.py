"""Deterministic fixtures; labels describe injected faults, not detector outputs."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Reading:
    timestamp: float
    sensor: str
    values: dict[str, float]
    restart: bool = False  # discard all engine state immediately before this input
    expected_rejection: bool = False  # specifically an out-of-order ValueError
    references: dict | None = None  # metric -> co-located peer readings (SensorHealth only)


@dataclass(frozen=True)
class Fault:
    id: str
    sensor: str
    metric: str
    category: str
    start: float
    end: float  # half-open interval [start, end)


@dataclass(frozen=True)
class Scenario:
    name: str
    readings: tuple[Reading, ...]  # arrival order, deliberately NOT sorted
    faults: tuple[Fault, ...]
    start: float
    end: float
    identities: tuple[tuple[str, str], ...]  # includes identities absent during gaps
    seed: int | None = None
    sampling_interval_seconds: float | None = None
    description: str = ""
    parameters: dict = field(default_factory=dict)


def synthetic_scenarios(seed: int = 1729) -> list[Scenario]:
    """Eighteen 7-day, 5-minute fixtures (including a startup-freeze control).

    Temperature is Celsius; PM is ug/m3. Each fixture gets an independent fixed
    child seed. Slow movement is a short seasonal-like segment, not a full year.
    The two *_with_reference fixtures pair a drift with a healthy trend of the
    same slope, each seen alongside two co-located peers: only a reference can
    tell them apart.
    """
    names = (
        "healthy_iid", "healthy_ar1", "healthy_diurnal", "healthy_slow_movement",
        "legitimate_pm_episode", "impossible_reading", "isolated_spikes",
        "abrupt_offset", "calibration_drift", "frozen_sensor", "missing_data",
        "increased_noise", "decreased_sensitivity", "out_of_order",
        "restart_during_fault", "startup_freeze",
        "calibration_drift_with_reference", "healthy_slow_movement_with_reference",
    )
    start, dt, n = 1735689600.0, 300.0, 2016
    times = start + np.arange(n) * dt
    result = []
    for index, name in enumerate(names):
        child_seed = seed + index
        rng = np.random.default_rng(child_seed)
        noise = rng.normal(0, 0.3, n)
        values = 20 + noise
        metric, sensor = "temperature", "synthetic-001"
        labels, omit, restarts, rejected = [], set(), set(), set()
        peers = None
        onset = 720

        def fault(category, first=onset, stop=n):
            labels.append(Fault(f"{name}:{len(labels)}", sensor, metric, category,
                                float(times[first]), start + stop * dt))

        if name == "healthy_ar1":
            ar = np.empty(n)
            ar[0] = noise[0]
            for i in range(1, n):
                ar[i] = 0.95 * ar[i - 1] + np.sqrt(1 - 0.95**2) * noise[i]
            values = 20 + ar
        elif name == "healthy_diurnal":
            values += 8 * np.sin(2 * np.pi * np.arange(n) / 288)
        elif name == "healthy_slow_movement":
            values += np.linspace(0, 12, n)
        elif name == "legitimate_pm_episode":
            metric = "pm2_5"
            values = 10 + noise + 80 * np.exp(-0.5 * ((np.arange(n) - 1000) / 90)**2)
        elif name == "impossible_reading":
            values[onset] = 150
            fault("physical_bounds", stop=onset + 1)
        elif name == "isolated_spikes":
            for i in range(onset, n - 100, 144):
                values[i] += 12
                fault("spike", i, i + 1)
        elif name in ("abrupt_offset", "restart_during_fault"):
            values[onset:] += 8
            fault("level_offset")
            if name == "restart_during_fault":
                restarts.add(onset + 1)
        elif name == "calibration_drift":
            values[onset:] += np.linspace(0, 8, n - onset)
            fault("calibration_drift")
        elif name == "frozen_sensor":
            values[onset:] = 20
            fault("freeze")
        elif name == "missing_data":
            omit.update(range(onset, onset + 288))
            fault("missing_data", stop=onset + 288)
        elif name == "increased_noise":
            values[onset:] = 20 + 5 * noise[onset:]
            fault("noise_increase")
        elif name == "decreased_sensitivity":
            stimulus = 3 * np.sin(2 * np.pi * np.arange(n) / 96)
            values += stimulus
            values[onset:] = 20 + 0.15 * stimulus[onset:] + noise[onset:]
            fault("sensitivity_loss")
        elif name == "out_of_order":
            rejected.add(onset)
            fault("timestamp_order", onset - 2, onset + 1)
        elif name == "startup_freeze":
            values[:] = 20
            fault("freeze", 0)
        elif name.endswith("_with_reference"):
            ambient = 20 + 8 * np.sin(2 * np.pi * np.arange(n) / 288)
            if name == "healthy_slow_movement_with_reference":
                ambient += np.linspace(0, 12, n)
            values = ambient + noise
            if name == "calibration_drift_with_reference":
                values[onset:] += np.linspace(0, 8, n - onset)
                fault("calibration_drift")
            peers = ambient + rng.normal(0, 0.3, (2, n))

        def references(i):
            if peers is None:
                return None
            return {metric: [{"sensor": f"reference-{k}", "value": float(peers[k, i]),
                              "timestamp": float(times[i])} for k in range(len(peers))]}

        readings = tuple(
            Reading(float(times[i - 2] if i in rejected else times[i]), sensor,
                    {metric: float(values[i])}, i in restarts, i in rejected, references(i))
            for i in range(n) if i not in omit
        )
        result.append(Scenario(
            name, readings, tuple(labels), start, start + n * dt,
            ((sensor, metric),), child_seed, dt, name.replace("_", " "),
            {"baseline": 20, "noise_sigma": 0.3, "fault_onset_index": onset,
             "generator_revision": 1},
        ))
    return result
