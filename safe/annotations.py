"""Operator labels and chronological split boundaries; labels never feed detectors."""

from dataclasses import dataclass
import csv
import math

from safe.health import utc_seconds
from safe.scenarios import Fault, Reading, Scenario

LABEL_CATEGORIES = {"freeze": "freeze", "offline": "missing_data", "offset": "level_offset",
                    "spike": "spike", "invalid": "physical_bounds", "drift": "calibration_drift",
                    "noise": "noise_increase", "sensitivity": "sensitivity_loss",
                    "clock": "timestamp_order", "ordering": "bin_ordering", "restart": "restart"}
CONTEXT_LABELS = {"healthy", "pollution_event", "maintenance", "unknown"}
ANNOTATION_COLUMNS = ("sensor", "metric", "start", "end", "label", "confidence", "notes")


@dataclass(frozen=True)
class Annotation:
    sensor: str
    metric: str
    start: float
    end: float
    label: str
    confidence: float
    notes: str = ""

    def __post_init__(self):
        if not self.sensor.strip() or not self.metric.strip():
            raise ValueError("annotation identity cannot be blank")
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.end <= self.start:
            raise ValueError("annotation must have an ordered finite interval")
        if self.label not in LABEL_CATEGORIES and self.label not in CONTEXT_LABELS:
            raise ValueError(f"unknown annotation label: {self.label}")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("annotation confidence must lie in [0,1]")


def read_annotations(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(ANNOTATION_COLUMNS):
            raise ValueError("annotation columns must be " + ",".join(ANNOTATION_COLUMNS))
        result = [Annotation(row["sensor"], row["metric"], utc_seconds(row["start"]),
                             utc_seconds(row["end"]), row["label"], float(row["confidence"]),
                             row["notes"]) for row in reader]
    previous = {}
    for item in sorted(result, key=lambda a: (a.sensor, a.metric, a.start)):
        key = (item.sensor, item.metric)
        if key in previous and item.start < previous[key]:
            raise ValueError("overlapping labels for the same sensor/metric require adjudication")
        previous[key] = item.end
    return result


def labeled_scenarios(readings, annotations, sampling_interval_seconds=300, minimum_confidence=0.8):
    """Score only explicit healthy/pollution/fault intervals; exclude unknown,
    maintenance, unlabeled time, and low-confidence labels from the denominator.
    Replay continuously across labels to preserve baselines. Unlabeled/maintenance
    time is ingested as context and excluded from scoring, never assumed healthy.
    """
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must lie in [0,1]")
    readings = tuple(readings)
    result = []
    identities = sorted({(a.sensor, a.metric) for a in annotations})
    for index, (sensor, metric) in enumerate(identities):
        labels = sorted((a for a in annotations if (a.sensor, a.metric) == (sensor, metric)
                         and a.confidence >= minimum_confidence and a.label not in {"unknown", "maintenance"}),
                        key=lambda a: a.start)
        if not labels:
            continue
        start, end = min(a.start for a in labels), max(a.end for a in labels)
        selected = tuple(Reading(r.timestamp, r.sensor, {metric: r.values[metric]}, r.restart,
                                 r.expected_rejection) for r in readings if r.sensor == sensor
                         and metric in r.values and start <= r.timestamp < end)
        faults = tuple(Fault(f"label-{index}-{i}", sensor, metric, LABEL_CATEGORIES[a.label], a.start, a.end)
                       for i, a in enumerate(labels) if a.label in LABEL_CATEGORIES)
        result.append(Scenario(f"annotation-{index}", selected, faults, start, end,
                               ((sensor, metric),), sampling_interval_seconds=sampling_interval_seconds,
                               parameters={"scoring_intervals": [[a.start, a.end] for a in labels]}))
    return result


def chronological_split(scenario, calibration_end, development_end):
    """Explicit time boundaries; reject faults straddling boundaries rather than
    counting one physical fault as independent examples in multiple partitions.
    """
    calibration_end, development_end = utc_seconds(calibration_end), utc_seconds(development_end)
    if not scenario.start < calibration_end < development_end < scenario.end:
        raise ValueError("split boundaries must lie in chronological order inside observation")
    if any(f.start < boundary < f.end for f in scenario.faults
           for boundary in (calibration_end, development_end)):
        raise ValueError("a fault crosses a split boundary")
    result = {}
    bounds = (scenario.start, calibration_end, development_end, scenario.end)
    for name, start, end in zip(("calibration", "development", "holdout"), bounds, bounds[1:]):
        result[name] = Scenario(f"{scenario.name}-{name}", tuple(r for r in scenario.readings if start <= r.timestamp < end),
            tuple(f for f in scenario.faults if start <= f.start and f.end <= end), start, end,
            scenario.identities, scenario.seed, scenario.sampling_interval_seconds, scenario.description,
            {**scenario.parameters, "partition": name})
    return result
