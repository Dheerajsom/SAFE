"""Incident-unit scoring for SensorHealth, alongside the preserved v1 baseline."""

from dataclasses import asdict
import json
from pathlib import Path
import platform

import numpy as np

from safe.evaluation import evaluate, validate_scenario
from safe.health import SensorHealth, ENGINE_VERSION
from safe.scenarios import synthetic_scenarios


HEALTH_CATEGORIES = {
    "invalid_measurement": {"physical_bounds"},
    "isolated_anomaly": {"spike", "level_offset", "calibration_drift", "noise_increase"},
    "abrupt_shift": {"level_offset", "calibration_drift"},
    "gradual_degradation": {"level_offset", "calibration_drift"},
    "noise_change": {"noise_increase", "sensitivity_loss"},
    "sensor_freeze": {"freeze"},
    "missing_data": {"missing_data"},
    "completeness_loss": {"missing_data"},
    "cadence_degradation": {"missing_data"},
    "timestamp_disorder": {"timestamp_order"},
    "duplicate_timestamp": {"timestamp_order"},
}


def replay_health(scenario, configuration=None):
    validate_scenario(scenario)
    observations, notifications, final = [], [], {}

    def collect(action, event):
        final[event["id"]] = event
        if action in {"opened", "updated", "escalated"}:
            # Use the detector evidence from this observation, never a future
            # upgraded incident category to reclassify earlier detections.
            current = {k: v for k, v in event["evidence"].items()
                       if v["observed_at"] == event["last_seen_at"]}
            observations.append(dict(id=event["id"], sensor=event["sensor"], metric=event["metric"],
                                     timestamp=event["last_seen_at"], categories=sorted(current),
                                     severity=event["severity"]))

    engine = SensorHealth(**(configuration or {}), on_event=collect,
                          on_notification=notifications.append)
    for sensor, metric in scenario.identities:
        engine.register(sensor, metric, scenario.start)
    interval = scenario.sampling_interval_seconds or 300
    clock = scenario.start
    for reading in scenario.readings:
        # Explicit timers run over absent readings. Never synthesize measurements.
        while clock < reading.timestamp:
            engine.tick(clock)
            clock += interval
        if reading.restart:
            engine = SensorHealth.from_snapshot(engine.snapshot(), on_event=collect,
                                                on_notification=notifications.append)
        engine.data_processing(reading.sensor, {**reading.values, "unix_timestamp": reading.timestamp},
                               references=reading.references)
    while clock < scenario.end:
        engine.tick(clock)
        clock += interval
    return list(final.values()), observations, notifications, engine.configuration()


def score_incidents(scenario, events, observations, notifications):
    """One incident matches at most one fault; one fault may have several incidents.

    Report diagnostic recall and actionable recall separately. False actionable
    incidents include every unmatched warning/critical incident, even if rate
    limiting suppressed its notification. No post-fault grace hides late alarms.
    """
    faults = [{**asdict(f), "incident_ids": [], "detection_delay_seconds": None,
               "actionable_detection_delay_seconds": None} for f in sorted(scenario.faults, key=lambda f: (f.start, f.id))]
    attribution = {}
    for observation in observations:
        if observation["id"] in attribution:
            candidate = next(f for f in faults if f["id"] == attribution[observation["id"]])
            candidates = [candidate]
        else:
            candidates = faults
        eligible = set().union(*(HEALTH_CATEGORIES.get(c, set()) for c in observation["categories"]))
        matched = next((f for f in candidates if f["sensor"] == observation["sensor"]
                        and f["metric"] == observation["metric"] and f["category"] in eligible
                        and f["start"] <= observation["timestamp"] < f["end"]), None)
        if matched is None:
            continue
        identity = observation["id"]
        attribution[identity] = matched["id"]
        if identity not in matched["incident_ids"]:
            matched["incident_ids"].append(identity)
        delay = observation["timestamp"] - matched["start"]
        if matched["detection_delay_seconds"] is None:
            matched["detection_delay_seconds"] = delay
        if observation["severity"] != "info" and matched["actionable_detection_delay_seconds"] is None:
            matched["actionable_detection_delay_seconds"] = delay
    intervals = scenario.parameters.get("scoring_intervals", [(scenario.start, scenario.end)])
    previous_end = scenario.start
    for start, end in intervals:
        if not scenario.start <= previous_end <= start < end <= scenario.end:
            raise ValueError("scoring intervals must be ordered, disjoint, and inside observation")
        previous_end = end
    if not intervals:
        raise ValueError("at least one scoring interval is required")
    scored_ids = {o["id"] for o in observations if o["severity"] != "info"
                  and any(start <= o["timestamp"] < end for start, end in intervals)}
    actionable = [e for e in events if e["id"] in scored_ids]
    false = [e for e in actionable if e["id"] not in attribution]
    delays = [f["detection_delay_seconds"] for f in faults if f["incident_ids"]]
    actionable_delays = [f["actionable_detection_delay_seconds"] for f in faults
                         if f["actionable_detection_delay_seconds"] is not None]
    exposure = sum(end - start for start, end in intervals) / 86400 * len({s for s, _ in scenario.identities})
    matched_actionable = sum(e["id"] in attribution for e in actionable)
    return dict(name=scenario.name, sensor_days=exposure, expected_faults=len(faults),
                detected_faults=len(delays), missed_faults=len(faults) - len(delays),
                recall=len(delays) / len(faults) if faults else None,
                actionable_recall=len(actionable_delays) / len(faults) if faults else None,
                incidents=len(events), actionable_incidents=len(actionable),
                false_actionable_incidents=len(false), false_actionable_incidents_per_sensor_day=len(false) / exposure,
                incident_precision=matched_actionable / len(actionable) if actionable else None,
                median_detection_delay_seconds=float(np.median(delays)) if delays else None,
                median_actionable_detection_delay_seconds=float(np.median(actionable_delays)) if actionable_delays else None,
                median_actionable_delay_readings=float(np.median(actionable_delays)) / scenario.sampling_interval_seconds
                    if actionable_delays and scenario.sampling_interval_seconds else None,
                observations=len(observations), observations_per_incident=len(observations) / len(events) if events else None,
                notifications=len(notifications),
                max_notifications_per_incident=max((e["notification_count"] for e in events), default=0),
                scored_actionable_ids=sorted(scored_ids),
                faults=faults, events=events)


def evaluate_health(scenarios, configuration=None):
    scenarios = list(scenarios)
    if not scenarios or len({s.name for s in scenarios}) != len(scenarios):
        raise ValueError("provide scenarios with unique names")
    rows = []
    for scenario in scenarios:
        events, observations, notifications, resolved = replay_health(scenario, configuration)
        rows.append(score_incidents(scenario, events, observations, notifications))
    expected = sum(r["expected_faults"] for r in rows)
    detected = sum(r["detected_faults"] for r in rows)
    false = sum(r["false_actionable_incidents"] for r in rows)
    days = sum(r["sensor_days"] for r in rows)
    by_category = {}
    for category in sorted({f["category"] for r in rows for f in r["faults"]}):
        faults = [f for r in rows for f in r["faults"] if f["category"] == category]
        delays = [f["detection_delay_seconds"] for f in faults if f["detection_delay_seconds"] is not None]
        detected_count = sum(bool(f["incident_ids"]) for f in faults)
        by_category[category] = dict(expected=len(faults), detected=detected_count, recall=detected_count / len(faults),
            actionable_detected=sum(f["actionable_detection_delay_seconds"] is not None for f in faults),
            median_detection_delay_seconds=float(np.median(delays)) if delays else None)
    by_metric = {}
    for metric in sorted({m for s in scenarios for _, m in s.identities}):
        metric_faults = [f for r in rows for f in r["faults"] if f["metric"] == metric]
        metric_events = [(r["name"], e) for r in rows for e in r["events"]
                         if e["metric"] == metric and e["id"] in r["scored_actionable_ids"]]
        matched = {(r["name"], identity) for r in rows for f in r["faults"]
                   if f["metric"] == metric for identity in f["incident_ids"]}
        metric_fp = sum((name, e["id"]) not in matched for name, e in metric_events)
        metric_exposure = sum(sum(end - start for start, end in s.parameters.get("scoring_intervals", [(s.start, s.end)]))
                              / 86400 * len({ss for ss, mm in s.identities if mm == metric}) for s in scenarios)
        by_metric[metric] = dict(expected=len(metric_faults), detected=sum(bool(f["incident_ids"]) for f in metric_faults),
                                actionable_incidents=len(metric_events), false_actionable_incidents=metric_fp,
                                false_actionable_incidents_per_sensor_day=metric_fp / metric_exposure,
                                sensor_days=metric_exposure)
    return dict(schema_version=2, engine_version=ENGINE_VERSION, python=platform.python_version(),
                configuration=resolved, by_scenario=rows, by_fault_category=by_category, by_metric=by_metric,
                overall=dict(expected_faults=expected, detected_faults=detected, missed_faults=expected - detected,
                             recall=detected / expected if expected else None,
                             false_actionable_incidents=false, sensor_days=days,
                             false_actionable_incidents_per_sensor_day=false / days,
                             notifications=sum(r["notifications"] for r in rows)))


def acceptance(report):
    categories = report["by_fault_category"]

    def recall(category, actionable=False):
        group = categories.get(category)
        if group is None or not group["expected"]:
            return False
        return group["actionable_detected" if actionable else "detected"] / group["expected"]

    rows = {r["name"]: r for r in report["by_scenario"]}
    delays = [r["median_actionable_delay_readings"] for r in report["by_scenario"]
              if any(f["category"] == "level_offset" for f in r["faults"])
              and r["median_actionable_delay_readings"] is not None]
    return {"hard_bound_recall_100_percent": recall("physical_bounds") == 1,
            "missing_recall_at_least_95_percent": recall("missing_data") >= .95,
            "freeze_recall_at_least_95_percent": recall("freeze") >= .95,
            "abrupt_actionable_recall_at_least_90_percent": recall("level_offset", True) >= .9,
            "false_actionable_incidents_below_0_1_per_sensor_day": report["overall"]["false_actionable_incidents_per_sensor_day"] < .1,
            "abrupt_median_delay_at_most_15_readings": bool(delays) and float(np.median(delays)) <= 15,
            "reference_calibration_drift_detected":
                rows.get("calibration_drift_with_reference", {}).get("actionable_recall") == 1,
            "no_reference_drift_on_healthy_trend":
                rows.get("healthy_slow_movement_with_reference", {}).get("false_actionable_incidents") == 0,
            "one_notification_per_incident": all(r["max_notifications_per_incident"] <= 1 for r in report["by_scenario"]),
            "no_page_hinkley_notifications_on_diurnal_data": all(
                not any(v.get("stationary_residuals") for v in e["evidence"].values())
                for r in report["by_scenario"] if r["name"] == "healthy_diurnal" for e in r["events"])}


def write_comparison(output, seeds=(1729, 2718, 31415), configuration=None):
    """Disjoint predeclared seeds: calibration, development, untouched validation.

    This runner never adjusts a parameter against any of these results. The seed
    split is a synthetic check, not an independent field deployment holdout.
    """
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("supply three distinct split seeds")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    report = {"splits": {}, "tuning_policy": "fixed configuration; no automated fitting to evaluation labels"}
    lines = ["# SAFE health evaluation", "", "Synthetic evidence only; field validation pending.", "",
             "Legacy FP counts raw alerts; health FP counts actionable incidents. These are different units.", "",
             "| Split | Legacy alerts | Legacy unmatched alerts | Health incidents | False actionable incidents | Detected faults | Notifications |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for split, seed in zip(("calibration", "development", "holdout"), seeds):
        scenarios = synthetic_scenarios(seed)
        baseline = evaluate(scenarios, seed=seed)
        health = evaluate_health(scenarios, configuration)
        report["splits"][split] = dict(seed=seed, baseline=baseline, health=health, acceptance=acceptance(health))
        b, h = baseline["overall"], health["overall"]
        lines.append(f"| {split} ({seed}) | {b['reading_level_alerts']} | {b['false_positive_alerts']} | "
                     f"{sum(r['incidents'] for r in health['by_scenario'])} | {h['false_actionable_incidents']} | "
                     f"{h['detected_faults']}/{h['expected_faults']} | {h['notifications']} |")
    lines += ["", "Holdout targets:", ""]
    lines += [f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in report["splits"]["holdout"]["acceptance"].items()]
    lines += ["", "Per-fault misses and delays, metric/category results, configurations, engine versions,",
              "and event evidence are in comparison.json. Diagnostic recall includes informational anomalies;",
              "actionable recall separately requires warning/critical evidence during the labeled interval.",
              "A single-sensor ambient trend cannot establish calibration failure; PM episodes without peers remain uncertain."]
    (root / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
