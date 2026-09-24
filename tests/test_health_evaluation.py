from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from safe.annotations import Annotation, chronological_split, labeled_scenarios, read_annotations
from safe.health_evaluation import acceptance, evaluate_health, replay_health, score_incidents
from safe.scenarios import Fault, Reading, Scenario, synthetic_scenarios


@pytest.fixture(scope="module")
def health_report():
    return evaluate_health(synthetic_scenarios(1729))


def test_operating_targets_on_calibration(health_report):
    assert all(acceptance(health_report).values())
    for row in health_report["by_scenario"]:
        assert row["detected_faults"] + row["missed_faults"] == row["expected_faults"]
        assert row["max_notifications_per_incident"] <= 1
    rows = {r["name"]: r for r in health_report["by_scenario"]}
    for name in ("healthy_iid", "healthy_ar1", "healthy_diurnal", "healthy_slow_movement",
                 "legitimate_pm_episode", "healthy_slow_movement_with_reference"):
        assert rows[name]["false_actionable_incidents"] == 0
    # Keep the known ambiguity visible instead of broadening attribution to pass.
    assert rows["calibration_drift"]["missed_faults"] == 1
    # A reference resolves it: the drift is attributed to the reference-drift check.
    drift = rows["calibration_drift_with_reference"]
    assert drift["actionable_recall"] == 1
    assert drift["median_actionable_detection_delay_seconds"] <= 24 * 3600
    evidence = [e["evidence"].get("gradual_degradation", {}) for e in drift["events"]]
    assert any(ev.get("evidence_source") == "reference" for ev in evidence)
    assert rows["abrupt_offset"]["notifications"] == 1
    assert rows["startup_freeze"]["actionable_recall"] == 1
    assert rows["missing_data"]["actionable_recall"] == 1
    assert rows["increased_noise"]["actionable_recall"] == 1


def test_restart_uses_saved_state_and_never_renotifies():
    scenario = next(s for s in synthetic_scenarios() if s.name == "restart_during_fault")
    events, _, notifications, _ = replay_health(scenario)
    change = [e for e in events if e["family"] == "change"]
    assert len(change) == 1 and len(notifications) == 1
    assert change[0]["detection_count"] >= 1200


def test_incident_attribution_no_future_category_credit_or_early_tolerance():
    scenario = Scenario("s", (), (Fault("f", "s", "m", "level_offset", 10, 20),),
                        0, 100, (("s", "m"),), sampling_interval_seconds=1)
    events = [{"id": "e", "severity": "warning", "notification_count": 1}]
    observations = [dict(id="e", sensor="s", metric="m", timestamp=9, categories=["abrupt_shift"], severity="warning"),
                    dict(id="e", sensor="s", metric="m", timestamp=15, categories=["sensor_freeze"], severity="warning")]
    result = score_incidents(scenario, events, observations, [])
    assert result["missed_faults"] == 1
    assert result["false_actionable_incidents"] == 1
    observations.append(dict(id="e", sensor="s", metric="m", timestamp=16, categories=["abrupt_shift"], severity="warning"))
    result = score_incidents(scenario, events, observations, [])
    assert result["detected_faults"] == 1
    assert result["faults"][0]["actionable_detection_delay_seconds"] == 6


def test_one_incident_cannot_detect_two_separate_faults():
    scenario = Scenario("s", (), (Fault("f", "s", "m", "level_offset", 10, 20),
                                  Fault("g", "s", "m", "level_offset", 30, 40)),
                        0, 100, (("s", "m"),), sampling_interval_seconds=1)
    events = [{"id": "e", "severity": "warning", "notification_count": 1}]
    observations = [dict(id="e", sensor="s", metric="m", timestamp=t, categories=["abrupt_shift"], severity="warning") for t in (10, 30)]
    result = score_incidents(scenario, events, observations, [])
    assert result["detected_faults"] == 1 and result["missed_faults"] == 1


def test_annotation_roundtrip_unknown_exposure_and_timestamp_normalization(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("sensor,metric,start,end,label,confidence,notes\n"
                    "001,temperature,2025-01-01T00:00:00,2025-01-01T01:00:00,healthy,1,ok\n"
                    "001,temperature,2025-01-01T01:00:00Z,2025-01-01T02:00:00Z,unknown,0.2,unclear\n"
                    "001,temperature,2025-01-01T02:00:00Z,2025-01-01T03:00:00Z,freeze,0.9,fixed value\n")
    labels = read_annotations(path)
    assert labels[0].sensor == "001"
    scenarios = labeled_scenarios([], labels)
    assert len(scenarios) == 1
    assert len(scenarios[0].faults) == 1
    result = score_incidents(scenarios[0], [], [], [])
    assert result["sensor_days"] == pytest.approx(2 / 24)


def test_annotation_overlap_and_invalid_labels_fail(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("sensor,metric,start,end,label,confidence,notes\n"
                    "s,m,2025-01-01,2025-01-03,healthy,1,a\n"
                    "s,m,2025-01-02,2025-01-04,freeze,1,b\n")
    with pytest.raises(ValueError, match="overlapping"):
        read_annotations(path)
    with pytest.raises(ValueError):
        Annotation("s", "m", 0, 1, "guess", .5)
    with pytest.raises(ValueError):
        Annotation("s", "m", 0, 1, "freeze", float("nan"))


def test_time_split_is_disjoint_and_rejects_crossing_fault():
    scenario = Scenario("s", tuple(Reading(t, "s", {"m": t}) for t in range(30)), (),
                        0, 30, (("s", "m"),))
    splits = chronological_split(scenario, 10, 20)
    assert [len(s.readings) for s in splits.values()] == [10, 10, 10]
    assert splits["holdout"].readings[0].timestamp == 20
    with pytest.raises(ValueError, match="crosses"):
        chronological_split(replace(scenario, faults=(Fault("f", "s", "m", "freeze", 9, 11),)), 10, 20)


def write_csv(path, start=0):
    import pandas as pd
    rows = []
    for i in range(20):
        rows.append({"_time": pd.Timestamp(1735689600 + start + i * 300, unit="s", tz="UTC").isoformat(),
                     "_value": 150 if i < 2 else 20 + .1 * (i % 3), "_field": "temperature",
                     "_measurement": "test", "device_id": "001"})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_health_cli_writes_jsonl_and_restart_safe_state(tmp_path):
    path, continuation = tmp_path / "input.csv", tmp_path / "next.csv"
    write_csv(path)
    write_csv(continuation, 6000)
    state, output = tmp_path / "state.json", tmp_path / "output"
    proc = subprocess.run([sys.executable, "-m", "safe.cli", "health", str(path),
                           "--state-out", str(state), "--event-update-interval", "0", "-o", str(output)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert state.exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["total_incidents"] == 1
    assert summary["total_notifications"] == 1
    lines = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert {line["action"] for line in lines} >= {"opened", "updated", "closed"}
    proc = subprocess.run([sys.executable, "-m", "safe.cli", "health", str(continuation),
                           "--state-in", str(state), "--state-out", str(state), "-o", str(output)], capture_output=True)
    assert proc.returncode == 0
    proc = subprocess.run([sys.executable, "-m", "safe.cli", "health", str(path),
                           "--state-in", str(state), "-o", str(output)], capture_output=True)
    assert proc.returncode != 0


def test_health_cli_failures_are_nonzero(tmp_path):
    path = tmp_path / "input.csv"
    write_csv(path)
    config = tmp_path / "config.json"
    config.write_text('{"max_series": -1}')
    for extra in ([str(tmp_path / "missing.csv")], [str(path), "--config", str(config)],
                  [str(path), "--metric", "not_present"]):
        proc = subprocess.run([sys.executable, "-m", "safe.cli", "health", *extra,
                               "-o", str(tmp_path / "out")], capture_output=True)
        assert proc.returncode != 0


def test_wrong_category_alarm_inside_fault_is_misdiagnosed_not_false_or_detected():
    scenario = Scenario("s", (), (Fault("f", "s", "m", "freeze", 10, 20),),
                        0, 100, (("s", "m"),), sampling_interval_seconds=1)
    events = [{"id": e, "severity": "warning", "notification_count": 1} for e in ("inside", "before", "other")]
    observations = [
        # Reference-drift evidence during a freeze: right time, incompatible category.
        dict(id="inside", sensor="s", metric="m", timestamp=12, categories=["gradual_degradation"], severity="warning"),
        # First alarmed before the fault; later observations inside it do not excuse it.
        dict(id="before", sensor="s", metric="m", timestamp=5, categories=["gradual_degradation"], severity="warning"),
        dict(id="before", sensor="s", metric="m", timestamp=15, categories=["gradual_degradation"], severity="warning"),
        # Same time, different metric: no fault there, so it is false.
        dict(id="other", sensor="s", metric="n", timestamp=12, categories=["gradual_degradation"], severity="warning")]
    result = score_incidents(scenario, events, observations, [])
    assert result["misdiagnosed_ids"] == ["inside"]
    assert result["misdiagnosed_actionable_incidents"] == 1
    assert result["false_actionable_incidents"] == 2
    assert result["detected_faults"] == 0
