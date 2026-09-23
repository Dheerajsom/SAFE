import pytest

from safe.evaluation import evaluate
from safe.scenarios import synthetic_scenarios


@pytest.fixture(scope="module")
def baseline():
    return evaluate(synthetic_scenarios(), seed=1729)


def test_generators_are_reproducible_and_cover_required_scenarios():
    first = synthetic_scenarios()
    assert first == synthetic_scenarios()
    assert first != synthetic_scenarios(1730)
    assert len(first) == 18
    assert len({s.name for s in first}) == 18
    assert all(s.end - s.start == 7 * 86400 for s in first)


def test_healthy_false_alert_baseline_is_visible(baseline):
    # Fixed fixture operating ranges, not claims about arbitrary random streams.
    rows = baseline["by_scenario"]
    assert 1 <= rows["healthy_iid"]["false_positive_alerts"] <= 5
    assert 10 <= rows["healthy_ar1"]["false_positive_alerts"] <= 25
    assert rows["healthy_diurnal"]["false_positive_alerts"] == 0
    assert rows["healthy_slow_movement"]["false_alerts_per_sensor_day"] > 2
    assert rows["legitimate_pm_episode"]["false_positive_alerts"] >= 10
    for name in ("healthy_iid", "healthy_ar1", "healthy_diurnal"):
        assert rows[name]["expected_faults"] == 0
        assert rows[name]["matched_alerts"] == 0


def test_known_faults_detected_and_unsupported_modes_missed(baseline):
    rows = baseline["by_scenario"]
    for name in ("impossible_reading", "abrupt_offset"):
        assert rows[name]["true_positive_events"] == 1
        assert rows[name]["mean_detection_delay_seconds"] == 0
    abrupt = next(s for s in baseline["scenarios"] if s["name"] == "abrupt_offset")
    assert any(a["alert_type"] == "Step-Change Detected" and a["fault_id"] for a in abrupt["alerts"])
    for name in ("missing_data", "startup_freeze", "decreased_sensitivity", "out_of_order"):
        assert rows[name]["false_negative_events"] == 1
        assert rows[name]["true_positive_events"] == 0
    missing = next(s for s in baseline["scenarios"] if s["name"] == "missing_data")
    assert missing["reading_count"] == 2016 - 288
    assert missing["metrics"]["sensor_days"] == 7
    disorder = next(s for s in baseline["scenarios"] if s["name"] == "out_of_order")
    assert len(disorder["ingestion_errors"]) == 1


def test_all_alerts_and_events_accounted_for(baseline):
    for s in baseline["scenarios"]:
        m = s["metrics"]
        assert m["matched_alerts"] + m["false_positive_alerts"] == len(s["alerts"])
        assert m["true_positive_events"] + m["false_negative_events"] == len(s["faults"])
        assert sum(e["alert_count"] for e in s["faults"]) == m["matched_alerts"]
    assert sum(r["reading_level_alerts"] for r in baseline["by_alert_type"].values()) == baseline["overall"]["reading_level_alerts"]
