import numpy as np
import pandas as pd
import pytest

from safe.loader import load_pivoted_dataframe, replay_csv


def write_influx_csv(path, rows):
    """rows: list of (time, value, field, measurement, device_id)."""
    lines = ["#comment line that must be skipped",
             ",result,table,_time,_value,_field,_measurement,device_id"]
    for i, (t, v, f, m, d) in enumerate(rows):
        lines.append(f",_result,0,{t},{v},{f},{m},{d}")
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture
def sample_csv(tmp_path):
    rows = []
    for i in range(5):
        t = f"2026-01-01T00:{i:02d}:00Z"
        rows.append((t, 10.0 + i, "pm1_0", "IPS7100MHC001", "dev1"))
        rows.append((t, 21.5, "temperature", "IPS7100MHC001", "dev1"))
    return write_influx_csv(tmp_path / "sample.csv", rows)


class TestLoadPivotedDataframe:
    def test_pivot_shape_and_metrics(self, sample_csv):
        df, metric_cols = load_pivoted_dataframe(sample_csv)
        assert sorted(metric_cols) == ["pm1_0", "temperature"]
        assert len(df) == 5
        assert list(df["pm1_0"]) == [10.0, 11.0, 12.0, 13.0, 14.0]

    def test_datetime_index_utc_naive(self, sample_csv):
        df, _ = load_pivoted_dataframe(sample_csv)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is None
        assert df.index[0] == pd.Timestamp("2026-01-01 00:00:00")

    def test_sensor_display_name_mapping(self, sample_csv):
        df, _ = load_pivoted_dataframe(sample_csv)
        assert (df["_sensor_name"] == "IPS7100_MHC_001_dev1").all()

    def test_unknown_measurement_fallback_name(self, tmp_path):
        csv = write_influx_csv(tmp_path / "u.csv",
                               [("2026-01-01T00:00:00Z", 1.0, "pm1_0", "OTHER", "devX")])
        df, _ = load_pivoted_dataframe(csv)
        assert df["_sensor_name"].iloc[0] == "OTHER_devX"

    def test_duplicate_rows_keep_first(self, tmp_path):
        t = "2026-01-01T00:00:00Z"
        csv = write_influx_csv(tmp_path / "d.csv", [
            (t, 1.0, "pm1_0", "M", "d"),
            (t, 999.0, "pm1_0", "M", "d"),  # duplicate key, dropped
        ])
        df, _ = load_pivoted_dataframe(csv)
        assert len(df) == 1
        assert df["pm1_0"].iloc[0] == 1.0

    def test_non_numeric_values_dropped(self, tmp_path):
        csv = write_influx_csv(tmp_path / "n.csv", [
            ("2026-01-01T00:00:00Z", "abc", "pm1_0", "M", "d"),
            ("2026-01-01T00:01:00Z", 2.0, "pm1_0", "M", "d"),
        ])
        df, _ = load_pivoted_dataframe(csv)
        assert len(df) == 1

    def test_missing_file(self):
        df, cols = load_pivoted_dataframe("does/not/exist.csv")
        assert df is None and cols is None

    def test_missing_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("a,b,c\n1,2,3\n")
        df, cols = load_pivoted_dataframe(str(p))
        assert df is None and cols is None


class TestReplayCsv:
    def test_replay_feeds_engine(self, tmp_path):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(80):
            t = f"2026-01-01T{i // 60:02d}:{i % 60:02d}:00Z"
            # one impossible value in the middle
            v = -50.0 if i == 60 else round(float(rng.normal(10, 0.5)), 3)
            rows.append((t, v, "pm1_0", "IPS7100MHC001", "dev1"))
        csv = write_influx_csv(tmp_path / "r.csv", rows)

        engine = replay_csv(csv)
        assert engine is not None
        assert "invalid_measurement" in [e["category"] for e in engine.events]

    def test_absent_fields_are_not_measurements(self, tmp_path):
        # Pivot NaNs for fields missing at a timestamp must not become invalid readings.
        rows = [("2026-01-01T00:00:00Z", 10.0, "pm1_0", "M", "d"),
                ("2026-01-01T00:05:00Z", 21.5, "temperature", "M", "d")]
        engine = replay_csv(write_influx_csv(tmp_path / "sparse.csv", rows))
        assert engine is not None
        assert not any(e["category"] == "invalid_measurement" for e in engine.events)

    def test_overlapping_replay_is_rejected(self, sample_csv):
        engine = replay_csv(sample_csv)
        assert replay_csv(sample_csv, engine=engine) is None

    def test_replay_missing_file_returns_none(self):
        assert replay_csv("does/not/exist.csv") is None
