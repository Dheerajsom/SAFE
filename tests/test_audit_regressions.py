"""Regression cases found during the SAFE 2.5 repository audit."""

import importlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from safe.cli import main
from safe.health import SensorHealth
from safe.loader import load_pivoted_dataframe, replay_csv
from safe.periods import bucketize, run_period_analysis
from safe.stats import sample_comparison
from tests.test_loader import write_influx_csv


def test_normalize_before_deduplicating_and_sorting(tmp_path):
    csv = write_influx_csv(tmp_path / "mixed.csv", [
        ("2026-01-01T01:00:00+01:00", 1, "x", "M", "001"),
        ("2026-01-01T00:00:00Z", 9, "x", "M", "001"),
        ("2025-12-31T23:59:59.500", 2, "x", "M", "001"),
        ("bad", 3, "x", "M", "001"),
        ("2026-01-01T00:02:00Z", "inf", "x", "M", "001"),
    ])
    df, _ = load_pivoted_dataframe(csv)
    assert df["x"].tolist() == [2, 1]
    assert df["_sensor_name"].tolist() == ["M_001", "M_001"]
    assert df["_unix_time"].diff().iloc[1] == 0.5
    assert df["_str_time"].iloc[1] == "2026-01-01 00:00:00"


@pytest.mark.parametrize("content", ["", "_time,_value,_field,_measurement,device_id\n"])
def test_empty_input_is_failure(tmp_path, content):
    path = tmp_path / "empty.csv"
    path.write_text(content)
    assert load_pivoted_dataframe(path) == (None, None)


def test_known_measurement_devices_do_not_share_state(tmp_path):
    csv = write_influx_csv(tmp_path / "devices.csv", [
        ("2026-01-01T00:00:00Z", 1, "x", "IPS7100MHC001", "001e064a1520"),
        ("2026-01-01T00:00:00Z", 8, "x", "IPS7100MHC001", "second"),
    ])
    engine = replay_csv(csv)
    assert engine._states[("IPS7100_MHC_001", "x")].last_value == 1
    assert engine._states[("IPS7100_MHC_001_second", "x")].last_value == 8


def test_partial_replay_reports_failure(sample_csv):
    class BrokenEngine:
        def data_processing(self, *args):
            raise ValueError("bad row")
    assert replay_csv(sample_csv, engine=BrokenEngine()) is None


@pytest.fixture
def sample_csv(tmp_path):
    return write_influx_csv(tmp_path / "readings.csv", [
        (f"2026-01-0{day}T0{hour}:00:00Z", day + hour, "pm1_0", "M", "d")
        for day in (1, 2) for hour in (0, 1, 2)
    ])


def test_nonfinite_timestamp_does_not_mutate_engine():
    engine = SensorHealth()
    with pytest.raises(ValueError):
        engine.data_processing("s", {"unix_timestamp": np.nan, "pm1_0": -1})
    assert not engine.events
    assert not engine._states


@pytest.mark.parametrize("values", [[], [1], [1, np.nan], [1, np.inf], [[1, 2], [3, 4]]])
def test_stats_reject_invalid_samples(values):
    with pytest.raises(ValueError):
        sample_comparison(values, [1, 2, 3])


def test_bucketize_filters_infinite_unknown_metrics():
    s = pd.Series([1., np.inf, 2.], index=pd.date_range("2026-01-01", periods=3, freq="h"))
    assert bucketize(s, "D", 2)[0][1].tolist() == [1., 2.]


def test_plot_failure_propagates_to_period_result(sample_csv, tmp_path, monkeypatch):
    import safe.plotting
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(safe.plotting, "run_all_plotting", fail)
    assert run_period_analysis(sample_csv, tmp_path / "out") is False


def test_plots_separate_sensors_and_forward_alpha(tmp_path, monkeypatch):
    import safe.plotting as plotting
    frame = pd.DataFrame({"sensor": ["A", "B"], "metric": ["pm1_0"] * 2,
                          "new_period": ["2026-01"] * 2})
    csv = tmp_path / "period_month_to_month.csv"
    frame.to_csv(csv, index=False)
    calls = []
    monkeypatch.setattr(plotting, "_sparse_metric_panels", lambda *args: None)
    monkeypatch.setattr(plotting, "_sparse_significance", lambda df, title, path, alpha: calls.append((df.sensor.tolist(), path, alpha)))
    plotting.generate_category_plots(csv, str(tmp_path / "plots"), p_alpha=0.05)
    assert [call[0] for call in calls] == [["A"], ["B"]]
    assert len({call[1] for call in calls}) == 2
    assert all(call[2] == 0.05 for call in calls)


@pytest.mark.parametrize("args", [["stream", "missing", "--event-update-interval", "nan"],
                                  ["periods", "missing", "-o", "unused", "--alpha", "nan"]])
def test_cli_bad_configuration_returns_failure(args, caplog):
    assert main(args) == 1
    assert "must be" in caplog.text


def test_visualizer_import_is_offline_and_does_not_read_csv(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("CSV read on import"))
    import dataVisualizer
    importlib.reload(dataVisualizer)


def test_normality_period_windows_do_not_overlap():
    from standardNormalVisualizer import COMPARISONS, normal_test_rows
    for comparison in COMPARISONS:
        first, second = comparison["samples"]
        assert pd.Timestamp(first[2]) < pd.Timestamp(second[1])
    assert normal_test_rows("x", "period", "sample", np.arange(30.))[0]["p_value"] is None


def test_window_selection_excludes_left_endpoint():
    from mintsXU4.mintsWindowPdfAnimation import window_values
    s = pd.Series(range(4), index=pd.date_range("2026-01-01", periods=4, freq="30min"))
    assert window_values(s, s.index[-1]).tolist() == [2, 3]


def test_offline_import_does_not_load_live_modules():
    code = "import safe; import sys; assert not any(x in sys.modules for x in ['serial', 'paho.mqtt.client', 'mintsXU4.mintsDefinitions', 'mintsXU4.mintsLatest'])"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_daily_loader_annotated_csv_overlap_and_nonfinite(tmp_path, monkeypatch):
    import gzip
    from mintsXU4 import mints1sLoader as loader
    header = "#annotation\n_time,_value,_field\n"
    for day, rows in enumerate([
        "2026-01-01T00:00:00Z,1,pm1_0\n2026-01-01T00:01:00Z,inf,pm1_0\n",
        "2026-01-01T00:00:00Z,9,pm1_0\n2026-01-02T00:00:00Z,2,pm1_0\n",
    ]):
        with gzip.open(tmp_path / f"valo_node_01_{day}.csv.gz", "wt") as f:
            f.write(header + rows)
    monkeypatch.setattr(loader, "DATA_DIR", str(tmp_path))
    df = loader.load_wide(use_cache=False)
    assert df.index.is_unique
    assert df["pm1_0"].dropna().tolist() == [1, 2]
    assert df["pm1_0"].dtype == np.float32
    with pytest.raises(ValueError):
        loader.load_wide(max_files=0)


def test_downloader_rejects_zero_chunk_before_network(monkeypatch):
    import mintsInfluxDownloader as downloader
    monkeypatch.setattr(sys, "argv", ["download", "--chunk-days", "0"])
    with pytest.raises(SystemExit) as exc:
        downloader.main()
    assert exc.value.code == 2


def test_downloader_auto_start_uses_earliest_table():
    import mintsInfluxDownloader as downloader
    class Backend:
        def stream_lines(self, flux):
            yield "_time,_value,_field\n2026-02-01T00:00:00Z,1,x\n"
            yield "_time,_value,_field\n2026-01-01T00:00:00Z,1,y\n"
    start = downloader.detect_start(Backend(), "b", "n", "d", "m", ["x", "y"])
    assert start.isoformat() == "2026-01-01T00:00:00+00:00"


def test_downloader_cli_drains_large_stderr(monkeypatch):
    import mintsInfluxDownloader as downloader
    original = subprocess.Popen
    def child(command, **kwargs):
        return original([sys.executable, "-c",
                         "import sys; sys.stderr.write('x'*200000); print('row')"], **kwargs)
    monkeypatch.setattr(downloader.subprocess, "Popen", child)
    backend = downloader.Backend("cli", "unused", "org", "dummy", "influx", 1)
    assert list(backend.stream_lines("query")) == ["row\n"]
