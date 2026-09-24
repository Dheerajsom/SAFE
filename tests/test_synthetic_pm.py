import numpy as np
import pytest

from safe.annotations import read_annotations
from safe.cli import main
from safe.config import PM_METRICS
from safe.synthetic_pm import (DEFAULT_DIRECTORY, NODES, dataset_files, generate, loader_check,
                               sensor_name)
from tests.test_loader import write_influx_csv


@pytest.fixture(scope="module")
def dataset():
    return generate()


def test_committed_dataset_matches_generator(dataset):
    for name, data in dataset_files(dataset).items():
        assert (DEFAULT_DIRECTORY / name).read_bytes() == data, name


def test_one_file_per_pm_bin_and_valid_labels():
    assert sorted(p.name for p in DEFAULT_DIRECTORY.glob("*.csv.gz")) == sorted(f"{m}.csv.gz" for m in PM_METRICS)
    labels = read_annotations(DEFAULT_DIRECTORY / "labels.csv")
    context = read_annotations(DEFAULT_DIRECTORY / "context.csv")
    names = {sensor_name(n) for n in NODES}
    assert {a.sensor for a in labels} < names  # node01 and node06 are healthy controls
    assert {a.label for a in labels} >= {"invalid", "freeze", "offline", "spike", "offset", "drift",
                                         "noise", "sensitivity", "ordering", "restart", "clock"}
    assert {a.label for a in context} == {"pollution_event", "healthy"}


def test_healthy_bins_are_ordered_and_faults_change_the_data(dataset):
    healthy = dataset.values["node01"]
    assert (np.diff(healthy, axis=1) >= 0).all()
    assert (healthy >= 0).all() and not np.isnan(healthy).any()
    assert (healthy[:, 0] == 0).any()  # clean-air zeros are present
    faulty = dataset.values["node02"]
    assert np.isnan(faulty).any() and (np.nan_to_num(faulty) < 0).any() and (faulty > 10000).any()


def test_loader_removes_injected_duplicates_and_restores_order():
    check = loader_check()
    assert check["removed_rows"] == 48 * len(PM_METRICS)
    assert check["time_ordered"]


def test_per_field_exports_need_merge_to_replay_together(tmp_path):
    a = write_influx_csv(tmp_path / "pm1_0.csv", [(f"2026-01-01T00:0{i}:00Z", 5, "pm1_0", "M", "d") for i in range(3)])
    b = write_influx_csv(tmp_path / "pm2_5.csv", [(f"2026-01-01T00:0{i}:00Z", 8, "pm2_5", "M", "d") for i in range(3)])
    assert main(["health", a, b, "-o", str(tmp_path / "separate")]) == 1  # second file overlaps the first
    assert main(["health", a, b, "--merge", "-o", str(tmp_path / "merged")]) == 0
