import numpy as np
import pytest

from safe.annotations import read_annotations
from safe.cli import main
from safe.config import PC_METRICS, PM_METRICS
from safe.synthetic_pm import (DIRECTORIES, FAMILIES, MASS_PER_COUNT, NODES, dataset_files, derive_pm,
                               file_content, generate, loader_check, sensor_name)
from tests.test_loader import write_influx_csv


@pytest.fixture(scope="module")
def dataset():
    return generate()


@pytest.mark.parametrize("family", FAMILIES)
def test_committed_dataset_matches_generator(dataset, family):
    for name, data in dataset_files(dataset, family).items():
        committed = (DIRECTORIES[family] / name).read_bytes()
        assert file_content(name, committed) == file_content(name, data), name


@pytest.mark.parametrize("family", FAMILIES)
def test_one_file_per_bin_and_valid_labels(family):
    directory = DIRECTORIES[family]
    assert sorted(p.name for p in directory.glob("*.csv.gz")) == sorted(f"{m}.csv.gz" for m in FAMILIES[family])
    labels = read_annotations(directory / "labels.csv")
    context = read_annotations(directory / "context.csv")
    names = {sensor_name(n) for n in NODES}
    assert {a.sensor for a in labels} < names  # node01 and node06 are healthy controls
    assert {a.metric for a in labels} <= set(FAMILIES[family])
    expected = {"invalid", "freeze", "offline", "spike", "offset", "drift", "noise", "sensitivity", "restart", "clock"}
    assert {a.label for a in labels} >= (expected | {"ordering"} if family == "pm" else expected)
    assert {a.label for a in context} == {"pollution_event", "healthy"}


def test_healthy_pm_is_derived_from_counts_and_ordered(dataset):
    counts, pm = dataset.values["pc"]["node01"], dataset.values["pm"]["node01"]
    assert np.allclose(pm, derive_pm(counts), rtol=1e-4, atol=1e-4)
    assert (np.diff(pm, axis=1) >= 0).all()
    assert (pm > 0).all() and not np.isnan(pm).any()
    assert (counts[:, PC_METRICS.index("pc10_0")] == 0).mean() > 0.95  # large bins read exact zeros
    assert 0 < (counts[:, PC_METRICS.index("pc5_0")] == 0).mean() < 0.3
    assert counts.max() < 1_000_000  # healthy counts stay near the field maximum
    assert np.median(pm[:, PM_METRICS.index("pm2_5")]) == pytest.approx(2.9, rel=0.2)  # field median


def test_physical_faults_reach_pm_and_output_faults_do_not(dataset):
    t = dataset.times
    labels = {f: {(a.node, a.metric, a.notes.split(":")[0]) for a in dataset.labels[f]} for f in FAMILIES}
    assert ("node04", "pc0_3", "D3") in labels["pc"] and ("node04", "pm0_3", "D3") in labels["pm"]
    assert ("node03", "pm1_0", "S2") in labels["pm"] and not any(k[2] == "S2" for k in labels["pc"])
    assert ("node03", "pc0_5", "P8") in labels["pc"] and not any(k[2] == "P8" for k in labels["pm"])
    faulty = dataset.values["pc"]["node02"]
    assert np.isnan(faulty).any() and (np.nan_to_num(faulty) < 0).any() and (faulty > 4e9).any()
    assert (dataset.values["pm"]["node02"] > 10000).any()
    assert len(t) == len(faulty) and len(MASS_PER_COUNT) == len(PC_METRICS)


@pytest.mark.parametrize("family", FAMILIES)
def test_loader_removes_injected_duplicates_and_restores_order(family):
    check = loader_check(DIRECTORIES[family], family)
    assert check["removed_rows"] == 48 * len(FAMILIES[family])
    assert check["time_ordered"]


def test_per_field_exports_need_merge_to_replay_together(tmp_path):
    a = write_influx_csv(tmp_path / "pm1_0.csv", [(f"2026-01-01T00:0{i}:00Z", 5, "pm1_0", "M", "d") for i in range(3)])
    b = write_influx_csv(tmp_path / "pm2_5.csv", [(f"2026-01-01T00:0{i}:00Z", 8, "pm2_5", "M", "d") for i in range(3)])
    assert main(["health", a, b, "-o", str(tmp_path / "separate")]) == 1  # second file overlaps the first
    assert main(["health", a, b, "--merge", "-o", str(tmp_path / "merged")]) == 0
