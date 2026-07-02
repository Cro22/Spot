"""Hito 5 tests: the false-positive measurement (we measure, we do not solve)."""

from __future__ import annotations

import json

from spot import bench
from spot.__main__ import main
from spot.bench import BENCHMARK_SCHEMA_VERSION


def test_smooth_controls_stay_clean():
    entries = {e.name: e for e in bench.run_benchmark()}
    # Smooth surfaces must not false-positive: the local test treats uniform
    # curvature as normal.
    assert entries["sphere"].n_flagged == 0
    assert entries["torus"].n_flagged == 0


def test_features_do_get_false_flagged():
    # The doorway is real: isolated sharp features (cube corners, cone apex) are
    # flagged because they are geometrically indistinguishable from a defect.
    entries = {e.name: e for e in bench.run_benchmark()}
    assert entries["cube"].n_flagged > 0
    assert entries["cone"].n_flagged > 0
    # Uniform ridges are not: the cylinder rim is "normal for around here".
    assert entries["cylinder"].n_flagged == 0


def test_threshold_sweep_is_monotone():
    sweep = bench.threshold_sweep()
    totals = [sweep[t] for t in sorted(sweep)]
    # Raising the threshold can only shed false positives, never add them.
    assert all(a >= b for a, b in zip(totals, totals[1:]))


def test_benchmark_document_schema():
    entries = bench.run_benchmark()
    doc = bench.benchmark_document(entries, k=2, threshold=3.5)
    assert doc["benchmark_schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert "aggregate" in doc and "feature_false_positive_rate" in doc["aggregate"]
    assert len(doc["meshes"]) == len(entries)
    # Every listed feature vertex is a false positive labelled implicitly by kind.
    feat = [m for m in doc["meshes"] if m["kind"] == "feature"]
    assert sum(m["n_flagged"] for m in feat) == doc["aggregate"]["feature_false_positives"]


def test_cli_bench_writes_benchmark(tmp_path):
    rc = main(["bench", "--out-dir", str(tmp_path)])
    assert rc == 0
    bfile = tmp_path / "benchmark.json"
    assert bfile.exists()
    doc = json.loads(bfile.read_text(encoding="utf-8"))
    assert doc["benchmark_schema_version"] == BENCHMARK_SCHEMA_VERSION
    # Per-mesh artifacts exist and reports are valid.
    assert (tmp_path / "cube.obj").exists()
    assert (tmp_path / "cube.report.json").exists()
    rep = json.loads((tmp_path / "cube.report.json").read_text(encoding="utf-8"))
    assert rep["schema_version"] == "1.0.0"
