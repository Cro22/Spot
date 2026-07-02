"""Hito 4 tests: the `fix` CLI end to end, and the report schema."""

from __future__ import annotations

import json

import numpy as np
import trimesh

from spot import mesh as mesh_io
from spot import synthetic
from spot.__main__ import main
from spot.report import SCHEMA_VERSION


def _sphere_with_defect_obj(path, magnitude_scale=1.0):
    g = mesh_io.from_trimesh(trimesh.creation.icosphere(subdivisions=4, radius=1.0))
    edge = synthetic.mean_edge_length(g)
    d = synthetic.inject_defect(g, vertex_id=100, magnitude=magnitude_scale * edge)
    mesh_io.save(path, d.graph.vertices, d.graph.faces)
    return d


def test_fix_end_to_end_obj(tmp_path):
    obj = tmp_path / "defect.obj"
    out = tmp_path / "fixed.obj"
    rep = tmp_path / "report.json"
    d = _sphere_with_defect_obj(obj)

    rc = main(["fix", str(obj), "--out", str(out), "--report", str(rep)])
    assert rc == 0
    assert out.exists() and rep.exists()

    # The written mesh loads back and has the same topology.
    fixed = mesh_io.load(out)
    assert fixed.n_faces == d.graph.n_faces
    assert fixed.n_vertices == d.graph.n_vertices

    # The defective vertex is closer to its true position after the fix.
    err_before = np.linalg.norm(d.graph.vertices[d.vertex_id] - d.original_position)
    err_after = np.linalg.norm(fixed.vertices[d.vertex_id] - d.original_position)
    assert err_after < err_before


def test_report_schema(tmp_path):
    obj = tmp_path / "defect.obj"
    rep = tmp_path / "report.json"
    d = _sphere_with_defect_obj(obj)

    main(["fix", str(obj), "--report", str(rep)])
    report = json.loads(rep.read_text(encoding="utf-8"))

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["tool"]["name"] == "spot"
    assert report["source"]["n_vertices"] == d.graph.n_vertices
    assert report["summary"]["n_flagged"] >= 1
    # Non-flagged vertices never move.
    assert report["summary"]["max_nonflagged_displacement"] == 0.0

    ids = [f["vertex_id"] for f in report["flagged"]]
    assert d.vertex_id in ids

    rec = next(f for f in report["flagged"] if f["vertex_id"] == d.vertex_id)
    for key in ("original_position", "corrected_position", "displacement",
                "displacement_magnitude", "local_z", "signals", "signal_z"):
        assert key in rec
    assert set(rec["signals"]) == {"curvature", "displacement", "normal"}
    assert rec["displacement_magnitude"] > 0
    # displacement == corrected - original, componentwise.
    disp = np.array(rec["corrected_position"]) - np.array(rec["original_position"])
    assert np.allclose(disp, rec["displacement"], atol=1e-9)


def test_fix_clean_mesh_flags_nothing(tmp_path):
    obj = tmp_path / "clean.obj"
    rep = tmp_path / "report.json"
    g = mesh_io.from_trimesh(trimesh.creation.icosphere(subdivisions=4, radius=1.0))
    mesh_io.save(obj, g.vertices, g.faces)

    main(["fix", str(obj), "--report", str(rep)])
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["summary"]["n_flagged"] == 0
    assert report["flagged"] == []


def test_fix_handmade_obj(tmp_path):
    """A tiny OBJ written by hand: a flat grid with one vertex spiked upward."""
    # 5x5 grid of vertices in the z=0 plane, spacing 1.0.
    lines = []
    n = 5
    idx = {}
    vid = 1
    for j in range(n):
        for i in range(n):
            z = 0.0
            lines.append(f"v {i}.0 {j}.0 {z}")
            idx[(i, j)] = vid
            vid += 1
    # Spike the center vertex up out of the plane.
    center = idx[(2, 2)]
    lines[center - 1] = "v 2.0 2.0 0.9"
    # Two triangles per grid cell.
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, dd = idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]
            lines.append(f"f {a} {b} {c}")
            lines.append(f"f {a} {c} {dd}")
    obj = tmp_path / "handmade.obj"
    obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = tmp_path / "report.json"
    out = tmp_path / "fixed.obj"

    rc = main(["fix", str(obj), "--out", str(out), "--report", str(rep)])
    assert rc == 0
    report = json.loads(rep.read_text(encoding="utf-8"))
    # The spiked center vertex (0-indexed center-1) should be flagged and pulled down.
    ids = [f["vertex_id"] for f in report["flagged"]]
    assert (center - 1) in ids
    fixed = mesh_io.load(out)
    assert fixed.vertices[center - 1][2] < 0.9  # pulled back toward the plane
