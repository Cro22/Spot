"""Hito 2 tests: the synthetic-defect oracle.

Done criterion: on a clean mesh with one synthetically displaced vertex, that
vertex is flagged and clean vertices are not. Plus a precision/recall sweep over
injected offsets.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from spot import detect
from spot import mesh as mesh_io
from spot import synthetic


@pytest.fixture(scope="module")
def sphere():
    tm = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    return mesh_io.from_trimesh(tm)


def test_clean_mesh_flags_nothing(sphere):
    # A clean sphere is uniformly curved: no vertex is a local outlier.
    result = detect.detect(sphere)
    assert result.flags.sum() == 0


def test_single_defect_is_flagged(sphere):
    edge = synthetic.mean_edge_length(sphere)
    d = synthetic.inject_defect(sphere, vertex_id=100, magnitude=1.0 * edge)
    result = detect.detect(d.graph)

    assert result.flags[d.vertex_id], "the displaced vertex must be flagged"
    # Clean vertices well away from the defect must stay unflagged.
    others = np.delete(result.flagged_indices, np.where(result.flagged_indices == d.vertex_id))
    assert len(others) <= 2, f"too many false positives near the defect: {others}"


def test_precision_recall_sweep(sphere):
    """Sweep injected offsets; report precision/recall. Larger shoves must land."""
    edge = synthetic.mean_edge_length(sphere)
    rng = np.random.default_rng(0)
    sample = rng.choice(sphere.n_vertices, size=25, replace=False)

    rows = []
    for scale in (0.25, 0.5, 1.0, 2.0):
        tp = fp = fn = 0
        for vid in sample:
            d = synthetic.inject_defect(sphere, int(vid), magnitude=scale * edge)
            flagged = set(detect.detect(d.graph).flagged_indices.tolist())
            if d.vertex_id in flagged:
                tp += 1
            else:
                fn += 1
            fp += len(flagged - {d.vertex_id})
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        rows.append((scale, precision, recall))
        print(f"offset={scale:.2f}*edge  precision={precision:.3f}  recall={recall:.3f}")

    by_scale = {s: (p, r) for s, p, r in rows}
    # A full-edge shove or larger is unambiguous: always caught.
    assert by_scale[1.0][1] == 1.0
    assert by_scale[2.0][1] == 1.0
    # And with few false positives.
    assert by_scale[1.0][0] > 0.8
    assert by_scale[2.0][0] > 0.8


def test_defect_survives_on_non_sphere():
    # Detection must not depend on the sphere's constant curvature. A torus is
    # smooth with genuinely varying curvature (and no legitimate sharp features
    # for V1 to false-positive on -- that is Hito 5 / Suzanne's problem).
    tm = trimesh.creation.torus(major_radius=1.0, minor_radius=0.35,
                                major_sections=48, minor_sections=24)
    g = mesh_io.from_trimesh(tm)
    edge = synthetic.mean_edge_length(g)
    d = synthetic.inject_defect(g, vertex_id=200, magnitude=1.0 * edge)
    result = detect.detect(d.graph)
    assert result.flags[d.vertex_id]
    # Clean torus: no spurious flags.
    clean = detect.detect(g)
    assert clean.flags.sum() == 0
