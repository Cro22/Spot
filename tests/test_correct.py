"""Hito 3 tests: local Taubin correction, displacement-capped.

Done criterion: the injected vertex returns near its original position AND the
maximum displacement over all non-flagged vertices is about zero.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from spot import correct, detect
from spot import mesh as mesh_io
from spot import synthetic


@pytest.fixture(scope="module")
def sphere():
    return mesh_io.from_trimesh(trimesh.creation.icosphere(subdivisions=4, radius=1.0))


def test_non_flagged_vertices_are_untouched(sphere):
    edge = synthetic.mean_edge_length(sphere)
    d = synthetic.inject_defect(sphere, vertex_id=100, magnitude=1.0 * edge)
    result = detect.detect(d.graph)
    fix = correct.taubin_fix(d.graph, result.flags)

    # Exactly zero: non-flagged vertices are never written.
    assert fix.max_nonflagged_displacement == 0.0
    # Byte-for-byte identical for every non-flagged vertex.
    mask = ~result.flags
    assert np.array_equal(fix.vertices[mask], d.graph.vertices[mask])


def test_injected_vertex_returns_near_original(sphere):
    edge = synthetic.mean_edge_length(sphere)
    for scale in (0.5, 1.0, 2.0):
        d = synthetic.inject_defect(sphere, vertex_id=100, magnitude=scale * edge)
        result = detect.detect(d.graph)
        assert result.flags[d.vertex_id]

        fix = correct.taubin_fix(d.graph, result.flags)

        err_before = np.linalg.norm(d.graph.vertices[d.vertex_id] - d.original_position)
        err_after = np.linalg.norm(fix.vertices[d.vertex_id] - d.original_position)
        # The fix must strictly improve the position...
        assert err_after < err_before, f"scale={scale}: {err_after:.4f} vs {err_before:.4f}"
        # ...landing within a small fraction of an edge of the true position. The
        # residual floor is the ring centroid sitting just inside the sphere; a
        # single vertex pinned by a fixed ring cannot do better than its neighbor
        # average, so ~0.1*edge is the discretization limit, not a weak fix.
        assert err_after < 0.1 * edge


def test_displacement_cap_is_respected(sphere):
    edge = synthetic.mean_edge_length(sphere)
    d = synthetic.inject_defect(sphere, vertex_id=100, magnitude=2.0 * edge)
    result = detect.detect(d.graph)

    cap = 0.5  # deliberately tight cap, in neighborhood-scale units
    fix = correct.taubin_fix(d.graph, result.flags, max_displacement=cap)
    # The cap is measured in ring-radius units; on a sphere that is ~one edge.
    ring_radius = correct._neighborhood_scale(d.graph, np.array([d.vertex_id]))[0]
    moved = np.linalg.norm(fix.displacement[d.vertex_id])
    assert moved <= cap * ring_radius * 1.02  # clamp is exact, tiny float slack


def test_clean_mesh_is_a_noop(sphere):
    # Nothing flagged -> nothing moves at all.
    result = detect.detect(sphere)
    assert result.flags.sum() == 0
    fix = correct.taubin_fix(sphere, result.flags)
    assert np.array_equal(fix.vertices, sphere.vertices)
