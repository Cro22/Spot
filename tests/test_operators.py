"""Hito 1 tests: validate the operators on spheres (ground truth is free).

For a sphere of radius R the exact mean curvature is 1/R everywhere, so a good
discrete operator should recover ~1/R and be roughly uniform across vertices.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from spot import mesh as mesh_io
from spot import operators as ops


@pytest.fixture
def sphere():
    tm = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    return mesh_io.from_trimesh(tm)


def test_laplacian_symmetric(sphere):
    L = ops.cotangent_laplacian(sphere)
    asym = (L - L.T)
    assert np.abs(asym.data).max() < 1e-9 if asym.nnz else True


def test_laplacian_rows_sum_to_zero(sphere):
    L = ops.cotangent_laplacian(sphere)
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    assert np.abs(row_sums).max() < 1e-9


def test_mass_matrix_totals_surface_area(sphere):
    # Lumped Voronoi areas partition the mesh, so they must sum to the total area.
    M = ops.voronoi_mass(sphere)
    total = M.diagonal().sum()
    expected = sphere_area = 4.0 * np.pi  # unit sphere; the mesh underestimates slightly
    # Icosphere is inscribed, so its area is a bit under the true sphere area.
    assert 0.9 * expected < total <= expected + 1e-6


def test_mean_curvature_unit_sphere(sphere):
    H, _ = ops.mean_curvature(sphere)
    # Trim the tails before judging "roughly uniform": the discretization has a
    # little spread, but the bulk should sit near 1/R = 1.
    med = np.median(H)
    assert 0.9 < med < 1.1
    # Roughly uniform: robust spread (MAD) small relative to the median.
    mad = np.median(np.abs(H - med))
    assert mad < 0.05 * med


def test_mean_curvature_scales_with_radius():
    for R in (0.5, 2.0):
        tm = trimesh.creation.icosphere(subdivisions=4, radius=R)
        g = mesh_io.from_trimesh(tm)
        H, _ = ops.mean_curvature(g)
        assert 0.9 / R < np.median(H) < 1.1 / R


def test_uniform_laplacian_rows_sum_to_zero(sphere):
    Lu = ops.uniform_laplacian(sphere)
    row_sums = np.asarray(Lu.sum(axis=1)).ravel()
    assert np.abs(row_sums).max() < 1e-9


def test_cross_check_libigl_if_available(sphere):
    """Optional cross-check against libigl's cotmatrix, if it imported cleanly."""
    igl = pytest.importorskip("igl")
    L = ops.cotangent_laplacian(sphere).toarray()
    L_igl = igl.cotmatrix(sphere.vertices, sphere.faces).toarray()
    assert np.abs(L - L_igl).max() < 1e-6
