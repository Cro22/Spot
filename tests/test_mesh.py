"""Hito 0 tests: loading, graph construction, and the manifold guard."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from spot import mesh as mesh_io
from spot.mesh import NotAManifoldTriangleMesh


def _icosphere() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=2, radius=1.0)


def test_from_trimesh_counts_and_graph():
    tm = _icosphere()
    g = mesh_io.from_trimesh(tm)

    assert g.n_vertices == len(tm.vertices)
    assert g.n_faces == len(tm.faces)
    assert g.faces.shape[1] == 3

    # One adjacency list and one incidence list per vertex.
    assert len(g.neighbors) == g.n_vertices
    assert len(g.vertex_faces) == g.n_vertices

    # Adjacency is symmetric: if j is a neighbor of i, i is a neighbor of j.
    for i, nbrs in enumerate(g.neighbors):
        for j in nbrs:
            assert i in g.neighbors[j]

    # Every incident face of a vertex actually contains that vertex.
    for i, faces in enumerate(g.vertex_faces):
        for f in faces:
            assert i in g.faces[f]


def test_load_roundtrip_obj(tmp_path):
    tm = _icosphere()
    path = tmp_path / "sphere.obj"
    tm.export(path)

    g = mesh_io.load(path)
    assert g.n_vertices == len(tm.vertices)
    assert g.n_faces == len(tm.faces)
    assert g.source == path


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        mesh_io.load(tmp_path / "nope.obj")


def test_non_manifold_edge_rejected():
    # Three triangles sharing a single edge (0-1): a classic non-manifold edge.
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    with pytest.raises(NotAManifoldTriangleMesh):
        mesh_io._validate_manifold(tm)
