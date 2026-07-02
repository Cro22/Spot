"""Per-vertex defect signals.

Three geometric signals, each large at a carelessly misplaced vertex, computed
from geometry alone. They are combined into a single defect signal that the
detector then flags as a *local* outlier (see :mod:`spot.detect`).

    1. mean curvature      |H_i|                     -- spikes at a bad vertex
    2. laplacian displacement  |v_i - avg_neighbors|  -- "it is offset" directly
    3. normal coherence    angle(n_i, neighborhood n) -- disagrees with neighbors
"""

from __future__ import annotations

import numpy as np

from .mesh import MeshGraph
from .operators import mean_curvature, uniform_laplacian

_EPS = 1e-12


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), _EPS)


def face_normals_areas(graph: MeshGraph) -> tuple[np.ndarray, np.ndarray]:
    """Unit face normals and triangle areas."""
    V, F = graph.vertices, graph.faces
    raw = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = 0.5 * np.linalg.norm(raw, axis=1)
    return _normalize_rows(raw), area


def vertex_normals(graph: MeshGraph) -> np.ndarray:
    """Area-weighted mean of each vertex's incident face normals, unit length."""
    fn, area = face_normals_areas(graph)
    acc = np.zeros_like(graph.vertices)
    for c in range(3):
        np.add.at(acc, graph.faces[:, c], area[:, None] * fn)
    return _normalize_rows(acc)


def mean_curvature_signal(graph: MeshGraph) -> np.ndarray:
    """|H_i|, the discrete mean-curvature magnitude."""
    H, _ = mean_curvature(graph)
    return H


def laplacian_displacement(graph: MeshGraph) -> np.ndarray:
    """|v_i - (average of neighbors)|.

    The uniform Laplacian gives (L_u V)_i = mean_j v_j - v_i, i.e. exactly the
    offset from where local smoothness predicts the vertex should sit, so its
    row-norm is the displacement magnitude. The most direct "it is offset" cue.
    """
    Lu = uniform_laplacian(graph)
    return np.linalg.norm(Lu @ graph.vertices, axis=1)


def normal_coherence(graph: MeshGraph) -> np.ndarray:
    """Angle (radians) between a vertex's normal and its neighborhood's normal.

    The brief's cue is "a shoved vertex disagrees sharply with its neighborhood".
    We take each vertex normal (area-weighted from its incident faces) and the
    smoothed neighborhood normal (mean of the 1-ring's vertex normals); a shoved
    vertex tilts its incident faces, swinging its normal away from that smoothed
    field and opening the angle.
    """
    vn = vertex_normals(graph)
    nbr = np.zeros_like(vn)
    for i, nbrs in enumerate(graph.neighbors):
        if len(nbrs):
            nbr[i] = vn[nbrs].mean(axis=0)
    nbr = _normalize_rows(nbr)
    cos = np.clip(np.einsum("ij,ij->i", vn, nbr), -1.0, 1.0)
    return np.arccos(cos)


def local_edge_length(graph: MeshGraph) -> np.ndarray:
    """Mean incident-edge length per vertex -- the local length scale.

    Used to make the signals dimensionless so a single threshold works across
    meshes (and across regions of a mesh with uneven vertex density).
    """
    V = graph.vertices
    out = np.zeros(graph.n_vertices)
    for i, nbrs in enumerate(graph.neighbors):
        if len(nbrs):
            out[i] = np.mean(np.linalg.norm(V[nbrs] - V[i], axis=1))
    # Isolated vertices (no edges) get the global mean so we never divide by 0.
    out[out == 0] = out[out > 0].mean() if np.any(out > 0) else 1.0
    return out


def defect_signals(graph: MeshGraph) -> dict[str, np.ndarray]:
    """The three signals, each made dimensionless by the local length scale.

    Raw units differ (curvature ~ 1/length, displacement ~ length, normal ~ rad),
    so we express the first two against the local edge length h_i:

        curvature   -> H_i * h_i    (how much the surface bends over one edge)
        displacement-> |offset|/h_i (offset in units of an edge)
        normal      -> angle (already dimensionless)

    On a smooth surface all three are small; a shoved vertex makes at least one
    O(1). The local-outlier test then decides if that is abnormal *for here*.
    """
    h = local_edge_length(graph)
    return {
        "curvature": mean_curvature_signal(graph) * h,
        "displacement": laplacian_displacement(graph) / h,
        "normal": normal_coherence(graph),
    }
