"""Discrete differential operators on a triangle mesh.

This is the part of Spot worth understanding, so it is built by hand from the
triangle geometry rather than pulled from a library. Everything here follows the
standard cotangent / mixed-Voronoi discretization (Meyer, Desbrun, Schroeder &
Barr, "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds",
2003).

Sign convention (matches libigl's ``cotmatrix`` / ``massmatrix``):
    L is the negative-semidefinite cotangent *stiffness* matrix,
        L_ij = (cot alpha_ij + cot beta_ij) / 2   for an edge (i, j),
        L_ii = -sum_{j != i} L_ij,
    so every row sums to zero and L is symmetric.
    M is the lumped (diagonal) mixed-Voronoi *mass* matrix.
    The discrete Laplace-Beltrami operator is then M^{-1} L, and applied to the
    vertex coordinates it recovers the mean-curvature normal:
        (M^{-1} L V)_i  ~=  -2 H_i n_i,     hence   H_i ~= 0.5 * ||(M^{-1} L V)_i||.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .mesh import MeshGraph


def _corner_cotangents(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Cotangent of each triangle's interior angle, per corner.

    Returns an (n_faces, 3) array where column c holds the cotangent of the
    angle at the c-th vertex of the face. cot(theta) = cos/sin = (u . v) / |u x v|
    for the two edge vectors u, v emanating from that corner.
    """
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    # Edge vectors from each corner to the other two vertices.
    u0, v0 = p1 - p0, p2 - p0  # at corner 0
    u1, v1 = p0 - p1, p2 - p1  # at corner 1
    u2, v2 = p0 - p2, p1 - p2  # at corner 2

    def cot(u: np.ndarray, w: np.ndarray) -> np.ndarray:
        dot = np.einsum("ij,ij->i", u, w)
        cross = np.linalg.norm(np.cross(u, w), axis=1)
        # cross == 2*area > 0 for a non-degenerate triangle; guard against /0.
        return dot / np.maximum(cross, 1e-12)

    return np.stack([cot(u0, v0), cot(u1, v1), cot(u2, v2)], axis=1)


def cotangent_laplacian(graph: MeshGraph) -> sp.csr_matrix:
    """Sparse cotangent (stiffness) Laplacian L, negative-semidefinite.

    Each triangle contributes cot(theta)/2 to the weight of the edge *opposite*
    the corner theta. An interior edge is shared by two triangles, so its two
    contributions sum to the familiar (cot alpha + cot beta) / 2.
    """
    V, F = graph.vertices, graph.faces
    cots = _corner_cotangents(V, F)

    # The angle at corner 0 sits opposite edge (v1, v2), and so on cyclically.
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    # (row, col, half-weight) for each opposite edge, both orientations.
    rows = np.concatenate([i1, i2, i2, i0, i0, i1])
    cols = np.concatenate([i2, i1, i0, i2, i1, i0])
    half = 0.5 * cots  # cot/2 per corner
    data = np.concatenate([half[:, 0], half[:, 0],
                           half[:, 1], half[:, 1],
                           half[:, 2], half[:, 2]])

    n = graph.n_vertices
    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()  # off-diagonal weights
    # L = W - diag(row sums of W): off-diagonals are the edge weights, and the
    # diagonal is set so each row sums to exactly zero.
    d = np.asarray(W.sum(axis=1)).ravel()
    L = W - sp.diags(d)
    return L.tocsr()


def voronoi_mass(graph: MeshGraph) -> sp.dia_matrix:
    """Lumped (diagonal) mixed-Voronoi mass matrix M.

    Per Meyer et al.: a non-obtuse triangle splits by its circumcenter, giving
    each vertex an exact Voronoi area; an obtuse triangle would put the
    circumcenter outside the triangle, so it is split by area instead (half to
    the obtuse corner, a quarter to each of the others).
    """
    V, F = graph.vertices, graph.faces
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cots = _corner_cotangents(V, F)
    c0, c1, c2 = cots[:, 0], cots[:, 1], cots[:, 2]

    # Squared edge lengths, named by the two corners they connect.
    l01 = np.einsum("ij,ij->i", p1 - p0, p1 - p0)
    l12 = np.einsum("ij,ij->i", p2 - p1, p2 - p1)
    l02 = np.einsum("ij,ij->i", p2 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    # Voronoi area for each corner: (1/8) * sum over its two edges of
    # |edge|^2 * cot(angle opposite that edge).
    vor0 = (l02 * c1 + l01 * c2) / 8.0
    vor1 = (l01 * c2 + l12 * c0) / 8.0
    vor2 = (l02 * c1 + l12 * c0) / 8.0
    vor = np.stack([vor0, vor1, vor2], axis=1)

    # Obtuse fallback. A corner is obtuse iff its cotangent is negative.
    obtuse_corner = cots < 0.0
    obtuse_face = obtuse_corner.any(axis=1)
    # For obtuse faces: area/2 at the obtuse corner, area/4 elsewhere.
    fallback = np.where(obtuse_corner, (area / 2.0)[:, None], (area / 4.0)[:, None])
    contrib = np.where(obtuse_face[:, None], fallback, vor)

    m = np.zeros(graph.n_vertices)
    np.add.at(m, F.ravel(), contrib.ravel())
    return sp.diags(m)


def uniform_laplacian(graph: MeshGraph) -> sp.csr_matrix:
    """Combinatorial (umbrella) Laplacian for comparison.

    (L_u x)_i = mean_{j in N(i)} x_j - x_i: it ignores geometry and weights every
    neighbor equally. Rows sum to zero. Used as a baseline against the cotangent
    operator and as the averaging kernel for the Laplacian-displacement signal.
    """
    n = graph.n_vertices
    rows, cols, data = [], [], []
    for i, nbrs in enumerate(graph.neighbors):
        deg = len(nbrs)
        if deg == 0:
            continue
        rows.extend([i] * deg)
        cols.extend(nbrs.tolist())
        data.extend([1.0 / deg] * deg)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return (A - sp.identity(n, format="csr")).tocsr()


def mean_curvature(graph: MeshGraph) -> tuple[np.ndarray, np.ndarray]:
    """Discrete mean curvature via the cotangent Laplacian.

    Returns ``(H, Hvec)`` where ``Hvec = M^{-1} (L V)`` is the mean-curvature
    normal per vertex and ``H = 0.5 * ||Hvec||`` its magnitude. On a unit sphere
    H should be ~= 1/R and roughly uniform.
    """
    L = cotangent_laplacian(graph)
    M = voronoi_mass(graph)
    minv = 1.0 / np.maximum(M.diagonal(), 1e-12)
    Hvec = minv[:, None] * (L @ graph.vertices)
    H = 0.5 * np.linalg.norm(Hvec, axis=1)
    return H, Hvec
