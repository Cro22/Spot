"""Local, displacement-capped correction.

We only ever move flagged vertices. Their neighbors are held fixed and act as
anchors, so a flagged vertex is pulled back toward where its (correct) 1-ring
says it should sit, and every non-flagged vertex stays byte-for-byte unchanged
(CLAUDE-spot.md step 4).

Smoothing uses Taubin's lambda|mu scheme rather than plain Laplacian smoothing:
plain smoothing shrinks a mesh, whereas Taubin alternates a shrinking pass
(lambda > 0) with an unshrinking pass (mu < 0) so the low-frequency shape is
preserved. Every pass has the same fixed point, Lu v = 0, i.e. a vertex sitting
at the average of its neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import MeshGraph
from .operators import uniform_laplacian


@dataclass
class Correction:
    vertices: np.ndarray      # corrected vertex positions
    displacement: np.ndarray  # (n, 3) vector actually applied per vertex (0 where unmoved)
    flags: np.ndarray         # the flags that were corrected

    @property
    def max_nonflagged_displacement(self) -> float:
        """Largest movement of any non-flagged vertex (should be exactly 0)."""
        mask = ~self.flags
        if not np.any(mask):
            return 0.0
        return float(np.linalg.norm(self.displacement[mask], axis=1).max())


def _neighborhood_scale(graph: MeshGraph, active: np.ndarray) -> np.ndarray:
    """Local mesh resolution around each active vertex, robust to the defect.

    Uses the ring radius -- the mean distance from a vertex's (un-shoved)
    neighbors to their centroid -- rather than the vertex's own incident edges,
    which are inflated precisely because the vertex was shoved. This is the scale
    the displacement cap is measured in.
    """
    V = graph.vertices
    scale = np.ones(active.shape[0])
    for k, i in enumerate(active):
        nbrs = graph.neighbors[i]
        if nbrs.size:
            centroid = V[nbrs].mean(axis=0)
            scale[k] = np.linalg.norm(V[nbrs] - centroid, axis=1).mean()
    return scale


def taubin_fix(
    graph: MeshGraph,
    flags: np.ndarray,
    lamb: float = 0.5,
    mu: float = -0.53,
    iterations: int = 20,
    max_displacement: float = 3.0,
) -> Correction:
    """Taubin-smooth the flagged vertices, holding everything else fixed.

    Parameters
    ----------
    lamb, mu : Taubin's shrink / unshrink factors (|mu| > lamb > 0).
    iterations : number of lambda|mu pairs.
    max_displacement : per-vertex cap on total movement, in units of the local
        edge length -- we nudge, we do not melt. A flagged vertex never moves
        further than this from its original position.
    """
    active = np.flatnonzero(flags)
    original = graph.vertices.copy()
    V = graph.vertices.copy()

    if active.size:
        Lu = uniform_laplacian(graph)
        for _ in range(iterations):
            for factor in (lamb, mu):
                # (Lu V)_i = mean_of_neighbors(i) - V_i. Compute for all vertices
                # but commit only to the active (flagged) ones, so neighbors stay
                # fixed and non-flagged vertices are never touched.
                delta = Lu @ V
                V[active] += factor * delta[active]

        # Displacement cap: clamp each flagged vertex's total move to
        # max_displacement * local neighborhood scale.
        moved = V[active] - original[active]
        dist = np.linalg.norm(moved, axis=1)
        cap = max_displacement * _neighborhood_scale(graph, active)
        over = dist > cap
        if np.any(over):
            scale = np.ones_like(dist)
            scale[over] = cap[over] / dist[over]
            V[active] = original[active] + moved * scale[:, None]

    displacement = V - original
    return Correction(vertices=V, displacement=displacement, flags=flags.copy())
