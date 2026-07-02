"""Local-outlier flagging.

The crux of the tool (CLAUDE-spot.md step 3): a defect is "weird compared to
right around me", not "weird compared to the whole model". Curvature is
legitimately high in curved regions, so a *global* threshold would flag every
genuine feature. Instead, for each vertex we measure how far its defect signal
sits from the median of its own k-ring, in robust (MAD) units, and flag only the
local outliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from . import signals
from .mesh import MeshGraph

_EPS = 1e-12


@dataclass
class DetectionResult:
    flags: np.ndarray            # bool, per vertex
    local_z: np.ndarray          # combined local z-score (max over the three signals)
    component_z: dict            # per-signal local z-scores
    components: dict             # per-signal dimensionless signal arrays
    k: int
    threshold: float

    @property
    def flagged_indices(self) -> np.ndarray:
        return np.flatnonzero(self.flags)


def kring_membership(graph: MeshGraph, k: int) -> sp.csr_matrix:
    """Boolean CSR where row i marks every vertex within k hops of i (i included).

    Built by booleanized powers of (I + A): reachability in <=k steps.
    """
    n = graph.n_vertices
    rows, cols = [], []
    for i, nbrs in enumerate(graph.neighbors):
        rows.extend([i] * len(nbrs))
        cols.extend(nbrs.tolist())
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    reach = (A + sp.identity(n, format="csr")) > 0  # 1-ring + self, boolean
    out = reach.copy()
    for _ in range(k - 1):
        out = (out @ reach) > 0  # booleanize each step so entries stay 0/1
    return out.tocsr()


def local_robust_z(signal: np.ndarray, kring: sp.csr_matrix, floor: float) -> np.ndarray:
    """Robust z-score of each vertex's signal within its k-ring (median/MAD).

    ``floor`` is an additive floor on the scale, in the (dimensionless) units of
    the signal. It is what keeps a perfectly smooth region from turning float
    noise into outliers: where the true local spread is ~0, the floor dominates
    and any sub-floor wobble scores ~0. A real, edge-scale defect is O(1) and
    still scores far above threshold.
    """
    indptr, indices = kring.indptr, kring.indices
    z = np.zeros_like(signal, dtype=np.float64)
    for i in range(signal.shape[0]):
        neigh = indices[indptr[i] : indptr[i + 1]]
        vals = signal[neigh]
        med = np.median(vals)
        scale = 1.4826 * np.median(np.abs(vals - med)) + floor
        z[i] = (signal[i] - med) / scale
    return z


# Additive scale floors, in each signal's dimensionless units. Variation below
# these is treated as discretization noise, not a defect.
_FLOORS = {"curvature": 0.02, "displacement": 0.02, "normal": 0.02}


def _suppress_non_maxima(flags: np.ndarray, z: np.ndarray, kring: sp.csr_matrix) -> np.ndarray:
    """Keep only k-ring local peaks among flagged vertices.

    We target *isolated* misplaced vertices, but a single spike inflates the
    signals of everything around it -- a far shove even deforms the surrounding
    ring into a ridge whose vertices become peaks of their own 1-ring. A genuine
    single-vertex defect is the peak of its whole k-ring neighborhood, so we
    suppress any flagged vertex that has a strictly higher-scoring vertex within
    that same k-ring (the neighborhood the outlier test already used).
    """
    keep = flags.copy()
    indptr, indices = kring.indptr, kring.indices
    for i in np.flatnonzero(flags):
        neigh = indices[indptr[i] : indptr[i + 1]]
        neigh = neigh[neigh != i]
        if neigh.size and z[neigh].max() > z[i]:
            keep[i] = False
    return keep


def detect(graph: MeshGraph, k: int = 2, threshold: float = 3.5,
           suppress: bool = True) -> DetectionResult:
    """Flag vertices whose defect signal is a local (k-ring) outlier.

    Each of the three dimensionless signals is scored against its own k-ring; the
    combined score is the max over the three, so a vertex flags if it stands out
    on *any* signal. One-sided: a defect means an *elevated* signal. Non-maximum
    suppression then keeps only the 1-ring peaks (see :func:`_suppress_non_maxima`).
    """
    components = signals.defect_signals(graph)
    kring = kring_membership(graph, k)
    component_z = {
        name: local_robust_z(sig, kring, _FLOORS[name]) for name, sig in components.items()
    }
    z = np.maximum.reduce(list(component_z.values()))
    flags = z > threshold
    if suppress:
        flags = _suppress_non_maxima(flags, z, kring)
    return DetectionResult(
        flags=flags,
        local_z=z,
        component_z=component_z,
        components=components,
        k=k,
        threshold=threshold,
    )
