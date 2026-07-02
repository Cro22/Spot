"""Synthetic-defect harness: the geometry is the oracle.

Take a clean mesh, shove one vertex by a known offset, and we know exactly which
vertex the detector should flag and where the fix should return it. Built in
Hito 2 and reused by every later change (CLAUDE-spot.md, "Testing").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import signals
from .mesh import MeshGraph


def mean_edge_length(graph: MeshGraph) -> float:
    """Average edge length, the natural unit for a defect offset."""
    V = graph.vertices
    lengths = [
        np.linalg.norm(V[i] - V[j]) for i, nbrs in enumerate(graph.neighbors) for j in nbrs
    ]
    return float(np.mean(lengths))


@dataclass
class Defect:
    graph: MeshGraph        # mesh with one vertex displaced (topology unchanged)
    vertex_id: int          # the displaced vertex
    original_position: np.ndarray
    offset: np.ndarray      # displacement applied


def inject_defect(graph: MeshGraph, vertex_id: int, magnitude: float,
                  direction: np.ndarray | None = None) -> Defect:
    """Displace one vertex by ``magnitude`` (in mesh units).

    Default direction is the vertex normal, i.e. shove it straight off the
    surface -- the classic careless-sculpt spike. Topology is untouched, so the
    adjacency carries over unchanged.
    """
    if direction is None:
        direction = signals.vertex_normals(graph)[vertex_id]
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    offset = magnitude * direction

    V = graph.vertices.copy()
    original = V[vertex_id].copy()
    V[vertex_id] = original + offset
    moved = replace(graph, vertices=V)
    return Defect(graph=moved, vertex_id=vertex_id, original_position=original, offset=offset)
