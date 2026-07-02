"""Mesh IO and graph construction.

A mesh is a graph: vertices are nodes, edges are edges, triangles supply the
weights (later, in the Laplacian). Hito 0 only builds the graph structure --
1-ring vertex adjacency and vertex->face incidence -- and validates that we were
handed a manifold triangle mesh, failing loudly if not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


class NotAManifoldTriangleMesh(ValueError):
    """Raised when the input is not a clean manifold triangle mesh.

    V1 refuses to guess about degenerate input. Repair is explicitly out of
    scope (see CLAUDE-spot.md), so we surface the problem and stop.
    """


@dataclass
class MeshGraph:
    """A triangle mesh viewed as a graph, plus the adjacency we build on it.

    Attributes
    ----------
    vertices : (n_vertices, 3) float64
        Vertex positions.
    faces : (n_faces, 3) int64
        Triangle vertex indices.
    neighbors : list[np.ndarray]
        1-ring adjacency: ``neighbors[i]`` is the sorted array of vertex indices
        sharing an edge with vertex ``i``.
    vertex_faces : list[np.ndarray]
        Face incidence: ``vertex_faces[i]`` is the array of face indices that
        contain vertex ``i``.
    source : Path | None
        Where the mesh came from, for reporting. We never let this influence the
        geometry.
    """

    vertices: np.ndarray
    faces: np.ndarray
    neighbors: list[np.ndarray]
    vertex_faces: list[np.ndarray]
    source: Path | None = None

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def _as_single_trimesh(loaded: object) -> trimesh.Trimesh:
    """Collapse whatever ``trimesh.load`` returned into one Trimesh.

    GLB and other scene formats can load as a ``trimesh.Scene`` with several
    geometries; we concatenate them into a single mesh so the rest of the tool
    sees one graph.
    """
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.dump().geometry.values()) if hasattr(loaded.dump(), "geometry") else loaded.dump()
        meshes = [g for g in geometries if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise NotAManifoldTriangleMesh("Scene contains no triangle geometry.")
        return trimesh.util.concatenate(meshes)
    raise NotAManifoldTriangleMesh(f"Unsupported mesh object: {type(loaded).__name__}")


def _validate_manifold(mesh: trimesh.Trimesh) -> None:
    """Require a manifold triangle mesh; raise loudly with a reason otherwise."""
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise NotAManifoldTriangleMesh("Mesh is not a pure triangle mesh (faces are not all triangles).")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise NotAManifoldTriangleMesh("Mesh is empty.")

    # Edge-manifold check: every undirected edge belongs to at most two faces.
    # trimesh gives us the sorted unique edges and, for each face-edge, which
    # unique edge it maps to; counting those is exactly the per-edge face count.
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    over = int(np.count_nonzero(counts > 2))
    if over:
        raise NotAManifoldTriangleMesh(
            f"Non-manifold mesh: {over} edge(s) shared by more than two faces."
        )


def load(path: str | Path) -> MeshGraph:
    """Load a mesh from OBJ/STL/GLB (etc.) and build its graph.

    Fails loudly if the result is not a manifold triangle mesh.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such mesh file: {path}")

    # process=False: keep the vertices exactly as authored so a flagged, unchanged
    # vertex stays byte-for-byte identical through the pipeline. trimesh's default
    # processing would merge/reorder vertices out from under us.
    loaded = trimesh.load(path, process=False, force="mesh")
    mesh = _as_single_trimesh(loaded)
    _validate_manifold(mesh)

    return from_trimesh(mesh, source=path)


def from_trimesh(mesh: trimesh.Trimesh, source: Path | None = None) -> MeshGraph:
    """Build a :class:`MeshGraph` from an in-memory Trimesh (used by tests too)."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # trimesh already computes clean 1-ring adjacency and vertex->face incidence.
    neighbors = [np.sort(np.asarray(n, dtype=np.int64)) for n in mesh.vertex_neighbors]
    vertex_faces = [
        np.asarray(vf[vf >= 0], dtype=np.int64)  # trimesh pads with -1; drop the padding
        for vf in mesh.vertex_faces
    ]

    return MeshGraph(
        vertices=vertices,
        faces=faces,
        neighbors=neighbors,
        vertex_faces=vertex_faces,
        source=source,
    )
