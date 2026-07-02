"""Polyscope visualization.

Kept deliberately thin in Hito 0: register the mesh and open the viewer. Later
hitos will layer flagged-vertex highlights and a before/after toggle on top.
"""

from __future__ import annotations

from .mesh import MeshGraph


def show(graph: MeshGraph, name: str = "spot") -> None:
    """Open a polyscope window showing the mesh.

    Imports polyscope lazily so that headless environments (tests, CI) can use
    the rest of the package without a display or GL context.
    """
    import polyscope as ps

    ps.init()
    ps.register_surface_mesh(name, graph.vertices, graph.faces)
    ps.show()
