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


def show_result(graph, detection, correction, name: str = "spot") -> None:
    """Before/after view: original and fixed meshes with flagged vertices marked.

    The two meshes overlap; toggle their visibility in polyscope's panel to see
    before vs after. Flagged vertices are shown as a highlighted point cloud.
    """
    import numpy as np
    import polyscope as ps

    ps.init()
    before = ps.register_surface_mesh(f"{name} (before)", graph.vertices, graph.faces)
    after = ps.register_surface_mesh(f"{name} (after)", correction.vertices, graph.faces)
    after.set_enabled(False)  # start on "before"; toggle in the panel

    flagged = detection.flagged_indices
    if len(flagged):
        pts = graph.vertices[flagged]
        cloud = ps.register_point_cloud("flagged", pts)
        cloud.add_scalar_quantity("local_z", np.asarray(detection.local_z)[flagged], enabled=True)
    ps.show()
