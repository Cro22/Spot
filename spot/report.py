"""The defect report -- Spot's public API.

This report is the input contract for Suzanne, the judgment layer built on top of
Spot (CLAUDE-spot.md step 5). Treat the schema as public and version it: consumers
should read ``schema_version`` and refuse anything with an unexpected major
version. Bump MAJOR on breaking changes, MINOR on additive ones.

Schema (v1):
    schema_version : str, "MAJOR.MINOR.PATCH"
    tool           : {name, version}
    source         : {path, n_vertices, n_faces}
    parameters     : the detection/correction knobs used
    summary        : {n_flagged, max_displacement, max_nonflagged_displacement}
    flagged        : list of per-vertex records, each:
        vertex_id            : int
        original_position    : [x, y, z]   (as loaded -- for a defect, the wrong spot)
        corrected_position   : [x, y, z]
        displacement         : [dx, dy, dz] (corrected - original)
        displacement_magnitude : float
        local_z              : float        (combined local outlier score)
        signals              : {curvature, displacement, normal}  (dimensionless)
        signal_z             : {curvature, displacement, normal}  (per-signal local z)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import __version__
from .correct import Correction
from .detect import DetectionResult
from .mesh import MeshGraph

SCHEMA_VERSION = "1.0.0"


def build_report(
    graph: MeshGraph,
    detection: DetectionResult,
    correction: Correction,
    parameters: dict | None = None,
) -> dict:
    """Assemble the report dict from a detection + correction of ``graph``."""
    orig = graph.vertices
    comp = detection.components
    comp_z = detection.component_z

    flagged = []
    for i in detection.flagged_indices:
        i = int(i)
        disp = correction.displacement[i]
        flagged.append(
            {
                "vertex_id": i,
                "original_position": orig[i].tolist(),
                "corrected_position": correction.vertices[i].tolist(),
                "displacement": disp.tolist(),
                "displacement_magnitude": float(np.linalg.norm(disp)),
                "local_z": float(detection.local_z[i]),
                "signals": {name: float(comp[name][i]) for name in comp},
                "signal_z": {name: float(comp_z[name][i]) for name in comp_z},
            }
        )

    mags = [f["displacement_magnitude"] for f in flagged]
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "spot", "version": __version__},
        "source": {
            "path": str(graph.source) if graph.source else None,
            "n_vertices": graph.n_vertices,
            "n_faces": graph.n_faces,
        },
        "parameters": parameters or {},
        "summary": {
            "n_flagged": len(flagged),
            "max_displacement": max(mags) if mags else 0.0,
            "max_nonflagged_displacement": correction.max_nonflagged_displacement,
        },
        "flagged": flagged,
    }


def write_report(report: dict, path: str | Path) -> None:
    """Write the report to a JSON file."""
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
