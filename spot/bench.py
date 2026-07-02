"""Hito 5: measure V1's false positives on genuine features.

This is the doorway to the research version, and it stays shut on purpose
(CLAUDE-spot.md). We do NOT try to tell a real sharp corner from a defect --
that needs a learned prior of intent, which is Suzanne's job. We only *measure*
how often V1 flags legitimate features, and export that flagged set as the
"feature" benchmark class Suzanne is scored against.

Every mesh here is clean (no injected defect), so every flag is a false positive
by construction. The interesting finding the local-outlier test produces:

    * uniform ridges (a cylinder rim, most of a cube's edges) are NOT flagged --
      a whole ridge is "normal for around here";
    * ISOLATED sharp features (a cube corner, a cone apex) ARE flagged -- they
      are geometrically indistinguishable from a careless spike without intent.

That isolated-feature set is exactly what Suzanne must learn to spare.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import trimesh

from . import detect
from . import mesh as mesh_io
from .mesh import MeshGraph

BENCHMARK_SCHEMA_VERSION = "1.0.0"


def build_suite() -> dict[str, tuple[MeshGraph, str]]:
    """The benchmark suite: feature-bearing meshes plus smooth controls.

    Returns ``{name: (graph, kind)}`` where kind is "feature" (has genuine sharp
    features V1 may false-positive on) or "smooth" (a control that should stay
    clean). All are clean -- no defect is injected.
    """
    raw = {
        "cube": ("feature", trimesh.creation.box(extents=(1, 1, 1)).subdivide().subdivide()),
        "cylinder": ("feature", trimesh.creation.cylinder(radius=0.5, height=1.5, sections=48)),
        "cone": ("feature", trimesh.creation.cone(radius=0.6, height=1.2, sections=48)),
        "sphere": ("smooth", trimesh.creation.icosphere(subdivisions=3, radius=1.0)),
        "torus": ("smooth", trimesh.creation.torus(1.0, 0.35, major_sections=48, minor_sections=24)),
    }
    return {name: (mesh_io.from_trimesh(tm), kind) for name, (kind, tm) in raw.items()}


@dataclass
class BenchEntry:
    name: str
    kind: str
    n_vertices: int
    n_flagged: int      # every flag is a false positive (mesh is clean)
    fp_rate: float
    feature_vertices: list[int] = field(default_factory=list)


def measure(name: str, graph: MeshGraph, kind: str, k: int = 2, threshold: float = 3.5) -> BenchEntry:
    """Run detection on a clean mesh and record its false positives."""
    result = detect.detect(graph, k=k, threshold=threshold)
    flagged = result.flagged_indices.tolist()
    n = graph.n_vertices
    return BenchEntry(
        name=name,
        kind=kind,
        n_vertices=n,
        n_flagged=len(flagged),
        fp_rate=len(flagged) / n if n else 0.0,
        feature_vertices=[int(i) for i in flagged],
    )


def run_benchmark(k: int = 2, threshold: float = 3.5) -> list[BenchEntry]:
    """Measure false positives across the whole suite."""
    return [measure(name, g, kind, k, threshold) for name, (g, kind) in build_suite().items()]


def threshold_sweep(thresholds=(3.0, 3.5, 4.5, 6.0, 8.0), k: int = 2) -> dict:
    """Total false positives on the feature meshes as the threshold is raised.

    Characterizes the doorway: how much of the false-positive load a stricter
    threshold sheds, and where genuine isolated features stop being flagged.
    """
    suite = build_suite()
    out = {}
    for t in thresholds:
        total = sum(
            measure(name, g, kind, k, t).n_flagged
            for name, (g, kind) in suite.items()
            if kind == "feature"
        )
        out[t] = total
    return out


def benchmark_document(entries: list[BenchEntry], k: int, threshold: float) -> dict:
    """The 'feature' benchmark class Suzanne consumes: what V1 wrongly flags.

    Its own versioned schema, separate from the per-mesh defect report.
    """
    feature_entries = [e for e in entries if e.kind == "feature"]
    total_feat_v = sum(e.n_vertices for e in feature_entries)
    total_feat_fp = sum(e.n_flagged for e in feature_entries)
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "description": "Vertices V1 flags on clean feature meshes. Ground-truth "
        "label: 'feature' (legitimate, must NOT be corrected). Suzanne's task is "
        "to spare these while still accepting real defects.",
        "parameters": {"k": k, "threshold": threshold},
        "aggregate": {
            "feature_false_positive_rate": total_feat_fp / total_feat_v if total_feat_v else 0.0,
            "feature_false_positives": total_feat_fp,
            "feature_vertices_total": total_feat_v,
        },
        "meshes": [asdict(e) for e in entries],
    }
