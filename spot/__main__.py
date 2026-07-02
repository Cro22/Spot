"""Command-line entry point: ``python -m spot ...``.

    spot view MODEL                     -- load, print counts, show in polyscope
    spot fix  MODEL --out F --report R  -- detect, correct, write mesh + report
"""

from __future__ import annotations

import argparse
import sys

import json
from pathlib import Path

from . import bench, correct, detect, report
from . import mesh as mesh_io


def _cmd_view(args: argparse.Namespace) -> int:
    graph = mesh_io.load(args.model)
    print(f"Loaded {graph.source}")
    print(f"  vertices: {graph.n_vertices}")
    print(f"  faces:    {graph.n_faces}")
    if args.no_show:
        return 0
    from . import view

    view.show(graph, name=graph.source.stem if graph.source else "spot")
    return 0


def _cmd_fix(args: argparse.Namespace) -> int:
    graph = mesh_io.load(args.model)
    det = detect.detect(graph, k=args.k, threshold=args.threshold)
    corr = correct.taubin_fix(
        graph,
        det.flags,
        iterations=args.iterations,
        max_displacement=args.max_displacement,
    )

    parameters = {
        "k": args.k,
        "threshold": args.threshold,
        "iterations": args.iterations,
        "max_displacement": args.max_displacement,
    }
    rep = report.build_report(graph, det, corr, parameters=parameters)

    print(f"Loaded {graph.source}  ({graph.n_vertices} verts, {graph.n_faces} faces)")
    print(f"  flagged: {rep['summary']['n_flagged']} vertex(es)")
    print(f"  max displacement applied: {rep['summary']['max_displacement']:.6f}")
    print(f"  max non-flagged displacement: {rep['summary']['max_nonflagged_displacement']:.2e}")

    if args.out:
        mesh_io.save(args.out, corr.vertices, graph.faces)
        print(f"  wrote fixed mesh -> {args.out}")
    if args.report:
        report.write_report(rep, args.report)
        print(f"  wrote report     -> {args.report}")
    if args.view:
        from . import view

        view.show_result(graph, det, corr, name=graph.source.stem if graph.source else "spot")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    entries = bench.run_benchmark(k=args.k, threshold=args.threshold)

    print("Feature false-positive benchmark (clean meshes -- every flag is a false positive)")
    print(f"  {'mesh':10s} {'kind':8s} {'verts':>6s} {'flagged':>8s} {'fp_rate':>8s}")
    for e in entries:
        print(f"  {e.name:10s} {e.kind:8s} {e.n_vertices:6d} {e.n_flagged:8d} {100 * e.fp_rate:7.2f}%")

    doc = bench.benchmark_document(entries, k=args.k, threshold=args.threshold)
    agg = doc["aggregate"]
    print(f"  feature FP rate: {100 * agg['feature_false_positive_rate']:.2f}% "
          f"({agg['feature_false_positives']}/{agg['feature_vertices_total']})")

    print("\nThreshold sweep on feature meshes (total false positives):")
    for t, total in bench.threshold_sweep(k=args.k).items():
        print(f"  threshold={t:.1f}  false_positives={total}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # The Suzanne "feature" benchmark class.
        (out / "benchmark.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
        # Each mesh + the standard defect report (what Spot would do to it).
        for name, (graph, kind) in bench.build_suite().items():
            mesh_io.save(out / f"{name}.obj", graph.vertices, graph.faces)
            det = detect.detect(graph, k=args.k, threshold=args.threshold)
            corr = correct.taubin_fix(graph, det.flags)
            rep = report.build_report(graph, det, corr,
                                      parameters={"k": args.k, "threshold": args.threshold})
            report.write_report(rep, out / f"{name}.report.json")
        print(f"\nwrote benchmark -> {out / 'benchmark.json'} (+ per-mesh meshes and reports)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_view = sub.add_parser("view", help="Load a mesh, print its counts, and show it in polyscope.")
    p_view.add_argument("model", help="Path to a mesh file (OBJ/STL/GLB/...).")
    p_view.add_argument("--no-show", action="store_true",
                        help="Load and report counts only; skip opening the viewer.")
    p_view.set_defaults(func=_cmd_view)

    p_fix = sub.add_parser("fix", help="Detect misplaced vertices, correct them, write mesh + report.")
    p_fix.add_argument("model", help="Path to a mesh file (OBJ/STL/GLB/...).")
    p_fix.add_argument("--out", help="Where to write the corrected mesh (format by extension).")
    p_fix.add_argument("--report", help="Where to write the JSON defect report.")
    p_fix.add_argument("--k", type=int, default=2, help="k-ring radius for the local outlier test.")
    p_fix.add_argument("--threshold", type=float, default=3.5, help="Local z-score flag threshold.")
    p_fix.add_argument("--iterations", type=int, default=20, help="Taubin lambda|mu iteration pairs.")
    p_fix.add_argument("--max-displacement", type=float, default=3.0, dest="max_displacement",
                       help="Per-vertex displacement cap, in local neighborhood-scale units.")
    p_fix.add_argument("--view", action="store_true", help="Open a before/after polyscope view.")
    p_fix.set_defaults(func=_cmd_fix)

    p_bench = sub.add_parser(
        "bench",
        help="Measure V1's false positives on genuine features (Suzanne's benchmark).",
    )
    p_bench.add_argument("--out-dir", dest="out_dir",
                         help="Write the benchmark class + per-mesh reports here.")
    p_bench.add_argument("--k", type=int, default=2, help="k-ring radius for the local outlier test.")
    p_bench.add_argument("--threshold", type=float, default=3.5, help="Local z-score flag threshold.")
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
