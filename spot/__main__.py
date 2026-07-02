"""Command-line entry point: ``python -m spot ...``.

Hito 0 ships a single ``view`` subcommand. The ``fix`` subcommand promised in
Hito 4 will be added once detection and correction exist.
"""

from __future__ import annotations

import argparse
import sys

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_view = sub.add_parser("view", help="Load a mesh, print its counts, and show it in polyscope.")
    p_view.add_argument("model", help="Path to a mesh file (OBJ/STL/GLB/...).")
    p_view.add_argument(
        "--no-show",
        action="store_true",
        help="Load and report counts only; skip opening the viewer (for headless use).",
    )
    p_view.set_defaults(func=_cmd_view)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
