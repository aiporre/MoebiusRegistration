#!/usr/bin/env python3
"""Scale the vertex positions of a PLY mesh by a constant factor."""

import argparse
from pathlib import Path

import trimesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input PLY mesh")
    parser.add_argument("output", type=Path, help="output PLY mesh")
    parser.add_argument("factor", type=float, help="scale factor, e.g. 2.0 or 0.5")
    args = parser.parse_args()

    mesh = trimesh.load(args.input, force="mesh")
    mesh.vertices *= args.factor
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)

    print(f"Scaled {len(mesh.vertices)} vertices by {args.factor}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
