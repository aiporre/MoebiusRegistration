#!/usr/bin/env python3
"""Run SphereMap on an example OBJ and count collapsed spherical triangles."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spheremap import SphereMap, extract_sphere
from cmcf_metrics import measure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        dest="input",
        type=Path,
        default=ROOT / "data/examples/cow.obj",
        help="input OBJ/PLY/OFF mesh",
    )
    parser.add_argument(
        "--output-file",
        dest="output",
        type=Path,
        default=ROOT / "data/examples/cow_spheremap.ply",
        help="output SphereMap PLY file",
    )
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--degree",
        type=int,
        default=4,
        choices=(1, 2, 3, 4),
        help="spherical harmonic degree",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_is_ply = args.output.suffix.lower() == ".ply"
    temporary = tempfile.TemporaryDirectory(prefix="spheremap-example-")
    sphere_map_output = args.output if output_is_ply else Path(temporary.name) / "spheremap_result.ply"
    job = SphereMap(
        iters=args.iters,
        step_size=1.0,
        threads=args.threads,
        degree=args.degree,
        c2i=0,
        a_steps=10,
        a_step_size=0.05,
        poincare_max_norm=2.0,
        gss_tolerance=1e-6,
        lump=False,
        no_center=False,
        verbose=True,
    ).run(args.input, sphere_map_output)
    result = job.result()
    metrics = measure(sphere_map_output)
    print("full metrics: ")
    for k,v in metrics.items():
        print(f"{k}: {v}")

    if not output_is_ply:
        try:
            extract_sphere(sphere_map_output, args.output)
        finally:
            temporary.cleanup()

    print(f"SphereMap return code: {result.returncode}")
    print(f"Generated output: {args.output}")
    print(
        "Collapsed triangles: "
        f"{metrics['collapsed_count']} / {metrics['face_count']} "
        f"({100.0 * metrics['collapsed_frac']:.6f}%)"
    )
    print(f"Minimum twice-area: {metrics['min_twice_area']:.9g}")
    print(f"Minimum triangle area: {metrics['min_area']:.9g}")
    print(f"Collapse threshold: {metrics['collapse_threshold']:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
