#!/usr/bin/env python3
"""Count area-collapsed triangles in an OBJ mesh."""

import argparse
import math
from pathlib import Path


def vertex_index(token: str, vertex_count: int) -> int:
    """Read the vertex part of an OBJ face token and return a zero-based index."""
    index = int(token.split("/", 1)[0])
    return index - 1 if index > 0 else vertex_count + index


def triangle_area(vertices, face):
    a, b, c = (vertices[index] for index in face)
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(value * value for value in cross))


def percentile(sorted_values, percentage):
    """Return a linearly interpolated percentile from sorted values."""
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def print_area_distribution(areas, bin_count):
    """Print summary statistics and a log-spaced histogram of triangle areas."""
    print("Triangle area distribution:")
    if not areas:
        print("  no triangles")
        return

    sorted_areas = sorted(areas)
    print(f"  min:    {sorted_areas[0]:.9g}")
    print(f"  p25:    {percentile(sorted_areas, 25):.9g}")
    print(f"  median: {percentile(sorted_areas, 50):.9g}")
    print(f"  p75:    {percentile(sorted_areas, 75):.9g}")
    print(f"  p95:    {percentile(sorted_areas, 95):.9g}")
    print(f"  max:    {sorted_areas[-1]:.9g}")
    print(f"  mean:   {math.fsum(areas) / len(areas):.9g}")

    zero_count = sum(area == 0.0 for area in areas)
    print("  histogram (log-spaced positive areas):")
    if zero_count:
        print(f"    area = 0: {zero_count} ({100.0 * zero_count / len(areas):.3f}%)")

    positive_areas = [area for area in areas if area > 0.0]
    if not positive_areas:
        return

    minimum = min(positive_areas)
    maximum = max(positive_areas)
    if minimum == maximum:
        print(
            f"    area = {minimum:.6g}: {len(positive_areas)} "
            f"({100.0 * len(positive_areas) / len(areas):.3f}%)"
        )
        return

    log_minimum = math.log(minimum)
    log_range = math.log(maximum) - log_minimum
    edges = [math.exp(log_minimum + log_range * index / bin_count) for index in range(bin_count + 1)]
    counts = [0] * bin_count
    for area in positive_areas:
        index = min(bin_count - 1, int((math.log(area) - log_minimum) / log_range * bin_count))
        counts[index] += 1

    for index, count in enumerate(counts):
        closing_bracket = "]" if index == bin_count - 1 else ")"
        print(
            f"    [{edges[index]:.6g}, {edges[index + 1]:.6g}{closing_bracket}: "
            f"{count} ({100.0 * count / len(areas):.3f}%)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj", type=Path, help="spherical OBJ file")
    parser.add_argument(
        "--area-tolerance",
        type=float,
        default=1e-12,
        help="triangles with area <= this value are collapsed (default: 1e-12)",
    )
    parser.add_argument(
        "--area-bins",
        type=int,
        default=10,
        help="number of log-spaced bins in the area histogram (default: 10)",
    )
    args = parser.parse_args()
    if args.area_tolerance < 0.0:
        parser.error("--area-tolerance must be non-negative")
    if args.area_bins < 1:
        parser.error("--area-bins must be at least 1")

    vertices = []
    triangles = []
    with args.obj.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif fields[0] == "f":
                face = [vertex_index(token, len(vertices)) for token in fields[1:]]
                if len(face) < 3:
                    raise ValueError(f"line {line_number}: face has fewer than 3 vertices")
                # OBJ supports polygons; triangulate them as a fan.
                for index in range(1, len(face) - 1):
                    triangles.append((face[0], face[index], face[index + 1]))

    collapsed = 0
    repeated_vertex = 0
    near_zero_area = 0
    areas = []
    for face in triangles:
        area = triangle_area(vertices, face)
        areas.append(area)
        if len(set(face)) < 3:
            repeated_vertex += 1
        if area <= args.area_tolerance:
            collapsed += 1
            if len(set(face)) == 3:
                near_zero_area += 1

    total = len(triangles)
    percentage = 100.0 * collapsed / total if total else 0.0
    print(f"Mesh: {args.obj}")
    print(f"Vertices: {len(vertices)}")
    print(f"Triangles: {total}")
    print(
        f"Area-collapsed triangles (area <= {args.area_tolerance:.9g}): "
        f"{collapsed} ({percentage:.6f}%)"
    )
    print(f"  repeated vertex index: {repeated_vertex}")
    print(f"  near-zero geometric area: {near_zero_area}")
    print_area_distribution(areas, args.area_bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
