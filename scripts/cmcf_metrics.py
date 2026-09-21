#!/usr/bin/env python3
"""Metrics for SphereMap output PLY files.

The collapse rule intentionally mirrors the CEM validator: coordinates are
normalized to the unit sphere, twice-area is used, and the threshold is
max(1e-12, 1e-4 * median(twice-area)).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class PlyMesh:
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    properties: list[str]


def _read_ply(path: Path) -> PlyMesh:
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        marker = b"end_header\r\n"
        end = raw.find(marker)
    if end < 0:
        raise ValueError(f"{path}: missing PLY end_header")
    header = raw[:end].decode("ascii")
    body = raw[end + len(marker):]
    fmt = next((line.split()[1] for line in header.splitlines() if line.startswith("format ")), None)
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"{path}: unsupported PLY format {fmt!r}")
    vertices_n = next((int(line.split()[2]) for line in header.splitlines() if line.startswith("element vertex ")), None)
    faces_n = next((int(line.split()[2]) for line in header.splitlines() if line.startswith("element face ")), None)
    if vertices_n is None or faces_n is None:
        raise ValueError(f"{path}: vertex/face elements are required")
    vprops: list[tuple[str, str]] = []
    face_count_type, face_index_type = "uchar", "int"
    in_vertex = False
    for line in header.splitlines():
        fields = line.split()
        if fields[:2] == ["element", "vertex"]:
            in_vertex = True
        elif fields and fields[0] == "element":
            in_vertex = False
        elif fields[:2] == ["property", "list"] and len(fields) >= 5 and not in_vertex:
            face_count_type, face_index_type = fields[2], fields[3]
        elif in_vertex and fields[:1] == ["property"] and len(fields) >= 3 and fields[1] != "list":
            vprops.append((fields[1], fields[2]))
    names = [name for _, name in vprops]
    wanted = ["px", "py", "pz"] if all(x in names for x in ("px", "py", "pz")) else ["x", "y", "z"]
    indices = [names.index(x) for x in wanted]

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    if fmt == "ascii":
        lines = body.decode("ascii", errors="strict").splitlines()
        for line in lines[:vertices_n]:
            values = line.split()
            vertices.append(tuple(float(values[i]) for i in indices))
        for line in lines[vertices_n:vertices_n + faces_n]:
            values = line.split()
            count = int(values[0])
            if count != 3:
                raise ValueError(f"{path}: non-triangle face")
            faces.append(tuple(int(x) for x in values[1:4]))
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        types = {"float": "f", "float32": "f", "double": "d", "float64": "d",
                 "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
                 "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
                 "int": "i", "int32": "i", "uint": "I", "uint32": "I"}
        vfmt = endian + "".join(types[t] for t, _ in vprops)
        size = struct.calcsize(vfmt)
        offset = 0
        for _ in range(vertices_n):
            row = struct.unpack_from(vfmt, body, offset)
            offset += size
            vertices.append(tuple(float(row[i]) for i in indices))
        # PLY face list scalar types vary between the dataset and SphereMap.
        count_fmt = endian + types[face_count_type]
        index_fmt = endian + types[face_index_type]
        count_size = struct.calcsize(count_fmt); index_size = struct.calcsize(index_fmt)
        for _ in range(faces_n):
            count = struct.unpack_from(count_fmt, body, offset)[0]
            offset += count_size
            if count != 3:
                raise ValueError(f"{path}: non-triangle face")
            face = tuple(struct.unpack_from(index_fmt, body, offset + i * index_size)[0] for i in range(3))
            faces.append(face)
            offset += 3 * index_size
    return PlyMesh(vertices, faces, names)


def _twice_area(a, b, c) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
    return math.sqrt(sum(x * x for x in cross))


def _unit(v):
    length = math.sqrt(sum(x * x for x in v))
    if not math.isfinite(length) or length == 0:
        raise ValueError("non-finite or zero spherical coordinate")
    return tuple(x / length for x in v)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    p = (len(values) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] if lo == hi else values[lo] * (hi - p) + values[hi] * (p - lo)


def minimum_angle_degrees(vertices, face) -> float:
    points = [vertices[i] for i in face]
    angles = []
    for i in range(3):
        a, b, c = points[i], points[(i + 1) % 3], points[(i + 2) % 3]
        u = tuple(b[d] - a[d] for d in range(3)); v = tuple(c[d] - a[d] for d in range(3))
        den = math.sqrt(sum(x*x for x in u) * sum(x*x for x in v))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, sum(u[d]*v[d] for d in range(3)) / den)))) if den else 0.0)
    return min(angles)


def _orientation_counts(vertices, faces, reference=(0.0, 0.0, 0.0)):
    """Classify face orientation relative to a mesh reference point."""
    signs = []
    eps = 1e-12
    for a, b, c in faces:
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        cross = ((pb[1]-pa[1])*(pc[2]-pa[2])-(pb[2]-pa[2])*(pc[1]-pa[1]),
                 (pb[2]-pa[2])*(pc[0]-pa[0])-(pb[0]-pa[0])*(pc[2]-pa[2]),
                 (pb[0]-pa[0])*(pc[1]-pa[1])-(pb[1]-pa[1])*(pc[0]-pa[0]))
        center = tuple((pa[d] + pb[d] + pc[d]) / 3 for d in range(3))
        signs.append(sum(cross[d] * (center[d] - reference[d]) for d in range(3)))
    outward = sum(x > eps for x in signs)
    inward = sum(x < -eps for x in signs)
    near = len(signs) - outward - inward
    return {"outward_count": outward, "inward_count": inward,
            "folded_count": min(outward, inward), "near_zero_orientation_count": near}


def _edge_orientation_inconsistencies(faces):
    """Count faces participating in same-direction shared-edge windings."""
    edges = {}
    for face_index, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            direction = 1 if (u, v) == key else -1
            edges.setdefault(key, []).append((face_index, direction))
    bad_faces = set()
    for uses in edges.values():
        for i, (face_i, direction_i) in enumerate(uses):
            for face_j, direction_j in uses[i + 1:]:
                if direction_i == direction_j:
                    bad_faces.update((face_i, face_j))
    return len(bad_faces)


def measure(output: Path, input_mesh: Path | None = None) -> dict:
    result = {"output_path": str(output)}
    out = _read_ply(output)
    result.update(vertex_count=len(out.vertices), face_count=len(out.faces), nonfinite_count=0,
                  topology_ok=True, collapsed_count=0, collapsed_frac=float("nan"),
                  min_twice_area=float("nan"), min_area=float("nan"),
                  sub1deg_count=0, sub1deg_frac=float("nan"), min_angle_deg=float("nan"),
                  outward_count=0, inward_count=0, folded_count=0, near_zero_orientation_count=0)
    result.update(input_outward_count=None, input_inward_count=None,
                  input_folded_count=None, input_near_zero_orientation_count=None,
                  input_orientation_inconsistent_face_count=None)
    if input_mesh is not None:
        inp = _read_ply(input_mesh)
        result["topology_ok"] = len(inp.vertices) == len(out.vertices) and inp.faces == out.faces
        result["input_orientation_inconsistent_face_count"] = _edge_orientation_inconsistencies(inp.faces)
        finite_input = all(all(math.isfinite(x) for x in point) for point in inp.vertices)
        if finite_input and all(name in inp.properties for name in ("px", "py", "pz")):
            input_reference = tuple(sum(point[d] for point in inp.vertices) / len(inp.vertices) for d in range(3))
            input_orientation = _orientation_counts(inp.vertices, inp.faces, input_reference)
            result.update({f"input_{key}": value for key, value in input_orientation.items()})
    normalized = []
    for point in out.vertices:
        try:
            normalized.append(_unit(point))
        except ValueError:
            result["nonfinite_count"] += 1
            normalized.append((float("nan"),) * 3)
    if result["nonfinite_count"]:
        return result
    areas = [_twice_area(normalized[a], normalized[b], normalized[c]) for a, b, c in out.faces]
    median = _percentile(areas, 0.5)
    threshold = max(1e-12, 1e-4 * median)
    collapsed = [a <= threshold for a in areas]
    angles = [minimum_angle_degrees(normalized, f) for f in out.faces]
    output_orientation = _orientation_counts(normalized, out.faces)
    result.update(median_twice_area=median, collapse_threshold=threshold, collapsed_count=sum(collapsed),
                  collapsed_frac=sum(collapsed) / len(areas) if areas else 0.0,
                  min_twice_area=min(areas) if areas else float("nan"),
                  min_area=min(areas) / 2.0 if areas else float("nan"),
                  sub1deg_count=sum(x < 1.0 for x in angles), sub1deg_frac=sum(x < 1.0 for x in angles) / len(angles) if angles else 0.0,
                  min_angle_deg=min(angles) if angles else float("nan"), **output_orientation)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path, help="input PLY for topology validation")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = measure(args.output, args.input)
    print(json.dumps(result, sort_keys=True) if args.json else "\n".join(f"{k}: {v}" for k, v in result.items()))
    return 0 if result["topology_ok"] and not result["nonfinite_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
