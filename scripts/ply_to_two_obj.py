#!/usr/bin/env python3
"""Export a spherical PLY as two OBJ meshes.

The input PLY must contain vertex properties x, y, z and px, py, pz.
The script writes the same faces twice:

* an OBJ using the original x, y, z coordinates;
* an OBJ using the spherical px, py, pz coordinates.

The implementation uses only the Python standard library and supports ASCII
and binary little/big-endian PLY files.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import BinaryIO, Dict, List, Sequence, Tuple


SCALAR_TYPES = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


class PlyError(ValueError):
    pass


def parse_header(stream: BinaryIO) -> Tuple[str, Dict[str, object]]:
    lines: List[str] = []
    while True:
        line = stream.readline()
        if not line:
            raise PlyError("PLY header ended before end_header")
        decoded = line.decode("ascii").strip()
        lines.append(decoded)
        if decoded == "end_header":
            break

    if not lines or lines[0] != "ply":
        raise PlyError("input is not a PLY file")

    format_name = None
    elements: List[Dict[str, object]] = []
    current = None
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0] in {"comment", "obj_info"}:
            continue
        if fields[0] == "format":
            format_name = fields[1]
        elif fields[0] == "element":
            current = {"name": fields[1], "count": int(fields[2]), "properties": []}
            elements.append(current)
        elif fields[0] == "property":
            if current is None:
                raise PlyError("property appears before an element declaration")
            properties = current["properties"]
            assert isinstance(properties, list)
            if fields[1] == "list":
                if len(fields) != 5:
                    raise PlyError(f"invalid list property: {line}")
                properties.append(("list", fields[2], fields[3], fields[4]))
            else:
                if len(fields) != 3:
                    raise PlyError(f"invalid scalar property: {line}")
                properties.append(("scalar", fields[1], fields[2]))

    if format_name is None:
        raise PlyError("PLY header has no format declaration")
    return format_name, {"elements": elements}


def scalar_value(stream: BinaryIO, type_name: str, endian: str):
    try:
        code, size = SCALAR_TYPES[type_name]
    except KeyError as exc:
        raise PlyError(f"unsupported PLY scalar type: {type_name}") from exc
    data = stream.read(size)
    if len(data) != size:
        raise PlyError("unexpected end of PLY data")
    return struct.unpack(endian + code, data)[0]


def read_ascii_value(token: str, type_name: str):
    if type_name in {"float", "float32", "double", "float64"}:
        return float(token)
    return int(token)


def read_ply(path: Path) -> Tuple[List[Dict[str, float]], List[List[int]]]:
    with path.open("rb") as stream:
        format_name, header = parse_header(stream)
        if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise PlyError(f"unsupported PLY format: {format_name}")

        elements = header["elements"]
        assert isinstance(elements, list)
        vertices: List[Dict[str, float]] = []
        faces: List[List[int]] = []

        if format_name == "ascii":
            lines = iter(stream.readline, b"")
            for element in elements:
                name = element["name"]
                count = element["count"]
                properties = element["properties"]
                assert isinstance(name, str) and isinstance(count, int)
                assert isinstance(properties, list)
                for _ in range(count):
                    line = next(lines, b"")
                    if not line:
                        raise PlyError("unexpected end of ASCII PLY data")
                    tokens = line.decode("ascii").split()
                    cursor = 0
                    record: Dict[str, object] = {}
                    for prop in properties:
                        if prop[0] == "scalar":
                            _, type_name, prop_name = prop
                            record[prop_name] = read_ascii_value(tokens[cursor], type_name)
                            cursor += 1
                        else:
                            _, count_type, item_type, prop_name = prop
                            list_count = int(read_ascii_value(tokens[cursor], count_type))
                            cursor += 1
                            record[prop_name] = [
                                read_ascii_value(tokens[cursor + i], item_type)
                                for i in range(list_count)
                            ]
                            cursor += list_count
                    if name == "vertex":
                        vertices.append({key: float(record[key]) for key in record if key in {"x", "y", "z", "px", "py", "pz"}})
                    elif name == "face":
                        index_values = record.get("vertex_indices", record.get("vertex_index"))
                        if index_values is not None:
                            faces.append([int(index) for index in index_values])
        else:
            endian = "<" if format_name == "binary_little_endian" else ">"
            for element in elements:
                name = element["name"]
                count = element["count"]
                properties = element["properties"]
                assert isinstance(name, str) and isinstance(count, int)
                assert isinstance(properties, list)
                for _ in range(count):
                    record: Dict[str, object] = {}
                    for prop in properties:
                        if prop[0] == "scalar":
                            _, type_name, prop_name = prop
                            record[prop_name] = scalar_value(stream, type_name, endian)
                        else:
                            _, count_type, item_type, prop_name = prop
                            list_count = int(scalar_value(stream, count_type, endian))
                            record[prop_name] = [
                                scalar_value(stream, item_type, endian) for _ in range(list_count)
                            ]
                    if name == "vertex":
                        vertices.append({key: float(record[key]) for key in record if key in {"x", "y", "z", "px", "py", "pz"}})
                    elif name == "face":
                        index_values = record.get("vertex_indices", record.get("vertex_index"))
                        if index_values is not None:
                            faces.append([int(index) for index in index_values])

    if not vertices:
        raise PlyError("PLY contains no vertex element")
    required = {"x", "y", "z", "px", "py", "pz"}
    missing = required - set(vertices[0])
    if missing:
        raise PlyError("PLY is missing required vertex properties: " + ", ".join(sorted(missing)))
    if not faces:
        raise PlyError("PLY contains no faces")
    return vertices, faces


def write_obj(path: Path, vertices: Sequence[Dict[str, float]], faces: Sequence[Sequence[int]], coordinate_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"# Generated from a PLY mesh; coordinates: {', '.join(coordinate_names)}\n")
        for vertex in vertices:
            stream.write("v " + " ".join(f"{vertex[name]:.9g}" for name in coordinate_names) + "\n")
        for face in faces:
            if len(face) < 3:
                continue
            stream.write("f " + " ".join(str(index + 1) for index in face) + "\n")


def default_output_paths(input_path: Path) -> Tuple[Path, Path]:
    return (
        input_path.with_name(input_path.stem + "_original.obj"),
        input_path.with_name(input_path.stem + "_spherical.obj"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input PLY containing x,y,z and px,py,pz")
    parser.add_argument("--original-out", type=Path, help="OBJ output using x,y,z")
    parser.add_argument("--spherical-out", type=Path, help="OBJ output using px,py,pz")
    args = parser.parse_args()

    original_default, spherical_default = default_output_paths(args.input)
    original_out = args.original_out or original_default
    spherical_out = args.spherical_out or spherical_default

    vertices, faces = read_ply(args.input)
    write_obj(original_out, vertices, faces, ("x", "y", "z"))
    write_obj(spherical_out, vertices, faces, ("px", "py", "pz"))
    print(f"Wrote {len(vertices)} vertices and {len(faces)} faces")
    print(f"Original mesh:  {original_out}")
    print(f"Spherical mesh: {spherical_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
