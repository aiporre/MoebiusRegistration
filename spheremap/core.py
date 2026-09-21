"""Background Python wrapper for the repository's SphereMap binary."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import trimesh
except ImportError:  # pragma: no cover - packaging/install error is clearer at use time
    trimesh = None


class SphereMapError(RuntimeError):
    """Raised when conversion, execution, or output validation fails."""


@dataclass(frozen=True)
class SphereMapResult:
    returncode: int
    output: Path
    ply_output: Path | None
    stdout: str
    stderr: str
    duration_sec: float
    command: tuple[str, ...]


class SphereMapJob:
    """A running SphereMap process.

    ``run`` starts the process immediately. Use ``wait``/``result`` to join it,
    or ``poll`` to inspect it without blocking.
    """

    def __init__(self, process: subprocess.Popen[str], output: Path, ply_output: Path | None,
                 command: tuple[str, ...], cleanup: tempfile.TemporaryDirectory[str] | None,
                 started: float, postprocess=None):
        self._process = process
        self.output = output
        self.ply_output = ply_output
        self.command = command
        self._cleanup = cleanup
        self._started = started
        self._postprocess = postprocess
        self._stdout = ""
        self._stderr = ""
        self._result: SphereMapResult | None = None
        self._lock = threading.Lock()

    @property
    def stdout(self) -> str:
        return self._stdout

    @property
    def stderr(self) -> str:
        return self._stderr

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    def poll(self) -> int | None:
        return self._process.poll()

    def cancel(self) -> bool:
        if self._process.poll() is not None:
            return False
        self._process.terminate()
        return True

    def wait(self, timeout: float | None = None) -> SphereMapResult:
        with self._lock:
            if self._result is not None:
                return self._result
        try:
            stdout, stderr = self._process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise
        self._stdout, self._stderr = stdout, stderr
        result_ply = self.ply_output if self.output.suffix.lower() == ".ply" else None
        result = SphereMapResult(self._process.returncode, self.output, result_ply,
                                 stdout, stderr, time.monotonic() - self._started, self.command)
        if result.returncode != 0:
            if self._cleanup is not None:
                self._cleanup.cleanup(); self._cleanup = None
            raise SphereMapError(f"SphereMap exited with code {result.returncode}: {stderr[-1000:]}")
        process_output = self.ply_output or self.output
        if not process_output.exists():
            if self._cleanup is not None:
                self._cleanup.cleanup(); self._cleanup = None
            raise SphereMapError(f"SphereMap completed without creating {self.output}")
        if self._postprocess is not None:
            try:
                self._postprocess(process_output, self.output)
            except Exception:
                if self._cleanup is not None:
                    self._cleanup.cleanup(); self._cleanup = None
                raise
        with self._lock:
            self._result = result
        if self._cleanup is not None:
            self._cleanup.cleanup(); self._cleanup = None
        return result

    result = wait


_PARAMETERS = {
    "iters": ("iters", int), "step_size": ("stepSize", float), "threads": ("threads", int),
    "resolution": ("res", int), "degree": ("degree", int), "a_steps": ("aSteps", int),
    "mesh": ("mesh", int), "fill": ("fill", int), "cut_off": ("cutOff", float),
    "smooth": ("smooth", float), "a_step_size": ("aStepSize", float), "gss_tolerance": ("gssTolerance", float),
    "poincare_max_norm": ("poincareMaxNorm", float), "c2i": ("c2i", int),
}
_FLAGS = {
    "verbose": "verbose", "full_verbose": "fullVerbose", "ascii": "ascii", "randomize": "random",
    "no_center": "noCenter", "collapse": "collapse", "no_orient": "noOrient",
    "no_grid_scale": "noGridScale", "lump": "lump", "spherical": "spherical",
}


def _scalar(code: str, endian: str):
    import struct
    return struct.calcsize(endian + code), endian + code


def _read_result_ply(path: Path):
    """Read the result fields needed to export spherical OBJ/OFF."""
    import struct
    raw = path.read_bytes()
    marker = b"end_header\n"
    pos = raw.find(marker)
    if pos < 0:
        raise SphereMapError(f"{path}: unsupported PLY header")
    header = raw[:pos].decode("ascii").splitlines()
    body = raw[pos + len(marker):]
    fmt = next(x.split()[1] for x in header if x.startswith("format "))
    endian = "<" if fmt == "binary_little_endian" else ">" if fmt == "binary_big_endian" else ""
    vn = next(int(x.split()[2]) for x in header if x.startswith("element vertex "))
    fn = next(int(x.split()[2]) for x in header if x.startswith("element face "))
    names, types, face_count, face_index = [], [], "uchar", "int"; in_v = False
    for line in header:
        f = line.split()
        if f[:2] == ["element", "vertex"]: in_v = True
        elif f and f[0] == "element": in_v = False
        elif in_v and f[:1] == ["property"] and f[1] != "list": types.append(f[1]); names.append(f[2])
        elif f[:2] == ["property", "list"]: face_count, face_index = f[2], f[3]
    wanted = [names.index(x) for x in ("px", "py", "pz")]
    codes = {"char":"b", "uchar":"B", "short":"h", "ushort":"H", "int":"i", "uint":"I", "float":"f", "double":"d"}
    vertices, faces = [], []
    if fmt == "ascii":
        lines = body.decode("ascii").splitlines()
        vertices = [tuple(float(lines[i].split()[j]) for j in wanted) for i in range(vn)]
        for line in lines[vn:vn + fn]:
            f = line.split(); faces.append(tuple(map(int, f[1:4])))
    else:
        vf = endian + "".join(codes[t] for t in types); vs = struct.calcsize(vf); off = 0
        for _ in range(vn):
            row = struct.unpack_from(vf, body, off); off += vs; vertices.append(tuple(float(row[i]) for i in wanted))
        cf, inf = endian + codes[face_count], endian + codes[face_index]; cs, is_ = struct.calcsize(cf), struct.calcsize(inf)
        for _ in range(fn):
            count = struct.unpack_from(cf, body, off)[0]; off += cs
            if count != 3: raise SphereMapError("only triangle PLY output is supported")
            faces.append(tuple(struct.unpack_from(inf, body, off + i * is_)[0] for i in range(3))); off += count * is_
    return vertices, faces


def extract_sphere(input: str | os.PathLike[str], output: str | os.PathLike[str]) -> Path:
    """Extract ``px,py,pz`` from a SphereMap PLY into OBJ/OFF/PLY output.

    The output contains the same face connectivity as the input PLY, but its
    vertex coordinates are the spherical coordinates produced by SphereMap.
    """
    if trimesh is None:
        raise SphereMapError("trimesh is required; install it with `pip install .`")
    source = Path(input).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    try:
        vertices, faces = _read_result_ply(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(destination)
    except SphereMapError:
        raise
    except Exception as exc:
        raise SphereMapError(f"could not extract spherical coordinates from {source}") from exc
    return destination


class SphereMap:
    """Configure and launch the compiled SphereMap executable."""

    def __init__(self, binary: str | os.PathLike[str] | None = None, auto_build: bool = True, **parameters: Any):
        self.binary = Path(binary).expanduser().resolve() if binary else None
        self.auto_build = auto_build
        unknown = set(parameters) - set(_PARAMETERS) - set(_FLAGS)
        if unknown:
            raise TypeError(f"unknown SphereMap parameters: {', '.join(sorted(unknown))}")
        self.parameters = dict(parameters)

    def _binary(self) -> Path:
        candidates = []
        if self.binary: candidates.append(self.binary)
        if os.environ.get("SPHEREMAP_BINARY"): candidates.append(Path(os.environ["SPHEREMAP_BINARY"]))
        root = Path(__file__).resolve().parents[1]
        candidates += [Path(__file__).resolve().parent / "_bin" / "SphereMap", root / "Bin/Linux/SphereMap"]
        for candidate in candidates:
            if candidate.exists() and os.access(candidate, os.X_OK): return candidate.resolve()
        if not self.auto_build: raise SphereMapError("SphereMap binary not found; pass binary=... or enable auto_build")
        try:
            subprocess.run(["make", "-j2", "Bin/Linux/SphereMap"], cwd=root, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SphereMapError("could not build SphereMap; install a C++ compiler, FFTW, and OpenMP") from exc
        candidate = root / "Bin/Linux/SphereMap"
        if not candidate.exists(): raise SphereMapError("build completed but SphereMap binary is missing")
        return candidate.resolve()

    def _command(self, binary: Path, input_path: Path, output_path: Path) -> list[str]:
        command = [str(binary), "--in", str(input_path), "--out", str(output_path)]
        for name, (flag, caster) in _PARAMETERS.items():
            if name in self.parameters:
                command += [f"--{flag}", str(caster(self.parameters[name]))]
        for name, flag in _FLAGS.items():
            if self.parameters.get(name, False): command.append(f"--{flag}")
        return command

    def run(self, input: str | os.PathLike[str], output: str | os.PathLike[str]) -> SphereMapJob:
        source = Path(input).expanduser().resolve(); destination = Path(output).expanduser().resolve()
        if not source.exists(): raise FileNotFoundError(source)
        if trimesh is None: raise SphereMapError("trimesh is required; install it with `pip install .`")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="spheremap-")
        work = Path(temporary.name); input_ply = source
        if source.suffix.lower() != ".ply":
            try:
                mesh = trimesh.load(source, force="mesh", process=False)
                input_ply = work / "input.ply"; mesh.export(input_ply)
            except Exception as exc:
                temporary.cleanup(); raise SphereMapError(f"could not convert {source} to PLY") from exc
        output_is_ply = destination.suffix.lower() == ".ply"
        ply_output = destination if output_is_ply else work / "result.ply"
        command = tuple(self._command(self._binary(), input_ply, ply_output))
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception:
            temporary.cleanup(); raise
        postprocess = None
        if not output_is_ply:
            def export_spherical(ply_path, output_path):
                try:
                    vertices, faces = _read_result_ply(ply_path)
                    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(output_path)
                except Exception as exc:
                    raise SphereMapError(f"could not export spherical result to {output_path}") from exc
            postprocess = export_spherical
        cleanup = temporary if (source.suffix.lower() != ".ply" or not output_is_ply) else None
        return SphereMapJob(process, destination, ply_output, command, cleanup, time.monotonic(), postprocess)
