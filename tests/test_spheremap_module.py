import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spheremap import SphereMap, SphereMapError, extract_sphere


class SphereMapModuleTests(unittest.TestCase):
    def test_parameter_mapping(self):
        wrapper = SphereMap(binary="/tmp/SphereMap", iters=7, step_size=0.03, no_center=True, threads=2)
        command = wrapper._command(Path("/tmp/SphereMap"), Path("in.ply"), Path("out.ply"))
        self.assertEqual(command[0:5], ["/tmp/SphereMap", "--in", "in.ply", "--out", "out.ply"])
        self.assertIn("--iters", command); self.assertIn("7", command)
        self.assertIn("--stepSize", command); self.assertIn("--noCenter", command)

    def test_unknown_parameter(self):
        with self.assertRaises(TypeError): SphereMap(foo=1)

    def test_missing_input(self):
        with self.assertRaises(FileNotFoundError): SphereMap(binary="/tmp/SphereMap", auto_build=False).run("missing.ply", "x.ply")

    def test_extract_sphere(self):
        source = Path(__file__).resolve().parents[1] / "data/tr_reg_001_default.ply"
        with tempfile.TemporaryDirectory() as directory:
            output = extract_sphere(source, Path(directory) / "sphere.obj")
            self.assertTrue(output.exists())
            self.assertTrue(output.read_text().splitlines()[1].startswith("v "))


if __name__ == "__main__": unittest.main()
