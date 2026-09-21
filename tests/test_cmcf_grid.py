import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cmcf_metrics import measure
from run_cmcf_faust_grid import configs_stage1, configs_stage3_advection, cohorts, config_id, _parse_diag, rank, Runner, FIELDS


def ply(points, faces):
    lines = ["ply", "format ascii 1.0", f"element vertex {len(points)}", "property float px", "property float py", "property float pz", f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
    return "\n".join(lines + ["%.17g %.17g %.17g" % p for p in points] + ["3 %d %d %d" % f for f in faces]) + "\n"


class MetricsTests(unittest.TestCase):
    def test_unit_normalization_and_threshold_equality(self):
        points = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (-2, 0, 0)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.ply"; path.write_text(ply(points, [(0, 1, 2), (0, 2, 1)]))
            result = measure(path)
        self.assertEqual(result["vertex_count"], 4)
        self.assertEqual(result["collapsed_count"], 0)
        self.assertEqual(result["outward_count"], 1)
        self.assertEqual(result["inward_count"], 1)
        self.assertEqual(result["folded_count"], 1)

    def test_zero_area_and_topology(self):
        points = [(1, 0, 0), (1, 0, 0), (0, 1, 0)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.ply"; path.write_text(ply(points, [(0, 1, 2)]))
            result = measure(path)
        self.assertEqual(result["collapsed_count"], 1)
        self.assertEqual(result["min_twice_area"], 0.0)
        self.assertEqual(result["min_area"], 0.0)
        self.assertEqual(result["sub1deg_count"], 1)

    def test_input_orientation_is_reported_separately(self):
        points = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (-2, 0, 0)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.ply"; path.write_text(ply(points, [(0, 1, 2), (0, 2, 1)]))
            result = measure(path, path)
        self.assertEqual(result["input_outward_count"], 1)
        self.assertEqual(result["input_inward_count"], 1)
        self.assertEqual(result["input_folded_count"], 1)
        self.assertEqual(result["input_orientation_inconsistent_face_count"], 0)


class PlannerTests(unittest.TestCase):
    def test_cohorts_are_fixed_and_disjoint(self):
        c = cohorts(20260920)
        self.assertEqual(c["pilot"], ["007", "028", "049", "066", "085"])
        self.assertEqual(len(c["stage4"]), 20); self.assertEqual(len(c["stage5"]), 20)
        self.assertTrue(set(c["stage4"]).isdisjoint(c["stage5"]))
        self.assertFalse(set(c["stage4"]) & set(c["pilot"]))

    def test_configuration_counts(self):
        self.assertEqual(len(list(configs_stage1())), 25)
        base = next(configs_stage1())
        self.assertEqual(len(configs_stage3_advection(base, 2)), 8)
        self.assertEqual(config_id(base), config_id(dict(base)))

    def test_diagnostics_ranking_and_failure_exclusion(self):
        self.assertEqual(_parse_diag("CMCF[4] x: D-Norm=0.1 / QC-Ratio=1.2 / R-Deviation=0.3"), ("0.1", "1.2", "0.3"))
        good = {"config_id": "a", "return_code": "0", "error": "", "folded_count": "0", "collapsed_frac": ".1", "qc_ratio": "1.1", "wall_time_sec": "2", "iters": "25", "stepSize": ".1"}
        bad = dict(good, config_id="b", folded_count="1", collapsed_frac="0")
        self.assertEqual(rank([good, bad])[0][-1], "a")

    def test_resume_key_is_loaded(self):
        import argparse, csv
        with tempfile.TemporaryDirectory() as d:
            output = Path(d); path = output / "results.csv"
            with path.open("w", newline="") as f:
                csv.DictWriter(f, FIELDS).writeheader(); csv.DictWriter(f, FIELDS).writerow({"stage": "1", "config_id": "a", "subject": "007"})
            args = argparse.Namespace(output=output, input=output, binary=output / "missing", threads_per_run=4, jobs=1)
            runner = Runner(args, {})
            self.assertIn(("1", "a", "007"), runner.done)


if __name__ == "__main__":
    unittest.main()
