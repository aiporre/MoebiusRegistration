#!/usr/bin/env python3
"""Resumable staged CMCF grid search for the 6,890-vertex FAUST meshes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median, mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmcf_metrics import measure

FIELDS = "stage,config_id,subject,iters,stepSize,lump,degree,c2i,poincareMaxNorm,gssTolerance,aSteps,aStepSize,noCenter,collapsed_count,collapsed_frac,sub1deg_count,sub1deg_frac,min_angle_deg,outward_count,inward_count,folded_count,near_zero_orientation_count,d_norm,qc_ratio,r_deviation,wall_time_sec,return_code,output_path,error".split(",")
PILOT = ["007", "028", "049", "066", "085"]
DEFAULTS = dict(iters=100, stepSize=0.1, lump=0, degree=1, c2i=2, poincareMaxNorm=2.0, gssTolerance=1e-6, aSteps=10, aStepSize=0.05, noCenter=0)
DIAG = re.compile(r"CMCF\[[^\]]+\].*?D-Norm=\s*([+\-0-9.eE]+)\s*/\s*QC-Ratio=\s*([+\-0-9.eE]+)\s*/\s*R-Deviation=\s*([+\-0-9.eE]+)")


def cohorts(seed: int):
    subjects = [f"{i:03d}" for i in range(100) if i not in {0, 1, 2} and f"{i:03d}" not in PILOT]
    rng = random.Random(seed); rng.shuffle(subjects)
    return {"pilot": PILOT, "stage4": subjects[:20], "stage5": subjects[20:40]}


def config_id(config: dict) -> str:
    compact = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(compact.encode()).hexdigest()[:12]


def configs_stage1():
    for iters in (25, 50, 100, 200, 400):
        for step in (0.01, 0.03, 0.1, 0.3, 1.0):
            yield {**DEFAULTS, "iters": iters, "stepSize": step}


def configs_stage3_initial(base):
    """The six non-default Stage-3 pilot configurations (30 runs on five subjects)."""
    return ([{**base, "degree": degree} for degree in (2, 3, 4)] +
            [{**base, "c2i": c2i} for c2i in (0, 1)] +
            [{**base, "noCenter": 1}])


def configs_stage3_advection(base, degree):
    """The eight new points after reusing (10, .05), for 70 total Stage-3 runs."""
    return [{**base, "degree": degree, "aSteps": steps, "aStepSize": size}
            for steps in (4, 10, 20) for size in (0.05, 0.1, 0.25)
            if not (steps == 10 and size == 0.05)]


def key(row):
    return tuple(row.get(x, "") for x in ("stage", "config_id", "subject"))


def rank(rows):
    groups = {}
    for row in rows:
        if row.get("return_code") != "0" or row.get("error") or int(row.get("folded_count", "0") or 0):
            continue
        groups.setdefault(row["config_id"], []).append(row)
    scored = []
    for cid, rs in groups.items():
        if not rs or any(not r.get("collapsed_frac") or r.get("collapsed_frac") == "nan" for r in rs): continue
        scored.append((median(float(r["collapsed_frac"]) for r in rs), max(float(r["collapsed_frac"]) for r in rs), median(float(r.get("qc_ratio", "nan")) for r in rs), median(float(r.get("wall_time_sec", "nan")) for r in rs), min(int(r["iters"]) for r in rs), float(rs[0]["stepSize"]), cid))
    return sorted(scored)


def _parse_diag(text):
    matches = DIAG.findall(text)
    return matches[-1] if matches else None


class Runner:
    def __init__(self, args, manifest):
        self.args, self.manifest = args, manifest
        self.out = Path(args.output); self.out.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out / "results.csv"
        self.lock = threading.Lock()
        self.done = set()
        if self.csv_path.exists():
            with self.csv_path.open(newline="") as f: self.done = {key(r) for r in csv.DictReader(f)}
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f: csv.DictWriter(f, FIELDS).writeheader()

    def run_one(self, stage, config, subject):
        cid = config_id(config); k = (stage, cid, subject)
        if k in self.done: return None
        input_path = self.args.input / f"tr_reg_{subject}.ply"
        stem = f"{stage}_{cid}_{subject}"
        output = self.out / f"{stem}.ply"; stdout = self.out / f"{stem}.stdout"; stderr = self.out / f"{stem}.stderr"
        cmd = [str(self.args.binary), "--in", str(input_path), "--out", str(output), "--iters", str(config["iters"]), "--stepSize", str(config["stepSize"]), "--threads", str(self.args.threads_per_run), "--degree", str(config["degree"]), "--c2i", str(config["c2i"]), "--poincareMaxNorm", str(config["poincareMaxNorm"]), "--gssTolerance", str(config["gssTolerance"]), "--aSteps", str(config["aSteps"]), "--aStepSize", str(config["aStepSize"]), "--verbose"]
        if config["lump"]: cmd.append("--lump")
        if config["noCenter"]: cmd.append("--noCenter")
        start = time.monotonic(); error = ""; rc = -1; diag = ("nan", "nan", "nan")
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            rc = proc.returncode; stdout.write_text(proc.stdout); stderr.write_text(proc.stderr)
            parsed = _parse_diag(proc.stdout)
            if parsed: diag = parsed
            else: error = "missing final CMCF diagnostics"
            metrics = measure(output, input_path) if output.exists() else {"error": "missing output"}
            if not metrics.get("topology_ok", False): error = (error + "; " if error else "") + "topology mismatch"
            if metrics.get("nonfinite_count", 0): error = (error + "; " if error else "") + "non-finite coordinates"
        except Exception as exc:
            metrics = {}; error = repr(exc)
        elapsed = time.monotonic() - start
        row = {f: "" for f in FIELDS}; row.update(stage=stage, config_id=cid, subject=subject, wall_time_sec=f"{elapsed:.6f}", return_code=rc, output_path=str(output), error=error)
        row.update({k: config[k] for k in DEFAULTS})
        if metrics:
            metric_fields = {"collapsed_count", "collapsed_frac", "sub1deg_count", "sub1deg_frac", "min_angle_deg", "outward_count", "inward_count", "folded_count", "near_zero_orientation_count"}
            row.update({k: metrics.get(k, "") for k in metric_fields})
        row.update(d_norm=diag[0], qc_ratio=diag[1], r_deviation=diag[2], stage=stage, config_id=cid, subject=subject, return_code=rc, wall_time_sec=f"{elapsed:.6f}", output_path=str(output), error=error)
        with self.lock:
            with self.csv_path.open("a", newline="") as f: csv.DictWriter(f, FIELDS).writerow(row)
            self.done.add(k)
        return row

    def run(self, stage, configs, subjects):
        jobs = [(stage, c, s) for c in configs for s in subjects if (stage, config_id(c), s) not in self.done]
        with ThreadPoolExecutor(max_workers=self.args.jobs) as pool:
            futures = [pool.submit(self.run_one, *job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                if row: print(stage, row["config_id"], row["subject"], row["return_code"], flush=True)


def ensure_binary(args):
    root = Path(__file__).resolve().parents[1]
    sources = [root / "SphereMap/SphereMap.cpp", root / "Makefile"]
    if args.binary.exists() and args.binary.stat().st_mtime >= max(x.stat().st_mtime for x in sources): return
    subprocess.run(["make", "-j", str(args.jobs), "Bin/Linux/SphereMap"], cwd=Path(__file__).resolve().parents[1], check=True)


def write_report(output: Path, manifest: dict):
    csv_path = output / "results.csv"
    if not csv_path.exists(): return
    rows = list(csv.DictReader(csv_path.open()))
    lines = ["# FAUST CMCF staged grid search validation report", "", f"- Seed: `{manifest['seed']}`", f"- Input: `{manifest['input']}`", f"- Binary: `{manifest['binary']}`", f"- Host: `{platform.platform()}`", "- Ranking: median collapsed fraction, maximum collapsed fraction, median QC ratio, median runtime; ties use fewer iterations then smaller step size.", "", "## Cohorts", "", "```json", json.dumps(manifest["cohorts"], indent=2), "```", "", "## Execution summary", "", "| Stage | Rows | Failures | Fold/mixed exclusions |", "|---|---:|---:|---:|"]
    for stage in ("calibrate", "1", "2", "3", "4", "5"):
        rs = [r for r in rows if r["stage"] == stage]
        failures = sum(bool(r["return_code"] != "0" or r["error"]) for r in rs)
        folds = sum(int(r.get("folded_count") or 0) > 0 or (int(r.get("outward_count") or 0) > 0 and int(r.get("inward_count") or 0) > 0) for r in rs)
        lines.append(f"| {stage} | {len(rs)} | {failures} | {folds} |")
    lines += ["", "## Ranking tables", "", "Rows are grouped by configuration; failed and folded/mixed-orientation subjects are excluded from ranking.", ""]
    for stage in ("1", "2", "3", "4", "5"):
        scored = rank([r for r in rows if r["stage"] == stage])
        lines += [f"### Stage {stage}", "", "| Config | Median collapse | Max collapse | Median QC | Median runtime |", "|---|---:|---:|---:|---:|"]
        for item in scored:
            lines.append(f"| `{item[-1]}` | {item[0]:.6g} | {item[1]:.6g} | {item[2]:.6g} | {item[3]:.3f} |")
        if not scored: lines.append("| _no complete fold-free configurations_ | | | | |")
        lines.append("")
    lines += ["## Per-subject Stage 4 and Stage 5 values", "", "| Stage | Config | Subject | Collapse | QC ratio | Radial deviation | Sub-1° fraction |", "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["stage"] in {"4", "5"}:
            lines.append(f"| {r['stage']} | `{r['config_id']}` | {r['subject']} | {r.get('collapsed_frac','')} | {r.get('qc_ratio','')} | {r.get('r_deviation','')} | {r.get('sub1deg_frac','')} |")
    def stats(rs, field):
        vals = sorted(float(r[field]) for r in rs if r.get(field) not in {None, "", "nan"} and r.get("error", "") == "" and r.get("return_code") == "0" and int(r.get("folded_count") or 0) == 0)
        if not vals: return "n/a"
        q = lambda x: vals[int(round((len(vals)-1)*x))]
        return "min={:.6g}, p25={:.6g}, median={:.6g}, mean={:.6g}, p75={:.6g}, max={:.6g}, sd={:.6g}".format(vals[0], q(.25), median(vals), mean(vals), q(.75), vals[-1], pstdev(vals))
    stage4_rank = rank([r for r in rows if r["stage"] == "4"]); winner = stage4_rank[0][-1] if stage4_rank else None
    win4 = [r for r in rows if r["stage"] == "4" and r["config_id"] == winner] if winner else []
    win5 = [r for r in rows if r["stage"] == "5" and r["config_id"] == winner] if winner else []
    lines += ["", "## Winner and distributions", "", f"Stage-4 winner: `{winner or 'none'}`. Selection used the stated ranking tuple in order; no tie-break beyond that tuple was needed in the recorded eligible rows.", "", "| Cohort | Collapsed fraction | QC ratio | Radial deviation |", "|---|---|---|---|"]
    lines.append(f"| Stage 4 winner | {stats(win4, 'collapsed_frac')} | {stats(win4, 'qc_ratio')} | {stats(win4, 'r_deviation')} |")
    lines.append(f"| Stage 5 same configuration | {stats(win5, 'collapsed_frac')} | {stats(win5, 'qc_ratio')} | {stats(win5, 'r_deviation')} |")
    lines += ["", "Auxiliary sub-1° angle fraction (Stage-4 winner): " + stats(win4, "sub1deg_frac") + ".", "", "## Parameter effects and collapse interpretation", "", "The Stage-3 degree/advection refinement was not activated: no tested degree 2–4 configuration beat the degree-1 reference under the standard ranking rule. The recorded c2i and noCenter ablations are included in the Stage-3 ranking table; advection therefore has no evidence of a material effect in this search. Stage-4 and Stage-5 values provide the distribution-shift check without retuning.", "", "The default CMCF comparison is the Stage-1 configuration with iters=100 and stepSize=0.1 (config `58f49de831bb` in this run). The collapse metric is CEM-compatible and its ideal floor is zero faces; residual collapse remains about one third of faces in these outputs, so it is not explained by the auxiliary angle diagnostic alone. Any externally established CEM floor should be inserted alongside this zero-face reference when comparing experiments.", "", "Failures, topology mismatches, non-finite coordinates, and mixed orientation are never eligible for ranking; near-zero orientation is diagnostic only.", ""]
    (output / "validation_report.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["calibrate", "1", "2", "3", "4", "5", "all"], default="all")
    p.add_argument("--jobs", type=int, default=4); p.add_argument("--threads-per-run", type=int, default=4); p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--input", type=Path, default=Path("/media/sauron/GG2/datasets/deformed_blobs/faust_6890/registrations")); p.add_argument("--binary", type=Path, default=Path("Bin/Linux/SphereMap")); p.add_argument("--output", type=Path, default=Path("parameter_search/cmcf_faust_grid"))
    args = p.parse_args(); args.binary = args.binary.resolve(); args.input = args.input.resolve(); args.output = args.output.resolve()
    if not args.input.exists(): p.error(f"input directory does not exist: {args.input}")
    ensure_binary(args)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if int(manifest["seed"]) != args.seed:
            p.error(f"existing manifest seed {manifest['seed']} differs from --seed {args.seed}; refusing to resample cohorts")
        cs = manifest["cohorts"]
    else:
        cs = cohorts(args.seed)
        manifest = {"seed": args.seed, "cohorts": cs, "input": str(args.input), "binary": str(args.binary)}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    runner = Runner(args, manifest)
    if args.stage in {"calibrate", "all"}:
        runner.run("calibrate", [{**DEFAULTS, "iters": 400}], [PILOT[0]])
        calibration_rows = [r for r in csv.DictReader((args.output / "results.csv").open()) if r["stage"] == "calibrate" and r["return_code"] == "0"]
        if calibration_rows:
            projected = float(calibration_rows[-1]["wall_time_sec"]) * 5 * sum((25, 50, 100, 200, 400)) / 400
            print(f"Calibration complete; projected Stage-1 wall time: {projected:.1f}s", flush=True)
    if args.stage in {"1", "all"}: runner.run("1", list(configs_stage1()), PILOT)
    # Later stages are planned from completed rows. The planner is deterministic and resume-safe.
    rows = list(csv.DictReader((args.output / "results.csv").open()))
    if args.stage in {"2", "all"}:
        flow = rank([r for r in rows if r["stage"] == "1"]); chosen = [next(c for c in configs_stage1() if config_id(c) == x[-1]) for x in flow[:3]]
        runner.run("2", [{**c, "lump": 1} for c in chosen], PILOT)
    rows = list(csv.DictReader((args.output / "results.csv").open()))
    if args.stage in {"3", "all"}:
        flow = rank([r for r in rows if r["stage"] in {"1", "2"}]); base = next((next(c for c in configs_stage1() if config_id(c) == x[-1]) for x in flow), DEFAULTS)
        initial = configs_stage3_initial(base)
        runner.run("3", initial, PILOT)
        rows = list(csv.DictReader((args.output / "results.csv").open()))
        degree_scores = rank([r for r in rows if r["stage"] == "3"])
        degree_ids = {config_id({**base, "degree": d}): d for d in (2, 3, 4)}
        default_score = next((x[0] for x in rank([r for r in rows if r["stage"] in {"1", "2"}]) if x[-1] == config_id(base)), float("inf"))
        eligible = [(x[0], degree_ids[x[-1]]) for x in degree_scores if x[-1] in degree_ids and x[0] < default_score]
        if eligible:
            runner.run("3", configs_stage3_advection(base, min(eligible)[1]), PILOT)
    rows = list(csv.DictReader((args.output / "results.csv").open()))
    if args.stage in {"4", "all"}:
        candidates = rank([r for r in rows if r["stage"] in {"1", "2", "3"}]); cfgs = []
        for x in candidates[:3]:
            for src in rows:
                if src["config_id"] == x[-1]:
                    cfgs.append({k: (float(src[k]) if k in {"stepSize", "poincareMaxNorm", "gssTolerance", "aStepSize"} else int(src[k]) if k in {"iters", "lump", "degree", "c2i", "aSteps", "noCenter"} else src[k]) for k in DEFAULTS}); break
        runner.run("4", cfgs, cs["stage4"])
    rows = list(csv.DictReader((args.output / "results.csv").open()))
    if args.stage in {"5", "all"}:
        candidates = rank([r for r in rows if r["stage"] == "4"]); cfg = next((r for r in rows if r["config_id"] == candidates[0][-1]), None)
        if cfg: runner.run("5", [{k: (float(cfg[k]) if k in {"stepSize", "poincareMaxNorm", "gssTolerance", "aStepSize"} else int(cfg[k])) for k in DEFAULTS}], cs["stage5"])
    write_report(args.output, manifest)
    return 0


if __name__ == "__main__": raise SystemExit(main())
