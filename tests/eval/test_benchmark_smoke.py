"""End-to-end smoke test of scripts/benchmark.py on a temporary outputs/ tree built from synthetic plans.

Checks the discovery conventions, that missing pieces never crash the run, and that known differences surface as
the right gate statuses.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from synthetic import KNOWN_T, base_specs, make_plan, scale_plan
from test_eval_synthetic import _pair

ROOT = Path(__file__).resolve().parents[2]


def _run(outputs: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark.py"), "--outputs", str(outputs)],
                          capture_output=True, text=True, timeout=300)


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_benchmark_empty_outputs(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "no plan.json found" in (tmp_path / "benchmark" / "report.md").read_text()


def test_benchmark_end_to_end(tmp_path):
    for cap in ("capA", "capB"):
        (tmp_path / cap).mkdir()
        (tmp_path / cap / "scene.npz").write_bytes(b"")          # discovery only needs the scene's name
    _write(tmp_path / "eval" / "registration" / "capA__capB.json",
           dict(a="capA", b="capB", T2=KNOWN_T.tolist(), yaw_deg=91.3, residual_rmse_m=0.01, inlier_frac_b=0.6,
                overlap_iou=0.9, distinctiveness=2.0))
    A, B = _pair()
    _write(tmp_path / "plan_x" / "capA" / "plan.json", A)
    _write(tmp_path / "plan_x" / "capB" / "plan.json", B)
    rooms, ops, adj = base_specs()
    V = scale_plan(make_plan(rooms, ops, adj, tier="video", sig_len=0.08), 1.02)
    _write(tmp_path / "video_x" / "capA" / "plan.json", V)
    _write(tmp_path / "plan_x" / "junk" / "plan.json", {**A, "meta": {"synthetic": True}})
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(tmp_path / "benchmark" / "gates.csv")))
    get = lambda gate, prod: [x for x in rows if x["gate"] == gate and x["producer"] == prod]   # noqa: E731
    rep = get("repeatability", "plan_x")
    assert len(rep) == 1 and rep[0]["status"] == "fail" and rep[0]["n"] == "12"
    assert get("ceiling_spread", "plan_x")[0]["status"] == "fail"           # R2 differs by 2 cm
    video = get("wall_length_video", "video_x")
    assert video and video[0]["status"] == "pass" and video[0]["basis"] == "lidar_reference"
    assert all(x["status"] == "not_verifiable" for x in get("opening_width", "plan_x"))
    assert "synthetic test plan" in (tmp_path / "benchmark" / "report.md").read_text()
