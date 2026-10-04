"""Ablation harness for plan extractor v2 (floorplan/plan/beta) against the frozen v1 (floorplan/plan/beta_v1).

    python scripts/plan_v2_eval.py register
    python scripts/plan_v2_eval.py run --producer beta_v1 --capture <capture> --tag v1
    python scripts/plan_v2_eval.py run --producer beta --capture <capture> --tag v2 [--set KEY=VALUE ...]
    python scripts/plan_v2_eval.py run --producer beta --capture <capture> --tag v2_rot90 --rot90 1
    python scripts/plan_v2_eval.py score v1 v2 [...]          -> outputs/plan_v2/scores.json

Why a separate script and not the judge's folders: the judge scored the OLD drift-on scenes (outputs/drift/*/scene_on)
with ITS registration. v2 runs on the canonical scenes (outputs/scenes_v2, D-011 intrinsics, raw_frame), which
need their own scene-to-scene registration. Everything else is the judge's code, imported unchanged:
judge_common.compare (floorplan/eval matching + gate + topology split + Wilson), the perturbations, and
judge_plausibility's floor recall. So every number here is directly comparable with the judge's tables.

Each scored tag reports, for the repeat pair (with_ceiling = A, floor_only = B):
  pass rate + Wilson 95% interval, rooms matched, median |delta|, same-topology n/pass/median, footprints and
  floor recall of both captures (a repeatability gain that shrinks the plan is a red flag, judge J-I-5),
  repeat-pair interval coverage, and the invariance tests if <tag>_shift / _rot90 / _thin90 runs exist.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_common import (PAIR, ROOT, compare, jsonable, plan_T_original_to_moved, thin_scene,  # noqa: E402
                          transform_scene)
from judge_plausibility import floor_raster, floor_recall  # noqa: E402

from floorplan.eval.match import as_plan_dict  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402

import os  # noqa: E402

OUT = Path(os.environ.get("PLAN_V2_OUT", ROOT / "outputs" / "plan_v2"))        # evidence folder (env override)
SCENES = Path(os.environ.get("PLAN_V2_SCENES", ROOT / "outputs" / "scenes_v2"))
PERTURB = {"shift": ((0.013, 0.007), 0), "rot90": ((0.0, 0.0), 1), "thin90": ((0.0, 0.0), 0),
           # raster-phase test (stability part): same scene, the raster moved by a quarter cell in u and v
           # (<tag>_phase run with --set grid_phase=0.25,0.25); the data-anchored "shift" test cannot see the phase,
           # because the raster corner moves with the data
           "phase": ((0.0, 0.0), 0)}
KEYS = ("rooms_a", "rooms_b", "rooms_matched", "walls_pass", "walls_fail", "walls_unmatched", "wall_pass_rate",
        "wilson_pass", "wall_delta_median_m", "walls_within_2cm", "same_topology", "different_topology",
        "wall_interval_coverage", "footprint_a", "footprint_b", "openings_matched")


# ----------------------------------------------------------------------------------------------- registration

def register_pair() -> dict:
    """Scene-to-scene registration of the repeat pair on scenes_v2 (from geometry, never from a plan: judge J-2)."""
    from floorplan.eval.register import RegisterParams, local_disagreement, register, scene_evidence
    ea, eb = (scene_evidence(*load_scene(SCENES / c), c) for c in PAIR)
    reg = register(ea, eb, RegisterParams())
    d = reg.to_dict()
    tiles = local_disagreement(ea, eb, np.array(reg.T2))
    disp = [float(np.linalg.norm(t["displacement_m"])) for t in tiles if t["well_constrained"]]
    d["tiles_summary"] = dict(n=len(disp), median_disp_m=float(np.median(disp)) if disp else None,
                              p90_disp_m=float(np.percentile(disp, 90)) if disp else None)
    d.pop("yaw_curve", None)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "registration.json").write_text(json.dumps(jsonable(d), indent=1, default=float))
    return d


def registration_T() -> np.ndarray:
    return np.array(json.loads((OUT / "registration.json").read_text())["T2"])


# ----------------------------------------------------------------------------------------------- running

def _params(cls, overrides: dict[str, str]):
    """Parameter dataclass with KEY=VALUE overrides parsed with each default's type (bool/tuple aware)."""
    base = cls()
    kw = {}
    for k, v in overrides.items():
        d = getattr(base, k)
        if v.lower() == "none":
            kw[k] = None
            continue
        kw[k] = (v.lower() in ("1", "true", "yes")) if isinstance(d, bool) else \
            tuple(float(x) for x in v.split(",")) if isinstance(d, tuple) else float(v) if d is None else type(d)(v)
    return dataclasses.replace(base, **kw)


def thin(scene: dict, keep: float, seed: int = 1) -> dict:
    """judge_common.thin_scene (same seed, same draws), also applied to the per-raw-point arrays it predates
    (raw_frame), so single-visit measurement still knows each kept point's source frame."""
    out = thin_scene(scene, keep, seed)
    rng = np.random.default_rng(seed)
    rng.random(len(scene["points"]))
    m = rng.random(len(scene["raw_points"])) < keep
    if "raw_frame" in scene:
        out["raw_frame"] = scene["raw_frame"][m]
    return out


def run(producer: str, capture: str, tag: str, overrides: dict, rot90: int, shift, keep_frac, draw: bool,
        scene_dir: Path | None = None) -> dict:
    """Run one extractor version on one capture (scenes_v2 unless scene_dir is given); save plan.json, run.json
    and the debug PNG under outputs/plan_v2/runs/<tag>/<capture>/."""
    mod = importlib.import_module(f"floorplan.plan.{producer}")
    scene, info = load_scene(scene_dir or SCENES / capture)
    if rot90 or any(shift):
        scene = transform_scene(scene, rot90, tuple(shift))
    if keep_frac is not None:
        scene = thin(scene, keep_frac)
    out = OUT / "runs" / tag / capture
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(producer=producer, capture=capture, tag=tag, overrides=overrides, rot90=rot90, shift=list(shift),
               keep_frac=keep_frac, scene_dir=str(scene_dir or SCENES / capture))
    try:
        t0 = time.time()
        plan, debug = mod.extract_plan_debug(scene, info, info.get("capture_id", capture),
                                             _params(mod.BetaParams, overrides))
        rec.update(status="ok", runtime_s=round(time.time() - t0, 2), rooms=len(plan.rooms), walls=len(plan.walls),
                   openings=len(plan.openings))
        (out / "plan.json").write_text(json.dumps(dataclasses.asdict(plan), indent=1, default=float))
        if draw:
            importlib.import_module(f"floorplan.plan.{producer}.render").render_debug(plan, debug, out / "debug.png")
    except Exception as e:  # noqa: BLE001 - a crash is a result to record
        rec.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    (out / "run.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


# ----------------------------------------------------------------------------------------------- scoring

def load(tag: str, capture: str) -> dict | None:
    f = OUT / "runs" / tag / capture / "plan.json"
    return as_plan_dict(f) if f.exists() else None


def _floor_rasters() -> dict:
    out = {}
    for c in PAIR:
        s, i = load_scene(SCENES / c)
        out[c] = floor_raster(s, i)
    return out


def score_tag(tag: str, T: np.ndarray, rasters: dict) -> dict:
    A, B = load(tag, PAIR[0]), load(tag, PAIR[1])
    if A is None or B is None:
        return dict(error="missing plan")
    res = compare(A, B, T, PAIR[0], PAIR[1])
    _overlay(A, B, res, T, tag)
    s = res["summary"]
    row = {k: s.get(k) for k in KEYS}
    row["same_topology_coverage"] = same_topology_coverage(A, B, res)
    row["floor_recall_a"] = floor_recall(A, *rasters[PAIR[0]])
    row["floor_recall_b"] = floor_recall(B, *rasters[PAIR[1]])
    row["openings_a"], row["openings_b"] = s["openings_a"], s["openings_b"]
    row["opening_deltas_cm"] = [None if o["delta"] is None else round(100 * o["delta"], 1) for o in s["openings"]]
    row["ceilings"] = _ceilings(A, B, s)
    row["invariance"] = {}
    for name, (shift, quarter) in PERTURB.items():
        for c in PAIR:
            base, moved = load(tag, c), load(f"{tag}_{name}", c)
            if base is None or moved is None:
                continue
            T_ba = np.linalg.inv(plan_T_original_to_moved(quarter, shift))
            si = compare(base, moved, T_ba, c, f"{c}+{name}")["summary"]
            n = si["walls_pass"] + si["walls_fail"] + si["walls_unmatched"]
            row["invariance"][f"{name}/{c.split('__')[0]}"] = dict(passed=si["walls_pass"], judged=n,
                                                                    rooms=(si["rooms_a"], si["rooms_b"]))
    return row


def _overlay(A: dict, B: dict, res: dict, T: np.ndarray, tag: str) -> None:
    """The judge's overlay figure (floor_only walls on with_ceiling, coloured by gate status) for one tag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from judge_repeatability import COLOURS, plot_overlay
    fig, ax = plt.subplots(figsize=(11, 11))
    plot_overlay(ax, A, B, res, T, f"plan_v2 [{tag}]: floor_only walls on with_ceiling (grey)")
    for k, c in COLOURS.items():
        ax.plot([], [], color=c, label=k)
    for w in A["walls"]:
        mid = (np.array(w["p0"]) + np.array(w["p1"])) / 2
        ax.text(mid[0], mid[1], w["id"], fontsize=5, color="0.3")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"repeat_{tag}.png", dpi=90)
    plt.close(fig)


def same_topology_coverage(A: dict, B: dict, res: dict) -> dict:
    """Interval coverage on same-topology walls only (B-4 target ~0.95): share with |z| <= 1.96, where
    z = delta / sqrt(sigma_a^2 + sigma_b^2) as computed by floorplan/eval. Different-topology walls are a
    segmentation error that no measurement sigma should be asked to cover."""
    from judge_common import _neighbours
    m, rep = res["match"], res["report"]
    pair = {q.a: q.b for q in m.walls}
    na, nb = _neighbours(A), _neighbours(B)
    z = []
    for w in rep.walls:
        if w.delta_m is None or w.z is None:
            continue
        pa, qa = na[w.wall_a]
        if pair.get(pa) in set(nb[w.wall_b]) and pair.get(qa) in set(nb[w.wall_b]):
            z.append(abs(w.z))
    z = np.array(z)
    return dict(n=len(z), coverage=float(np.mean(z <= 1.96)) if len(z) else None,
                median_abs_z=float(np.median(z)) if len(z) else None)


def _ceilings(A: dict, B: dict, s: dict) -> list[dict]:
    """Ceiling height of each matched room pair (cross-capture spread; the gate wants <= 1 cm)."""
    ra, rb = {r["id"]: r for r in A["rooms"]}, {r["id"]: r for r in B["rooms"]}
    out = []
    for row in s["room_bbox"]:
        ca, cb = ra[row["a"]]["ceiling_height"], rb[row["b"]]["ceiling_height"]
        out.append(dict(a=row["a"], b=row["b"], ceil_a=ca.get("value"), ceil_b=cb.get("value"),
                        status=(ca.get("status"), cb.get("status"))))
    return out


def score(tags: list[str]) -> dict:
    T = registration_T()
    rasters = _floor_rasters()
    f = OUT / "scores.json"
    allrows = json.loads(f.read_text()) if f.exists() else {}
    for tag in tags:
        allrows[tag] = jsonable(score_tag(tag, T, rasters))
        r = allrows[tag]
        if "error" in r:
            print(tag, r)
            continue
        n = r["walls_pass"] + r["walls_fail"] + r["walls_unmatched"]
        wl = r["wilson_pass"] or (0, 0)
        st = r["same_topology"]
        print(f"{tag:28s} rooms {r['rooms_a']}/{r['rooms_b']} m{r['rooms_matched']}  pass {r['walls_pass']}/{n}="
              f"{100 * r['walls_pass'] / max(n, 1):.1f}% [{100 * wl[0]:.1f},{100 * wl[1]:.1f}]  "
              f"med {100 * (r['wall_delta_median_m'] or np.nan):.1f}cm  same-topo {st['n']}/{st['passed']}/"
              f"{100 * (st['median_m'] or np.nan):.1f}cm  fp {r['footprint_a']:.1f}/{r['footprint_b']:.1f}  "
              f"recall {r['floor_recall_a']:.3f}/{r['floor_recall_b']:.3f}  cov {r['wall_interval_coverage']:.2f} "
              f"st-cov {r['same_topology_coverage']['coverage']} |z|med {r['same_topology_coverage']['median_abs_z']} "
              f"inv {r['invariance']}")
    (OUT / "scores.json").write_text(json.dumps(allrows, indent=1, default=float))
    return allrows


def plausibility(tags: list[str]) -> dict:
    """The judge's plausibility checks (judge_plausibility.plan_stats) on all three captures, per tag."""
    from judge_common import CAPTURES
    from judge_plausibility import plan_stats
    f = OUT / "plausibility.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    for c in CAPTURES:
        scene, info = load_scene(SCENES / c)
        occ, origin = floor_raster(scene, info)
        for tag in tags:
            plan = load(tag, c)
            if plan is None:
                continue
            st = plan_stats(plan, scene, info, occ, origin)
            st["runtime_s"] = json.loads((OUT / "runs" / tag / c / "run.json").read_text()).get("runtime_s")
            out[f"{tag}/{c}"] = jsonable(st)
            print(tag, c.split("__")[0], {k: st[k] for k in ("rooms", "walls", "footprint_m2", "floor_recall",
                  "unvisited_rooms", "sliver_rooms", "overlap_m2", "connected", "openings", "openings_width_measured",
                  "wall_length_measured_frac", "median_wall_halfwidth_cm", "runtime_s")})
    f.write_text(json.dumps(out, indent=1, default=float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("register")
    r = sub.add_parser("run")
    r.add_argument("--producer", default="beta", choices=("beta", "beta_v1"))
    r.add_argument("--capture", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--rot90", type=int, default=0)
    r.add_argument("--shift", type=float, nargs=2, default=(0.0, 0.0))
    r.add_argument("--keep-frac", type=float, default=None)
    r.add_argument("--no-render", action="store_true")
    r.add_argument("--scene-dir", type=Path, default=None, help="another prepared scene (drift OFF, photo tier)")
    s = sub.add_parser("score")
    s.add_argument("tags", nargs="+")
    q = sub.add_parser("plaus")
    q.add_argument("tags", nargs="+")
    a = ap.parse_args()
    if a.cmd == "register":
        d = register_pair()
        print({k: d.get(k) for k in ("yaw_deg", "residual_rmse_m", "inlier_frac_b", "tiles_summary")})
    elif a.cmd == "run":
        rec = run(a.producer, a.capture, a.tag, dict(kv.split("=", 1) for kv in a.set), a.rot90, a.shift,
                  a.keep_frac, not a.no_render, a.scene_dir)
        print({k: v for k, v in rec.items() if k != "traceback"})
    elif a.cmd == "plaus":
        plausibility(a.tags)
    else:
        score(a.tags)


if __name__ == "__main__":
    main()
