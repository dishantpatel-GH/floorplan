"""Shared helpers for the plan-extractor judge (scripts/judge_*.py).

Why a judge needs its own runner: the two extractors (plan_alpha, plan_beta) must be compared on exactly the same
inputs, with the same evaluation code (floorplan/eval), under the same perturbations. Each extractor's own run
script writes to its own folder layout and its own self-check, so here both are driven through one function and
one output layout: outputs/judge/runs/<variant>/<producer>/<capture>/plan.json + run.json.

Scene perturbations exist to answer one question the repeat pair cannot: how much of the cross-capture
disagreement is caused by the extractor itself? A rigid move of the whole scene (a sub-cell shift, a 90 deg turn)
changes nothing physical, so an ideal extractor returns the same plan, moved. Whatever changes is the extractor's
own instability, a floor under which repeatability can never go.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplan.eval.match import as_plan_dict, match_plans  # noqa: E402
from floorplan.eval.repeatability import repeatability, wall_tolerance  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402

OUT = ROOT / "outputs" / "judge"
PRODUCERS = ("alpha", "beta")
CAPTURES = ("single_room__c00a170fe1", "single_scan_floor_only__1a8384c3f6", "single_scan_with_ceiling__c7d28f72c6")
PAIR = ("single_scan_with_ceiling__c7d28f72c6", "single_scan_floor_only__1a8384c3f6")   # same apartment twice


def scene_dir(capture: str, variant: str) -> Path:
    """Prepared scene of a capture: 'raw' = ARKit poses as recorded, 'drift_on' = drift module's corrected poses."""
    if variant == "drift_on":
        return ROOT / "outputs" / "drift" / capture / "scene_on"
    return ROOT / "outputs" / capture


# ----------------------------------------------------------------------------------------------- perturbations

def yaw_matrix(quarter_turns: int) -> np.ndarray:
    """Rotation about +y by k * 90 deg (exact integers, so no resampling error is introduced)."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][quarter_turns % 4]
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def transform_scene(scene: dict, quarter_turns: int = 0, shift_xz: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Rigidly move every geometric array of the scene about the vertical axis (heights untouched)."""
    R = yaw_matrix(quarter_turns)
    t = np.array([shift_xz[0], 0.0, shift_xz[1]])
    out = dict(scene)
    for k in ("points", "raw_points", "traj"):
        out[k] = (scene[k].astype(np.float64) @ R.T + t).astype(scene[k].dtype)
    for k in ("normals", "raw_ray"):
        out[k] = (scene[k].astype(np.float32) @ R.T.astype(np.float32)).astype(scene[k].dtype)
    return out


def plan_T_original_to_moved(quarter_turns: int, shift_xz: tuple[float, float]) -> np.ndarray:
    """3x3 map of plan coordinates (u, v) = (x, z) from the original scene to the moved one."""
    R = yaw_matrix(quarter_turns)
    T = np.eye(3)
    T[:2, :2] = R[np.ix_([0, 2], [0, 2])]
    T[:2, 2] = shift_xz
    return T


def crop_scene(scene: dict, box_uv: tuple[float, float, float, float]) -> dict:
    """Keep only geometry inside an axis-aligned plan box (u0, u1, v0, v1): simulates a capture of part of a home."""
    u0, u1, v0, v1 = box_uv

    def inside(P: np.ndarray) -> np.ndarray:
        return (P[:, 0] >= u0) & (P[:, 0] <= u1) & (P[:, 2] >= v0) & (P[:, 2] <= v1)

    out = dict(scene)
    m = inside(scene["points"])
    for k in ("points", "normals", "colors"):
        if k in scene and len(scene[k]) == len(m):
            out[k] = scene[k][m]
    m = inside(scene["raw_points"])
    for k in ("raw_points", "raw_range", "raw_ray"):
        out[k] = scene[k][m]
    out["traj"] = scene["traj"][inside(scene["traj"])]
    return out


def thin_scene(scene: dict, keep: float, seed: int = 1) -> dict:
    """Keep a random share of the TSDF surface and of the raw points (fixed seed).

    The same walls, sampled slightly differently: the closest thing to a second capture without drift or coverage
    changes. It also moves the data extremes that both extractors anchor their raster to, so it tests sensitivity
    to the raster phase as well."""
    rng = np.random.default_rng(seed)
    out = dict(scene)
    m = rng.random(len(scene["points"])) < keep
    for k in ("points", "normals", "colors"):
        if k in scene and len(scene[k]) == len(m):
            out[k] = scene[k][m]
    m = rng.random(len(scene["raw_points"])) < keep
    for k in ("raw_points", "raw_range", "raw_ray"):
        out[k] = scene[k][m]
    return out


# ----------------------------------------------------------------------------------------------- running

def _params(cls, overrides: dict[str, str]):
    """The extractor's parameter dataclass with KEY=VALUE overrides, each parsed with its default's type."""
    base = cls()
    kw = {k: type(getattr(base, k))(v) for k, v in overrides.items()}   # unknown key -> AttributeError, loudly
    return dataclasses.replace(base, **kw)


def run_producer(producer: str, scene: dict, info: dict, capture_id: str, overrides: dict | None = None):
    """Run one extractor (default parameters unless overridden). Returns (Plan, debug-or-None, wall-clock s)."""
    t0 = time.time()
    if producer == "alpha":
        from floorplan.plan.alpha import extract_plan
        from floorplan.plan.alpha.params import AlphaParams
        plan, debug = extract_plan(scene, info, capture_id, _params(AlphaParams, overrides or {})), None
    elif producer == "beta":
        from floorplan.plan.beta import BetaParams, extract_plan_debug
        plan, debug = extract_plan_debug(scene, info, capture_id, _params(BetaParams, overrides or {}))
    else:
        raise ValueError(f"unknown producer {producer!r}")
    return plan, debug, time.time() - t0


def render(producer: str, plan, debug, scene: dict, info: dict, png: Path) -> None:
    """Each extractor's own debug figure, so the judge looks at what the authors looked at."""
    if producer == "alpha":
        from floorplan.plan.alpha.render import render_plan
        render_plan(plan, scene, info, png)
    else:
        from floorplan.plan.beta.render import render_debug
        render_debug(plan, debug, png)


def run_and_save(producer: str, scene: dict, info: dict, out_dir: Path, extra: dict, draw: bool = True,
                 overrides: dict | None = None) -> dict:
    """Run, save plan.json + run.json (+ debug.png). A crash is recorded, not raised: failure behaviour is judged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = dict(producer=producer, capture_id=info.get("capture_id"), **extra)
    try:
        plan, debug, secs = run_producer(producer, scene, info, info.get("capture_id", "unknown"), overrides)
        (out_dir / "plan.json").write_text(json.dumps(dataclasses.asdict(plan), indent=1, default=float))
        rec.update(status="ok", runtime_s=round(secs, 2), rooms=len(plan.rooms), walls=len(plan.walls),
                   openings=len(plan.openings))
        if draw:
            render(producer, plan, debug, scene, info, out_dir / "debug.png")
    except Exception as e:  # noqa: BLE001 - the judge must record any failure mode, whatever its type
        rec.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
    (out_dir / "run.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def load_run(variant: str, producer: str, capture: str) -> dict | None:
    f = OUT / "runs" / variant / producer / capture / "plan.json"
    return as_plan_dict(f) if f.exists() else None


def load_scene_for(capture: str, variant: str):
    return load_scene(scene_dir(capture, variant))


# ----------------------------------------------------------------------------------------------- comparison

def compare(plan_a: dict, plan_b: dict, T_ba: np.ndarray, name_a: str = "A", name_b: str = "B") -> dict:
    """Match B onto A with floorplan/eval and summarise everything the judge ranks on.

    Beyond the gate's own pass rate (every wall of every matched room, unmatched counted as fail), it reports
    numbers that separate *why* a plan fails: walls matched at all (segmentation agreement), the median |delta|
    of matched walls (measurement agreement), the pass rate on long walls only, room bounding-box agreement (what
    a homeowner checks with a tape) and opening-width agreement.
    """
    m = match_plans(plan_a, plan_b, T_ba)
    rep = repeatability(plan_a, plan_b, m, name_a, name_b, "lidar")
    s = rep.summary()
    matched = [w for w in rep.walls if w.delta_m is not None]
    d = np.array([abs(w.delta_m) for w in matched])
    long_ = [w for w in matched if (w.length_a_m + w.length_b_m) / 2 >= 1.0]
    walls_a = [w for w in plan_a["walls"] if w["room_id"] in {r.a for r in m.rooms}]
    walls_b = [w for w in plan_b["walls"] if w["room_id"] in {r.b for r in m.rooms}]
    len_total = sum(w["length"]["value"] for w in walls_a) + sum(w["length"]["value"] for w in walls_b)
    len_pass = sum((w.length_a_m + w.length_b_m) for w in matched if w.status == "pass")
    s.update(
        rooms_a=len(plan_a["rooms"]), rooms_b=len(plan_b["rooms"]),
        walls_a_in_matched_rooms=len(walls_a), walls_b_in_matched_rooms=len(walls_b),
        wall_match_rate=(2 * len(matched) / max(len(walls_a) + len(walls_b), 1)),
        wall_delta_mean_m=float(d.mean()) if len(d) else None,
        walls_within_2cm=int((d <= 0.02).sum()), walls_within_5cm=int((d <= 0.05).sum()),
        long_walls_matched=len(long_), long_walls_pass=sum(w.status == "pass" for w in long_),
        **_topology_split(plan_a, plan_b, rep, m),
        wilson_pass=wilson(s["walls_pass"], s["walls_pass"] + s["walls_fail"] + s["walls_unmatched"]),
        length_weighted_pass=len_pass / len_total if len_total else None,
        room_bbox=_bbox_rows(plan_a, plan_b, m),
        openings=[dict(a=o.opening_a, b=o.opening_b, kind=f"{o.kind_a}/{o.kind_b}", width_a=o.width_a_m,
                       width_b=o.width_b_m, delta=o.delta_m) for o in rep.openings],
        openings_a=len(plan_a["openings"]), openings_b=len(plan_b["openings"]),
        footprint_a=plan_a["footprint_area"]["value"], footprint_b=plan_b["footprint_area"]["value"],
    )
    return dict(summary=s, report=rep, match=m)


def _neighbours(plan: dict) -> dict[str, tuple[str, str]]:
    """wall id -> (previous wall id, next wall id) around its room polygon (wall_ids are in polygon order)."""
    out = {}
    for r in plan["rooms"]:
        ids = r["wall_ids"]
        for i, w in enumerate(ids):
            out[w] = (ids[i - 1], ids[(i + 1) % len(ids)])
    return out


def _topology_split(A: dict, B: dict, rep, m) -> dict:
    """Split matched walls into 'same topology' (both neighbours also matched to each other, so both corners are
    built from the same walls) and 'different topology' (a jog, notch or different neighbour moved a corner).

    Same-topology walls measure the extractor's *measurement* repeatability; the rest, plus the unmatched walls,
    measure its *segmentation* repeatability. The two need different fixes, so the judge reports them apart."""
    pair = {p.a: p.b for p in m.walls}
    na, nb = _neighbours(A), _neighbours(B)
    same, diff = [], []
    for w in rep.walls:
        if w.delta_m is None:
            continue
        pa, qa = na[w.wall_a]
        nbs = set(nb[w.wall_b])
        (same if pair.get(pa) in nbs and pair.get(qa) in nbs else diff).append(w)

    def stats(ws):
        d = np.array([abs(w.delta_m) for w in ws])
        return dict(n=len(ws), passed=sum(w.status == "pass" for w in ws),
                    median_m=float(np.median(d)) if len(d) else None)

    return dict(same_topology=stats(same), different_topology=stats(diff))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval of a pass rate k/n (is 8/94 really different from 11/98?)."""
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(c - h), float(c + h))


def _bbox_rows(A: dict, B: dict, m) -> list[dict]:
    """Room length x width (aligned bbox) per matched room, shorter side paired with shorter side.

    Pairing by size, not by direction, keeps this independent of the 90 deg turn between captures; it is wrong only
    for near-square rooms whose sides swap order, which then show up as large deltas (visible, not hidden)."""
    ra, rb = {r["id"]: r for r in A["rooms"]}, {r["id"]: r for r in B["rooms"]}
    rows = []
    for p in m.rooms:
        a, b = ra[p.a], rb[p.b]
        # the published plan.json stores bbox_dims as a dict ({length, width}); the dataclass dump as a list
        dims = lambda r: r["bbox_dims"].values() if isinstance(r["bbox_dims"], dict) else r["bbox_dims"]  # noqa: E731
        da = sorted(x["value"] for x in dims(a))
        db = sorted(x["value"] for x in dims(b))
        rows.append(dict(a=p.a, b=p.b, iou=round(p.iou_local, 3), area_a=a["floor_area"]["value"],
                         area_b=b["floor_area"]["value"], dims_a=da, dims_b=db,
                         dims_delta=[db[i] - da[i] for i in range(2)],
                         dims_pass=[abs(db[i] - da[i]) <= wall_tolerance((da[i] + db[i]) / 2) for i in range(2)]))
    return rows


def jsonable(x):
    """Strip numpy types for json.dumps."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x
