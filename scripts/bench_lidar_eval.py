#!/usr/bin/env python
"""Benchmark, LiDAR tier on the sample captures: score the outputs of scripts/bench_lidar_runs.sh.

    python scripts/bench_lidar_eval.py            -> outputs/benchmark/lidar/{results.json, *.csv, *.png}

Inputs are ONLY the run folders written by the one command (scripts/run_capture.py --tier lidar):
outputs/benchmark/lidar/<capture>/<variant>/{plan.json, scene/, run_report.json, time.log}.

What it computes (each block reuses existing, already-reviewed code; nothing is re-implemented):
  1. timing: run_report.json runtime + /usr/bin/time (wall, CPU, peak RSS);
  2. drift accountability (on vs off, floor_only and with_ceiling): plan footprint / bbox / rooms / walls, the
     judge's plausibility stats (judge_plausibility.plan_stats), the drift module's independent scene metrics
     (drift_ablation.wall_sharpness, drift_ablation.footprint), and an overlay of the OFF plan on the ON plan in one
     frame (drift_ablation.to_common_frame: both worlds share the gauge of the first fragment);
  3. repeatability (with_ceiling = A, floor_only = B, drift on): registration from geometry
     (floorplan/eval/register, as plan_v2_eval.register_pair) on THESE run scenes, then judge_common.compare and
     plan_v2_eval.same_topology_coverage, unchanged; per-wall CSV; opening widths; single_room vs each big capture;
  4. ceilings per room; 5. damage confirmed / review counts; 6. interval coverage on the repeat pair.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from drift_ablation import footprint, to_common_frame, wall_sharpness  # noqa: E402
from judge_common import _neighbours, compare, jsonable  # noqa: E402
from judge_plausibility import floor_raster, plan_stats  # noqa: E402
from plan_v2_eval import same_topology_coverage  # noqa: E402

from floorplan.eval.match import as_plan_dict  # noqa: E402
from floorplan.eval.register import RegisterParams, local_disagreement, register, scene_evidence  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402

B = ROOT / "outputs" / "benchmark" / "lidar"
CAPS = ("single_room", "floor_only", "with_ceiling")


def d(cap, var="drift_on") -> Path:
    return B / cap / var


def plan(cap, var="drift_on") -> dict:
    """plan.json in the published-schema form, normalised by floorplan/eval (ci95 -> lo/hi). One adapter: the
    schema writes a room's bbox_dims as {length, width}, the judge's _bbox_rows expects the dataclass list form."""
    p = as_plan_dict(d(cap, var) / "plan.json")
    for r in p["rooms"]:
        if isinstance(r.get("bbox_dims"), dict):
            r["bbox_dims"] = [r["bbox_dims"]["length"], r["bbox_dims"]["width"]]
    return p


def raw_plan(cap, var="drift_on") -> dict:
    return json.loads((d(cap, var) / "plan.json").read_text())


_scenes: dict = {}


def scene(cap, var="drift_on"):
    k = (cap, var)
    if k not in _scenes:
        _scenes[k] = load_scene(d(cap, var) / "scene")
    return _scenes[k]


# ------------------------------------------------------------------------------------------------ 1 timing
def timing() -> dict:
    out = {}
    for f in sorted(B.glob("*/*/run_report.json")):
        cap, var = f.parent.parent.name, f.parent.name
        rr = json.loads(f.read_text())
        t = {}
        for line in (f.parent / "time.log").read_text().splitlines():
            line = line.strip()
            if line.startswith("Elapsed (wall clock)"):
                v = line.split("):")[-1].strip().split(":")
                t["wall_s"] = round(sum(float(x) * 60 ** i for i, x in enumerate(reversed(v))), 1)
            elif line.startswith("User time"):
                t["user_s"] = float(line.split(":")[-1])
            elif line.startswith("System time"):
                t["sys_s"] = float(line.split(":")[-1])
            elif line.startswith("Maximum resident"):
                t["max_rss_gb"] = round(int(line.split(":")[-1]) / 1e6, 2)
        log = (f.parent / "stdout.log").read_text().splitlines()
        stage = {}
        for line in log:
            try:
                ts = float(line[1:line.index("s]")])
            except ValueError:
                continue
            if "scene saved" in line:
                stage["scene_saved_at_s"] = ts
            if line.split("] ", 1)[-1].startswith("plan:"):
                stage["plan_done_at_s"] = ts
        dm = (rr.get("front_end") or {}).get("drift_runtime_s")
        out[f"{cap}/{var}"] = dict(runtime_s=rr.get("runtime_s"), drift_s=dm, **stage, **t)
    return out


# ------------------------------------------------------------------------------------------------ 2 drift
def plan_geometry(p: dict) -> dict:
    pts = np.array([q for r in p["rooms"] for q in r["polygon"]])
    lo, hi = pts.min(0), pts.max(0)
    return dict(rooms=len(p["rooms"]), walls=len(p["walls"]), openings=len(p["openings"]),
                footprint_m2=round(p["footprint_area"]["value"], 2),
                footprint_interval=[round(p["footprint_area"].get("lo") or np.nan, 2),
                                    round(p["footprint_area"].get("hi") or np.nan, 2)],
                bbox_m=[round(float(hi[0] - lo[0]), 3), round(float(hi[1] - lo[1]), 3)],
                room_areas=sorted([round(r["floor_area"]["value"], 2) for r in p["rooms"]], reverse=True))


def drift_block() -> dict:
    out = {}
    for cap in ("floor_only", "with_ceiling"):
        rows = {}
        for var in ("drift_on", "drift_off"):
            s, i = scene(cap, var)
            p = plan(cap, var)
            occ, origin = floor_raster(s, i)
            st = plan_stats(p, s, i, occ, origin)
            rows[var] = dict(plan=plan_geometry(p), plausibility=jsonable(st),
                             wall_sharpness=wall_sharpness(s, i), scene_footprint=footprint(s, i),
                             ceiling_y=i.get("ceiling_y"), floor_y=i.get("floor_y"),
                             ceiling_height_global=i.get("ceiling_height_global"))
        rows["drift_module"] = drift_module_summary(cap)
        rows["on_vs_off_plan"] = on_off_compare(cap)
        rows["off_onto_on_registration"] = fig_drift(cap)
        out[cap] = rows
    return out


def drift_module_summary(cap: str) -> dict:
    """The drift module's own report from the default run (run_report.json -> front_end.drift): in-sample loop
    surface separation ARKit vs corrected, anchor spreads (ceiling level is an independent witness), corrections."""
    d = json.loads((d_(cap) / "run_report.json").read_text())["front_end"]["drift"]
    lr = d["loop_residuals"]
    out = dict(loops={k: v for k, v in d["loops"].items() if not isinstance(v, list)},
               frame_correction=d["frame_correction"], runtime_s=d.get("runtime_s"))
    for kind in ("revisit", "local"):
        if kind in lr:
            out[f"{kind}_surface_sep_cm"] = {v: {k: lr[kind][v]["surface_sep_cm"][k] for k in ("n", "median", "p95")}
                                             for v in ("arkit", "corrected")}
    out["anchors"] = {v: {k: d["anchors"][v].get(k) for k in ("ceiling_mad_sigma_cm", "floor_mad_sigma_cm",
                                                               "wall_yaw_std_deg")} for v in ("arkit", "corrected")}
    return jsonable(out)


def d_(cap):
    return d(cap, "drift_on")


def T_off_to_on(cap: str) -> np.ndarray:
    """Rigid 2-D map of OFF plan coordinates into ON plan coordinates through the shared world gauge."""
    s_on, i_on = scene(cap, "drift_on")
    s_off, i_off = scene(cap, "drift_off")
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Q = _plan_to_frame(P, s_off, i_off, s_on)
    R = np.c_[Q[1] - Q[0], Q[2] - Q[0]]
    U, _, Vt = np.linalg.svd(R)
    T = np.eye(3)
    T[:2, :2] = U @ Vt
    T[:2, 2] = Q[0]
    return T


def on_off_compare(cap: str) -> dict:
    """Score the OFF plan against the ON plan with the same repeatability code (shared gauge, no registration):
    how many walls / rooms / ceilings the drift handling changes."""
    A, Bp = plan(cap, "drift_on"), plan(cap, "drift_off")
    res = compare(A, Bp, T_off_to_on(cap), "on", "off")
    s = res["summary"]
    n = s["walls_pass"] + s["walls_fail"] + s["walls_unmatched"]
    return jsonable(dict(rooms_on=s["rooms_a"], rooms_off=s["rooms_b"], rooms_matched=s["rooms_matched"],
                         walls_judged=n, walls_same_within_gate=s["walls_pass"], walls_fail=s["walls_fail"],
                         walls_unmatched=s["walls_unmatched"], median_abs_delta_cm=100 * (s["wall_delta_median_m"] or np.nan),
                         rooms=[dict(on=r.room_a, off=r.room_b, area_on=r.area_a_m2, area_off=r.area_b_m2,
                                     ceil_on=r.ceiling_a_m, ceil_off=r.ceiling_b_m) for r in res["report"].rooms]))


def _plan_to_frame(poly_uv: np.ndarray, s_from: dict, i_from: dict, s_to: dict) -> np.ndarray:
    P = np.c_[poly_uv[:, 0], np.full(len(poly_uv), i_from["floor_y"]), poly_uv[:, 1]]
    Q = to_common_frame(P, i_from, s_from["T_align"], s_to["T_align"])
    return Q[:, [0, 2]]


def fig_drift(cap: str) -> dict:
    """Left: OFF plan in ON's frame through the shared world gauge (shows the accumulated heading/position drift).
    Right: OFF plan registered onto ON by whole-scene geometry (floorplan/eval/register), so only SHAPE differences
    (doubled / shifted walls, lost rooms) remain. Returns the registration summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s_on, i_on = scene(cap, "drift_on")
    s_off, i_off = scene(cap, "drift_off")
    p_on, p_off = plan(cap, "drift_on"), plan(cap, "drift_off")
    reg = registration(cap, cap, "drift_on", "drift_off")
    T = np.array(reg["T2"])
    fig, axes = plt.subplots(1, 2, figsize=(20, 10.5))
    g_on, g_off = plan_geometry(p_on), plan_geometry(p_off)
    maps = [lambda P: _plan_to_frame(P, s_off, i_off, s_on), lambda P: P @ T[:2, :2].T + T[:2, 2]]
    titles = ["OFF placed through the shared gauge (first fragment fixed): accumulated drift visible",
              f"OFF registered onto ON by scene geometry (yaw {reg['yaw_deg']:.2f} deg, rmse "
              f"{100 * reg['residual_rmse_m']:.1f} cm): shape differences only"]
    for ax, fn, tt in zip(axes, maps, titles):
        ax.plot(s_on["traj"][:, 0], s_on["traj"][:, 2], color="0.8", lw=0.5, label="camera path (corrected)")
        for k, (p, col, lab, f) in enumerate([(p_off, "#eb6834", "drift OFF (ARKit poses as-is)", fn),
                                              (p_on, "#2a78d6", "drift ON (default)", lambda P: P)]):
            for j, r in enumerate(p["rooms"]):
                P = f(np.array(r["polygon"]))
                P = np.vstack([P, P[:1]])
                ax.plot(P[:, 0], P[:, 1], color=col, lw=1.6 if k else 1.0, label=lab if j == 0 else None,
                        ls="-" if k else "--")
        ax.set_title(tt, fontsize=10)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlabel("u (m)"), ax.set_ylabel("v (m)")
    fig.suptitle(f"{cap}: stitched plan, drift correction ON vs OFF | ON: {g_on['rooms']} rooms, "
                 f"{g_on['footprint_m2']} m2, bbox {g_on['bbox_m'][0]:.2f} x {g_on['bbox_m'][1]:.2f} m | OFF: "
                 f"{g_off['rooms']} rooms, {g_off['footprint_m2']} m2, bbox {g_off['bbox_m'][0]:.2f} x "
                 f"{g_off['bbox_m'][1]:.2f} m", fontsize=12)
    fig.tight_layout()
    fig.savefig(B / f"drift_overlay_{cap}.png", dpi=90)
    plt.close(fig)
    return {k: reg.get(k) for k in ("yaw_deg", "residual_rmse_m", "inlier_frac_b", "tiles_summary")}


# ------------------------------------------------------------------------------------------------ 3 repeatability
def registration(cap_a: str, cap_b: str, var_a: str = "drift_on", var_b: str = "drift_on") -> dict:
    ea = scene_evidence(*scene(cap_a, var_a), cap_a)
    eb = scene_evidence(*scene(cap_b, var_b), cap_b)
    reg = register(ea, eb, RegisterParams())
    r = reg.to_dict()
    tiles = local_disagreement(ea, eb, np.array(reg.T2))
    disp = [float(np.linalg.norm(t["displacement_m"])) for t in tiles if t["well_constrained"]]
    r["tiles_summary"] = dict(n=len(disp), median_disp_m=float(np.median(disp)) if disp else None,
                              p90_disp_m=float(np.percentile(disp, 90)) if disp else None)
    r.pop("yaw_curve", None)
    return jsonable(r)


def _width(o):
    w = o.get("width") or {}
    return w.get("value"), w.get("lo"), w.get("hi"), w.get("status")


def opening_table(A: dict, Bp: dict, s: dict) -> list[dict]:
    oa, ob = {o["id"]: o for o in A["openings"]}, {o["id"]: o for o in Bp["openings"]}
    rows = []
    for o in s["openings"]:
        va, la, ha, sa = _width(oa[o["a"]])
        vb, lb, hb, sb = _width(ob[o["b"]])
        z = None
        if None not in (va, vb, la, ha, lb, hb):
            sig = np.hypot((ha - la) / 3.92, (hb - lb) / 3.92)
            z = (vb - va) / sig if sig > 0 else None
        rows.append(dict(a=o["a"], b=o["b"], kind=o["kind"], width_a_cm=None if va is None else round(100 * va, 1),
                         width_b_cm=None if vb is None else round(100 * vb, 1), status_a=sa, status_b=sb,
                         delta_cm=None if o["delta"] is None else round(100 * o["delta"], 1),
                         within_2cm=None if o["delta"] is None else abs(o["delta"]) <= 0.02,
                         z=None if z is None else round(float(z), 2)))
    return rows


def pair_block(cap_a: str, cap_b: str, tag: str, var: str = "drift_on") -> dict:
    reg = registration(cap_a, cap_b, var, var)
    (B / f"registration_{tag}.json").write_text(json.dumps(reg, indent=1, default=float))
    A, Bp = plan(cap_a, var), plan(cap_b, var)
    res = compare(A, Bp, np.array(reg["T2"]), cap_a, cap_b)
    s = res["summary"]
    na, nb = _neighbours(A), _neighbours(Bp)
    pair = {q.a: q.b for q in res["match"].walls}
    wa = {w["id"]: w for w in A["walls"]}
    wb = {w["id"]: w for w in Bp["walls"]}
    walls = []
    for w in res["report"].walls:
        same = None
        if w.delta_m is not None:
            pa, qa = na[w.wall_a]
            same = pair.get(pa) in set(nb[w.wall_b]) and pair.get(qa) in set(nb[w.wall_b])
        walls.append(dict(wall_a=w.wall_a, wall_b=w.wall_b, room_a=w.room_a or (wb[w.wall_b]["room_id"] + " (B)"),
                          len_a_m=None if w.length_a_m is None else round(w.length_a_m, 3),
                          len_b_m=None if w.length_b_m is None else round(w.length_b_m, 3),
                          delta_cm=None if w.delta_m is None else round(100 * w.delta_m, 1),
                          tol_cm=None if w.tol_m is None else round(100 * w.tol_m, 2), status=w.status,
                          same_topology=same, z=None if w.z is None else round(w.z, 2),
                          status_a=wa[w.wall_a]["length"]["status"] if w.wall_a else None,
                          status_b=wb[w.wall_b]["length"]["status"] if w.wall_b else None))
    with open(B / f"walls_{tag}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(walls[0]))
        wr.writeheader()
        wr.writerows(walls)
    openings = opening_table(A, Bp, s)
    n_judged = s["walls_pass"] + s["walls_fail"] + s["walls_unmatched"]
    od = [o for o in openings if o["delta_cm"] is not None]
    oz = [abs(o["z"]) for o in openings if o["z"] is not None]
    m = res["match"]
    ra, rb = {q.a for q in m.rooms}, {q.b for q in m.rooms}
    oa_ = {o["id"]: o for o in A["openings"]}
    ob_ = {o["id"]: o for o in Bp["openings"]}
    un_a = [o for o in m.unmatched.get("openings_a", []) if set(oa_[o].get("room_ids") or []) & ra]
    un_b = [o for o in m.unmatched.get("openings_b", []) if set(ob_[o].get("room_ids") or []) & rb]
    n_open_slots = len(openings) + len(un_a) + len(un_b)   # missed and phantom openings count as misses
    out = dict(
        registration={k: reg.get(k) for k in ("yaw_deg", "residual_rmse_m", "inlier_frac_b", "tiles_summary")},
        rooms_a=s["rooms_a"], rooms_b=s["rooms_b"], rooms_matched=s["rooms_matched"],
        walls_judged=n_judged, walls_pass=s["walls_pass"], walls_fail=s["walls_fail"],
        walls_unmatched=s["walls_unmatched"], walls_not_verifiable=s["walls_not_verifiable"],
        pass_rate=s["walls_pass"] / n_judged if n_judged else None, wilson=s["wilson_pass"],
        median_abs_delta_cm=None if s["wall_delta_median_m"] is None else 100 * s["wall_delta_median_m"],
        walls_within_2cm=s["walls_within_2cm"], same_topology=s["same_topology"],
        different_topology=s["different_topology"], wall_interval_coverage=s["wall_interval_coverage"],
        same_topology_coverage=same_topology_coverage(A, Bp, res),
        footprint_a=s["footprint_a"], footprint_b=s["footprint_b"],
        room_bbox=s["room_bbox"],
        rooms=[dict(a=r.room_a, b=r.room_b, area_a=r.area_a_m2, area_b=r.area_b_m2,
                    ceil_a=r.ceiling_a_m, ceil_b=r.ceiling_b_m, ceiling_status=r.ceiling_status)
               for r in res["report"].rooms],
        openings_a=len(A["openings"]), openings_b=len(Bp["openings"]), openings_matched=len(openings),
        openings_unmatched=s["openings_unmatched"], openings_unmatched_in_matched_rooms=dict(a=un_a, b=un_b),
        openings=openings,
        openings_width_both_measured=len(od),
        openings_within_2cm=sum(bool(o["within_2cm"]) for o in od),
        openings_within_2cm_over_all_slots=f"{sum(bool(o['within_2cm']) for o in od)}/{n_open_slots}",
        opening_median_abs_delta_cm=float(np.median([abs(o["delta_cm"]) for o in od])) if od else None,
        opening_interval_coverage=float(np.mean(np.array(oz) <= 1.96)) if oz else None,
    )
    _overlay(A, Bp, res, np.array(reg["T2"]), tag)
    return jsonable(out)


def _overlay(A, Bp, res, T, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from judge_repeatability import COLOURS, plot_overlay
    fig, ax = plt.subplots(figsize=(11, 11))
    plot_overlay(ax, A, Bp, res, T, f"benchmark LiDAR [{tag}]: B walls on A (grey), coloured by gate status")
    for k, c in COLOURS.items():
        ax.plot([], [], color=c, label=k)
    for w in A["walls"]:
        mid = (np.array(w["p0"]) + np.array(w["p1"])) / 2
        ax.text(mid[0], mid[1], w["id"], fontsize=5, color="0.3")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(B / f"repeat_{tag}.png", dpi=90)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ 4 ceilings
def ceilings() -> dict:
    out = {}
    for cap in CAPS:
        rp = plan(cap)
        out[cap] = [dict(room=r["id"], area_m2=round(r["floor_area"]["value"], 2),
                         value_m=r["ceiling_height"].get("value"), lo=r["ceiling_height"].get("lo"),
                         hi=r["ceiling_height"].get("hi"), status=r["ceiling_height"].get("status"),
                         method=r["ceiling_height"].get("method"))
                    for r in rp["rooms"]]
    return out


# ------------------------------------------------------------------------------------------------ 5 damage
def damage() -> dict:
    out = {}
    for f in sorted(B.glob("*/*/plan.json")):
        rp = json.loads(f.read_text())
        dm = rp.get("damage") or []
        st = [x.get("status") for x in dm]
        out[f"{f.parent.parent.name}/{f.parent.name}"] = dict(
            detections=len(dm), confirmed=st.count("confirmed"), review=st.count("review"),
            scope_items=len(rp.get("scope_items") or []),
            items=[dict(id=x.get("id"), cls=x.get("class") or x.get("type") or x.get("kind"), status=x.get("status"),
                        confidence=x.get("confidence"), room=x.get("room_id")) for x in dm])
    return out


def main() -> None:
    res = dict(timing=timing())
    print(json.dumps(res["timing"], indent=1))
    res["ceilings"] = ceilings()
    res["damage"] = damage()
    print(json.dumps(res["damage"], indent=1, default=str)[:3000])
    res["repeat_pair"] = pair_block("with_ceiling", "floor_only", "with_ceiling__floor_only")
    print({k: v for k, v in res["repeat_pair"].items() if k not in ("room_bbox", "openings", "rooms")})
    for big in ("with_ceiling", "floor_only"):
        try:
            res[f"single_room_vs_{big}"] = pair_block(big, "single_room", f"{big}__single_room")
        except Exception as e:  # noqa: BLE001 - record, do not hide
            res[f"single_room_vs_{big}"] = dict(error=f"{type(e).__name__}: {e}")
        print(big, {k: v for k, v in res[f"single_room_vs_{big}"].items() if k not in ("room_bbox", "openings")})
    if all((d(c, "drift_on_rerun2") / "plan.json").exists() for c in ("with_ceiling", "floor_only")):
        # the same repeat pair from the second (sequential) run of the same command: run-to-run spread of the gate
        res["repeat_pair_rerun2"] = pair_block("with_ceiling", "floor_only", "with_ceiling__floor_only__rerun2",
                                               "drift_on_rerun2")
        print("rerun2", {k: v for k, v in res["repeat_pair_rerun2"].items()
                         if k not in ("room_bbox", "openings", "rooms")})
    res["drift"] = drift_block()
    res["drift_module_single_room"] = drift_module_summary("single_room")
    (B / "results.json").write_text(json.dumps(jsonable(res), indent=1, default=float))
    print("wrote", B / "results.json")


if __name__ == "__main__":
    main()
