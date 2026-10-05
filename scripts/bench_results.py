#!/usr/bin/env python
"""Score one fresh benchmark folder (scripts/bench_fresh.sh) and write results/: summary.json, tables.md, plan images.

    env -u PYTHONPATH python scripts/bench_results.py outputs/benchmark/fresh_<HHMM> [--out results]

Scorers reused unchanged:
  * eval_own_capture.py   own house vs tape (photo dim, photo lit, the video runs), k65 vs the simulator's GT
                          (geometric wall pairing on the GT polygons; walls, openings, ceilings, calibration)
  * eval_door_stitch.py   k65 whole-plan IoU against the GT rooms, and the overlay figure
  * bench_tier_ref.py     sample video plans vs the LiDAR plan of the same capture (REFERENCE-BASED, no GT there)
  * bench_lidar_eval.py   LiDAR repeat pair (with_ceiling vs floor_only) and drift on/off
Nothing is made up: a run that is missing becomes "missing" in summary.json and in the tables.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
O = ROOT / "outputs"
PY = sys.executable


def _first(*cands: Path) -> Path:
    return next((c for c in cands if c.exists()), cands[-1])


OWN_GT = _first(ROOT / "data/own_house/gt/ground_truth.csv", O / "own_house/capture/gt/ground_truth.csv")
OWN_POLY = OWN_GT.with_name("gt_polygons.json")
K65_GT = _first(ROOT / "data/k65/gt/ground_truth.csv", O / "sim/k65v2/capture_iphone15/gt/ground_truth.csv")
K65_JSON = K65_GT.with_name("sim_gt.json")
K65_SESSION = O / "sim/k65v2/old_v21/session.json"
CEILING_TAPE_M = 2.6289
OWN_VIDEO = ["own_2328", "own_b1", "own_b2", "own_a1", "own_a2", "own_p1", "own_pp1", "own_p2", "own_pp2"]
SAMPLE_VIDEO = {"single_room": ["sr_r1", "sr_r2", "sr_r3", "sr_a"], "floor_only": ["fo_a", "fo_d066"],
                "with_ceiling": ["wc_a"]}
SAMPLES = ("single_room", "floor_only", "with_ceiling")
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


# ------------------------------------------------------------------------------------------------ helpers
def elapsed_s(run: Path) -> float | None:
    """Wall-clock seconds of a run from /usr/bin/time -v (time.log)."""
    t = run / "time.log"
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t.read_text()) if t.exists() else None
    if not m:
        return None
    s = 0.0
    for part in m.group(1).split(":"):
        s = 60 * s + float(part)
    return round(s, 1)


def max_rss_gb(run: Path) -> float | None:
    t = run / "time.log"
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t.read_text()) if t.exists() else None
    return round(int(m.group(1)) / 1024 ** 2, 2) if m else None


def mval(m: dict | None) -> dict | None:
    if not m or m.get("value") is None:
        return None
    lo, hi = (m.get("ci95") or [None, None])[:2]
    return dict(value=round(m["value"], 4), lo=None if lo is None else round(lo, 4),
                hi=None if hi is None else round(hi, 4))


def plan_summary(p: dict) -> dict:
    kinds = [o.get("kind") for o in p.get("openings", [])]
    return dict(rooms=len(p["rooms"]), room_names=[r.get("name") or r.get("label") for r in p["rooms"]],
                walls=len(p["walls"]), openings=len(kinds), doors=kinds.count("door"), windows=kinds.count("window"),
                footprint_m2=mval(p.get("footprint_area")),
                ceilings_m=[dict(room=r["id"], **(mval(r.get("ceiling_height")) or {})) for r in p["rooms"]],
                warnings=(p.get("meta") or {}).get("warnings"), reliability=(p.get("meta") or {}).get("reliability"))


def run_info(run: Path) -> dict:
    """Timing and status of one run folder made by bench_fresh.sh (one.sh)."""
    d = dict(path=str(run.relative_to(O)) if run.is_relative_to(O) else str(run))
    for k in ("started", "finished", "exit_code"):
        f = run / f"{k}.txt"
        d[k] = f.read_text().strip() if f.exists() else None
    d["wall_clock_s"], d["max_rss_gb"] = elapsed_s(run), max_rss_gb(run)
    rr = run / "run_report.json"
    if rr.exists():
        r = json.loads(rr.read_text())
        d["pipeline_runtime_s"] = r.get("runtime_s") or (r.get("timing") or {}).get("total_s")
        src = Path(r["replanned_from"]) if r.get("replanned_from") else None
        if src is not None:                                   # a replan of a replay of a cached live video run
            d["replanned_from"] = str(src)
            rp = load(src / "REPLAY.json")
            live = Path(rp["run"]) if rp else src            # no REPLAY.json: src is the cached live run itself
            lr = load(live / "run_report.json") if live else None
            if lr:
                d["cached_live_run"], d["cached_live_run_s"] = str(live), lr.get("runtime_s")
    return d


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def shoelace(poly) -> float:
    a = np.asarray(poly, float)
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


# ------------------------------------------------------------------------------------------------ GT scoring
def eval_gt(gt: Path, gt_json: Path, plans: dict[str, Path], out: Path) -> dict:
    """eval_own_capture.py on every plan that exists; returns its own_eval.json (label -> result)."""
    plans = {k: v for k, v in plans.items() if v.exists()}
    if not plans:
        return {}
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, ROOT / "scripts/eval_own_capture.py", "--gt", gt, "--gt-json", gt_json, "--out", out]
    for k, v in plans.items():
        cmd += ["--plan", f"{k}={v}"]
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=ENV, capture_output=True, text=True)
    (out / "eval.log").write_text(r.stdout + r.stderr)
    return load(out / "own_eval.json") or {}


def n_gt_walls(gt_csv: Path) -> int:
    from floorplan.benchmark.gt_eval import read_gt
    return sum(1 for it in read_gt(gt_csv) if it.kind == "wall")


def gt_block(res: dict, tier: str, plan: dict | None, all_walls: int | None = None) -> dict:
    """The gate numbers of one plan scored against GT (eval_own_capture result). all_walls: GT walls of the whole
    capture; a video or LiDAR walk covers every room, so a GT room the plan does not have counts its walls as missed
    (the scorer only lists walls of the GT rooms it paired)."""
    rows, g = res["rows"], res["gates"]
    walls = [r for r in rows if r["kind"] == "wall"]
    meas = [r for r in walls if r["value"] is not None]
    seen = {g for g, q in (res.get("room_pairs") or {}).items() if q}   # GT rooms the plan has (photo: one room)
    ops = [r for r in rows if r["kind"] in ("door_width", "window_width") and (not seen or r["room"] in seen)]
    ceil = [r for r in rows if r["kind"] == "ceiling_height" and (not seen or r["room"] in seen)]
    gto = [r for r in ops if r["gt"] is not None]                                # GT openings in the plan's rooms
    found = [r for r in gto if not str(r.get("note") or "").startswith("MISSED")]   # placed on the right wall
    wm = [r for r in found if r["value"] is not None and r["err"] is not None]     # ... and given a width
    tol = {"photo": 0.08, "video": 0.03}.get(tier)
    n_walls = all_walls if (all_walls and tier in ("video", "lidar")) else len(walls)
    b = dict(
        walls=dict(gt=n_walls, measured=len(meas), missed=n_walls - len(meas),
                   tol_rel=tol, within_tol_frac=(g.get("wall_gate") or {}).get("pass_frac"),
                   within_tol_n=sum(1 for r in meas if tol is not None and r["rel"] <= tol),
                   median_rel_err=_r(g.get("wall_median_rel_err")), median_abs_err_m=_r(g.get("wall_median_abs_err_m")),
                   within_1cm=sum(1 for r in meas if abs(r["err"]) <= 0.01),
                   within_2cm=sum(1 for r in meas if abs(r["err"]) <= 0.02),
                   ci95_covers=f"{sum(1 for r in meas if r['inside'])}/{len(meas)}"),
        openings=dict(gt=len(gto), found=len(found), width_measured=len(wm), missed=len(gto) - len(found),
                      phantom=sum(1 for r in ops if r["gt"] is None), gt_rooms_in_plan=sorted(seen),
                      width_err_m=[dict(item=f"{r['room']} {r['item']}", err=_r(r["err"])) for r in wm],
                      width_within_2cm=sum(1 for r in wm if abs(r["err"]) <= 0.02),
                      width_median_abs_err_m=_r(st.median(abs(r["err"]) for r in wm)) if wm else None),
        ceiling=[dict(room=r["room"], value=_r(r["value"]), lo=_r(r["lo"]), hi=_r(r["hi"]), gt=r["gt"],
                      err_m=_r(r["err"]), inside=r["inside"]) for r in ceil],
        calibration_coverage95=(g.get("calibration") or {}).get("coverage_95"),
        stitch=g.get("stitch"),
        rows=[dict(room=r["room"], item=r["item"], kind=r["kind"], value=_r(r["value"]), lo=_r(r["lo"]),
                   hi=_r(r["hi"]), gt=r["gt"], err_m=_r(r["err"]), rel=_r(r["rel"]), inside=r["inside"])
              for r in rows])
    if plan is not None:
        b["plan"] = plan_summary(plan)
    return b


def _r(x, n=4):
    return None if x is None else round(float(x), n)


def wall_values(res: dict) -> dict:
    return {f"{r['room']} {r['item']}": r["value"] for r in res["rows"] if r["kind"] == "wall"}


def item_values(res: dict) -> dict:
    return {f"{r['room']} {r['item']}": (r["value"], r["gt"]) for r in res["rows"] if r["gt"] is not None}


# ------------------------------------------------------------------------------------------------ images
def save_png(src: Path, dst: Path, max_w: int = 1600) -> str | None:
    if not src.exists():
        return None
    from PIL import Image
    im = Image.open(src)
    if im.mode not in ("RGB", "L"):
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1] if im.mode in ("RGBA", "LA") else None)
        im = bg
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, optimize=True)
    if dst.stat().st_size > 1_000_000:                       # keep every image under 1 MB
        im.convert("P", palette=Image.ADAPTIVE, colors=128).save(dst, optimize=True)
    return str(dst.relative_to(dst.parent.parent))   # plans/... or overlays/..., relative to results/


def plan_images(run: Path, name: str, out: Path) -> dict:
    return dict(presentation=save_png(run / "plan_presentation.png", out / "plans" / f"{name}_presentation.png"),
                technical=save_png(run / "plan.png", out / "plans" / f"{name}_technical.png"))


def side_by_side(left: Path, right: Path, dst: Path, labels: tuple[str, str]) -> str | None:
    if not (left.exists() and right.exists()):
        return None
    from PIL import Image, ImageDraw
    a, b = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
    h = 900
    a = a.resize((round(a.width * h / a.height), h))
    b = b.resize((round(b.width * h / b.height), h))
    im = Image.new("RGB", (a.width + b.width + 30, h + 50), "white")
    im.paste(a, (0, 50))
    im.paste(b, (a.width + 30, 50))
    dr = ImageDraw.Draw(im)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=26)
    except (TypeError, OSError):
        font = None
    dr.text((10, 12), labels[0], fill="black", font=font)
    dr.text((a.width + 40, 12), labels[1], fill="black", font=font)
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, optimize=True)
    return str(dst.relative_to(dst.parent.parent))


# ------------------------------------------------------------------------------------------------ tables.md
def _pct(x, n=1):
    return "—" if x is None else f"{100 * x:+.{n}f}%"


def _fp(b: dict) -> str:
    f = b.get("footprint") or {}
    if f.get("rel_err") is not None:
        return f"{f['plan_m2']['value']:.2f} vs {f['tape_outline_m2']:.2f} m² ({_pct(f['rel_err'])})"
    s = b.get("stitch") or {}
    fp = s.get("footprint") if isinstance(s, dict) else None
    if fp and fp.get("gt"):
        return f"{fp['value']:.2f} vs {fp['gt']:.2f} m² ({_pct(fp['rel_err'])})"
    return "—"


def _ceil(b: dict) -> str:
    c = [x for x in b.get("ceiling", []) if x.get("value") is not None]
    if not c:
        return "not measured"
    return ", ".join(f"{x['value']:.2f} ({100 * x['err_m']:+.0f} cm)" for x in c)


def _time(b: dict) -> str:
    r = b.get("run") or {}
    t = r.get("pipeline_runtime_s") or r.get("wall_clock_s")
    if t is not None and r.get("cached_live_run_s"):
        return f"{t:.0f} replan (live run {r['cached_live_run_s']:.0f})"
    return "—" if t is None else f"{t:.0f}"


def _pctu(x, n=1):
    return "—" if x is None else f"{100 * x:.{n}f}%"


def _gt_row(name: str, tier: str, how: str, b: dict) -> str:
    w, o = b["walls"], b["openings"]
    lid = w.get("lidar_within_max_1cm_0p5pct")
    if lid is not None:
        within = f"{lid}/{w['gt']} (±max(1 cm, 0.5%))"
    else:
        within = f"{w['within_tol_n']}/{w['gt']} (±{100 * w['tol_rel']:.0f}%)"
    wmed = o["width_median_abs_err_m"]
    return (f"| {name} | {tier} | {how} | {within} | {_pctu(w['median_rel_err'])} | {w['missed']} | {w['ci95_covers']} "
            f"| {_fp(b)} | {_ceil(b)} | {o['found']}/{o['gt']}, {o['phantom']} phantom | "
            f"{'—' if wmed is None else f'{100 * wmed:.1f} cm'} ({o['width_within_2cm']} within 2 cm) | {_time(b)} |")


def write_tables(summ: dict, path: Path) -> None:
    L = [f"# Benchmark tables ({summ['code']})", "",
         "Numbers from `results/summary.json` (scripts/bench_results.py). Wall gate: photo ±8%, video ±3% of the "
         "tape / simulator length; MISSED walls count as failures. Footprint against the tape outline (own house) or "
         "the simulator's rooms (k65). Openings: GT openings in the rooms the plan has, found = placed on the right "
         "wall; width error over the found ones that carry a width. Time = pipeline runtime of the run (s).", "",
         "## 1. Gates against ground truth", "",
         "| Capture | Tier | Run | Walls within tol | Median wall err | Missed walls | GT inside 95% CI | Footprint "
         "| Ceiling (tape 2.6289 m own house) | Openings found | Width err median | Time s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    oh = summ["tiers"]["own_house"]["runs"]
    for lab, how in (("photo_dim", "live"), ("photo_lit", "live")):
        if lab in oh:
            L.append(_gt_row("own house (dim)" if lab.endswith("dim") else "own house (lit)", "photo", how, oh[lab]))
    for lab, b in oh.items():
        if lab.startswith("video_own"):
            L.append(_gt_row("own house take1", "video", f"replay {lab[6:]}", b))
    if "video_live" in oh:
        L.append(_gt_row("own house take1", "video", "live", oh["video_live"]))
    for lab, b in summ["tiers"]["k65_sim"]["runs"].items():
        how = ("replan of the cached 4 Oct run (older video front end)" if lab.startswith("video") else
               "replan of cached scene" if "replan" in (b.get("run") or {}).get("path", "") else "live")
        iou = b.get("plan_iou_vs_gt")
        L.append(_gt_row(f"k65 (sim){'' if iou is None else f', IoU {iou}'}", lab.split("_")[0], how, b))
    rep = summ["tiers"]["own_house"]["repeatability"]
    L += ["", "## 2. Repeatability", ""]
    if "photo_dim_vs_lit" in rep:
        r = rep["photo_dim_vs_lit"]
        L += ["### Own house, photo: the same room captured twice (dim light vs lit)", "",
              "| Item | Dim | Lit | Tape | Dim − lit | |dim − lit| / mean |", "|---|---|---|---|---|---|"]
        for x in r["rows"]:
            if x["dim"] is None and x["lit"] is None:
                continue
            f = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
            dm = "—" if x["diff_m"] is None else f"{100 * x['diff_m']:+.1f} cm"
            dr = "—" if x["diff_rel"] is None else f"{100 * x['diff_rel']:.1f}%"
            L.append(f"| {x['item']} | {f(x['dim'])} | {f(x['lit'])} | {x['gt']:.3f} | {dm} | {dr} |")
        L += ["", f"Walls in both: {r['walls_both']}; median |dim − lit| {100 * (r['wall_median_abs_diff_m'] or 0):.1f} cm, "
                  f"max {100 * (r['wall_max_abs_diff_m'] or 0):.1f} cm, median relative {_pctu(r['wall_median_rel_diff'])}; "
                  f"{r['walls_within_8pct_of_each_other']} of {r['walls_both']} within 8% of each other.", ""]
    if "video_take1_run_to_run" in rep:
        r = rep["video_take1_run_to_run"]
        L += [f"### Own house, video take1: {r['runs']} cached DPVO/MoGe-2 runs replayed with the HEAD code", "",
              "| Item | Tape | Found in | Mean | SD | Min | Max | Range |", "|---|---|---|---|---|---|---|---|"]
        g = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        for x in r["per_item"]:
            if not x["found"]:
                continue
            L.append(f"| {x['item']} | {x['gt']:.3f} | {x['found']}/{x['runs']} | {g(x['mean'])} | {g(x['std'])} | "
                     f"{g(x['min'])} | {g(x['max'])} | {g(x['range'])} |")
        L += ["", f"Median wall error per run: {_pctu(r['median_rel_err_range'][0])} to {_pctu(r['median_rel_err_range'][1])}"
                  if r.get("median_rel_err_range") else "",
              f"Footprint per run: {r['footprint_m2_range'][0]:.2f} to {r['footprint_m2_range'][1]:.2f} m²"
              if r.get("footprint_m2_range") else "", ""]
    sb = summ["tiers"]["samples"]["captures"]
    L += ["## 3. Sample captures (no ground truth)", "",
          "LiDAR runs live; video = replays of cached tracks scored against the LiDAR plan of the same capture "
          "(REFERENCE-BASED, not ground truth; room dimensions within ±3%).", "",
          "| Capture | LiDAR rooms / walls / openings | LiDAR footprint m² | LiDAR time s | Same plan as the 10:04 run | "
          "Video run | Rooms matched | Room dims within 3% | Median dim err | Footprint vs LiDAR | One stitched plan | Video plan s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cap in SAMPLES:
        e = sb.get(cap) or {}
        lp = (e.get("lidar") or {}).get("plan")
        if not isinstance(lp, dict):
            L.append(f"| {cap} | missing | | | | | | | | | | |")
            continue
        det = e.get("lidar_vs_1004_run") or {}
        same = "—" if not det else ("yes (max wall diff 0.0 cm)" if det.get("max_abs_wall_diff_m") == 0 else
                                    f"no (max wall diff {100 * (det.get('max_abs_wall_diff_m') or 0):.1f} cm)"
                                    if det.get("same_wall_count") else "no (wall count differs)")
        head = (f"| {cap} | {lp['rooms']} / {lp['walls']} / {lp['openings']} | {lp['footprint_m2']['value']:.2f} | "
                f"{_time(e['lidar'])} | {same} |")
        vids = e.get("video_vs_lidar") or {}
        if not vids:
            L.append(head + " — | | | | | | |")
        for n, v in vids.items():
            d, fp, rm = v.get("dims") or {}, v.get("footprint") or {}, v.get("rooms") or {}
            L.append(head + f" {n} | {rm.get('matched')}/{rm.get('reference')} | {d.get('within_tol', '—')}/{d.get('n', '—')} | "
                     f"{_pctu(d.get('median_abs_rel_err'))} | {_pct(fp.get('rel_err'))} | "
                     f"{(v.get('stitch') or {}).get('one_stitched_plan')} | {_time(v)} |")
            head = "| | | | | |"
    rp = sb.get("lidar_repeat_pair")
    if isinstance(rp, dict):
        w = [round(x, 3) for x in rp.get("wilson") or [] if x is not None]
        L += ["", "LiDAR repeat pair (with_ceiling vs floor_only: two scans of the same flat, registered from the scene "
                  "geometry; a wall passes when the two lengths agree within max(1 cm, 0.5%), unmatched walls fail):", "",
              f"- walls judged {rp.get('walls_judged')}, pass {rp.get('walls_pass')}, fail {rp.get('walls_fail')}, "
              f"unmatched {rp.get('walls_unmatched')}; pass rate {_pctu(rp.get('pass_rate'))} (Wilson 95% {w}); "
              f"median |delta| {rp.get('median_abs_delta_cm') or 0:.2f} cm; within 2 cm {rp.get('walls_within_2cm')}",
              f"- rooms {rp.get('rooms_a')} / {rp.get('rooms_b')}, matched {rp.get('rooms_matched')}; footprint "
              f"{_r(rp.get('footprint_a'), 2)} / {_r(rp.get('footprint_b'), 2)} m²; same-topology walls "
              f"{(rp.get('same_topology') or {}).get('passed')}/{(rp.get('same_topology') or {}).get('n')} pass "
              f"(median {100 * ((rp.get('same_topology') or {}).get('median_m') or 0):.1f} cm)",
              f"- openings matched {rp.get('openings_matched')}, widths within 2 cm "
              f"{rp.get('openings_within_2cm_over_all_slots')}, median |delta| {rp.get('opening_median_abs_delta_cm')} cm"]
    dr = sb.get("drift_on_off")
    if isinstance(dr, dict):
        L += ["", "## 4. Drift on / off (LiDAR, stitched footprint, same capture)", "",
              "| Capture | Footprint on m² | Footprint off m² | Rooms on / off | Walls same within gate | Median wall delta cm |",
              "|---|---|---|---|---|---|"]
        for cap, x in dr.items():
            on, off = (x.get("drift_on") or {}).get("plan") or {}, (x.get("drift_off") or {}).get("plan") or {}
            c = x.get("on_vs_off_plan") or {}
            L.append(f"| {cap} | {on.get('footprint_m2')} | {off.get('footprint_m2')} | {on.get('rooms')} / {off.get('rooms')} | "
                     f"{c.get('walls_same_within_gate')}/{c.get('walls_judged')} | {_r(c.get('median_abs_delta_cm'), 1)} |")
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ main
def orient_by_openings(plan: dict, geom: dict, pm: dict, tie_iou: float = 0.02) -> dict:
    """A near-rectangular room fits its tape outline almost as well turned 180 deg (own lit: 0.878 both ways), so
    the IoU fit can come out upside down. Between the fit and the fit turned 180 deg about the plan's centre, keep
    the one whose doors and windows land on tape walls holding that kind of opening; the IoU must stay within
    tie_iou. The scorer (wall_match) breaks the same tie with the door walls."""
    import eval_door_stitch as EDS
    wm = EDS.wm
    glob = pm.get("glob")
    if not glob:
        return pm
    M = np.diag([1.0, -1.0]) if pm.get("mirror") else np.eye(2)
    R, t = np.asarray(glob["R"], float), np.asarray(glob["t"], float)
    polys = [np.asarray(r["polygon"], float) @ M.T for r in plan["rooms"] if len(r.get("polygon") or []) >= 3]
    if not polys:
        return pm
    walls = [(name, w["a"], w["b"], room) for room, g in geom.items() for name, w in g["walls"].items()]
    held = {(room, o["wall"], o["kind"]) for room, g in geom.items() for o in g["openings"]}

    def seg_d(q, a, b):
        ab = b - a
        s = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        return float(np.linalg.norm(q - (a + s * ab)))

    def score(R_, t_):
        agree = 0
        for o in plan.get("openings", []):
            ctr = o.get("position") or o.get("center")
            if ctr is None or o.get("kind") not in ("door", "window"):
                continue
            q = np.asarray(ctr, float) @ M.T @ R_.T + t_
            name, a, b, room = min(walls, key=lambda w: seg_d(q, w[1], w[2]))
            agree += (room, name, o["kind"]) in held
        P = [p @ R_.T + t_ for p in polys]
        iou = np.mean([max(wm.iou(p, np.asarray(g["poly"], float)) for g in geom.values()) for p in P])
        return agree, float(iou)

    a0, i0 = score(R, t)
    R1 = -R                                     # turned 180 deg; its shift fitted again: centres matched, then the
    from shapely.geometry import Polygon      # best IoU on a 2 cm grid, +-0.6 m around the matched area centres
    gc = np.mean([Polygon(np.asarray(g["poly"], float)).centroid.coords[0] for g in geom.values()], axis=0)
    t1 = gc - np.mean([Polygon(P @ R1.T).centroid.coords[0] for P in polys], axis=0)
    best = score(R1, t1)[1], t1
    for dx in np.arange(-0.6, 0.601, 0.02):
        for dy in np.arange(-0.6, 0.601, 0.02):
            tt = t1 + np.array([dx, dy])
            P = [q @ R1.T + tt for q in polys]
            i = np.mean([max(wm.iou(q, np.asarray(g["poly"], float)) for g in geom.values()) for q in P])
            if i > best[0]:
                best = i, tt
    t1 = best[1]
    a1, i1 = score(R1, t1)
    out = dict(pm)
    out["orientation"] = dict(agree_fit=a0, agree_turned=a1, iou_fit=round(i0, 3), iou_turned=round(i1, 3), turned=False)
    if a1 > a0 and i1 >= i0 - tie_iou:
        out["glob"] = dict(glob, R=R1, t=t1)
        out["iou"] = round(i1, 3)
        out["orientation"]["turned"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fresh", type=Path, help="outputs/benchmark/fresh_<HHMM>")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--skip-lidar-pairs", action="store_true", help="skip the LiDAR registration (repeat pair, drift)")
    a = ap.parse_args()
    F, R = a.fresh.absolute(), a.out.absolute()
    S = F / "score"
    S.mkdir(exist_ok=True)
    (R / "plans").mkdir(parents=True, exist_ok=True)
    (R / "overlays").mkdir(parents=True, exist_ok=True)
    code = (F / "CODE.txt").read_text().strip()
    summ = dict(code=code, folder=str(F.relative_to(O)) if F.is_relative_to(O) else str(F),
                ceiling_tape_m=CEILING_TAPE_M, tiers={})
    images = {}

    # ---------------- own house: photo dim + lit (live), video take1 (replans of cached tracks, + live if made)
    own_plans = {"photo_dim": F / "photo/own_dim/plan.json", "photo_lit": F / "photo/own_lit/plan.json"}
    for n in OWN_VIDEO:
        own_plans[f"video_{n}"] = F / f"video/{n}/plan.json"
    own_plans["video_live"] = F / "video_live/take1/plan.json"
    own = eval_gt(OWN_GT, OWN_POLY, own_plans, S / "own")
    polys = {r["id"]: shoelace(r["polygon"]) for r in json.loads(OWN_POLY.read_text())["rooms"]}
    oh = {}
    for label, res in own.items():
        tier = label.split("_")[0]
        plan = load(own_plans[label])
        b = gt_block(res, tier, plan, n_gt_walls(OWN_GT))
        matched = [g for g, p in (res.get("room_pairs") or {}).items() if p]
        if tier == "video":                 # the walkthrough covers every outlined room, matched or not
            matched = sorted(polys)
        if plan and matched:
            gt_area = sum(polys.get(g, 0.0) for g in matched)
            fp = mval(plan.get("footprint_area"))
            b["footprint"] = dict(plan_m2=fp, tape_outline_m2=round(gt_area, 3), gt_rooms=matched,
                                  rel_err=_r(fp["value"] / gt_area - 1) if fp and gt_area else None,
                                  inside=bool(fp and fp["lo"] is not None and fp["lo"] <= gt_area <= fp["hi"]),
                                  note="tape outline = gt_polygons.json (tape lengths on the sketch layout); photo: the "
                                       "photographed room, video: room + hall + kitchen (no outline for bath/passage)")
        run = own_plans[label].parent
        b["run"] = run_info(run)
        oh[label] = b
    # repeatability: photo dim vs lit, the same room taken twice
    rep = {}
    if "photo_dim" in own and "photo_lit" in own:
        vd, vl = item_values(own["photo_dim"]), item_values(own["photo_lit"])
        rows = []
        for k in vd:
            (d, gt), (l, _) = vd[k], vl.get(k, (None, None))
            if d is not None and l is not None:
                rows.append(dict(item=k, dim=_r(d), lit=_r(l), gt=gt, diff_m=_r(d - l),
                                 diff_rel=_r(abs(d - l) / ((d + l) / 2))))
            else:
                rows.append(dict(item=k, dim=_r(d), lit=_r(l), gt=gt, diff_m=None, diff_rel=None))
        wr = [r for r in rows if r["item"].split()[1].startswith("W") and r["diff_m"] is not None]
        rep["photo_dim_vs_lit"] = dict(
            rows=rows, walls_both=len(wr),
            wall_median_abs_diff_m=_r(st.median(abs(r["diff_m"]) for r in wr)) if wr else None,
            wall_max_abs_diff_m=_r(max(abs(r["diff_m"]) for r in wr)) if wr else None,
            wall_median_rel_diff=_r(st.median(r["diff_rel"] for r in wr)) if wr else None,
            walls_within_8pct_of_each_other=sum(1 for r in wr if r["diff_rel"] <= 0.08))
    vids = [f"video_{n}" for n in OWN_VIDEO if f"video_{n}" in own]
    if vids:
        per = {}
        for lab in vids:
            for k, (v, gt) in item_values(own[lab]).items():
                per.setdefault(k, dict(gt=gt, values=[]))["values"].append(v)
        rows = []
        for k, d in per.items():
            vals = [v for v in d["values"] if v is not None]
            rows.append(dict(item=k, gt=d["gt"], runs=len(d["values"]), found=len(vals),
                             mean=_r(np.mean(vals)) if vals else None, std=_r(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                             min=_r(min(vals)) if vals else None, max=_r(max(vals)) if vals else None,
                             range=_r(max(vals) - min(vals)) if vals else None))
        runs = [dict(run=lab, walls_within_3pct=oh[lab]["walls"]["within_tol_n"], walls_gt=oh[lab]["walls"]["gt"],
                     walls_missed=oh[lab]["walls"]["missed"], median_rel_err=oh[lab]["walls"]["median_rel_err"],
                     footprint_m2=(oh[lab].get("footprint") or {}).get("plan_m2"),
                     rooms=oh[lab]["plan"]["rooms"] if "plan" in oh[lab] else None,
                     ceiling=[c["value"] for c in oh[lab]["ceiling"]]) for lab in vids]
        meds = [r["median_rel_err"] for r in runs if r["median_rel_err"] is not None]
        fps = [r["footprint_m2"]["value"] for r in runs if r["footprint_m2"]]
        rep["video_take1_run_to_run"] = dict(
            runs=len(vids), note="replays of 9 cached DPVO/MoGe-2 take1 runs with the HEAD code (deterministic per "
                                 "cache); the spread is the run-to-run spread of the live tracker", per_item=rows,
            per_run=runs, median_rel_err_range=[_r(min(meds)), _r(max(meds))] if meds else None,
            footprint_m2_range=[_r(min(fps)), _r(max(fps))] if fps else None)
    summ["tiers"]["own_house"] = dict(gt="tape (outputs/own_house/capture/gt/ground_truth.csv); ceiling 2.6289 m",
                                      runs=oh, repeatability=rep)
    import eval_door_stitch as EDS
    from floorplan.benchmark.gt_eval import read_gt_geometry
    own_geom = read_gt_geometry(OWN_GT, OWN_POLY)
    for lab, p in own_plans.items():
        if p.exists() and lab in ("photo_dim", "photo_lit", "video_own_2328", "video_live"):
            images[f"own_{lab}"] = plan_images(p.parent, f"own_{lab}", R)
            try:   # plan rooms (red, dashed) over the tape outline under one rigid transform (eval_door_stitch.py)
                plan = json.loads(p.read_text())
                geom = {g: v for g, v in own_geom.items() if g in (oh.get(lab, {}).get("openings", {})
                                                                   .get("gt_rooms_in_plan") or own_geom)}
                pm = orient_by_openings(plan, geom, EDS.plan_metrics(plan, geom))
                png = S / "own" / f"{lab}_vs_tape.png"
                EDS.overlay(plan, geom, pm, f"own house {lab}: plan (red dashed; dots: its doors and windows) over the tape outline "
                            f"(filled; sketch layout + tape lengths); IoU {pm['iou']}", png)
                images[f"own_{lab}"]["overlay_vs_tape"] = save_png(png, R / "overlays" / f"own_{lab}_vs_tape.png")
                oh[lab]["plan_iou_vs_tape_outline"] = pm["iou"]
            except Exception as e:  # noqa: BLE001
                images[f"own_{lab}"]["overlay_vs_tape"] = f"failed: {type(e).__name__}: {e}"

    # ---------------- k65 (simulated flat, exact GT): photo (live), LiDAR (live or replan), video (replay)
    k65_runs = {"photo_k65": F / "photo/k65", "lidar_k65": F / "lidar/k65", "video_k65": F / "video/k65"}
    if not (k65_runs["lidar_k65"] / "plan.json").exists() and (F / "lidar/k65_replan/plan.json").exists():
        k65_runs["lidar_k65"] = F / "lidar/k65_replan"
    k65 = eval_gt(K65_GT, K65_JSON, {k: v / "plan.json" for k, v in k65_runs.items()}, S / "k65")
    kb = {}
    for lab, res in k65.items():
        run = k65_runs[lab]
        b = gt_block(res, lab.split("_")[0], load(run / "plan.json"), n_gt_walls(K65_GT))
        if lab.startswith("lidar"):
            meas = [r for r in res["rows"] if r["kind"] == "wall" and r["value"] is not None]
            b["walls"]["lidar_within_max_1cm_0p5pct"] = sum(1 for r in meas if abs(r["err"]) <= max(0.01, 0.005 * r["gt"]))
        cmd = [PY, ROOT / "scripts/eval_door_stitch.py", "--run", run, "--gt", K65_GT,
               "--png", S / "k65" / f"{lab}_vs_gt.png"]
        if lab.startswith("photo") and K65_SESSION.exists():   # simulator photo poses: the door table
            cmd += ["--session", K65_SESSION]
        r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=ENV, capture_output=True, text=True)
        try:
            ds = json.loads(r.stdout)
            b["plan_iou_vs_gt"] = ds.get("iou")
            if "doors" in ds:
                b["door_stitch_doors"] = ds["doors"]
        except json.JSONDecodeError:
            b["plan_iou_vs_gt"] = f"scorer failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?'}"
        gtf = json.loads(K65_JSON.read_text())
        b["run"] = run_info(run)
        kb[lab] = b
        images[f"k65_{lab}"] = plan_images(run, f"k65_{lab}", R)
        ov = save_png(S / "k65" / f"{lab}_vs_gt.png", R / "overlays" / f"k65_{lab}_vs_gt.png")
        if ov:
            images[f"k65_{lab}"]["overlay_vs_gt"] = ov
    summ["tiers"]["k65_sim"] = dict(gt="simulator geometry (outputs/sim/k65v2/capture_iphone15/gt)", runs=kb)

    # ---------------- samples: LiDAR live (the reference), video replans scored against it (REFERENCE-BASED)
    import bench_tier_ref as TR
    sb = {}
    for cap in SAMPLES:
        lid = F / "lidar" / cap
        lp = load(lid / "plan.json")
        entry = dict(lidar=dict(run=run_info(lid), plan=plan_summary(lp) if lp else "missing"))
        images[f"sample_{cap}_lidar"] = plan_images(lid, f"sample_{cap}_lidar", R)
        vids = {}
        for n in SAMPLE_VIDEO[cap]:
            vp = F / "video" / n / "plan.json"
            if not (vp.exists() and lp):
                continue
            res = TR.score(vp, lid / "plan.json", "video", lid / "scene", None)
            (S / f"tier_ref_{n}.json").write_text(json.dumps(res, indent=1, default=float))
            vids[n] = dict(dims=res.get("dims"), areas=res.get("areas"), footprint=res.get("footprint"),
                           rooms=res.get("rooms"), adjacency={k: v for k, v in (res.get("adjacency") or {}).items()
                                                               if k not in ("pairs", "missing")},
                           overlaps={k: v for k, v in (res.get("overlaps") or {}).items() if k != "pairs"},
                           stitch=res.get("stitch"), calibration=res.get("calibration"),
                           plan=plan_summary(load(vp)), run=run_info(vp.parent))
            if n == SAMPLE_VIDEO[cap][0] or (cap == "floor_only" and n == "fo_a"):
                images[f"sample_{cap}_video_{n}"] = plan_images(vp.parent, f"sample_{cap}_video_{n}", R)
                try:
                    TR.plot_overlay(vp, lid / "plan.json", res, S / f"tier_ref_{n}.png")
                    ov = save_png(S / f"tier_ref_{n}.png", R / "overlays" / f"sample_{cap}_video_{n}_vs_lidar.png")
                    images[f"sample_{cap}_video_{n}"]["overlay_vs_lidar"] = ov
                except Exception as e:  # noqa: BLE001
                    images[f"sample_{cap}_video_{n}"]["overlay_vs_lidar"] = f"failed: {e}"
        entry["video_vs_lidar"] = vids
        # determinism: the same LiDAR command on the same capture, earlier final run (10:04) vs now
        old = O / "benchmark/final/lidar" / cap / "drift_on_1004" / "plan.json"
        if old.exists() and lp:
            po = json.loads(old.read_text())
            la = sorted(round(w["length"]["value"], 4) for w in lp["walls"])
            lb = sorted(round(w["length"]["value"], 4) for w in po["walls"])
            entry["lidar_vs_1004_run"] = dict(same_wall_count=len(la) == len(lb),
                                              max_abs_wall_diff_m=_r(max(abs(x - y) for x, y in zip(la, lb)))
                                              if len(la) == len(lb) else None,
                                              footprint_now=mval(lp["footprint_area"]), footprint_1004=mval(po["footprint_area"]))
        sb[cap] = entry
    # LiDAR repeat pair and drift on/off with the reused scorer (registration from scene geometry)
    if not a.skip_lidar_pairs:
        lay = S / "lidar_layout"
        for cap in SAMPLES:
            for var, src in (("drift_on", F / "lidar" / cap), ("drift_off", F / "lidar" / f"{cap}_drift_off")):
                dst = lay / cap / var
                if (src / "plan.json").exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.symlink_to(src)
        import bench_lidar_eval as BL
        BL.B = lay
        BL._scenes.clear()
        try:
            if all((lay / c / "drift_on").exists() for c in ("floor_only", "with_ceiling")):
                pb = BL.pair_block("with_ceiling", "floor_only", "with_ceiling__floor_only")
                sb["lidar_repeat_pair"] = BL.jsonable({k: v for k, v in pb.items() if k not in ("walls",)})
        except Exception as e:  # noqa: BLE001
            sb["lidar_repeat_pair"] = f"failed: {type(e).__name__}: {e}"
        try:
            if not all((lay / c / "drift_off").exists() for c in ("floor_only", "with_ceiling")):
                sb["drift_on_off"] = ("not re-run in this folder: see docs/modules/drift.md and "
                                      "outputs/benchmark/final/REPORT.md (drift_off runs of 10:04, same LiDAR code)")
            else:
                sb["drift_on_off"] = BL.jsonable(BL.drift_block())
                for c in ("floor_only", "with_ceiling"):
                    for f in lay.glob(f"*drift*{c}*.png"):
                        images[f"drift_{c}"] = dict(overlay=save_png(f, R / "overlays" / f"drift_on_off_{c}.png"))
        except Exception as e:  # noqa: BLE001
            sb["drift_on_off"] = f"failed: {type(e).__name__}: {e}"
        for f in lay.glob("repeat_*.png"):
            images[f"lidar_{f.stem}"] = dict(overlay=save_png(f, R / "overlays" / f"lidar_{f.stem}.png"))
    summ["tiers"]["samples"] = dict(gt="none: video scored against the LiDAR plan of the same capture "
                                       "(REFERENCE-BASED, not ground truth)", captures=sb)

    # ---------------- CubiCasa side by side (own house)
    cc = ROOT.parent / "cubicasa" / "NearMaheshwariBhawan_dim_0.jpg"
    if not cc.exists():
        cc = ROOT / "data" / "cubicasa" / "NearMaheshwariBhawan_dim_0.jpg"
    ours = F / "video" / "own_2328" / "plan_presentation.png"
    images["cubicasa_side_by_side"] = side_by_side(ours, cc, R / "overlays" / "own_house_ours_vs_cubicasa.png",
                                                   ("ours: video take1 (replay own_2328), HEAD code",
                                                    "CubiCasa (Play Store app) export, same flat"))
    summ["images"] = images
    (R / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    write_tables(summ, R / "tables.md")
    print(f"wrote {R / 'summary.json'} and {R / 'tables.md'}")


if __name__ == "__main__":
    main()
