#!/usr/bin/env python
"""Final benchmark report (case study Deliverable 5): gates at all three tiers, repeatability, head-to-head, timing.

    scripts/bench_final_runs.sh lidar | lidar_off | video | photo <label>   # the one command on the samples
    python scripts/bench_final_arkit.py                                      # ARKitScenes vs Faro laser, current code
    python scripts/bench_final.py [--skip-lidar]                             # score everything + write REPORT.md

Idempotent: it only READS run folders and re-derives every number, so it can be re-run at any time (e.g. after the
photo v3 lands, or after `scripts/process_own_capture.py` has written outputs/own/eval/own_eval.json). Missing
inputs become "not run yet" rows, never invented numbers.

Inputs (all under outputs/benchmark/final/ unless noted):
  lidar/<capture>/{drift_on,drift_off,drift_on_rerun}/  run_capture --tier lidar (time.log, stdout.log, plan.json, scene/)
  video/<capture>/default/                              run_capture --tier video
  photo/<capture>/<label>/                              run_capture --tier photo (label: pre_v3, v3, ...)
  arkitscenes/summary.json                              scripts/bench_final_arkit.py
  outputs/own/eval/own_eval.json, outputs/own/*/run_report.json   scripts/process_own_capture.py (optional)
Outputs: results.json (every number), REPORT.md, tier_scores/<tier>__<capture>__<label>.json, lidar/results.json.

Scoring code reused unchanged: scripts/bench_lidar_eval.py (LiDAR repeat pair, drift on/off, ceilings, damage),
scripts/bench_tier_ref.py (video/photo vs the LiDAR plan of the same capture: REFERENCE-BASED, not ground truth).
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

F = ROOT / "outputs" / "benchmark" / "final"
OWN = Path(os.environ.get("BENCH_OWN", ROOT / "outputs" / "own"))   # override only for the smoke test
CAPS = ("single_room", "floor_only", "with_ceiling")
PHOTO_TRUTH = {"single_room": "single_room__c00a170fe1__rotate",
               "floor_only": "single_scan_floor_only__1a8384c3f6__rotate",
               "with_ceiling": "single_scan_with_ceiling__c7d28f72c6__rotate"}


def jl(p: Path):
    return json.loads(Path(p).read_text())


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------------------------------------ run metadata
def code_changed_since(started_iso: str, tier: str) -> list[str]:
    """Pipeline files modified after a run started (other agents edit the code while the benchmark runs). For the LiDAR
    tier floorplan/photo and floorplan/video are not on the path, so they are ignored."""
    import datetime as dt
    t0 = dt.datetime.fromisoformat(started_iso).timestamp()
    files = list((ROOT / "floorplan").rglob("*.py")) + [ROOT / "scripts" / "run_capture.py"]
    files = [f for f in files if not rel(f).startswith("floorplan/benchmark/")]   # scoring code, not the pipeline
    skip = ("floorplan/photo/", "floorplan/video/") if tier == "lidar" else \
        ("floorplan/photo/",) if tier == "video" else ("floorplan/video/",) if tier == "photo" else ()
    return sorted(rel(f) for f in files if f.stat().st_mtime > t0 and not rel(f).startswith(skip))


def run_meta(d: Path, tier: str = "") -> dict:
    """Timing and status of one run_capture folder (written by scripts/bench_final_runs.sh)."""
    m = dict(dir=rel(d), exists=d.exists())
    if not d.exists():
        return m
    rc = (d / "exit_code.txt")
    m["exit_code"] = int(rc.read_text().strip()) if rc.exists() and rc.read_text().strip() else None
    m["finished"] = (d / "finished.txt").exists()
    m["has_plan"] = (d / "plan.json").exists()
    for k in ("code_fingerprint", "started", "finished"):
        f = d / f"{k}.txt"
        if f.exists():
            m[k if k != "finished" else "finished_at"] = f.read_text().strip()
    if m.get("started"):
        m["code_changed_since_start"] = code_changed_since(m["started"], tier)
        fp = m.get("code_fingerprint", "").splitlines()
        m["code_fingerprint"] = fp[0] if fp else None
        if len(fp) > 1:
            m["code_fingerprint_lidar_path"] = fp[1]
    if (d / "load_before.txt").exists():
        m["load_before"] = (d / "load_before.txt").read_text().split("load average:")[-1].strip()
    t = {}
    if (d / "time.log").exists():
        for line in (d / "time.log").read_text().splitlines():
            line = line.strip()
            if line.startswith("Elapsed (wall clock)"):
                v = line.split("):")[-1].strip().split(":")
                t["wall_s"] = round(sum(float(x) * 60 ** i for i, x in enumerate(reversed(v))), 1)
            elif line.startswith("Maximum resident"):
                t["max_rss_gb"] = round(int(line.split(":")[-1]) / 1e6, 2)
            elif line.startswith("User time"):
                t["user_s"] = float(line.split(":")[-1])
    m.update(t)
    if (d / "run_report.json").exists():
        rr = jl(d / "run_report.json")
        m["runtime_s"] = rr.get("runtime_s")
        fe = rr.get("front_end") or {}
        m["drift_s"] = fe.get("drift_runtime_s")
    # final-run additions: what the plan says about itself (D-032 floor fallback, D-033 dropped openings, D-034
    # reliability flag + sigma floor) and what the front end said (segments, whole_scene_consistent, floor height)
    if (d / "plan.json").exists():
        pm = jl(d / "plan.json").get("meta") or {}
        unc = pm.get("uncertainty") or {}
        fb = pm.get("floor_fallback")
        m["plan_flags"] = dict(
            reliability=pm.get("reliability"),
            floor_fallback=(fb.get("source") if isinstance(fb, dict) else fb),
            dropped_openings=[f"{o.get('id')} {o.get('kind')} {o.get('width_m')} m" for o in pm.get("dropped_openings") or []],
            tier_scale_sigma_rel=unc.get("tier_scale_sigma_rel") if isinstance(unc, dict) else None,
            floor_y=pm.get("floor_y"))
    si = d / "scene" / "scene_info.json"
    if si.exists() and tier in ("video", "photo"):
        s = jl(si)
        pg = s.get("pose_graph") or {}
        m["front_end_flags"] = dict(whole_scene_consistent=s.get("whole_scene_consistent"),
                                    n_segments=pg.get("n_segments") if isinstance(pg, dict) else None,
                                    floor_y=s.get("floor_y"), scale_sigma_rel=s.get("scale_sigma_rel"))
    if (d / "stdout.log").exists():
        st = {}
        lines = (d / "stdout.log").read_text().splitlines()
        for line in lines:
            try:
                ts = float(line[1:line.index("s]")])
            except ValueError:
                continue
            if "scene saved" in line:
                st["scene_saved_at_s"] = ts
            if line.split("] ", 1)[-1].startswith("plan:"):
                st["plan_done_at_s"] = ts
        m.update(st)
        if m.get("exit_code") not in (0, None) or (m["finished"] and not m["has_plan"]):
            # stderr (the traceback) is in time.log, before /usr/bin/time's own report
            tl = (d / "time.log").read_text().split("Command exited")[0].split("\tCommand being timed")[0] \
                if (d / "time.log").exists() else ""
            tail = [x for x in tl.splitlines() if x.strip()][-6:] or [x for x in lines if x.strip()][-6:]
            err = [x for x in tail if "Error" in x or "error" in x]
            m["error"] = (err[-1] if err else tail[-1] if tail else "no output").strip()[:400]
            m["stdout_tail"] = tail
    return m


# ------------------------------------------------------------------------------------------------ LiDAR
def plan_internal(p: dict) -> dict:
    """Stitch checks on one plan: rooms, adjacency components, pairwise room overlaps."""
    from bench_tier_ref import OVERLAP_EPS_M2, _components, _pairs, _poly, load_plan  # noqa: F401
    ids = [r["id"] for r in p["rooms"]]
    polys = {r["id"]: _poly(json.loads(r["polygon"]) if isinstance(r["polygon"], str) else r["polygon"])
             for r in p["rooms"]}
    ov = 0.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = polys[ids[i]].intersection(polys[ids[j]]).area
            ov += a if a > OVERLAP_EPS_M2 else 0.0
    return dict(rooms=len(ids), adjacency_pairs=len(_pairs(p)), components=_components(ids, _pairs(p)),
                dropped_openings=[f"{o.get('id')} {o.get('kind')} {o.get('width_m')} m"
                                  for o in (p.get("meta") or {}).get("dropped_openings") or []],
                overlap_m2=round(ov, 3), footprint_m2=p["footprint_area"]["value"],
                footprint_ci95=p["footprint_area"].get("ci95"))


def lidar_block(skip: bool) -> dict:
    out_f = F / "lidar" / "results.json"
    plans = sorted((F / "lidar").glob("*/*/plan.json"))
    if skip and out_f.exists() and all(out_f.stat().st_mtime > p.stat().st_mtime for p in plans):
        res = jl(out_f)
    else:
        import bench_lidar_eval as BL
        BL.B = F / "lidar"
        BL._scenes.clear()
        res = dict(timing={k: v for k, v in ((f"{c}/{v}", run_meta(F / "lidar" / c / v, "lidar")) for c in CAPS
                                              for v in ("drift_on", "drift_off", "drift_on_rerun"))
                           if v.get("exists")})
        have = {c: (F / "lidar" / c / "drift_on" / "plan.json").exists() for c in CAPS}
        if all(have.values()):
            res["ceilings"] = BL.ceilings()
        res["damage"] = BL.damage()
        if have["with_ceiling"] and have["floor_only"]:
            res["repeat_pair"] = BL.pair_block("with_ceiling", "floor_only", "with_ceiling__floor_only")
        for big in ("with_ceiling", "floor_only"):
            if have[big] and have["single_room"]:
                try:
                    res[f"single_room_vs_{big}"] = BL.pair_block(big, "single_room", f"{big}__single_room")
                except Exception as e:  # noqa: BLE001
                    res[f"single_room_vs_{big}"] = dict(error=f"{type(e).__name__}: {e}")
        if all((F / "lidar" / c / v / "plan.json").exists() for c in ("floor_only", "with_ceiling")
               for v in ("drift_on", "drift_off")):
            res["drift"] = BL.drift_block()
        if have["single_room"]:
            res["drift_module_single_room"] = BL.drift_module_summary("single_room")
        res["internal"] = {f"{c}/{v}": plan_internal(jl(F / "lidar" / c / v / "plan.json"))
                           for c in CAPS for v in ("drift_on", "drift_off")
                           if (F / "lidar" / c / v / "plan.json").exists()}
        # same capture, same command, twice (D-029 determinism claim)
        a, b = F / "lidar" / "floor_only" / "drift_on" / "plan.json", F / "lidar" / "floor_only" / "drift_on_rerun" / "plan.json"
        if a.exists() and b.exists():
            res["rerun_floor_only"] = rerun_compare(jl(a), jl(b))
        # same command, earlier code state (first pass kept as drift_on_0343): did the code edits made by other agents
        # since then change any LiDAR plan?
        res["code_change_check"] = {}
        # snapshots of earlier passes, kept without scene/: drift_on_0343 (first pass), drift_on_0432 / drift_off_0445
        # (the 05:05 report's runs, before D-032/D-033 moved to run_capture.py and D-034)
        for c, snap, cur in [(c, s_, "drift_on") for s_ in ("drift_on_0343", "drift_on_0432") for c in CAPS] + \
                            [(c, "drift_off_0445", "drift_off") for c in CAPS]:
            a, b = F / "lidar" / c / snap / "plan.json", F / "lidar" / c / cur / "plan.json"
            if a.exists() and b.exists():
                r = rerun_compare(jl(a), jl(b))
                r["started_a"] = (a.parent / "started.txt").read_text().strip() if (a.parent / "started.txt").exists() else None
                r["started_b"] = (b.parent / "started.txt").read_text().strip() if (b.parent / "started.txt").exists() else None
                res["code_change_check"][f"{c} {cur} ({snap})"] = r
        res = BL.jsonable(res) if hasattr(BL, "jsonable") else res
        out_f.write_text(json.dumps(res, indent=1, default=float))
    return res


def rerun_compare(a: dict, b: dict) -> dict:
    def sig(p):
        return dict(footprint=p["footprint_area"]["value"], rooms=sorted(r["floor_area"]["value"] for r in p["rooms"]),
                    walls=sorted(w["length"]["value"] for w in p["walls"]), openings=len(p["openings"]))
    sa, sb = sig(a), sig(b)
    strip = lambda p: {k: v for k, v in p.items() if k != "meta"}  # noqa: E731 - meta holds run timings
    same_walls = len(sa["walls"]) == len(sb["walls"]) and np.allclose(sa["walls"], sb["walls"], atol=1e-4)
    return dict(json_identical_except_meta=strip(a) == strip(b),
                footprint_a=sa["footprint"], footprint_b=sb["footprint"], rooms_a=sa["rooms"], rooms_b=sb["rooms"],
                walls_a=len(sa["walls"]), walls_b=len(sb["walls"]), identical_geometry=bool(
                    same_walls and np.allclose(sa["rooms"], sb["rooms"], atol=1e-4) and sa["openings"] == sb["openings"]))


# ------------------------------------------------------------------------------------------------ ARKitScenes
def arkit_block() -> dict:
    f = F / "arkitscenes" / "summary.json"
    if not f.exists():
        return dict(status="not run (scripts/bench_final_arkit.py)")
    d = jl(f)
    rows = {}
    for t in d["table"]:
        if t["drift"] and t["role"] in ("all", "holdout") and t["correction"] in ("raw", "corrected") and t.get("n"):
            rows[f"{t['role']}/{t['kind']}/{t['correction']}"] = t
    return dict(source=rel(f), rows=rows, rooms=d.get("rooms"), timing=d.get("timing"),
                written=time.strftime("%H:%M", time.localtime(f.stat().st_mtime)))


# ------------------------------------------------------------------------------------------------ video / photo
def tier_block(tier: str) -> dict:
    from bench_tier_ref import score
    out = {}
    base = F / tier
    labels = sorted({p.name for p in base.glob("*/*") if p.is_dir()}) if base.exists() else []
    for label in labels:
        for cap in CAPS:
            d = base / cap / label
            if not d.exists():
                continue
            m = run_meta(d, tier)
            rec = dict(run=m)
            ref = F / "lidar" / cap / "drift_on"
            if m.get("has_plan") and (ref / "plan.json").exists():
                truth = F / "photo_inputs" / PHOTO_TRUTH[cap] / "truth" / "photo_truth.json"
                if not truth.exists():
                    truth = ROOT / "outputs" / "photo_inputs" / PHOTO_TRUTH[cap] / "truth" / "photo_truth.json"
                rec["photo_input"] = rel(truth.parent.parent / "photos")
                try:
                    s = score(d / "plan.json", ref / "plan.json", tier, ref / "scene",
                              truth if tier == "photo" else None)
                    (F / "tier_scores").mkdir(parents=True, exist_ok=True)
                    (F / "tier_scores" / f"{tier}__{cap}__{label}.json").write_text(json.dumps(s, indent=1, default=float))
                    rec["score"] = s
                    if s.get("status") != "no_rooms":
                        from bench_tier_ref import plot_overlay
                        plot_overlay(d / "plan.json", ref / "plan.json", s,
                                     F / "tier_scores" / f"{tier}__{cap}__{label}.png")
                except Exception as e:  # noqa: BLE001 - a scoring failure is reported, not hidden
                    rec["score_error"] = f"{type(e).__name__}: {e}"
            elif not m.get("has_plan"):
                rec["status"] = "FAILED (no plan.json)" if m.get("finished") else "running or not run"
            out[f"{cap}/{label}"] = rec
    return out


# ------------------------------------------------------------------------------------------------ own capture
def own_block() -> dict:
    f = OWN / "eval" / "own_eval.json"
    out = dict(source=rel(f), exists=f.exists())
    runs = {}
    for rr in sorted(OWN.glob("*/run_report.json")):
        runs[rr.parent.name] = run_meta(rr.parent)
    out["runs"] = runs
    if not f.exists():
        return out
    d = jl(f)
    out["tiers"] = {k: v.get("gates") for k, v in d.items() if isinstance(v, dict) and "gates" in v}
    out["head_to_head"] = d.get("head_to_head")
    # repeatability on the own capture: photo repeat room (take1 vs take2) and video take1 vs take2, per tape item:
    # the same wall measured from both takes agrees within max(1 cm, 0.5%) (the case-study gate)
    out["repeat"] = {}
    for name, k1, k2 in (("photo", "photo_repeat_take1", "photo_repeat_take2"), ("video", "video", "video_take2")):
        t1, t2 = d.get(k1), d.get(k2)
        if not (t1 and t2):
            continue
        a = {(r["room"], r["item"]): r for r in t1["rows"] if r["kind"] == "wall" and r["value"] is not None}
        b = {(r["room"], r["item"]): r for r in t2["rows"] if r["kind"] == "wall" and r["value"] is not None}
        n_gt = len({(r["room"], r["item"]) for r in t1["rows"] if r["kind"] == "wall"})
        rows = []
        for k in sorted(set(a) & set(b)):
            dv = abs(a[k]["value"] - b[k]["value"])
            tol = max(0.01, 0.005 * a[k]["gt"])
            rows.append(dict(room=k[0], item=k[1], take1=a[k]["value"], take2=b[k]["value"], delta_m=dv,
                             tol_m=tol, passed=bool(dv <= tol)))
        out["repeat"][name] = dict(rows=rows, n=len(rows), n_tape_walls=n_gt, passed=sum(r["passed"] for r in rows),
                                   median_delta_m=float(np.median([r["delta_m"] for r in rows])) if rows else None)
    out["raw"] = {k: v for k, v in d.items() if k not in ("head_to_head",)}
    return out


# ------------------------------------------------------------------------------------------------ report
def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def stale(m: dict) -> str:
    c = m.get("code_changed_since_start")
    if c is None:
        return "–"
    return "none" if not c else f"{len(c)}: " + ", ".join(Path(x).name for x in c[:4]) + (" …" if len(c) > 4 else "")


def verdict(ok):
    return "not verifiable" if ok is None else ("**PASS**" if ok else "**FAIL**")


def latest_keys(block: dict) -> list[str]:
    """Newest run per capture (by start time): the gate table shows these; older code states go to a history table."""
    best: dict[str, tuple[str, str]] = {}
    for key, rec in block.items():
        cap = key.split("/")[0]
        if key.endswith("rerun"):         # a second run of the same code is a repeatability sample, not a newer state
            continue                      # (variants "rerun" and "<label>_rerun")
        t = rec["run"].get("started") or ""
        if cap not in best or t > best[cap][0]:
            best[cap] = (t, key)
    return [best[c][1] for c in CAPS if c in best]


def short_status(rec: dict) -> str:
    if "score" not in rec:
        m = rec["run"]
        return f"FAIL, no plan (exit {m.get('exit_code')}: `{(rec.get('score_error') or m.get('error') or rec.get('status') or '')[:120]}`)"
    s = rec["score"]
    if s.get("status") == "no_rooms":
        return "FAIL, 0 rooms"
    fp, dm, adj, st = s["footprint"], s["dims"], s["adjacency"], s["stitch"]
    return (f"{s['rooms']['tier']} rooms, footprint {num(fp['value'])} vs {num(fp['ref'])} m² ({pct(fp['rel_err'])}), dims within "
            f"{dm['within_tol']}/{dm['n']}, adjacency {adj['correct']}/{adj['reported']} correct, "
            f"{st['components']} component(s), overlaps {s['overlaps']['n']}, coverage {pct(s['calibration']['coverage95'], 0)}")


def tier_gate_rows(tier: str, block: dict) -> list[str]:
    rows = []
    tol = 0.03 if tier == "video" else 0.08
    for key in latest_keys(block):
        rec = block[key]
        cap, label = key.split("/")
        src = f"`tier_scores/{tier}__{cap}__{label}.json`"
        tag = f"{tier} {cap} [{label}]"
        if "score" not in rec:
            m = rec["run"]
            why = rec.get("score_error") or m.get("error") or rec.get("status") or "no plan"
            if not m.get("finished") and not rec.get("score_error"):
                rows.append(f"| {tag}: whole run | still running / not run yet (started {m.get('started', '–')[11:19]}) | – | `{m['dir']}` |")
                continue
            rows.append(f"| {tag}: whole run | **FAIL** (no plan) | exit {m.get('exit_code')}: `{why}` | `{m['dir']}/time.log` (stderr) |")
            continue
        s = rec["score"]
        if s.get("status") == "no_rooms":
            rows.append(f"| {tag}: whole run | **FAIL** | plan has 0 rooms | {src} |")
            continue
        dm, fp, adj, ov, st, rm = s["dims"], s["footprint"], s["adjacency"], s["overlaps"], s["stitch"], s["rooms"]
        n_in = rm.get("reference_with_input", rm["reference"])
        dims_ok = dm["n"] > 0 and dm["within_tol"] == dm["n"] and rm["matched"] >= n_in
        rows.append(f"| {tag}: room dimensions within ±{tol:.0%} (wall-length gate, bbox proxy) | {verdict(dims_ok)} | "
                    f"{dm['within_tol']}/{dm['n']} dims within; median \\|err\\| {pct(dm['median_abs_rel_err'])}, max "
                    f"{pct(dm['max_abs_rel_err'])}; rooms matched {rm['matched']}/{n_in} "
                    + (f"(reference rooms with a photo folder; {rm['reference']} in the reference)" if n_in != rm["reference"]
                       else "(reference)") + f" | {src} |")
        rows.append(f"| {tag}: footprint within ±8%" + (" (photo gate; shown for video)" if tier == "video" else "") + f" | {verdict(fp['within_tol'] if tier == 'photo' else abs(fp['rel_err']) <= 0.08)} | "
                    f"{num(fp['value'])} m² [{num(fp['lo'])}, {num(fp['hi'])}] vs ref {num(fp['ref'])} m² "
                    f"({pct(fp['rel_err'])}); interval covers ref: {fp['covered']} | {src} |")
        fi = s.get("footprint_input_rooms")
        if fi:
            rows.append(f"| {tag}: footprint vs the reference rooms that have a photo folder (fair subset) | "
                        f"{verdict(fi['within_tol'])} | {num(fi['value'])} m² vs {num(fi['ref'])} m² ({pct(fi['rel_err'])}); "
                        f"interval covers: {fi['covered']} | {src} |")
        adj_ok = (adj["recall_all"] == 1.0 and (adj["precision"] in (1.0, None))) if adj["reference_pairs"] else (adj["reported"] == 0)
        rows.append(f"| {tag}: correct adjacency | {verdict(adj_ok)} | {adj['correct']}/{adj['reported']} reported pairs "
                    f"correct; {round((adj['recall_all'] or 0) * adj['reference_pairs'])}/{adj['reference_pairs']} reference pairs "
                    f"found" + (f" ({round((adj.get('recall_input') or 0) * adj['reference_pairs_between_input_rooms'])}/"
                                f"{adj['reference_pairs_between_input_rooms']} between rooms with a folder)"
                                if adj.get("reference_pairs_between_input_rooms") not in (None, adj["reference_pairs"]) else "")
                    + f" | {src} |")
        rows.append(f"| {tag}: one stitched plan, no overlaps | {verdict(st['one_stitched_plan'] and ov['n'] == 0)} | "
                    f"{st['rooms']} rooms in {st['components']} connected component(s); overlaps {ov['n']} "
                    f"({ov['total_m2']} m²) | {src} |")
        cal = s["calibration"]
        rows.append(f"| {tag}: calibrated intervals (95%) | {verdict(None if cal['coverage95'] is None else cal['coverage95'] >= 0.9)} | "
                    f"coverage of ref values {pct(cal['coverage95'], 0)} (n {cal['n']}), median interval half-width "
                    f"{pct(cal['median_rel_halfwidth'])} of the value | {src} |")
        pf, ff = rec["run"].get("plan_flags") or {}, rec["run"].get("front_end_flags") or {}
        rows.append(f"| {tag}: plan self-report (D-032 floor fallback, D-033 dropped openings, D-034 reliability) | reported | "
                    f"reliability: {pf.get('reliability') or 'not flagged'}; tier scale sigma {pf.get('tier_scale_sigma_rel', '–')}; "
                    f"front end: segments {ff.get('n_segments', '–')}, whole_scene_consistent {ff.get('whole_scene_consistent', '–')}, "
                    f"floor_y {num(ff.get('floor_y'))} m; floor fallback: {pf.get('floor_fallback') or 'not used'}; dropped openings: "
                    f"{', '.join(pf.get('dropped_openings') or []) or 'none'} | `{rec['run']['dir']}/plan.json` → meta |")
    return rows


def write_report(R: dict) -> str:
    L, lid, ak = [], R["lidar"], R["arkitscenes"]
    L += ["# Benchmark report (Deliverable 5)", "",
          f"Generated by `scripts/bench_final.py` at {R['generated']} from the run folders under "
          "`outputs/benchmark/final/`. Every number below is re-derived from those runs each time the script runs. "
          "Write-up and method: `docs/modules/benchmark_final.md`.", "",
          "**Ground truth.** The three sample captures have no tape or laser ground truth. So:",
          "- LiDAR absolute accuracy comes from ARKitScenes (iPad LiDAR against a Faro laser, 5 rooms), re-run on the "
          "current code;",
          "- video and photo are scored against the LiDAR plan of the **same capture**. These rows are "
          "**REFERENCE-BASED, not ground truth**: the reference itself carries LiDAR error (about 1 cm on ARKitScenes);",
          "- the own capture (tape GT) fills the last section after `scripts/process_own_capture.py` has run.", ""]
    L += ["**Code version.** Other agents edit the pipeline while the benchmark runs. Each run folder keeps a code "
          "fingerprint, and the timing table's last column lists pipeline files modified after that run started "
          "(for LiDAR, `floorplan/photo` and `floorplan/video` are ignored: they are not on its path). A run with "
          "\"none\" reflects the current code.", ""]

    # ---------------- gates
    L += ["## 1. Gates at all three tiers", "",
          "| Gate | Verdict | Number | Source |", "|---|---|---|---|"]
    rp = lid.get("repeat_pair")
    if rp:
        L.append(f"| LiDAR: repeatability, every wall within max(1 cm, 0.5%) (with_ceiling vs floor_only) | "
                 f"{verdict(rp['walls_pass'] == rp['walls_judged'])} | {rp['walls_pass']}/{rp['walls_judged']} walls "
                 f"= {pct(rp['pass_rate'])} (Wilson 95% {pct(rp['wilson'][0])}–{pct(rp['wilson'][1])}); "
                 f"{rp['walls_unmatched']} unmatched; same-topology {rp['same_topology']['passed']}/{rp['same_topology']['n']}, "
                 f"median {num(100 * (rp['same_topology']['median_m'] or 0), 1)} cm; rooms matched "
                 f"{rp['rooms_matched']}/{rp['rooms_a']} | `lidar/results.json` → repeat_pair |")
        L.append(f"| LiDAR: opening widths ≤ 2 cm on ≥ 85% | not verifiable (no GT); repeat-pair proxy "
                 f"{verdict(False if rp['openings_within_2cm_over_all_slots'] else None)} | "
                 f"{rp['openings_within_2cm_over_all_slots']} opening slots agree within 2 cm across the repeat pair "
                 f"(missed/phantom count as misses); matched median \\|Δ\\| {num(rp['opening_median_abs_delta_cm'], 1)} cm | "
                 "`lidar/results.json` → repeat_pair.openings |")
    ce = ak.get("rows", {}).get("all/ceiling_height/corrected")
    if ce:
        L.append(f"| LiDAR: ceiling height ≤ 1.5 cm per room (ARKitScenes, laser GT) | {verdict(ce['within_1.5cm'] == 1.0)} | "
                 f"{pct(ce['within_1.5cm'], 0)} of {ce['n']} rooms within 1.5 cm; median \\|e\\| {ce['median_abs_cm']} cm, "
                 f"max {ce['max_abs_cm']} cm | `arkitscenes/summary.json` |")
    cl = lid.get("ceilings", {})
    if cl:
        obs = {c: sum(r["value_m"] is not None for r in v) for c, v in cl.items()}
        L.append(f"| LiDAR: ceiling spread across captures ≤ 1 cm | not verifiable | ceilings observed per capture: "
                 f"{', '.join(f'{c} {obs[c]}/{len(cl[c])}' for c in cl)}; only one capture of each room sees the ceiling | "
                 "`lidar/results.json` → ceilings |")
    for kind, lab in (("wall_length", "wall lengths"), ("room_dim", "room dimensions")):
        w = ak.get("rows", {}).get(f"all/{kind}/corrected")
        if w:
            L.append(f"| LiDAR: absolute {lab} (ARKitScenes, laser GT; no case-study threshold, 1.5 cm shown) | reported | "
                     f"median \\|e\\| {w['median_abs_cm']} cm, {pct(w['within_1.5cm'], 0)} within 1.5 cm, "
                     f"{pct(w['within_2.0cm'], 0)} within 2 cm, max {w['max_abs_cm']} cm (n {w['n']}) | `arkitscenes/summary.json` |")
    dr = lid.get("drift")
    if dr:
        fo = dr["floor_only"]
        on, off = fo["drift_on"]["plan"], fo["drift_off"]["plan"]
        wc = dr["with_ceiling"]
        L.append(f"| LiDAR: drift accountability (method + on/off ablation) | **PASS** | floor_only ON {on['rooms']} rooms / "
                 f"{on['footprint_m2']} m² vs OFF {off['rooms']} rooms / {off['footprint_m2']} m²; with_ceiling ON "
                 f"{wc['drift_on']['plan']['rooms']} / {wc['drift_on']['plan']['footprint_m2']} m² vs OFF "
                 f"{wc['drift_off']['plan']['rooms']} / {wc['drift_off']['plan']['footprint_m2']} m² | "
                 "`lidar/drift_overlay_*.png`, `lidar/results.json` → drift |")
    else:
        L.append("| LiDAR: drift accountability | not run yet | drift_off runs missing | – |")
    it = lid.get("internal", {})
    if it:
        s = "; ".join(f"{k.split('/')[0]} {v['rooms']} rooms, {v['adjacency_pairs']} adjacency pairs, {v['components']} comp., "
                      f"overlap {v['overlap_m2']} m²" + (f", dropped openings (D-033): {', '.join(v['dropped_openings'])}"
                                                         if v.get("dropped_openings") else "")
                      for k, v in it.items() if k.endswith("drift_on"))
        ok = all(v["components"] == 1 and v["overlap_m2"] == 0 for k, v in it.items() if k.endswith("drift_on"))
        L.append(f"| LiDAR: one stitched plan, no overlaps (internal; adjacency GT not available) | {verdict(ok)} | {s} | "
                 "`lidar/results.json` → internal |")
    dmg = lid.get("damage", {})
    if dmg:
        conf = {k: v["confirmed"] for k, v in dmg.items() if k.endswith("drift_on")}
        L.append(f"| LiDAR: damage false positives on clean captures (0 confirmed) | {verdict(sum(conf.values()) == 0)} | "
                 + ", ".join(f"{k.split('/')[0]} {v} confirmed / {dmg[k]['review']} review / {dmg[k]['scope_items']} scope"
                             for k, v in conf.items()) + " | `lidar/results.json` → damage |")
    cov = ak.get("rows", {}).get("all/lengths_all/corrected")
    if rp or cov:
        L.append(f"| LiDAR: calibration (95% intervals) | reported | ARKitScenes lengths coverage "
                 f"{pct(cov and cov['coverage95'], 0)} (n {cov and cov['n']}, median half-width {cov and cov['median_ci_halfwidth_cm']} cm); "
                 f"repeat pair same-topology walls {pct(rp and rp['same_topology_coverage']['coverage'], 0)}, all matched walls "
                 f"{pct(rp and rp['wall_interval_coverage'], 0)} | `arkitscenes/summary.json`, `lidar/results.json` |")
    for tier in ("video", "photo"):
        rows = tier_gate_rows(tier, R[tier])
        L += rows if rows else [f"| {tier}: all gates | not run yet | – | – |"]
    L += ["", "Video and photo rows are **REFERENCE-BASED** (the LiDAR plan of the same capture, `lidar/<capture>/drift_on`). "
          "Room dimensions stand in for the wall-length gate (exact for rectangular rooms). The photo \"fair subset\" row compares "
          "against only the reference rooms that have a photo folder. Overlays: `tier_scores/<tier>__<capture>__<label>.png`; "
          "drift on/off overlays: `lidar/drift_overlay_*.png`.", ""]
    hist = [(t, k, rec) for t in ("video", "photo") for k, rec in R[t].items() if k not in latest_keys(R[t])]
    if hist:
        L += ["**Earlier code states (not in the gate table, kept for the trend):**", "",
              "| Tier | Capture [label] | Started | Result |", "|---|---|---|---|"]
        L += [f"| {t} | {k.replace('/', ' [')}] | {rec['run'].get('started', '–')[11:19]} | {short_status(rec)} |" for t, k, rec in hist]
        L += [""]

    # ---------------- summary of verdicts per tier (inserted after the header)
    import re
    gate_lines = [x for x in L if x.startswith("| ") and ("**PASS**" in x or "**FAIL**" in x or "not verifiable" in x)]
    summ = ["## 0. Summary", "", "| Tier | PASS | FAIL | not verifiable / reported |", "|---|---|---|---|"]
    for t, pref in (("LiDAR", "| LiDAR"), ("video", "| video"), ("photo", "| photo")):
        g = [x for x in L if x.startswith(pref) and x.count("|") >= 5]
        cells = [x.split("|")[2] for x in g]
        summ.append(f"| {t} | {sum('**PASS**' in c for c in cells)} | {sum('**FAIL**' in c for c in cells)} | "
                    f"{sum(('not verifiable' in c or 'reported' in c or 'not run' in c) and '**' not in c for c in cells)} |")
    summ += ["", "Counts are rows of the gate table in section 1 (latest run per capture).", ""]
    i = L.index("## 1. Gates at all three tiers")
    L[i:i] = summ
    del re, gate_lines

    # ---------------- repeatability
    L += ["## 2. Repeatability", "",
          "Gate: two captures of the same room at the same tier agree within max(1 cm, 0.5%) per wall.", "",
          "| Tier | Pair | Walls judged | Pass | Pass rate (Wilson 95%) | Median \\|Δ\\| | Same-topology pass / n (median) | Footprints m² | Source |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, lab in (("repeat_pair", "with_ceiling vs floor_only (whole flat)"),
                     ("single_room_vs_with_ceiling", "with_ceiling vs single_room"),
                     ("single_room_vs_floor_only", "floor_only vs single_room")):
        p = lid.get(key)
        if p and "error" not in p:
            L.append(f"| LiDAR | {lab} | {p['walls_judged']} | {p['walls_pass']} | {pct(p['pass_rate'])} "
                     f"({pct(p['wilson'][0])}–{pct(p['wilson'][1])}) | {num(p['median_abs_delta_cm'], 1)} cm | "
                     f"{p['same_topology']['passed']}/{p['same_topology']['n']} ({num(100 * (p['same_topology']['median_m'] or 0), 1)} cm) | "
                     f"{num(p['footprint_a'])} / {num(p['footprint_b'])} | `lidar/walls_*.csv` |")
        elif p:
            L.append(f"| LiDAR | {lab} | error: {p['error']} | | | | | | |")
    rr = lid.get("rerun_floor_only")
    if rr:
        L.append(f"| LiDAR | floor_only, same command run twice | {rr['walls_a']} / {rr['walls_b']} walls | – | identical geometry: "
                 f"**{rr['identical_geometry']}** | – | – | {num(rr['footprint_a'])} / {num(rr['footprint_b'])} | `lidar/results.json` → rerun_floor_only |")
    for c, r in (lid.get("code_change_check") or {}).items():
        L.append(f"| LiDAR | {c}: run at {(r.get('started_a') or '')[11:16]} vs run at {(r.get('started_b') or '')[11:16]} "
                 f"(other agents edited the code in between) | {r['walls_a']} / {r['walls_b']} walls | – | identical geometry: "
                 f"**{r['identical_geometry']}** | – | – | {num(r['footprint_a'])} / {num(r['footprint_b'])} | "
                 "`lidar/results.json` → code_change_check |")
    for key, rec in R["video"].items():
        if not key.endswith("rerun"):
            continue
        cap, var = key.split("/")
        base = "default" if var == "rerun" else var[: -len("_rerun")]   # rerun pairs with default, X_rerun with X
        a, b = F / "video" / cap / base / "plan.json", F / "video" / cap / var / "plan.json"
        if not (a.exists() and b.exists()):
            st = ("rerun still running" if not rec["run"].get("finished") else "rerun has no plan") if a.exists() \
                else "first run has no plan"
            L.append(f"| video | {cap}: same command run twice | – | – | {st} | – | – | – | `video/{cap}/*/stdout.log` |")
            continue
        pa, pb = jl(a), jl(b)
        ra = (R["video"].get(f"{cap}/{base}", {}).get("score") or {})
        rb = rec.get("score") or {}
        fa = (R["video"].get(f"{cap}/{base}", {}).get("run") or {}).get("front_end_flags") or {}
        fb_ = (rec.get("run") or {}).get("front_end_flags") or {}
        same = rerun_compare(pa, pb)["identical_geometry"] if pa["rooms"] and pb["rooms"] else (len(pa["rooms"]) == len(pb["rooms"]) == 0)
        L.append(f"| video | {cap} [{base} vs {var}]: same command run twice | rooms {len(pa['rooms'])} / {len(pb['rooms'])}, walls "
                 f"{len(pa['walls'])} / {len(pb['walls'])} | – | identical geometry: **{same}** | – | segments "
                 f"{fa.get('n_segments', '–')} / {fb_.get('n_segments', '–')}, floor_y {num(fa.get('floor_y'))} / {num(fb_.get('floor_y'))} m, "
                 f"whole_scene_consistent {fa.get('whole_scene_consistent', '–')} / {fb_.get('whole_scene_consistent', '–')} | "
                 f"{num(pa['footprint_area']['value']) if pa['rooms'] else '0 rooms'} / {num(pb['footprint_area']['value']) if pb['rooms'] else '0 rooms'} | "
                 f"dims within 3% of LiDAR: {(ra.get('dims') or {}).get('within_tol', '–')}/"
                 f"{(ra.get('dims') or {}).get('n', '–')} vs {(rb.get('dims') or {}).get('within_tol', '–')}/{(rb.get('dims') or {}).get('n', '–')}; "
                 f"footprint interval covers LiDAR: {(ra.get('footprint') or {}).get('covered', '–')} / {(rb.get('footprint') or {}).get('covered', '–')} |")
    orp = R["own"].get("repeat") or {}
    for name, rp_ in orp.items():
        L.append(f"| {name} (own capture) | take1 vs take2, walls matched to the tape in both | {rp_['n']} of {rp_['n_tape_walls']} tape walls | "
                 f"{rp_['passed']} | {pct(rp_['passed'] / rp_['n'] if rp_['n'] else None)} | "
                 f"{num(100 * rp_['median_delta_m'] if rp_['median_delta_m'] is not None else None, 1)} cm | – | – | "
                 f"`{rel(OWN / 'eval' / 'own_eval.json')}` |")
    if not orp:
        L.append("| photo / video (own capture) | repeat room take1 vs take2 | pending the 08:00 own capture | | | | | | – |")
    L += [""]

    # ---------------- head to head
    L += ["## 3. Head-to-head against a consumer app (Part 3)", ""]
    h = R["own"].get("head_to_head")
    if h and h.get("rows"):
        L += [f"Beat or tie on {pct(h['beat_or_tie_frac'], 0)} of {len(h['rows'])} shared dimensions (gate ≥ 70%): "
              f"{verdict(h['passed'])}. Tie = within 5 mm.", "",
              "| Room | Item | Tape GT (m) | Ours (m) | App (m) | \\|err\\| ours | \\|err\\| app | Outcome |",
              "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['room']} | {r['item']} | {r['gt']:.3f} | {r['ours']:.3f} | {r['app']:.3f} | {100 * r['err_ours']:.1f} cm | "
              f"{100 * r['err_app']:.1f} cm | {r['outcome']} |" for r in h["rows"]]
    else:
        L += ["**Placeholder until the own capture.** Filled automatically from `outputs/own/eval/own_eval.json` → "
              "`head_to_head` (written by `scripts/process_own_capture.py <root>` when `app_export/app_dimensions.csv` "
              "exists). Note: the case study asks for our **LiDAR** tier against the app; the candidate's phone (OnePlus "
              "Nord) has no LiDAR, so the comparison that can be run is our video (or photo) tier, and the report must say so.",
              "", "| Room | Item | Tape GT (m) | Ours (m) | App (m) | \\|err\\| ours | \\|err\\| app | Outcome |",
              "|---|---|---|---|---|---|---|---|", "| – | – | – | – | – | – | – | pending |"]
    L += [""]

    # ---------------- timing
    L += ["## 4. Timing (one command, end to end)", "",
          "| Tier | Capture | Variant | Wall time | Peak RSS | Drift | Scene saved at | Plan done at | Exit | Load before (1/5/15 min) | Pipeline files changed since the run |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in lid.get("timing", {}).items():
        c, v = k.split("/")
        L.append(f"| LiDAR | {c} | {v} | {num(m.get('wall_s'), 0)} s | {num(m.get('max_rss_gb'), 1)} GB | "
                 f"{num(m.get('drift_s'), 1)} s | {num(m.get('scene_saved_at_s'), 0)} s | {num(m.get('plan_done_at_s'), 0)} s | "
                 f"{m.get('exit_code')} | {m.get('load_before', '–')} | {stale(m)} |")
    for tier in ("video", "photo"):
        for key, rec in R[tier].items():
            m = rec["run"]
            c, v = key.split("/")
            L.append(f"| {tier} | {c} | {v} | {num(m.get('wall_s'), 0)} s | {num(m.get('max_rss_gb'), 1)} GB | – | "
                     f"{num(m.get('scene_saved_at_s'), 0)} s | {num(m.get('plan_done_at_s'), 0)} s | {m.get('exit_code')} | "
                     f"{m.get('load_before', '–')} | {stale(m)} |")
    for room, t in (ak.get("timing") or {}).items():
        if t:
            L.append(f"| LiDAR (ARKitScenes build) | {room} | drift on | {t.get('pipeline_total_s')} s (pipeline) | – | "
                     f"{t.get('drift_s')} s | – | – | 0 | – | – |")
    for k, m in R["own"].get("runs", {}).items():
        L.append(f"| own | {k} | – | {num(m.get('wall_s') or m.get('runtime_s'), 0)} s | {num(m.get('max_rss_gb'), 1)} GB | – | "
                 f"{num(m.get('scene_saved_at_s'), 0)} s | {num(m.get('plan_done_at_s'), 0)} s | {m.get('exit_code')} | – | {stale(m)} |")
    L += ["", "Wall times were measured while other jobs shared the CPU and GPU (load column); treat them as upper bounds.", ""]

    # ---------------- calibration
    L += ["## 5. Calibration summary (share of reference values inside the reported 95% interval)", "",
          "Nominal is about 95%. Much lower is over-confident (\"confident garbage\"); about 100% with very wide intervals "
          "is honest but uninformative, so the median interval half-width is shown next to it.", "",
          "| Tier | Data | Quantity | n | Coverage | Median half-width | Reference |", "|---|---|---|---|---|---|---|"]
    for key, lab in (("all/lengths_all/raw", "lengths, no bias correction"), ("all/lengths_all/corrected", "lengths, default"),
                     ("all/ceiling_height/corrected", "ceiling height, default")):
        t = ak.get("rows", {}).get(key)
        if t:
            L.append(f"| LiDAR | ARKitScenes 5 rooms | {lab} | {t['n']} | {pct(t['coverage95'], 0)} | {t['median_ci_halfwidth_cm']} cm | Faro laser (GT) |")
    if rp:
        L.append(f"| LiDAR | repeat pair | same-topology walls (\\|z\\| ≤ 1.96) | {rp['same_topology_coverage']['n']} | "
                 f"{pct(rp['same_topology_coverage']['coverage'], 0)} | – | other capture |")
        L.append(f"| LiDAR | repeat pair | opening widths | {rp['openings_width_both_measured']} | {pct(rp['opening_interval_coverage'], 0)} | – | other capture |")
    for tier in ("video", "photo"):
        for key, rec in R[tier].items():
            s = rec.get("score")
            if not s or "calibration" not in s:
                continue
            for q, blk in (("room dims", s["dims"]), ("room areas", s["areas"])):
                if blk.get("n"):
                    L.append(f"| {tier} | {key} | {q} | {blk['n']} | {pct(blk['coverage95'], 0)} | {pct(blk['median_rel_halfwidth'])} | LiDAR plan (reference) |")
            fp = s["footprint"]
            L.append(f"| {tier} | {key} | footprint | 1 | {'100%' if fp['covered'] else '0%'} | {pct(fp['rel_halfwidth'])} | LiDAR plan (reference) |")
    for tier, g in (R["own"].get("tiers") or {}).items():
        c = (g or {}).get("calibration")
        if c:
            L.append(f"| {tier} | own capture | walls, openings, ceilings | {c['n']} | {pct(c['coverage_95'], 0)} | – | tape (GT) |")
    L += [""]

    # ---------------- own capture
    L += ["## 6. Own capture (tape ground truth)", ""]
    o = R["own"]
    if o.get("tiers"):
        L += ["| Tier run | Wall gate | Median wall error | Openings ≤ 2 cm | Ceilings ≤ 1.5 cm | Interval coverage |",
              "|---|---|---|---|---|---|"]
        for tier, g in o["tiers"].items():
            g = g or {}
            wg, og, cg, cal = g.get("wall_gate") or {}, g.get("opening_gate") or {}, g.get("ceiling_gate") or {}, g.get("calibration") or {}
            L.append(f"| {tier} | {verdict(wg.get('passed')) if wg else '–'} ({pct(wg.get('pass_frac'), 0)} within ±{pct(wg.get('tol'), 0)}) | "
                     f"{pct(g.get('wall_median_rel_err'))} ({num(100 * g['wall_median_abs_err_m'] if g.get('wall_median_abs_err_m') is not None else None, 1)} cm) | "
                     f"{pct(og.get('pass_frac'), 0)} (n {og.get('n', '–')}, missed {og.get('missed', '–')}, phantom {og.get('phantom', '–')}) | "
                     f"{pct(cg.get('pass_frac'), 0)} | {pct(cal.get('coverage_95'), 0)} (n {cal.get('n', '–')}) |")
        L += ["", f"Per-item tables: `outputs/own/eval/own_eval.md`.", ""]
    else:
        L += ["**Pending.** After the 08:00 capture, run", "",
              "```", "python scripts/process_own_capture.py <capture root>      # runs photo + video, scores against tape",
              "python scripts/bench_final.py --skip-lidar                  # refreshes this report", "```", "",
              "This section then shows, per tier run: wall-length gate (photo ±8%, video ±3%), opening widths, "
              "ceiling heights, and interval coverage against the tape; section 2 gains the photo repeat room and "
              "section 3 the head-to-head table.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-lidar", action="store_true", help="reuse lidar/results.json when newer than every LiDAR plan")
    a = ap.parse_args()
    F.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = dict(generated=time.strftime("%Y-%m-%d %H:%M %Z"))
    R["lidar"] = lidar_block(a.skip_lidar)
    print(f"[{time.time() - t0:5.0f}s] lidar scored", flush=True)
    R["arkitscenes"] = arkit_block()
    R["video"] = tier_block("video")
    R["photo"] = tier_block("photo")
    print(f"[{time.time() - t0:5.0f}s] video/photo scored", flush=True)
    R["own"] = own_block()
    (F / "results.json").write_text(json.dumps(R, indent=1, default=float))
    (F / "REPORT.md").write_text(write_report(R))
    print(f"[{time.time() - t0:5.0f}s] wrote {F / 'REPORT.md'}")


if __name__ == "__main__":
    main()
