#!/usr/bin/env python
"""One command per capture (case study, Part 2 + Deliverable 3).

    python scripts/run_capture.py <input> --tier lidar|video|photo [--out DIR]

  lidar : <input> = a Stray Scanner folder (rgb.mp4, depth/, confidence/, odometry.csv, camera_matrix.csv)
  video : <input> = a video file (.mp4/.mov) OR a Stray Scanner folder (only its rgb.mp4 is used)
  photo : <input> = a folder with one sub-folder of 2-8 photos per room

Outputs in --out (default outputs/runs/<name>/<tier>/): plan.json (schema/plan.schema.json), plan.svg / plan.png
(technical drawing: every wall dimensioned with its interval), plan_presentation.svg / .png (clean drawing to show a
homeowner), plan.dxf, scene/ (the aligned 3D scene, for audit), run_report.json (timings, every stage's report).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplan.config import Config  # noqa: E402
from floorplan.pipeline.scene import save_scene  # noqa: E402
from floorplan.uncertainty.tier_budget import widen_plan  # noqa: E402


VIDEO_INCONSISTENT_SIGMA_REL = 0.20   # D-034: measured -37% footprint errors on unverified multi-segment videos
MIN_OPENING_WIDTH_M = {"door": 0.45, "passage": 0.45, "window": 0.25}   # D-033: physically implausible below this


def drop_implausible_openings(plan) -> list[dict]:
    """Remove openings narrower than any real one (an extractor artefact, e.g. a 1 cm 'passage' on single_room);
    keep a record in plan.meta['dropped_openings'] and clean wall / adjacency references."""
    drop = {o.id: o for o in plan.openings
            if o.width is not None and o.width.value is not None
            and o.width.value < MIN_OPENING_WIDTH_M.get(o.kind, 0.25)}
    if not drop:
        return []
    plan.openings = [o for o in plan.openings if o.id not in drop]
    for w in plan.walls:
        w.opening_ids = [i for i in w.opening_ids if i not in drop]
    plan.adjacency = [x for x in plan.adjacency if x.via not in drop]
    rec = [dict(id=o.id, kind=o.kind, width_m=round(o.width.value, 3),
                reason=f"narrower than {MIN_OPENING_WIDTH_M.get(o.kind, 0.25):.2f} m: not a real {o.kind}")
           for o in drop.values()]
    plan.meta.setdefault("dropped_openings", []).extend(rec)
    return rec


PHOTO_OVERRIDES: dict = {}
VIDEO_SCALE: str | None = None        # --video-scale: VideoParams.scale_method (D-076); None = its default
VIDEO_STEEP: bool = False             # --video-steep: VideoParams.steep_frames (D-084, off by default)
SEMANTIC_HINT = {"no segmentation env": "build it with setup/seg_env.sh",
                 "disabled": "turned off with --photo-param semantic_walls=false"}


def semantic_warning(info: dict) -> str | None:
    """D-060: without the wall segmenter, wardrobes and cabinets can become walls (k65 footprint -14.9% instead of
    -8.3%). The photo tier still runs, so the plan must say it was made from geometry only."""
    status = (info.get("semantic") or {}).get("status", "not run")
    if status == "ok":
        return None
    hint = SEMANTIC_HINT.get(status, "see 'semantic' in scene/scene_info.json")
    return (f"no semantic wall masks ({status}; {hint}): photo rooms from geometry only, so wardrobes and "
            "cabinets can be taken for walls (D-060)")


def front_end(inp: Path, tier: str, cfg: Config, out: Path, drift: bool, log):
    report = {}
    if tier == "lidar":
        from floorplan.io.stray import load_stray
        from floorplan.pipeline.scene import build_scene
        from floorplan.recon.keyframes import select_keyframes
        cap = load_stray(inp)
        T_wc = cap.T_wc
        if drift:
            from floorplan.recon.drift import correct_drift
            t0 = time.time()
            T_wc, report["drift"] = correct_drift(cap, select_keyframes(cap, cfg), cap.T_wc, cfg, log=log)
            report["drift_runtime_s"] = round(time.time() - t0, 1)
        else:
            report["drift"] = "DISABLED (ablation only: poses used as-is fails the drift gate)"
        scene, info, _ = build_scene(inp, cfg, T_wc=T_wc, log=log)
        return scene, info, report, 0.0, "LiDAR depth is metric; no scale term"
    if tier == "video":
        from floorplan.video import VideoParams, build_scene_from_video
        kw = dict(scale_method=VIDEO_SCALE) if VIDEO_SCALE else {}
        params = VideoParams(**kw, steep_frames=True) if VIDEO_STEEP else (VideoParams(**kw) if kw else None)
        scene, info = build_scene_from_video(inp, params, work_dir=out / "work", log=log)
        rel = float(info.get("scale_sigma_rel", info.get("scale", {}).get("sigma_rel", 0.03)) or 0.03)
        reason = "video scale from learned metric depth + priors; no reference object needed (D-067)"
        if info.get("whole_scene_consistent") is False:
            # D-034: segments whose joins were not verified by geometry can be placed/scaled wrongly relative to each
            # other. Measured on the sample's long walkthroughs: footprint -37.9% / -36.8% vs LiDAR while intervals
            # covered only 31-38% (confident garbage). Floor the relative sigma at 20% (95% ~ +-40%) and flag it.
            rel = max(rel, VIDEO_INCONSISTENT_SIGMA_REL)
            reason = (f"video: segments not joined by verified geometry (whole_scene_consistent=False); relative "
                      f"sigma floored at {100 * VIDEO_INCONSISTENT_SIGMA_REL:.0f}% (D-034)")
            report["reliability"] = "low: " + reason
        return scene, info, report, rel, reason
    if tier == "photo":
        from floorplan.photo import build_scene_from_photos
        scene, info = build_scene_from_photos(inp, params=PHOTO_OVERRIDES or None)
        rel = float(info.get("scale_sigma_rel", info.get("scale", {}).get("sigma_rel", 0.05)) or 0.05)
        reason = "photo scale from learned metric depth + priors; no reference object needed (D-067)"
        warning = semantic_warning(info)
        if warning:
            report["warnings"] = [warning]
        return scene, info, report, rel, reason
    raise ValueError(tier)


def clip_nonnegative(plan) -> int:
    """Sizes (areas, lengths, heights, widths) are >= 0, so a lower bound below 0 says nothing: clip it at 0. This
    never removes the true value from an interval. Photo rooms with fallback placement (widen x4) and a 0.5 m
    inferred-wall sigma otherwise print negative lower bounds (plan_beta.md Part v4). Returns how many were clipped."""
    ms = [plan.footprint_area]
    for r in plan.rooms:
        ms += [r.floor_area, r.perimeter, r.ceiling_height, *(r.bbox_dims or ())]
    for w in plan.walls:
        ms += [w.length, w.thickness]
    for o in plan.openings:
        ms += [o.width, o.height, o.sill_height]
    n = 0
    for m in ms:
        if m is not None and m.lo is not None and m.lo < 0:
            m.lo, n = 0.0, n + 1
    return n


class StageTimeout(Exception):
    """A pipeline stage exceeded its time limit."""


def _with_time_limit(fn, seconds: int):
    """Run fn() with a SIGALRM time limit (main thread, POSIX). Long numpy/GEOS calls are interrupted when they return
    to Python, which is enough for the plan step's Python-level loops."""
    import signal

    def _alarm(signum, frame):
        raise StageTimeout()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(1, int(seconds)))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--tier", required=True, choices=["lidar", "video", "photo"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--extractor", default="beta", choices=["alpha", "beta"])
    ap.add_argument("--no-drift", action="store_true", help="ablation only")
    ap.add_argument("--no-damage", action="store_true")
    ap.add_argument("--no-bias-correction", action="store_true", help="LiDAR tier: report raw (uncorrected) values")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--plan-timeout", type=int, default=300, help="seconds before the plan step falls back (D-058)")
    ap.add_argument("--damage-timeout", type=int, default=300, help="seconds before damage detection is skipped")
    ap.add_argument("--photo-param", action="append", default=[], metavar="KEY=VALUE",
                    help="override a photo-tier parameter (floorplan/photo/params.py), e.g. f35_rule=diagonal")
    ap.add_argument("--video-scale", choices=["depth_agreement", "pnp", "auto"],
                    help="video tier: local scale method (D-076); default from floorplan/video/params.py")
    ap.add_argument("--video-steep", action="store_true",
                    help="video tier: leave ceiling looks and straight-down looks out of the scale and the fusion; a "
                         "look at the end ends the walk (D-084, off by default)")
    ap.add_argument("--no-semantic-openings", action="store_true",
                    help="video/photo: skip the doors and windows from the segmenter (floorplan/openings/semantic.py)")
    ap.add_argument("--no-room-names", action="store_true",
                    help="skip room names (Bedroom, Kitchen, ...) from the classes seen in each room (D-078)")
    ap.add_argument("--door-priors", default="d", metavar="RULES",
                    help="doors from priors where the door itself is not seen (floorplan/openings/priors.py, D-085): "
                         "a = gaps 0.55-1.15 m with wall on both sides, b = doorway photos, c = the video path "
                         "crossing between rooms, d = segmenter doors cut by the frame get the prior width; default "
                         "'d' (the only rule that measured better), 'none' turns them off")
    ap.add_argument("--video-rooms", type=Path, metavar="JSON",
                    help="video tier, experimental: measure each room from its own turning frames with the photo tier "
                         "(floorplan/video/rooms_as_photos.py; JSON: the rooms' time windows and a video-tier run "
                         "for the camera rotations)")
    a = ap.parse_args()
    if a.video_rooms and a.tier != "video":
        ap.error("--video-rooms needs --tier video")
    global VIDEO_SCALE, VIDEO_STEEP
    VIDEO_SCALE = a.video_scale
    VIDEO_STEEP = a.video_steep
    for kv in a.photo_param:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            v = {"true": True, "false": False}.get(v.lower(), v)  # True/False (Python spelling) are booleans too
        PHOTO_OVERRIDES[k] = v
    name = a.input.stem if a.input.is_file() else f"{a.input.parent.name}__{a.input.name}"
    out = a.out or ROOT / "outputs" / "runs" / name / a.tier
    out.mkdir(parents=True, exist_ok=True)
    if a.video_rooms:
        # video as per-room photos: no camera path is chained through the house; each room is measured from its own
        # turning frames (oracle test on my home, 5 Oct: see the module docstring for what it can and cannot do)
        from floorplan.video.rooms_as_photos import prepare_photo_input
        a.input, extra = prepare_photo_input(a.input, a.video_rooms, out / "rooms_as_photos")
        a.tier = "photo"
        PHOTO_OVERRIDES.update({k: v for k, v in extra.items() if k not in PHOTO_OVERRIDES})
    cfg = Config.load(a.config)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)  # noqa: E731
    report = dict(input=str(a.input), tier=a.tier, extractor=a.extractor, config=cfg.to_dict())

    scene, info, fe_report, rel_sigma, rel_reason = front_end(a.input, a.tier, cfg, out, not a.no_drift, log)
    run_warnings = fe_report.pop("warnings", [])
    if run_warnings:
        report["warnings"] = run_warnings      # top level, so nobody has to dig for them
    report["front_end"] = fe_report
    if info.get("floor_y") is None:
        # D-032: never crash on a missing floor; estimate it, say how, and keep it in the report
        from floorplan.plan.floor_fallback import estimate_floor
        fb = estimate_floor(scene, info)
        info["floor_y"] = fb["floor_y"]
        info["floor_fallback"] = fb
        report["floor_fallback"] = fb
        log(f"floor not detected by the front-end -> {fb['source']} (floor_y {fb['floor_y']})")
    save_scene(scene, info, out / "scene")
    log("scene saved")

    def _extract(extractor):
        if extractor == "alpha":
            from floorplan.plan.alpha import extract_plan
            return extract_plan(scene, info, name)
        from floorplan.plan.beta import extract_plan
        # plan_beta v4: tier-aware parameters (video/photo: raw-point surface, surface-noise term; photo: one room
        # per folder, named after it, per-folder widening, adjacency from the photo links). LiDAR is unchanged.
        return extract_plan(scene, info, name, tier=a.tier)

    # D-058: the plan step must never hang a live run (an orphaned photo run sat in extract_plan for 3 h). A hard
    # time limit; on expiry the alpha extractor gets its own limit; if that fails too, a clean error, not a hang.
    try:
        plan = _with_time_limit(lambda: _extract(a.extractor), a.plan_timeout)
    except StageTimeout:
        report["plan_timeout"] = dict(extractor=a.extractor, limit_s=a.plan_timeout)
        log(f"PLAN STEP TIMED OUT after {a.plan_timeout} s ({a.extractor}); retrying with the alpha extractor")
        if a.extractor == "alpha":
            raise SystemExit(f"plan extraction exceeded {a.plan_timeout} s")
        plan = _with_time_limit(lambda: _extract("alpha"), a.plan_timeout)
        report["plan_timeout"]["fallback"] = "alpha"
    plan.tier = a.tier
    dropped = drop_implausible_openings(plan)
    if dropped:
        log(f"dropped {len(dropped)} implausible opening(s): {[d['id'] for d in dropped]}")
    if info.get("floor_fallback"):
        plan.meta["floor_fallback"] = info["floor_fallback"]
    if a.tier in ("video", "photo") and not a.no_semantic_openings:
        # doors and windows from the segmenter's door / window pixels, measured on the plan's walls; geometric openings
        # win where both exist. Before the tier widening, so the new widths get the scale term like every other one.
        from floorplan.openings.semantic import add_semantic_openings
        work = out / "work" if a.tier == "video" else a.input.parent / f"photo_work__{a.input.name}"
        report["semantic_openings"] = add_semantic_openings(plan, scene, info, a.tier, work, log=log)
    if a.tier == "lidar":
        # D-021: Apple LiDAR surfaces sit ~0.7 cm into the room vs a laser; correct it and carry its uncertainty.
        # With --no-bias-correction the values stay raw but the interval is still widened one-sided (D-027).
        from floorplan.uncertainty.bias import LidarBiasConfig, apply_to_plan
        apply_to_plan(plan, LidarBiasConfig.load(enabled=not a.no_bias_correction))
    widen_plan(plan, rel_sigma, rel_reason)
    if fe_report.get("reliability"):
        plan.meta["reliability"] = fe_report["reliability"]
    if run_warnings:
        plan.meta["warnings"] = list(run_warnings)
    report["intervals_clipped_at_zero"] = clip_nonnegative(plan)
    log(f"plan: {len(plan.rooms)} rooms, {len(plan.walls)} walls, {len(plan.openings)} openings")

    if not a.no_damage:
        try:
            from floorplan.damage import run_damage_on_plan  # wired after the damage module lands
            # D-058: damage is an add-on to the plan; if it overruns, the plan is written without it (sim 88 s)
            plan = _with_time_limit(lambda: run_damage_on_plan(plan, scene, info, a.input, a.tier, log=log,
                                                               out_dir=out / "damage"), a.damage_timeout)
        except ImportError as e:
            report["damage"] = f"not available yet: {e}"
        except StageTimeout:
            report["damage"] = f"skipped: exceeded {a.damage_timeout} s (the plan is complete without it)"
            log(f"DAMAGE STEP TIMED OUT after {a.damage_timeout} s; plan written without damage annotations")

    if not a.no_room_names:
        # D-078: Bedroom, Kitchen, ... from the ADE20K classes of each room's images, then shape. Photo: the class
        # maps of the photo work dir; video: the keyframes in work/; LiDAR: frames decoded from rgb.mp4. Video and
        # LiDAR pixels vote for the room their 3D point lies in (depth + pose).
        from floorplan.plan.room_types import name_rooms
        work = a.input.parent / f"photo_work__{a.input.name}" if a.tier == "photo" else out / "work"
        cache = {"photo": work, "video": out / "work", "lidar": out / "names"}[a.tier]
        rep = name_rooms(plan, scene, info, a.tier, a.input, work, cache, log=log)
        report["room_names"] = {k: v for k, v in rep.items() if k != "rooms"}

    if a.door_priors and a.door_priors != "none":
        # D-085: after the names (a bathroom door's prior is 0.70 m) and the widening (a prior width is not a scaled
        # measurement)
        from floorplan.openings.priors import add_prior_doors
        report["door_priors"] = add_prior_doors(plan, scene, info, a.tier, log=log, rules=a.door_priors)

    from floorplan.export.dxf import export_dxf
    from floorplan.export.json_export import save_plan_json
    from floorplan.export.render import render_plan
    problems = save_plan_json(plan, out / "plan.json")
    report["schema_problems"] = problems
    render_plan(plan, out / "plan")
    try:
        from floorplan.export.presentation import render_presentation
        st = render_presentation(plan, out / "plan_presentation")
        report["presentation"] = {"doors": st.doors, "windows": st.windows, "passages": st.passages,
                                  "labels_forced": st.labels_forced, "warnings": st.warnings}
    except Exception as e:  # noqa: BLE001 - a drawing failure must not cost the run its plan files
        report["presentation"] = f"failed: {type(e).__name__}: {e}"
        log(f"WARNING: presentation drawing failed: {type(e).__name__}: {e}")
    export_dxf(plan, out / "plan.dxf")
    report["runtime_s"] = round(time.time() - t0, 1)
    (out / "run_report.json").write_text(json.dumps(report, indent=1, default=str))
    for w in run_warnings:
        log(f"WARNING: {w}")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
