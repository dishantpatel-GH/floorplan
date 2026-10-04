#!/usr/bin/env python
"""Render a recorded session (session.json) into clean camera frames with Isaac Sim (RTX, headless).

Only colour is rendered. Depth, the LiDAR artefacts, phone image processing and the file formats are added by
emulate.py, which runs in the main .venv and can be re-run with other noise settings without rendering again
(decision D-035; the ray-cast depth equals Isaac's own depth to 0.00 mm, sim/README.md).

Human imperfection (decision D-038): the recorded or scripted path is the *intended* path. Before rendering, every
walk gets the slow drift of height (~+-7 cm), tilt (~+-5 deg), roll (~+-2 deg), aim and side sway that real people
show, plus hand tremor (rig.humanize); every photo gets its own aim/height/tilt error (rig.humanize_photo, on top of
the varying turn steps that auto_capture.py already scripts). The perturbed pose is the true pose of the frame.

Output (<session>/render/):
  <take>/frames.mp4   1920x1440, near-lossless H.264 (crf 12), constant frame rate
  <take>/poses.npz    t, T (n,4,4) camera-to-world in the scene's USD world with USD camera axes, K, lighting
  photos/NNNN.png     2000x1500 stills, and photos.json (true pose, folder, capture time, lighting)

Usage: envs/isaacsim_np126/bin/python floorplan-capture/sim/render.py <session_dir> [--fps 30] [--human 1.0] [--seed 0]
       [--takes take1,take2] [--no-photos] [--no-takes] [--max-frames N] [--subframes 1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import zlib
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session", type=Path)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--human", type=float, default=None, help="imperfection level (default: session meta or 1.0)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--takes", default="", help="comma list (default all)")
ap.add_argument("--no-photos", action="store_true")
ap.add_argument("--no-takes", action="store_true")
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--subframes", type=int, default=1, help="RTX subframes per video frame (photos use 8)")
ap.add_argument("--lighting", default="", help="override every take/photo: day|dim|dark")
ap.add_argument("--only", default="", help="photos: re-render just these photo indices (comma list); the other photos "
                "and their photos.json entries are kept (already-humanised poses render identically)")
ap.add_argument("--chunks", type=int, default=1, help="render N time-chunks of a walk at once with N cameras "
                "(the Replicator step overhead, not pixels, dominates; D-048)")
ap.add_argument("--poses-from", type=Path, help="reuse the TRUE (already humanised) poses of another render's poses.npz, "
                "interpolated to --fps (same path at another frame rate; D-041 check)")
args = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import isaac_common as ic  # noqa: E402

sess = json.loads((args.session / "session.json").read_text())
scene = sess["scene"]
app = ic.start_app(headless=True)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from sim.rig import camera_to_world, humanize, humanize_photo, resample  # noqa: E402

carb_settings = None
try:
    import carb
    carb_settings = carb.settings.get_settings()
    carb_settings.set("/omni/replicator/captureOnPlay", False)
except Exception:
    pass

gt_dir, gt = ic.load_scene_assets(scene)
stage = ic.open_stage(app, scene)
ic.add_sheets(stage, gt["sheets"])
out = args.session / "render"
out.mkdir(parents=True, exist_ok=True)
level = args.human if args.human is not None else float(sess.get("meta", {}).get("human_level", 1.0))
cur_light = None


def light(preset):
    global cur_light
    preset = args.lighting or preset or "day"
    if preset != cur_light:
        ic.set_lighting(stage, preset)
        cur_light = preset
    return preset


def K_of(W, H, f):
    return np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1.0]])


def render_take(tk):
    tdir = out / tk["name"]
    tdir.mkdir(parents=True, exist_ok=True)
    if args.poses_from:
        src = np.load(args.poses_from)["pose_rows"]
        tmax = tk["samples"][-1][0] - tk["samples"][0][0]
        src = src[src[:, 0] <= tmax + 1e-6]
        P = resample(src.tolist(), args.fps)                  # same humanised path, new frame rate
    else:
        P = resample(tk["samples"], args.fps)
        P = humanize(P, level=level if not tk.get("humanized") else 0.0, seed=args.seed * 1000 + zlib.crc32(tk["name"].encode()) % 997)
    if args.max_frames:
        P = P[: args.max_frames]
    if args.chunks > 1:
        return render_take_chunked(tk, tdir, P)
    cam, op = ic.make_camera(stage, f"/World/Cam_{tk['name']}", ic.WALK_W, ic.WALK_H, ic.WALK_F)
    rp = rep.create.render_product(str(cam.GetPath()), (ic.WALK_W, ic.WALK_H))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    lt = light(tk.get("lighting"))
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                            f"{ic.WALK_W}x{ic.WALK_H}", "-r", f"{args.fps}", "-i", "-", "-c:v", "libx264", "-crf", "12",
                            "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", f"{args.fps}",
                            str(tdir / "frames.mp4")], stdin=subprocess.PIPE)
    Ts = np.empty((len(P), 4, 4))
    t0 = time.time()
    for i, row in enumerate(P):
        T = camera_to_world(*row[1:7])
        Ts[i] = T
        ic.set_pose(op, T)
        rep.orchestrator.step(rt_subframes=8 if i == 0 else args.subframes, pause_timeline=True)
        rgb = np.asarray(ann.get_data())[..., :3]
        enc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        if i % 100 == 0:
            el = time.time() - t0
            print(f"[render] {tk['name']} {i}/{len(P)}  {el / max(i, 1):.2f} s/frame", flush=True)
    enc.stdin.close()
    enc.wait()
    ann.detach()
    rp.destroy()
    np.savez(tdir / "poses.npz", t=P[:, 0], T=Ts, nominal=resample(tk["samples"], args.fps)[: len(P)], K=K_of(ic.WALK_W, ic.WALK_H, ic.WALK_F),
             W=ic.WALK_W, H=ic.WALK_H, fps=args.fps, lighting=lt, human_level=level, seed=args.seed,
             pose_rows=P)
    print(f"[render] {tk['name']}: {len(P)} frames in {time.time() - t0:.0f}s -> {tdir}", flush=True)


def render_take_chunked(tk, tdir, P):
    """N cameras render N contiguous chunks of the walk in the same Replicator steps; each chunk is encoded to its
    own file and the parts are concatenated losslessly. Temporal accumulation stays per camera (continuous motion)."""
    N = args.chunks
    bounds = np.linspace(0, len(P), N + 1).astype(int)
    lt = light(tk.get("lighting"))
    cams = []
    for k in range(N):
        cam, op = ic.make_camera(stage, f"/World/Cam_{tk['name']}_{k}", ic.WALK_W, ic.WALK_H, ic.WALK_F)
        rp = rep.create.render_product(str(cam.GetPath()), (ic.WALK_W, ic.WALK_H))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        part = tdir / f"part{k:02d}.mp4"
        enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                f"{ic.WALK_W}x{ic.WALK_H}", "-r", f"{args.fps}", "-i", "-", "-c:v", "libx264", "-crf", "12",
                                "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", f"{args.fps}", str(part)],
                               stdin=subprocess.PIPE)
        cams.append(dict(op=op, ann=ann, enc=enc, part=part, a=bounds[k], b=bounds[k + 1]))
    Ts = np.empty((len(P), 4, 4))
    steps = int(max(c["b"] - c["a"] for c in cams))
    t0 = time.time()
    for j in range(steps):
        live = [c for c in cams if c["a"] + j < c["b"]]
        for c in live:
            i = c["a"] + j
            T = camera_to_world(*P[i, 1:7])
            Ts[i] = T
            ic.set_pose(c["op"], T)
        rep.orchestrator.step(rt_subframes=8 if j == 0 else args.subframes, pause_timeline=True)
        for c in live:
            rgb = np.asarray(c["ann"].get_data())[..., :3]
            c["enc"].stdin.write(np.ascontiguousarray(rgb).tobytes())
        if j % 100 == 0:
            print(f"[render] {tk['name']} chunked x{N}: step {j}/{steps}  {(time.time() - t0) / max(j * N, 1):.3f} s/frame",
                  flush=True)
    for c in cams:
        c["enc"].stdin.close()
        c["enc"].wait()
    lst = tdir / "parts.txt"
    lst.write_text("".join(f"file '{c['part'].name}'\n" for c in cams))
    subprocess.call(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                     str(tdir / "frames.mp4")])
    for c in cams:
        c["part"].unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    np.savez(tdir / "poses.npz", t=P[:, 0], T=Ts, nominal=P, K=K_of(ic.WALK_W, ic.WALK_H, ic.WALK_F),
             W=ic.WALK_W, H=ic.WALK_H, fps=args.fps, lighting=lt, human_level=level, seed=args.seed, pose_rows=P)
    print(f"[render] {tk['name']}: {len(P)} frames in {time.time() - t0:.0f}s with {N} cameras -> {tdir}", flush=True)


def render_photos(photos):
    pdir = out / "photos"
    pdir.mkdir(parents=True, exist_ok=True)
    cam, op = ic.make_camera(stage, "/World/PhotoCam", ic.PHOTO_W, ic.PHOTO_H, ic.PHOTO_F)
    rp = rep.create.render_product(str(cam.GetPath()), (ic.PHOTO_W, ic.PHOTO_H))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    from PIL import Image
    rng = np.random.default_rng(args.seed + 7)
    meta = []
    t0 = time.time()
    only = {int(x) for x in args.only.split(",") if x.strip()}
    kept = {}
    if only and (pdir / "photos.json").exists():
        kept = {m["file"]: m for m in json.loads((pdir / "photos.json").read_text())["photos"]}
    for p in photos:
        if only and p["index"] not in only and f"{p['index']:04d}.jpg" in kept:
            meta.append(dict(kept[f"{p['index']:04d}.jpg"], folder=p["folder"], t=p["t"]))
            continue
        pose = humanize_photo(p["pose"], level=level if not p.get("humanized") else 0.0, rng=rng)
        f_p = float(p.get("f_px", ic.PHOTO_F))                # v2.2 (D-065): per-photo lens (0.5x ultra-wide)
        cam.GetHorizontalApertureAttr().Set(18.0 * ic.PHOTO_W / f_p)
        cam.GetVerticalApertureAttr().Set(18.0 * ic.PHOTO_H / f_p)
        T = camera_to_world(*pose)
        lt = light(p.get("lighting"))
        ic.set_pose(op, T)
        rep.orchestrator.step(rt_subframes=12, pause_timeline=True)
        rgb = np.asarray(ann.get_data())[..., :3]
        name = f"{p['index']:04d}.jpg"                     # 12 MP: near-lossless JPEG (q98) instead of 25 MB PNGs
        Image.fromarray(np.ascontiguousarray(rgb)).save(pdir / name, quality=98)
        meta.append(dict(file=name, folder=p["folder"], t=p["t"], lighting=lt, pose=pose, nominal_pose=p["pose"],
                         T=T.tolist(), K=K_of(ic.PHOTO_W, ic.PHOTO_H, f_p).tolist(), lens=p.get("lens", "1x")))
    ann.detach()
    rp.destroy()
    (pdir / "photos.json").write_text(json.dumps(dict(K=K_of(ic.PHOTO_W, ic.PHOTO_H, ic.PHOTO_F).tolist(),
                                                     W=ic.PHOTO_W, H=ic.PHOTO_H, human_level=level, photos=meta), indent=1))
    print(f"[render] {len(meta)} photos in {time.time() - t0:.0f}s -> {pdir}", flush=True)


want = set(filter(None, args.takes.split(",")))
if not args.no_takes:
    for tk in sess["takes"]:
        if not want or tk["name"] in want:
            render_take(tk)
if not args.no_photos and sess["photos"]:
    render_photos(sess["photos"])
(out / "render.json").write_text(json.dumps(dict(session=str(args.session), scene=scene, fps=args.fps, human_level=level,
                                                seed=args.seed, finished=time.strftime("%Y-%m-%d %H:%M:%S"))))
app.close()
