"""Stage 1 of the LiDAR tier: capture -> aligned 3D scene (fused surface + raw points + levels).

Output (saved as outputs/<capture>/scene.npz):
  points/normals/colors   TSDF surface in the ALIGNED frame (gravity-levelled, walls along x and z)
  raw_points, raw_range   filtered LiDAR points in the aligned frame (used for final measurements, D-007)
  raw_ray, raw_frame      per raw point: viewing-ray direction and source frame index (single-visit measurement)
  T_align                 4x4 capture-world -> aligned-world
  traj                    camera centres (aligned frame), all frames; kf = keyframe indices
  floor_y, ceiling_y      global levels (None if not observed)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from floorplan.config import Config
from floorplan.io.stray import load_stray
from floorplan.plan.align import align_scene
from floorplan.recon.fusion import collect_raw_points, fuse_tsdf
from floorplan.recon.keyframes import select_keyframes


def build_scene(capture_dir, cfg: Config, T_wc: np.ndarray | None = None, log=print):
    t0 = time.time()
    cap = load_stray(capture_dir)
    T_wc = cap.T_wc if T_wc is None else T_wc
    kf = select_keyframes(cap, cfg, T_wc)
    log(f"[scene] {cap.capture_id}: {cap.n} frames, {len(kf)} keyframes ({time.time()-t0:.1f}s)")
    fused = fuse_tsdf(cap, kf, T_wc, cfg, with_color=True)
    log(f"[scene] TSDF: {len(fused['points'])} surface points from {fused['frames_used']} frames ({time.time()-t0:.1f}s)")
    raw = collect_raw_points(cap, kf, T_wc, cfg)
    log(f"[scene] raw LiDAR points kept: {len(raw['points'])} ({time.time()-t0:.1f}s)")
    T_align, P, N, info = align_scene(fused["points"], fused["normals"], T_wc[:, :3, 3], cfg)
    R, t = T_align[:3, :3], T_align[:3, 3]
    scene = dict(
        points=P.astype(np.float32), normals=N.astype(np.float32), colors=fused["colors"],
        raw_points=(raw["points"] @ R.T + t).astype(np.float32), raw_range=raw["range"],
        raw_ray=(raw["ray"].astype(np.float32) @ R.T).astype(np.float16), raw_frame=raw["frame"],
        T_align=T_align, traj=(T_wc[:, :3, 3] @ R.T + t).astype(np.float32), kf=kf,
        T_wc=T_wc, timestamps=cap.timestamps)
    info.update(capture_id=cap.capture_id, frames=cap.n, keyframes=int(len(kf)),
                surface_points=int(len(P)), raw_points=int(len(raw["points"])),
                runtime_s=round(time.time() - t0, 1), config=cfg.to_dict())
    log(f"[scene] aligned: tilt {info['floor_tilt_before_deg']:.2f} deg, yaw {info['manhattan_yaw_deg']:.1f} deg, "
        f"Manhattan score {info['manhattan_score']:.2f}, floor {info['floor_y']}, ceiling {info['ceiling_y']}")
    return scene, info, cap


def save_scene(scene: dict, info: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "scene.npz", **{k: v for k, v in scene.items() if v is not None})
    (out_dir / "scene_info.json").write_text(json.dumps(info, indent=2, default=float))


def load_scene(out_dir: Path):
    z = np.load(Path(out_dir) / "scene.npz", allow_pickle=False)
    scene = {k: z[k] for k in z.files}
    info = json.loads((Path(out_dir) / "scene_info.json").read_text())
    return scene, info
