"""Call DPVO in its own interpreter and return up-to-scale camera-to-world poses per video frame.

DPVO needs compiled CUDA extensions (lietorch, cuda_corr, cuda_ba) built against its own environment, so it runs as
a subprocess (`envs/dpvo/bin/python floorplan/video/dpvo_runner.py ...`). Paths come from environment variables so
nothing machine-specific is hard-coded: FLOORPLAN_DPVO_PYTHON, FLOORPLAN_DPVO_REPO, FLOORPLAN_DPVO_SHIMS.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# default: envs/ and third_party/ sit next to the repo, in its parent folder
MAPPING_ROOT = Path(os.environ.get("FLOORPLAN_THIRD_PARTY_ROOT", Path(__file__).resolve().parents[3]))
DPVO_PYTHON = Path(os.environ.get("FLOORPLAN_DPVO_PYTHON", MAPPING_ROOT / "envs/dpvo/bin/python"))
DPVO_REPO = Path(os.environ.get("FLOORPLAN_DPVO_REPO", MAPPING_ROOT / "third_party/dpvo"))
DPVO_SHIMS = Path(os.environ.get("FLOORPLAN_DPVO_SHIMS", MAPPING_ROOT / "third_party/dpvo_shims"))


def run_dpvo(video: Path, out: Path, rot_cw: int, f_px: float, upright_size: tuple[int, int], params,
             log=print, start: int = 0, end: int = -1) -> dict:
    """Run DPVO; f_px is the focal length in pixels of the full-resolution upright frame of size (W, H)."""
    W, H = upright_size
    s = params.dpvo_height / H
    w_used = int(round(W * s)) // 16 * 16
    fx = f_px * s
    cmd = [str(DPVO_PYTHON), str(Path(__file__).with_name("dpvo_runner.py")), str(video), str(out),
           "--rot", str(rot_cw), "--stride", str(params.dpvo_stride), "--height", str(params.dpvo_height),
           "--fx", f"{fx}", "--fy", f"{fx}", "--cx", f"{w_used / 2}", "--cy", f"{params.dpvo_height / 2}",
           "--repo", str(DPVO_REPO), "--seed", str(params.seed), "--start", str(start), "--end", str(end)]
    if params.dpvo_loop_closure:
        cmd.append("--loop-closure")
    env = dict(os.environ, PYTHONPATH=str(DPVO_SHIMS), HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
    r = subprocess.run(cmd, cwd=DPVO_REPO, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"DPVO failed:\n{r.stderr[-3000:]}")
    log(f"[vo] {r.stdout.strip().splitlines()[-1]}")
    return load_dpvo(out)


def load_dpvo(out: Path) -> dict:
    """Read a DPVO result file written by dpvo_runner.py."""
    z = np.load(out)
    P = z["poses"]
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(P[:, 3:]).as_matrix()
    T[:, :3, 3] = P[:, :3]
    return dict(frames=z["frames"], T_wc=T, runtime_s=float(z["runtime_s"]), peak_gb=float(z["peak_gb"]))


def interpolate_poses(frames: np.ndarray, T: np.ndarray, n: int) -> np.ndarray:
    """Poses for every video frame 0..n-1: linear in position, spherical-linear (slerp) in rotation."""
    from scipy.spatial.transform import Slerp
    q = np.clip(np.arange(n), frames[0], frames[-1])
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = Slerp(frames, Rotation.from_matrix(T[:, :3, :3]))(q).as_matrix()
    for a in range(3):
        out[:, a, 3] = np.interp(q, frames, T[:, a, 3])
    return out
