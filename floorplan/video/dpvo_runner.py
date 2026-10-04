"""Run DPVO (Deep Patch Visual Odometry, MIT) on a video and save one up-to-scale camera pose per used frame.

This file is executed as a SCRIPT by the separate `envs/dpvo` interpreter (DPVO's compiled CUDA ops and lietorch
live there), so it imports nothing from `floorplan` and only needs numpy + OpenCV + DPVO. floorplan/video/vo.py
builds the command line and reads the result.

Why DPVO: the sample videos were recorded for LiDAR scanning, so the phone sweeps 0.3-1 m from blank walls and
curtains. Sparse-feature SfM finds no matches on those stretches and breaks into fragments (decision V-3). DPVO
tracks every frame at 30-60 fps, where consecutive frames overlap almost completely, and its learned patch
correlation works on weak texture where keypoint detectors find nothing.

Usage: python dpvo_runner.py VIDEO OUT.npz --rot 90 --stride 2 --height 640 --fx .. --fy .. --cx .. --cy ..
       [--loop-closure] [--repo PATH_TO_DPVO]
"""
import argparse
import sys
import time

import cv2
import numpy as np

ROTATE = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--height", type=int, default=640)       # upright image height after resizing (multiple of 16)
    for k in ("fx", "fy", "cx", "cy"):
        ap.add_argument(f"--{k}", type=float, required=True)  # at the resized upright resolution
    ap.add_argument("--loop-closure", action="store_true")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)        # first decoded frame (v2: fresh run per VO segment)
    ap.add_argument("--end", type=int, default=-1)         # one past the last decoded frame; -1 = to the end
    a = ap.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    if a.repo:
        sys.path.insert(0, a.repo)
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    torch.manual_seed(a.seed)
    torch.backends.cudnn.deterministic = True     # D-034 note: cuDNN picks non-deterministic kernels by default
    torch.backends.cudnn.benchmark = False
    np.random.seed(a.seed)
    cfg.merge_from_file(f"{a.repo}/config/default.yaml" if a.repo else "config/default.yaml")
    if a.loop_closure:
        cfg.merge_from_list(["LOOP_CLOSURE", True])
    ckpt = hf_hub_download("pablovela5620/dpvo", "dpvo.pth", revision="c998d3b57bf47c619f851d37dff0aa1fa43e1c34")
    intr = torch.tensor([a.fx, a.fy, a.cx, a.cy], dtype=torch.float32, device="cuda")

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    slam, used, t0, i = None, [], time.time(), -1
    with torch.no_grad():                                 # required: DPVO otherwise keeps autograd graphs (OOM)
        while True:
            ok = cap.grab()
            if not ok:
                break
            i += 1
            if a.end >= 0 and i >= a.end:
                break
            if i < a.start or i % a.stride:
                continue
            _, bgr = cap.retrieve()
            if ROTATE[a.rot] is not None:
                bgr = cv2.rotate(bgr, ROTATE[a.rot])
            h, w = bgr.shape[:2]
            W = int(round(w * a.height / h)) // 16 * 16
            bgr = cv2.resize(bgr, (W, a.height), interpolation=cv2.INTER_AREA)
            img = torch.from_numpy(bgr).permute(2, 0, 1).cuda()
            if slam is None:
                slam = DPVO(cfg, ckpt, ht=img.shape[1], wd=img.shape[2], viz=False)
            slam(float(len(used)), img, intr)
            used.append(i)
        poses, _ = slam.terminate()                         # (N,7) camera-to-world: tx ty tz qx qy qz qw
    np.savez(a.out, frames=np.asarray(used), poses=poses, runtime_s=time.time() - t0,
             peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    print(f"DPVO: {len(used)} frames in {time.time() - t0:.1f}s, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
