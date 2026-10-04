"""Upright rotation, focal-length prior and gravity from the images alone, with GeoCalib (Apache-2.0 code,
CC-BY-4.0 weights).

Why this is needed:
  * Rotation. A phone held in portrait stores sideways frames unless the container carries a rotation flag. Learned
    models (depth, features, GeoCalib itself) are trained on upright photos, and their accuracy drops on sideways
    input. Real iPhone Camera-app videos carry the flag; Stray Scanner's rgb.mp4 does not, and we may not read the
    IMU in this tier. So we try all four 90-degree rotations on a few frames and keep the one where GeoCalib's up vector
    points up the screen. The container flag, when present, is used directly.
  * Focal length. The video tier has no intrinsics file. GeoCalib predicts the focal length from vanishing
    structure in a single image; the median over many frames is a good prior for SfM self-calibration.
  * Gravity. SfM reconstructs in an arbitrary frame. GeoCalib's per-frame up vector, rotated into the SfM frame by
    the reconstructed camera orientations, gives "up" for the whole scene (refined later on the floor plane).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from floorplan.video.frames import ROTATE_CODES, iter_frames, resize_long, upright

GEOCALIB_INPUT = 320   # GeoCalib resizes internally to 320 px on the short side; feeding more only costs decode time


def _load_geocalib(device: str):
    from geocalib import GeoCalib
    return GeoCalib(weights="pinhole").to(device).eval()


def _to_tensor(bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)


@torch.inference_mode()
def _calibrate(model, img: torch.Tensor, seed: int) -> dict:
    torch.manual_seed(seed)                          # GeoCalib's head draws random bases on every call
    res = model.calibrate(img)
    up = res["gravity"].vec3d[0].cpu().numpy()       # UP in the OpenCV camera frame (x right, y down, z forward)
    roll, pitch = np.degrees(res["gravity"].rp[0].cpu().numpy())
    f = float(res["camera"].f[0, 1])
    return dict(up=up, roll=float(roll), pitch=float(pitch), f=f, f_sigma=float(res["focal_uncertainty"][0]))


def detect_rotation(video: Path, frame_ids: np.ndarray, n_probe: int, rotation_meta: int,
                    device: str = "cuda", seed: int = 0) -> tuple[int, dict]:
    """Clockwise rotation (0/90/180/270) that makes the stored frames upright, and the vote evidence."""
    if rotation_meta:
        return rotation_meta, dict(source="container metadata", rotation=rotation_meta)
    probe = frame_ids[np.linspace(0, len(frame_ids) - 1, min(n_probe, len(frame_ids))).round().astype(int)]
    model = _load_geocalib(device)
    angles = {r: [] for r in ROTATE_CODES}
    for _, bgr in iter_frames(video, probe):
        small = resize_long(bgr, 2 * GEOCALIB_INPUT)
        for r in ROTATE_CODES:
            up = _calibrate(model, _to_tensor(upright(small, r), device), seed)["up"]
            # direction of "up" projected into the image, measured from the image's -y axis (screen up).
            # GeoCalib's roll alone is ambiguous here: an upside-down frame can also report a small roll.
            angles[r].append(float(np.degrees(np.arctan2(up[0], -up[1]))))
    del model
    torch.cuda.empty_cache()
    # the upright candidate is the one where up points up the screen (angle near 0) in MOST FRAMES (per-frame votes).
    # D-059: the median angle is not a usable score: GeoCalib reads almost any rotated frame as "nearly upright", so
    # the medians tie (sim walk: 0 deg 1.39 vs 180 deg 1.33; real samples 4.7-12 for every candidate) and the smaller
    # one is noise. That picked 180 deg for an upright sim video -> depth on upside-down frames -> 0 rooms. The votes
    # are decisive on every capture we have (sim 0: 7 of 12; real 90: 7-8 of 12); ties go to the container value.
    score = {r: float(np.median(np.abs(v))) for r, v in angles.items()}
    stacked = np.abs(np.array([angles[r] for r in ROTATE_CODES]))
    winners = np.array(list(ROTATE_CODES))[np.argmin(stacked, axis=0)]
    votes = {r: int(np.sum(winners == r)) for r in ROTATE_CODES}
    top = max(votes.values())
    tied = [r for r in ROTATE_CODES if votes[r] == top]
    best = rotation_meta if rotation_meta in tied else min(tied, key=score.get)
    return best, dict(source="GeoCalib up-vector vote", rotation=best, median_abs_up_angle_deg=score, votes=votes,
                      probe_frames=probe.tolist())


def gravity_and_focal(image_files: list[Path], every: int, device: str = "cuda", seed: int = 0) -> dict:
    """GeoCalib on every `every`-th upright keyframe: per-frame up vector (camera frame), pitch, focal (px)."""
    model = _load_geocalib(device)
    idx, ups, pitch, focal, fsig = [], [], [], [], []
    for k in range(0, len(image_files), every):
        bgr = cv2.imread(str(image_files[k]))
        r = _calibrate(model, _to_tensor(bgr, device), seed)
        idx.append(k)
        ups.append(r["up"])
        pitch.append(r["pitch"])
        focal.append(r["f"])
        fsig.append(r["f_sigma"])
    del model
    torch.cuda.empty_cache()
    return dict(index=np.asarray(idx), up_cam=np.asarray(ups), pitch_deg=np.asarray(pitch),
                focal_px=np.asarray(focal), focal_sigma_px=np.asarray(fsig))
