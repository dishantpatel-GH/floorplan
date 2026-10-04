"""Evaluation of the photo tier against the simulated-photo truth and the LiDAR scene (NEVER used as pipeline input).

What is scored, and why:
  * depth scale vs LiDAR, per photo: the truth file names the video frame each photo came from, so the LiDAR depth of
    that frame is the reference for the photo's predicted depth (median ratio = scale error of that photo);
  * camera poses vs truth after a similarity alignment: is the room's internal geometry right (shape), separately
    from its scale;
  * whole-property camera placement after ONE similarity: are the rooms placed correctly relative to each other
    (the stitch), and what single scale error remains;
  * point-to-LiDAR-surface distances and per-room extents vs the LiDAR plan.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def umeyama(X: np.ndarray, Y: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising ||Y - (s R X + t)||: maps estimate X onto reference Y."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / max((Xc ** 2).sum() / len(X), 1e-12)) if with_scale else 1.0
    return s, R, my - s * R @ mx


def load_truth(photo_root: Path) -> dict | None:
    f = Path(photo_root).parent / "truth" / "photo_truth.json"
    return json.loads(f.read_text()) if f.exists() else None


def lidar_depth_for_photo(capture_dir: Path, t: dict, full_hw: tuple[int, int], crop, model_wh) -> np.ndarray:
    """LiDAR depth of the photo's source frame, rotated and cropped exactly like the photo (0 = invalid)."""
    d = cv2.imread(str(Path(capture_dir) / "depth" / f"{t['frame']:06d}.png"), -1).astype(np.float32) / 1000
    c = cv2.imread(str(Path(capture_dir) / "confidence" / f"{t['frame']:06d}.png"), -1)
    d[c < 2] = 0
    d = np.rot90(d, k=t["rot90"]).copy()
    h, w = full_hw
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    ox, oy, s = crop
    W, H = model_wh
    d = d[int(oy):int(oy + round(H / s)), int(ox):int(ox + round(W / s))]
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)


def depth_ratio(pred: np.ndarray, ref: np.ndarray, max_m: float = 4.0) -> tuple[float, float, int]:
    """(median pred/ref, abs-rel error after removing that median, valid pixel count)."""
    ok = (ref > 0.2) & (ref < max_m) & (pred > 0)
    if ok.sum() < 100:
        return float("nan"), float("nan"), int(ok.sum())
    r = pred[ok] / ref[ok]
    m = float(np.median(r))
    return m, float(np.median(np.abs(pred[ok] / m - ref[ok]) / ref[ok])), int(ok.sum())


def relative_rotation_errors(T_est: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """Angle (deg) between estimated and true RELATIVE rotations of every camera pair: frame-independent."""
    out = []
    for i in range(len(T_est)):
        for j in range(i + 1, len(T_est)):
            a = T_est[i, :3, :3].T @ T_est[j, :3, :3]
            b = T_ref[i, :3, :3].T @ T_ref[j, :3, :3]
            out.append(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))
    return np.array(out)


def align_rotation_first(T_est: np.ndarray, T_ref: np.ndarray, with_scale: bool = True):
    """Similarity estimate->reference whose rotation comes from the camera ORIENTATIONS (chordal mean), then scale and
    shift from the centres. With 2-8 photos whose centres span a metre or two, a centres-only (Umeyama) fit is
    ill-conditioned; orientations are not."""
    M = sum(b[:3, :3] @ a[:3, :3].T for a, b in zip(T_est, T_ref))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    X = T_est[:, :3, 3] @ R.T
    Y = T_ref[:, :3, 3]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    s = float((Xc * Yc).sum() / max((Xc ** 2).sum(), 1e-12)) if with_scale else 1.0
    return s, R, Y.mean(0) - s * X.mean(0)


def pose_errors(T_est: np.ndarray, T_ref: np.ndarray, with_scale: bool = True) -> dict:
    """Relative-rotation errors, and camera-centre errors after a rotation-first similarity alignment."""
    if len(T_est) < 2:
        return dict(n=len(T_est))
    s, R, t = align_rotation_first(T_est, T_ref, with_scale)
    C = s * T_est[:, :3, 3] @ R.T + t
    err = np.linalg.norm(C - T_ref[:, :3, 3], axis=1)
    rot = relative_rotation_errors(T_est, T_ref)
    return dict(n=len(T_est), scale=s, scale_error_pct=100 * (1 / s - 1) if s > 0 else None,
                centre_rmse_m=float(np.sqrt((err ** 2).mean())), centre_max_m=float(err.max()),
                rel_rot_median_deg=float(np.median(rot)), rel_rot_max_deg=float(np.max(rot)), sim3=(s, R, t))


def truth_T_wc(t: dict) -> np.ndarray:
    """True camera-to-world pose of the UPRIGHT photo. The simulator rotated the video frame by k quarter turns
    counter-clockwise (np.rot90), which rolls the camera: p_photo = A^k p_frame with A = [[0,1,0],[-1,0,0],[0,0,1]],
    so R_photo = R_frame (A^k)^T."""
    A = np.linalg.matrix_power(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float), int(t["rot90"]) % 4)
    T = np.array(t["T_wc"], float)
    T[:3, :3] = T[:3, :3] @ A.T
    return T
