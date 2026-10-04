"""Score the video tier against the LiDAR tier of the SAME capture (a reference, not ground truth).

The sample data is a Stray Scanner LiDAR capture, so every video frame has an ARKit pose and a LiDAR depth map.
The video tier never reads them; this module does, only to measure how far the image-only result is from them:

  * trajectory: camera centres after the best rigid (SE3, scale fixed to our metric estimate) alignment, which is
    what matters for measurements, and after the best similarity (Sim3) alignment, which isolates shape error from
    scale error. The Sim3 scale factor IS the scale error of the video tier.
  * cloud-to-cloud: distance from video-tier surface points to the LiDAR TSDF surface after the SE3 alignment.

Frame correspondence: OpenCV drops the first packet of rgb.mp4 (negative presentation time stamp), so decoded frame
i is odometry row i+1. We match by timestamp instead of by index so this cannot silently go wrong.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from floorplan.io.stray import load_stray


def camera_axes_after_rotation(rot_cw: int) -> np.ndarray:
    """R_SU: axes of the upright camera U expressed in the stored camera S, for an image rotated rot_cw clockwise.

    Rotating the picture 90 deg clockwise moves stored pixel (u, v) to (H-1-v, u), so x_U = -y_S and y_U = x_S.
    The optical axis (z) does not change."""
    c, s = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[rot_cw]
    # x_U = c*x_S - s*y_S ; y_U = s*x_S + c*y_S   (columns = U axes in S coordinates)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], float)


def match_frames(video_ts: np.ndarray, odom_ts: np.ndarray) -> np.ndarray:
    """Odometry row for each decoded video frame, by nearest timestamp (decoded frame 0 = odometry row 1)."""
    t = odom_ts - odom_ts[1]
    j = np.clip(np.searchsorted(t, video_ts), 1, len(t) - 1)
    return np.where(np.abs(t[j - 1] - video_ts) < np.abs(t[j] - video_ts), j - 1, j)


def arkit_reference(capture_dir, video_ts: np.ndarray, kf: np.ndarray, rot_cw: int) -> dict:
    """ARKit camera-to-world poses of the keyframes, expressed for the UPRIGHT image axes."""
    cap = load_stray(capture_dir)
    rows = match_frames(video_ts, cap.timestamps)[kf]
    T = cap.T_wc[rows].copy()
    T[:, :3, :3] = T[:, :3, :3] @ camera_axes_after_rotation(rot_cw)
    return dict(cap=cap, rows=rows, T_wc=T)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares s, R, t with dst ~ s * R @ src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / A.var(0).sum()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def rotation_errors(T_est: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """Per-frame camera orientation error (deg) after the best global rotation (chordal mean of R_ref R_est^T).

    Aligned on orientations, not on camera centres: a short, nearly straight camera path constrains the
    rotation about its own direction poorly, which would inflate the error."""
    M = np.einsum("nij,nkj->ik", T_ref[:, :3, :3], T_est[:, :3, :3])
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    rel = np.einsum("ij,njk->nik", R, T_est[:, :3, :3])
    rel = np.einsum("nji,njk->nik", rel, T_ref[:, :3, :3])
    return np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def trajectory_metrics(T_est: np.ndarray, T_ref: np.ndarray) -> dict:
    """SE3 and Sim3 alignment of camera centres; returns errors in metres and the residual scale."""
    C, G = T_est[:, :3, 3], T_ref[:, :3, 3]
    out = {}
    for tag, with_scale in (("se3", False), ("sim3", True)):
        s, R, t = umeyama(C, G, with_scale)
        err = np.linalg.norm((s * C @ R.T + t) - G, axis=1)
        out[tag] = dict(scale=s, ate_rmse_m=float(np.sqrt(np.mean(err ** 2))), ate_median_m=float(np.median(err)),
                        ate_max_m=float(err.max()),
                        R=R.tolist(), t=t.tolist())
    out["rot_err_median_deg"] = float(np.median(rotation_errors(T_est, T_ref)))
    path = float(np.sum(np.linalg.norm(np.diff(G, axis=0), axis=1)))
    out["path_length_m"] = path
    out["scale_error_pct"] = 100.0 * (1.0 / out["sim3"]["scale"] - 1.0)   # >0: our scene is too large
    out["frames"] = int(len(C))
    return out


def cloud_to_cloud(points: np.ndarray, ref_points: np.ndarray, max_d: float = 0.5) -> dict:
    """Nearest-neighbour distance from each point to the reference surface (one direction: accuracy)."""
    d, _ = cKDTree(ref_points).query(points, distance_upper_bound=max_d, workers=-1)
    far = ~np.isfinite(d)
    d = np.where(far, max_d, d)
    return dict(median_m=float(np.median(d)), p90_m=float(np.percentile(d, 90)), mean_m=float(d.mean()),
                frac_within_5cm=float((d < 0.05).mean()), frac_beyond_50cm=float(far.mean()), n=int(len(d)))


def window_scale_errors(T_est: np.ndarray, T_ref: np.ndarray, size: int = 30) -> list[dict]:
    """Scale error and shape error per window of `size` keyframes: shows whether scale drift was removed locally."""
    out = []
    for a in range(0, len(T_est) - size // 2, size):
        m = trajectory_metrics(T_est[a:a + size], T_ref[a:a + size])
        out.append(dict(start=a, scale_error_pct=m["scale_error_pct"], sim3_ate_m=m["sim3"]["ate_rmse_m"],
                        path_m=m["path_length_m"]))
    return out


def segment_errors(T_est: np.ndarray, T_ref: np.ndarray, seg: np.ndarray, info: dict) -> list[dict]:
    """Per scale segment (stretch between VO restarts): its own Sim3 scale error and shape error, the claimed sigma,
    and whether the claimed 95% interval covers the error. Segments shorter than 10 keyframes are skipped."""
    claimed = {s["id"]: s.get("sigma_rel", s.get("sigma_rel_learned")) for s in info["scale"]["segments"]}
    out = []
    for g in np.unique(seg):
        ids = np.where(seg == g)[0]
        if len(ids) < 10:
            continue
        m = trajectory_metrics(T_est[ids], T_ref[ids])
        sig = claimed.get(int(g))
        out.append(dict(segment=int(g), keyframes=[int(ids[0]), int(ids[-1])], path_m=m["path_length_m"],
                        scale_error_pct=m["scale_error_pct"], sim3_ate_m=m["sim3"]["ate_rmse_m"],
                        se3_ate_m=m["se3"]["ate_rmse_m"], claimed_sigma_pct=None if sig is None else 100 * sig,
                        covered_95=None if sig is None else bool(abs(np.log(1 + m["scale_error_pct"] / 100))
                                                                 <= 1.96 * sig)))
    return out


def evaluate_against_lidar(scene: dict, info: dict, capture_dir, lidar_scene: dict) -> dict:
    """Trajectory, scale and cloud-to-cloud metrics of a video-tier scene against the LiDAR scene of the same capture.

    Returns the metrics and the 4x4 transforms that map video-aligned points into the LiDAR-aligned frame
    (rigid: what a user would get; similarity: shape only)."""
    kf = scene["kf"]
    ref = arkit_reference(capture_dir, scene["timestamps"], kf, info["rotation"]["final_rotation"])
    traj = trajectory_metrics(scene["T_wc"][kf], ref["T_wc"])
    out = dict(trajectory=traj, transforms={}, windows=window_scale_errors(scene["T_wc"][kf], ref["T_wc"]))
    if "kf_segment" in scene:
        out["segments"] = segment_errors(scene["T_wc"][kf], ref["T_wc"], np.asarray(scene["kf_segment"]), info)
    if "kf_trusted" in scene and np.asarray(scene["kf_trusted"]).sum() >= 10:
        t = np.asarray(scene["kf_trusted"]).astype(bool)
        out["trajectory_trusted"] = trajectory_metrics(scene["T_wc"][kf][t], ref["T_wc"][t])
    inv_al = np.linalg.inv(scene["T_align"])
    for tag in ("se3", "sim3"):
        s, R, t = traj[tag]["scale"], np.array(traj[tag]["R"]), np.array(traj[tag]["t"])
        S = np.eye(4)
        S[:3, :3], S[:3, 3] = s * R, t
        M = lidar_scene["T_align"] @ S @ inv_al                # video aligned -> LiDAR aligned
        out["transforms"][tag] = M.tolist()
        ref_pts = lidar_scene["points"]
        for key in ("points", "raw_points"):
            P = scene[key] @ M[:3, :3].T + M[:3, 3]
            out[f"c2c_{key}_{tag}"] = cloud_to_cloud(P, ref_pts)
    return out
