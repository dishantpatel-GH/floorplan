"""Gravity and Manhattan alignment (D-009), plus global floor and ceiling levels.

Why align at all:
  * Gravity: heights (ceiling height, sill height) are measured along "up". ARKit's gravity estimate is good to a
    fraction of a degree, but the floor plane is a direct measurement of "level". A 0.5 deg tilt across a 5 m room
    shifts heights by 4 cm at the far wall, which is larger than the 1.5 cm ceiling-height gate.
  * Manhattan yaw: most interior walls meet at right angles. Rotating the scene so walls run along x and z makes
    wall fitting a 1-D problem per wall (fit an offset, not an angle). That is more robust, and it is how
    magicplan / poly.cam plans look. Walls that are not axis-aligned are kept and fitted at their own angle.
"""
from __future__ import annotations

import numpy as np


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b (Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def _level_peaks(y: np.ndarray, bin_m: float = 0.01):
    """Histogram of heights -> (bin centres, counts) smoothed over 3 bins."""
    if len(y) == 0:
        return np.array([]), np.array([])
    edges = np.arange(y.min() - 2 * bin_m, y.max() + 2 * bin_m, bin_m)
    h, e = np.histogram(y, bins=edges)
    h = np.convolve(h, np.ones(3) / 3, mode="same")
    return (e[:-1] + e[1:]) / 2, h


def estimate_floor(points, normals, cam_heights, min_frac=0.02):
    """Floor = the lowest strong peak among UP-facing horizontal surfaces below the camera.

    TSDF normals point toward the observed side, so floors face up (+y) and ceilings face down (-y). A table top
    also faces up, but it is higher and smaller, so we take the lowest peak holding at least min_frac of
    up-facing points."""
    up = normals[:, 1] > 0.95
    y = points[up, 1]
    ch = np.asarray(cam_heights, float)
    ch = ch[np.isfinite(ch)]
    y = y[np.isfinite(y)]                               # D-058: a degenerate video segment can put NaN/inf points in
    if not len(y) or not len(ch):
        return None
    y = y[y < np.median(ch) - 0.5]
    c, h = _level_peaks(y)
    if len(h) == 0 or not np.isfinite(h).all():
        return None
    strong = np.where(h >= max(min_frac * h.sum(), 0.3 * h.max()))[0]
    return float(c[strong.min()]) if len(strong) else None


def estimate_ceiling(points, normals, floor_y, min_height=1.9, min_frac=0.02):
    down = normals[:, 1] < -0.95
    y = points[down, 1]
    y = y[np.isfinite(y)]
    y = y[y > floor_y + min_height]
    if len(y) < 200:
        return None
    c, h = _level_peaks(y)
    strong = np.where(h >= max(min_frac * h.sum(), 0.3 * h.max()))[0]
    return float(c[strong.max()]) if len(strong) else None


def gravity_correction(points, normals, floor_y, max_tilt_deg):
    """Rotation that makes the fitted floor plane exactly horizontal (if its tilt is plausible)."""
    sel = (normals[:, 1] > 0.95) & (np.abs(points[:, 1] - floor_y) < 0.04)
    P = points[sel]
    if len(P) < 500:
        return np.eye(3), 0.0
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2] * np.sign(vt[2][1])
    tilt = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    if tilt > max_tilt_deg:
        return np.eye(3), tilt
    return _rot_between(n, np.array([0.0, 1.0, 0.0])), tilt


def manhattan_yaw(normals, tol_deg=5.0):
    """Dominant wall direction modulo 90 deg, estimated from wall normals (|n_y| < 0.2).

    The 4*theta trick: angles that differ by 90 deg become identical after multiplying by 4, so the circular mean
    of exp(4i*theta) gives the shared Manhattan orientation."""
    w = np.abs(normals[:, 1]) < 0.2
    th = np.arctan2(normals[w, 2], normals[w, 0])
    if len(th) < 100:
        return 0.0, 0.0
    hist, edges = np.histogram(np.mod(th, np.pi / 2), bins=180, range=(0, np.pi / 2))
    hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7) / 7, mode="same")[3:-3]
    peak = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    d = np.angle(np.exp(4j * (th - peak))) / 4
    near = np.abs(d) < np.radians(tol_deg)
    yaw = peak + np.angle(np.mean(np.exp(4j * d[near]))) / 4
    score = float(near.mean())   # fraction of wall points within tol of a Manhattan direction
    return float(yaw), score


def align_scene(points, normals, cam_positions, cfg):
    """Return T_align (4x4) mapping capture-world -> aligned world, plus diagnostics."""
    floor0 = estimate_floor(points, normals, cam_positions[:, 1])
    R_g, tilt = gravity_correction(points, normals, floor0, cfg.max_floor_tilt_deg)
    n_g = normals @ R_g.T
    yaw, score = manhattan_yaw(n_g, cfg.manhattan_tol_deg)
    c, s = np.cos(-yaw), np.sin(-yaw)
    # rotate about y by -yaw: maps a wall normal at angle yaw (in the x-z plane) onto the +x axis
    R_y = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    R = R_y @ R_g
    T = np.eye(4)
    T[:3, :3] = R
    P = points @ R.T
    N = normals @ R.T
    floor_y = estimate_floor(P, N, (cam_positions @ R.T)[:, 1])
    ceil_y = estimate_ceiling(P, N, floor_y)
    info = dict(floor_tilt_before_deg=tilt, manhattan_yaw_deg=float(np.degrees(yaw)), manhattan_score=score,
                floor_y=floor_y, ceiling_y=ceil_y,
                ceiling_height_global=(ceil_y - floor_y) if ceil_y is not None else None)
    return T, P, N, info
