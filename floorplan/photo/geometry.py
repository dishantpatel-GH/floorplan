"""Small geometric tools of the photo tier: gravity from cameras + floor, normals, gravity-constrained registration.

Why gravity first: once every room's frame is levelled (+y up), the unknown placement of one room relative to another
has only 4 degrees of freedom (a yaw about "up" and a 3-D shift) instead of 6. Fewer unknowns means a cross-room link
can be estimated from fewer and noisier correspondences, and it can never tilt a room.
"""
from __future__ import annotations

import numpy as np


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b (Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def up_from_cameras(T_wc: np.ndarray) -> np.ndarray:
    """First guess of 'up' from how people hold phones.

    A photo taken level-ish has its image x axis horizontal (little roll), so 'up' is the direction most orthogonal
    to all camera x axes (smallest eigenvector of sum x x^T). That needs cameras facing different ways (true for
    corner-to-corner shots); with one camera, or nearly parallel ones, we use -y (image up) averaged instead."""
    X = T_wc[:, :3, 0]
    Ym = -T_wc[:, :3, 1].mean(0)
    w, V = np.linalg.eigh(X.T @ X)
    up = V[:, 0] if len(T_wc) >= 2 and w[1] > 0.05 * len(T_wc) else Ym
    up = up * np.sign(np.dot(up, Ym) or 1.0)
    return up / np.linalg.norm(up)


def estimate_normals(P: np.ndarray, radius: float = 0.08, max_nn: int = 30) -> np.ndarray:
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pc.normals)


def orient_normals(P: np.ndarray, N: np.ndarray, view_centre: np.ndarray) -> np.ndarray:
    """Flip normals to face the camera that saw the point (the LiDAR TSDF convention: floors face up)."""
    s = np.sign(np.einsum("ij,ij->i", N, view_centre - P))
    s[s == 0] = 1
    return N * s[:, None]


def refine_up_with_floor(P: np.ndarray, N: np.ndarray, up0: np.ndarray, cam_h: float,
                         max_dev_deg: float = 12.0) -> tuple[np.ndarray, float | None, int]:
    """Refine 'up' with the floor plane: up-facing points clearly below the cameras, plane fit by SVD (with RANSAC
    against furniture tops). Returns (up, floor height along up, number of floor points)."""
    h = P @ up0
    sel = (N @ up0 > np.cos(np.radians(max_dev_deg))) & (h < cam_h - 0.6)
    F = P[sel]
    if len(F) < 300:
        return up0, None, int(len(F))
    hf = F @ up0
    # the floor is the lowest strong level: take the 5th-percentile height band, then plane-fit it
    lo = np.percentile(hf, 5)
    band = F[np.abs(hf - np.median(hf[hf < lo + 0.15])) < 0.06]
    c = band.mean(0)
    _, _, vt = np.linalg.svd(band - c, full_matrices=False)
    n = vt[2] * np.sign(np.dot(vt[2], up0))
    if np.degrees(np.arccos(np.clip(np.dot(n, up0), -1, 1))) > max_dev_deg:
        return up0, float(np.median(band @ up0)), int(len(band))
    return n, float(c @ n), int(len(band))


def yaw_rigid_ransac(A: np.ndarray, B: np.ndarray, thr: np.ndarray, iters: int = 2000, seed: int = 0):
    """4-DoF (yaw about +y, translation) transform mapping B onto A, robust to outliers.

    A, B: (k,3) corresponding points in two LEVELLED frames. thr: (k,) inlier distance per correspondence (grows
    with depth, since learned depth errors are relative). Returns (T 4x4, inlier mask)."""
    rng = np.random.default_rng(seed)
    k = len(A)
    best, best_in = None, np.zeros(k, bool)
    if k < 3:
        return None, best_in
    for _ in range(iters):
        i, j = rng.choice(k, 2, replace=False)
        T = _yaw_fit(A[[i, j]], B[[i, j]])
        if T is None:
            continue
        r = np.linalg.norm(B @ T[:3, :3].T + T[:3, 3] - A, axis=1)
        inl = r < thr
        if inl.sum() > best_in.sum():
            best, best_in = T, inl
    if best is None or best_in.sum() < 3:
        return None, best_in
    for _ in range(3):                                  # refit on inliers, re-score
        best = _yaw_fit(A[best_in], B[best_in])
        r = np.linalg.norm(B @ best[:3, :3].T + best[:3, 3] - A, axis=1)
        best_in = r < thr
    return best, best_in


def _yaw_fit(A: np.ndarray, B: np.ndarray) -> np.ndarray | None:
    """Least-squares yaw + translation (closed form: 2-D Procrustes in x-z, mean offset in y)."""
    a, b = A[:, [0, 2]], B[:, [0, 2]]
    ca, cb = a.mean(0), b.mean(0)
    H = (b - cb).T @ (a - ca)
    if np.linalg.norm(H) < 1e-9:
        return None
    U, _, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1] *= -1
        R2 = Vt.T @ U.T
    t2 = ca - R2 @ cb
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1]
    T[0, 3], T[2, 3] = t2
    T[1, 3] = float(np.mean(A[:, 1] - B[:, 1]))
    return T


def similarity_scale(A: np.ndarray, B: np.ndarray) -> float:
    """Scale of the best similarity B -> A (Umeyama); used as a diagnostic of cross-room scale consistency."""
    a, b = A - A.mean(0), B - B.mean(0)
    return float(np.sqrt((a ** 2).sum() / max((b ** 2).sum(), 1e-12)))


def manhattan_yaw(N: np.ndarray, min_pts: int = 100) -> float:
    """Dominant wall direction modulo 90 deg from wall normals (the 4-theta circular mean, as in plan/align.py)."""
    w = np.abs(N[:, 1]) < 0.2
    th = np.arctan2(N[w, 2], N[w, 0])
    if len(th) < min_pts:
        return 0.0
    return float(np.angle(np.mean(np.exp(4j * th))) / 4)


def yaw_matrix(yaw: float) -> np.ndarray:
    """Rotation about +y by -yaw: maps a wall normal at angle yaw (x-z plane) onto +x (same as plan/align.py)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
