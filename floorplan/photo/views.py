"""Per-photo metric geometry in a LEVELLED camera frame: MoGe-2 depth + gravity (GeoCalib, refined by the floor).

Each photo becomes a small metric point cloud whose +y axis is "up" and whose yaw is the camera's heading. Only the
yaw, the position and a scale correction remain unknown per photo; link.py estimates them from feature matches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floorplan.photo.geometry import estimate_normals, orient_normals, refine_up_with_floor


@dataclass
class PhotoView:
    name: str
    room: str
    rgb: np.ndarray         # (h,w,3) uint8 at the depth resolution
    K: np.ndarray           # (3,3) intrinsics at the depth resolution
    depth: np.ndarray       # (h,w) metres (MoGe-2 scale), 0 = invalid
    R_lev: np.ndarray       # (3,3) camera -> levelled frame (+y up, camera heading along +z)
    up_source: str          # "floor" (refined by the floor plane) or "geocalib" or "camera_prior"
    floor_h: float | None   # floor height in the levelled frame (camera at 0), metres, if seen
    sfm_scale: float        # depth-res pixels per SfM-res pixel (to map feature matches)
    normal_cam: np.ndarray | None = None   # (h,w,3) MoGe-2 normals, camera frame, facing the camera

    def points_cam(self, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(k,3) camera-frame points of valid pixels and their flat pixel indices."""
        h, w = self.depth.shape
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        d = self.depth[v, u]
        ok = d > 0
        u, v, d = u[ok], v[ok], d[ok]
        x = (u - self.K[0, 2]) / self.K[0, 0] * d
        y = (v - self.K[1, 2]) / self.K[1, 1] * d
        return np.stack([x, y, d], 1), v * w + u

    def lift(self, xy_sfm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Levelled-frame 3-D points of feature locations given in SfM pixels; mask of valid ones."""
        uv = xy_sfm * self.sfm_scale
        h, w = self.depth.shape
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        d = self.depth[v, u]
        P = np.stack([(uv[:, 0] - self.K[0, 2]) / self.K[0, 0] * d, (uv[:, 1] - self.K[1, 2]) / self.K[1, 1] * d, d], 1)
        return P @ self.R_lev.T, d > 0


def levelling_rotation(up_cam: np.ndarray) -> np.ndarray:
    """Rotation camera -> levelled frame: y = up, z = camera forward projected on the horizontal, x = y cross z."""
    y = up_cam / np.linalg.norm(up_cam)
    f = np.array([0.0, 0.0, 1.0])
    z = f - np.dot(f, y) * y
    if np.linalg.norm(z) < 1e-6:                     # looking straight up/down: any horizontal heading
        z = np.array([1.0, 0, 0]) - y[0] * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def gravity_geocalib(rgbs: list[np.ndarray], focals: list[float], seed: int = 0) -> list[np.ndarray | None]:
    """Up vector in the OpenCV camera frame per photo from GeoCalib (learned single-image gravity), focal as prior."""
    try:
        import torch
        from geocalib import GeoCalib
    except ImportError:
        return [None] * len(rgbs)
    model = GeoCalib(weights="pinhole").to("cuda")
    out = []
    for rgb, f in zip(rgbs, focals):
        img = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1).cuda()
        torch.manual_seed(seed)                      # GeoCalib's head draws random bases on every call
        with torch.inference_mode():
            res = model.calibrate(img, priors={"focal": torch.tensor(float(f), device="cuda")})
        out.append(res["gravity"].vec3d[0].cpu().numpy().astype(float))
    del model
    torch.cuda.empty_cache()
    return out


def build_view(name, room, rgb, K, depth, normal_cam, up_cam, up_source, sfm_scale) -> PhotoView:
    """Levelled view; the floor plane (if seen) refines GeoCalib's up vector (a plane over metres beats a learned
    single-image estimate, the same reasoning as LiDAR decision D-009)."""
    R0 = levelling_rotation(up_cam)
    pv = PhotoView(name, room, rgb, K, depth, R0, up_source, None, sfm_scale)
    Pall, idx_all = pv.points_cam(stride=1)
    if normal_cam is None:
        normal_cam = np.zeros(depth.shape + (3,))
        normal_cam.reshape(-1, 3)[idx_all] = estimate_normals(Pall)
    flat = normal_cam.reshape(-1, 3).copy()
    flat[idx_all] = orient_normals(Pall, flat[idx_all], np.zeros(3))
    pv.normal_cam = flat.reshape(depth.shape + (3,))
    Pc, idx = pv.points_cam(stride=2)
    if len(Pc) > 500:
        Nc = pv.normal_cam.reshape(-1, 3)[idx]
        up, floor_h, n = refine_up_with_floor(Pc, Nc, up_cam / np.linalg.norm(up_cam), cam_h=0.0)
        if floor_h is not None and n >= 300:
            pv.R_lev = levelling_rotation(up)
            pv.floor_h = float(floor_h)
            pv.up_source = "floor"
    return pv
