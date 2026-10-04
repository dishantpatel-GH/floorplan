#!/usr/bin/env python
"""Absolute-accuracy check of the LiDAR tier against a laser scanner (ARKitScenes iPad capture + Faro scan).

Why: the sample captures have no ground truth, so they can only show repeatability, not accuracy. ARKitScenes
ships an iPad Pro LiDAR capture AND a survey-grade Faro Focus S70 scan of the same room. We run the SAME stage-1 code
as floorplan/pipeline/scene.py (keyframes -> TSDF -> raw points -> gravity/Manhattan alignment) on it through the
ArkitScenesCapture adapter, register the laser scan into our frame, and measure:

  1. surface accuracy: cloud-to-cloud distances (point-to-point and point-to-plane) from our TSDF surface and from
     our raw LiDAR points to the laser surface (median / p90 / p95), plus laser coverage of our surface;
  2. what the floor plan actually reports: distances between facing planar surfaces (wall to wall, floor to
     ceiling), measured with the SAME robust plane fit on our raw points, on our TSDF surface and on the laser.
     Surfaces are found in our cloud; each one names the laser surface to compare with, whose position is then fitted
     from laser points only, on the stretch both sensors see. The "room box" (width x, depth z, height around the
     camera path) is the headline; all facing pairs give the error distribution;
  3. where errors come from: raw-point error by range, by incidence angle and by distance to a room corner;
  4. ablations: pose interpolation (10 Hz poses vs 60 Hz depth), the depth-confidence threshold (D-006) and an edge
     margin that leaves the rim of each surface out of the plane fit.

Registration (the dataset ships no laser -> ARKit transform): both clouds are levelled and Manhattan-aligned by the
pipeline's own align_scene, which leaves only a 90-degree yaw ambiguity and a translation. We try the 4 yaws, find
the horizontal shift by FFT cross-correlation of wall slices, take the vertical shift from the floor levels, and
refine with point-to-plane ICP. Dimensions are wall-to-wall distances, so they do not depend on the translation and
depend on rotation only as cos(error) (1e-5 for 0.25 deg): they are the registration-independent part of the check.

Usage:
  PYTHONPATH=. python scripts/validate_arkitscenes.py \
      --scene  <ARKitScenes>/raw/Training/47895909 \
      --laser  <ARKitScenes>/laser_scanner_point_clouds/482587/191738.ply \
      [--out outputs/arkitscenes_validation] [--no-ablations] [--reference-transform T_laser_to_arkit.txt]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.config import Config  # noqa: E402
from floorplan.io.arkitscenes import Z_UP_TO_Y_UP, load_arkitscenes  # noqa: E402
from floorplan.plan.align import align_scene, estimate_ceiling, estimate_floor  # noqa: E402
from floorplan.recon.fusion import collect_raw_points, fuse_tsdf  # noqa: E402
from floorplan.recon.keyframes import select_keyframes  # noqa: E402

# --- laser preparation ---
LASER_STRIDE = 10            # read every 10th of the ~41 M Faro points (still < 5 mm spacing on nearby walls)
LASER_RADIUS_M = 8.0         # keep laser points within 8 m of a scanner position (drops far-away rooms)
EVAL_VOXEL_M = 0.005         # laser reference density for distance queries (5 mm, same as our raw points)
REG_VOXEL_M = 0.02           # clouds used for coarse alignment and ICP (2 cm, same as our TSDF voxel)
NORMAL_RADIUS_M, NORMAL_NN = 0.05, 30   # PCA normal neighbourhood
# --- registration ---
SLICE_BAND_M = (1.0, 1.5)    # wall slice used for the 2D cross-correlation (above most furniture)
FFT_RES_M = 0.02
ICP_SCHEDULE_M = (0.10, 0.05, 0.02)   # coarse-to-fine max correspondence distance
# --- metrics ---
OVERLAP_M = 0.10             # a point counts as "seen by both" if the laser has a point within 10 cm
COMPLETE_M = 0.05            # laser point "covered" by our surface if we have a point within 5 cm
RANGE_BINS_M = (0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
INCIDENCE_BINS_DEG = (0, 15, 30, 45, 60, 75, 90)   # angle between the viewing ray and the surface normal
# --- dimension measurement (robust plane fitting, same code for our clouds and for the laser) ---
FAMILIES = {"+x": (0, 1), "-x": (0, -1), "+z": (2, 1), "-z": (2, -1), "floor": (1, 1), "ceiling": (1, -1)}
NORMAL_COS = 0.9             # a point belongs to a family if its normal is within 25 deg of the family direction
PEAK_BIN_M = 0.01            # 1 cm histogram bins along the family axis
PEAK_NMS_M = 0.05            # two parallel surfaces closer than 5 cm are one surface (tile vs plaster, noise)
SEED_WINDOW_M = 0.03         # points within 3 cm of a peak / plane seed the trimmed fit
MATCH_WINDOW_M = 0.10        # the laser surface must lie within 10 cm of ours to be "the same surface"
INLIER_M = 0.015             # support of a fitted plane: points within 1.5 cm
CELL_M = 0.05                # 5 cm in-plane grid, used to compare the SAME stretch of surface in both clouds
MIN_AREA_M2 = 0.15           # measure only surfaces covering >= 0.15 m2 (e.g. 30 x 50 cm)
MIN_COMMON_M2 = 0.05         # ... of which >= 0.05 m2 is seen by both sensors
MIN_FACING_GAP_M = 0.3       # interior distances: two facing surfaces at least 30 cm apart
FULL_HEIGHT_M = 2.0          # a vertical surface is "full height" (wall, tall cabinet) if it reaches 2.0 m
MAX_SURFACES_PER_FAMILY = 40
EDGE_MARGIN_M = 0.0          # rim of each surface left out of the comparison; 0 chosen by the ablation (see docs)
EDGE_MARGINS_TESTED_M = (0.0, 0.05, 0.10, 0.20)
CORNER_BINS_M = (0.0, 0.05, 0.10, 0.20, 0.40, 1.0)   # distance from a room-box point to the nearest OTHER box plane
CORNER_EXCLUSIONS_M = (0.0, 0.2, 0.3, 0.4)   # room box re-measured without points this close to another box plane
TRIM_SCHEDULE_M = (0.03, 0.02, 0.015, 0.01)   # iterative trimmed least squares
CENTRE_TOL_M = 0.2           # a room-box wall must span the room centre (+-20 cm) along the wall

# colours (validated categorical slots 1-3 of the reference palette)
C_OURS, C_LASER, C_TSDF = "#2a78d6", "#eb6834", "#1baf7a"


# ---------------------------------------------------------------------------------------------------- utilities
def voxel_subsample(P: np.ndarray, size: float) -> np.ndarray:
    """Indices of one point per voxel (the first in input order). Deterministic, unlike a hash-map downsample."""
    k = np.floor((P - P.min(0)) / size).astype(np.int64)
    key = k[:, 0] + (k[:, 1] << 21) + (k[:, 2] << 42)
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def pca_normals(P: np.ndarray) -> np.ndarray:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=NORMAL_NN))
    return np.asarray(pc.normals)


def orient_towards(N: np.ndarray, P: np.ndarray, viewpoints: np.ndarray) -> np.ndarray:
    """Flip normals to face the nearest viewpoint (scanner): same 'faces the observed side' rule as TSDF normals."""
    vp = viewpoints[cKDTree(viewpoints).query(P)[1]]
    flip = np.einsum("ij,ij->i", N, vp - P) < 0
    N = N.copy()
    N[flip] *= -1
    return N


def transform(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def rot_y(k: int) -> np.ndarray:
    """4x4 rotation by k*90 deg about +y (the up axis)."""
    c, s = round(math.cos(k * math.pi / 2)), round(math.sin(k * math.pi / 2))
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return T


def rigid_delta(T: np.ndarray) -> tuple[float, float]:
    """(rotation deg, translation m) of a 4x4 rigid transform."""
    ang = math.degrees(math.acos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    return ang, float(np.linalg.norm(T[:3, 3]))


def stats_cm(d: np.ndarray) -> dict:
    d = np.abs(d)
    if len(d) == 0:
        return dict(n=0)
    q = np.percentile(d, [50, 90, 95]) * 100
    return dict(n=int(len(d)), median_cm=round(float(q[0]), 3), p90_cm=round(float(q[1]), 3),
                p95_cm=round(float(q[2]), 3), mean_cm=round(float(d.mean() * 100), 3))


# ---------------------------------------------------------------------------------------------------- stage 1
def build_stage1(cap, cfg: Config) -> tuple[dict, dict]:
    """The same four steps as floorplan.pipeline.scene.build_scene (which only accepts a Stray folder)."""
    kf = select_keyframes(cap, cfg)
    fused = fuse_tsdf(cap, kf, cap.T_wc, cfg, with_color=False)   # colour does not affect geometry
    raw = collect_raw_points(cap, kf, cap.T_wc, cfg)
    T_align, P, N, info = align_scene(fused["points"], fused["normals"], cap.T_wc[:, :3, 3], cfg)
    R = T_align[:3, :3]
    raw_P = transform(T_align, raw["points"].astype(np.float64))
    raw_ray = raw["ray"].astype(np.float64) @ R.T
    raw_N = pca_normals(raw_P)
    raw_N[np.einsum("ij,ij->i", raw_N, raw_ray) > 0] *= -1          # face the camera that saw the point
    order = np.lexsort(np.round(P, 4).T[::-1])                       # TSDF extraction order is thread-dependent
    scene = dict(points=P[order].astype(np.float64), normals=N[order].astype(np.float64),
                 raw_points=raw_P, raw_normals=raw_N, raw_range=raw["range"].astype(np.float64), raw_ray=raw_ray,
                 traj=transform(T_align, cap.T_wc[:, :3, 3]), T_align=T_align)
    info.update(frames=cap.n, keyframes=int(len(kf)), frames_fused=int(fused["frames_used"]),
                surface_points=int(len(P)), raw_points=int(len(raw_P)))
    return scene, info


# ---------------------------------------------------------------------------------------------------- laser
def read_ply_xyz(path: Path, stride: int) -> tuple[np.ndarray, int]:
    """Memory-mapped reader for big binary little-endian vertex-only PLYs (Faro scans are ~1.8 GB)."""
    types = {"double": "<f8", "float": "<f4", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4",
             "short": "<i2", "ushort": "<u2", "char": "i1", "float32": "<f4", "float64": "<f8"}
    props, nv, in_vertex = [], 0, False
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format") and "binary_little_endian" not in line:
                raise ValueError(f"{path}: unsupported PLY format {line!r}")
            if line.startswith("element"):
                _, name, cnt = line.split()
                in_vertex = name == "vertex"
                nv = int(cnt) if in_vertex else nv
            elif line.startswith("property") and in_vertex:
                _, typ, name = line.split()
                props.append((name, types[typ]))
            elif line == "end_header":
                offset = f.tell()
                break
    mm = np.memmap(path, dtype=np.dtype(props), mode="r", offset=offset, shape=(nv,))
    sub = np.array(mm[::stride])
    return np.stack([sub["x"], sub["y"], sub["z"]], 1).astype(np.float64), nv


def scanner_position(ply: Path) -> np.ndarray:
    """Scanner centre from `<scan>_pose.txt` (4x4 row-vector style: row 3 = translation). The PLY is already in
    this registered frame (README), so the pose is used ONLY to know where the scanner stood."""
    M = np.loadtxt(ply.with_name(f"{ply.stem}_pose.txt"), delimiter=",")
    return M[3, :3]


def load_laser(plys: list[Path]) -> dict:
    """Laser scan(s) in the pipeline's +y-up convention, cropped around the scanners, at two densities with normals."""
    parts, scanners, n_total = [], [], 0
    for p in plys:
        xyz, n = read_ply_xyz(p, LASER_STRIDE)
        parts.append(xyz)
        scanners.append(scanner_position(p))
        n_total += n
    C = Z_UP_TO_Y_UP[:3, :3]
    P = np.vstack(parts) @ C.T
    S = np.asarray(scanners) @ C.T
    P = P[cKDTree(S).query(P)[0] < LASER_RADIUS_M]
    out = dict(scanners=S, n_points_file=n_total, n_points_read=int(sum(len(x) for x in parts)))
    for name, vox in (("eval", EVAL_VOXEL_M), ("reg", REG_VOXEL_M)):
        Q = P[voxel_subsample(P, vox)]
        out[name] = (Q, orient_towards(pca_normals(Q), Q, S))
    nn = cKDTree(out["eval"][0]).query(out["eval"][0], k=2)[0][:, 1]
    out["eval_spacing_median_mm"] = round(float(np.median(nn)) * 1000, 2)
    return out


# ---------------------------------------------------------------------------------------------------- registration
def wall_slice_xz(P: np.ndarray, N: np.ndarray, floor_y: float) -> np.ndarray:
    m = (np.abs(N[:, 1]) < 0.3) & (P[:, 1] > floor_y + SLICE_BAND_M[0]) & (P[:, 1] < floor_y + SLICE_BAND_M[1])
    return P[m][:, [0, 2]]


def fft_shift(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """2D translation t maximising the overlap of rasterised point sets A and B + t, and its normalised score."""
    lo = np.minimum(A.min(0), B.min(0)) - 0.1
    hi = np.maximum(A.max(0), B.max(0)) + 0.1
    size = np.ceil((hi - lo) / FFT_RES_M).astype(int) + 1
    shape = (2 * size[1], 2 * size[0])

    def raster(X):
        g = np.zeros(shape, np.float32)
        ij = np.floor((X - lo) / FFT_RES_M).astype(int)
        g[ij[:, 1], ij[:, 0]] = 1.0
        return cv2.GaussianBlur(g, (0, 0), 1.5)

    GA, GB = raster(A), raster(B)
    corr = np.fft.irfft2(np.fft.rfft2(GA) * np.conj(np.fft.rfft2(GB)), s=shape)
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    iy = iy - shape[0] if iy > shape[0] // 2 else iy
    ix = ix - shape[1] if ix > shape[1] // 2 else ix
    score = float(corr.max() / math.sqrt((GA ** 2).sum() * (GB ** 2).sum()))
    return np.array([ix, iy], float) * FFT_RES_M, score


def register_laser(scene: dict, info: dict, laser: dict, cfg: Config) -> tuple[np.ndarray, dict]:
    """T (4x4) mapping laser (+y-up) -> our aligned frame, with diagnostics."""
    LP, LN = laser["reg"]
    T_aL, LPa, LNa, linfo = align_scene(LP, LN, laser["scanners"], cfg)            # level + Manhattan the laser
    ours_slice = wall_slice_xz(scene["points"], scene["normals"], info["floor_y"])
    cands = []
    for k in range(4):                                                              # 90-degree yaw ambiguity
        R = rot_y(k)
        B = wall_slice_xz(transform(R, LPa), LNa @ R[:3, :3].T, linfo["floor_y"])
        t_xz, score = fft_shift(ours_slice, B)
        T = R.copy()
        T[:3, 3] = [t_xz[0], info["floor_y"] - linfo["floor_y"], t_xz[1]]
        cands.append((score, k, T))
    score, k_best, T_coarse = max(cands, key=lambda c: c[0])
    T0 = T_coarse @ T_aL                                                            # laser -> ours (coarse)

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene["points"][voxel_subsample(scene["points"], REG_VOXEL_M)]))
    Lc, Ln = transform(T0, LP), LN @ T0[:3, :3].T
    lo, hi = scene["points"].min(0) - 0.3, scene["points"].max(0) + 0.3
    inside = np.all((Lc > lo) & (Lc < hi), axis=1)
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Lc[inside]))
    tgt.normals = o3d.utility.Vector3dVector(Ln[inside])
    T_icp, reg = np.eye(4), None
    for dist in ICP_SCHEDULE_M:                                                     # source = ours, target = laser
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, dist, T_icp, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T_icp = reg.transformation
    T = np.linalg.inv(T_icp) @ T0
    rot, trans = rigid_delta(T_icp)
    diag = dict(laser_floor_tilt_deg=round(linfo["floor_tilt_before_deg"], 3),
                yaw_scores={int(90 * c[1]): round(c[0], 3) for c in cands}, yaw_chosen_deg=90 * k_best,
                fft_score=round(score, 3), icp_fitness=round(reg.fitness, 4),
                icp_inlier_rmse_cm=round(reg.inlier_rmse * 100, 3),
                icp_correction_deg=round(rot, 3), icp_correction_cm=round(trans * 100, 2))
    return T, diag


# ---------------------------------------------------------------------------------------------------- metrics
def signed_distances(Q: np.ndarray, ref_P: np.ndarray, ref_N: np.ndarray, tree: cKDTree):
    """Point-to-point distance to the nearest reference point, and signed point-to-plane distance along the
    reference normal (+ = on the side the reference was observed from, i.e. in front of the true surface)."""
    d_pp, j = tree.query(Q, workers=-1)
    d_pl = np.einsum("ij,ij->i", Q - ref_P[j], ref_N[j])
    return d_pp, d_pl, j


def surface_metrics(Q: np.ndarray, ref_P: np.ndarray, ref_N: np.ndarray, tree: cKDTree):
    """Accuracy of cloud Q against the laser; returns (summary, d_pp, d_pl, nearest laser index)."""
    d_pp, d_pl, j = signed_distances(Q, ref_P, ref_N, tree)
    ov = d_pp < OVERLAP_M
    m = dict(points=int(len(Q)), overlap_frac=round(float(ov.mean()), 4),
             point_to_point_all=stats_cm(d_pp), point_to_point_overlap=stats_cm(d_pp[ov]),
             point_to_plane_overlap=stats_cm(d_pl[ov]),
             signed_point_to_plane_median_cm=round(float(np.median(d_pl[ov])) * 100, 3),
             outliers_over_10cm_frac=round(float((~ov).mean()), 4))
    return m, d_pp, d_pl, j


def error_by_corner_distance(P: np.ndarray, d_pl: np.ndarray, overlap: np.ndarray, box: dict,
                             centre: np.ndarray) -> list[dict]:
    """Signed error of raw points on the room-box surfaces, by distance to the nearest OTHER box surface (i.e. to the
    wall-wall / wall-ceiling / wall-floor corner). Shows whether depth is pulled inward near concave corners."""
    axes = {"width_x": 0, "width_z": 2, "height": 1}
    pairs = [(axes[name], a["ours"].plane, b["ours"].plane) for name, (a, b) in box.items()]
    if not pairs:
        return []
    inside = np.ones(len(P), bool)
    for axis, a, b in pairs:
        inside &= (P[:, axis] > a.coord_at(axis, centre) - 0.05) & (P[:, axis] < b.coord_at(axis, centre) + 0.05)
    planes = [pl for _, a, b in pairs for pl in (a, b)]
    D = np.abs(np.stack([P @ np.asarray(pl.n) - pl.d for pl in planes], 1))
    own = np.argmin(D, 1)
    on = inside & (D[np.arange(len(P)), own] < 0.05)
    D[np.arange(len(P)), own] = np.inf
    return error_by_bins(d_pl, D.min(1), overlap & on, CORNER_BINS_M, "corner_distance_m")


def beyond_room_box(P: np.ndarray, overlap: np.ndarray, box: dict, centre: np.ndarray) -> dict:
    """Raw points more than 10 cm behind a room-box wall, floor or ceiling. Mirrors and glass create such 'ghost'
    points (a reflection looks like a room behind the wall); real openings (a window, a vent) do too, but then the
    laser sees the same geometry. So: how many are there, and how many does the laser confirm?"""
    axes = {"width_x": 0, "width_z": 2, "height": 1}
    out = np.zeros(len(P), bool)
    for name, (a, b) in box.items():
        k = axes[name]
        out |= (P[:, k] < a["ours"].plane.coord_at(k, centre) - 0.10) | (P[:, k] > b["ours"].plane.coord_at(k, centre) + 0.10)
    return dict(points=int(out.sum()), frac_of_raw=round(float(out.mean()), 5),
                laser_confirmed_frac=round(float(overlap[out].mean()), 3) if out.any() else None)


def completeness(laser_P: np.ndarray, ours: np.ndarray) -> dict:
    """Fraction of laser points inside our capture's box that have one of our points within COMPLETE_M."""
    lo, hi = ours.min(0), ours.max(0)
    L = laser_P[np.all((laser_P > lo) & (laser_P < hi), axis=1)]
    d = cKDTree(ours).query(L, workers=-1)[0]
    return dict(laser_points_in_box=int(len(L)), covered_frac=round(float((d < COMPLETE_M).mean()), 4))


def error_by_bins(d_pl: np.ndarray, value: np.ndarray, overlap: np.ndarray, bins: tuple, key: str) -> list[dict]:
    """|error| statistics and signed median per bin of `value` (range or incidence angle); feeds the noise model."""
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = overlap & (value >= a) & (value < b)
        if m.sum() >= 100:
            out.append({key: [a, b], **stats_cm(d_pl[m]),
                        "signed_median_cm": round(float(np.median(d_pl[m])) * 100, 3)})
    return out



# ---------------------------------------------------------------------------------------------------- dimensions
@dataclass
class Plane:
    n: list               # unit normal (3,), pointing to the observed side ("into the room")
    d: float              # n . p = d
    inliers: int
    rmse_mm: float
    offset_se_mm: float   # statistical standard error of the offset (ignores correlated / systematic errors)

    def coord_at(self, axis: int, p: np.ndarray) -> float:
        """Coordinate along `axis` where the plane crosses the line through p parallel to that axis."""
        n = np.asarray(self.n)
        rest = sum(n[a] * p[a] for a in range(3) if a != axis)
        return float((self.d - rest) / n[axis])


@dataclass
class Surface:
    """One planar surface found in a cloud: its plane, where it is (5 cm in-plane cells) and its extent."""
    family: str
    plane: Plane
    cells: np.ndarray     # sorted unique int64 keys of occupied in-plane cells
    lo: np.ndarray        # bounding box of the support points
    hi: np.ndarray

    @property
    def axis(self) -> int:
        return FAMILIES[self.family][0]

    @property
    def area_m2(self) -> float:
        return len(self.cells) * CELL_M ** 2

    def summary(self) -> dict:
        return dict(family=self.family, area_m2=round(self.area_m2, 3), lo=np.round(self.lo, 3).tolist(),
                    hi=np.round(self.hi, 3).tolist(), plane=asdict(self.plane))


def fit_plane_trimmed(X: np.ndarray) -> Plane | None:
    """Least-squares plane, refitted on points within a shrinking distance (3 -> 1 cm). Robust to clutter and to
    the few mm of LiDAR 'fuzz' while still averaging thousands of points (a median-like but plane-shaped estimate)."""
    keep = np.ones(len(X), bool)
    for thr in TRIM_SCHEDULE_M:
        if keep.sum() < 50:
            return None
        mu = X[keep].mean(0)
        n = np.linalg.svd(X[keep] - mu, full_matrices=False)[2][2]
        keep = np.abs((X - mu) @ n) < thr
    if keep.sum() < 50:
        return None
    rmse = float(np.sqrt(np.mean(((X[keep] - mu) @ n) ** 2)))
    return Plane(n=n.tolist(), d=float(n @ mu), inliers=int(keep.sum()), rmse_mm=round(rmse * 1000, 3),
                 offset_se_mm=round(rmse / math.sqrt(keep.sum()) * 1000, 4))


def cell_keys(X: np.ndarray, axis: int) -> np.ndarray:
    """In-plane 5 cm cell id of each point (the two coordinates other than `axis`)."""
    a, b = [k for k in range(3) if k != axis]
    i = np.floor(X[:, a] / CELL_M).astype(np.int64) + (1 << 20)
    j = np.floor(X[:, b] / CELL_M).astype(np.int64) + (1 << 20)
    return (i << 21) | j


def histogram_peaks(c: np.ndarray) -> list[float]:
    """Peaks of a 1 cm histogram (3-bin smoothing), strongest first, at least PEAK_NMS_M apart."""
    if len(c) < 50:
        return []
    edges = np.arange(c.min() - PEAK_BIN_M, c.max() + 2 * PEAK_BIN_M, PEAK_BIN_M)
    h, e = np.histogram(c, edges)
    h = np.convolve(h, np.ones(3) / 3, mode="same")
    centres = (e[:-1] + e[1:]) / 2
    peaks: list[float] = []
    for i in np.argsort(-h, kind="stable"):
        if h[i] < 20:
            break
        if all(abs(centres[i] - p) >= PEAK_NMS_M for p in peaks):
            peaks.append(float(centres[i]))
    return peaks


@dataclass
class FamilyCloud:
    """Points of one cloud split by orientation family, plus how many points a 5 cm cell needs to count as 'seen'
    (20% of a fully observed cell at the cloud's resolution). Without the density rule, a sparse haze of noisy
    points 3-5 cm off a wall would look like a large second surface."""
    fam: dict
    min_cell_pts: int


def family_cloud(P: np.ndarray, N: np.ndarray, resolution_m: float) -> FamilyCloud:
    fam = {f: P[s * N[:, a] > NORMAL_COS] for f, (a, s) in FAMILIES.items()}
    return FamilyCloud(fam, max(1, int(0.2 * (CELL_M / resolution_m) ** 2)))


def make_surface(X: np.ndarray, family: str, min_cell_pts: int, cells_allowed: np.ndarray | None = None) -> Surface | None:
    """Trimmed plane fit on X (points of one family near one seed), oriented to the family, with its support."""
    axis, sign = FAMILIES[family]
    if cells_allowed is not None:
        X = X[np.isin(cell_keys(X, axis), cells_allowed)]
    plane = fit_plane_trimmed(X) if len(X) >= 50 else None
    if plane is None:
        return None
    if plane.n[axis] * sign < 0:
        plane.n, plane.d = [-v for v in plane.n], -plane.d
    S = X[np.abs(X @ np.asarray(plane.n) - plane.d) < INLIER_M]
    keys, counts = np.unique(cell_keys(S, axis), return_counts=True)
    cells = keys[counts >= min_cell_pts]
    if len(cells) == 0:
        return None
    S = S[np.isin(cell_keys(S, axis), cells)]
    return Surface(family, plane, cells, S.min(0), S.max(0))


def find_surfaces(cloud: FamilyCloud) -> list[Surface]:
    """Planar surfaces of each axis-aligned family covering >= MIN_AREA_M2 (walls, floor, ceiling, cabinet fronts).

    Sequential extraction, like sequential RANSAC but 1-D because align_scene has put walls on the x and z axes:
    take the strongest histogram peak, fit a plane, remove its points, repeat. Removing points stops one noisy wall
    from being reported several times."""
    out = []
    for family, X in cloud.fam.items():
        axis = FAMILIES[family][0]
        left = np.ones(len(X), bool)
        for _ in range(MAX_SURFACES_PER_FAMILY):
            peaks = histogram_peaks(X[left, axis])
            if not peaks:
                break
            seed = np.abs(X[:, axis] - peaks[0]) < SEED_WINDOW_M
            s = make_surface(X[left & seed], family, cloud.min_cell_pts)
            left &= ~seed
            if s is not None:
                left &= ~(np.abs(X @ np.asarray(s.plane.n) - s.plane.d) < SEED_WINDOW_M)
                if s.area_m2 >= MIN_AREA_M2:
                    out.append(s)
    return out


def erode_cells(cells: np.ndarray, margin_m: float) -> np.ndarray:
    """Cells whose whole neighbourhood within margin_m is also occupied: drops the rim of a surface patch."""
    r = int(round(margin_m / CELL_M))
    if r == 0:
        return cells
    i, j = cells >> 21, cells & ((1 << 21) - 1)
    keep = np.ones(len(cells), bool)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            keep &= np.isin(((i + di) << 21) | (j + dj), cells)
    return cells[keep]


def match_surface(s: Surface, ours: FamilyCloud, laser: FamilyCloud, edge_margin_m: float) -> dict | None:
    """The laser's version of surface s, compared on the stretch both sensors see.

    Our surface only says WHICH laser surface to look at (same family, within 10 cm, same cells); the laser plane
    position is then found from laser points alone (its own histogram peak), and both planes are refitted on the
    cells they have in common, minus a rim of edge_margin_m (iPad depth rounds concave corners inward), so a wall
    that one sensor sees only partly is compared like for like."""
    axis = s.axis
    L = laser.fam[s.family]
    near = np.abs(L @ np.asarray(s.plane.n) - s.plane.d) < MATCH_WINDOW_M
    near &= np.isin(cell_keys(L, axis), s.cells)
    peaks = histogram_peaks(L[near, axis])
    if not peaks:
        return None
    ls = make_surface(L[near & (np.abs(L[:, axis] - peaks[0]) < SEED_WINDOW_M)], s.family, laser.min_cell_pts)
    if ls is None:
        return None
    common = np.intersect1d(erode_cells(s.cells, edge_margin_m), ls.cells)   # erode OUR dense patch, not the
    #                                                                           patchy single-scan laser coverage
    if len(common) * CELL_M ** 2 < MIN_COMMON_M2:
        return None
    X = ours.fam[s.family]
    a = make_surface(X[np.abs(X @ np.asarray(s.plane.n) - s.plane.d) < SEED_WINDOW_M], s.family, 1, common)
    b = make_surface(L[np.abs(L @ np.asarray(ls.plane.n) - ls.plane.d) < SEED_WINDOW_M], s.family, 1, common)
    if a is None or b is None:
        return None
    mid = (a.lo + a.hi) / 2
    offset = a.plane.coord_at(axis, mid) - b.plane.coord_at(axis, mid)
    return dict(ours=a, laser=b, area_m2=round(s.area_m2, 3), common_area_m2=round(len(common) * CELL_M ** 2, 3),
                inward_offset_cm=round(offset * FAMILIES[s.family][1] * 100, 2))


def spans(s: Surface, axis: int, value: float, tol: float = 0.0) -> bool:
    return s.lo[axis] - tol <= value <= s.hi[axis] + tol


def pair_measure(a: dict, b: dict, axis: int, floor_y: float) -> dict:
    """Distance between facing matched surfaces a (low side) and b (high side), ours vs laser, measured on the line
    parallel to `axis` through the middle of the two surfaces: what a tape measure between them would read."""
    A, B = a["ours"], b["ours"]
    p = (A.lo + A.hi + B.lo + B.hi) / 4
    ours = B.plane.coord_at(axis, p) - A.plane.coord_at(axis, p)
    ref = b["laser"].plane.coord_at(axis, p) - a["laser"].plane.coord_at(axis, p)
    tall = A.hi[1] > floor_y + FULL_HEIGHT_M and B.hi[1] > floor_y + FULL_HEIGHT_M
    kind = "height" if axis == 1 else ("full_height" if tall else "partial_height")
    return dict(axis="xyz"[axis], kind=kind, at=p.round(3).tolist(), ours_m=round(ours, 4), laser_m=round(ref, 4),
                error_cm=round((ours - ref) * 100, 2), error_pct=round(100 * (ours - ref) / ref, 3),
                se_mm=round(math.hypot(A.plane.offset_se_mm, B.plane.offset_se_mm), 3))


def facing_pairs(matches: list[dict], floor_y: float) -> list[dict]:
    """Every pair of facing surfaces (+axis surface on the low side, -axis on the high side) that overlap in plan
    and are at least MIN_FACING_GAP_M apart. Floor-ceiling pairs give heights."""
    pairs = []
    for lo_f, hi_f, axis in (("+x", "-x", 0), ("+z", "-z", 2), ("floor", "ceiling", 1)):
        horiz = [k for k in (0, 2) if k != axis]
        for a in (m for m in matches if m["ours"].family == lo_f):
            for b in (m for m in matches if m["ours"].family == hi_f):
                A, B = a["ours"], b["ours"]
                if all(min(A.hi[k], B.hi[k]) > max(A.lo[k], B.lo[k]) for k in horiz):
                    pm = pair_measure(a, b, axis, floor_y)
                    if pm["ours_m"] >= MIN_FACING_GAP_M:
                        pairs.append(pm)
    return pairs


def room_box(matches: list[dict], centre: np.ndarray, floor_y: float) -> tuple[dict, dict]:
    """Width (x), depth (z) and height of the room around `centre`. On each side of the centre we take the LARGEST
    surface of the right orientation that spans the centre (walls are the largest surfaces; cabinet fronts and
    shower screens are smaller), then measure the pair. Returns (measurements, chosen surface pairs)."""
    out, chosen = {}, {}
    for axis, lo_f, hi_f, name in ((0, "+x", "-x", "width_x"), (2, "+z", "-z", "width_z"), (1, "floor", "ceiling", "height")):
        horiz = [k for k in (0, 2) if k != axis]

        def side(family: str, below: bool) -> dict | None:
            c = [m for m in matches if m["ours"].family == family
                 and (m["ours"].plane.coord_at(axis, centre) < centre[axis]) == below
                 and all(spans(m["ours"], k, centre[k], CENTRE_TOL_M) for k in horiz)]
            return max(c, key=lambda m: m["area_m2"]) if c else None

        a, b = side(lo_f, True), side(hi_f, False)
        if a and b:
            out[name], chosen[name] = pair_measure(a, b, axis, floor_y), (a, b)
    return out, chosen


def edge_summary(d: dict) -> dict:
    """The headline numbers of one measure_against_laser run (for the edge-margin ablation table)."""
    return dict(edge_margin_m=d["edge_margin_m"], surfaces_matched=d["surfaces_matched"],
                room_box_error_cm={k: v["error_cm"] for k, v in d["room_box"].items()},
                pair_stats=d["pair_stats"], inward_offset_cm=d["inward_offset_cm"])


def box_without_corners(chosen: dict, ours: FamilyCloud, laser: FamilyCloud, exclusion_m: float) -> dict:
    """Room box re-measured with every plane fitted only on points at least exclusion_m away from the OTHER box
    planes, in both clouds. Tests the fix for corner rounding: if depth is pulled inward near concave corners, fitting
    on the middle of each surface should remove the bias."""
    axes = {"width_x": 0, "width_z": 2, "height": 1}
    box_planes = [m["ours"].plane for pair in chosen.values() for m in pair]

    def refit(m: dict, which: str, cloud: FamilyCloud) -> Plane | None:
        s = m[which]
        X = cloud.fam[s.family]
        sel = (np.abs(X @ np.asarray(s.plane.n) - s.plane.d) < SEED_WINDOW_M) & np.isin(cell_keys(X, s.axis), s.cells)
        others = [pl for pl in box_planes if pl is not m["ours"].plane]
        far = np.min(np.abs(np.stack([X @ np.asarray(pl.n) - pl.d for pl in others], 1)), 1) >= exclusion_m
        return fit_plane_trimmed(X[sel & far])

    out = {}
    for name, (a, b) in chosen.items():
        k = axes[name]
        planes = [refit(m, w, c) for m in (a, b) for w, c in (("ours", ours), ("laser", laser))]
        if None in planes:
            continue
        pa, la, pb, lb = planes
        p = (a["ours"].lo + a["ours"].hi + b["ours"].lo + b["ours"].hi) / 4
        d_ours = pb.coord_at(k, p) - pa.coord_at(k, p)
        d_laser = lb.coord_at(k, p) - la.coord_at(k, p)
        out[name] = dict(ours_m=round(d_ours, 4), laser_m=round(d_laser, 4), error_cm=round((d_ours - d_laser) * 100, 2),
                         inliers_ours=[pa.inliers, pb.inliers])
    return out


def pair_stats(pairs: list[dict]) -> dict:
    out = {}
    for kind in ("full_height", "partial_height", "height"):
        e = np.array([p["error_cm"] for p in pairs if p["kind"] == kind])
        if len(e):
            a = np.abs(e)
            out[kind] = dict(n=int(len(e)), median_abs_cm=round(float(np.median(a)), 2),
                             p90_abs_cm=round(float(np.percentile(a, 90)), 2), max_abs_cm=round(float(a.max()), 2),
                             median_signed_cm=round(float(np.median(e)), 2),
                             within_1cm=round(float((a <= 1).mean()), 3), within_2cm=round(float((a <= 2).mean()), 3))
    return out


def error_vs_distance(pairs: list[dict]) -> dict | None:
    """Fit error = offset + slope * distance over all facing pairs. A scale error shows up as slope, a per-surface
    depth bias as offset (every interior distance loses twice the bias, whatever its length)."""
    if len(pairs) < 5:
        return None
    d = np.array([p["laser_m"] for p in pairs])
    e = np.array([p["error_cm"] for p in pairs])
    A = np.stack([np.ones_like(d), d], 1)
    (c0, c1), *_ = np.linalg.lstsq(A, e, rcond=None)
    return dict(n=int(len(d)), offset_cm=round(float(c0), 3), slope_cm_per_m=round(float(c1), 3))


def measure_against_laser(P: np.ndarray, N: np.ndarray, resolution_m: float, laser: FamilyCloud,
                          centre: np.ndarray, floor_y: float, edge_margin_m: float = EDGE_MARGIN_M) -> dict:
    """Find surfaces in our cloud, match each to the laser, and compare every facing distance and the room box."""
    ours = family_cloud(P, N, resolution_m)
    surfaces = find_surfaces(ours)
    matches = [m for m in (match_surface(s, ours, laser, edge_margin_m) for s in surfaces) if m is not None]
    pairs = facing_pairs(matches, floor_y)
    offs = np.array([m["inward_offset_cm"] for m in matches])
    box, chosen = room_box(matches, centre, floor_y)
    return dict(edge_margin_m=edge_margin_m, surfaces_found=len(surfaces), surfaces_matched=len(matches),
                room_box=box, pair_stats=pair_stats(pairs),
                error_vs_distance=error_vs_distance(pairs),
                inward_offset_cm=dict(median=round(float(np.median(offs)), 2),
                                      median_abs=round(float(np.median(np.abs(offs))), 2)) if len(offs) else None,
                surfaces=[dict(ours=m["ours"].summary(), laser=m["laser"].summary(), area_m2=m["area_m2"],
                               common_area_m2=m["common_area_m2"], inward_offset_cm=m["inward_offset_cm"])
                          for m in matches],
                pairs=pairs, _matches=matches, _box=chosen, _fc=ours)


# ---------------------------------------------------------------------------------------------------- one variant
def evaluate_variant(scene_dir: Path, pose_mode: str, cfg: Config, laser: dict, log) -> dict:
    t0 = time.time()
    cap = load_arkitscenes(scene_dir, pose_mode)
    scene, info = build_stage1(cap, cfg)
    T_LA, reg = register_laser(scene, info, laser, cfg)
    EP, EN = laser["eval"]
    LP, LN = transform(T_LA, EP), EN @ T_LA[:3, :3].T
    tree = cKDTree(LP)
    tsdf_m, *_ = surface_metrics(scene["points"], LP, LN, tree)
    raw_m, raw_pp, raw_pl, raw_j = surface_metrics(scene["raw_points"], LP, LN, tree)
    raw_ov = raw_pp < OVERLAP_M
    incidence = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", scene["raw_ray"], LN[raw_j])), 0, 1)))
    floor_y = info["floor_y"]
    med = np.median(scene["traj"][:, [0, 2]], axis=0)          # the operator walks inside the room
    centre = np.array([med[0], floor_y + 1.2, med[1]])
    laser_fc = family_cloud(LP, LN, EVAL_VOXEL_M)
    dims = {name: measure_against_laser(P, N, res, laser_fc, centre, floor_y) for name, (P, N, res) in
            (("raw", (scene["raw_points"], scene["raw_normals"], cfg.raw_voxel_m)),
             ("tsdf", (scene["points"], scene["normals"], cfg.voxel_m)))}
    res = dict(pose_mode=pose_mode, min_conf=cfg.min_conf, capture=cap.meta,
               stage1={k: info[k] for k in ("frames", "keyframes", "frames_fused", "surface_points", "raw_points",
                                            "floor_tilt_before_deg", "manhattan_yaw_deg", "manhattan_score")},
               registration=reg, surface=dict(tsdf=tsdf_m, raw=raw_m),
               completeness=dict(tsdf=completeness(LP, scene["points"]), raw=completeness(LP, scene["raw_points"])),
               raw_error_by_range=error_by_bins(raw_pl, scene["raw_range"], raw_ov, RANGE_BINS_M, "range_m"),
               raw_error_by_incidence=error_by_bins(raw_pl, incidence, raw_ov, INCIDENCE_BINS_DEG, "incidence_deg"),
               raw_error_by_corner_distance=error_by_corner_distance(scene["raw_points"], raw_pl, raw_ov,
                                                                     dims["raw"]["_box"], centre),
               raw_beyond_room_box=beyond_room_box(scene["raw_points"], raw_ov, dims["raw"]["_box"], centre),
               raw_corner_exclusion_ablation={str(m): box_without_corners(dims["raw"]["_box"], dims["raw"]["_fc"],
                                                                         laser_fc, m) for m in CORNER_EXCLUSIONS_M},
               raw_edge_margin_ablation=[edge_summary(measure_against_laser(
                   scene["raw_points"], scene["raw_normals"], cfg.raw_voxel_m, laser_fc, centre, floor_y, mg))
                   for mg in EDGE_MARGINS_TESTED_M],
               room_centre=centre.round(3).tolist(),
               dims={k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in dims.items()},
               runtime_s=round(time.time() - t0, 1))
    box = {k: v["error_cm"] for k, v in dims["raw"]["room_box"].items()}
    log(f"[{pose_mode}, conf>={cfg.min_conf}] {info['keyframes']} kf, ICP rmse {reg['icp_inlier_rmse_cm']} cm; "
        f"raw p2plane median {raw_m['point_to_plane_overlap']['median_cm']} cm; room-box error (cm) {box}; "
        f"pairs {dims['raw']['pair_stats']} ({res['runtime_s']} s)")
    return dict(result=res, scene=scene, T_LA=T_LA, laser=(LP, LN), raw_pl=raw_pl, raw_pp=raw_pp, dims=dims,
                floor_y=floor_y, centre=centre)


# ---------------------------------------------------------------------------------------------------- figures
def plot_plan(ev: dict, out: Path):
    """Top view of the wall band: laser vs our raw points, with every matched wall surface drawn from each cloud."""
    sc, (LP, LN), fy = ev["scene"], ev["laser"], ev["floor_y"]
    traj = sc["traj"][:, [0, 2]]
    lo, hi = traj.min(0) - 1.2, traj.max(0) + 1.2

    def band(P, N):
        m = (np.abs(N[:, 1]) < 0.3) & (P[:, 1] > fy + 1.0) & (P[:, 1] < fy + 1.5)
        return P[m & np.all((P[:, [0, 2]] > lo) & (P[:, [0, 2]] < hi), axis=1)]

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
    for a, (P, N, col) in zip(ax, ((LP, LN, C_LASER), (sc["raw_points"], sc["raw_normals"], C_OURS))):
        B = band(P, N)
        a.scatter(B[:, 0], B[:, 2], s=0.2, c=col, lw=0)
    box = {id(m["ours"]) for pair in ev["dims"]["raw"]["_box"].values() for m in pair}
    for m in ev["dims"]["raw"]["_matches"]:
        lw = 2.2 if id(m["ours"]) in box else 0.9
        for s, ls, col in ((m["laser"], "--", "#52514e"), (m["ours"], "-", "#0b0b0b")):
            if s.axis == 1:
                continue
            k = 2 if s.axis == 0 else 0                       # the horizontal direction along the wall
            t = np.array([s.lo[k], s.hi[k]])
            c = [s.plane.coord_at(s.axis, np.where(np.arange(3) == k, tt, (s.lo + s.hi) / 2)) for tt in t]
            for a in ax:
                a.plot(*((c, t) if s.axis == 0 else (t, c)), ls, c=col, lw=lw)
    handles = [Line2D([], [], ls="", marker="o", c=C_LASER, label="Faro laser (registered), 1.0-1.5 m above floor"),
               Line2D([], [], ls="", marker="o", c=C_OURS, label="our raw LiDAR points, 1.0-1.5 m above floor"),
               Line2D([], [], ls="-", c="#0b0b0b", label="surface fitted on our raw points (thick = room box)"),
               Line2D([], [], ls="--", c="#52514e", label="same surface fitted on the laser"),
               Line2D([], [], ls="-", c="#9a9992", label="camera path"),
               Line2D([], [], ls="", marker="+", c="#e34948", mew=2, label="room centre")]
    for a in ax:
        a.plot(traj[:, 0], traj[:, 1], "-", c="#9a9992", lw=0.6)
        a.plot(*ev["centre"][[0, 2]], "+", c="#e34948", ms=14, mew=2)
        a.set_aspect("equal"); a.grid(alpha=0.3); a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1])
        a.set_xlabel("x (m)"); a.set_ylabel("z (m)")
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    box = ev["result"]["dims"]["raw"]["room_box"]
    txt = ", ".join(f"{k} {v['ours_m']:.3f} vs {v['laser_m']:.3f} m ({v['error_cm']:+.1f} cm)" for k, v in box.items())
    fig.suptitle(f"ARKitScenes room, top view (left: laser, right: ours). Room box, ours vs laser: {txt}", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out / "plan_overlay.png", dpi=120); plt.close(fig)


def plot_error_map(ev: dict, out: Path):
    """Where our raw points disagree with the laser: a top view, then each room-box wall and the ceiling seen
    face-on (points within 10 cm of the plane), coloured by |point-to-plane error| in cm."""
    P = ev["scene"]["raw_points"]
    err = np.abs(ev["raw_pl"]) * 100
    ov = ev["raw_pp"] < OVERLAP_M
    views = [("top view, all raw points", np.ones(len(P), bool), 0, 2)]
    for name, pair in ev["dims"]["raw"]["_box"].items():
        for m in pair:
            s = m["ours"]
            near = np.abs(P @ np.asarray(s.plane.n) - s.plane.d) < 0.10
            if s.axis == 1:
                views.append((f"{s.family} ({name}), seen from inside", near, 0, 2))
            else:
                k = 2 if s.axis == 0 else 0
                near &= (P[:, k] > s.lo[k] - 0.1) & (P[:, k] < s.hi[k] + 0.1)
                views.append((f"wall '{s.family}' at {'xyz'[s.axis]} = {s.plane.coord_at(s.axis, (s.lo + s.hi) / 2):.2f} m",
                              near, k, 1))
    views = [v for v in views if v[1].any() and "floor" not in v[0]]
    ncol = 3
    nrow = math.ceil(len(views) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow), squeeze=False)
    for a, (title, m, i, j) in zip(axs.flat, views):
        o = np.flatnonzero(m & ov)
        o = o[np.argsort(err[o])]                       # large errors drawn last so they stay visible
        g = m & ~ov
        a.scatter(P[g, i], P[g, j], s=0.3, c="#c3c2b7", lw=0, label="no laser point within 10 cm")
        im = a.scatter(P[o, i], P[o, j], s=0.3, c=err[o], cmap="Blues", vmin=0, vmax=5, lw=0)
        a.set_title(title, fontsize=10); a.set_aspect("equal"); a.grid(alpha=0.2)
        a.set_xlabel(f"{'xyz'[i]} (m)"); a.set_ylabel(f"{'xyz'[j]} (m)" + (" (up)" if j == 1 else ""))
    for a in axs.flat[len(views):]:
        a.axis("off")
    axs[0, 0].legend(loc="upper left", fontsize=8, markerscale=12, frameon=False)
    fig.colorbar(im, ax=axs, shrink=0.6, label="|raw point - laser surface| (cm), clipped at 5")
    fig.suptitle("Raw LiDAR point error against the Faro laser scan")
    fig.savefig(out / "raw_error_map.png", dpi=100, bbox_inches="tight"); plt.close(fig)


def plot_error_stats(ev: dict, out: Path):
    """Signed error distributions (raw vs TSDF), and raw-point error by range and by incidence angle."""
    sc, (LP, LN) = ev["scene"], ev["laser"]
    t_pp, t_pl, _ = signed_distances(sc["points"], LP, LN, cKDTree(LP))
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    bins = np.linspace(-5, 5, 101)
    for d_pl, d_pp, col, lbl in ((ev["raw_pl"], ev["raw_pp"], C_OURS, "raw LiDAR points"),
                                 (t_pl, t_pp, C_TSDF, "TSDF surface")):
        v = d_pl[d_pp < OVERLAP_M] * 100
        ax[0].hist(v, bins=bins, histtype="step", lw=2, color=col, density=True,
                   label=f"{lbl}: median |e| {np.median(np.abs(v)):.2f} cm, signed median {np.median(v):+.2f} cm")
    ax[0].axvline(0, c="#52514e", lw=0.8)
    ax[0].set_xlabel("signed point-to-plane distance to laser (cm)\n+ = in front of the true surface (towards the camera)")
    ax[0].set_ylabel("density"); ax[0].legend(frameon=False, fontsize=8, loc="upper left"); ax[0].grid(alpha=0.3)
    for a, key, unit in ((ax[1], "raw_error_by_range", "range from camera (m)"),
                         (ax[2], "raw_error_by_incidence", "incidence angle (deg, 0 = head-on)")):
        rows = ev["result"][key]
        name = "range_m" if "range" in key else "incidence_deg"
        x = np.arange(len(rows))
        a.bar(x - 0.2, [r["median_cm"] for r in rows], 0.38, color=C_OURS, label="median |e|")
        a.bar(x + 0.2, [r["p90_cm"] for r in rows], 0.38, color="#9ec3ef", label="p90 |e|")
        a.plot(x, [r["signed_median_cm"] for r in rows], "o-", c="#0b0b0b", lw=1, ms=4, label="signed median")
        a.set_xticks(x, [f"{r[name][0]}-{r[name][1]}\nn={r['n']:,}" for r in rows], fontsize=8)
        a.set_xlabel(unit); a.set_ylabel("raw point error (cm)")
        a.axhline(0, c="#52514e", lw=0.8); a.legend(frameon=False, fontsize=8); a.grid(alpha=0.3, axis="y")
    fig.suptitle("Raw LiDAR and TSDF error against the Faro laser (points with a laser point within 10 cm)")
    fig.tight_layout(); fig.savefig(out / "error_stats.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------------------------------- main
def compare_reference(T_LA: np.ndarray, T_align: np.ndarray, ref_path: Path) -> dict:
    """Cross-check against an independently estimated laser -> ARKitScenes-world transform (z-up, column vectors)."""
    T_ref = T_align @ Z_UP_TO_Y_UP @ np.loadtxt(ref_path) @ np.linalg.inv(Z_UP_TO_Y_UP)
    rot, trans = rigid_delta(np.linalg.inv(T_ref) @ T_LA)
    return dict(path=str(ref_path), rotation_diff_deg=round(rot, 3), translation_diff_cm=round(trans * 100, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", type=Path, required=True, help="ARKitScenes raw scene folder (raw/<split>/<video_id>)")
    ap.add_argument("--laser", type=Path, nargs="+", required=True, help="Faro PLY(s) of the visit (+ *_pose.txt)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/arkitscenes_validation")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--reference-transform", type=Path, default=None,
                    help="optional independent laser->ARKitScenes-world 4x4 to cross-check the registration")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    log = lambda m: print(m, flush=True)  # noqa: E731

    t0 = time.time()
    laser = load_laser(a.laser)
    log(f"[laser] {laser['n_points_file']:,} points in file, {len(laser['eval'][0]):,} at {EVAL_VOXEL_M*1000:.0f} mm "
        f"(median spacing {laser['eval_spacing_median_mm']} mm), {len(laser['reg'][0]):,} at {REG_VOXEL_M*100:.0f} cm "
        f"({time.time()-t0:.1f} s)")

    base_cfg = Config.load()
    base = evaluate_variant(a.scene, "nearest", base_cfg, laser, log)
    results = dict(scene=str(a.scene), laser=[str(p) for p in a.laser],
                   laser_info={k: laser[k] for k in ("n_points_file", "n_points_read", "eval_spacing_median_mm")},
                   laser_scanners=laser["scanners"].tolist(), config=base_cfg.to_dict(), baseline=base["result"])
    if a.reference_transform:
        results["registration_cross_check"] = compare_reference(base["T_LA"], base["scene"]["T_align"],
                                                                a.reference_transform)
        log(f"[cross-check] {results['registration_cross_check']}")
    if not a.no_ablations:
        variants = [("interpolate", base_cfg)] + [("nearest", Config.load(min_conf=c)) for c in (1, 0)]
        results["ablations"] = [evaluate_variant(a.scene, m, c, laser, log)["result"] for m, c in variants]

    plot_plan(base, a.out)
    plot_error_map(base, a.out)
    plot_error_stats(base, a.out)
    np.savetxt(a.out / "T_laser_to_aligned.txt", base["T_LA"], fmt="%.8f",
               header="p_aligned = T @ p_laser_yup, p_laser_yup = (x, z, -y) of the Faro PLY; column vectors")
    (a.out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    log(f"[done] {time.time()-t0:.1f} s -> {a.out}")


if __name__ == "__main__":
    main()
