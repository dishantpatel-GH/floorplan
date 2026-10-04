"""Detectors for surfaces and conditions that corrupt measurements: mirrors, glass, glossy floors, low light, blur.

Why this module exists
----------------------
Every tier measures geometry from light that comes back to the phone. Four common surfaces/conditions break that:

* **Mirrors** return the *reflected* room, so a reconstruction (LiDAR, multi-view, learned depth) can contain a phantom
  copy of the room *behind* the mirror plane. If a wall is fitted through that, the room grows.
* **Glass** (windows, shower screens) returns nothing, a weak low-confidence echo, or what is *behind* it. A glazed
  wall section looks like an open doorway.
* **Glossy / wet-look floors** mirror the ceiling lights and furniture *below* the floor plane and blow out highlights.
* **Low light and motion** blur the images and add sensor noise; feature matching and learned depth degrade.

Design rule: detectors never silently delete data. Each returns a `SurfaceFlag` with the evidence (numbers) and the
recommended `Handling`, so the caller decides (exclude points, re-label an opening, widen an interval, ask for a
re-capture) and the decision is auditable in the output JSON.

Conventions: world +y is up; plan coordinates (u, v) = (x, z) of the aligned world (see floorplan/model.py).
Everything is deterministic and has no capture-specific constants: thresholds are physical (cm, degrees) or relative
to the capture itself (e.g. sharpness relative to temporal neighbours).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------------------------------------------------
# Flags and handling vocabulary
# --------------------------------------------------------------------------------------------------------------------
class Handling:
    """Recommended actions. Strings (not an Enum) so they serialise directly into the plan JSON."""
    EXCLUDE_POINTS = "exclude_points"              # drop the flagged points before fitting walls / measuring
    PLANE_IS_WALL = "treat_reflector_plane_as_wall"  # the mirror/glass plane bounds the room: measure to it
    LABEL_GLASS = "label_opening_as_glass"          # a see-through gap with a sill/frame is a window/screen, not a door
    WIDEN_INTERVAL = "widen_interval"               # keep the measurement, enlarge its 95% interval
    DOWNWEIGHT_FRAME = "downweight_frame"           # use the frame with a lower weight / skip it for matching
    SKIP_FRAME = "skip_frame"
    RECAPTURE = "ask_for_recapture"                 # tell the user what to fix (e.g. "turn on the lights")
    MASK_FOR_DAMAGE = "mask_for_damage_detection"   # specular highlights/reflections are not stains


@dataclass
class SurfaceFlag:
    kind: str                     # mirror | see_through | glass_lowconf | specular_floor | blur | low_light | ...
    score: float                  # 0..1 strength of the evidence
    where: dict                   # plan location (segment / bbox / centroid) or frame indices
    evidence: dict                # the numbers that triggered it
    handling: list[str] = field(default_factory=list)
    message: str = ""             # one plain-language sentence for the report / the user

    def to_dict(self) -> dict:
        d = asdict(self)
        return _jsonable(d)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return round(float(x), 4)
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# --------------------------------------------------------------------------------------------------------------------
# 1. Image quality per frame (all tiers: photos, video, LiDAR RGB)
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class FrameQuality:
    sharpness: float      # variance of the Laplacian at a fixed working resolution (higher = sharper)
    brightness: float     # median luma, 0..255
    dark_frac: float      # fraction of pixels with luma < 25 (crushed shadows)
    clipped_frac: float   # fraction of pixels with luma >= 250 (blown highlights: lamps, specular reflections)
    noise_sigma: float    # robust sensor-noise estimate in 8-bit grey levels (Immerkaer operator, median based)


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def frame_quality(img: np.ndarray, long_side: int = 640) -> FrameQuality:
    """Cheap, model-free image quality numbers for one frame.

    Why a fixed working resolution: the Laplacian variance depends on pixel scale, so a 12 MP photo and a 1080p video
    frame would not be comparable. Resizing to the same long side makes the number comparable across devices.
    Why this noise estimate: the Immerkaer operator cancels smooth image content; restricting it to the flatter half of
    the pixels removes edges, so what remains is sensor noise, which grows sharply in low light (high ISO).
    """
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    s = long_side / max(h, w)
    gs = cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else g
    gf = gs.astype(np.float32)
    lap = cv2.Laplacian(gf, cv2.CV_32F, ksize=3)
    # noise on a 2x-larger image than the sharpness one, so downsampling does not average the noise away
    s2 = min(1.0, 2 * long_side / max(h, w))
    gn = cv2.resize(g, (round(w * s2), round(h * s2)), interpolation=cv2.INTER_AREA).astype(np.float32) if s2 < 1 else g.astype(np.float32)
    r = np.abs(cv2.filter2D(gn, cv2.CV_32F, _IMMERKAER)[1:-1, 1:-1])
    gx = cv2.Sobel(gn, cv2.CV_32F, 1, 0)[1:-1, 1:-1]
    gy = cv2.Sobel(gn, cv2.CV_32F, 0, 1)[1:-1, 1:-1]
    gm = np.hypot(gx, gy)
    flat = gm <= np.percentile(gm, 50)                    # textured pixels would read as "noise"
    # mean (not median) form of Immerkaer: sigma = sqrt(pi/2) * mean|r| / 6. The median collapses to 0 or 1 grey
    # level on compressed video (the codec removes most noise), the mean stays continuous.
    noise = float(np.sqrt(np.pi / 2) * r[flat].mean() / 6.0)
    return FrameQuality(sharpness=float(lap.var()), brightness=float(np.median(gs)),
                        dark_frac=float((gs < 25).mean()), clipped_frac=float((gs >= 250).mean()), noise_sigma=noise)


def _rolling_median(x: np.ndarray, t: np.ndarray, half_window_s: float) -> np.ndarray:
    lo = np.searchsorted(t, t - half_window_s, side="left")
    hi = np.searchsorted(t, t + half_window_s, side="right")
    return np.array([np.median(x[a:b]) for a, b in zip(lo, hi)])


def flag_frames(q: Sequence[FrameQuality], timestamps: np.ndarray | None = None, rel_blur: float = 0.5,
                half_window_s: float = 1.0, min_brightness: float = 60.0, max_noise: float = 3.0,
                capture_blur_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray, list[SurfaceFlag]]:
    """Per-frame weights (0..1) and blur/low-light flags, plus capture-level flags.

    Blur is judged RELATIVE to the frame's temporal neighbours (video) or to the folder median (photos, no timestamps):
    a white wall is never "sharp" in absolute terms, but a frame that is half as sharp as the frames 1 s around it is
    motion-blurred. Low light is absolute (median luma, noise) because it is a property of the room, not of the frame.
    Returns (weight, is_blurry, flags).
    """
    sh = np.array([f.sharpness for f in q], float)
    br = np.array([f.brightness for f in q], float)
    nz = np.array([f.noise_sigma for f in q], float)
    if timestamps is not None and len(sh) > 2:
        ref = _rolling_median(sh, np.asarray(timestamps, float), half_window_s)
    else:
        ref = np.full_like(sh, np.median(sh))
    ratio = sh / np.maximum(ref, 1e-6)
    blurry = ratio < rel_blur
    dark = (br < min_brightness) | (nz > max_noise)
    w = np.clip(ratio, 0, 1) * np.where(dark, 0.5, 1.0)
    flags: list[SurfaceFlag] = []
    if len(sh) and blurry.mean() > capture_blur_frac:
        flags.append(SurfaceFlag("blur", float(blurry.mean()), {"frames": "capture"},
                                 {"blurry_frac": float(blurry.mean()), "median_sharpness": float(np.median(sh))},
                                 [Handling.RECAPTURE, Handling.WIDEN_INTERVAL],
                                 "Many frames are motion-blurred: move/turn more slowly or add light."))
    if len(br) and dark.mean() > 0.5:
        flags.append(SurfaceFlag("low_light", float(dark.mean()), {"frames": "capture"},
                                 {"dark_frac": float(dark.mean()), "median_brightness": float(np.median(br)),
                                  "median_noise": float(np.median(nz))},
                                 [Handling.RECAPTURE, Handling.WIDEN_INTERVAL],
                                 "The capture is dark and noisy: turn on all lights."))
    return w, blurry, flags


def pick_sharpest(sharpness: np.ndarray, timestamps: np.ndarray, window_s: float = 0.5) -> np.ndarray:
    """Indices of the sharpest frame in each consecutive time window: the frame-selection rule for the video tier.
    Why: within half a second the viewpoint barely changes, but blur varies a lot with hand shake."""
    t = np.asarray(timestamps, float)
    bins = np.floor((t - t[0]) / window_s).astype(int)
    out = []
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        out.append(idx[np.argmax(sharpness[idx])])
    return np.array(out, int)


# --------------------------------------------------------------------------------------------------------------------
# 2. LiDAR depth confidence (LiDAR tier)
# --------------------------------------------------------------------------------------------------------------------
def depth_confidence_stats(conf: np.ndarray) -> dict:
    """Fractions of low (0), medium (1) and high (2) confidence pixels in one ARKit confidence map."""
    n = conf.size
    return {"low": float((conf == 0).sum() / n), "medium": float((conf == 1).sum() / n),
            "high": float((conf == 2).sum() / n)}


def lowconf_blob_mask(conf: np.ndarray, min_conf: int = 2, open_px: int = 2, min_area_frac: float = 0.004) -> np.ndarray:
    """Pixels below `min_conf` that form BLOBS, not thin lines.

    Why: ARKit marks every depth discontinuity (object edges) as low confidence; those are 1-3 px lines and harmless.
    Glass, mirrors, dark glossy panels and far see-through regions produce 2D blobs. A morphological opening removes
    the lines; a minimum area removes specks.
    """
    m = (conf < min_conf).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_frac * conf.size
    return keep[lab]


def pixel_rays_world(K: np.ndarray, T_wc: np.ndarray, h: int, w: int) -> np.ndarray:
    """(h, w, 3) world-frame direction of each pixel's ray, scaled so that camera z = 1 (multiply by z-depth)."""
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    rc = np.stack([x, y, np.ones_like(x)], -1)
    return rc @ T_wc[:3, :3].T


def floor_reflection_stats(depth_m: np.ndarray, conf: np.ndarray, K: np.ndarray, T_wc: np.ndarray, floor_y: float,
                           rgb_small: np.ndarray | None = None, margin_m: float = 0.04,
                           max_range_m: float = 4.0) -> dict:
    """How a glossy floor corrupts one depth frame.

    For every pixel whose ray would hit the floor plane within range, compare the measured 3D point with the floor:
    a point clearly BELOW the floor (y < floor - margin) can only be a reflection in a glossy/wet floor (or a hole/
    stair). Also reports the low-confidence fraction on floor pixels and, if the RGB frame is given at depth resolution,
    the fraction of blown-out highlights on the floor (lamps reflected in tiles), which also fool damage detection.
    """
    h, w = depth_m.shape
    d = pixel_rays_world(K, T_wc, h, w)
    cy = T_wc[1, 3]
    dy = d[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_floor = (floor_y - cy) / dy
    floorward = (dy < -1e-3) & (z_floor > 0.2) & (z_floor < max_range_m)
    y = cy + depth_m * dy
    valid = floorward & (depth_m > 0)
    on_floor = valid & (np.abs(y - floor_y) <= margin_m)
    below = valid & (y < floor_y - margin_m)
    nf = max(int(floorward.sum()), 1)
    out = {"floor_px": int(floorward.sum()), "on_floor_frac": float(on_floor.sum() / nf),
           "below_floor_frac": float(below.sum() / nf),
           "below_floor_highconf_frac": float((below & (conf == 2)).sum() / nf),
           "lowconf_on_floor_frac": float((floorward & (conf < 2)).sum() / nf)}
    if rgb_small is not None:
        g = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2GRAY) if rgb_small.ndim == 3 else rgb_small
        out["clipped_on_floor_frac"] = float(((g >= 250) & floorward).sum() / nf)
    return out


def below_floor_points(points: np.ndarray, floor_y: float, up_axis: int = 1, margin_m: float = 0.10,
                       min_points: int = 500) -> tuple[np.ndarray, list[SurfaceFlag]]:
    """Points more than `margin_m` below the floor plane: either a real lower level (a stairwell - this is what the
    sample apartment's below-floor points are, see docs/FAILURE_MODES.md) or reflections in a glossy/wet floor (the
    specular floor returns the reflected path, not the floor). Either way they must not shape the room's floor or
    walls. Cheap and plan-free, so every tier can run it after levelling; the plan extractor decides which it is
    (a stair has risers at regular steps; a reflection mirrors the room above)."""
    m = points[:, up_axis] < floor_y - margin_m
    flags = []
    if m.sum() >= min_points:
        P = points[m]
        plan_axes = [i for i in range(3) if i != up_axis]
        flags.append(SurfaceFlag("below_floor", float(min(1.0, m.mean() * 100)),
                                 {"plan_bbox": [P[:, plan_axes].min(0), P[:, plan_axes].max(0)]},
                                 {"n_points": int(m.sum()), "frac": float(m.mean()),
                                  "depth_below_floor_m_p50_p95": np.percentile(floor_y - P[:, up_axis], [50, 95])},
                                 [Handling.EXCLUDE_POINTS, Handling.MASK_FOR_DAMAGE],
                                 "Points below the floor: a stairwell / lower level, or reflections in a glossy floor."))
    return m, flags


# --------------------------------------------------------------------------------------------------------------------
# 3. Geometry consistency: phantom geometry behind observed surfaces (mirrors, see-through) - any tier with 3D points
# --------------------------------------------------------------------------------------------------------------------
class VoxelOccupancy:
    """Sparse voxel grid of observed surfaces (count + mean normal per voxel), built from a fused surface.

    Why sparse (sorted keys + searchsorted): a 14 x 14 x 3.5 m apartment at 5 cm has 5.5 M voxels, of which only
    ~2% are occupied; a sorted key array keeps memory small and lookups vectorised.
    """

    def __init__(self, points: np.ndarray, normals: np.ndarray | None = None, voxel: float = 0.05):
        self.voxel = float(voxel)
        self.origin = points.min(0) - 2 * voxel
        ijk = np.floor((points - self.origin) / voxel).astype(np.int64)
        self.dims = ijk.max(0) + 3
        keys = self._key(ijk)
        self.keys, inv, self.count = np.unique(keys, return_inverse=True, return_counts=True)
        if normals is not None:
            ns = np.zeros((len(self.keys), 3))
            np.add.at(ns, inv, normals.astype(np.float64))
            self.normal = ns / np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-9)
        else:
            self.normal = None

    def _key(self, ijk: np.ndarray) -> np.ndarray:
        return (ijk[..., 0] * self.dims[1] + ijk[..., 1]) * self.dims[2] + ijk[..., 2]

    def lookup(self, xyz: np.ndarray) -> np.ndarray:
        """Index into self.keys for each query point, -1 where the voxel is empty."""
        ijk = np.floor((xyz - self.origin) / self.voxel).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < self.dims), axis=-1)
        k = self._key(np.where(inside[..., None], ijk, 0))
        pos = np.clip(np.searchsorted(self.keys, k), 0, len(self.keys) - 1)
        hit = inside & (self.keys[pos] == k)
        return np.where(hit, pos, -1)


def occlusion_violations(points: np.ndarray, origins: np.ndarray, occ: VoxelOccupancy, min_count: int = 3,
                         start_m: float = 0.3, margin_m: float = 0.15, min_cross: int = 2, cos_cross: float = 0.34,
                         chunk: int = 40000) -> tuple[np.ndarray, np.ndarray]:
    """Which points were seen THROUGH an observed opaque surface (a physical impossibility for a real surface).

    For each point, march along its viewing ray from the camera and count samples that fall in occupied voxels whose
    surface normal faces the ray (|cos| > cos_cross, so grazing along a wall does not count) before reaching
    `margin_m` short of the point. A real surface cannot be seen through a wall; a mirror's reflected copy, a surface
    seen through glass that other views saw as opaque, or a door that moved during capture can.
    Returns (violation bool (N,), first-crossing position (N,3), NaN where none).
    """
    step = occ.voxel / 2
    N = len(points)
    viol = np.zeros(N, bool)
    first = np.full((N, 3), np.nan)
    for a in range(0, N, chunk):
        p, o = points[a:a + chunk].astype(np.float64), origins[a:a + chunk].astype(np.float64)
        v = p - o
        L = np.linalg.norm(v, axis=1)
        d = v / np.maximum(L[:, None], 1e-9)
        n_steps = int(np.ceil((L.max() - margin_m - start_m) / step)) if len(L) else 0
        if n_steps <= 0:
            continue
        t = start_m + step * np.arange(n_steps)
        valid_t = t[None, :] < (L - margin_m)[:, None]
        xyz = o[:, None, :] + t[None, :, None] * d[:, None, :]
        idx = occ.lookup(xyz)
        occd = (idx >= 0) & valid_t
        cnt = np.where(occd, occ.count[np.maximum(idx, 0)], 0)
        occd &= cnt >= min_count
        if occ.normal is not None:
            cosang = np.abs(np.einsum("nsk,nk->ns", occ.normal[np.maximum(idx, 0)], d))
            occd &= cosang > cos_cross
        ncross = occd.sum(1)
        vi = ncross >= min_cross
        viol[a:a + chunk] = vi
        if vi.any():
            j = np.argmax(occd[vi], axis=1)
            first[a:a + chunk][vi] = xyz[vi, j]
    return viol, first


def cluster_plan(xz: np.ndarray, cell: float = 0.1, min_pts_cell: int = 3, min_cells: int = 4,
                 dilate: int = 1) -> np.ndarray:
    """Connected-component labels (-1 = noise) of points in plan view. Why plan view: mirrors, windows and screens are
    vertical, so their phantom content forms a compact footprint behind the wall line."""
    if len(xz) == 0:
        return np.zeros(0, int)
    o = xz.min(0)
    ij = np.floor((xz - o) / cell).astype(int)
    H = np.zeros(ij.max(0) + 1, int)
    np.add.at(H, (ij[:, 0], ij[:, 1]), 1)
    occ = H >= min_pts_cell
    if dilate:
        occ = ndimage.binary_dilation(occ, iterations=dilate) & (H > 0)
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    sizes = ndimage.sum(occ, lab, index=np.arange(1, n + 1))
    good = np.zeros(n + 1, bool)
    good[1:] = sizes >= min_cells
    l = lab[ij[:, 0], ij[:, 1]]
    return np.where(good[l], l, -1)


def fit_plane(P: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares plane n.x + d = 0 through points; returns (n, d, rms)."""
    c = P.mean(0)
    _, s, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    return n, float(-n @ c), float(s[-1] / np.sqrt(len(P)))


def reflection_consistency(phantom: np.ndarray, n: np.ndarray, d: float, reference: np.ndarray,
                           cameras: np.ndarray, tol_m: float = 0.05, shift_m: float = 0.25) -> dict:
    """Mirror test: reflect the phantom points across the candidate plane and check they land on real geometry.

    A mirror's phantom is an exact reflected copy of the room in front of it, so after reflection the points must
    coincide with real surfaces ON THE CAMERA SIDE of the plane (the reference is restricted to that side, otherwise
    the phantom would trivially match itself). Content seen through glass (a real room behind) does not.
    Control: reflecting across the same plane shifted by +-`shift_m` must match much worse. Without the control, a
    cluttered room "matches" almost anything within 5 cm; the contrast (true - control) is what proves a mirror.
    """
    if np.mean(cameras @ n + d) < 0:                  # orient the plane so the cameras are on the + side
        n, d = -n, -d
    ref = reference[(reference @ n + d) > 0.03]
    if len(ref) == 0:
        return {"match_frac_reflected": 0.0, "match_frac_control": 0.0, "contrast": 0.0}
    tree = cKDTree(ref)

    def match(dd: float) -> tuple[float, float]:
        s_ = phantom @ n + dd
        dist, _ = tree.query(phantom - 2 * s_[:, None] * n[None, :], k=1, distance_upper_bound=1.0)
        return float((dist < tol_m).mean()), float(np.median(np.minimum(dist, 1.0)))

    m_true, med = match(d)
    m_ctrl = max(match(d + shift_m)[0], match(d - shift_m)[0])
    return {"match_frac_reflected": m_true, "match_frac_control": m_ctrl, "contrast": m_true - m_ctrl,
            "median_nn_reflected_m": med,
            "median_dist_behind_plane_m": float(np.median(np.abs(phantom @ n + d)))}


def is_mirror(rc: dict, min_match: float = 0.5, min_contrast: float = 0.25) -> bool:
    """A mirror needs BOTH a good reflected match and a clear margin over the shifted-plane control."""
    return rc.get("match_frac_reflected", 0) >= min_match and rc.get("contrast", 0) >= min_contrast


def detect_mirrors(points: np.ndarray, origins: np.ndarray, occ: VoxelOccupancy, reference: np.ndarray,
                   up_axis: int = 1, max_query: int = 400000, min_cluster_pts: int = 150,
                   mirror_match: float = 0.5, max_plane_rms_m: float = 0.05,
                   seed: int = 0) -> tuple[list[SurfaceFlag], np.ndarray]:
    """Find phantom geometry seen through observed surfaces and classify each cluster.

    Inputs are tier-agnostic: 3D points with the camera centre each was seen from (LiDAR raw points, or a
    multi-view/learned point map with its camera centres) and an occupancy grid of the fused surface.
    Classes:
      * mirror            plane vertical, reflected copy matches real geometry -> exclude points, plane is a wall
      * floor_reflection  crossing plane horizontal (a glossy floor/ceiling)  -> exclude points
      * see_through       planar crossing but no reflection match (glass that other views saw as opaque) -> exclude,
                          widen the interval of the nearby wall
      * inconsistent_geometry  the crossings do not form a plane (rms > max_plane_rms_m): drift between passes,
                          moved objects, people -> exclude, widen; the drift module should look here
    Returns (flags, phantom mask over the input points).
    """
    rng = np.random.default_rng(seed)
    N = len(points)
    q = np.sort(rng.choice(N, size=min(N, max_query), replace=False))
    viol, first = occlusion_violations(points[q], origins[q], occ)
    mask = np.zeros(N, bool)
    mask[q[viol]] = True
    flags: list[SurfaceFlag] = []
    if viol.sum() < min_cluster_pts:
        return flags, mask
    P, F = points[q][viol], first[viol]
    plan_axes = [i for i in range(3) if i != up_axis]
    lab = cluster_plan(P[:, plan_axes])
    O = origins[q][viol]
    for c in sorted(set(lab.tolist()) - {-1}):
        m = lab == c
        if m.sum() < min_cluster_pts:
            continue
        n, d, rms = fit_plane(F[m])
        vertical = abs(n[up_axis]) < 0.3
        horizontal = abs(n[up_axis]) > 0.9
        ev = {"n_points": int(m.sum()), "plane_normal": n, "plane_rms_m": rms,
              "phantom_height_range_m": [float(P[m, up_axis].min()), float(P[m, up_axis].max())],
              "crossing_centroid": F[m].mean(0)}
        flat = rms < max_plane_rms_m
        ev["crossing_plane"] = ("not_planar" if not flat else
                                "horizontal" if horizontal else ("vertical" if vertical else "oblique"))
        below = flat and horizontal and np.median(P[m, up_axis]) < np.median(F[m, up_axis]) - 0.05
        if not flat:
            # the surfaces it was seen through do not form one plane: not a mirror/glass pane (those are flat) but
            # two observations of the scene that disagree - pose drift between passes, a moved object, a person
            kind, score = "inconsistent_geometry", float(min(1.0, m.sum() / 2000))
            handling = [Handling.EXCLUDE_POINTS, Handling.WIDEN_INTERVAL]
            msg = "Two views disagree about what is solid here (drift between passes, moved object or person)."
        elif below:
            kind, score = "floor_reflection", 1.0
            handling = [Handling.EXCLUDE_POINTS, Handling.MASK_FOR_DAMAGE]
            msg = "Points below the floor plane: reflections in a glossy floor."
        elif horizontal:
            # seen through a horizontal surface but NOT below it: e.g. a bed/table top that moved, or two passes
            # of a drifting capture disagreeing; not a reflection, so no mirror test
            kind, score = "see_through", float(min(1.0, m.sum() / 2000))
            handling = [Handling.EXCLUDE_POINTS, Handling.WIDEN_INTERVAL]
            msg = "Geometry seen through a horizontal surface (moved object or multi-pass drift)."
        else:
            rc = reflection_consistency(P[m], n, d, reference, O[m]) if vertical else {"match_frac_reflected": 0.0}
            ev.update(rc)
            if vertical and is_mirror(rc, mirror_match):
                kind, score = "mirror", rc["match_frac_reflected"]
                handling = [Handling.EXCLUDE_POINTS, Handling.PLANE_IS_WALL, Handling.WIDEN_INTERVAL]
                msg = "A mirror: geometry behind it is a reflected copy of the room."
            else:
                kind, score = "see_through", float(min(1.0, m.sum() / 2000))
                handling = [Handling.EXCLUDE_POINTS, Handling.WIDEN_INTERVAL]
                msg = "Geometry seen through a surface that other views saw as solid (glass, moved door, person)."
        bb = P[m][:, plan_axes]
        flags.append(SurfaceFlag(kind, float(score),
                                 {"plan_bbox": [bb.min(0), bb.max(0)], "crossing_uv": F[m].mean(0)[plan_axes]},
                                 ev, handling, msg))
    return flags, mask


# --------------------------------------------------------------------------------------------------------------------
# 4. Wall-aware checks (needs a plan): points behind a wall line, mirror vs glass vs opening
# --------------------------------------------------------------------------------------------------------------------
def _wall_arrays(walls: Iterable) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    p0, p1, nn, ids = [], [], [], []
    for w in walls:
        g = w if isinstance(w, dict) else w.__dict__
        p0.append(g["p0"]); p1.append(g["p1"]); nn.append(g["normal"]); ids.append(str(g.get("id", len(ids))))
    return np.asarray(p0, float), np.asarray(p1, float), np.asarray(nn, float), ids


def points_behind_walls(uv: np.ndarray, origin_uv: np.ndarray, walls: Iterable, margin_m: float = 0.10,
                        thickness_m: float = 0.30) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each point (plan coords), the wall whose segment its viewing ray crosses with the point lying behind it.

    "Behind" = on the outside of the wall (normal points into the room) by more than `margin_m` + an assumed wall
    thickness, so the wall's own back face is ignored, while the next room seen THROUGH A REAL OPENING still shows up
    (the caller tells them apart with the reflection test and the crossing height).
    Returns (wall index or -1, distance behind (m), where the ray crosses the wall: metres along it from p0, and the
    ray parameter t of the crossing (0 = camera, 1 = point) so the caller can get the crossing HEIGHT).
    """
    p0, p1, nn, _ = _wall_arrays(walls)
    N = len(uv)
    best = np.full(N, -1)
    behind = np.zeros(N)
    s_m = np.zeros(N)
    t_x = np.zeros(N)
    o, r = origin_uv, uv - origin_uv
    for k in range(len(p0)):
        e = p1[k] - p0[k]
        den = r[:, 0] * e[1] - r[:, 1] * e[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            w0 = p0[k] - o
            t = (w0[:, 0] * e[1] - w0[:, 1] * e[0]) / den      # along the ray (0 = camera, 1 = point)
            s = (w0[:, 0] * r[:, 1] - w0[:, 1] * r[:, 0]) / den  # along the wall (0..1)
        dist = -((uv - p0[k]) @ nn[k])                           # > 0 outside the room
        cam_inside = ((o - p0[k]) @ nn[k]) > 0
        hit = (np.abs(den) > 1e-9) & (t > 0) & (t < 1) & (s > 0.02) & (s < 0.98) & cam_inside \
            & (dist > margin_m + thickness_m)
        upd = hit & ((best < 0) | (t < t_x))                     # the FIRST wall the ray crosses
        best[upd], behind[upd], s_m[upd], t_x[upd] = k, dist[upd], s[upd] * np.linalg.norm(e), t[upd]
    return best, behind, s_m, t_x


def _sill_coverage(reference: np.ndarray, p0: np.ndarray, t_hat: np.ndarray, n_hat: np.ndarray, s_lo_m: float,
                   s_hi_m: float, floor_y: float, up_axis: int, plan_axes: list[int], band=(0.1, 0.5),
                   cell: float = 0.05, near_m: float = 0.08) -> float:
    """Fraction of the (along-wall x height) cells in the low band under a see-through section that contain observed
    wall surface. A window has a sill (wall below it, coverage high); a doorway does not (coverage ~0)."""
    uv = reference[:, plan_axes]
    d = (uv - p0) @ n_hat
    s = (uv - p0) @ t_hat
    h = reference[:, up_axis] - floor_y
    m = (np.abs(d) < near_m) & (s >= s_lo_m) & (s < s_hi_m) & (h >= band[0]) & (h < band[1])
    ns = max(1, int(np.ceil((s_hi_m - s_lo_m) / cell)))
    nh = max(1, int(np.ceil((band[1] - band[0]) / cell)))
    if not m.any():
        return 0.0
    ii = np.clip(((s[m] - s_lo_m) / cell).astype(int), 0, ns - 1)
    jj = np.clip(((h[m] - band[0]) / cell).astype(int), 0, nh - 1)
    occ = np.zeros((ns, nh), bool)
    occ[ii, jj] = True
    return float(occ.mean())


def classify_wall_gaps(points: np.ndarray, origins: np.ndarray, walls: Sequence, reference: np.ndarray,
                       floor_y: float, up_axis: int = 1, bin_m: float = 0.2, min_pts: int = 200,
                       mirror_match: float = 0.5, sill_min_coverage: float = 0.6) -> list[SurfaceFlag]:
    """Group points seen behind each wall into along-wall windows and say what each window is.

    * mirror      reflected copy matches the room -> the wall is solid there; exclude the points.
    * glazing     see-through with a SILL: the wall surface is observed below the see-through section (0.1-0.5 m
                  above the floor) -> a window / fixed glass, not a doorway.
    * opening     no wall below it -> a doorway or passage (or a glass door; ambiguous, flagged).
    Why a sill test and not the height where rays cross: a phone at chest height rarely sends rays through the
    bottom of a doorway to a point within LiDAR range, so the lowest crossing is ~0.5 m even for doors (measured).
    The plan extractor decides with its own opening list; this gives it the evidence.
    """
    plan_axes = [i for i in range(3) if i != up_axis]
    uv, ouv = points[:, plan_axes], origins[:, plan_axes]
    widx, dist, s_all, t_all = points_behind_walls(uv, ouv, walls)
    y_cross = origins[:, up_axis] + t_all * (points[:, up_axis] - origins[:, up_axis])   # height where ray meets wall
    p0, p1, nn, ids = _wall_arrays(walls)
    flags = []
    for k in np.unique(widx[widx >= 0]):
        m = widx == k
        e = p1[k] - p0[k]
        Lk = np.linalg.norm(e)
        s = s_all[m]                    # where the rays cross the wall: the extent of the pane / opening
        bins = np.floor(s / bin_m).astype(int)
        cnt = np.bincount(bins - bins.min())
        occ = cnt >= max(10, min_pts // 10)
        lab, n = ndimage.label(occ)
        for c in range(1, n + 1):
            sel_b = np.flatnonzero(lab == c) + bins.min()
            sm = np.isin(bins, sel_b)
            if sm.sum() < min_pts:
                continue
            P = points[m][sm]
            nrm3 = np.zeros(3); nrm3[plan_axes[0]], nrm3[plan_axes[1]] = nn[k]
            d3 = -float(nrm3[plan_axes] @ p0[k])
            rc = reflection_consistency(P, nrm3, d3, reference, origins[m][sm])
            yc = y_cross[m][sm] - floor_y
            lo_h = float(np.percentile(yc, 2))
            sill_cov = _sill_coverage(reference, p0[k], e / Lk, nn[k], s_lo_m=float(sel_b.min() * bin_m),
                                      s_hi_m=float((sel_b.max() + 1) * bin_m), floor_y=floor_y, up_axis=up_axis,
                                      plan_axes=plan_axes)
            s_lo, s_hi = float(sel_b.min() * bin_m), float((sel_b.max() + 1) * bin_m)
            ev = {"wall_id": ids[k], "n_points": int(sm.sum()), "along_wall_m": [s_lo, s_hi],
                  "sill_coverage": sill_cov, "lowest_crossing_above_floor_m": lo_h, "highest_crossing_above_floor_m": float(np.percentile(yc, 98)),
                  **rc}
            where = {"wall_id": ids[k], "segment_uv": [p0[k] + e / Lk * s_lo, p0[k] + e / Lk * min(s_hi, Lk)]}
            if is_mirror(rc, mirror_match):
                flags.append(SurfaceFlag("mirror", rc["match_frac_reflected"], where, ev,
                                         [Handling.EXCLUDE_POINTS, Handling.PLANE_IS_WALL, Handling.WIDEN_INTERVAL],
                                         "Mirror on this wall: content behind it is a reflection; the wall is solid."))
            elif sill_cov >= sill_min_coverage:
                flags.append(SurfaceFlag("glazing", float(min(1.0, sm.sum() / 2000)), where, ev,
                                         [Handling.LABEL_GLASS, Handling.EXCLUDE_POINTS],
                                         "See-through section with a sill: a glazed window or screen, not a doorway."))
            else:
                flags.append(SurfaceFlag("opening_or_glass_door", float(min(1.0, sm.sum() / 2000)), where, ev,
                                         [Handling.WIDEN_INTERVAL],
                                         "See-through down to the floor: a doorway, or a glass door/screen."))
    return flags
