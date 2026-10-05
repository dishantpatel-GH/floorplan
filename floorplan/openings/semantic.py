"""Doors and windows from the segmenter, measured on the plan's walls (video and photo tiers).

Why: the plan step finds an opening only where the geometry has a gap: a free-space neck between two rooms, or rays
that crossed a wall plane. Phone captures without LiDAR rarely leave such a gap, so the own-house plans showed almost
no doors and no windows (user review, 5 Oct). The photo tier already runs SegFormer-B5 (ADE20K) for its wall masks;
ADE20K has "door", "double door", "screen door" and "windowpane".

Per view (a photo, or a video keyframe) with its metric depth, intrinsics and pose in the plan frame:
  1. The view is levelled on the floor plane it sees (video poses are tilted by up to 45 deg, level_view).
  2. Door / window pixels -> blobs; specks are dropped.
  3. A blob goes on the plan wall that holds most of its pixels: on the wall line, or behind it (a door: at most 1 m
     behind). The camera must be on the wall's room side.
  4. The wall's surface in this view = a line fitted to the view's own "wall" pixels near the blob.
  5. Each blob pixel gives a position along that line, where its ray crosses it. Pixels on the surface and behind it
     (seen through the opening) count; pixels in front of it (an open leaf, a cupboard, a curtain) do not. A blob
     mostly in front is an open leaf: its length in plan is the width, hinged at the wall (_leaf_interval).
  6. The interval = the longest run of 2 cm bins with support, plus half a pixel at each end. An end touching the
     image's left or right border is cut: it does not measure the opening's end.
Across views: intervals on parallel walls within 0.45 m that overlap are one opening. Width = median width of the views
that saw both ends (a pose error moves both ends of a view together); else each end = median of the views that saw it.
Rejected: doors that do not reach the floor or reach less than 1.45 m (cupboards; wardrobe fronts stand in front of
the wall and never get here), one-view openings without both ends and a large blob, openings with an end no view saw
in fewer than 4 views, widths outside the ranges in SemParams.
Into the plan: where the geometry already has an opening on that wall, the geometric one stays with its width and
becomes a door when the segmenter saw one there. The rest are added with "source": "segmentation".
Measured: docs/modules/openings.md.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from floorplan.model import Adjacency, Measurement, Opening, Plan, Wall

DOOR_LABELS = ("door", "double door", "screen door")
WINDOW_LABELS = ("windowpane",)
WALL_LABELS = ("wall",)
FLOOR_LABELS = ("floor", "rug")


@dataclass
class SemParams:
    min_blob_frac: float = 0.002        # blobs below this share of the image are specks
    assign_tol_m: float = 0.35          # a pixel within this of the plan wall line counts as on it (wall choice)
    min_assign_frac: float = 0.3        # the chosen wall must take at least this share of the blob's pixels
    wall_extent_margin_m: float = 0.3   # a pixel may cross the wall line this far beyond the wall's ends
    min_host_len_m: float = 0.25        # shorter walls (jogs) do not host openings
    camera_side_m: float = 0.05         # the camera must be this far on the wall's room side
    min_incidence_deg: float = 15.0     # rays more grazing than this to the wall give unstable positions
    wall_band_m: float = 0.5            # "wall" pixels this close to the plan wall give the surface offset ...
    wall_min_px: int = 60               # ... when there are at least this many of them
    door_beyond_m: float = 1.0          # door pixels beyond a wall belong to it up to a leaf's width behind it
    floor_min_px: int = 150             # floor pixels that give a view its own floor height (else the scene's)
    max_level_deg: float = 50.0         # a view is levelled on its floor by at most this much
    plane_tol_m: float = 0.12           # a pixel this close to the surface is on it
    bin_m: float = 0.02
    gap_m: float = 0.08                 # a run along the wall may skip this much without support
    bin_min_rel: float = 0.04           # a bin is supported if it holds this share of the fullest bin
    border_px: int = 3
    end_zone_m: float = 0.03            # pixels this close to an end decide whether the frame cut that end
    door_max_bottom_m: float = 0.35     # a door reaches the floor ...
    door_min_top_m: float = 1.45        # ... and this high (kitchen cupboards: 0.9 m; wall cupboards do not reach the
                                        # floor). Not 1.8 m: take1's entrance door tops measured 1.5-1.8 m on levelled
                                        # keyframes; the rest of the error is in the video depth and poses
    door_min_visible_m: float = 1.2     # ... or, cut by the frame at both, at least this much of it is seen
    min_visible_height_m: dict = field(default_factory=lambda: {"door": 0.6, "window": 0.3})   # per view, uncut:
                                        # flatter on its wall means seen at a grazing angle through the wrong wall
    join_gap_m: float = 0.3             # blobs on one wall this close in one view are one opening
    door_width_m: tuple = (0.5, 2.6)
    window_width_m: tuple = (0.3, 4.0)
    match_angle_deg: float = 10.0       # intervals on walls this parallel ...
    match_offset_m: float = 0.45        # ... and this close (both faces of one partition) can be one opening
    match_overlap: float = 0.3          # ... when they overlap by this share of the shorter one
    min_views: int = 2
    min_views_end_unseen: int = 4       # an opening with an end no view saw needs this many views (take1: a 3-view
                                        # "door" on the bedroom's window wall, left end never seen)
    single_view_min_frac: float = 0.01  # one view is enough only with both ends seen and a blob this big ...
    partial_min_frac: float = 2.0       # ... or with one end seen and a blob this big (off; photo tier: 0.05)
    lower_bound_min_m: float = 0.3      # an opening with an end never seen: its seen width (a lower bound) only has
                                        # to be this wide
    end_sigma_floor_m: float = 0.03     # segmentation edge (1-2 px of a 1/4-resolution mask) + depth noise
    geometry_overlap: float = 0.3       # a geometric opening overlapping this share is the same opening
    cut_end_extra_m: float = 0.45       # an end never seen: the opening may be this much wider than what was seen

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


@dataclass
class SemView:
    name: str
    sem: np.ndarray          # (h, w) ADE20K class ids at the depth resolution
    depth: np.ndarray        # (h, w) metres; 0 = no depth
    K: np.ndarray            # (3, 3) intrinsics at the depth resolution
    T: np.ndarray            # (4, 4) camera -> aligned 3-D frame (+y up); plan (u, v) = (x, z)


@dataclass
class WallLine:
    idx: int
    wall: Wall
    p0: np.ndarray
    e: np.ndarray            # unit direction p0 -> p1
    n: np.ndarray            # unit normal into the wall's room
    L: float


# ------------------------------------------------------------------------------------------------ loading views
def _resize_nearest(m: np.ndarray, h: int, w: int) -> np.ndarray:
    yi = np.minimum((np.arange(h) * m.shape[0] / h).astype(int), m.shape[0] - 1)
    xi = np.minimum((np.arange(w) * m.shape[1] / w).astype(int), m.shape[1] - 1)
    return m[yi][:, xi]


def _class_maps(npz: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    z = np.load(npz)
    labels = [str(x).strip() for x in z["names"]]
    return {str(k): z[f"m{i}"] for i, k in enumerate(z["keys"])}, labels


def photo_views(scene: dict, work: Path) -> tuple[list[SemView], list[str]]:
    """Photo tier: class maps from semantic_raw.npz, final metric depth (after link and scale corrections) from
    depth_final.npz, intrinsics from the view cache, poses = scene T_wc (camera -> aligned frame)."""
    work = Path(work)
    maps, labels = _class_maps(work / "semantic_raw.npz")
    dep = np.load(work / "depth_final.npz")
    _, views = pickle.loads((work / "views_cache.pkl").read_bytes())
    out = []
    for i, n in enumerate(scene["cam_names"]):
        n = str(n)
        key = n.replace("/", "__")
        if n not in maps or key not in dep.files or n not in views:
            continue
        D = dep[key].astype(np.float32)
        h, w = D.shape
        out.append(SemView(n, _resize_nearest(maps[n], h, w), D, np.asarray(views[n].K, float),
                           np.asarray(scene["T_wc"][i], float)))
    return out, labels


def video_keyframe_files(info: dict, work: Path, names) -> list[Path]:
    rot = int((info.get("rotation") or {}).get("final_rotation", 0))
    return [Path(work) / f"frames_r{rot}" / "sfm" / str(n) for n in names]


def video_views(scene: dict, info: dict, work: Path, sem_npz: Path) -> tuple[list[SemView], list[str]]:
    """Video tier: keyframe depth as the run predicted it (work/depth_<model>.npz) with only the flying-pixel filter
    (far pixels stay: a window shows the outside), times the segment's scale correction; poses = T_align @ T_wc of the
    keyframes, the same poses the fused scene was built from. Keyframes of untrusted segments are left out when the
    run left them out of the scene."""
    from floorplan.video.depth import clean_depth
    work = Path(work)
    params = info.get("params") or {}
    z = np.load(work / f"depth_{info.get('depth_model', params.get('depth_model', 'moge2'))}.npz")
    D_all, names, K = z["depth"], [str(n) for n in z["names"]], np.asarray(z["K"], float)
    maps, labels = _class_maps(sem_npz)
    kf = np.asarray(scene["kf"])
    T = np.einsum("ij,njk->nik", np.asarray(scene["T_align"], float), np.asarray(scene["T_wc"], float)[kf])
    use = np.ones(len(kf), bool)
    trusted = scene.get("kf_trusted")
    if trusted is not None and params.get("drop_untrusted_segments", True):
        trusted = np.asarray(trusted, bool)
        if trusted.any() and not trusted.all():
            use = trusted
    corr = {int(g["id"]): float(g.get("scale_correction") or 1.0)
            for g in (info.get("scale") or {}).get("segments", [])}
    seg = np.asarray(scene.get("kf_segment", np.zeros(len(kf))), int)
    edge = float(params.get("depth_edge_rel", 0.05))
    out = []
    for i, n in enumerate(names):
        if i >= len(kf) or not use[i] or n not in maps:
            continue
        d = clean_depth(D_all[i].astype(np.float32) * corr.get(int(seg[i]), 1.0), np.inf, edge)
        h, w = d.shape
        out.append(SemView(n, _resize_nearest(maps[n], h, w), d, K, T[i]))
    return out, labels


def segment_images(items: list[tuple[str, Path]], out_npz: Path, log=print, timeout_s: int = 1800) -> dict:
    """Run the segmenter (scripts/seg_walls.py in envs/seg, as the photo tier does) on (name, path) items."""
    from floorplan.photo.semantic import _seg_python, seg_hf_home
    out_npz = Path(out_npz)
    if out_npz.exists():
        try:
            if {n for n, _ in items} <= set(str(k) for k in np.load(out_npz)["keys"]):
                return dict(status="cached")
        except Exception:                                  # a broken cache is recomputed
            pass
    py = _seg_python()
    script = Path(__file__).resolve().parents[2] / "scripts" / "seg_walls.py"
    if py is None or not script.exists():
        return dict(status="no segmentation env")
    lst = out_npz.with_suffix(".txt")
    lst.write_text("".join(f"{n}\t{Path(p).resolve()}\n" for n, p in items))
    env = dict(os.environ)
    env["HF_HOME"] = str(seg_hf_home(py))
    t0 = time.time()
    try:
        r = subprocess.run([str(py), str(script), "--list", str(lst), "--out", str(out_npz)], env=env,
                           capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return dict(status="timed out")
    if r.returncode != 0 or not out_npz.exists():
        return dict(status="failed", stderr=r.stderr[-500:])
    log(f"[openings] segmented {len(items)} images in {time.time() - t0:.0f} s")
    return dict(status="ok", seconds=round(time.time() - t0, 1))


# ------------------------------------------------------------------------------------------------ one view
def _wall_lines(plan: Plan) -> list[WallLine]:
    out = []
    for i, w in enumerate(plan.walls):
        p0, p1 = np.asarray(w.p0, float), np.asarray(w.p1, float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 1e-6:
            continue
        n = np.asarray(w.normal, float)
        out.append(WallLine(i, w, p0, (p1 - p0) / L, n / max(np.linalg.norm(n), 1e-9), L))
    return out


def _rays(v: SemView, vv: np.ndarray, uu: np.ndarray) -> np.ndarray:
    K = v.K
    rc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones(len(uu))], 1)
    return rc @ v.T[:3, :3].T                     # direction in the aligned frame, scaled so that P = C + depth * r


def _cross(C: np.ndarray, r: np.ndarray, d: np.ndarray, wl: WallLine, delta: float = 0.0):
    """Ray crossings with the vertical plane `delta` in front of the wall line: lam (ray parameter, = depth units),
    t (position along the wall), s_p (signed distance of the pixel's 3-D point from that plane; + = room side),
    incidence (sine of the angle between ray and plane), s_c (camera's distance)."""
    Cxz, rxz = C[[0, 2]], r[:, [0, 2]]
    s_c = float((Cxz - wl.p0) @ wl.n) - delta
    den = rxz @ wl.n
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(den < -1e-6, -s_c / den, -1.0)
    X = Cxz[None] + lam[:, None] * rxz
    t = (X - wl.p0) @ wl.e
    s_p = s_c + d * den
    inc = -den / np.maximum(np.linalg.norm(r, axis=1), 1e-9)
    return lam, t, s_p, inc, s_c


def _longest_run(t: np.ndarray, p: SemParams) -> np.ndarray:
    """Mask of the positions in the best-supported run of 2 cm bins (gaps up to gap_m allowed)."""
    lo = np.floor(t.min() / p.bin_m)
    b = (np.floor(t / p.bin_m) - lo).astype(int)
    cnt = np.bincount(b)
    ok = cnt >= max(2, p.bin_min_rel * cnt.max())
    gap = int(round(p.gap_m / p.bin_m))
    runs, start, last = [], None, None
    for i in np.flatnonzero(ok):
        if start is None:
            start = last = i
        elif i - last > gap + 1:
            runs.append((start, last))
            start = last = i
        else:
            last = i
    if start is not None:
        runs.append((start, last))
    if not runs:
        return np.zeros(len(t), bool)
    a, z = max(runs, key=lambda r: cnt[r[0]:r[1] + 1].sum())
    return (b >= a) & (b <= z)


def _local_line(wl: WallLine, wall_pts: np.ndarray, floor_y: float, t_lo: float, t_hi: float,
                p: SemParams) -> tuple[WallLine, float, str]:
    """The wall's surface as this view sees it: a line fitted to the view's own wall pixels within 1 m of the blob
    (and within wall_band_m of the plan wall). A view whose yaw is off by 20 deg sees a door at 20 deg to the plan
    wall; measured along its own wall the width does not shrink with the cosine. The local line keeps the plan wall's
    origin (projected onto it) so positions stay comparable. Too few pixels: the plan line, moved to their median
    offset (or not moved). A fit more than 35 deg off the plan wall: the plan direction."""
    q = wall_pts[:, [0, 2]] - wl.p0
    sw, tw, hw = q @ wl.n, q @ wl.e, wall_pts[:, 1] - floor_y
    sel = (np.abs(sw) <= p.wall_band_m) & (tw >= t_lo - 1.0) & (tw <= t_hi + 1.0) & (hw > 0.3) & (hw < 2.2)
    if sel.sum() < p.wall_min_px:
        return wl, 0.0, "plan line"
    X = wall_pts[sel][:, [0, 2]]
    keep = np.ones(len(X), bool)
    for _ in range(2):
        c = X[keep].mean(0)
        e = np.linalg.svd(X[keep] - c, full_matrices=False)[2][0]
        n = np.array([-e[1], e[0]])
        res = np.abs((X - c) @ n)
        keep = res < max(0.04, 2.5 * 1.4826 * float(np.median(res[keep])))
        if keep.sum() < p.wall_min_px / 2:
            break
    e = e if e @ wl.e >= 0 else -e
    if e @ wl.e < np.cos(np.radians(35)) or keep.sum() < p.wall_min_px / 2:
        return wl, float(np.median(sw[sel])), "plan direction, offset from the view's wall pixels"
    n = np.array([-e[1], e[0]])
    n = n if n @ wl.n >= 0 else -n
    c = X[keep].mean(0)
    p0 = c + e * float((wl.p0 - c) @ e)
    return WallLine(wl.idx, wl.wall, p0, e, n, wl.L), 0.0, "fitted to the view's wall pixels"


def _see_through(v: SemView, wl: WallLine, delta: float, floor_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Positions along the wall where the view looks THROUGH its surface (pixels of any class 0.3 m or more beyond
    it, crossing it 0.3-1.8 m above the floor): the doorway beside a hinge."""
    h, w = v.sem.shape
    gv, gu = np.mgrid[0:h:3, 0:w:3]
    gv, gu = gv.ravel(), gu.ravel()
    d = v.depth[gv, gu].astype(float)
    r = _rays(v, gv, gu)
    C = v.T[:3, 3]
    lam, t, s_p, inc, _ = _cross(C, r, d, wl, delta)
    hx = C[1] + lam * r[:, 1] - floor_y
    sel = (d > 0) & (lam > 0) & (s_p < -0.3) & (hx > 0.3) & (hx < 1.8) & (inc > 0.2)
    return t[sel], gu[sel]


def _leaf_interval(P: np.ndarray, uu: np.ndarray, vv: np.ndarray, v: SemView, wl: WallLine, delta: float,
                   floor_y: float, w: int, h: int, p: SemParams) -> tuple[dict | None, str]:
    """An open door leaf in the room: a thin vertical slab hinged at the wall. Its length in plan is the door width;
    the hinge is the end nearest the wall; the doorway lies on the side where the view sees through the wall (else
    on the side the free end points to)."""
    hgt = P[:, 1] - floor_y
    keep = (hgt > 0.15) & (hgt < 1.95)
    if keep.sum() < 50:
        return None, "open leaf: too few pixels between 0.15 and 1.95 m"
    Q, uu_k = P[keep][:, [0, 2]], uu[keep]
    c = Q.mean(0)
    ev, evec = np.linalg.eigh(np.cov((Q - c).T))
    a = evec[:, 1]
    if np.sqrt(max(ev[0], 0.0)) > 0.08:
        return None, f"open leaf: not a thin slab in plan ({np.sqrt(ev[0]):.2f} m across)"
    q = (Q - c) @ a
    q0, q1 = np.quantile(q, [0.01, 0.99])
    E = [c + q0 * a, c + q1 * a]
    s = [float((x - wl.p0) @ wl.n) - delta for x in E]
    ang = np.degrees(np.arccos(min(1.0, abs(float(a @ wl.e)))))
    if ang < 25:
        return None, f"flat front {min(s):.2f} m in front of the wall: cupboard or wardrobe"
    ih = int(np.argmin(s))
    if s[ih] > 0.3:
        return None, f"open leaf not hinged on the wall (nearest end {s[ih]:.2f} m from it)"
    t_h, t_f = (float((E[ih] - wl.p0) @ wl.e), float((E[1 - ih] - wl.p0) @ wl.e))
    length = float(q1 - q0)
    tt, _ = _see_through(v, wl, delta, floor_y)
    n_pos = int(((tt > t_h) & (tt < t_h + length)).sum())
    n_neg = int(((tt < t_h) & (tt > t_h - length)).sum())
    if max(n_pos, n_neg) >= 20 and max(n_pos, n_neg) >= 2 * max(min(n_pos, n_neg), 1):
        side, how = (1.0 if n_pos > n_neg else -1.0), "see-through"
    elif abs(t_f - t_h) >= 0.1:
        side, how = float(np.sign(t_f - t_h)), "free end"
    else:
        return None, "open leaf at 90 deg with no see-through: side of the doorway unknown"
    lr = (uu_k <= p.border_px) | (uu_k >= w - 1 - p.border_px)
    hinge_cut = bool(lr[(q <= q0 + p.end_zone_m) if ih == 0 else (q >= q1 - p.end_zone_m)].any())
    free_cut = bool(lr[(q >= q1 - p.end_zone_m) if ih == 0 else (q <= q0 + p.end_zone_m)].any())
    t_far = t_h + side * length
    lo, hi = sorted([t_h, t_far])
    cut_lo, cut_hi = (hinge_cut, free_cut) if side > 0 else (free_cut, hinge_cut)
    hk = hgt[keep]
    return dict(t0=lo, t1=hi, cut0=cut_lo, cut1=cut_hi, bottom=float(np.quantile(hk, 0.02)),
                top=float(np.quantile(P[:, 1] - floor_y, 0.98)),
                bottom_cut=bool((vv >= h - 1 - p.border_px).any()), top_cut=bool((vv <= p.border_px).any()),
                used_px=int(keep.sum()), leaf=dict(angle_deg=round(float(ang), 1), side_from=how,
                                                   hinge_offset_m=round(s[ih], 3))), ""


def _rot_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def level_view(v: SemView, floor_ids: np.ndarray, floor_y: float, p: SemParams) -> tuple[SemView, float, dict]:
    """Level one view on the floor it sees, and take its floor height from there.

    Why: video keyframe poses are tilted. On take1 the floor plane in a keyframe's own depth and the up axis of its
    pose differ by 15-45 deg (after/r1) and 5-30 deg (before/r2), while that floor gives plausible camera heights
    (1.0-1.7 m) and pitches (6-37 deg down). Untouched, heights on a wall 3 m away were 0.5 m off and an entrance door's
    top came out at 1.5 m, like a cupboard's. The floor pixels (within 3.5 m) give a plane; the view is turned about its
    camera centre so that plane is level, and its floor height is the plane's. Tilts above max_level_deg are taken
    as a bad fit (a wall labelled floor). Without floor pixels the view is used as it is, with the scene's floor."""
    h, w = v.sem.shape
    gv, gu = np.mgrid[0:h:3, 0:w:3]
    m = np.isin(v.sem[gv, gu], floor_ids) & (v.depth[gv, gu] > 0) & (v.depth[gv, gu] < 3.5)
    if m.sum() < p.floor_min_px:
        return v, floor_y, dict(levelled=False, floor_px=int(m.sum()))
    C = v.T[:3, 3]
    P = C[None] + v.depth[gv[m], gu[m]][:, None].astype(float) * _rays(v, gv[m], gu[m])
    keep = np.ones(len(P), bool)
    for _ in range(3):
        c = P[keep].mean(0)
        n = np.linalg.svd(P[keep] - c, full_matrices=False)[2][2]
        n = n if n[1] > 0 else -n
        res = (P - c) @ n
        keep = np.abs(res) < max(0.03, 2.5 * 1.4826 * float(np.median(np.abs(res[keep]))))
        if keep.sum() < p.floor_min_px / 2:
            return v, floor_y, dict(levelled=False, floor_px=int(m.sum()), reason="no floor plane")
    tilt = float(np.degrees(np.arccos(min(1.0, n[1]))))
    if tilt > p.max_level_deg:
        return v, floor_y, dict(levelled=False, floor_px=int(m.sum()), tilt_deg=round(tilt, 1))
    R = _rot_to(n, np.array([0.0, 1.0, 0.0]))
    T = v.T.copy()
    T[:3, :3] = R @ v.T[:3, :3]
    y = float(np.median((C[None] + (P[keep] - C) @ R.T)[:, 1]))
    return SemView(v.name, v.sem, v.depth, v.K, T), y, dict(levelled=True, tilt_deg=round(tilt, 1),
                                                            floor_px=int(m.sum()))


def view_intervals(v: SemView, walls: list[WallLine], ids: dict[str, np.ndarray], wall_ids: np.ndarray,
                   floor_y: float, p: SemParams, floor_ids: np.ndarray | None = None) -> tuple[list[dict], list[dict]]:
    """Door / window intervals of one view on the plan's walls, and the blobs that were left out (with why)."""
    h, w = v.sem.shape
    C = v.T[:3, 3]
    hosts = [wl for wl in walls if wl.L >= p.min_host_len_m]
    sin_min = np.sin(np.radians(p.min_incidence_deg))
    found, dropped = [], []
    wall_pts = None
    if floor_ids is not None and len(floor_ids):
        v, floor_y, _ = level_view(v, floor_ids, floor_y, p)
        C = v.T[:3, 3]
    for kind, cls in ids.items():
        mask = np.isin(v.sem, cls).astype(np.uint8)
        min_px = p.min_blob_frac * h * w
        if mask.sum() < min_px:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        nb, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for b in range(1, nb):
            if stats[b, cv2.CC_STAT_AREA] < min_px:
                continue
            vv, uu = np.nonzero(lab == b)
            d = v.depth[vv, uu].astype(float)
            r = _rays(v, vv, uu)
            usable = d > 0 if kind == "door" else np.ones(len(d), bool)   # glass / outside may have no depth
            rec = dict(view=v.name, kind=kind, px=int(len(vv)), frac=round(len(vv) / (h * w), 4))
            if usable.sum() < min_px:
                dropped.append(dict(rec, reason="no depth on the blob"))
                continue
            # wall choice on every 2nd pixel: pixels on the wall line count 1, pixels beyond it 0.5 (a door: only
            # up to a leaf's width beyond it; farther, the door is on some other wall seen through this one)
            sub = np.flatnonzero(usable)[::2]
            best, best_score = None, 0.0
            for wl in hosts:
                lam, t, s_p, inc, s_c = _cross(C, r[sub], d[sub], wl)
                if s_c < p.camera_side_m:
                    continue
                inside = (lam > 0) & (t >= -p.wall_extent_margin_m) & (t <= wl.L + p.wall_extent_margin_m) & \
                         (inc >= sin_min)
                has_d = d[sub] > 0
                on = inside & has_d & (np.abs(s_p) <= p.assign_tol_m)
                thru = inside & ((has_d & (s_p < -p.assign_tol_m)) | ~has_d)
                if kind == "door":
                    thru &= has_d & (s_p >= -p.door_beyond_m)
                score = on.sum() + 0.5 * thru.sum()
                if score > best_score:
                    best, best_score = wl, score
            if best is None or best_score < p.min_assign_frac * len(sub):
                dropped.append(dict(rec, reason="on no plan wall"))
                continue
            # the wall's surface in this view, from its own "wall" pixels
            if wall_pts is None:
                gv, gu = np.mgrid[0:h:4, 0:w:4]
                m = np.isin(v.sem[gv, gu], wall_ids) & (v.depth[gv, gu] > 0)
                gv, gu = gv[m], gu[m]
                wall_pts = C[None] + v.depth[gv, gu][:, None].astype(float) * _rays(v, gv, gu)
            lam, t, s_p, inc, s_c = _cross(C, r, d, best)
            near = (d > 0) & (lam > 0) & (np.abs(s_p) <= p.assign_tol_m)
            tt = t[near] if near.sum() >= 10 else t[lam > 0]
            plan_dist = float(s_c)                      # camera to the plan wall
            best, delta, how = _local_line(best, wall_pts, floor_y, float(np.min(tt)), float(np.max(tt)), p)
            rec["surface"] = how
            lam, t, s_p, inc, s_c = _cross(C, r, d, best, delta)
            has_d = d > 0
            ok = usable & (lam > 0) & (inc >= sin_min) & \
                ((has_d & (s_p <= p.plane_tol_m)) | (~has_d))             # on the surface or beyond it
            front = usable & has_d & (s_p > p.plane_tol_m)
            n_front = int(front.sum())
            if kind == "door" and n_front > max(ok.sum(), min_px):
                # mostly an open leaf standing in the room: measure the leaf, hinged on the wall
                P = C[None] + d[front, None] * r[front]
                leaf, why = _leaf_interval(P, uu[front], vv[front], v, best, delta, floor_y, w, h, p)
                if leaf is None:
                    dropped.append(dict(rec, reason=why, wall=best.wall.id, front_px=n_front))
                else:
                    found.append(dict(rec, wall=best.idx, wall_id=best.wall.id, delta=round(delta, 3),
                                      front_px=n_front, cam_dist=round(float(s_c + delta), 2), **leaf))
                continue
            if ok.sum() < min_px:
                dropped.append(dict(rec, reason="in front of the wall (open leaf, cupboard, curtain)",
                                    wall=best.wall.id, front_px=n_front))
                continue
            t, lam_ok, uu_ok, vv_ok = t[ok], lam[ok], uu[ok], vv[ok]
            hgt = C[1] + lam_ok * r[ok, 1] - floor_y
            foot = lam_ok * np.linalg.norm(r[ok], axis=1) / (v.K[0, 0] * np.maximum(inc[ok], 0.2))
            run = _longest_run(t, p)
            t, hgt, uu_ok, vv_ok, foot = t[run], hgt[run], uu_ok[run], vv_ok[run], foot[run]
            # pixel centres stop half a pixel short of the blob's edge: add that half pixel's footprint on the wall
            # (1 cm at 3 m for a 300 px focal; measured on the ray-cast test: 0.79 m for a 0.80 m door without it)
            half = 0.5 * float(np.median(foot))
            t0, t1 = float(np.quantile(t, 0.002)) - half, float(np.quantile(t, 0.998)) + half
            lr = (uu_ok <= p.border_px) | (uu_ok >= w - 1 - p.border_px)
            bottom, top = float(np.quantile(hgt, 0.02)), float(np.quantile(hgt, 0.98))
            b_cut, t_cut = bool((vv_ok >= h - 1 - p.border_px).any()), bool((vv_ok <= p.border_px).any())
            tall = p.min_visible_height_m[kind]
            if top - bottom < tall and not (b_cut or t_cut):
                # a blob squashed flat on this wall was put on the wrong wall: seen at a grazing angle through it
                # (take1 before r2: the bedroom window, 0.05-0.3 m tall on a wall it is not on)
                dropped.append(dict(rec, reason=f"only {top - bottom:.2f} m tall on {best.wall.id}: wrong wall",
                                    wall=best.wall.id))
                continue
            found.append(dict(
                rec, wall=best.idx, wall_id=best.wall.id, t0=t0, t1=t1,
                cut0=bool(lr[t <= t0 + p.end_zone_m].any()), cut1=bool(lr[t >= t1 - p.end_zone_m].any()),
                bottom=bottom, top=top, bottom_cut=b_cut, top_cut=t_cut,
                delta=round(delta, 3), used_px=int(len(t)), front_px=n_front, cam_dist=round(float(s_c + delta), 3),
                plan_dist=round(plan_dist, 3)))
    return found, dropped


# ------------------------------------------------------------------------------------------------ all views
def _clusters(items: list[dict], walls: dict[int, WallLine], p: SemParams) -> list[list[int]]:
    """Union-find over view intervals of one kind: parallel walls within match_offset_m, overlapping intervals."""
    par = list(range(len(items)))

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    cos_min = np.cos(np.radians(p.match_angle_deg))
    ends = []
    for it in items:
        wl = walls[it["wall"]]
        ends.append((wl.p0 + it["t0"] * wl.e, wl.p0 + it["t1"] * wl.e))
    for i, a in enumerate(items):
        wa = walls[a["wall"]]
        for j in range(i + 1, len(items)):
            wb = walls[items[j]["wall"]]
            if abs(float(wa.e @ wb.e)) < cos_min or abs(float((wb.p0 - wa.p0) @ wa.n)) > p.match_offset_m:
                continue
            ta = sorted(float((x - wa.p0) @ wa.e) for x in ends[i])
            tb = sorted(float((x - wa.p0) @ wa.e) for x in ends[j])
            ov = min(ta[1], tb[1]) - max(ta[0], tb[0])
            if ov > p.match_overlap * min(ta[1] - ta[0], tb[1] - tb[0]):
                par[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _robust_sigma(x: list[float], floor: float) -> float:
    if len(x) < 2:
        return floor
    return max(floor, 1.4826 * float(np.median(np.abs(np.asarray(x) - np.median(x)))))


def join_split_blobs(items: list[dict], p: SemParams) -> list[dict]:
    """In one view, blobs of one kind on one wall with a gap of at most join_gap_m are one opening: the two leaves of
    a double door, a grille door, a window with a mullion (take1's entrance: 0.64 + 0.27 m blobs, 0.22 m apart)."""
    out, groups = [], {}
    for it in items:
        groups.setdefault((it["view"], it["kind"], it["wall"]), []).append(it)
    for g in groups.values():
        g = sorted(g, key=lambda it: it["t0"])
        cur = dict(g[0])
        for it in g[1:]:
            if it["t0"] - cur["t1"] <= p.join_gap_m:
                if it["t1"] > cur["t1"]:
                    cur["t1"], cur["cut1"] = it["t1"], it["cut1"]
                for k in ("px", "used_px", "front_px"):
                    cur[k] = cur.get(k, 0) + it.get(k, 0)
                cur["frac"] = cur["frac"] + it["frac"]
                cur["bottom"], cur["top"] = min(cur["bottom"], it["bottom"]), max(cur["top"], it["top"])
                cur["bottom_cut"] |= it["bottom_cut"]
                cur["top_cut"] |= it["top_cut"]
                cur["joined"] = cur.get("joined", 1) + 1
            else:
                out.append(cur)
                cur = dict(it)
        out.append(cur)
    return out


def merge_views(items: list[dict], walls: dict[int, WallLine], p: SemParams) -> tuple[list[dict], list[dict]]:
    """One record per opening (host wall, interval, width with its interval, views, heights), and the rejects."""
    kept, rejected = [], []
    for kind in ("door", "window"):
        its = [it for it in items if it["kind"] == kind]
        for g in _clusters(its, walls, p):
            mem = [its[i] for i in g]
            votes: dict[int, float] = {}
            for m in mem:
                votes[m["wall"]] = votes.get(m["wall"], 0.0) + m["used_px"]
            host = walls[max(votes, key=votes.get)]
            lo_obs, hi_obs, lo_all, hi_all, whole = [], [], [], [], []
            for m in mem:
                wl = walls[m["wall"]]
                a = float((wl.p0 + m["t0"] * wl.e - host.p0) @ host.e)
                b = float((wl.p0 + m["t1"] * wl.e - host.p0) @ host.e)
                ca, cb = m["cut0"], m["cut1"]
                if a > b:                                  # that wall runs the other way
                    a, b, ca, cb = b, a, cb, ca
                lo_all.append(a)
                hi_all.append(b)
                if not ca:
                    lo_obs.append(a)
                if not cb:
                    hi_obs.append(b)
                if not ca and not cb:
                    whole.append((a, b))
            views = sorted({m["view"] for m in mem})
            if len(whole) >= 2:
                # Width = median of the widths of views that saw both ends; centre = median of their centres. A view's
                # pose error shifts both of its ends together (take1: the same door 0.5 m apart early and late in the
                # walk), so ends pooled over views would mix shifts into the width.
                wd = float(np.median([b - a for a, b in whole]))
                c = float(np.median([(a + b) / 2 for a, b in whole]))
                lo, hi = c - wd / 2, c + wd / 2
                sig = _robust_sigma([b - a for a, b in whole], p.end_sigma_floor_m * np.sqrt(2)) / np.sqrt(2)
                sig_ends = [sig, sig]
            else:
                lo = float(np.median(lo_obs)) if lo_obs else float(min(lo_all))
                hi = float(np.median(hi_obs)) if hi_obs else float(max(hi_all))
                sig_ends = [_robust_sigma(lo_obs, p.end_sigma_floor_m), _robust_sigma(hi_obs, p.end_sigma_floor_m)]
            bottoms = [m["bottom"] for m in mem if not m["bottom_cut"]]
            tops = [m["top"] for m in mem if not m["top_cut"]]
            rec = dict(kind=kind, host=host.idx, host_id=host.wall.id, room_id=host.wall.room_id, t0=lo, t1=hi,
                       width=hi - lo, views=len(views), view_names=views[:12], ends_seen=[len(lo_obs), len(hi_obs)],
                       whole_views=len(whole), sigma_ends=sig_ends,
                       bottom=float(np.quantile(bottoms, 0.25)) if bottoms else None,
                       top=float(np.quantile(tops, 0.75)) if tops else None,
                       px=int(sum(m["px"] for m in mem)), max_frac=max(m["frac"] for m in mem),
                       walls_voted={walls[k].wall.id: int(v) for k, v in votes.items()})
            why = None
            wmin, wmax = p.door_width_m if kind == "door" else p.window_width_m
            both = bool(lo_obs and hi_obs)
            # a large blob with one end seen (a photo taken close to a door): the opening is there, its width is a
            # lower bound (drawn as low confidence; photo tier only, see params_for)
            big_partial = not both and bool(lo_obs or hi_obs) and rec["max_frac"] >= p.partial_min_frac
            if not both:
                wmin = min(wmin, p.lower_bound_min_m)
            if rec["width"] < wmin or rec["width"] > wmax:
                why = f"width {rec['width']:.2f} m outside {wmin}-{wmax} m"
            elif kind == "door" and rec["bottom"] is not None and rec["bottom"] > p.door_max_bottom_m:
                why = f"does not reach the floor (bottom {rec['bottom']:.2f} m): cupboard or furniture door"
            elif kind == "door" and rec["top"] is not None and rec["top"] < p.door_min_top_m:
                why = f"too low for a door (top {rec['top']:.2f} m): cupboard or furniture door"
            elif kind == "door" and rec["bottom"] is None and rec["top"] is None and \
                    max(m["top"] - m["bottom"] for m in mem) < p.door_min_visible_m:
                why = "frame cut it at the top and the bottom, and less than 1.2 m of it is seen"
            elif len(views) < p.min_views and not (both and rec["max_frac"] >= p.single_view_min_frac) \
                    and not big_partial:
                why = f"seen in {len(views)} view(s) only, without both ends or with a small blob"
            elif not both and len(views) < p.min_views_end_unseen and not big_partial:
                why = (f"one end never seen (cut by the frame in all {len(views)} views): needs "
                       f"{p.min_views_end_unseen} views")
            elif (hi + lo) / 2 < -p.wall_extent_margin_m or (hi + lo) / 2 > host.L + p.wall_extent_margin_m:
                why = "centre beyond the ends of its wall"
            (rejected if why else kept).append(dict(rec, reason=why) if why else rec)
    return kept, rejected


# ------------------------------------------------------------------------------------------------ into the plan
def _interval_on(o: Opening, host: WallLine, walls_by_id: dict[str, WallLine], p: SemParams):
    """A plan opening's interval along `host`, if one of its walls (or its centre) lies on host's line."""
    c = np.asarray(o.center, float)
    on_line = [walls_by_id[i] for i in o.wall_ids if i in walls_by_id]
    if on_line and not any(abs(float(wl.e @ host.e)) > 0.98 for wl in on_line):
        return None                                 # its walls are not parallel to this one
    if abs(float((c - host.p0) @ host.n)) > p.match_offset_m:
        return None
    tc = float((c - host.p0) @ host.e)
    if o.width is None or o.width.value is None:    # no width (a doorway pair of photos): a 0.3 m catch zone
        return tc - 0.3, tc + 0.3
    half = o.width.value / 2
    return tc - half, tc + half


def add_to_plan(plan: Plan, kept: list[dict], walls: dict[int, WallLine], tier: str, p: SemParams) -> list[dict]:
    """Add the segmenter's openings to the plan (geometry wins where both exist); return what was done per opening."""
    walls_by_id = {wl.wall.id: wl for wl in walls.values()}
    cos_min = np.cos(np.radians(p.match_angle_deg))
    rooms = {r.id for r in plan.rooms}
    log, k = [], 1
    existing_ids = {o.id for o in plan.openings}
    for rec in sorted(kept, key=lambda r: (r["kind"], -r["views"])):
        host = walls[rec["host"]]
        lo, hi = rec["t0"], rec["t1"]
        same = None
        for o in plan.openings:
            if (o.kind == "window") != (rec["kind"] == "window"):
                continue
            iv = _interval_on(o, host, walls_by_id, p)
            if iv is None:
                continue
            ov = min(iv[1], hi) - max(iv[0], lo)
            if ov > p.geometry_overlap * max(min(iv[1] - iv[0], hi - lo), 1e-6):
                same = o
                break
        sig = float(np.hypot(*rec["sigma_ends"]))
        how = (f"segmenter ({tier}): {rec['kind']} pixels of {rec['views']} view(s) on the wall surface or seen "
               f"through it; ends = median of the views that saw them ({rec['ends_seen'][0]} / {rec['ends_seen'][1]})")
        both = rec["ends_seen"][0] > 0 and rec["ends_seen"][1] > 0
        if not both:          # a lower bound; symmetric so that the tier widening (also symmetric) keeps it covered
            sig = max(sig, p.cut_end_extra_m / 1.96)
        width = Measurement.from_sigma(rec["width"], sig, method=how, status="measured" if both else "inferred")
        if not both:
            width.method += "; an end was cut by the image frame in every view: the width is a lower bound"
        if same is not None:
            has_w = same.width is not None and same.width.value is not None
            note = (f"; segmenter: {rec['kind']} seen in {rec['views']} view(s), width {rec['width']:.2f} m"
                    + (" (geometry width kept)" if has_w else " (the geometry had no width: the segmenter's is used)"))
            if not has_w:
                same.width = width
                centre = host.p0 + host.e * (lo + hi) / 2
                same.center = (float(centre[0]), float(centre[1]))
            if same.kind == "passage" and rec["kind"] == "door":
                same.kind = "door"
                note += "; kind passage -> door (a door leaf was seen)"
            same.evidence = (same.evidence or "") + note
            log.append(dict(rec, action="matched geometric opening" + ("" if has_w else ", width from segmenter"),
                            opening_id=same.id))
            continue
        wall_ids, room_ids = [host.wall.id], [host.wall.room_id]
        if rec["kind"] == "door":                       # the same partition seen from the neighbouring room
            for wl in walls.values():
                if wl.wall.id in wall_ids or abs(float(wl.e @ host.e)) < cos_min:
                    continue
                if abs(float((wl.p0 - host.p0) @ host.n)) > p.match_offset_m:
                    continue
                a, b = sorted(float((x - host.p0) @ host.e) for x in (wl.p0, wl.p0 + wl.L * wl.e))
                if min(b, hi) - max(a, lo) > 0.5 * (hi - lo):
                    wall_ids.append(wl.wall.id)
                    if wl.wall.room_id not in room_ids and wl.wall.room_id in rooms:
                        room_ids.append(wl.wall.room_id)
        while f"S{k}" in existing_ids:
            k += 1
        oid = f"S{k}"
        existing_ids.add(oid)
        centre = host.p0 + host.e * (lo + hi) / 2
        conf = min(0.9, 0.45 + 0.1 * rec["views"]) if rec["views"] >= p.min_views else 0.5
        if not both:
            conf = min(conf, 0.45)
        height = sill = None
        if rec["kind"] == "door" and rec["top"] is not None:
            height = Measurement.from_sigma(rec["top"], 0.05, method="top of the door pixels on the wall surface "
                                            "(75th percentile over views)", status="inferred")
        if rec["kind"] == "window" and rec["bottom"] is not None:
            sill = Measurement.from_sigma(max(rec["bottom"], 0.0), 0.05, method="bottom of the window pixels on the "
                                          "wall surface (25th percentile over views)", status="inferred")
        o = Opening(id=oid, kind=rec["kind"], wall_ids=wall_ids, room_ids=[r for r in room_ids if r in rooms],
                    center=(float(centre[0]), float(centre[1])), width=width, height=height, sill_height=sill,
                    confidence=round(conf, 2), source="segmentation",
                    evidence=(f"segmenter ({tier}): {rec['kind']} in {rec['views']} view(s), {rec['px']} pixels; "
                              f"host {host.wall.id}; ends seen in {rec['ends_seen'][0]} / {rec['ends_seen'][1]} views"))
        plan.openings.append(o)
        for wid in wall_ids:
            wl = walls_by_id.get(wid)
            if wl is not None and oid not in wl.wall.opening_ids:
                wl.wall.opening_ids.append(oid)
        if len(o.room_ids) == 2:
            pair = set(o.room_ids)
            if not any({a.room_a, a.room_b} == pair for a in plan.adjacency):
                plan.adjacency.append(Adjacency(o.room_ids[0], o.room_ids[1], oid))
        log.append(dict(rec, action="added", opening_id=oid))
    return log


def confirm_doors(plan: Plan, rejected: list[dict], walls: dict[int, WallLine], p: SemParams) -> list[dict]:
    """A geometric passage where the segmenter saw door pixels in 2+ views is a door, even when those views saw only
    part of it (walking through a doorway, the leaf is seen close up and cut by the frame: take1's bedroom door gave
    0.2-0.5 m pieces). The geometric width stays."""
    walls_by_id = {wl.wall.id: wl for wl in walls.values()}
    out = []
    for rec in rejected:
        if rec["kind"] != "door" or rec["views"] < p.min_views or not str(rec.get("reason", "")).startswith("width"):
            continue
        host = walls[rec["host"]]
        for o in plan.openings:
            if o.kind != "passage" or o.source != "geometry":
                continue
            iv = _interval_on(o, host, walls_by_id, p)
            if iv is None:
                continue
            ov = min(iv[1], rec["t1"]) - max(iv[0], rec["t0"])
            if ov > p.geometry_overlap * max(rec["t1"] - rec["t0"], 1e-6):
                o.kind = "door"
                o.evidence = (o.evidence or "") + (f"; kind passage -> door: the segmenter saw door pixels here in "
                                                   f"{rec['views']} view(s) (pieces {rec['width']:.2f} m wide)")
                out.append(dict(rec, action="confirmed geometric passage as a door", opening_id=o.id))
                break
    return out


def detect_openings(views: list[SemView], labels: list[str], plan: Plan, floor_y: float,
                    p: SemParams | None = None) -> dict:
    p = p or SemParams()
    walls = {wl.idx: wl for wl in _wall_lines(plan)}
    idx = {n: i for i, n in enumerate(labels)}
    ids = {"door": np.array([idx[n] for n in DOOR_LABELS if n in idx]),
           "window": np.array([idx[n] for n in WINDOW_LABELS if n in idx])}
    wall_ids = np.array([idx[n] for n in WALL_LABELS if n in idx])
    floor_ids = np.array([idx[n] for n in FLOOR_LABELS if n in idx])
    items, dropped = [], []
    for v in views:
        f, d = view_intervals(v, list(walls.values()), ids, wall_ids, floor_y, p, floor_ids)
        items += f
        dropped += d
    items = join_split_blobs(items, p)
    kept, rejected = merge_views(items, walls, p)
    return dict(items=items, dropped_blobs=dropped, kept=kept, rejected=rejected, walls=walls)


def params_for(tier: str) -> SemParams:
    """Video keyframes come about 1 s apart, so a real opening is in several of them: 3 views, no one-view openings
    (take1: the one-view detections were pieces of windows already found, on other walls). A room's 6-8 photos
    often show an opening once: 2 views, or one large view with both ends, or a blob of 5% of the photo with one end
    seen (taken close to a door: the door is there, its width is a lower bound)."""
    if tier == "video":
        return SemParams(min_views=3, single_view_min_frac=2.0)
    return SemParams(partial_min_frac=0.05)


def add_semantic_openings(plan: Plan, scene: dict, info: dict, tier: str, work: Path, log=print,
                          p: SemParams | None = None, sem_npz: Path | None = None) -> dict:
    """The whole step for a run: views -> detections -> plan. Never raises: on any failure the plan is unchanged
    and the report says why."""
    p = p or params_for(tier)
    t0 = time.time()
    try:
        floor_y = info.get("floor_y")
        if floor_y is None:
            return dict(status="no floor height")
        if tier == "photo":
            views, labels = photo_views(scene, Path(work))
        elif tier == "video":
            work = Path(work)
            names = [str(n) for n in np.load(work / f"depth_{info.get('depth_model', 'moge2')}.npz")["names"]]
            sem_npz = Path(sem_npz) if sem_npz else work / "semantic_kf.npz"
            files = video_keyframe_files(info, work, names)
            seg = segment_images(list(zip(names, files)), sem_npz, log)
            if seg["status"] not in ("ok", "cached"):
                log(f"[openings] WARNING: keyframe segmentation {seg['status']}: no segmenter openings")
                return dict(status=f"segmentation {seg['status']}")
            views, labels = video_views(scene, info, work, sem_npz)
        else:
            return dict(status=f"tier {tier}: not used")
        det = detect_openings(views, labels, plan, float(floor_y), p)
        done = add_to_plan(plan, det["kept"], det["walls"], tier, p)
        done += confirm_doors(plan, det["rejected"], det["walls"], p)
        added = [d for d in done if d["action"] == "added"]
        rep = dict(status="ok", views=len(views), view_intervals=len(det["items"]),
                   blobs_left_out=len(det["dropped_blobs"]), added=len(added),
                   matched_geometry=len(done) - len(added), rejected=len(det["rejected"]),
                   doors=sum(o.kind == "door" for o in plan.openings),
                   windows=sum(o.kind == "window" for o in plan.openings),
                   seconds=round(time.time() - t0, 1), params=p.to_dict(),
                   openings=[_brief(d) for d in done], rejected_openings=[_brief(d) for d in det["rejected"]])
        plan.meta["semantic_openings"] = {k: v for k, v in rep.items() if k != "params"}
        log(f"[openings] segmenter: {len(views)} views, {len(det['items'])} wall intervals -> {len(added)} added, "
            f"{rep['matched_geometry']} on geometric openings, {rep['rejected']} rejected ({rep['seconds']} s)")
        return rep
    except Exception as e:                               # never lose the plan over an add-on
        log(f"[openings] WARNING: segmenter openings failed: {type(e).__name__}: {e}")
        return dict(status=f"failed: {type(e).__name__}: {e}")


def _brief(d: dict) -> dict:
    keys = ("opening_id", "action", "reason", "kind", "host_id", "room_id", "width", "t0", "t1", "views",
            "ends_seen", "sigma_ends", "bottom", "top", "px", "walls_voted")
    out = {}
    for k in keys:
        if k in d:
            v = d[k]
            out[k] = round(v, 3) if isinstance(v, float) else ([round(x, 3) for x in v] if isinstance(v, list) and v
                                                               and isinstance(v[0], float) else v)
    return out
