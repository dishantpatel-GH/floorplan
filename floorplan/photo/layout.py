"""v3: per-room layout from the room's own spin photos (docs/modules/photo_tier.md "photo_tier v3").

Panorama layout methods (LayoutNet, HorizonNet) estimate a room's box from ONE viewpoint looking all round. A D-017
spin is the same thing in 6-7 stills: the photos are taken from (nearly) one spot, so their metric depth maps can be
placed around a common centre with the spin rotations (Manhattan-snapped steps, protocol.spin_edges) without any
feature match. From that local cloud:

  1. wall points = horizontal normals (MoGe-2), in a height band above furniture and below the ceiling;
  2. the cloud is rotated onto its Manhattan axes;
  3. for each of the 4 sides, the wall offset is the 5 cm bin whose wall-facing points cover the largest wall AREA
     (distinct 10 x 10 cm (tangent, height) cells), not the most points: a far wall seen through a door covers a
     door-sized patch, a real wall most of the room's width. The offset is the median of the inliers;
     3b. D-077: the chosen surface can be furniture (a wardrobe front, an open door leaf) when the room's wall is
     seen BESIDE it, 0.5-1.5 m farther out, past the end of that surface and inside the room's extent: then that
     wall is the side (_far_wall, _far_wall_in_room);
  4. each photo that sees a side gives its own offset; the per-wall uncertainty is a Monte Carlo over the photo
     position (spin sway), per-photo depth scale and the surface noise, so walls seen by one photo are wider;
  5. a side no photo saw is INFERRED: at least as far as the observed extent of the neighbouring walls (a corner
     cannot be inside the room), else "photographer near the middle" (the opposite wall's distance), with a wide
     sigma, status inferred;
  6. room height = ceiling points minus floor points, when both are seen.

The layout lives in a local frame (centre of the spin = origin). frontend.py moves it into the common frame with the
pose of any placed spin photo and then into the aligned scene frame (T_align); plan_beta (tiers.layout_rooms) builds
the rooms from it.
"""
from __future__ import annotations

import numpy as np

from floorplan.photo.semantic import semantic_wall_points

from floorplan.photo.geometry import manhattan_yaw

SIDES = ("+x", "-x", "+z", "-z")


def _ry(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def local_poses(group: list[str], steps_deg: list[float]) -> dict[str, float]:
    """Spin photos -> yaw in the room's local frame (first photo = 0), from the Manhattan-snapped steps."""
    th = np.concatenate([[0.0], np.cumsum(np.radians(steps_deg))])
    return {n: float(t) for n, t in zip(group, th)}


def _photo_cloud(view, yaw: float, t: np.ndarray, scale: float, stride: int):
    Pc, idx = view.points_cam(stride)
    R = _ry(yaw) @ view.R_lev
    P = scale * Pc @ R.T + t
    N = view.normal_cam.reshape(-1, 3)[idx] @ R.T if view.normal_cam is not None else np.zeros_like(P)
    return P, N, idx % view.depth.shape[1]


def _farthest_per_column(P: np.ndarray, col: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the farthest point (horizontal range from the camera) in every image column.
    Why: in a column, the room's wall is the LAST vertical surface; furniture stands in front of it. Above furniture
    height the wall is usually visible, so the farthest band point of a column is the wall (or, through a door, the
    next room: a door-wide patch, outvoted by the wall's columns)."""
    if len(P) == 0:
        return np.zeros(0, int)
    r = np.hypot(P[:, 0] - t[0], P[:, 2] - t[2])
    order = np.lexsort((-r, col))
    first = np.r_[True, col[order][1:] != col[order][:-1]]
    return order[first]


def _side_axis(side: str) -> tuple[int, float]:
    return (0 if side[1] == "x" else 2), (1.0 if side[0] == "+" else -1.0)


def fit_room_layout(views: dict, yaws: dict[str, float], scale: float, p, offsets: dict | None = None) -> dict:
    """Manhattan rectangle of one room from its spin photos placed around a common centre.

    views: name -> PhotoView; yaws: name -> local yaw (rad); scale: global metric scale factor (fused);
    offsets: optional name -> (3,) local camera position (default 0: the spin centre).
    Returns a dict with the local Manhattan rotation, per-side offsets/sigmas/status, height and diagnostics."""
    rng = np.random.default_rng(p.seed)
    names = [n for n in yaws if n in views and views[n].normal_cam is not None]
    if not names:
        return dict(ok=False, reason="no spin photo with depth")
    floor_h = [views[n].floor_h * scale for n in names if views[n].floor_h is not None]
    cam_h = float(-np.median(floor_h)) if floor_h else p.layout_cam_height_m      # camera height above the floor
    per = {}
    sem_used = []
    allW, allN, allid = [], [], []
    geoW, geoN, geoid = [], [], []                 # D-060: the same points without the semantic mask
    ceil_y, floor_y = [], []
    for k, n in enumerate(names):
        t = np.zeros(3) if offsets is None else np.asarray(offsets.get(n, np.zeros(3)), float)
        P, N, col = _photo_cloud(views[n], yaws[n], t, scale, p.layout_stride)
        h = P[:, 1] + cam_h                                                     # height above the floor
        wall = (np.abs(N[:, 1]) < p.layout_wall_max_ny) & (h > p.layout_band_lo_m) & (h < p.layout_band_hi_m)
        # D-060: only points the segmenter labels WALL may define a side (wardrobes, cabinets, shower partitions,
        # mirrors and columns are vertical too); a photo with too few wall-labelled points keeps the geometric set
        sw = semantic_wall_points(views[n], p.layout_stride) if getattr(p, "semantic_walls", True) else None
        wall_geo = wall.copy()
        if sw is not None and int((wall & sw).sum()) >= p.semantic_min_pts:
            wall = wall & sw
            sem_used.append(n)
        wg = np.flatnonzero(wall_geo)
        if p.layout_farthest_per_column:
            wg = wg[_farthest_per_column(P[wg], col[wg], t)]
        geoW.append(P[wg]); geoN.append(N[wg]); geoid.append(np.full(len(wg), k))
        wi = np.flatnonzero(wall)
        if p.layout_farthest_per_column:
            wi = wi[_farthest_per_column(P[wi], col[wi], t)]
        allW.append(P[wi]); allN.append(N[wi]); allid.append(np.full(len(wi), k))
        ceil_y.append(P[(N[:, 1] < -0.85), 1])
        floor_y.append(P[(N[:, 1] > 0.85), 1])
        per[n] = int(wall.sum())
    W, NW, pid = np.concatenate(allW), np.concatenate(allN), np.concatenate(allid)
    GW, GN, gid = np.concatenate(geoW), np.concatenate(geoN), np.concatenate(geoid)
    if len(W) < p.layout_min_wall_pts and len(GW) >= p.layout_min_wall_pts:
        W, NW, pid = GW, GN, gid                   # too few wall-labelled points in the whole room: geometry only
    if len(W) < p.layout_min_wall_pts:
        return dict(ok=False, reason=f"too few wall points ({len(W)})")
    m = manhattan_yaw(GN if len(GN) > len(NW) else NW, min_pts=50)   # directions: all vertical surfaces agree
    Rm = _ry(m)                                    # rotate by +m about y: a normal at angle m -> angle 0 (x axis)
    W, NW = W @ Rm.T, NW @ Rm.T
    GW, GN = GW @ Rm.T, GN @ Rm.T
    sides, side_pts = {}, {}
    for side in SIDES:
        ax, sgn = _side_axis(side)
        tan = 2 if ax == 0 else 0
        sel = (NW[:, ax] * sgn < -p.layout_normal_min) & (W[:, ax] * sgn > p.layout_min_dist_m)
        side_pts[side] = (W[sel, ax] * sgn, W[sel, tan], W[sel, 1], pid[sel])
        sides[side] = _fit_side(*side_pts[side], p, far_wall=True)
        if sides[side]["status"] != "measured" and sem_used:
            # D-060: no wall-labelled side here (a kitchen side is all cabinets): keep the geometric fit, as before
            gsel = (GN[:, ax] * sgn < -p.layout_normal_min) & (GW[:, ax] * sgn > p.layout_min_dist_m)
            g = _fit_side(GW[gsel, ax] * sgn, GW[gsel, tan], GW[gsel, 1], gid[gsel], p)
            if g["status"] == "measured":
                sides[side] = dict(g, how=str(g.get("how", "")) + " (geometry only: no wall-labelled points)")
    _far_wall_in_room(sides, side_pts, p)
    # extent of the fitted walls along their tangent: a perpendicular wall cannot be nearer than where they end
    for side in SIDES:
        s = sides[side]
        if s["status"] == "measured":
            continue
        ax, sgn = _side_axis(side)
        reach = [sides[o]["tan_extent"][0 if sgn < 0 else 1] * sgn for o in SIDES
                 if _side_axis(o)[0] != ax and sides[o]["status"] == "measured"]
        opp = sides[("-" if side[0] == "+" else "+") + side[1]]
        centre = opp["offset"] if opp["status"] == "measured" else None
        lo = max(reach) if reach else None
        val = max([v for v in (lo, centre) if v is not None], default=p.layout_unseen_default_m)
        s.update(offset=float(val), lower_bound=None if lo is None else float(lo), status="inferred",
                 sigma=float(max(p.layout_unseen_sigma_rel * val, p.layout_unseen_sigma_min_m)),
                 how=("max(extent of the neighbouring walls, opposite wall distance)" if lo is not None and
                      centre is not None else "extent of the neighbouring walls" if lo is not None else
                      "opposite wall distance (photographer near the middle)" if centre is not None else
                      "default (nothing seen)"))
    height = None
    cy = np.concatenate(ceil_y) if ceil_y else np.zeros(0)
    fy = np.concatenate(floor_y) if floor_y else np.zeros(0)
    if len(cy) >= p.layout_min_level_pts:
        floor_level = float(np.median(fy)) if len(fy) >= p.layout_min_level_pts else -cam_h
        hv = float(np.median(cy) - floor_level)
        if p.layout_height_min_m <= hv <= p.layout_height_max_m:
            height = dict(value=hv,
                          sigma=float(np.hypot(p.layout_height_sigma_m, 1.4826 * np.median(np.abs(cy - np.median(cy))))),
                          ceiling_pts=int(len(cy)), floor_from="points" if len(fy) >= p.layout_min_level_pts else "camera")
        else:      # D-049: downward-facing points this low are cabinet/lamp undersides, not the ceiling
            height = None
    out = dict(ok=True, manhattan_yaw=float(m), cam_height_m=cam_h, photos=names, wall_points=per, sides=sides,
               height=height, semantic_photos=sem_used)
    out["mc"] = _monte_carlo(out, names, p, rng)
    return out


def _fit_side(d: np.ndarray, tan: np.ndarray, y: np.ndarray, pid: np.ndarray, p, far_wall: bool = False) -> dict:
    """Wall offset along one side: the bin with the longest observed wall (distinct 10 cm tangent cells).
    far_wall: D-077, a wall seen beside the chosen surface, farther out, replaces it (_far_wall)."""
    none = dict(status="unseen", offset=None, sigma=None, photos=[], len_m=0.0, tan_extent=(0.0, 0.0))
    if len(d) < p.layout_min_side_pts:
        return none
    b = p.layout_bin_m
    k = np.floor(d / b).astype(int)
    cell = np.floor(tan / p.layout_cell_m).astype(np.int64)        # wall LENGTH seen at this offset (tangent cells)
    kmin = k.min()
    area = np.zeros(k.max() - kmin + 3)
    for kk in np.unique(k):
        area[kk - kmin + 1] = len(np.unique(cell[k == kk]))
    sm = np.convolve(area, np.ones(3), "same") * p.layout_cell_m
    best = int(np.argmax(sm))
    if p.layout_prefer_near_ratio < 1.0:
        # a strong peak NEARER than the longest one wins: beyond a wall, only door-wide patches of the next room are
        # seen, and with one point per column furniture rarely makes a long peak (photo_tier.md v3, PT-15)
        peaks = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]
                 and sm[i] >= p.layout_prefer_near_ratio * sm[best] and sm[i] >= p.layout_min_wall_len_m]
        if peaks:
            best = min(peaks)
    if sm[best] < p.layout_min_wall_len_m:
        return dict(none, len_m=float(sm[best]))
    centre = (best - 1 + kmin + 0.5) * b
    inl = np.abs(d - centre) < 1.5 * b
    off = float(np.median(d[inl]))
    moved = None
    if far_wall and getattr(p, "layout_far_wall", False):
        fw = _far_wall(d, tan, pid, sm, kmin, off, p)
        if fw is not None:
            best, moved = fw
            centre = (best - 1 + kmin + 0.5) * b
            inl = np.abs(d - centre) < 1.5 * b
            off = float(np.median(d[inl]))
            moved["to_m"] = round(off, 3)
    inl = np.abs(d - off) < p.layout_inlier_m
    photos = {}
    for q in np.unique(pid[inl]):
        sel = inl & (pid == q)
        if sel.sum() >= p.layout_min_side_pts // 4:
            photos[int(q)] = float(np.median(d[sel]))
    # rival: second-best peak further than 0.3 m from the chosen one (ambiguity diagnostic)
    far = np.abs((np.arange(len(sm)) - 1 + kmin + 0.5) * b - off) > 0.3
    rival = float(sm[far].max()) if far.any() else 0.0
    t_in = tan[inl]
    out = dict(status="measured", offset=off, photos=photos, len_m=float(sm[best]),
               rival_ratio=round(rival / max(float(sm[best]), 1e-9), 3), inliers=int(inl.sum()),
               fit_mad_m=float(1.4826 * np.median(np.abs(d[inl] - off))),
               tan_extent=(float(np.percentile(t_in, 1)), float(np.percentile(t_in, 99))))
    if moved is not None:
        out["far_wall"] = moved
    return out


def _seen_cells(tan: np.ndarray, pid: np.ndarray, sel: np.ndarray, cell: float, min_pts: int = 10) -> np.ndarray:
    """10 cm tangent cells of the points in sel, from photos with >= min_pts of them (a few stray points of another
    photo do not make a surface)."""
    keep = np.zeros(len(tan), bool)
    for q in np.unique(pid[sel]):
        s = sel & (pid == q)
        if s.sum() >= min_pts:
            keep |= s
    return np.unique(np.floor(tan[keep] / cell).astype(np.int64))


def _far_wall(d: np.ndarray, tan: np.ndarray, pid: np.ndarray, sm: np.ndarray, kmin: int, off: float, p):
    """D-077: is the chosen surface (offset off) furniture standing in front of the room's wall?

    Own Room (tape): the rule above took a wardrobe front (lit and dim takes) and an open door leaf (dim) as walls;
    the segmenter labels both "wall". In each case one photo sees the real wall BESIDE that surface, 0.57-1.01 m
    farther out: on the other side of the door next to the wardrobe, or past the door leaf. A farther wall-length
    peak wins when all of these hold:
      - >= layout_far_wall_min_len_m of wall and >= layout_min_side_pts points within 0.3 m (what any side needs;
        sim k65: 8 points of one doorway photo made a 0.8 m peak), layout_far_wall_min/max_gap_m behind the chosen
        surface, and >= layout_far_wall_min_ratio x its distance. One wall seen by two photos (another photo's depth
        scale, or a photo away from the spin centre) gave peaks 0.30-0.45 m apart on the simulator, at ratios
        1.15-1.33;
      - shadow: <= layout_far_wall_max_shadow of its rays (from the spin centre) cross the chosen plane where the
        chosen surface was seen. Nothing behind a surface is visible through it, so such a peak is the same surface
        at another depth scale (sim k65 bedroom: a blank wall close-up 30% too deep);
      - past an end: >= layout_far_wall_min_beyond of its rays cross the chosen plane OUTSIDE the chosen surface's
        extent, not in a gap between two of its parts. A wardrobe ends (at the door) and the wall line continues
        beyond it; a wall seen on both sides of the far patch is a wall with an opening (own lit: the white door
        between a pillar face and the door frame), and it stays.
    The longest qualifying peak wins; inside +-0.3 m of it the longest bin, as for the code's own choice. Once all
    4 sides are fitted, _far_wall_in_room drops a far wall that lies outside the room (the next room through a door).
    Returns (bin index, diagnostics) or None."""
    b = p.layout_bin_m
    c = p.layout_cell_m
    near = np.abs(d - off) < p.layout_inlier_m
    cells = _seen_cells(tan, pid, near, c)
    if len(cells) == 0:
        return None
    covered = np.unique(np.concatenate([cells - 1, cells, cells + 1]))      # one cell of slack either side
    # the chosen surface's extent: its cells joined across gaps <= 0.3 m, pieces < 0.3 m dropped
    runs = []
    for q in cells:
        if runs and q - runs[-1][1] <= 4:
            runs[-1][1] = q
        else:
            runs.append([q, q])
    runs = [r for r in runs if (r[1] + 1 - r[0]) * c >= 0.3 - 1e-9]
    if not runs:
        return None
    lo, hi = runs[0][0] * c, (runs[-1][1] + 1) * c
    best, info = None, None
    for i in range(1, len(sm) - 1):
        if not (sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] >= p.layout_far_wall_min_len_m):
            continue
        D = (i - 1 + kmin + 0.5) * b
        if not (off + p.layout_far_wall_min_gap_m <= D <= off + p.layout_far_wall_max_gap_m
                and D >= p.layout_far_wall_min_ratio * off):
            continue
        if (np.abs(d - D) <= 0.3).sum() < p.layout_min_side_pts:
            continue
        t = tan[np.abs(d - D) < 2 * b] * (off / D)            # where the far peak's rays cross the chosen plane
        if len(t) == 0:
            continue
        shadow = float(np.isin(np.floor(t / c).astype(np.int64), covered).mean())
        beyond = float(((t < lo) | (t > hi)).mean())
        if shadow > p.layout_far_wall_max_shadow or beyond < p.layout_far_wall_min_beyond:
            continue
        if best is None or sm[i] > sm[best]:
            best, info = i, dict(from_m=round(off, 3), peak_m=round(float(D), 3), len_m=round(float(sm[i]), 2),
                                 shadow=round(shadow, 2), beyond_end=round(beyond, 2))
    if best is None:
        return None
    win = [i for i in range(len(sm)) if abs((i - best) * b) <= 0.3 + 1e-9]
    return int(max(win, key=lambda i: (sm[i], -i))), info


def _far_wall_in_room(sides: dict, side_pts: dict, p) -> None:
    """D-077, once all 4 sides are fitted: a far wall must be THIS room's wall, so it must overlap the room's extent
    along it (between the measured perpendicular sides) by >= layout_far_wall_min_overlap_m. A patch beyond a
    perpendicular wall is the next room seen through a door at the end of the chosen surface (sim k38_s1 bathroom:
    1.15 -> 2.43 m, truth 1.06 m; overlap -0.13 m, own Room 0.96-1.94 m). Such a side keeps its first choice."""
    for side, s in list(sides.items()):
        if not s.get("far_wall"):
            continue
        hi_s, lo_s = ("+z", "-z") if side[1] == "x" else ("+x", "-x")
        hi = sides[hi_s]["offset"] if sides[hi_s]["status"] == "measured" else np.inf
        lo = -sides[lo_s]["offset"] if sides[lo_s]["status"] == "measured" else -np.inf
        a, b = s["tan_extent"]
        overlap = min(b, hi) - max(a, lo)
        if overlap < p.layout_far_wall_min_overlap_m:
            sides[side] = dict(_fit_side(*side_pts[side], p),
                               far_wall_rejected=dict(s["far_wall"], overlap_m=round(float(overlap), 2),
                                                      reason="outside the room's extent along it (next room)"))


def _monte_carlo(lay: dict, names: list[str], p, rng) -> dict:
    """Per-side offset samples: each photo's own offset, moved by its position error (sway around the spin centre)
    and scaled by its own depth-scale error, averaged over the photos that saw the side; plus surface noise.
    Unseen sides: Gaussian around the inferred value, cut at the lower bound (a wall cannot be nearer than where the
    neighbouring walls were seen to end). The global scale error is NOT here (run_capture adds the tier scale term)."""
    S = p.layout_mc
    dt = rng.normal(0, p.layout_cam_sigma_m, (S, len(names), 2))
    ds = np.exp(rng.normal(0, p.prior_sigma_log_s, (S, len(names))))
    samples = {}
    for side in SIDES:
        s = lay["sides"][side]
        ax, sgn = _side_axis(side)
        j = 0 if ax == 0 else 1
        if s["status"] == "measured":
            q = np.array(list(s["photos"].keys()) or [0])
            o = np.array(list(s["photos"].values()) or [s["offset"]])
            o = o - np.mean(o) + s["offset"]                     # centre on the pooled estimate
            v = (o[None, :] * ds[:, q] + sgn * dt[:, q, j]).mean(1)
            v = v + rng.normal(0, np.hypot(p.layout_geom_sigma_m, s["fit_mad_m"] / np.sqrt(max(s["inliers"], 1))), S)
            s["sigma"] = float(np.std(v))
        else:
            v = rng.normal(s["offset"], s["sigma"], S)
            if s.get("lower_bound") is not None:
                v = np.maximum(v, s["lower_bound"])
        samples[side] = v
    return samples


# ------------------------------------------------------------------------------ into the scene frame (frontend)
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ceiling_photo_scales(views: dict, matches: dict, ceiling: dict, min_matches: int = 12, max_pairs: int = 4000,
                         seed: int = 0) -> dict:
    """D-057: a ceiling photo's depth scale RELATIVE to its room's other photos, from their feature matches.

    MoGe-2's scale on a photo tilted up at a mostly blank ceiling is unreliable. Sim truth, 1 BHK: metric/predicted
    0.59-1.18 on the ceiling photos, against a median 0.975 and a 5.6% spread on the others. Kitchen and bathroom
    ceilings came out +50% and +33%. A ceiling photo shares features with its room's other photos (upper walls, door
    heads). The ratio of distances between matched 3-D points in the two photos is their scale ratio, independent
    of any pose (levelling is a rotation). Returns {ceiling photo: dict(scale, refs, matches, spread)}; photos with
    no usable matches are absent (their height then gets a wide interval)."""
    rng = np.random.default_rng(seed)
    by_room = {}
    for n, v in views.items():
        by_room.setdefault(v.room, []).append(n)
    out = {}
    for room, cs in (ceiling or {}).items():
        refs = [n for n in by_room.get(room, []) if n not in cs]
        for c in cs:
            ests, ws = [], []
            for n in refs:
                if (c, n) in matches:
                    xy_c, xy_n = matches[(c, n)]
                elif (n, c) in matches:
                    xy_n, xy_c = matches[(n, c)]
                else:
                    continue
                if len(xy_c) < min_matches:
                    continue
                Pc, okc = views[c].lift(np.asarray(xy_c, float))
                Pn, okn = views[n].lift(np.asarray(xy_n, float))
                ok = okc & okn
                if ok.sum() < min_matches:
                    continue
                Pc, Pn = Pc[ok], Pn[ok]
                i, j = rng.integers(0, len(Pc), max_pairs), rng.integers(0, len(Pc), max_pairs)
                dc, dn = np.linalg.norm(Pc[i] - Pc[j], axis=1), np.linalg.norm(Pn[i] - Pn[j], axis=1)
                keep = (dc > 0.3) & (dn > 0.3)
                if keep.sum() < 50:
                    continue
                ests.append(float(np.median(dn[keep] / dc[keep])))
                ws.append(int(ok.sum()))
            if ests:
                e, w = np.array(ests), np.array(ws, float)
                o = np.argsort(e)
                sc = float(e[o][min(int(np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)), len(e) - 1)])
                out[c] = dict(scale=sc, refs=len(e), matches=int(w.sum()),
                              spread=float(np.median(np.abs(np.log(e / sc)))) if len(e) > 1 else None)
    return out


def ceiling_above_camera(views: dict, names: list[str], scale: float, p, rel: dict | None = None) -> float | None:
    """Ceiling height above the camera from the room's ceiling photos (protocol v2, D-052): points with a downward
    normal, more than 0.4 m above the camera, in the levelled frame (yaw does not matter for heights). rel: D-057
    relative scale of each ceiling photo to its room's other photos (ceiling_photo_scales)."""
    ys = []
    for n in names:
        v = views.get(n)
        if v is None or v.normal_cam is None:
            continue
        Pc, idx = v.points_cam(p.layout_stride)
        P = scale * float((rel or {}).get(n, {}).get("scale", 1.0)) * Pc @ v.R_lev.T
        N = v.normal_cam.reshape(-1, 3)[idx] @ v.R_lev.T
        sel = (N[:, 1] < -0.85) & (P[:, 1] > 0.4)
        if sel.sum() >= p.layout_min_level_pts:
            ys.append(float(np.median(P[sel, 1])))
    return float(np.median(ys)) if ys else None


def _refit_with_door_shots(lay, views, group, yaws, shots, poses, scale, p):
    """D-054: a hub room (many doors) has only 3-4 spin photos, which leave walls unseen; its doorway-pair photos look
    into it from the thresholds. Add them to the fit with their pose-graph position and yaw RELATIVE to the placed spin
    photos (same connected placement), if they lie within 6 m of the spin centre. Keep the refit only if it measures
    more sides than the spin alone."""
    placed = [n for n in group if n in poses]
    extra = [d for d in shots if d in poses and d not in group]
    if len(placed) < 2 or not extra:
        return lay
    d = np.array([_wrap(poses[n][0] - yaws[n]) for n in placed])
    dyaw = float(np.angle(np.mean(np.exp(1j * d))))
    c = np.mean([poses[n][1] for n in placed], 0)
    yaws2, offs = dict(yaws), {n: np.zeros(3) for n in yaws}
    for e in extra:
        o = _ry(-dyaw) @ (np.asarray(poses[e][1]) - c)
        if np.linalg.norm(o[[0, 2]]) > 6.0:
            continue
        yaws2[e], offs[e] = float(_wrap(poses[e][0] - dyaw)), o
    if len(yaws2) == len(yaws):
        return lay
    lay2 = fit_room_layout(views, yaws2, scale, p, offsets=offs)
    n1 = sum(v["status"] == "measured" for v in lay["sides"].values())
    n2 = sum(v["status"] == "measured" for v in lay2.get("sides", {}).values()) if lay2.get("ok") else -1
    if n2 > n1:
        lay2["door_shots_used"] = [e for e in yaws2 if e not in yaws]
        return lay2
    return lay


def _closeup_spin(views: dict, group: list[str], p) -> bool:
    """D-062: a 'spin' whose photos are mostly close-ups of a wall carries no room geometry. Sim 2 BHK bedroom2
    (8.9 m2, bed in the middle): the only free spot was against a wall, 2 of 4 turning photos saw blank wall at
    0.3-0.6 m, and the box came out 4.2 m along one axis (true 2.9) and 1.5 m along the other (true 3.0). A person
    in a small furnished bedroom does the same; the doorway photo looks in across the room instead."""
    if len(group) < 3:
        return False
    near = 0
    for n in group:
        d = views[n].depth
        h = d.shape[0]
        d = d[int(0.15 * h):int(0.6 * h)]          # the band around the horizon: walls, not the floor at one's feet
        dv = d[d > 0]                              # (a 0.5x photo's bottom edge sees floor at ~0.9 m, D-065)
        if dv.size and float(np.median(dv)) < p.closeup_depth_m:
            near += 1
    return near >= 0.5 * len(group)


def _threshold_extras(lay, views, door_shot, extras, poses, scale, p):
    """v2.1 (D-055): a small room's second doorway photo (turned the other way, taken on the same threshold after the
    ceiling photo) joins the threshold box fit, so the side wall the first photo missed is measured.

    Placed by the pose graph within same_spot_m of the first photo: its relative yaw and offset are used. NOT placed
    (D-063: the two doorway photos overlap little, so the graph rarely links them; sim 2 BHK: bathroom, bedroom2
    and kitchen each got one side "inferred", widths 1.0-1.2 m vs 1.7-3.0 m true): the protocol says same spot, so
    try the four yaws that put its walls on the room's axes (both photos' Manhattan directions) and keep the one
    that measures MORE sides while moving no side the first photo measured by more than 25 cm. A photo from
    another spot (a balcony seen from 2 m back) fails that test and is ignored."""
    from floorplan.photo.protocol import photo_manhattan
    if not lay.get("ok"):
        return lay
    base_meas = {k: v["offset"] for k, v in lay["sides"].items() if v["status"] == "measured"}
    yaws, offs, used = {door_shot: 0.0}, {door_shot: np.zeros(3)}, []
    loose = []
    for e in extras:
        if e == door_shot or e not in views:
            continue
        if door_shot in poses and e in poses:
            th0, t0 = poses[door_shot][0], np.asarray(poses[door_shot][1])
            o = _ry(-th0) @ (np.asarray(poses[e][1]) - t0)
            if np.linalg.norm(o[[0, 2]]) <= p.same_spot_m:
                yaws[e], offs[e] = float(_wrap(poses[e][0] - th0)), o
                used.append(e)
            continue                                   # placed elsewhere: another spot
        loose.append(e)

    def _score(lay2):
        if not lay2.get("ok"):
            return None
        n = sum(v["status"] == "measured" for v in lay2["sides"].values())
        moved = [abs(lay2["sides"][k]["offset"] - off) for k, off in base_meas.items()
                 if lay2["sides"][k]["status"] == "measured"]
        lost = sum(lay2["sides"][k]["status"] != "measured" for k in base_meas)
        if lost or (moved and max(moved) > 0.25):
            return None
        return n, -sum(moved)

    best_lay, best = None, None
    if used:
        lay2 = fit_room_layout(views, yaws, scale, p, offsets=offs)
        sc = _score(lay2)
        if sc is not None:
            best_lay, best = lay2, sc
    m_d = photo_manhattan(views[door_shot])
    if not getattr(p, "threshold_same_spot_unplaced", False):
        loose = []                                 # D-063 off by default: k65 kitchen picked a -130 deg candidate, +78% area
    for e in loose:
        m_e = photo_manhattan(views[e])
        if m_d is None or m_e is None:
            continue
        for k in range(4):
            th = float(_wrap(m_e - m_d + k * np.pi / 2))
            y2, o2 = dict(yaws), dict(offs)
            y2[e], o2[e] = th, np.zeros(3)
            lay2 = fit_room_layout(views, y2, scale, p, offsets=o2)
            sc = _score(lay2)
            if sc is not None and (best is None or sc > best):
                best_lay, best = lay2, sc
                best_lay["threshold_extra_same_spot"] = dict(photo=e, yaw_deg=round(float(np.degrees(th)), 1))
    n0 = len(base_meas)
    if best_lay is not None and best[0] > n0:
        best_lay["threshold_extras_used"] = [e for e in best_lay.get("photos", []) if e != door_shot]
        return best_lay
    return lay


def room_layouts(views: dict, spin_info: dict, poses: dict, scale: float, Ra: np.ndarray, ta: np.ndarray, p,
                 log=print, ceiling_photos: dict | None = None, doorway_into: dict | None = None,
                 all_pair_photos: set | None = None, extra_photos: dict | None = None,
                 ceiling_scale: dict | None = None) -> dict:
    """Fit every room's layout from its spin photos and express it in the ALIGNED scene frame.

    poses: final pose-graph poses (yaw, t, log_s) in the common frame, already multiplied by the fused scale.
    A layout is anchored by the room's PLACED spin photos: local yaw phi_n vs common yaw theta_n gives the rotation,
    the placed photos' mean position the centre (the spin centre). Rooms whose spin photos were not placed keep the
    layout (sizes are still measured) with anchor None; plan_beta then centres it on the free-space room."""
    out = {}
    pair_shots_of, door_shots_of = {}, {}
    for n, v in views.items():                         # every doorway-pair photo that looks into a room (D-054)
        if n in set((doorway_into or {}).values()) or n in (all_pair_photos or set()):
            pair_shots_of.setdefault(v.room, []).append(n)
    for room in set(pair_shots_of) | set(extra_photos or {}):   # + photos from other spots (v2.1, D-055)
        door_shots_of[room] = pair_shots_of.get(room, []) + list((extra_photos or {}).get(room, []))
    for room, si in sorted(spin_info.items()):
        group = si.get("photos") or []
        door_shot = (doorway_into or {}).get(room)
        orig_group, orig_si = group, si
        cnames = (ceiling_photos or {}).get(room) or []
        ex = [e for e in (extra_photos or {}).get(room, []) if e in poses]
        ex_all = list((extra_photos or {}).get(room, []))        # D-063: unplaced extras too (same-spot hypothesis)
        cands = [d for d in pair_shots_of.get(room, []) if d in poses]
        if ex and cands:                               # the threshold photo the room's extra photos were taken beside
            def _near(d):
                t0 = np.asarray(poses[d][1])
                return sum(np.linalg.norm((np.asarray(poses[e][1]) - t0)[[0, 2]]) <= p.same_spot_m for e in ex)
            best = max(cands, key=_near)
            if _near(best) > 0:
                door_shot = best
        closeup = _closeup_spin(views, group, p)
        if door_shot and p.small_room_from_doorway and (len(group) < 2 or (len(group) == 2 and cnames) or closeup):
            # protocol v2 small room (D-053): the photo taken ON the threshold looking in defines the box. Far wall
            # and side walls are measured; the door wall is where the camera stands (its room-side face is half a
            # wall thickness in front of the camera). v2.1: the 2 photos from the far end are not a spin of the room
            # (they look back at the door); a ceiling photo marks the v2 protocol (v1 captures are unchanged).
            group, si = [door_shot], dict(si, photos=[door_shot], steps_deg=[], small_room=True,
                                          closeup_spin=bool(closeup))
        if not group:
            out[room] = dict(ok=False, reason=si.get("note", "no spin photos"))
            continue
        steps = si.get("steps_deg") or []
        if len(group) > 1 and len(steps) != len(group) - 1:
            out[room] = dict(ok=False, reason=si.get("note", "spin steps unknown"))
            continue
        yaws = local_poses(group, steps)
        try:
            lay = fit_room_layout(views, yaws, scale, p)
            if si.get("small_room") and ex_all:
                lay = _threshold_extras(lay, views, door_shot, ex_all, poses, scale, p)
            if not lay.get("ok") and si.get("small_room") and orig_group:
                # D-054: the doorway view saw mostly glass (a balcony's window wall): use the far-end photo(s) instead
                steps0 = orig_si.get("steps_deg") or []
                group = orig_group
                si = dict(orig_si) if len(steps0) == len(group) - 1 else dict(orig_si, steps_deg=[])
                yaws = local_poses(group, si.get("steps_deg") or [])
                lay = fit_room_layout(views, yaws, scale, p)
            if lay.get("ok") and not si.get("small_room") and p.layout_use_door_shots and len(group) >= 2:
                lay = _refit_with_door_shots(lay, views, group, yaws, door_shots_of.get(room, []), poses, scale, p)
        except Exception as e:                      # a layout must never cost the capture: v4 room is the fallback
            out[room] = dict(ok=False, reason=f"layout fit failed: {type(e).__name__}: {e}")
            continue
        if not lay.get("ok"):
            out[room] = lay
            continue
        lay.pop("mc", None)
        if si.get("small_room"):
            fwd = _ry(lay["manhattan_yaw"]) @ views[group[0]].R_lev @ np.array([0.0, 0.0, 1.0])
            behind = max(SIDES, key=lambda sd: -(fwd[_side_axis(sd)[0]] * _side_axis(sd)[1]))
            lay["sides"][behind] = dict(status="measured", offset=-p.door_threshold_inset_m,
                                        sigma=p.door_threshold_sigma_m, photos=[], len_m=0.0, tan_extent=(0.0, 0.0),
                                        how="camera on the door threshold (protocol v2 small room)")
            lay["small_room"] = dict(door_photo=group[0], door_side=behind)
        if cnames:                                    # D-052: the ceiling photo measures this room's height
            linked = [c for c in cnames if c in (ceiling_scale or {})]
            above = ceiling_above_camera(views, linked or cnames, scale, p, rel=ceiling_scale)
            if above is not None:
                hv = lay["cam_height_m"] + above
                # D-057: linked to the room's photos by features -> its scale is theirs; unlinked -> MoGe-2's own
                # scale on an upward photo (sim spread ~25%): an honest, wide interval
                sig_above = p.ceiling_link_sigma_rel if linked else p.ceiling_unlinked_sigma_rel
                if p.layout_height_min_m <= hv <= p.layout_height_max_m:
                    sm = float(np.hypot(np.hypot(p.layout_height_sigma_m, 0.05), sig_above * above))
                    mu0, s0 = p.ceiling_prior_m
                    w1, w0 = 1 / sm ** 2, 1 / s0 ** 2      # D-057: fused with the residential prior (disclosed)
                    lay["height"] = dict(value=float((w1 * hv + w0 * mu0) / (w1 + w0)), sigma=float((w1 + w0) ** -0.5),
                                         measured_value=float(hv), measured_sigma=sm, prior=[mu0, s0],
                                         linked=bool(linked), ceiling_pts=None, floor_from="spin photos' floor plane",
                                         source=f"ceiling photo(s) {linked or cnames}"
                                                + (" scaled by matches to the room's photos (D-057)" if linked
                                                   else " (own depth scale, not linked: wide interval)")
                                                + f", fused with the residential prior {mu0:.2f} +- {s0:.2f} m",
                                         ceiling_scale={c: ceiling_scale[c] for c in linked})
        # ---- v3-poly (photo/polygon.py): rectangle + evidence-backed steps (alcoves, notches, the doorway photo on an
        # inferred side). Stored beside the rectangle; no existing key changes. plan_beta uses it when present.
        if getattr(p, "layout_polygon", True) and not si.get("small_room"):
            try:
                from floorplan.photo.polygon import room_polygon
                room_ph = [n for n in views if views[n].room == room and n not in cnames]
                lay["polygon_local"] = room_polygon(lay, views, group, yaws, poses, room_ph,
                                                    pair_shots_of.get(room, []), scale, p, log)
                if lay["polygon_local"].get("used"):
                    log(f"[photo/v3-poly] {room}: polygon with {lay['polygon_local']['n_vertices']} vertices "
                        f"({', '.join(c['kind'] for c in lay['polygon_local']['changes'])}), area "
                        f"{lay['polygon_local']['rect_area_m2']:.2f} -> {lay['polygon_local']['area_m2']:.2f} m2")
            except Exception as e:                  # the rectangle stands
                lay["polygon_local"] = dict(ok=False, reason=f"polygon fit failed: {type(e).__name__}: {e}")
        # ------------------------------------------------------------------------------------------------------
        placed = [n for n in group if n in poses]
        anchor = None
        if placed:
            d = np.array([_wrap(poses[n][0] - yaws[n]) for n in placed])
            dyaw = float(np.angle(np.mean(np.exp(1j * d))))
            c = np.mean([poses[n][1] for n in placed], 0)
            # layout frame L -> local (Rm^T) -> common (R_y(dyaw), + c) -> aligned (Ra, ta)
            R = Ra @ _ry(dyaw) @ _ry(lay["manhattan_yaw"]).T
            centre = Ra @ c + ta
            ex, ez = R[:, 0], R[:, 2]                          # aligned directions of the layout's +x and +z sides
            # side direction in the aligned plan (u = x, v = z); snap to the nearest axis
            amap, resid = {}, []
            for side, e in (("+x", ex), ("-x", -ex), ("+z", ez), ("-z", -ez)):
                ang = np.arctan2(e[2], e[0])
                q = int(np.round(ang / (np.pi / 2))) % 4
                resid.append(abs(np.degrees(_wrap(ang - q * np.pi / 2))))
                amap[side] = ["+x", "+z", "-x", "-z"][q]
            anchor = dict(centre_uv=[float(centre[0]), float(centre[2])], placed_spin_photos=placed,
                          side_to_plan=amap, snap_residual_deg=round(float(max(resid)), 2),
                          yaw_spread_deg=round(float(np.degrees(np.max(np.abs(_wrap(d - dyaw))))), 2))
        sides_plan = {}
        for side, s in lay["sides"].items():
            key = anchor["side_to_plan"][side] if anchor else side
            sides_plan[key] = dict(s, layout_side=side)
        lay.update(sides_plan=sides_plan, anchor=anchor, room=room)
        out[room] = lay
        log(f"[photo/v3] {room}: layout from {len(lay['photos'])} spin photo(s), sides "
            + " ".join(f"{k}={v['offset']:.2f}{'' if v['status'] == 'measured' else '(inf)'}"
                       for k, v in sorted(sides_plan.items()))
            + (f", anchored by {len(placed)} placed" if anchor else ", not anchored"))
    return _jsonable(out)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x
