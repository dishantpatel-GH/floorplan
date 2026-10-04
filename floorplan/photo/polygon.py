"""photo_tier v3-poly: a room's footprint as a rectilinear polygon (rectangle + evidence-backed steps), photo tier.

layout.fit_room_layout fits ONE Manhattan rectangle per room. Open-plan living rooms are not rectangles (sim k38 living
room: an alcove holding the bathroom and bedroom doors; k65: a door alcove and a far end no spin photo saw). This
module keeps the rectangle as the default and changes it only where the photos show it is wrong. In the room's layout
frame (spin centre = origin, axes = the rectangle's Manhattan axes):

  1. photos: the spin photos (local yaws from the spin steps, all at the centre) plus every other placed photo of the
     room folder (doorway-pair photos, extra photos) at its pose-graph position relative to the placed spin photos,
     the same placement as layout._refit_with_door_shots;
  2. wall points: the rectangle fit's rule (horizontal normal, 1-2 m band, farthest point per image column, the D-060
     semantic wall mask when the view has one). Visibility: per photo and azimuth, the farthest band point; a floor
     position nearer than it is SEEN FREE, one behind it is BLOCKED;
  3. per side, the offset peaks of that side's wall points (wall LENGTH in 10 cm tangent cells, like _fit_side);
  4. a MEASURED side whose wall was seen only outside the room's extent along it (a neighbour's wall through a wide
     door, beside the room) is replaced by the nearest farther wall of that side that overlaps the room;
  5. an INFERRED side reaches at least the doorway photo standing on it (taken on the threshold);
  6. alcove: doorway photo(s) of this room >= 0.4 m beyond a measured side, corroborated by a second doorway photo or
     by a wall seen just beyond; not if the spin photos saw that region mostly blocked;
  7. notch: a rectangle corner cell between seen wall lines that is mostly blocked, never seen free and fronted by
     seen walls (strict; cells nobody saw keep the rectangle: no evidence -> rectangle);
  8. the polygon is the boundary of the inside cells of the line grid (<= 2 changes); every edge keeps the line it
     lies on (offset, sigma, status).

Output (lay["polygon_local"]): vertices in the layout frame, per-edge status/sigma, the line grid and cell mask (for the
Monte Carlo in plan_beta), and the evidence behind each change. plan_beta (tiers.layout_plan) uses it when present
(params.layout_polygon). Evidence and limits: docs/modules/photo_tier.md "photo_tier v3-poly".
"""
from __future__ import annotations

import numpy as np

from floorplan.photo.layout import SIDES, _farthest_per_column, _photo_cloud, _ry, _side_axis, _wrap

# Defaults; a PhotoParams field of the same name overrides (getattr), so no params change is needed to tune them.
DEFAULTS = dict(
    layout_polygon=True,             # fit the polygon at all
    poly_max_photo_dist_m=6.0,       # extra photos further than this from the spin centre are not used (as D-054)
    poly_seg_min_len_m=0.5,          # a candidate wall line needs >= 0.5 m of observed wall at one offset
    poly_line_sep_m=0.3,             # ... and must be >= 0.3 m from every other line on its axis
    poly_max_ext_m=3.0,              # an alcove/extension reaches at most 3 m beyond the rectangle
    poly_min_cell_m=0.4,             # a cell narrower than this on either axis is not a room part / notch
    poly_notch_blocked=0.6,          # notch cell: >= 60% blocked ...
    poly_notch_free=0.05,            # ... <= 5% seen free
    poly_notch_min_area_m2=0.35,
    poly_vis_bin_deg=0.5,            # azimuth bin of the visibility profile
    poly_vis_margin_m=0.12,          # a position within this of the farthest band point is not decided
    poly_sample_m=0.1,               # visibility samples per cell
    poly_door_inset_m=0.10,          # the room-side face of a door's wall is ~half a wall in front of the doorway
                                     # photo (same convention as D-053, door_threshold_inset_m)
    poly_max_mods=2,                 # at most 2 changes (alcoves/notches) per room: rectangle + 2 steps
    poly_alcove_min_beyond_m=0.4,    # a doorway photo >= 0.4 m beyond a measured side (door shots are placed to
                                     # ~0.15 m typically, 0.65 m worst on the sim): the room reaches it there
    poly_alcove_max_blocked=0.4,     # an alcove the spin photos saw mostly behind a wall is not this room's
    poly_notch_wall_cover=0.4,       # a notch's walls toward the room must be >= 40% seen
    poly_side_min_overlap_m=0.3,     # a measured side's seen wall must overlap the room's extent along it by 0.3 m
)


def _par(p, k):
    return getattr(p, k, DEFAULTS[k])


# ------------------------------------------------------------------------------------------------ photos & points
def room_photo_frame(group: list[str], yaws: dict, poses: dict, candidates: list[str], p) -> tuple[dict, dict]:
    """Spin photos at the centre + other placed photos of the room at their position relative to the placed spin
    photos (local frame of the spin). Returns (yaws, offsets)."""
    yaws2 = dict(yaws)
    offs = {n: np.zeros(3) for n in yaws}
    placed = [n for n in group if n in poses and n in yaws]
    if len(placed) < 1:
        return yaws2, offs
    d = np.array([_wrap(poses[n][0] - yaws[n]) for n in placed])
    dyaw = float(np.angle(np.mean(np.exp(1j * d))))
    c = np.mean([np.asarray(poses[n][1], float) for n in placed], 0)
    for e in candidates:
        if e in yaws2 or e not in poses:
            continue
        o = _ry(-dyaw) @ (np.asarray(poses[e][1], float) - c)
        if np.linalg.norm(o[[0, 2]]) > _par(p, "poly_max_photo_dist_m"):
            continue
        yaws2[e], offs[e] = float(_wrap(poses[e][0] - dyaw)), o
    return yaws2, offs


def collect(views: dict, yaws: dict, offs: dict, scale: float, p, lay: dict) -> dict:
    """Wall points and per-photo visibility profiles in the LAYOUT frame (rectangle axes)."""
    Rm = _ry(lay["manhattan_yaw"])
    cam_h = lay["cam_height_m"]
    nb = int(round(360 / _par(p, "poly_vis_bin_deg")))
    W, NW, pid, cams, vis, names = [], [], [], [], [], []
    for k, n in enumerate(n for n in yaws if n in views and views[n].normal_cam is not None):
        v = views[n]
        t = np.asarray(offs.get(n, np.zeros(3)), float)
        P, N, col = _photo_cloud(v, yaws[n], t, scale, p.layout_stride)
        h = P[:, 1] + cam_h
        band = (h > p.layout_band_lo_m) & (h < p.layout_band_hi_m)
        wall = (np.abs(N[:, 1]) < p.layout_wall_max_ny) & band
        sw = None                                       # D-060 semantic wall mask, same rule as fit_room_layout
        if getattr(p, "semantic_walls", True):
            try:
                from floorplan.photo.semantic import semantic_wall_points
                sw = semantic_wall_points(v, p.layout_stride)
            except Exception:
                sw = None
        mask = getattr(v, "wall_mask", None)            # a per-view boolean pixel mask, if a view carries one
        if sw is None and mask is not None:
            try:
                _, idx = v.points_cam(p.layout_stride)
                sw = np.asarray(mask).reshape(-1)[idx].astype(bool)
            except Exception:
                sw = None
        if sw is not None and int((wall & sw).sum()) >= getattr(p, "semantic_min_pts", 150):
            wall = wall & sw
        wi = np.flatnonzero(wall)
        if p.layout_farthest_per_column:
            wi = wi[_farthest_per_column(P[wi], col[wi], t)]
        tc = Rm @ t
        W.append((P[wi] @ Rm.T)[:, [0, 2]])
        NW.append((N[wi] @ Rm.T)[:, [0, 2]])
        pid.append(np.full(len(wi), len(names)))
        # visibility: farthest band point (any normal) per azimuth bin
        bi = np.flatnonzero(band)
        Q = (P[bi] @ Rm.T)[:, [0, 2]] - tc[[0, 2]]
        az = np.arctan2(Q[:, 1], Q[:, 0])
        r = np.hypot(Q[:, 0], Q[:, 1])
        b = ((az + np.pi) / (2 * np.pi) * nb).astype(int) % nb
        R = np.full(nb, np.nan)
        if len(b):
            o = np.lexsort((r, b))
            last = np.r_[b[o][1:] != b[o][:-1], True]
            R[b[o][last]] = r[o][last]
        cams.append(tc[[0, 2]])
        vis.append(R)
        names.append(n)
    if not names:
        return dict(ok=False)
    return dict(ok=True, W=np.concatenate(W), N=np.concatenate(NW), pid=np.concatenate(pid), cams=np.array(cams),
                vis=np.array(vis), names=names, nb=nb)


def visibility(data: dict, Q: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """For floor positions Q (k,2): seen free by any photo; blocked (behind the farthest band point) by some photo
    and free for none."""
    free = np.zeros(len(Q), bool)
    blocked = np.zeros(len(Q), bool)
    nb = data["nb"]
    for c, R in zip(data["cams"], data["vis"]):
        D = Q - c
        az = np.arctan2(D[:, 1], D[:, 0])
        r = np.hypot(D[:, 0], D[:, 1])
        b = ((az + np.pi) / (2 * np.pi) * nb).astype(int) % nb
        Rb = R[b]
        ok = ~np.isnan(Rb)
        free |= ok & (r < Rb - margin)
        blocked |= ok & (r > Rb + margin)
    return free, blocked & ~free


# ------------------------------------------------------------------------------------------------ candidate lines
def side_segments(d: np.ndarray, tan: np.ndarray, p) -> list[dict]:
    """Offset peaks of one side's wall points, scored by wall length (distinct 10 cm tangent cells)."""
    if len(d) < 10:
        return []
    b = p.layout_bin_m
    k = np.floor(d / b).astype(int)
    cell = np.floor(tan / p.layout_cell_m).astype(np.int64)
    kmin = k.min()
    L = np.zeros(k.max() - kmin + 3)
    for kk in np.unique(k):
        L[kk - kmin + 1] = len(np.unique(cell[k == kk]))
    sm = np.convolve(L, np.ones(3), "same") * p.layout_cell_m
    peaks = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] - 1e-9
             and sm[i] >= _par(p, "poly_seg_min_len_m")]
    out = []
    for i in sorted(peaks, key=lambda i: -sm[i]):
        centre = (i - 1 + kmin + 0.5) * b
        inl = np.abs(d - centre) < 1.5 * b
        if inl.sum() < 5:
            continue
        off = float(np.median(d[inl]))
        inl = np.abs(d - off) < p.layout_inlier_m
        if any(abs(off - s["offset"]) < _par(p, "poly_line_sep_m") for s in out):
            continue
        t = tan[inl]
        out.append(dict(offset=off, len_m=float(sm[i]), t_lo=float(np.percentile(t, 2)),
                        t_hi=float(np.percentile(t, 98)),
                        n=int(inl.sum()), mad=float(1.4826 * np.median(np.abs(d[inl] - off)))))
    return out


# ------------------------------------------------------------------------------------------------ the polygon fit
def _line_sigma(seg: dict | None, fallback: float) -> float:
    if seg is None:
        return float(fallback)
    return float(max(np.hypot(0.05, seg["mad"] / np.sqrt(max(seg["n"], 1))), 0.15))


def fit_polygon(lay: dict, data: dict, p, door_photos: list[str] | None = None, log=None) -> dict:
    """Rectangle + evidence-backed steps -> rectilinear polygon (layout frame). See the module docstring."""
    S = lay["sides"]
    rect = {s: float(S[s]["offset"]) for s in SIDES}
    status = {s: S[s]["status"] for s in SIDES}
    sig = {s: float(S[s].get("sigma") or 0.3) for s in SIDES}
    notes = []
    W, N, cams = data["W"], data["N"], data["cams"]
    inset = _par(p, "poly_door_inset_m")
    sep = _par(p, "poly_line_sep_m")
    dcams = {n: cams[data["names"].index(n)] for n in (door_photos or []) if n in data["names"]}

    def t_range(j):          # tangent range of the rectangle for a side whose normal axis is j
        return (-rect["-z"], rect["+z"]) if j == 0 else (-rect["-x"], rect["+x"])

    # 3. wall segments per side (offset peaks), in layout coordinates
    segs = {}
    for s in SIDES:
        ax, sg = _side_axis(s)
        j = 0 if ax == 0 else 1
        sel = (N[:, j] * sg < -p.layout_normal_min) & (W[:, j] * sg > p.layout_min_dist_m)
        segs[s] = side_segments(W[sel, j] * sg, W[sel, 1 - j], p)
    # 2b. a MEASURED side whose wall was seen only OUTSIDE the room's extent along it (e.g. through a wide door, beside
    # the room) cannot bound the room: the nearest farther wall of that side that does overlap the room replaces it
    for s in SIDES:
        if status[s] != "measured" or not S[s].get("tan_extent"):
            continue
        ax, sg = _side_axis(s)
        j = 0 if ax == 0 else 1
        lo_t, hi_t = t_range(j)
        a, b = S[s]["tan_extent"]
        if min(b, hi_t) - max(a, lo_t) >= _par(p, "poly_side_min_overlap_m"):
            continue
        alt = [g for g in segs[s] if g["offset"] > rect[s] + 0.3 and g["offset"] <= rect[s] + _par(p, "poly_max_ext_m")
               and min(g["t_hi"], hi_t) - max(g["t_lo"], lo_t) >= 2 * _par(p, "poly_side_min_overlap_m")]
        if alt:
            g = min(alt, key=lambda g: g["offset"])
            notes.append(dict(kind="side_replaced", side=s, from_m=round(rect[s], 3), to_m=round(g["offset"], 3),
                              reason="the rectangle's wall was seen only outside the room's extent (through an "
                                     "opening); the nearest wall overlapping the room replaces it",
                              seen_extent=[round(a, 2), round(b, 2)], room_extent=[round(lo_t, 2), round(hi_t, 2)]))
            rect[s] = float(g["offset"])
            sig[s] = float(max(sig[s], _line_sigma(g, sig[s])))

    # 5. an inferred side reaches at least the doorway photo standing on it
    for n, c in dcams.items():
        for s in SIDES:
            if status[s] == "measured":
                continue
            ax, sg = _side_axis(s)
            j = 0 if ax == 0 else 1
            lo_t, hi_t = t_range(j)
            reach = sg * c[j] - inset
            if reach > rect[s] + 0.2 and lo_t - 0.3 <= c[1 - j] <= hi_t + 0.3:
                notes.append(dict(kind="door_photo_extends_inferred_side", side=s, photo=n, from_m=round(rect[s], 3),
                                  to_m=round(float(reach), 3)))
                rect[s] = float(reach)
                sig[s] = float(max(sig[s] * 0.6, 0.3))
    lines = {0: {}, 1: {}}                       # axis -> {position: (side, seg or None, kind)}

    def add_line(j, pos, side, seg, kind, tol=0.05):
        for q in lines[j]:
            if abs(q - pos) < tol:
                return q
        lines[j][float(pos)] = (side, seg, kind)
        return float(pos)
    for s in SIDES:
        ax, sg = _side_axis(s)
        add_line(0 if ax == 0 else 1, sg * rect[s], s, None, "rectangle")
    # 4a. alcoves: a doorway photo of this room standing beyond a MEASURED side (>= 0.4 m) -> the room reaches it,
    # when corroborated (a second doorway photo there, or a wall seen just beyond it).
    # Depth: to the photo (- half a wall) or to a seen wall on that side just beyond it; tangent span: the photo(s)
    # +- half a door, stretched to the nearest seen perpendicular wall outside the rectangle within 1.5 m.
    alcoves = []
    for s in SIDES:
        if status[s] != "measured":
            continue
        ax, sg = _side_axis(s)
        j = 0 if ax == 0 else 1
        lo_t, hi_t = t_range(j)
        ps = [(n, c) for n, c in dcams.items() if sg * c[j] - rect[s] >= _par(p, "poly_alcove_min_beyond_m")
              and lo_t - 0.3 <= c[1 - j] <= hi_t + 0.3]
        if not ps:
            continue
        reach = max(sg * c[j] for _, c in ps) - inset
        pt_lo, pt_hi = min(c[1 - j] for _, c in ps), max(c[1 - j] for _, c in ps)
        far = [g for g in segs[s] if reach - 0.1 <= g["offset"] <= reach + 1.0 and
               min(g["t_hi"], pt_hi + 0.5) - max(g["t_lo"], pt_lo - 0.5) > 0.2]
        depth_how = "doorway photo position"
        if far:
            reach, depth_how = min(g["offset"] for g in far), "seen wall beyond the doorway photo"
        elif len(ps) < 2:
            # one doorway photo and no wall seen behind it: a single point that may be its own placement error
            # (door shots are placed to 0.03-0.54 m on the sim) does not change the shape
            notes.append(dict(kind="alcove_rejected", side=s, photos=[n for n, _ in ps],
                              beyond_m=round(float(max(sg * c[j] for _, c in ps) - rect[s]), 3),
                              reason="one doorway photo and no wall seen beyond it: not corroborated"))
            continue
        if reach - rect[s] > _par(p, "poly_max_ext_m"):
            continue
        t_lo = min(c[1 - j] for _, c in ps) - 0.45
        t_hi = max(c[1 - j] for _, c in ps) + 0.45
        bound = {}
        for side_t, sgn_t in ((("+z" if j == 0 else "+x"), 1.0), (("-z" if j == 0 else "-x"), -1.0)):
            cand = []
            for g in segs[side_t]:
                pos = sgn_t * g["offset"]                 # perpendicular wall position on the tangent axis
                # its extent along OUR normal axis must lie (mostly) beyond the rectangle side
                a, b = (g["t_lo"], g["t_hi"])
                if sg > 0:
                    out_len = b - max(a, rect[s])
                else:
                    out_len = min(b, -rect[s]) - a
                if out_len < 0.3:
                    continue
                edge = t_hi if sgn_t > 0 else t_lo
                if 0 <= sgn_t * (pos - edge) + 0.45 <= 1.5:
                    cand.append(pos)
            if cand:
                bound[side_t] = min(cand) if sgn_t > 0 else max(cand)
        if ("+z" if j == 0 else "+x") in bound:
            t_hi = bound["+z" if j == 0 else "+x"]
        if ("-z" if j == 0 else "-x") in bound:
            t_lo = bound["-z" if j == 0 else "-x"]
        t_lo, t_hi = max(t_lo, lo_t), min(t_hi, hi_t)
        if t_hi - t_lo < _par(p, "poly_min_cell_m") or reach - rect[s] < _par(p, "poly_min_cell_m") * 0.75:
            continue
        # the spin photos must not have seen a wall in front of it (mostly blocked = not this room)
        st = _par(p, "poly_sample_m")
        ga = np.arange(rect[s] + st / 2, reach, st) * sg
        gb = np.arange(t_lo + st / 2, t_hi, st)
        Q = np.stack(np.meshgrid(ga, gb, indexing="ij"), -1).reshape(-1, 2)
        if j == 1:
            Q = Q[:, ::-1]
        f, b = visibility(data, Q, _par(p, "poly_vis_margin_m"))
        rec = dict(kind="alcove", side=s, photos=[n for n, _ in ps], depth_m=round(float(reach - rect[s]), 3),
                   span_m=round(float(t_hi - t_lo), 3), free=round(float(f.mean()), 2),
                   blocked=round(float(b.mean()), 2),
                   depth_from=depth_how, span_bounded_by_walls=sorted(bound))
        if b.mean() > _par(p, "poly_alcove_max_blocked"):
            notes.append(dict(rec, kind="alcove_rejected", reason="the spin photos saw a wall in front of it"))
            continue
        a_pos = add_line(j, sg * reach, s, min(far, key=lambda g: g["offset"]) if far else None, "alcove depth")
        t0 = add_line(1 - j, t_lo, s, None, "alcove span")
        t1 = add_line(1 - j, t_hi, s, None, "alcove span")
        alcoves.append(dict(rec, j=j, a=(min(sg * rect[s], a_pos), max(sg * rect[s], a_pos)), t=(t0, t1)))
    # notch candidate lines: the other seen walls of each side, inside the rectangle
    for s in SIDES:
        ax, sg = _side_axis(s)
        j = 0 if ax == 0 else 1
        for g in segs[s]:
            if not (p.layout_min_dist_m < g["offset"] < rect[s] - sep):
                continue
            if any(abs(sg * g["offset"] - q) < sep for q in lines[j]):
                continue
            add_line(j, sg * g["offset"], s, g, "seen wall")
    X = np.array(sorted(lines[0]))
    Z = np.array(sorted(lines[1]))
    nx, nz = len(X) - 1, len(Z) - 1
    x0, x1, z0, z1 = -rect["-x"], rect["+x"], -rect["-z"], rect["+z"]
    cx = 0.5 * (X[1:] + X[:-1])
    cz = 0.5 * (Z[1:] + Z[:-1])
    inside0 = (cx[:, None] > x0) & (cx[:, None] < x1) & (cz[None, :] > z0) & (cz[None, :] < z1)
    inside = inside0.copy()
    mods = []
    for al in alcoves:
        a0, a1 = al["a"]
        t0, t1 = al["t"]
        if al["j"] == 0:
            m = (cx[:, None] > a0) & (cx[:, None] < a1) & (cz[None, :] > t0) & (cz[None, :] < t1)
        else:
            m = (cz[None, :] > a0) & (cz[None, :] < a1) & (cx[:, None] > t0) & (cx[:, None] < t1)
        trial = inside | m
        if _connected(trial) and len(mods) < _par(p, "poly_max_mods"):
            inside = trial
            mods.append({k: v for k, v in al.items() if k not in ("j", "a", "t")})
    # 4b. notches: border cells of the rectangle, blocked behind a seen wall, never seen free
    wcell = _par(p, "poly_min_cell_m")

    def wall_cov(j, pos, lo, hi, facing):
        if hi - lo <= 0:
            return 0.0
        sel = (np.abs(W[:, j] - pos) < p.layout_inlier_m) & (W[:, 1 - j] > lo) & (W[:, 1 - j] < hi) \
            & (N[:, j] * facing > p.layout_normal_min)
        cells = np.unique(np.floor(W[sel, 1 - j] / p.layout_cell_m))
        return float(min(len(cells) * p.layout_cell_m / (hi - lo), 1.0))
    cams_cell = {(int(np.searchsorted(X, c[0]) - 1), int(np.searchsorted(Z, c[1]) - 1)) for c in cams}
    st = _par(p, "poly_sample_m")
    free = np.full((nx, nz), np.nan)
    blk = np.full((nx, nz), np.nan)
    for i in range(nx):
        for k in range(nz):
            if len(mods) >= _par(p, "poly_max_mods"):
                break
            if not inside0[i, k] or (i, k) in cams_cell:
                continue
            dx, dz = X[i + 1] - X[i], Z[k + 1] - Z[k]
            if min(dx, dz) < wcell or dx * dz < _par(p, "poly_notch_min_area_m2"):
                continue
            nb4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
            border = [(di, dk) for (di, dk) in nb4
                      if not (0 <= i + di < nx and 0 <= k + dk < nz and inside0[i + di, k + dk])]
            inner = [(di, dk) for (di, dk) in nb4 if (0 <= i + di < nx and 0 <= k + dk < nz and inside[i + di, k + dk])]
            if len(border) < 2 or not inner:        # a notch cuts a CORNER region of the rectangle
                continue
            gx = np.arange(X[i] + st / 2, X[i + 1], st)
            gz = np.arange(Z[k] + st / 2, Z[k + 1], st)
            Q = np.stack(np.meshgrid(gx, gz, indexing="ij"), -1).reshape(-1, 2)
            f, b = visibility(data, Q, _par(p, "poly_vis_margin_m"))
            free[i, k], blk[i, k] = f.mean(), b.mean()
            if blk[i, k] < _par(p, "poly_notch_blocked") or free[i, k] > _par(p, "poly_notch_free"):
                continue
            covs = []
            for (di, dk) in inner:
                if di:
                    covs.append(wall_cov(0, X[i + 1] if di > 0 else X[i], Z[k], Z[k + 1], di))
                else:
                    covs.append(wall_cov(1, Z[k + 1] if dk > 0 else Z[k], X[i], X[i + 1], dk))
            if min(covs) < _par(p, "poly_notch_wall_cover"):
                continue
            trial = inside.copy()
            trial[i, k] = False
            if not _connected(trial):
                continue
            inside = trial
            mods.append(dict(kind="notch", cell=[int(i), int(k)], free=round(float(free[i, k]), 2),
                             blocked=round(float(blk[i, k]), 2), wall_cover=round(float(min(covs)), 2),
                             size_m=[round(float(dx), 2), round(float(dz), 2)]))
    if not _connected(inside):
        inside, mods = inside0, []
    verts = trace(inside)
    if verts is None:
        return dict(ok=False, reason="boundary trace failed")

    def line_rec(j, pos):
        side, seg, kind = lines[j][pos]
        meas = seg is not None or (kind == "rectangle" and status[side] == "measured")
        return dict(offset=float(pos), side=side, kind=kind, status="measured" if meas else "inferred",
                    sigma=_line_sigma(seg, sig[side]))
    xl = [line_rec(0, q) for q in X]
    zl = [line_rec(1, q) for q in Z]
    V = [[xl[i]["offset"], zl[k]["offset"]] for i, k in verts]
    edges = []
    for a in range(len(verts)):
        (i0, k0), (i1, k1) = verts[a], verts[(a + 1) % len(verts)]
        ln = xl[i0] if i0 == i1 else zl[k0]
        edges.append(dict(axis="x" if i0 == i1 else "z", line=int(i0 if i0 == i1 else k0), status=ln["status"],
                          sigma=ln["sigma"], kind=ln["kind"]))
    area = _shoelace(np.array(V))
    used = bool(mods) or any(n["kind"] in ("door_photo_extends_inferred_side", "side_replaced") for n in notes)
    return dict(ok=True, used=used, vertices=V, vertex_lines=[[int(i), int(k)] for i, k in verts], edges=edges,
                x_lines=xl, z_lines=zl, inside=inside.astype(int).tolist(), changes=mods + notes,
                n_vertices=len(V), area_m2=round(float(area), 3),
                rect_area_m2=round(float((S["+x"]["offset"] + S["-x"]["offset"])
                                         * (S["+z"]["offset"] + S["-z"]["offset"])), 3),
                sides={s: dict(offset=rect[s], status=status[s], sigma=sig[s]) for s in SIDES},
                photos=data["names"], door_photos=sorted(dcams))


def _connected(inside: np.ndarray) -> bool:
    from scipy import ndimage as ndi
    if not inside.any():
        return False
    _, n = ndi.label(inside)
    return n == 1


def _shoelace(V: np.ndarray) -> float:
    x, y = V[:, 0], V[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def trace(inside: np.ndarray):
    """Counter-clockwise boundary (in (x, z) with x right, z up) of a 4-connected cell set as grid-corner indices
    (i, k); collinear corners removed. None if the set has holes or is not one piece."""
    nx, nz = inside.shape
    E = {}
    def add(a, b):
        E.setdefault(a, []).append(b)
    for i in range(nx):
        for k in range(nz):
            if not inside[i, k]:
                continue
            if k == 0 or not inside[i, k - 1]:
                add((i, k), (i + 1, k))              # bottom edge, left -> right
            if i == nx - 1 or not inside[i + 1, k]:
                add((i + 1, k), (i + 1, k + 1))      # right edge, up
            if k == nz - 1 or not inside[i, k + 1]:
                add((i + 1, k + 1), (i, k + 1))      # top edge, right -> left
            if i == 0 or not inside[i - 1, k]:
                add((i, k + 1), (i, k))              # left edge, down
    if not E:
        return None
    start = min(E)
    loop, cur, prev_dir = [start], start, None
    n_edges = sum(len(v) for v in E.values())
    for _ in range(n_edges + 1):
        nxt = E[cur]
        if len(nxt) > 1:                              # touching corners: keep turning left (CCW)
            nxt = sorted(nxt, key=lambda q: _turn(prev_dir, (q[0] - cur[0], q[1] - cur[1])))
        q = nxt.pop(0)
        if not E[cur]:
            del E[cur]
        prev_dir = (q[0] - cur[0], q[1] - cur[1])
        cur = q
        if cur == start:
            break
        loop.append(cur)
    if E:                                             # more than one loop: hole or second piece
        return None
    out = []
    m = len(loop)
    for a in range(m):
        p0, p1, p2 = loop[a - 1], loop[a], loop[(a + 1) % m]
        if (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0]) != 0:
            out.append(p1)
    return out


def _turn(prev, d):
    if prev is None:
        return 0
    cross = prev[0] * d[1] - prev[1] * d[0]
    return 0 if cross > 0 else 1 if cross == 0 else 2


# ------------------------------------------------------------------------------------------------ entry point
def room_polygon(lay: dict, views: dict, group: list[str], yaws: dict, poses: dict, room_photos: list[str],
                 door_photos: list[str], scale: float, p, log=None) -> dict:
    """lay["polygon_local"] for one room (called by layout.room_layouts after the rectangle fit)."""
    if not _par(p, "layout_polygon"):
        return dict(ok=False, reason="disabled")
    if not lay.get("ok"):
        return dict(ok=False, reason="no rectangle")
    if lay.get("small_room"):
        return dict(ok=False, reason="small room (threshold box): rectangle kept")
    yaws_l = {n: yaws[n] for n in group if n in yaws}
    yaws2, offs = room_photo_frame(group, yaws_l, poses, [n for n in room_photos if n not in yaws_l], p)
    data = collect(views, yaws2, offs, scale, p, lay)
    if not data.get("ok") or len(data["W"]) < p.layout_min_wall_pts:
        return dict(ok=False, reason="too few wall points")
    out = fit_polygon(lay, data, p, door_photos=[d for d in door_photos if d in yaws2 and d not in yaws_l], log=log)
    out["photo_offsets"] = {n: [round(float(offs[n][0]), 3), round(float(offs[n][2]), 3)] for n in yaws2}
    return out


# ------------------------------------------------------------------------------------------------ plan side (tiers)
_E = {"+x": (0, 1.0), "-x": (0, -1.0), "+z": (1, 1.0), "-z": (1, -1.0)}


def plan_polygon(lay: dict, rel: float, S: int, rng) -> dict | None:
    """polygon_local (layout frame) -> the plan frame (u, v) relative to the room centre, with Monte Carlo draws of
    every wall line. Same calibration as the rectangle sides in tiers.layout_plan: sigma = hypot(line sigma,
    rel * distance from the centre). Returns None when there is no usable polygon."""
    pl = lay.get("polygon_local") or {}
    if not pl.get("ok") or not pl.get("used"):
        return None
    amap = (lay.get("anchor") or {}).get("side_to_plan") or {s: s for s in SIDES}
    ax_x, sx = _E[amap["+x"]]
    ax_z, sz = _E[amap["+z"]]
    if ax_x == ax_z:
        return None
    lines = {ax_x: [(sx * q["offset"], q) for q in pl["x_lines"]], ax_z: [(sz * q["offset"], q) for q in pl["z_lines"]]}
    inside = np.array(pl["inside"], bool)                       # (len(x_lines)-1, len(z_lines)-1)
    if sx < 0:
        inside = inside[::-1, :]
    if sz < 0:
        inside = inside[:, ::-1]
    if ax_x == 1:                                               # layout x -> plan v: transpose
        inside = inside.T
    U = sorted(lines[0], key=lambda q: q[0])
    V = sorted(lines[1], key=lambda q: q[0])
    verts = trace(inside)
    if verts is None:
        return None

    def draws(pos, q):
        sig = float(np.hypot(q["sigma"], rel * abs(pos)))
        return rng.normal(pos, sig, S)
    DU = np.sort(np.stack([draws(pos, q) for pos, q in U], 1), 1)     # keep the line order in every draw
    DV = np.sort(np.stack([draws(pos, q) for pos, q in V], 1), 1)
    # plan side key of every rectangle side line, to share draws with the rectangle logic
    side_line = {}
    for a, (pos, q) in enumerate(U):
        if q["kind"] == "rectangle":
            side_line[amap[q["side"]]] = ("u", a)
    for a, (pos, q) in enumerate(V):
        if q["kind"] == "rectangle":
            side_line[amap[q["side"]]] = ("v", a)
    return dict(u=np.array([q[0] for q in U]), v=np.array([q[0] for q in V]), umeta=[q[1] for q in U],
                vmeta=[q[1] for q in V], DU=DU, DV=DV, inside=inside, verts=verts, side_line=side_line,
                changes=pl.get("changes", []), n_vertices=len(verts))


def sync_sides(pg: dict, val: dict, draws: dict) -> None:
    """Rectangle side lines of the polygon <- the room's side values/draws (after the neighbour clip): one truth."""
    for s, (ax, a) in pg["side_line"].items():
        if s not in val:
            continue
        sg = 1.0 if s[0] == "+" else -1.0
        if ax == "u":
            pg["u"][a] = sg * val[s]
            pg["DU"][:, a] = sg * draws[s]
        else:
            pg["v"][a] = sg * val[s]
            pg["DV"][:, a] = sg * draws[s]
    for k in ("DU", "DV"):
        pg[k] = np.sort(pg[k], 1)


def parts(pg: dict) -> list[tuple]:
    """Inside cells as rectangles (u0, u1, v0, v1) relative to the centre."""
    out = []
    for a in range(pg["inside"].shape[0]):
        for b in range(pg["inside"].shape[1]):
            if pg["inside"][a, b]:
                out.append((pg["u"][a], pg["u"][a + 1], pg["v"][b], pg["v"][b + 1]))
    return out


def plan_geometry(pg: dict, c: np.ndarray):
    """Vertices (absolute plan coordinates), per-edge (p0, p1, inward normal, length draws, value, status) and the
    area / perimeter / bbox draws of the polygon room."""
    vi = pg["verts"]
    P = np.array([[pg["u"][a], pg["v"][b]] for a, b in vi]) + c
    PU = pg["DU"][:, [a for a, _ in vi]]
    PV = pg["DV"][:, [b for _, b in vi]]
    area = 0.5 * np.abs((PU * np.roll(PV, -1, 1) - PV * np.roll(PU, -1, 1)).sum(1))
    edges = []
    m = len(vi)
    for k in range(m):
        (a0, b0), (a1, b1) = vi[k], vi[(k + 1) % m]
        L = np.abs(PU[:, (k + 1) % m] - PU[:, k]) + np.abs(PV[:, (k + 1) % m] - PV[:, k])
        d = P[(k + 1) % m] - P[k]
        nrm = (-float(np.sign(d[1])), float(np.sign(d[0])))      # CCW polygon: interior on the left
        if a0 == a1:      # along a u-line: its length is fixed by the v-lines at both ends
            own, ends = pg["umeta"][a0], (pg["vmeta"][b0], pg["vmeta"][b1])
        else:
            own, ends = pg["vmeta"][b0], (pg["umeta"][a0], pg["umeta"][a1])
        st = "measured" if all(e["status"] == "measured" for e in ends) else "inferred"
        edges.append(dict(p0=tuple(map(float, P[k])), p1=tuple(map(float, P[(k + 1) % m])), normal=nrm, draws=L,
                          value=float(np.abs(d).sum()), status=st, line_status=own["status"], kind=own["kind"]))
    perim = sum(e["draws"] for e in edges)
    du = pg["DU"][:, -1] - pg["DU"][:, 0]
    dv = pg["DV"][:, -1] - pg["DV"][:, 0]
    return P, edges, area, perim, du, dv


def push_apart_parts(boxes: dict, gap: float, iters: int = 200, fixed: list | None = None) -> dict:
    """tiers._push_apart for rooms made of several rectangles (polygon rooms); a rectangle room is one part. The same
    rules: only real overlaps, the least certain room moves (not anchored, larger widening, smaller), along the axis
    of least penetration of the most overlapping pair of parts."""
    def rects(f):
        b = boxes[f]
        c = b["c"]
        if b.get("poly") is not None:
            return [(c[0] + u0, c[0] + u1, c[1] + v0, c[1] + v1) for u0, u1, v0, v1 in parts(b["poly"])]
        return [(c[0] - b["val"]["-x"], c[0] + b["val"]["+x"], c[1] - b["val"]["-z"], c[1] + b["val"]["+z"])]

    def bbox(rs):
        r = np.array(rs)
        return (r[:, 0].min(), r[:, 1].max(), r[:, 2].min(), r[:, 3].max())

    def worst(RA, RB):
        best = None
        for A in RA:
            for B in RB:
                ou, ov = min(A[1], B[1]) - max(A[0], B[0]), min(A[3], B[3]) - max(A[2], B[2])
                if ou > 1e-3 and ov > 1e-3 and (best is None or ou * ov > best[0]):
                    best = (ou * ov, ou, ov, A, B)
        return best

    shift = {f: np.zeros(2) for f in boxes}
    names = sorted(boxes)
    for _ in range(iters):
        moved = False
        for i, a in enumerate(names):
            for bname in names[i + 1:]:
                w = worst(rects(a), rects(bname))
                if w is None:
                    continue
                _, ou, ov, _, _ = w
                pu, pv = ou + gap, ov + gap

                def key(f):
                    bb = bbox(rects(f))
                    return (boxes[f]["anchored"], -boxes[f]["widen"], (bb[1] - bb[0]) * (bb[3] - bb[2]))
                mover, other = (a, bname) if key(a) < key(bname) else (bname, a)
                w = worst(rects(mover), rects(other))
                _, ou, ov, M, O = w
                pu, pv = ou + gap, ov + gap
                d = np.zeros(2)
                if pu <= pv:
                    d[0] = pu if (M[0] + M[1]) >= (O[0] + O[1]) else -pu
                else:
                    d[1] = pv if (M[2] + M[3]) >= (O[2] + O[3]) else -pv
                boxes[mover]["c"] = boxes[mover]["c"] + d
                shift[mover] += d
                moved = True
        for f in names:
            for O in fixed or []:
                w = worst(rects(f), [O])
                if w is None:
                    continue
                _, ou, ov, M, _ = w
                d = np.zeros(2)
                if ou <= ov:
                    d[0] = (ou + gap) if (M[0] + M[1]) >= (O[0] + O[1]) else -(ou + gap)
                else:
                    d[1] = (ov + gap) if (M[2] + M[3]) >= (O[2] + O[3]) else -(ov + gap)
                boxes[f]["c"] = boxes[f]["c"] + d
                shift[f] += d
                moved = True
        if not moved:
            break
    return {f: round(float(np.hypot(*s)), 3) for f, s in shift.items()}
