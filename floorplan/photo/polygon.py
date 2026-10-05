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
     by a wall seen just beyond; not if the spin photos saw that region mostly blocked. D-082 (poly_ext_walls): the
     alcove is measured by its own walls (_alcove_walls: span cut to the open stretch of the side's wall line, depth
     from the far wall seen anywhere in it) and an alcove with no doorway photo in it is found from its walls alone
     (_open_mouths: a gap in the side's wall line, the alcove's side wall running out beyond it, no door head);
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
    poly_ext_walls=True,             # D-082: alcoves measured by their own walls (_alcove_walls)
    poly_open_mouths=False,          # D-082: alcoves with no photo in them (_open_mouths). Off: a fresh k38 run
                                     #     turned the balcony door (reveal 0.301 m) into a fake step
    poly_alcove_side_wall_m=0.3,     # ... a perpendicular wall seen this far beyond the side at the span's edge is
                                     #     the alcove's own side wall (a door reveal is one wall: sim 0.24 m, seen
                                     #     as 0.24-0.30 m; the k65 lobby's wall 0.37 m)
    poly_alcove_mouth_open=0.5,      # ... the side's own wall seen over more of the span than this: no mouth there
    poly_alcove_mouth_m=0.4,         # ... "in front of a wall" is judged on this strip just past the side line
    poly_alcove_min_free=0.2,        # ... and >= 20% of the alcove seen free by some photo (rays went into it)
    poly_head_lo_m=2.25,             # D-082 open mouth: above a door's head (sim doors 2.03-2.06 m high; k65's
                                     # balcony door 2.36 m) ...
    poly_head_hi_m=2.5,              # ... and below the ceiling (sim 2.4-2.8 m) a door has wall, an open alcove not
    poly_mouth_min_m=0.5,            # an open mouth in a side's wall line: 0.5-2.5 m wide, wall seen on both sides
    poly_mouth_max_m=2.5,
    poly_mouth_open_above=0.3,       # ... rays to points above door-head height crossed >= 30% of it, in plan view
                                     # (a ray through a door to a wall far beyond counts too: k38 balcony door 0.84)
    poly_mouth_head_max=0.2,         # ... and a wall at the side line up there covers <= 20% of it (no door head;
                                     # the k38 balcony door and a k22 bedroom door read 0.0 although they have one)
    poly_mouth_max_depth_m=1.5,      # ... an alcove, not a room: its far boundary within 1.5 m of the side
    poly_notch_wall_cover=0.4,       # a notch's walls toward the room must be >= 40% seen
    poly_side_min_overlap_m=0.3,     # a measured side's seen wall must overlap the room's extent along it by 0.3 m
    poly_skip_far_wall_lines=True,   # no notch line on a surface the far-wall rule set aside as furniture (D-077)
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
    H, HN, hid = [], [], []                             # D-082: points above door-head height (any surface)
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
        hi = np.flatnonzero((h > _par(p, "poly_head_lo_m")) & (h < _par(p, "poly_head_hi_m")))
        H.append((P[hi] @ Rm.T)[:, [0, 2]])
        HN.append((N[hi] @ Rm.T)[:, [0, 2]])
        hid.append(np.full(len(hi), len(names)))
        cams.append(tc[[0, 2]])
        vis.append(R)
        names.append(n)
    if not names:
        return dict(ok=False)
    return dict(ok=True, W=np.concatenate(W), N=np.concatenate(NW), pid=np.concatenate(pid), cams=np.array(cams),
                vis=np.array(vis), names=names, nb=nb, H=np.concatenate(H), HN=np.concatenate(HN),
                hid=np.concatenate(hid))


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
    # D-082 (poly_ext_walls): the same alcove measured by its own walls first (_alcove_walls): the span cut to the
    # mouth (where the side's wall line is open), the depth from the far wall seen anywhere in the mouth.
    alcoves = []
    v2 = bool(_par(p, "poly_ext_walls"))
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
        if v2:                                   # D-082 first; its rejections are final, "no note" leaves it to v1
            al = _alcove_walls(s, ps, rect, segs, W, N, data, p, (lo_t, hi_t), inset)
            if al.get("ok"):
                rec = {k: v for k, v in al.items() if k not in ("ok", "far", "t_lo", "t_hi", "reach")}
                a_pos = add_line(j, sg * al["reach"], s, al["far"], "alcove depth")
                t0 = add_line(1 - j, al["t_lo"], s, None, "alcove span")
                t1 = add_line(1 - j, al["t_hi"], s, None, "alcove span")
                alcoves.append(dict(rec, j=j, a=(min(sg * rect[s], a_pos), max(sg * rect[s], a_pos)), t=(t0, t1)))
                continue
            if al.get("note"):
                notes.append(al["note"])
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
    # D-082: open mouths: a gap in a measured side's wall line, the alcove's own wall beyond it and no door head
    # above it: the room continues there, without a doorway photo in it (_open_mouths; off by default, see DEFAULTS)
    if v2 and _par(p, "poly_open_mouths"):
        for s in SIDES:
            if status[s] != "measured" or any(al["side"] == s for al in alcoves):
                continue
            ax, sg = _side_axis(s)
            j = 0 if ax == 0 else 1
            for m in _open_mouths(s, rect, segs, W, N, data, p, t_range(j)):
                if not m.get("ok"):
                    notes.append(m["note"])
                    continue
                rec = {k: v for k, v in m.items() if k not in ("ok", "far", "t_lo", "t_hi", "reach")}
                a_pos = add_line(j, sg * m["reach"], s, m["far"], "alcove depth")
                t0 = add_line(1 - j, m["t_lo"], s, None, "alcove span")
                t1 = add_line(1 - j, m["t_hi"], s, None, "alcove span")
                alcoves.append(dict(rec, j=j, a=(min(sg * rect[s], a_pos), max(sg * rect[s], a_pos)), t=(t0, t1)))
    # notch candidate lines: the other seen walls of each side, inside the rectangle
    for s in SIDES:
        ax, sg = _side_axis(s)
        j = 0 if ax == 0 else 1
        # D-077 set this surface aside as furniture in front of the wall (own Room: the wardrobe front, an open door
        # leaf): it bounds no notch (the false wardrobe-corner notch, lit 0.69 x 0.70 m, dim 1.01 x 0.63 m)
        fw = (S[s].get("far_wall") or {}).get("from_m") if _par(p, "poly_skip_far_wall_lines") else None
        for g in segs[s]:
            if not (p.layout_min_dist_m < g["offset"] < rect[s] - sep):
                continue
            if any(abs(sg * g["offset"] - q) < sep for q in lines[j]):
                continue
            if fw is not None and abs(g["offset"] - fw) <= 2 * p.layout_bin_m:
                notes.append(dict(kind="notch_line_skipped", side=s, offset_m=round(g["offset"], 3),
                                  reason="the far-wall rule set this surface aside as furniture (D-077)"))
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


def _alcove_walls(s: str, ps: list, rect: dict, segs: dict, W: np.ndarray, N: np.ndarray, data: dict, p,
                  room_t: tuple, inset: float) -> dict:
    """D-082: an alcove beyond MEASURED side s, measured by its own walls. ps: this room's doorway photo(s) standing
    >= poly_alcove_min_beyond_m beyond the side (as v1). Layout frame; j = the side's normal axis.

      span:  the photo(s) +- half a door, stretched to the nearest perpendicular wall seen beyond the side within
             1.5 m (as v1), then cut to the MOUTH: the longest stretch of it where the side's own wall line was not
             seen (an alcove opens over its whole width; behind a seen wall there is none);
      walls: a perpendicular wall at the span's edge that runs >= poly_alcove_side_wall_m beyond the side is the
             alcove's own side wall (a door reveal is one wall thickness, seen as 0.24-0.30 m on the sim); a wall of
             this side seen beyond the photo inside the mouth is its far wall;
      depth: the far wall (sim k38 living room: seen 0.4 m behind the doorway photos, which v1 looked for only
             beside the photos); else, with two doorway photos, the farthest the side walls reach or the photos (- half
             a wall), whichever is farther;
      kept:  with a far wall or two doorway photos (as v1; a side wall alone is not enough: a door's reveals look the
             same, sim k38 bedroom); the photo(s) within half a door of the mouth; the strip just past the side line
             (poly_alcove_mouth_m) seen behind a wall on <= poly_alcove_max_blocked of it (the alcove's far corners
             may hide behind its own side walls) and >= poly_alcove_min_free of the alcove seen free.
    Returns dict(ok=True, reach, t_lo, t_hi, far, record fields) or dict(ok=False, note=None or a rejection); with
    ok False and no note, v1 decides."""
    ax, sg = _side_axis(s)
    j = 0 if ax == 0 else 1
    lo_t, hi_t = room_t
    cell = p.layout_cell_m
    names = [n for n, _ in ps]
    reach = max(sg * c[j] for _, c in ps) - inset
    pt_lo, pt_hi = min(c[1 - j] for _, c in ps), max(c[1 - j] for _, c in ps)
    t_lo, t_hi = pt_lo - 0.45, pt_hi + 0.45
    s_hi, s_lo = ("+z", "-z") if j == 0 else ("+x", "-x")
    walls = {}
    for side_t, sgn_t in ((s_hi, 1.0), (s_lo, -1.0)):
        cand = []
        for g in segs[side_t]:
            pos = sgn_t * g["offset"]                     # perpendicular wall position on the tangent axis
            a, b = g["t_lo"], g["t_hi"]                   # its extent along OUR normal axis
            out_len = (b - max(a, rect[s])) if sg > 0 else (min(b, -rect[s]) - a)
            edge = t_hi if sgn_t > 0 else t_lo
            if out_len >= 0.3 and 0 <= sgn_t * (pos - edge) + 0.45 <= 1.5:
                cand.append((pos, float(out_len), float(b if sg > 0 else -a)))
        if cand:
            walls[side_t] = min(cand, key=lambda q: sgn_t * q[0])
    if s_hi in walls:
        t_hi = walls[s_hi][0]
    if s_lo in walls:
        t_lo = walls[s_lo][0]
    t_lo, t_hi = max(t_lo, lo_t), min(t_hi, hi_t)
    rej = dict(kind="alcove_rejected", side=s, photos=names, rule="D-082 own walls")
    # the mouth: the side's own wall line must be open there
    on = (np.abs(sg * W[:, j] - rect[s]) < p.layout_inlier_m) & (N[:, j] * sg < -p.layout_normal_min)
    kc, nc = np.unique(np.floor(W[on, 1 - j] / cell).astype(int), return_counts=True)
    seen = set(kc[nc >= 2].tolist())                      # >= 2 points: one stray point does not close a mouth
    ks = np.arange(int(np.floor(t_lo / cell)), int(np.ceil(t_hi / cell)))
    best, run = None, None
    for k in ks:
        if k in seen:
            run = None
            continue
        run = [k, k] if run is None else [run[0], k]
        if best is None or run[1] - run[0] > best[1] - best[0]:
            best = list(run)
    if best is None or (best[1] + 1 - best[0]) * cell < _par(p, "poly_min_cell_m"):
        return dict(ok=False, note=dict(rej, reason="the side's own wall was seen across the span: no mouth"))
    covered = 1.0 - (best[1] + 1 - best[0]) * cell / max(t_hi - t_lo, 1e-9)
    if covered > _par(p, "poly_alcove_mouth_open"):
        return dict(ok=False, note=dict(rej, reason=f"the side's own wall was seen on {covered:.0%} of the span"))
    t_lo, t_hi = max(t_lo, best[0] * cell), min(t_hi, (best[1] + 1) * cell)
    if pt_lo < t_lo - 0.45 or pt_hi > t_hi + 0.45:
        return dict(ok=False, note=dict(rej, reason="the doorway photo stands more than half a door from the mouth"))
    side_walls = {k: w for k, w in walls.items() if w[1] >= _par(p, "poly_alcove_side_wall_m")
                  and (abs(w[0] - t_lo) <= 0.15 or abs(w[0] - t_hi) <= 0.15)}
    far = [g for g in segs[s] if reach - 0.1 <= g["offset"] <= reach + 1.0
           and min(g["t_hi"], t_hi) - max(g["t_lo"], t_lo) > 0.2]
    if far:
        g = min(far, key=lambda g: g["offset"])
        depth, how = g["offset"], "far wall seen in the alcove's mouth"
    elif len(ps) >= 2:
        g = None
        depth = max([reach] + [w[2] for w in side_walls.values()])
        how = "farthest seen point of the alcove's side wall(s) or the doorway photos"
    else:
        return dict(ok=False, note=None)                  # one photo and no far wall: v1 decides (not corroborated)
    if depth - rect[s] > _par(p, "poly_max_ext_m"):
        return dict(ok=False, note=dict(rej, reason="deeper than poly_max_ext_m: the next room"))
    if t_hi - t_lo < _par(p, "poly_min_cell_m") or depth - rect[s] < _par(p, "poly_min_cell_m") * 0.75:
        return dict(ok=False, note=dict(rej, reason="narrower or shallower than poly_min_cell_m once cut to the mouth"))
    st = _par(p, "poly_sample_m")
    ga = np.arange(rect[s] + st / 2, depth, st) * sg
    gb = np.arange(t_lo + st / 2, t_hi, st)
    Q = np.stack(np.meshgrid(ga, gb, indexing="ij"), -1).reshape(-1, 2)
    if j == 1:
        Q = Q[:, ::-1]
    f, b = visibility(data, Q, _par(p, "poly_vis_margin_m"))
    strip = np.abs(Q[:, j] - sg * rect[s]) < _par(p, "poly_alcove_mouth_m")
    bm = float(b[strip].mean()) if strip.any() else 1.0
    rec = dict(kind="alcove", side=s, photos=names, depth_m=round(float(depth - rect[s]), 3),
               span_m=round(float(t_hi - t_lo), 3), free=round(float(f.mean()), 2), blocked=round(float(b.mean()), 2),
               blocked_mouth=round(bm, 2), depth_from=how, span_bounded_by_walls=sorted(walls),
               side_walls=sorted(side_walls), mouth_m=[round(float(t_lo), 3), round(float(t_hi), 3)],
               rule="D-082 own walls")
    if bm > _par(p, "poly_alcove_max_blocked"):
        return dict(ok=False, note=dict(rec, kind="alcove_rejected", reason="the spin photos saw a wall in front of it"))
    if f.mean() < _par(p, "poly_alcove_min_free"):
        return dict(ok=False, note=dict(rec, kind="alcove_rejected", reason="no photo saw into it (not seen free)"))
    return dict(rec, ok=True, reach=float(depth), t_lo=float(t_lo), t_hi=float(t_hi), far=g)


def _open_mouths(s: str, rect: dict, segs: dict, W: np.ndarray, N: np.ndarray, data: dict, p, room_t: tuple) -> list:
    """D-082: alcoves of MEASURED side s shown by the walls alone (no doorway photo in them). Layout frame.

      mouth: a gap of poly_mouth_min_m..poly_mouth_max_m in the side's seen wall line, the wall seen on both sides of
             it (sim k65 living room: the west wall seen on both sides of the 1.0 m lobby);
      walls: a perpendicular wall at an end of the gap, facing into it, seen >= poly_alcove_side_wall_m beyond the
             side (the alcove's own side wall);
      no door head: a door has wall above its head (sim doors 2.03-2.06 m, k65's balcony door 2.36 m); rays of
             points above poly_head_lo_m crossed >= poly_mouth_open_above of the gap (in plan view, so a ray through
             a door to a far wall counts too), and a wall surface on the side line up there covers
             <= poly_mouth_head_max of it (the k38 balcony door and a k22 bedroom door read 0.0: in practice the
             side wall length is what keeps doors out);
      depth: a wall of this side seen in the mouth within poly_mouth_max_depth_m (far wall), else the farthest point
             of the side wall(s) (a lower bound; the edge is inferred);
      kept when >= poly_alcove_min_free of it was seen free and the strip just past the side line
             (poly_alcove_mouth_m) is <= poly_alcove_max_blocked behind a wall.
    Returns a list of dict(ok, reach, t_lo, t_hi, far, record fields) or dict(note) for rejected mouths."""
    ax, sg = _side_axis(s)
    j = 0 if ax == 0 else 1
    lo_t, hi_t = room_t
    cell = p.layout_cell_m
    d, t = sg * W[:, j] - rect[s], W[:, 1 - j]
    on = (np.abs(d) < p.layout_inlier_m) & (N[:, j] * sg < -p.layout_normal_min) & (t > lo_t) & (t < hi_t)
    kc, nc = np.unique(np.floor(t[on] / cell).astype(int), return_counts=True)
    seen = np.sort(kc[nc >= 2])
    H, HN, hid = data["H"], data["HN"], data["hid"]
    dh, th = sg * H[:, j] - rect[s], H[:, 1 - j]
    cam = data["cams"][hid] if len(hid) else np.zeros((0, 2))
    dc, tcm = sg * cam[:, j] - rect[s], cam[:, 1 - j]
    max_d = _par(p, "poly_mouth_max_depth_m")
    out = []
    for k1, k2 in zip(seen[:-1], seen[1:]):
        g0, g1 = float((k1 + 1) * cell), float(k2 * cell)          # the gap between two seen stretches of wall
        if not (_par(p, "poly_mouth_min_m") <= g1 - g0 <= _par(p, "poly_mouth_max_m")):
            continue
        rej = dict(kind="alcove_rejected", side=s, mouth_m=[round(g0, 3), round(g1, 3)], rule="D-082 open mouth")
        ends = {}
        for edge, face in ((g0, 1.0), (g1, -1.0)):
            sel = (np.abs(t - edge) < 0.15) & (N[:, 1 - j] * face > p.layout_normal_min) & (d > 0.05) & (d < max_d)
            if sel.sum() >= 5:
                ends[f"{edge:.2f}"] = round(float(np.percentile(d[sel], 95)), 3)
        walls = {e: x for e, x in ends.items() if x >= _par(p, "poly_alcove_side_wall_m")}
        if not walls:
            continue                                    # most gaps: a door with its reveals, or wall nobody saw
        beyond = (dh > 0.15) & (dc < 0)
        lam = -dc[beyond] / np.maximum(dh[beyond] - dc[beyond], 1e-9)
        tx = tcm[beyond] + lam * (th[beyond] - tcm[beyond])          # where those rays crossed the side line
        gc = set(range(int(round(g0 / cell)), int(round(g1 / cell))))
        f_open = len(set(np.floor(tx / cell).astype(int).tolist()) & gc) / max(len(gc), 1)
        head = (np.abs(dh) < 0.15) & (HN[:, j] * sg < -p.layout_normal_min) & (th > g0) & (th < g1)
        f_head = len(set(np.floor(th[head] / cell).astype(int).tolist()) & gc) / max(len(gc), 1)
        rej.update(open_above=round(f_open, 2), head=round(f_head, 2), side_walls_m=ends)
        if f_open < _par(p, "poly_mouth_open_above") or f_head > _par(p, "poly_mouth_head_max"):
            out.append(dict(ok=False, note=dict(rej, reason="not seen open above door-head height (a door, or "
                                                             "unseen): the space beyond may be the next room")))
            continue
        far = [g for g in segs[s] if rect[s] + 0.3 <= g["offset"] <= rect[s] + max_d
               and min(g["t_hi"], g1) - max(g["t_lo"], g0) > 0.2]
        if far:
            g = min(far, key=lambda g: g["offset"])
            depth, how = g["offset"], "far wall seen in the mouth"
        else:
            g, depth = None, rect[s] + max(walls.values())
            how = "farthest seen point of the alcove's side wall (a lower bound: its far wall was not seen)"
        if depth - rect[s] < _par(p, "poly_min_cell_m") * 0.75:
            continue
        st = _par(p, "poly_sample_m")
        ga = np.arange(rect[s] + st / 2, depth, st) * sg
        gb = np.arange(g0 + st / 2, g1, st)
        Q = np.stack(np.meshgrid(ga, gb, indexing="ij"), -1).reshape(-1, 2)
        if j == 1:
            Q = Q[:, ::-1]
        f, b = visibility(data, Q, _par(p, "poly_vis_margin_m"))
        strip = np.abs(Q[:, j] - sg * rect[s]) < _par(p, "poly_alcove_mouth_m")
        bm = float(b[strip].mean()) if strip.any() else 1.0
        rec = dict(kind="alcove", side=s, photos=[], depth_m=round(float(depth - rect[s]), 3),
                   span_m=round(g1 - g0, 3), free=round(float(f.mean()), 2), blocked=round(float(b.mean()), 2),
                   blocked_mouth=round(bm, 2), depth_from=how, mouth_m=[round(g0, 3), round(g1, 3)],
                   side_walls_m=ends, open_above=round(f_open, 2), head=round(f_head, 2), rule="D-082 open mouth")
        if bm > _par(p, "poly_alcove_max_blocked") or f.mean() < _par(p, "poly_alcove_min_free"):
            out.append(dict(ok=False, note=dict(rec, kind="alcove_rejected", reason="the photos did not see into it")))
            continue
        out.append(dict(rec, ok=True, reach=float(depth), t_lo=g0, t_hi=g1, far=g))
    return out


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
    if out.get("ok") and getattr(p, "pillars", True):          # D-083: pillars and wall steps (photo/pillars.py)
        try:
            from floorplan.photo.pillars import find_pillars
            res = find_pillars(lay, views, yaws2, offs, scale, p)
            cut_pillars(out, res.get("pillars", []), p)
            out["pillar_search"] = dict(candidates=res.get("candidates", 0), rejected=res.get("rejected", []))
        except Exception as e:                  # the polygon without pillars stands
            out["pillar_search"] = dict(error=f"{type(e).__name__}: {e}")
    return out


# ------------------------------------------------------------------------------------------------ pillars (D-083)
def _outline(xl: list[dict], zl: list[dict], inside: np.ndarray):
    """Vertices, vertex line indices, per-edge records and area of the cell set (same records as fit_polygon)."""
    verts = trace(inside)
    if verts is None:
        return None
    V = [[xl[i]["offset"], zl[k]["offset"]] for i, k in verts]
    edges = []
    for a in range(len(verts)):
        (i0, k0), (i1, k1) = verts[a], verts[(a + 1) % len(verts)]
        ln = xl[i0] if i0 == i1 else zl[k0]
        edges.append(dict(axis="x" if i0 == i1 else "z", line=int(i0 if i0 == i1 else k0), status=ln["status"],
                          sigma=ln["sigma"], kind=ln["kind"]))
    return V, [[int(i), int(k)] for i, k in verts], edges, float(_shoelace(np.array(V)))


def cut_pillars(pl: dict, pillars: list[dict], p) -> None:
    """Cut pillars / wall steps (photo/pillars.py, layout frame) into polygon_local as rectilinear notches.

    A pillar on side s takes the cells between its face line and the side line over its span along the wall. The
    side line must be the polygon's edge there (the cell beyond it outside), so a side the polygon moved or an alcove
    keeps its shape; a cut that leaves holes or splits the room is undone. The rectangle's lines never move: wall
    lengths stay wall to wall. In place: lines, cells, vertices, edges, area, changes, pillars."""
    pl.setdefault("pillars", [])
    if not pillars or not pl.get("ok"):
        return
    xl, zl = [dict(q) for q in pl["x_lines"]], [dict(q) for q in pl["z_lines"]]
    inside = np.array(pl["inside"], bool)
    sides = pl.get("sides") or {}
    for pr in pillars:
        s = pr["side"]
        ax, sg = _side_axis(s)
        j = 0 if ax == 0 else 1
        side_pos = sg * float(sides.get(s, {}).get("offset", pr["side_m"]))
        rec = dict(pr, kind=pr["kind"], cut=False)
        if abs(side_pos - sg * pr["side_m"]) > 0.05:
            pl["pillars"].append(dict(rec, reason="the polygon moved this side: not cut"))
            continue
        L = [xl, zl]
        sig = float(max(sides.get(s, {}).get("sigma") or 0.1, 0.05))
        new = [(j, sg * pr["face_m"], f"{pr['kind']} face"), (1 - j, pr["t0"], f"{pr['kind']} side"),
               (1 - j, pr["t1"], f"{pr['kind']} side")]
        L2 = [list(xl), list(zl)]
        for a, pos, kind in new:
            if all(abs(q["offset"] - pos) > 0.02 for q in L2[a]):
                L2[a].append(dict(offset=float(pos), side=s, kind=kind, status="measured", sigma=sig))
        L2 = [sorted(q, key=lambda r: r["offset"]) for q in L2]
        X0, Z0 = np.array([q["offset"] for q in L[0]]), np.array([q["offset"] for q in L[1]])
        X2, Z2 = np.array([q["offset"] for q in L2[0]]), np.array([q["offset"] for q in L2[1]])
        cx, cz = 0.5 * (X2[1:] + X2[:-1]), 0.5 * (Z2[1:] + Z2[:-1])
        io = np.clip(np.searchsorted(X0, cx) - 1, 0, len(X0) - 2)
        ko = np.clip(np.searchsorted(Z0, cz) - 1, 0, len(Z0) - 2)
        ins2 = inside[io][:, ko] & (cx[:, None] > X0[0]) & (cx[:, None] < X0[-1]) \
            & (cz[None, :] > Z0[0]) & (cz[None, :] < Z0[-1])
        a0, a1 = sorted((sg * pr["face_m"], side_pos))
        cn, ct = (cx, cz) if j == 0 else (cz, cx)
        band_n = (cn > a0) & (cn < a1)
        band_t = (ct > pr["t0"]) & (ct < pr["t1"])
        cut = (band_n[:, None] & band_t[None, :]) if j == 0 else (band_t[:, None] & band_n[None, :])
        cut &= ins2
        if not cut.any():
            pl["pillars"].append(dict(rec, reason="no room cells there: not cut"))
            continue
        # the side line is the outline there: just beyond it (same span), nothing of the room
        beyond = np.flatnonzero((cn > side_pos) if sg > 0 else (cn < side_pos))
        nxt = None if len(beyond) == 0 else int(beyond[0] if sg > 0 else beyond[-1])
        if nxt is not None:
            col = ins2[nxt, :] if j == 0 else ins2[:, nxt]
            if (col & band_t).any():
                pl["pillars"].append(dict(rec, reason="the room continues beyond the side there (alcove): not cut"))
                continue
        trial = ins2 & ~cut
        if not _connected(trial) or trace(trial) is None:
            pl["pillars"].append(dict(rec, reason="the cut would split the room or leave a hole: not cut"))
            continue
        xl, zl, inside = L2[0], L2[1], trial
        pl["pillars"].append(dict(rec, cut=True))
    if not any(q.get("cut") for q in pl["pillars"]):
        return
    o = _outline(xl, zl, inside)
    if o is None:
        return
    V, vl, edges, area = o
    pl.update(x_lines=xl, z_lines=zl, inside=inside.astype(int).tolist(), vertices=V, vertex_lines=vl, edges=edges,
              n_vertices=len(V), area_m2=round(area, 3), used=True)
    pl["changes"] = list(pl.get("changes", [])) + [
        dict(kind=q["kind"], side=q["side"], t_m=[q["t0"], q["t1"]], width_m=q["width_m"], depth_m=q["depth_m"],
             photos=q["photos"]) for q in pl["pillars"] if q.get("cut")]


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


def plan_pillars(lay: dict, c, room_id: str) -> list[dict]:
    """The pillars / steps cut into polygon_local (cut_pillars), in plan coordinates (u, v) for plan.json: kind, the
    plan side of the wall they stand on, the notch rectangle, its centre, width along the wall and depth out of it,
    the photos that saw it. Same mapping as plan_polygon (layout x/z -> plan axis and sign, + the room centre c)."""
    pl = lay.get("polygon_local") or {}
    if not pl.get("ok") or not pl.get("used"):
        return []
    amap = (lay.get("anchor") or {}).get("side_to_plan") or {s: s for s in SIDES}
    ax_x, sx = _E[amap["+x"]]
    ax_z, sz = _E[amap["+z"]]
    if ax_x == ax_z:
        return []
    c = np.asarray(c, float)

    def to_plan(x, z):
        q = np.zeros(2)
        q[ax_x], q[ax_z] = sx * x, sz * z
        return c + q
    out = []
    for k, q in enumerate(x for x in pl.get("pillars", []) if x.get("cut")):
        ax, sg = _side_axis(q["side"])
        n0, n1 = sg * q["face_m"], sg * q["side_m"]
        corners = [(n, t) if ax == 0 else (t, n) for n, t in ((n0, q["t0"]), (n1, q["t0"]), (n1, q["t1"]),
                                                               (n0, q["t1"]))]
        P = np.array([to_plan(x, z) for x, z in corners])
        out.append(dict(id=f"{room_id}-P{k + 1}", kind=q["kind"], wall_side=amap[q["side"]],
                        center=[round(float(v), 4) for v in P.mean(0)],
                        width_m=round(float(q["t1"] - q["t0"]), 3), depth_m=round(float(q["side_m"] - q["face_m"]), 3),
                        polygon=[[round(float(a), 4), round(float(b), 4)] for a, b in P],
                        views=list(q.get("photos", [])), status="measured",
                        method="photo tier: per-photo depth profile along the wall (D-083); cut into the outline, "
                               "the wall's length stays wall to wall"))
    return out


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
