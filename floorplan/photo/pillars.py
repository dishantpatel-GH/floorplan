"""Photo tier: pillars and wall steps cut into the room outline (D-083).

The room box (layout.fit_room_layout) and the polygon step (polygon.py) give a room's walls. A pillar (column) stands
out of a wall by 0.06-0.6 m and is 0.15-0.8 m wide; a wall step is the same thing running into a corner (own Room: the
0.44 m end of the door wall, 0.126 m proud of it). Neither changes a wall's length (tape and plans measure wall to
wall), but a floor plan draws them, and the box alone draws a plain rectangle.

Each photo is used on its own: MoGe-2's depth scale differs between photos of one spin by up to 40% (own Room: one W4
pillar at 1.10 m in one photo and 1.56 m in the next), but inside one depth map the pillar and the wall beside it are
measured together. Per photo and measured side, in the layout frame:

  1. points: the box fit's wall points (vertical normal, 1-2 m band, wall-labelled by the D-060 segmenter, farthest
     point per image column) that face the side and lie toward it from the camera;
  2. profile along the wall in 5 cm cells, split into RUNS: neighbouring cells within 3 cm (or 2%) of a level;
  3. a pillar: a run 0.15-0.8 m wide, in front of the wall beside it by 0.06-0.6 m on one side (a run >= 0.25 m long
     within 0.3 m) and closed on the other (a second wall run at least 3 cm deeper, or the pillar's own side face:
     points facing along the wall at the run's end). A step: the same with the run's outer end at the room's corner;
  4. the face reaches below 0.5 m (pillars stand on the floor; a wall cabinet does not), unless something stands in
     front of its foot, and up to the band top or as high as the photo sees;
  5. furniture: anything the segmenter does not call wall is out at step 1; the far-wall rule's set-aside surface
     (D-077: the own Room's wardrobe front, an open door leaf) is not a pillar; wider than 0.8 m or deeper than 0.6 m
     (the wardrobe: 1.05 m wide, 0.9 m deep) is not one either.

Into the box: the photo's wall run is the side, so depth and tangent positions scale by side offset / wall level (about
the camera). Detections of one side that overlap (or lie within 0.4 m) are one pillar; the photo whose scale agrees
best with the box (ratio closest to 1) gives the numbers. The box sides do not move.

Output: a list of dicts in the layout frame (side, kind, tangent range, depth, face offset, photos); polygon.py cuts
them into the outline as rectilinear notches.
"""
from __future__ import annotations

import numpy as np

from floorplan.photo.layout import SIDES, _photo_cloud, _ry, _side_axis

DEFAULTS = dict(
    pillars=True,                    # find pillars and wall steps and cut them into the photo-tier outline
    pillar_cell_m=0.05,              # profile cell along the wall
    pillar_run_tol_m=0.03,           # a run: cells within 3 cm (or 2% of the range) of its level
    pillar_min_w_m=0.15,             # pillar / step width along the wall
    pillar_max_w_m=0.8,              # ... wider is a wall jog or furniture (own wardrobe 1.05 m)
    pillar_min_depth_m=0.06,         # out of the wall by 0.06-0.6 m (own Room: 0.08-0.18 m; wardrobe 0.9 m)
    pillar_max_depth_m=0.6,
    pillar_closed_min_m=0.03,        # the other side: a wall run at least this much deeper ...
    pillar_ref_min_len_m=0.25,       # the wall beside it: a run >= 0.25 m (a 0.15 m door-frame edge is not a wall)
    pillar_max_gap_m=0.3,            # ... starting within 0.3 m of the run's end (the hidden side face)
    pillar_corner_m=0.2,             # a step's outer end lies within 0.2 m of the room's corner
    pillar_min_pts=8,                # points in the run (one per image column)
    pillar_floor_h_m=0.5,            # the face reaches below this height (own pillars: the floor), unless something
                                     #     stands in front of its foot (own steps: a box, a bag) ...
    pillar_top_slack_m=0.15,         # ... and up to the band top, or to what the photo sees there, minus this
    pillar_merge_m=0.4,              # detections of one side closer than this are one pillar
    pillar_far_wall_m=0.1,           # a run within this of the far-wall rule's set-aside surface is furniture
)


def _par(p, k):
    return getattr(p, k, DEFAULTS[k])


def profile_runs(t: np.ndarray, d: np.ndarray, cell: float, tol: float, min_pts: int = 2) -> list[dict]:
    """1-D profile (tangent t, offset d) -> runs of cells whose median offset stays within tol (or 2%) of the run's
    level; gaps of up to one empty cell are bridged. Each run: t0, t1, level (median of its cells), n points."""
    if len(t) == 0:
        return []
    k = np.floor(t / cell).astype(int)
    out: list[dict] = []
    for kk in np.unique(k):
        m = k == kk
        if m.sum() < min_pts:
            continue
        dv = float(np.median(d[m]))
        if out and kk - out[-1]["k1"] <= 2 and abs(dv - out[-1]["level"]) < max(tol, 0.02 * dv):
            r = out[-1]
            r["k1"] = int(kk)
            r["cells"].append(dv)
            r["n"] += int(m.sum())
            r["level"] = float(np.median(r["cells"]))
        else:
            out.append(dict(k0=int(kk), k1=int(kk), cells=[dv], n=int(m.sum()), level=dv))
    for r in out:
        r["t0"], r["t1"] = r["k0"] * cell, (r["k1"] + 1) * cell
        r["len"] = r["t1"] - r["t0"]
    return out


def _photo_points(v, yaw, t, scale, Rm, cam_h, p):
    """Layout-frame points of one photo: positions, normals, heights, image columns, wall flag, vertical flag."""
    P, N, col = _photo_cloud(v, yaw, t, scale, p.layout_stride)
    P, N = P @ Rm.T, N @ Rm.T
    h = P[:, 1] + cam_h
    vert = np.abs(N[:, 1]) < p.layout_wall_max_ny
    wl = np.ones(len(P), bool)
    if getattr(p, "semantic_walls", True) and getattr(v, "sem", None) is not None:
        from floorplan.photo.semantic import semantic_wall_points
        sw = semantic_wall_points(v, p.layout_stride)
        if sw is not None and int((sw & vert).sum()) >= getattr(p, "semantic_min_pts", 150):
            wl = sw
    return P, N, h, col, wl, vert


def _farthest(P, col, tc, sel):
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return idx
    r = np.hypot(P[idx, 0] - tc[0], P[idx, 2] - tc[2])
    o = np.lexsort((-r, col[idx]))
    first = np.r_[True, col[idx][o][1:] != col[idx][o][:-1]]
    return idx[o][first]


def _face_heights(P, N, h, vert, ax, sg, tn, r, p, bin_h: float = 0.15):
    """Height range of a run's face: from the band, follow the surface down and up in 15 cm height bins while each
    bin's median offset stays within 4 cm of the bin before (a wall leans by a few cm per metre when a photo's
    gravity is off by 2-5 deg: own 223851, 1.44 m at the floor and 1.56 m at 1.35 m). Going down, a bin much NEARER
    than the face is something standing in front of it (a box, a bag): 'occluded', the floor test is waived;
    farther or empty: the face ends there. Returns (lowest, highest, how the walk down ended)."""
    sel = vert & (P[:, tn] >= r["t0"]) & (P[:, tn] <= r["t1"]) & (N[:, ax] * sg < -p.layout_normal_min)
    near = vert & (P[:, tn] >= r["t0"]) & (P[:, tn] <= r["t1"])
    if sel.sum() < _par(p, "pillar_min_pts"):
        return None, None, None
    d = P[:, ax] * sg
    edges = np.arange(0.0, max(float(h[sel].max()), p.layout_band_hi_m) + bin_h, bin_h)
    med, mnear = {}, {}
    for i in range(len(edges) - 1):
        m = sel & (h >= edges[i]) & (h < edges[i + 1])
        if m.sum() >= 5:
            med[i] = float(np.median(d[m]))
        mn = near & (h >= edges[i]) & (h < edges[i + 1])
        if mn.sum() >= 5:
            mnear[i] = float(np.percentile(d[mn], 25))
    start = [i for i in med if p.layout_band_lo_m <= edges[i] < p.layout_band_hi_m and abs(med[i] - r["level"]) < 0.04]
    if not start:
        return None, None, None
    lo = hi = start[0]
    how = "floor"
    while True:                                          # down
        i = lo - 1
        if i < 0:
            break
        if i in med and abs(med[i] - med[lo]) < 0.04:
            lo = i
            continue
        if i in mnear and mnear[i] < med[lo] - 0.06:
            how = "occluded"
        else:
            how = "ends"
        break
    while hi + 1 in med and abs(med[hi + 1] - med[hi]) < 0.04:   # up
        hi += 1
    return float(edges[lo]), float(edges[hi + 1]), how


def photo_candidates(name, P, N, h, col, wl, vert, tc, lay, p) -> list[dict]:
    """Pillar / step candidates of one photo, in the photo's own depth scale, then mapped into the box."""
    S = lay["sides"]
    cell, tol = _par(p, "pillar_cell_m"), _par(p, "pillar_run_tol_m")
    band = (h > p.layout_band_lo_m) & (h < p.layout_band_hi_m)
    keep = _farthest(P, col, tc, band & vert & wl)
    out = []
    for s in SIDES:
        if S[s].get("status") != "measured" or not S[s].get("offset"):
            continue
        ax, sg = _side_axis(s)
        tn = 2 if ax == 0 else 0
        Ds = float(S[s]["offset"])
        c_ax = sg * float(tc[ax])                        # the camera's offset toward the side (0 for a spin photo)
        if Ds - c_ax < p.layout_min_dist_m:              # a placed photo at (or past) this side: no scale for it
            continue
        dd = P[keep, ax] * sg
        face = N[keep, ax] * sg < -p.layout_normal_min
        dirv = np.stack([P[keep, 0] - tc[0], P[keep, 2] - tc[2]], 1)
        cosa = dirv[:, 0 if ax == 0 else 1] * sg / (np.linalg.norm(dirv, axis=1) + 1e-9)
        sel = face & (dd > p.layout_min_dist_m) & (cosa > 0.5)
        if sel.sum() < p.layout_min_side_pts // 2:
            continue
        ts, ds = P[keep[sel], tn], dd[sel]
        runs = profile_runs(ts, ds, cell, tol)
        # perpendicular faces (normal along the wall) among the farthest points: a pillar's own side face
        perp = np.abs(N[keep, tn]) > p.layout_normal_min
        pt, pd = P[keep[perp], tn], dd[perp]
        # the room's corners along this side, in this photo's scale (set per candidate from its wall level)
        perp_sides = ("+z", "-z") if ax == 0 else ("+x", "-x")
        for i, r in enumerate(runs):
            w = r["len"]
            if not (_par(p, "pillar_min_w_m") <= w <= _par(p, "pillar_max_w_m")) or r["n"] < _par(p, "pillar_min_pts"):
                continue
            left = next((q for q in reversed(runs[:i]) if r["t0"] - q["t1"] <= _par(p, "pillar_max_gap_m")), None)
            right = next((q for q in runs[i + 1:] if q["t0"] - r["t1"] <= _par(p, "pillar_max_gap_m")), None)

            def is_ref(q):
                return (q is not None and q["len"] >= _par(p, "pillar_ref_min_len_m")
                        and _par(p, "pillar_min_depth_m") <= q["level"] - r["level"] <= _par(p, "pillar_max_depth_m"))
            refs = [q for q in (left, right) if is_ref(q)]
            if not refs:
                continue
            ref = max(refs, key=lambda q: q["len"])
            depth = float(np.mean([q["level"] - r["level"] for q in refs]))
            rho = (Ds - c_ax) / (ref["level"] - c_ax)    # this photo's scale -> the box's, about its camera
            tcam = tc[tn]

            def to_box(x):
                return float(tcam + (x - tcam) * rho)

            def closed(end, q):
                """The run's other end is closed: a deeper wall run beside it, or its own side face there."""
                if q is not None and q["level"] - r["level"] >= _par(p, "pillar_closed_min_m"):
                    return "wall"
                m = (np.abs(pt - end) < 0.08) & (pd > r["level"] - 0.03) & (pd < r["level"] + depth + 0.05)
                return "side face" if m.sum() >= 3 else None
            # corners along this side (box), and whether the run's outer end reaches one
            lo_c = -float(S[perp_sides[1]]["offset"]) if S[perp_sides[1]].get("status") == "measured" else None
            hi_c = float(S[perp_sides[0]]["offset"]) if S[perp_sides[0]].get("status") == "measured" else None
            b0, b1 = to_box(r["t0"]), to_box(r["t1"])
            kind, ends = None, {}
            ref_left, ref_right = left in refs, right in refs
            if ref_left and ref_right:
                kind, ends = "pillar", dict(lo="wall", hi="wall")
            elif ref_left or ref_right:
                other_end, other_q = (r["t1"], right) if ref_left else (r["t0"], left)
                corner = hi_c if ref_left else lo_c
                b_end = b1 if ref_left else b0
                if corner is not None and abs(b_end - corner) <= _par(p, "pillar_corner_m"):
                    kind = "step"
                    ends = dict(lo="wall", hi="corner") if ref_left else dict(lo="corner", hi="wall")
                else:
                    c = closed(other_end, other_q)
                    if c is not None:
                        kind = "pillar"
                        ends = dict(lo="wall", hi=c) if ref_left else dict(lo=c, hi="wall")
            if kind is None:
                continue
            # the face stands on the floor and reaches the band top (or as high as the photo sees there)
            h_lo, h_hi, low_how = _face_heights(P, N, h, vert, ax, sg, tn, r, p)
            tsel = (P[:, tn] >= r["t0"]) & (P[:, tn] <= r["t1"])
            seen_top = float(np.percentile(h[tsel], 99)) if tsel.any() else 0.0
            top_need = min(p.layout_band_hi_m, seen_top) - _par(p, "pillar_top_slack_m")
            if h_lo is None or h_hi < top_need or (h_lo > _par(p, "pillar_floor_h_m") and low_how != "occluded"):
                continue
            depth_box = depth * rho
            out.append(dict(side=s, kind=kind, photo=name, t0=min(b0, b1), t1=max(b0, b1), depth_m=depth_box,
                            face_m=Ds - depth_box, ends=ends, ratio=rho, n=int(r["n"]),
                            photo_level_m=round(r["level"], 3), photo_wall_m=round(ref["level"], 3),
                            photo_width_m=round(w, 3), photo_depth_m=round(depth, 3),
                            height_m=[round(h_lo, 2), round(h_hi, 2)], face_low=low_how))
    return out


def merge(cands: list[dict], lay: dict, p) -> tuple[list[dict], list[dict]]:
    """One pillar per cluster of overlapping detections of a side; drop furniture and out-of-room detections.
    Returns (pillars, rejected)."""
    S = lay["sides"]
    keep, rej = [], []
    for c in cands:
        fw = (S[c["side"]].get("far_wall") or {}).get("from_m")
        if fw is not None and abs(c["face_m"] - fw) <= _par(p, "pillar_far_wall_m") + 0.5 * c["depth_m"]:
            rej.append(dict(c, reason="at the surface the far-wall rule set aside as furniture (D-077)"))
            continue
        ax, _ = _side_axis(c["side"])
        lo_s, hi_s = ("-z", "+z") if ax == 0 else ("-x", "+x")
        lo, hi = -float(S[lo_s]["offset"]), float(S[hi_s]["offset"])
        if c["t1"] < lo + 0.05 or c["t0"] > hi - 0.05:
            rej.append(dict(c, reason="outside the room's extent along the side"))
            continue
        keep.append(c)
    out = []
    for s in SIDES:
        cs = sorted([c for c in keep if c["side"] == s], key=lambda c: abs(np.log(c["ratio"])))
        groups: list[list[dict]] = []
        for c in cs:                                   # most trusted first: ratio closest to 1
            g = next((g for g in groups if min(c["t1"], g[0]["t1"]) - max(c["t0"], g[0]["t0"])
                      > -_par(p, "pillar_merge_m")), None)
            if g is None:
                groups.append([c])
            else:
                g.append(c)
        for g in groups:
            b = g[0]
            ax, _ = _side_axis(s)
            lo_s, hi_s = ("-z", "+z") if ax == 0 else ("-x", "+x")
            lo, hi = -float(S[lo_s]["offset"]), float(S[hi_s]["offset"])
            kinds = {c["kind"] for c in g}
            kind = "step" if b["kind"] == "step" else "pillar"
            t0, t1 = max(b["t0"], lo), min(b["t1"], hi)
            if kind == "step":                          # a step runs into the corner
                if b["ends"].get("hi") == "corner":
                    t1 = hi
                else:
                    t0 = lo
            if t1 - t0 < _par(p, "pillar_min_w_m") * 0.75:
                rej.append(dict(b, reason="too narrow inside the room"))
                continue
            out.append(dict(side=s, kind=kind, t0=round(t0, 3), t1=round(t1, 3), width_m=round(t1 - t0, 3),
                            depth_m=round(b["depth_m"], 3), face_m=round(b["face_m"], 3),
                            side_m=round(float(S[s]["offset"]), 3), ends=b["ends"],
                            photos=sorted({c["photo"] for c in g}), kinds_seen=sorted(kinds),
                            evidence=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()
                                       if k not in ("side",)} for c in g]))
    return out, rej


def find_pillars(lay: dict, views: dict, yaws: dict, offs: dict, scale: float, p) -> dict:
    """Pillars and wall steps of one room (layout frame). yaws/offs: the polygon step's photo set (spin photos at the
    centre, other placed photos of the room at their positions)."""
    if not _par(p, "pillars"):
        return dict(ok=False, reason="disabled", pillars=[])
    if not lay.get("ok"):
        return dict(ok=False, reason="no box", pillars=[])
    Rm = _ry(lay["manhattan_yaw"])
    cam_h = lay["cam_height_m"]
    cands = []
    for n in yaws:
        v = views.get(n)
        if v is None or v.normal_cam is None:
            continue
        t = np.asarray(offs.get(n, np.zeros(3)), float)
        P, N, h, col, wl, vert = _photo_points(v, yaws[n], t, scale, Rm, cam_h, p)
        cands += photo_candidates(n, P, N, h, col, wl, vert, Rm @ t, lay, p)
    pillars, rej = merge(cands, lay, p)
    return dict(ok=True, pillars=pillars, rejected=rej, candidates=len(cands))
