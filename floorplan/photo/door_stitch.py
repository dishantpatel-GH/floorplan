"""Door-anchored stitching of the photo tier's rooms (D-081): stitch at the doors the out-looking photos see.

The idea is the user's (5 Oct): we know where the door is in each room, and the capture protocol always takes a photo
looking OUT through it (a room's first turning photo faces the door one came in by, a small room's far-end photos look
back at its door, a doorway pair stands on the threshold). Compare that out-looking photo with every other room's
photos: the room it matches is the neighbour, and the match says where; then snap the two rooms together at the door
they share. Each room keeps its own size: rooms only move (rigidly).

  1. Doors per room, in the room's own layout frame (layout.py): pixels whose depth runs more than door_beyond_m past a
     fitted wall at door heights, cast along their rays onto that wall, plus the ADE20K door classes on the wall
     (windows, sky, curtains and mirrors are not openings) -> door intervals (side, centre, width). A doorway-pair
     photo placed in the room stands in one of its doors (the threshold).
  2. Door correspondences: every photo of room A against every photo of every other room B with verified matches,
     by PnP of B-photo's pixels on A-photo's metric depth (RANSAC + LM). Kept only if the relative pose is level
     (tilt <= pnp_max_tilt_deg: both photos are levelled by gravity), has a real baseline (>= pnp_min_baseline_m:
     matches on the outdoor scene through two windows give a pure rotation, measured on sim k65: 643 'inliers' bedroom
     to balcony), the matched points lie outside A (seen through A's door) and inside B, and the two rooms do not
     overlap. The room with the most such inliers is A's neighbour through that door. A doorway pair names the two
     rooms of a door as well.
  3. Relative placement: the PnP pose puts B's frame in A's (rotation snapped to the rooms' Manhattan axes); the door
     snap then moves B so that the facing door intervals coincide along the wall and the wall faces are parallel, a
     wall thickness apart (jamb depth when the door's jambs are seen, else a 0.15 +- 0.05 m prior).
  4. Joint adjustment over all rooms: door constraints (strong), PnP-only links and doorway-pair same-spot priors
     (weaker), no overlaps (penalty); per-room rigid moves only, robust loss. A room with no matched door keeps its
     placement and is flagged.
"""
from __future__ import annotations

import copy

import numpy as np

SIDES = ("+x", "-x", "+z", "-z")
DOOR_CLASSES = ("door", "double door", "screen door")
SEE_THROUGH = ("windowpane", "sky", "curtain", "blind", "mirror", "screen")
_DIR = {"+x": np.array([1.0, 0.0]), "-x": np.array([-1.0, 0.0]), "+z": np.array([0.0, 1.0]), "-z": np.array([0.0, -1.0])}


def _ry(th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot2(th: float) -> np.ndarray:
    """The (x, z) part of _ry(th), acting on (x, z) column vectors."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, s], [-s, c]])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _ax(side: str) -> tuple[int, float]:
    """2-D axis index (0 = x, 1 = z) and outward sign of a layout side."""
    return (0 if side[1] == "x" else 1), (1.0 if side[0] == "+" else -1.0)


def _perp(side: str) -> tuple[str, str]:
    return ("-z", "+z") if side[1] == "x" else ("-x", "+x")


def _side_of_normal(n: np.ndarray) -> str:
    return max(SIDES, key=lambda s: float(_DIR[s] @ n))


# ------------------------------------------------------------------------------------------- 0. room frames
def room_frames(views: dict, lays: dict, poses: dict, extra_comps: list[dict], k: float) -> dict:
    """room -> its layout frame: photos with a pose in it {name: (yaw, (x, z))}, side offsets, box.

    The layout's own fitted photos are placed by their local yaw/offset (fit_poses). Other photos of the room are
    placed when the pose graph put them in one component with fitted photos (frontend poses, or a component the
    frontend dropped, solved here): their pose relative to those photos carries over."""
    out = {}
    comps = [poses] + list(extra_comps)
    for room, lay in lays.items():
        if not lay.get("ok") or not lay.get("fit_poses"):
            continue
        m = float(lay["manhattan_yaw"])
        fit = {n: np.asarray(v, float) for n, v in lay["fit_poses"].items() if n in views}
        photos = {}
        for n, f in fit.items():
            c = _ry(m) @ f[1:4]
            photos[n] = (float(_wrap(f[0] + m)), c[[0, 2]], "fit")
        best = None
        for cp in comps:
            g = [n for n in fit if n in cp]
            if g and (best is None or len(g) > len(best[1])):
                best = (cp, g)
        if best is not None:
            cp, g = best
            d = float(np.angle(np.mean([np.exp(1j * (fit[n][0] - cp[n][0])) for n in g])))
            b = np.mean([fit[n][1:4] - _ry(d) @ np.asarray(cp[n][1], float) for n in g], 0)
            for n, pose in cp.items():
                if n in photos or n not in views or views[n].room != room:
                    continue
                c = _ry(m) @ (_ry(d) @ np.asarray(pose[1], float) + b)
                photos[n] = (float(_wrap(pose[0] + d + m)), c[[0, 2]], "pose graph")
        off = {s: float(lay["sides"][s]["offset"]) for s in SIDES}
        out[room] = dict(photos=photos, off=off, first=next(iter(fit), None), box=(-off["-x"], off["+x"], -off["-z"], off["+z"]), m=m,
                         cam_h=float(lay.get("cam_height_m") or 1.4), small=bool(lay.get("small_room")),
                         status={s: lay["sides"][s].get("status") for s in SIDES})
    return out


def place_in_rooms(frames: dict, views: dict, matches: dict, k: float, p) -> dict:
    """A room's photos that the pose graph left out of its frame, placed by PnP of their pixels on the metric depth
    of a photo in the frame (the best-supported one). A threshold photo placed this way names its door. Sim k65: the
    living room's photo on the bathroom's threshold shared 18 verified matches with the living room's photo on the
    bedroom's threshold; the pose graph had dropped that 12-inlier edge as a weak bridge, so the bathroom stayed
    'beside the block'. Kept only if level (tilt <= pnp_max_tilt_deg), at chest height, on or inside the room's box
    (+ intra_box_margin_m) and looking into the room. Returns {photo: (room, from photo, inliers)}."""
    out = {}
    for room, fr in frames.items():
        best = {}
        for (a, b), (xa, xb) in matches.items():
            if views[a].room != room or views[b].room != room:
                continue
            for o, q, xo, xq in ((a, b, xa, xb), (b, a, xb, xa)):
                if o not in fr["photos"] or q in fr["photos"]:
                    continue
                r = _pnp(views[o], views[q], np.asarray(xo, float), np.asarray(xq, float), p, p.intra_min_inliers)
                if r is None or r["tilt"] > p.pnp_max_tilt_deg or abs(r["c"][1]) * k > p.pnp_max_dy_m:
                    continue
                yo, co, _ = fr["photos"][o]
                c3 = k * _ry(yo) @ r["c"] + np.array([co[0], 0.0, co[1]])
                c2, yaw = c3[[0, 2]], float(_wrap(yo + r["yaw"]))
                if not _inside(c2[None], fr["box"], p.intra_box_margin_m)[0]:
                    continue
                ctr = np.array([(fr["box"][0] + fr["box"][1]) / 2, (fr["box"][2] + fr["box"][3]) / 2])
                h, v = np.array([np.sin(yaw), np.cos(yaw)]), ctr - c2
                if np.linalg.norm(v) > 0.5 and v @ h < np.cos(np.radians(p.intra_max_view_deg)) * np.linalg.norm(v):
                    continue                                   # not looking into the room
                if q not in best or r["n"] > best[q][2]:
                    best[q] = (yaw, c2, r["n"], o)
        for q, (yaw, c2, n, o) in best.items():
            fr["photos"][q] = (yaw, c2, f"PnP on {o}")
            out[q] = (room, o, n)
    return out


def _cam_h(view, fr, k) -> float:
    return float(-view.floor_h * k) if view.floor_h is not None else fr["cam_h"]


def _points(view, yaw: float, c2, k: float, stride: int):
    """Points of one photo in its room frame (3-D, y up, camera at height 0) and their flat pixel indices."""
    Pc, idx = view.points_cam(stride)
    P = (k * Pc @ view.R_lev.T) @ _ry(yaw).T
    P[:, 0] += c2[0]
    P[:, 2] += c2[1]
    return P, idx


def _label_ids(view, names) -> np.ndarray:
    labels = getattr(view, "sem_labels", None)
    if labels is None:
        return np.zeros(0, int)
    return np.array([i for i, n in enumerate(labels) if n in names], int)


# ------------------------------------------------------------------------------------------- 1. doors per room
def detect_doors(room: str, fr: dict, views: dict, pairs: list[dict], k: float, p) -> list[dict]:
    """Door intervals on the room's 4 fitted walls (depth beyond the wall, door pixels), plus thresholds."""
    b = p.door_bin_m
    off = fr["off"]
    acc = {}
    jamb = {s: [] for s in SIDES}
    for s in SIDES:
        lo_s, hi_s = _perp(s)
        lo, hi = -off[lo_s] - 0.3, off[hi_s] + 0.3
        nb = max(int(np.ceil((hi - lo) / b)), 1)
        acc[s] = dict(lo=lo, nb=nb, mid_out=np.zeros(nb), mid_wall=np.zeros(nb), low_out=np.zeros(nb),
                      low_wall=np.zeros(nb), door=np.zeros(nb), photos=[set() for _ in range(nb)])
    for n, (yaw, c2, _) in fr["photos"].items():
        v = views[n]
        P, idx = _points(v, yaw, c2, k, p.door_stride)
        h = P[:, 1] + _cam_h(v, fr, k)
        sem = getattr(v, "sem", None)
        see = np.isin(sem.ravel()[idx], _label_ids(v, SEE_THROUGH)) if sem is not None else np.zeros(len(P), bool)
        dpx = np.isin(sem.ravel()[idx], _label_ids(v, DOOR_CLASSES)) if sem is not None else np.zeros(len(P), bool)
        X = P[:, [0, 2]]
        for s in SIDES:
            a, sg = _ax(s)
            t = 1 - a
            o = off[s]
            if sg * c2[a] >= o - 0.2:                         # the camera must be inside, in front of this wall
                continue
            ra, rt = X[:, a] - c2[a], X[:, t] - c2[t]
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = (sg * o - c2[a]) / ra
            ct = c2[t] + lam * rt
            inc = np.abs(ra) / np.maximum(np.hypot(ra, rt), 1e-9)
            d = sg * X[:, a] - o                               # how far past the wall face (m)
            ok = (lam > 0) & (inc > np.sin(np.radians(p.door_min_incidence_deg))) & (d > -0.25) & (h > 0.1) \
                & (h < p.door_top_m) & np.isfinite(ct)
            A = acc[s]
            j = np.floor((ct - A["lo"]) / b).astype(int)
            ok &= (j >= 0) & (j < A["nb"])
            out_ = ok & (d > p.door_beyond_m) & ~see
            wall = ok & (np.abs(d) <= 0.2) & ~dpx
            mid, low = h >= p.door_low_top_m, h < p.door_low_top_m
            for key, sel in (("mid_out", out_ & mid), ("mid_wall", wall & mid), ("low_out", out_ & low),
                             ("low_wall", wall & low), ("door", ok & dpx & (d > -0.2))):
                np.add.at(A[key], j[sel], 1)
            for q in np.unique(j[out_ & mid]):
                A["photos"][q].add(n)
            # jambs: surfaces facing along the wall, between this face and the far face (wall thickness)
            nc = getattr(v, "normal_cam", None)
            if nc is not None:
                N = (nc.reshape(-1, 3)[idx] @ v.R_lev.T) @ _ry(yaw).T
                js = ok & (np.abs(N[:, [0, 2]][:, t]) > 0.85) & (d > 0.02) & (d < 0.45) & (h > 0.5) & (h < 1.9)
                jamb[s].append(d[js])
    doors = []
    for s in SIDES:
        A = acc[s]
        mo, mw, lo_, lw, dp = A["mid_out"], A["mid_wall"], A["low_out"], A["low_wall"], A["door"]
        opening = (mo >= p.door_min_pts) & (mo >= p.door_min_out_frac * (mo + mw)) & ~(lw > 0.5 * (lw + lo_) + 2)
        doorpx = dp >= p.door_min_pts
        if fr["status"].get(s) != "measured":
            # an inferred side is not where the wall is: what lies beyond it is the rest of the room (sim k38 living
            # room: 3 phantom 'doors' 1.9-3.2 m wide on its short sides); its doors come from threshold photos only
            A["jamb"] = np.zeros(0)
            continue
        wallc = (mw >= p.door_min_pts) & (mw > mo)
        flank = int(round(p.door_flank_m / b))
        for kind, mask in (("depth", opening), ("door pixels", doorpx & ~opening)):
            for i0, i1 in _runs(mask, p.door_gap_bins):
                w = (i1 - i0 + 1) * b
                if not (p.door_min_w_m <= w <= p.door_max_w_m):
                    continue
                if not (wallc[max(i0 - flank, 0):i0].any() or wallc[i1 + 1:i1 + 1 + flank].any()):
                    continue                              # a door is a gap IN a wall: wall seen beside it
                lo = A["lo"] + i0 * b
                phs = sorted(set().union(*A["photos"][i0:i1 + 1])) if kind == "depth" else []
                doors.append(dict(room=room, side=s, c=float(lo + w / 2), w=float(w), lo=float(lo), hi=float(lo + w),
                                  src=kind, sigma_c=p.door_sigma_c_m, photos=phs, pairs=[], side_known=True))
        jd = np.concatenate(jamb[s]) if jamb[s] else np.zeros(0)
        A["jamb"] = jd
    # one door can come out of both rules: keep the depth one where they overlap
    doors = [d for d in doors if d["src"] == "depth" or not any(
        e["src"] == "depth" and e["side"] == d["side"] and min(e["hi"], d["hi"]) - max(e["lo"], d["lo"]) > 0
        for e in doors)]
    for d in doors:                                       # wall thickness from the jambs at this door's ends
        jd = acc[d["side"]]["jamb"]
        d["thickness"] = float(np.percentile(jd, 90)) if len(jd) >= p.jamb_min_pts else None
    for d in doors:
        a, sg = _ax(d["side"])
        d["face"] = sg * off[d["side"]]                  # axis coordinate of the room-side wall face
    # thresholds: a doorway-pair photo of this room stands in one of its doors (behind it). The camera fixes where
    # that wall is, even when the box's side there was only inferred (hub rooms: sim k65 living room, 1.8 m short)
    for pr in pairs:
        for n in (pr["a"], pr["b"]):
            if n not in fr["photos"] or views[n].room != room:
                continue
            yaw, c2, _ = fr["photos"][n]
            fwd = np.array([np.sin(yaw), np.cos(yaw)])
            s = max(SIDES, key=lambda q: -float(fwd @ _DIR[q]))
            a, sg = _ax(s)
            face = float(c2[a] - sg * p.door_threshold_inset_m)
            ct = float(c2[1 - a])
            near = [d for d in doors if d["side"] == s and d["src"] == "depth" and abs(d["face"] - face) <= 0.6
                    and abs(d["c"] - ct) <= max(d["w"] / 2 + 0.3, 0.6)]
            square = abs(np.degrees(np.arccos(np.clip(-float(fwd @ _DIR[s]), -1, 1)))) <= p.threshold_square_deg
            if not square:
                # not square to the wall behind it: a seen door counts only if the camera stands in its interval
                # (a corner between two doors: sim k38's bedroom photo merged into the bathroom door's wall)
                near = []
                for q in SIDES:
                    aq, sq = _ax(q)
                    fq = float(c2[aq] - sq * p.door_threshold_inset_m)
                    near += [d for d in doors if d["side"] == q and d["src"] == "depth" and abs(d["face"] - fq) <= 0.4
                             and d["lo"] - 0.1 <= c2[1 - aq] <= d["hi"] + 0.1]
                if len(near) > 1:
                    near = []
            if near:
                d = min(near, key=lambda d: abs(d["c"] - ct))
                d["pairs"].append(n)
            else:
                # the wall behind a threshold photo is the door's wall only if the photo looks square into the room
                # (sim k65: the bedroom door's photo looked 17 deg off an axis, into the living room's middle, and
                # the door was in the wall at 90 deg to the one behind it)
                doors.append(dict(room=room, side=s, c=ct, w=None, lo=ct, hi=ct, src="threshold", face=face,
                                  sigma_c=p.threshold_sigma_c_m, photos=[n], pairs=[n], thickness=None,
                                  side_known=bool(square or fr["small"])))
    for i, d in enumerate(doors):
        d["id"] = f"{room}#D{i + 1}"
    return doors


def _runs(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    runs = []
    for i in idx:
        if runs and i - runs[-1][1] <= gap + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [tuple(r) for r in runs]


def door_point(d: dict, fr: dict) -> np.ndarray:
    """2-D point of the door's centre on the room's wall face, in the room frame."""
    a, sg = _ax(d["side"])
    x = np.zeros(2)
    x[a] = d["face"] if d.get("face") is not None else sg * fr["off"][d["side"]]
    x[1 - a] = d["c"]
    return x


# ------------------------------------------------------------------------------------------- 2. PnP links
def _pnp(vo, vq, xo, xq, p, min_inliers: int | None = None) -> dict | None:
    """Pose of photo q relative to photo o from q's pixels on o's metric depth. Returns yaw/centre of q in o's
    levelled frame (x_o = R_y(yaw) x_q + c), the tilt of the relative rotation and o's inlier points (levelled)."""
    import cv2
    uv_o, uv_q = xo * vo.sfm_scale, xq * vq.sfm_scale
    h, w = vo.depth.shape
    u = np.clip(np.round(uv_o[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(uv_o[:, 1]).astype(int), 0, h - 1)
    hq, wq = vq.depth.shape
    uq = np.clip(np.round(uv_q[:, 0]).astype(int), 0, wq - 1)
    vq_ = np.clip(np.round(uv_q[:, 1]).astype(int), 0, hq - 1)
    d = vo.depth[v, u]
    ok = (d > 0.2) & (d < p.pnp_max_depth_m)
    for view, uu, vv in ((vo, u, v), (vq, uq, vq_)):        # no matches on windows / sky / mirrors
        sem = getattr(view, "sem", None)
        if sem is not None:
            ok &= ~np.isin(sem[vv, uu], _label_ids(view, SEE_THROUGH))
    min_inliers = p.pnp_min_inliers if min_inliers is None else min_inliers
    if ok.sum() < min_inliers:
        return None
    P = np.stack([(uv_o[:, 0] - vo.K[0, 2]) / vo.K[0, 0] * d, (uv_o[:, 1] - vo.K[1, 2]) / vo.K[1, 1] * d, d], 1)[ok]
    x2 = uv_q[ok].astype(np.float64)
    try:
        r, rvec, tvec, inl = cv2.solvePnPRansac(P.astype(np.float64), x2, vq.K.astype(np.float64), None,
                                                iterationsCount=1000, reprojectionError=p.pnp_px,
                                                confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return None
    if not r or inl is None or len(inl) < min_inliers:
        return None
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(P[inl], x2[inl], vq.K.astype(np.float64), None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    Rrel = vq.R_lev @ R @ vo.R_lev.T
    tl = vq.R_lev @ tvec.ravel()
    phi = float(np.arctan2(Rrel[0, 2], Rrel[0, 0]))
    yaw = -phi
    c = -_ry(yaw) @ tl
    return dict(n=int(len(inl)), yaw=yaw, c=c, tilt=float(np.degrees(np.arccos(np.clip(Rrel[1, 1], -1, 1)))),
                P=P[inl] @ vo.R_lev.T)


def _inside(X: np.ndarray, box, margin: float) -> np.ndarray:
    return (X[:, 0] > box[0] - margin) & (X[:, 0] < box[1] + margin) & (X[:, 1] > box[2] - margin) & \
        (X[:, 1] < box[3] + margin)


def _outer_box(fr: dict, p):
    """The room's box with each INFERRED side pushed out by inferred_slack_m (it may lie farther than inferred)."""
    b = list(fr["box"])
    for j, s in enumerate(("-x", "+x", "-z", "+z")):
        if fr["status"].get(s) != "measured":
            b[j] += (-1 if s[0] == "-" else 1) * p.inferred_slack_m
    return tuple(b)


def _box_moved(box, psi: float, t: np.ndarray):
    C = np.array([[box[0], box[2]], [box[1], box[2]], [box[1], box[3]], [box[0], box[3]]]) @ _rot2(psi).T + t
    return (C[:, 0].min(), C[:, 0].max(), C[:, 1].min(), C[:, 1].max())


def _overlap(b1, b2) -> float:
    ox = min(b1[1], b2[1]) - max(b1[0], b2[0])
    oz = min(b1[3], b2[3]) - max(b1[2], b2[2])
    return max(ox, 0.0) * max(oz, 0.0)


def _area(b) -> float:
    return max(b[1] - b[0], 0.0) * max(b[3] - b[2], 0.0)


def pnp_links(frames: dict, views: dict, matches: dict, k: float, p, log=print) -> list[dict]:
    """Every verified cross-room photo pair, both ways (the out-looking photo is the one whose matched points lie
    outside its own room). Returns the hypotheses that pass all checks: x_A = R(psi) x_B + t in the rooms' frames."""
    hyps, rejected = [], {}
    for (a, b), (xa, xb) in matches.items():
        ra, rb = views[a].room, views[b].room
        if ra == rb or ra not in frames or rb not in frames:
            continue
        for o, q, xo, xq in ((a, b, xa, xb), (b, a, xb, xa)):
            A, B = views[o].room, views[q].room
            if o not in frames[A]["photos"] or q not in frames[B]["photos"]:
                continue
            r = _pnp(views[o], views[q], np.asarray(xo, float), np.asarray(xq, float), p)
            why = None
            if r is None:
                why = "too few PnP inliers"
            elif r["tilt"] > p.pnp_max_tilt_deg:
                why = "not level"
            elif abs(r["c"][1]) * k > p.pnp_max_dy_m:
                why = "camera heights differ"
            elif np.hypot(r["c"][0], r["c"][2]) * k < p.pnp_min_baseline_m:
                why = "no baseline (pure rotation: far scene through windows)"
            if why:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            yo, co, _ = frames[A]["photos"][o]
            yq, cq, _ = frames[B]["photos"][q]
            co3 = np.array([co[0], 0.0, co[1]])
            qA_yaw = yo + r["yaw"]
            qA_c = (k * _ry(yo) @ r["c"] + co3)[[0, 2]]
            psi = float(_wrap(qA_yaw - yq))
            psi_s = float(np.round(psi / (np.pi / 2)) * (np.pi / 2))
            if abs(np.degrees(_wrap(psi - psi_s))) > p.pnp_max_axis_resid_deg:
                rejected["rooms not on common axes"] = rejected.get("rooms not on common axes", 0) + 1
                continue
            t = qA_c - _rot2(psi_s) @ cq
            X = ((k * r["P"]) @ _ry(yo).T + co3)[:, [0, 2]]
            boxA, boxB = frames[A]["box"], frames[B]["box"]
            outside = ~_inside(X, boxA, -p.out_margin_m)
            XB = (X - t) @ _rot2(psi_s)                     # R(psi)^T (X - t), row vectors
            inB = _inside(XB, _outer_box(frames[B], p), p.in_margin_m)
            good = outside & inB
            bB = _box_moved(boxB, psi_s, t)
            ov = _overlap(boxA, bB) / max(min(_area(boxA), _area(bB)), 1e-9)
            if good.sum() < p.pnp_min_inliers or good.mean() < p.pnp_min_good_frac:
                rejected["matched points not in the next room"] = rejected.get("matched points not in the next room",
                                                                                0) + 1
                continue
            if ov > p.max_room_overlap:
                rejected["rooms would overlap"] = rejected.get("rooms would overlap", 0) + 1
                continue
            # where the out-looking rays cross A's walls: the door they look through
            cross = _crossings(co, X[good], frames[A])
            exact = frames[A]["photos"][o][2] != "pose graph" and frames[B]["photos"][q][2] != "pose graph"
            hyps.append(dict(A=A, B=B, o=o, q=q, n=int(good.sum()), psi=psi_s, t=t, psi_raw=psi, exact=exact,
                             tilt=r["tilt"], overlap=round(ov, 3), cross=cross))
    log(f"[photo/door] PnP links: {len(hyps)} hypotheses kept; rejected " +
        ", ".join(f"{k_}: {v}" for k_, v in sorted(rejected.items())))
    return hyps


def _crossings(c2, X: np.ndarray, fr: dict) -> dict | None:
    """Median side and tangent where rays from the camera to points X first leave the room's box."""
    off = fr["off"]
    lam_all = np.full((len(X), 4), np.inf)
    tan_all = np.zeros((len(X), 4))
    for j, s in enumerate(SIDES):
        a, sg = _ax(s)
        lo_s, hi_s = _perp(s)
        ra = X[:, a] - c2[a]
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (sg * off[s] - c2[a]) / ra
        tg = c2[1 - a] + lam * (X[:, 1 - a] - c2[1 - a])
        ok = (lam > 0) & (lam <= 1.0) & np.isfinite(lam) & (tg > -off[lo_s] - 0.3) & (tg < off[hi_s] + 0.3)
        lam_all[ok, j] = lam[ok]
        tan_all[:, j] = tg
    first = np.argmin(lam_all, 1)
    hit = np.isfinite(lam_all[np.arange(len(X)), first])
    if not hit.any():
        return None
    j = int(np.bincount(first[hit], minlength=4).argmax())
    sel = hit & (first == j)
    return dict(side=SIDES[j], c=float(np.median(tan_all[sel, j])), n=int(sel.sum()))


# ------------------------------------------------------------------------------------------- 3. correspondences
def _invert(psi, t):
    return -psi, -(_rot2(-psi) @ t)


def _cluster(hyps: list[dict], tol: float) -> list[list[dict]]:
    cl = []
    for h in sorted(hyps, key=lambda h: -h["n"]):
        for c in cl:
            if abs(_wrap(c[0]["psi"] - h["psi"])) < 1e-6 and np.linalg.norm(c[0]["t"] - h["t"]) < tol:
                c.append(h)
                break
        else:
            cl.append([h])
    return sorted(cl, key=lambda c: -sum(h["n"] for h in c))


def _snap(dA: dict, frA: dict, dB: dict, frB: dict, tau: float):
    """Relative transform x_A = R(psi) x_B + t that puts B's door on A's door, faces tau apart."""
    nA, nB = _DIR[dA["side"]], _DIR[dB["side"]]
    psi = next(q * np.pi / 2 for q in range(4) if float(_rot2(q * np.pi / 2) @ nB @ nA) < -0.9)
    t = door_point(dA, frA) + tau * nA - _rot2(psi) @ door_point(dB, frB)
    return float(_wrap(psi)), t


def _nearest_door(doors: list[dict], side: str | None, x: np.ndarray, fr: dict, max_d: float):
    best = None
    for d in doors:
        if side is not None and d["side"] != side:
            continue
        dist = float(np.linalg.norm(door_point(d, fr) - x))
        if dist <= max_d and (best is None or dist < best[0]):
            best = (dist, d)
    return None if best is None else best[1]


def correspondences(frames, doors, hyps, pairs, views, p, log=print) -> list[dict]:
    """Room links with their doors: PnP clusters (out-looking photo -> neighbour) and doorway pairs."""
    by_pair = {}
    for h in hyps:                                   # canonical direction: sorted room names
        A, B = sorted((h["A"], h["B"]))
        hh = dict(h)
        if (h["A"], h["B"]) != (A, B):
            hh["psi"], hh["t"] = _invert(h["psi"], h["t"])
            hh["A"], hh["B"], hh["out_room"] = A, B, h["A"]
        else:
            hh["out_room"] = A
        by_pair.setdefault((A, B), []).append(hh)
    links = []
    for (A, B), hs in by_pair.items():
        cl = _cluster(hs, p.cluster_tol_m)
        n0 = sum(h["n"] for h in cl[0])
        n1 = sum(h["n"] for h in cl[1]) if len(cl) > 1 else 0
        entry = dict(A=A, B=B, kind="pnp", inliers=n0, rival=n1, photos=sorted({f"{h['o']}->{h['q']}" for h in cl[0]}),
                     exact=any(h["exact"] for h in cl[0]))
        if n0 < p.link_min_inliers:
            entry["rejected"] = f"weak: {n0} inliers < {p.link_min_inliers}"
        elif n1 > p.link_max_rival * n0:
            entry["rejected"] = f"ambiguous: rival placement with {n1} inliers"
        h0 = max(cl[0], key=lambda h: h["n"])
        w = np.array([h["n"] for h in cl[0]], float)
        entry["psi"] = h0["psi"]
        entry["t"] = (np.array([h["t"] for h in cl[0]]) * w[:, None]).sum(0) / w.sum()
        # the door the out-looking photo looks through, and the door of B facing it
        hc = max((h for h in cl[0] if h["cross"]), key=lambda h: h["n"], default=None)
        if hc is not None and "rejected" not in entry:
            R_, S_ = (A, B) if hc["out_room"] == A else (B, A)
            cr = hc["cross"]
            x = door_point(dict(side=cr["side"], c=cr["c"]), frames[R_])
            dR = _nearest_door(doors[R_], cr["side"], x, frames[R_], p.door_assoc_m)
            psi, t = (entry["psi"], entry["t"]) if R_ == A else _invert(entry["psi"], entry["t"])
            xS = _rot2(-psi) @ ((door_point(dR, frames[R_]) if dR else x) - t)    # in S's frame
            facing = _side_of_normal(_rot2(-psi) @ -_DIR[cr["side"]])
            dS = _nearest_door(doors[S_], facing, xS, frames[S_], p.door_assoc_m + 0.3)
            entry["doors"] = {R_: dR["id"] if dR else None, S_: dS["id"] if dS else None}
            entry["out_room"] = R_
            entry["cross"] = dict(side=cr["side"], c=round(cr["c"], 3))
            if dR is not None and dS is not None:
                entry["door_pair"] = (dR, dS) if R_ == A else (dS, dR)
        links.append(entry)
    # doorway pairs: the two photos stand in the same door
    claimed = {d["id"] for ds in doors.values() for d in ds if d["pairs"]}
    for L_ in links:
        if L_.get("door_pair") and not L_.get("rejected"):
            claimed |= {d["id"] for d in L_["door_pair"]}
    for pr in pairs:
        a, b = pr["a"], pr["b"]
        A, B = views[a].room, views[b].room
        if A == B or A not in frames or B not in frames:
            continue
        if A > B:
            A, B, a, b = B, A, b, a
        dA = next((d for d in doors[A] if a in d["pairs"]), None)
        dB = next((d for d in doors[B] if b in d["pairs"]), None)
        entry = dict(A=A, B=B, kind="doorway_pair", photos=[a, b], placed=[a in frames[A]["photos"],
                                                                       b in frames[B]["photos"]])
        entered = views[pr["b"]].room                # the pair is taken on the way INTO its second photo's room
        for side, dd, R_ in (("A", dA, A), ("B", dB, B)):
            if dd is None:                          # the photo is not placed in its room
                dd, how = _door_without_photo(R_, doors[R_], frames[R_], R_ == entered, claimed, p)
                if dd is not None:
                    entry[f"door_{side}_how"] = how
            if side == "A":
                dA = dd
            else:
                dB = dd
        if dA is not None and dB is not None and dA.get("side_known") and dB.get("side_known"):
            entry["door_pair"] = (dA, dB)
            entry["doors"] = {A: dA["id"], B: dB["id"]}
        elif dA is not None and dB is not None:
            entry["rejected"] = "door wall not known (threshold photo not square to it) in " + " and ".join(
                r for r, d in ((A, dA), (B, dB)) if not d.get("side_known"))
        else:
            entry["rejected"] = "door not found in " + " and ".join(r for r, d in ((A, dA), (B, dB)) if d is None)
        if a in frames[A]["photos"] and b in frames[B]["photos"]:
            entry["same_spot"] = (frames[A]["photos"][a][1], frames[B]["photos"][b][1])
        links.append(entry)
    return links


def _door_without_photo(room, ds, fr, entered: bool, claimed: set, p):
    """The door of a doorway pair whose photo in this room is not placed in the room's frame. The room the pair
    leads INTO started its turning photos facing that door (protocol: 'start facing the door you came in through'):
    the open door nearest the first photo's heading. Otherwise none."""
    cand = [d for d in ds if d["src"] == "depth" and d["id"] not in claimed]
    if entered and not fr["small"] and fr.get("first") in fr["photos"]:
        yaw, c2, _ = fr["photos"][fr["first"]]
        h = np.array([np.sin(yaw), np.cos(yaw)])
        ang = []
        for d in cand:
            v = door_point(d, fr) - c2
            ang.append((float(np.degrees(np.arccos(np.clip(v @ h / max(np.linalg.norm(v), 1e-9), -1, 1)))), d))
        ang = [x for x in ang if x[0] <= p.first_photo_max_angle_deg]
        if ang:
            return min(ang, key=lambda x: x[0])[1], "the door its first turning photo faces"
    return None, None          # 'the only door left' was tried: on sim k65 it put the bathroom on a phantom door


# ------------------------------------------------------------------------------------------- 4. joint adjustment
def _constraints(links, frames, p, fixed_rot: dict | None = None) -> list[dict]:
    """Relative transforms x_A = R(psi) x_B + t with sigmas along the door's normal and tangent (A's frame): the PnP
    placement of a room pair (isotropic sigma) and the door snap (faces a wall thickness apart, door centres on one
    line). A snap must agree with the pair's PnP placement (same rotation, within snap_max_move_m) when there is one."""
    pnp = {(L["A"], L["B"]): L for L in links if L["kind"] == "pnp" and not L.get("rejected")}
    fixed_rot = fixed_rot or {}
    cons = []
    for L in links:
        # (tried: a doorway pair whose door walls are unknown, as a same-spot link when the pose graph knows both
        # rooms' rotations. k38 bedroom 0.41 -> 0.07 m, but k22 bedroom 0.16 -> 0.80 m: the threshold photos'
        # positions inside their rooms come from the pose graph and were off. Not used.)
        if L.get("rejected"):
            continue
        A, B = L["A"], L["B"]
        if L["kind"] == "pnp":
            sg = p.pnp_sigma_m if L.get("exact") else p.pnp_sigma_pg_m
            cons.append(dict(A=A, B=B, psi=L["psi"], t=L["t"], n=None, sn=sg, st=sg,
                             kind="pnp", weight=L.get("inliers", 0)))
        if not L.get("door_pair"):
            continue
        dA, dB = L["door_pair"]
        th = [x for x in (dA.get("thickness"), dB.get("thickness")) if x is not None]
        tau = float(np.clip(np.median(th), 0.05, 0.40)) if th else p.wall_thickness_m
        psi, t = _snap(dA, frames[A], dB, frames[B], tau)
        ref = pnp.get((A, B))
        if ref is not None:
            dt = float(np.linalg.norm(t - ref["t"]))
            if abs(_wrap(psi - ref["psi"])) > 1e-6 or dt > p.snap_max_move_m:
                L["snap"] = dict(used=False, reason=f"door snap disagrees with the PnP placement ({dt:.2f} m, "
                                                    f"{np.degrees(_wrap(psi - ref['psi'])):+.0f} deg)")
                continue
        n = _DIR[dA["side"]]
        st = float(np.hypot(dA["sigma_c"], dB["sigma_c"]))
        cons.append(dict(A=A, B=B, psi=psi, t=t, n=n, sn=p.wall_thickness_sigma_m, st=st, kind=L["kind"] + "+door",
                         tau=tau, weight=L.get("inliers", 0) + 50))
        L["snap"] = dict(used=True, tau=round(tau, 3), tau_from="jambs" if th else "prior",
                         vs_pnp_m=None if ref is None else round(float(np.linalg.norm(t - ref["t"])), 3))
    return cons


def _pen(ox: float, oz: float, p) -> float:
    """Overlap penalty beyond overlap_tol_m (0 by default: rooms the stitch leaves overlapping are pushed apart again
    by plan_beta, one pair at a time, which undid the stitch on k22)."""
    if ox <= 0 or oz <= 0:
        return 0.0
    return max(min(ox, oz) - p.overlap_tol_m, 0.0) / p.overlap_sigma_m


def adjust(frames: dict, cons: list[dict], links: list[dict], ref: str, p,
           fixed_rot: dict | None = None, obstacles: dict | None = None, prior: dict | None = None) -> tuple[dict, list]:
    """Rooms' rigid placements (psi, T) in the reference room's frame: rotations by a spanning tree over the
    constraints (strongest first), translations by robust least squares with a no-overlap penalty.

    fixed_rot: rooms whose rotation relative to the reference the pose graph already measured (placed in one
    component with it): a constraint that turns such a room another way is dropped. A door association can be wrong
    by 90 deg (sim k38: a threshold photo in a corner between two doors put the bedroom on the bathroom's wall); the
    pose graph's rotations come from feature matches and the Manhattan snap and were right on all three flats."""
    from scipy.optimize import least_squares
    psi = {ref: 0.0}
    T0 = {ref: np.zeros(2)}
    used, dropped = [], []
    fixed_rot = dict(fixed_rot or {})
    todo = []
    for c in sorted(cons, key=lambda c: -c["weight"]):
        A, B = c["A"], c["B"]
        if A in fixed_rot and B in fixed_rot and abs(_wrap(fixed_rot[B] - fixed_rot[A] - c["psi"])) > 1e-6:
            dropped.append(dict(A=A, B=B, kind=c["kind"], reason="rotation contradicts the pose graph"))
            continue
        todo.append(c)
    changed = True
    while changed:
        changed = False
        for c in list(todo):
            A, B = c["A"], c["B"]
            new = (B, _wrap(psi[A] + c["psi"])) if A in psi and B not in psi else \
                (A, _wrap(psi[B] - c["psi"])) if B in psi and A not in psi else None
            if new and new[0] in fixed_rot and abs(_wrap(new[1] - fixed_rot[new[0]])) > 1e-6:
                dropped.append(dict(A=A, B=B, kind=c["kind"], reason="rotation contradicts the pose graph"))
                todo.remove(c)
                continue
            if A in psi and B not in psi:
                psi[B] = _wrap(psi[A] + c["psi"])
                T0[B] = _rot2(psi[A]) @ c["t"] + T0[A]
            elif B in psi and A not in psi:
                psi[A] = _wrap(psi[B] - c["psi"])
                T0[A] = T0[B] - _rot2(psi[A]) @ c["t"]
            elif A in psi and B in psi:
                if abs(_wrap(psi[B] - psi[A] - c["psi"])) > 1e-6:
                    dropped.append(dict(A=A, B=B, kind=c["kind"], reason="rotation contradicts the tree"))
                    todo.remove(c)
                    continue
            else:
                continue
            used.append(c)
            todo.remove(c)
            changed = True
    rooms = [r for r in psi if r != ref]
    if not rooms:
        return {ref: (0.0, np.zeros(2))}, dropped
    idx = {r: i for i, r in enumerate(rooms)}

    def T(x, r):
        return np.zeros(2) if r == ref else x[2 * idx[r]:2 * idx[r] + 2]

    boxes = {r: frames[r]["box"] for r in psi}
    same = [L for L in links if L.get("same_spot") and not L.get("rejected") and L["A"] in psi and L["B"] in psi
            and not L.get("used_as")]
    fixed_boxes = [_box_moved(frames[r]["box"], ps, T) for r, (ps, T) in (obstacles or {}).items() if r not in psi]

    def resid(x):
        out = []
        for r, (_, Tpg) in (prior or {}).items():         # the pose graph's own placement of the room
            if r in idx and p.pose_graph_sigma_m > 0:
                out += list((x[2 * idx[r]:2 * idx[r] + 2] - Tpg) / p.pose_graph_sigma_m)
        for c in used:
            A, B = c["A"], c["B"]
            e = _rot2(psi[A]).T @ (T(x, B) - T(x, A)) - c["t"]
            if c["n"] is not None:
                n = c["n"]
                tg = np.array([-n[1], n[0]])
                out += [float(e @ n) / c["sn"], float(e @ tg) / c["st"]]
            else:
                out += [e[0] / c["sn"], e[1] / c["st"]]
        for L in same:                                    # doorway pair: both photos on one spot (weak)
            ca, cb = L["same_spot"]
            A, B = L["A"], L["B"]
            e = (_rot2(psi[A]) @ ca + T(x, A)) - (_rot2(psi[B]) @ cb + T(x, B))
            out += list(e / p.same_spot_sigma_m)
        rs = list(psi)
        for i, a in enumerate(rs):                         # rooms do not overlap
            ba = _box_moved(boxes[a], psi[a], T(x, a))
            for b in rs[i + 1:]:
                bb = _box_moved(boxes[b], psi[b], T(x, b))
                ox = min(ba[1], bb[1]) - max(ba[0], bb[0])
                oz = min(ba[3], bb[3]) - max(ba[2], bb[2])
                out.append(_pen(ox, oz, p))
            for bb in fixed_boxes:                         # rooms the stitch does not move are obstacles
                ox = min(ba[1], bb[1]) - max(ba[0], bb[0])
                oz = min(ba[3], bb[3]) - max(ba[2], bb[2])
                out.append(_pen(ox, oz, p) if a != ref else 0.0)
        return np.array(out)

    x0 = np.concatenate([T0[r] for r in rooms])
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=2.0, max_nfev=400)
    return {r: (float(psi[r]), np.array(T(sol.x, r))) for r in psi}, dropped


# ------------------------------------------------------------------------------------------- 5. into the plan
def _M_from_anchor(anchor: dict) -> np.ndarray:
    amap = anchor["side_to_plan"]
    return np.stack([_DIR[amap["+x"]], _DIR[amap["+z"]]], 1)


def stitch_rooms(views: dict, matches: dict, lays: dict, pairs: list[dict], names: list[str], edges: list,
                 poses: dict, k: float, scene: dict, p, log=print) -> dict:
    """Door-anchored stitching (D-081). Updates the layouts' anchors (centre and side directions in the plan) and
    moves each stitched room's cameras with it in the scene; returns the report for scene_info["door_stitch"]."""
    from floorplan.photo import link as L
    comps = []                     # one pose set per pose-graph component (the frontend's block and fallbacks are
    for c in L.components(names, edges):          # one dict there, but only a component's poses are consistent)
        if all(n in poses for n in c):
            comps.append({n: poses[n] for n in c})
            continue
        cp = L.solve_component(c, [e for e in edges if e.a in c and e.b in c]) if len(c) > 1 else \
            {c[0]: (0.0, np.zeros(3), 0.0)}
        comps.append({n: (th, k * np.asarray(t, float), ls) for n, (th, t, ls) in cp.items()})
    frames = room_frames(views, lays, {}, comps, k)
    placed_by_pnp = place_in_rooms(frames, views, matches, k, p) if p.intra_pnp else {}
    if placed_by_pnp:
        log("[photo/door] placed in their rooms by PnP: " + ", ".join(
            f"{q} (on {o}, {n} inliers)" for q, (_, o, n) in sorted(placed_by_pnp.items())))
    doors = {r: detect_doors(r, fr, views, pairs, k, p) for r, fr in frames.items()}
    for r, ds in doors.items():
        log(f"[photo/door] {r}: {len(fr_ := frames[r]['photos'])} photo(s) in its frame; doors: " + (", ".join(
            f"{d['side']} at {d['c']:+.2f} m" + (f" w {d['w']:.2f}" if d["w"] else "") + f" ({d['src']})"
            for d in ds) or "none"))
    hyps = pnp_links(frames, views, matches, k, p, log)
    links = correspondences(frames, doors, hyps, pairs, views, p, log)
    anchored = [r for r in frames if (lays[r].get("anchor") or {}).get("side_to_plan")]
    cons = _constraints(links, frames, p)
    deg = {r: sum(1 for c in cons if r in (c["A"], c["B"])) for r in frames}
    report = dict(rooms={}, links=[_link_report(Lk) for Lk in links], doors={r: [_door_report(d) for d in ds]
                                                                             for r, ds in doors.items()},
                  hypotheses=len(hyps), placed_by_pnp={q: dict(room=r, on=o, inliers=n)
                                                       for q, (r, o, n) in placed_by_pnp.items()},
                  frames={r: dict(photos={n: [round(float(y), 4), round(float(c[0]), 3), round(float(c[1]), 3), how]
                                          for n, (y, c, how) in fr["photos"].items()},
                                  box=[round(float(x), 3) for x in fr["box"]]) for r, fr in frames.items()})
    if not anchored or not cons:
        report["result"] = "nothing stitched (no anchored room or no door link)"
        log(f"[photo/door] {report['result']}")
        return report
    ref = max(anchored, key=lambda r: (deg[r], _area(frames[r]["box"])))
    # rotations the pose graph measured: rooms anchored by photos in the reference room's component
    comp_of = {n: i for i, c in enumerate(comps) for n in c}
    ref_comp = {comp_of[n] for n in (lays[ref].get("fit_poses") or {}) if n in comp_of}
    M0 = _M_from_anchor(lays[ref]["anchor"])
    fixed_rot = {}
    for r in frames:
        a = lays[r].get("anchor") or {}
        fit = [n for n in (lays[r].get("fit_poses") or {}) if n in comp_of]
        if a.get("side_to_plan") and fit and {comp_of[n] for n in fit} & ref_comp:
            Mr = M0.T @ _M_from_anchor(a)
            fixed_rot[r] = float(np.arctan2(Mr[0, 1], Mr[0, 0]))
    report["rotation_from_pose_graph"] = sorted(fixed_rot)
    cons = _constraints(links, frames, p, fixed_rot)
    report["links"] = [_link_report(Lk) for Lk in links]
    # the pose graph's own placement of those rooms (reference frame): obstacles if not stitched, and a sanity bound
    pg = {r: (fixed_rot[r], M0.T @ (np.asarray(lays[r]["anchor"]["centre_uv"], float) -
                                    np.asarray(lays[ref]["anchor"]["centre_uv"], float))) for r in fixed_rot}
    capped = {}
    while True:
        use = [c for c in cons if c["A"] not in capped and c["B"] not in capped]
        place, dropped = adjust(frames, use, links, ref, p, fixed_rot, obstacles=pg, prior=pg)
        far = {r: float(np.linalg.norm(place[r][1] - pg[r][1])) for r in place if r in pg and r != ref}
        far = {r: d for r, d in far.items() if d > p.stitch_max_move_m}
        if not far:
            break
        r = max(far, key=far.get)          # the stitch would carry a room far from where the pose graph put it:
        capped[r] = round(far[r], 2)       # a wrong door association is likelier than a 1 m pose-graph error
    report["kept_by_move_cap"] = capped
    report["reference_room"] = ref
    report["dropped_constraints"] = dropped
    a_ref = lays[ref]["anchor"]
    M_ref, c_ref = _M_from_anchor(a_ref), np.asarray(a_ref["centre_uv"], float)
    cam_names = [str(x) for x in np.asarray(scene.get("cam_names", []))]
    moved_cams = {}
    # what the stitch changes, so one front-end run can be scored with and without it (scripts/ab_door_stitch.py)
    report["undo"] = dict(layouts={r: dict(anchor=lays[r].get("anchor"), sides_plan=lays[r].get("sides_plan"))
                                   for r in frames if r in place},
                          T_wc={n: np.asarray(scene["T_wc"])[cam_names.index(n)].tolist() for n in cam_names}
                          if "T_wc" in scene else {})
    for r in frames:
        if r not in place or r in capped:
            report["rooms"][r] = dict(stitched=False, reason="no matched door: placement kept" if r not in capped else
                                      f"the stitch would move it {capped[r]} m from the pose graph's placement "
                                      f"(> {p.stitch_max_move_m} m): placement kept")
            continue
        psi_r, T_r = place[r]
        M_new = M_ref @ _rot2(psi_r)
        c_new = M_ref @ T_r + c_ref
        old = lays[r].get("anchor")
        amap = {s: _side_of_normal(M_new @ _DIR[s]) for s in SIDES}
        entry = dict(stitched=True, rotation_deg=round(float(np.degrees(psi_r)), 1),
                     centre_uv=[round(float(x), 3) for x in c_new])
        if old and old.get("side_to_plan"):
            M_old, c_old = _M_from_anchor(old), np.asarray(old["centre_uv"], float)
            D = M_new @ M_old.T
            dt = c_new - D @ c_old
            entry["moved_m"] = round(float(np.linalg.norm(c_new - c_old)), 3)
            entry["turned_deg"] = round(float(np.degrees(np.arctan2(D[1, 0], D[0, 0]))), 1)
            for n in frames[r]["photos"]:
                if n in cam_names:
                    moved_cams[n] = (D, dt)
        else:
            entry["moved_m"] = None
            entry["note"] = "was not anchored (spin photos not placed): placed by its door"
        new_anchor = dict(old or {}, centre_uv=[float(c_new[0]), float(c_new[1])], side_to_plan=amap,
                          door_stitch=dict(reference=ref, rotation_deg=entry["rotation_deg"]))
        new_anchor.setdefault("placed_spin_photos", [])
        new_anchor.setdefault("snap_residual_deg", 0.0)
        new_anchor.setdefault("yaw_spread_deg", 0.0)
        lays[r]["anchor"] = new_anchor
        lays[r]["sides_plan"] = {amap[s]: dict(lays[r]["sides"][s], layout_side=s) for s in SIDES}
        report["rooms"][r] = entry
    # a doorway-pair photo placed with its partner's room (not with its own) moves with the partner's room
    partner = {pr["a"]: pr["b"] for pr in pairs} | {pr["b"]: pr["a"] for pr in pairs}
    for n in cam_names:
        if n in moved_cams or n not in partner or partner[n] not in moved_cams:
            continue
        if not any(n in fr["photos"] for fr in frames.values()):
            moved_cams[n] = moved_cams[partner[n]]
    if moved_cams and "T_wc" in scene:
        T_wc = np.array(scene["T_wc"], float)
        for n, (D, dt) in moved_cams.items():
            i = cam_names.index(n)
            R3 = np.array([[D[0, 0], 0, D[0, 1]], [0, 1, 0], [D[1, 0], 0, D[1, 1]]])
            T_wc[i, :3, :3] = R3 @ T_wc[i, :3, :3]
            T_wc[i, :3, 3] = R3 @ T_wc[i, :3, 3] + np.array([dt[0], 0.0, dt[1]])
        scene["T_wc"] = T_wc
        scene["traj"] = T_wc[:, :3, 3].astype(np.float32)
    report["cameras_moved"] = len(moved_cams)
    n_st = sum(1 for v in report["rooms"].values() if v.get("stitched"))
    report["result"] = f"{n_st}/{len(frames)} rooms stitched at doors (reference {ref})"
    log(f"[photo/door] {report['result']}: " + "; ".join(
        f"{r} " + (f"moved {v.get('moved_m')} m, turned {v.get('turned_deg', 0)} deg" if v.get("stitched") and
                   v.get("moved_m") is not None else ("placed by its door" if v.get("stitched") else "kept"))
        for r, v in sorted(report["rooms"].items())))
    return report


def moved_rooms(info: dict) -> list[str]:
    """Rooms the stitch moved or placed (not its reference room); [] when it did not run or moved nothing."""
    ds = info.get("door_stitch") or {}
    return sorted(r for r, v in (ds.get("rooms") or {}).items() if v.get("stitched") and r != ds.get("reference_room"))


def undo_stitch(scene: dict, info: dict) -> tuple[dict, dict]:
    """The scene and info as they were before the stitch moved the rooms (anchors, sides, cameras), from the
    stitch's undo record; the inputs are not changed. Paired A/B (scripts/ab_door_stitch.py) and run_capture's
    retry when plan_beta times out on a stitched scene."""
    scene, info = dict(scene), copy.deepcopy(info)
    u = (info.get("door_stitch") or {}).get("undo") or {}
    for r, v in (u.get("layouts") or {}).items():
        lay = info["room_layouts"][r]
        for k in ("anchor", "sides_plan"):
            if v.get(k) is None:
                lay.pop(k, None)
            else:
                lay[k] = v[k]
    if u.get("T_wc"):
        names = [str(x) for x in scene["cam_names"]]
        T = np.array(scene["T_wc"], float)
        for n, t in u["T_wc"].items():
            T[names.index(n)] = np.asarray(t, float)
        scene["T_wc"] = T
        scene["traj"] = T[:, :3, 3].astype(np.float32)
    return scene, info


def _door_report(d: dict) -> dict:
    return dict(id=d["id"], side=d["side"], centre_m=round(d["c"], 3), face_m=round(float(d["face"]), 3),
                width_m=None if d["w"] is None else
                round(d["w"], 3), source=d["src"], photos=d["photos"][:6], pairs=d["pairs"],
                thickness_m=None if d.get("thickness") is None else round(d["thickness"], 3))


def _link_report(Lk: dict) -> dict:
    out = {k: v for k, v in Lk.items() if k not in ("door_pair", "t", "psi", "same_spot")}
    if "t" in Lk:
        out["t_m"] = [round(float(x), 3) for x in Lk["t"]]
        out["rotation_deg"] = round(float(np.degrees(Lk["psi"])), 1)
    return out
