"""Pair predicted walls with ground-truth walls W1..Wn inside one matched room pair.

Why: pairing by cyclic ORDER alone breaks as soon as a plan has an extra short wall (a 3-10 cm door reveal or jog, a
0.1-0.3 m stub; common in LiDAR plans): every pair after it shifts by one, and a plan whose real walls are within 1 cm
scores tens of centimetres. Two pairings, the first is the default whenever it can run:

  * geometric (GT polygons available; the simulator writes them to <capture>/gt/sim_gt.json):
      1. put the predicted room in the GT room's frame with a rigid transform (no scale). The plan's (u, v) is
         mirrored seen from above (issue I-001), so it is un-mirrored first. Rotation: the 4 Manhattan candidates from
         the dominant wall directions. Translation: centroids, then per axis the shift that puts the most predicted
         wall length within 0.30 m of a same-facing GT wall (a consensus: a part of the room the plan lacks or adds
         must not drag the rest), re-centred on those walls; then point-to-line ICP on them (fine-tunes the
         rotation, capped at +-5 deg). When the whole plan sits well on the whole GT (LiDAR, video), the global
         transform refined per room (<= 0.5 m) is preferred: it fixes the 180 deg ambiguity of rectangular rooms, and
         unlabelled rooms are then paired by overlap. Otherwise (photo tier: rooms not stitched) the best of the 4
         room-local candidates by IoU wins, ties broken by door walls agreeing. Weak overlap (IoU < 0.5) or a
         90-deg-turned fit almost as good is reported as ambiguous.
      2. each predicted wall goes to the GT wall on the same line: same inward-facing direction (+-10 deg), offset
         <= 0.30 m, overlapping it along the line. Several predicted pieces on one GT wall are merged: end-to-end
         span along the line.
  * geometric on a reconstructed outline (tape GT, lengths only; the default for a tape GT, decision D-072): the
    room's outline is rebuilt from W1..Wn (clockwise from above, every corner a right angle: reconstruct_outlines),
    the predicted room is registered onto it by IoU alone (register_outline: 4 rotations, the translation of highest
    IoU; near-ties broken by W1 holding a predicted door) and paired as in 1-2 above. Lengths never choose the
    pairing. Several outlines fit the tape (a bump or the same notch turned inwards): every one is scored and the
    WORST is reported. No outline fits: constrained_pairs, an order pairing with the axis and the door as hard
    constraints.
  * order-merged (tape GT, lengths only): predicted walls clockwise from above, cyclic monotone alignment to W1..Wn
    in which a GT wall may take a run of consecutive predicted walls merged along their main line (collinear pieces
    split by 3-10 cm jogs or door reveals), short predicted walls are left out cheaply (0.1 per metre), a GT wall is
    left unpaired only if nothing is within ~50% of it, and a door wall is preferred as W1. The GT lengths decide
    whether to merge: a 0.2 m niche the tape measured stays a wall, a 3.6 cm LiDAR jog does not.

Merged length = geometric span + the plan's own per-corner correction at the two end corners (LiDAR plans add an
inward-bias correction of about +-0.7 cm per corner to each wall length; it is estimated from the room's walls as
(length value - geometric length) = c * (corner sign sum), convex +1, reflex -1).
"""
from __future__ import annotations

import itertools

import numpy as np

PARAMS = dict(angle_deg=10.0, offset_m=0.30, short_m=0.35, reverse_penalty_m=0.10, door_bonus=0.02,
              prefer_global_iou=0.05, global_min_iou=0.5, global_max_shift_m=0.5, room_min_iou=0.1,
              icp_max_rot_deg=5.0, weak_iou=0.5, ambiguous_iou_gap=0.03,
              max_run=9, skip_pred_per_m=0.1, skip_gt_m=0.1, skip_gt_rel=0.5,
              # anchored alignment (tape GT); costs in relative-error units, see anchored_pairs
              max_gt_run=5, miss_gt=0.5, merge_gt=0.15, skip_pred_rel=0.5, anchor_margin=0.25,
              ambiguous_margin=0.05,
              # tape GT outline (reconstruct_outlines, register_outline, constrained_pairs)
              close_abs_m=0.03, close_rel=0.005, max_outline_walls=20, raster_m=0.02, tie_iou=0.02,
              fallback_margin=0.25)


# ------------------------------------------------------------------------------------------------ small geometry

def _val(meas):
    if not meas:
        return None, None, None
    if meas.get("ci95") is not None:
        lo, hi = meas["ci95"]
        return meas.get("value"), lo, hi
    return meas.get("value"), meas.get("lo"), meas.get("hi")


def signed_area(poly) -> float:
    p = np.asarray(poly, float)
    return float(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def plan_mirrored(plan: dict) -> bool:
    """True when the plan's (u, v) is mirrored seen from above (I-001: (u, v) = (x, z), +y up, so v points DOWN).
    plan.json states it in coordinate_frame.view_from_above; mirrored is also the default."""
    vfa = (plan.get("coordinate_frame") or {}).get("view_from_above") or {}
    return str(vfa.get("v", "down")).lower() != "up"


class Segs:
    """Directed wall segments in one frame; inward normals from the orientation of each segment's polygon
    (sigma = +1 for a counter-clockwise polygon; a scalar, or one value per segment)."""

    def __init__(self, A, B, sigma):
        self.A, self.B = np.asarray(A, float).reshape(-1, 2), np.asarray(B, float).reshape(-1, 2)
        self.sigma = np.where(np.broadcast_to(np.asarray(sigma, float), (len(self.A),)) >= 0, 1.0, -1.0)
        D = self.B - self.A
        self.L = np.hypot(D[:, 0], D[:, 1])
        self.d = D / np.maximum(self.L, 1e-12)[:, None]
        self.n = self.sigma[:, None] * np.stack([-self.d[:, 1], self.d[:, 0]], axis=1)    # inward normal

    def moved(self, R, t) -> "Segs":
        return Segs(self.A @ R.T + t, self.B @ R.T + t, self.sigma * np.sign(np.linalg.det(R)))

    @staticmethod
    def cat(items: list["Segs"]) -> "Segs":
        return Segs(np.vstack([x.A for x in items]), np.vstack([x.B for x in items]),
                    np.concatenate([x.sigma for x in items]))


def _dominant_angle(S: Segs) -> float:
    phi = np.arctan2(S.d[:, 1], S.d[:, 0])
    return 0.25 * float(np.arctan2(np.sum(S.L * np.sin(4 * phi)), np.sum(S.L * np.cos(4 * phi))))


def _polygon(p):
    from shapely.geometry import Polygon
    g = Polygon(np.asarray(p, float))
    return g if g.is_valid else g.buffer(0)


def iou(P, G) -> float:
    a, b = _polygon(P), _polygon(G)
    u = a.union(b).area
    return float(a.intersection(b).area / u) if u > 0 else 0.0


def line_match(P: Segs, G: Segs, off_tol: float, ang_deg: float, gap_tol: float | None = None) -> list[tuple[int, float]]:
    """For every predicted segment: (index of the GT segment on the same line or -1, perpendicular offset).

    Same line = inward normals within ang_deg, offset of the predicted midpoint from the GT line <= off_tol, and
    overlapping along the line by at least half of the shorter of the two (gap_tol None; as floorplan/eval/match.py),
    or at most gap_tol apart (alignment only). A piece mostly past the GT wall's end corner is not part of that wall (e.g. the
    side of an alcove at the corner). Among candidates: smallest offset + gap, then most overlap (two collinear GT
    walls, e.g. either side of a bump, differ by overlap)."""
    cos_tol = np.cos(np.radians(ang_deg))
    out = []
    for i in range(len(P.L)):
        best, best_cost, best_off = -1, np.inf, np.nan
        if P.L[i] > 1e-6:
            mid = 0.5 * (P.A[i] + P.B[i])
            for j in range(len(G.L)):
                if P.n[i] @ G.n[j] < cos_tol:
                    continue
                off = abs(G.n[j] @ (mid - G.A[j]))
                s0, s1 = sorted(((P.A[i] - G.A[j]) @ G.d[j], (P.B[i] - G.A[j]) @ G.d[j]))
                ov = min(s1, G.L[j]) - max(s0, 0.0)
                gap = max(0.0, -ov)
                if off > off_tol or (ov < 0.5 * min(P.L[i], G.L[j]) if gap_tol is None else gap > gap_tol):
                    continue
                cost = off + gap + 0.1 * (1.0 - max(ov, 0.0) / P.L[i])
                if cost < best_cost:
                    best, best_cost, best_off = j, cost, off
        out.append((best, float(best_off)))
    return out


def _axis_shift(pairs: list[tuple[Segs, Segs]], axis: np.ndarray, tol: float, max_shift: float = np.inf) -> float:
    """1-D shift along `axis` (unit) that puts the most predicted wall length within tol of a same-facing GT wall line
    of the same room pair (walls whose inward normal is along +-axis); ties -> the smallest shift. Then re-centred on
    those inlier walls (length-weighted mean residual), so a room a little too big or too small sits symmetric.
    A consensus, not least squares over all walls: an extra or missing part of the room must not drag the rest."""
    cos20 = np.cos(np.radians(20.0))
    pp, fp, lp, rp, pg, fg, rg = [], [], [], [], [], [], []
    for k, (P, G) in enumerate(pairs):
        sp, sg = np.abs(P.n @ axis) > cos20, np.abs(G.n @ axis) > cos20
        pp.append((0.5 * (P.A + P.B))[sp] @ axis); fp.append(np.sign(P.n[sp] @ axis)); lp.append(P.L[sp])
        rp.append(np.full(sp.sum(), k))
        pg.append((0.5 * (G.A + G.B))[sg] @ axis); fg.append(np.sign(G.n[sg] @ axis)); rg.append(np.full(sg.sum(), k))
    pp, fp, lp, rp, pg, fg, rg = (np.concatenate(x) for x in (pp, fp, lp, rp, pg, fg, rg))
    if not len(pp) or not len(pg):
        return 0.0
    same = (fp[:, None] == fg[None, :]) & (rp[:, None] == rg[None, :])
    diff = pg[None, :] - pp[:, None]                      # shift that puts predicted wall i on GT wall j
    best = (-1.0, 0.0, None)
    for c in np.append(diff[same], 0.0):
        if abs(c) > max_shift:
            continue
        d = np.where(same, np.abs(diff - c), np.inf)
        inl = d.min(axis=1) <= tol
        score = float(lp[inl].sum())
        if score > best[0] + 1e-9 or (abs(score - best[0]) <= 1e-9 and abs(c) < abs(best[1])):
            best = (score, float(c), inl)
    score, c, inl = best
    if inl is None or not inl.any():
        return c
    j = np.argmin(np.where(same, np.abs(diff - c), np.inf), axis=1)
    r = (diff[np.arange(len(pp)), j] - c)[inl]
    return c + float(np.average(r, weights=lp[inl]))


def consensus_align(pairs: list[tuple[Segs, Segs]], R, t, p=None, max_shift: float = np.inf):
    """Translation by per-axis consensus (axes = the GT's dominant wall directions; shift per axis <= max_shift),
    then point-to-line ICP on the walls within the pairing tolerance only, to fine-tune rotation (capped) and
    translation."""
    p = p or PARAMS
    th = _dominant_angle(Segs.cat([G for _, G in pairs]))
    t = np.asarray(t, float)
    for _ in range(2):
        moved = [(P.moved(R, t), G) for P, G in pairs]
        for ax in (np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])):
            t = t + _axis_shift(moved, ax, p["offset_m"], max_shift) * ax
            moved = [(P.moved(R, t), G) for P, G in pairs]
        R, t = icp(pairs, R, t, max_d=p["offset_m"], max_rot_deg=p["icp_max_rot_deg"])
    return R, t


def icp(pairs: list[tuple[Segs, Segs]], R, t, max_d=1.0, max_rot_deg=5.0, min_len=0.35, iters=30):
    """Point-to-line ICP of predicted walls onto GT walls (same inward normal within 30 deg), x -> R x + t.

    pairs: (predicted segments in the un-mirrored plan frame, GT segments) per room pair. Least squares on both
    endpoints of every predicted wall >= min_len, weighted by length; the rotation correction is capped."""
    R, t = np.asarray(R, float), np.asarray(t, float)
    R0 = R.copy()
    for _ in range(iters):
        rows, wts, pts = [], [], []
        for P, G in pairs:
            Q = P.moved(R, t)
            for i, (j, _) in enumerate(line_match(Q, G, max_d, 30.0, gap_tol=max_d)):
                if j < 0 or Q.L[i] < min_len:
                    continue
                for x in (Q.A[i], Q.B[i]):
                    pts.append(x)
                    rows.append((j, G.n[j], G.A[j], x))
                    wts.append(0.5 * Q.L[i])
        if len(rows) < 3:
            break
        c = np.mean(np.asarray(pts), axis=0)
        J = np.array([[n @ np.array([-(x - c)[1], (x - c)[0]]), n[0], n[1]] for _, n, _, x in rows])
        r = np.array([n @ (x - a) for _, n, a, x in rows])
        w = np.sqrt(np.asarray(wts))
        delta = np.linalg.lstsq(J * w[:, None], -r * w, rcond=None)[0]
        dR = _rot(delta[0])
        R_new = dR @ R
        # cap the total rotation change against the starting rotation
        ang = np.degrees(np.arctan2(*(R_new @ R0.T)[[1, 0], [0, 0]]))
        if abs(ang) > max_rot_deg:
            break
        R, t = R_new, dR @ (t - c) + c + delta[1:]
        if abs(delta[0]) < 1e-5 and np.hypot(*delta[1:]) < 1e-5:
            break
    return R, t


# ------------------------------------------------------------------------------------------- plan-side helpers

def room_walls(plan: dict, room: dict) -> list[dict]:
    W = {w["id"]: w for w in plan["walls"]}
    return [W[i] for i in room["wall_ids"] if i in W]


def corner_model(room: dict, ws: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    """Corner signs (convex +1, reflex -1) at every wall's p0 and p1, and the per-corner length correction c the plan
    applied: (length value - geometric length) = c * (s0 + s1), fitted over the room's walls (0 for most tiers)."""
    poly = np.asarray(room.get("polygon") or [], float).reshape(-1, 2)
    if len(poly) < 3:
        return np.zeros(len(ws)), np.zeros(len(ws)), 0.0
    sig = np.sign(signed_area(poly)) or 1.0
    e_prev, e_next = poly - np.roll(poly, 1, axis=0), np.roll(poly, -1, axis=0) - poly
    cross = e_prev[:, 0] * e_next[:, 1] - e_prev[:, 1] * e_next[:, 0]
    vs = np.where(np.abs(cross) < 1e-9, 0, np.where(cross * sig > 0, 1, -1))

    def sign_at(p):
        return vs[int(np.argmin(np.linalg.norm(poly - np.asarray(p, float), axis=1)))]
    s0 = np.array([sign_at(w["p0"]) for w in ws], float)
    s1 = np.array([sign_at(w["p1"]) for w in ws], float)
    dl = np.array([((_val(w["length"])[0] or 0.0) - float(np.linalg.norm(np.subtract(w["p1"], w["p0"])))) for w in ws])
    s = s0 + s1
    c = float(dl @ s / (s @ s)) if s @ s > 0 else 0.0
    return s0, s1, c


def piece_length(ws: list[dict], idx: list[int], s0, s1, c) -> tuple[float, float | None, float | None]:
    """(value, lo, hi) of predicted walls idx taken as one wall: a single wall keeps its own measurement; several
    collinear pieces give the end-to-end span along their mean direction, plus the plan's corner correction at the
    two end corners; interval from the two end pieces (approximation)."""
    if len(idx) == 1:
        return _val(ws[idx[0]]["length"])
    p0 = np.array([ws[i]["p0"] for i in idx], float)
    p1 = np.array([ws[i]["p1"] for i in idx], float)
    u = (p1 - p0).sum(axis=0)
    u = u / (np.linalg.norm(u) or 1.0)
    a, b = p0 @ u, p1 @ u
    first, last = int(np.argmin(np.minimum(a, b))), int(np.argmax(np.maximum(a, b)))
    v = float(max(b.max(), a.max()) - min(a.min(), b.min()) + c * (s0[idx[first]] + s1[idx[last]]))
    lo_hi = []
    for k in (1, 2):                              # lo, hi offsets of the two end pieces
        e = [_val(ws[idx[j]]["length"]) for j in (first, last)]
        if any(x[k] is None or x[0] is None for x in e):
            return v, None, None
        lo_hi.append(float(np.sqrt(0.5 * sum((x[k] - x[0]) ** 2 for x in e))))
    return v, v - lo_hi[0], v + lo_hi[1]


def _has_door(w: dict, openings: dict) -> bool:
    return any(openings.get(o, {}).get("kind") in ("door", "passage") for o in w.get("opening_ids", []))


# ------------------------------------------------------------------------------------------ geometric pairing

def plan_room_segs(plan: dict, room: dict, mirror: bool) -> tuple[list[dict], Segs, np.ndarray]:
    ws = room_walls(plan, room)
    M = np.diag([1.0, -1.0]) if mirror else np.eye(2)
    poly = np.asarray(room["polygon"], float) @ M.T
    S = Segs(np.array([w["p0"] for w in ws], float) @ M.T, np.array([w["p1"] for w in ws], float) @ M.T,
             np.sign(signed_area(poly)) or 1.0)
    return ws, S, poly


def gt_room_segs(gtg: dict, names: list[str]) -> Segs:
    return Segs([gtg["walls"][n]["a"] for n in names], [gtg["walls"][n]["b"] for n in names],
                np.sign(signed_area(gtg["poly"])) or 1.0)


def _centroid(poly) -> np.ndarray:
    c = _polygon(poly).centroid
    return np.array([c.x, c.y])


def global_alignment(plan: dict, gt_geom: dict, mirror: bool, p=PARAMS) -> dict | None:
    """One rigid transform for the whole plan, independent of the room pairing: all plan walls against all GT walls,
    score = IoU of the union of the plan's rooms with the union of the GT rooms. dict(R, t, score) of the best of
    the 4 Manhattan candidates."""
    from shapely import affinity
    from shapely.ops import unary_union
    plan_segs = [plan_room_segs(plan, r, mirror) for r in plan["rooms"] if len(r.get("polygon") or []) >= 3]
    gts = [gt_room_segs(g, sorted(g["walls"], key=_wnum)) for g in gt_geom.values()]
    if not plan_segs or not gts:
        return None
    allP, allG = Segs.cat([x[1] for x in plan_segs]), Segs.cat(gts)
    pu = unary_union([_polygon(x[2]) for x in plan_segs])
    gu = unary_union([_polygon(g["poly"]) for g in gt_geom.values()])
    base = _dominant_angle(allG) - _dominant_angle(allP)
    cp, cg = np.array(pu.centroid.coords[0]), np.array(gu.centroid.coords[0])
    best = None
    for k in range(4):
        R = _rot(base + k * np.pi / 2)
        R, t = consensus_align([(allP, allG)], R, cg - R @ cp, p)
        moved = affinity.affine_transform(pu, [R[0, 0], R[0, 1], R[1, 0], R[1, 1], t[0], t[1]])
        score = moved.intersection(gu).area / max(moved.union(gu).area, 1e-9)
        if best is None or score > best["score"]:
            best = dict(R=R, t=t, score=float(score))
    return best


def pair_rooms_by_overlap(plan: dict, gt_geom: dict, glob: dict, mirror: bool, fixed: dict, p=PARAMS) -> dict:
    """GT room -> plan room index by polygon IoU under the global transform (Hungarian, IoU >= room_min_iou);
    pairs in `fixed` (exact label matches) are kept."""
    from scipy.optimize import linear_sum_assignment
    rest_g = [g for g in gt_geom if g not in fixed]
    rest_p = [i for i in range(len(plan["rooms"])) if i not in fixed.values()]
    out = dict(fixed)
    if not rest_g or not rest_p:
        return out
    C = np.zeros((len(rest_g), len(rest_p)))
    for b, i in enumerate(rest_p):
        room = plan["rooms"][i]
        if len(room.get("polygon") or []) < 3:
            continue
        _, _, ppoly = plan_room_segs(plan, room, mirror)
        for a, g in enumerate(rest_g):
            C[a, b] = iou(ppoly @ glob["R"].T + glob["t"], gt_geom[g]["poly"])
    for a, b in zip(*linear_sum_assignment(-C)):
        if C[a, b] >= p["room_min_iou"]:
            out[rest_g[a]] = rest_p[b]
    return out


def room_overlap(plan: dict, room: dict, gtg: dict, mirror: bool, glob: dict | None = None, p=PARAMS) -> float:
    """IoU of a plan room with a GT room outline, to choose among plan rooms that carry the GT room's label. Under the
    whole-plan transform when the whole plan fits the GT (glob score >= global_min_iou); else the room alone at its
    best fit (4 rotations in 90 deg steps from the dominant edge directions, translation of highest IoU), so only size
    and shape count."""
    if len(room.get("polygon") or []) < 3:
        return 0.0
    M = np.diag([1.0, -1.0]) if mirror else np.eye(2)
    Q, G = np.asarray(room["polygon"], float) @ M.T, np.asarray(gtg["poly"], float)
    if _polygon(Q).area <= 0:
        return 0.0
    if glob is not None and glob["score"] >= p["global_min_iou"]:
        return iou(Q @ glob["R"].T + glob["t"], G)
    edges = lambda X: Segs(X, np.roll(X, -1, axis=0), 1.0)     # noqa: E731
    base = _dominant_angle(edges(G)) - _dominant_angle(edges(Q))
    best = 0.0
    for k in range(4):
        Qk = Q @ _rot(base + k * np.pi / 2).T
        best = max(best, iou_translation(Qk, G, _centroid(G) - _centroid(Qk), p["raster_m"])[1])
    return best


def _door_agreement(assign, ws, openings, gt_doors: list[bool]) -> float:
    """Fraction of GT door walls whose matched predicted pieces include a wall with a door or passage."""
    n = sum(gt_doors)
    if not n:
        return 0.0
    hit = {j for i, (j, _) in enumerate(assign) if j >= 0 and gt_doors[j] and _has_door(ws[i], openings)}
    return len(hit) / n


def align_room(plan, room, gtg, names, mirror, glob, openings, p=PARAMS) -> dict:
    """Rigid transform of one predicted room into its GT room's frame (see the module docstring)."""
    ws, P, ppoly = plan_room_segs(plan, room, mirror)
    G = gt_room_segs(gtg, names)
    gpoly = gtg["poly"]
    gt_doors = [gtg["walls"][n].get("door", False) for n in names]
    cands = []
    base = _dominant_angle(G) - _dominant_angle(P)
    for k in range(4):
        R = _rot(base + k * np.pi / 2)
        R, t = consensus_align([(P, G)], R, _centroid(gpoly) - R @ _centroid(ppoly), p)
        a = line_match(P.moved(R, t), G, p["offset_m"], p["angle_deg"])
        cands.append(dict(R=R, t=t, iou=iou(ppoly @ R.T + t, gpoly), frame=f"room-local {90 * k}",
                          doors=_door_agreement(a, ws, openings, gt_doors)))
    kb = max(range(4), key=lambda k: cands[k]["iou"] + p["door_bonus"] * cands[k]["doors"])
    best = cands[kb]
    turned = max(cands[(kb + 1) % 4]["iou"], cands[(kb + 3) % 4]["iou"])     # same room turned by 90 deg
    ambiguity = (f"90 deg turn fits about as well (IoU {best['iou']:.2f} vs {turned:.2f})"
                 if best["iou"] - turned < p["ambiguous_iou_gap"] else None)
    if glob is not None and glob["score"] >= p["global_min_iou"]:
        R, t = consensus_align([(P, G)], glob["R"], glob["t"], p, max_shift=p["global_max_shift_m"])
        g = dict(R=R, t=t, iou=iou(ppoly @ R.T + t, gpoly), frame="global, refined")
        if g["iou"] >= best["iou"] - p["prefer_global_iou"]:
            best, ambiguity = g, None            # the whole-plan fit fixes the orientation
    if best["iou"] < p["weak_iou"]:
        ambiguity = f"weak overlap (IoU {best['iou']:.2f} < {p['weak_iou']}): pairing by position is uncertain"
    best["ambiguity"] = ambiguity
    best["rot_deg"] = float(np.degrees(np.arctan2(best["R"][1, 0], best["R"][0, 0])))
    best.update(ws=ws, P=P, G=G)
    return best


def geometric_pairs(plan, room, gtg, names, mirror, glob, openings, p=PARAMS) -> dict:
    """GT wall index -> predicted wall indices on its line, plus the unmatched predicted walls and the alignment."""
    return pairs_on_lines(align_room(plan, room, gtg, names, mirror, glob, openings, p), p)


def pairs_on_lines(al: dict, p=PARAMS) -> dict:
    """Step 2 of the geometric pairing for a given alignment al (align_room or register_outline)."""
    Q = al["P"].moved(al["R"], al["t"])
    assign = line_match(Q, al["G"], p["offset_m"], p["angle_deg"])
    by_gt: dict[int, list[int]] = {}
    for i, (j, _) in enumerate(assign):
        if j >= 0:
            by_gt.setdefault(j, []).append(i)
    unmatched = [i for i, (j, _) in enumerate(assign) if j < 0]
    return dict(by_gt=by_gt, offsets={i: off for i, (j, off) in enumerate(assign) if j >= 0}, unmatched=unmatched,
                align=al)


# ------------------------------------------------------------------------- tape GT: the outline from the lengths

_DIRS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])        # E, N, W, S; a left turn adds 1


def outline_tolerance(gt_len, p=PARAMS) -> float:
    """How far a taped outline may miss closing: max(close_abs_m, close_rel * perimeter) = max(3 cm, 0.5%). Each wall
    is taped to 1-3 mm, but walls are not quite straight or square and the tape runs 1 m above the floor, so about
    5 mm per wall accumulates over the 2-6 walls of one axis."""
    return max(p["close_abs_m"], p["close_rel"] * float(np.sum(gt_len)))


def reconstruct_outlines(gt_len, names=None, p=PARAMS) -> dict:
    """Every right-angled outline the tape lengths W1..Wn allow (clockwise from above, W1 first).

    Consecutive walls are perpendicular and every corner turns right or left; going clockwise round a simple outline
    makes 4 more right turns than left ones, so the (n - 4) / 2 left corners are enumerated (n = 12: 495 patterns).
    W1 runs east from the origin: that fixes the rotation, and there is no mirror (clockwise naming fixes the
    handedness). A pattern is kept when its outline closes within outline_tolerance (the misclosure is then spread
    over the walls of each axis in proportion to their length, so the polygon closes exactly) and does not cross
    itself. Returns dict(status, shapes, tol, best_misclosure, reason): status 'reconstructed' (one outline),
    'ambiguous' (several; e.g. a bump whose two sides are equal can equally be a notch) or 'unreconstructable' (none;
    reason says why); shapes: [dict(poly (n, 2) with vertex k = the start of W(k+1), misclosure m, left = indices of
    the walls after which the outline turns left, area m2)]."""
    gl = np.asarray(gt_len, float)
    n = len(gl)
    out = dict(status="unreconstructable", shapes=[], tol=None, best_misclosure=None, reason=None)
    if names is not None and [_wnum(x) for x in names] != list(range(1, n + 1)):
        out["reason"] = f"the walls are not W1..W{n} without gaps ({', '.join(names)})"
        return out
    if n < 4 or n % 2:
        out["reason"] = (f"{n} walls: an outline with right-angled corners alternates two axes, so it has an even "
                         f"number of walls, at least 4 (a wall taped in two pieces, or a wall not at a right angle?)")
        return out
    if n > p["max_outline_walls"]:
        out["reason"] = f"{n} walls: more than {p['max_outline_walls']}, too many turn patterns to enumerate"
        return out
    if np.any(gl <= 0):
        out["reason"] = "a wall length is not positive"
        return out
    from shapely.geometry import LinearRing
    tol = out["tol"] = outline_tolerance(gl, p)
    combos = list(itertools.combinations(range(n), (n - 4) // 2))
    left = np.array(combos, int).reshape(len(combos), (n - 4) // 2)
    turn = -np.ones((len(left), n), int)
    np.put_along_axis(turn, left, 1, axis=1)
    d = np.concatenate([np.zeros((len(left), 1), int), np.cumsum(turn[:, :-1], axis=1)], axis=1) % 4
    disp = gl[None, :, None] * _DIRS[d]                                   # (patterns, walls, 2)
    e = disp.sum(axis=1)
    err = np.hypot(e[:, 0], e[:, 1])
    out["best_misclosure"] = float(err.min())
    for c in np.flatnonzero(err <= tol):
        adj = disp[c].copy()
        for ax in (0, 1):                                                 # spread the misclosure (compass rule)
            on = _DIRS[d[c]][:, ax] != 0
            adj[on, ax] -= e[c, ax] * gl[on] / gl[on].sum()
        V = np.vstack([[0.0, 0.0], np.cumsum(adj, axis=0)[:-1]])
        if signed_area(V) >= 0 or not LinearRing(V).is_simple:
            continue
        out["shapes"].append(dict(poly=V, misclosure=float(err[c]), left=[int(i) for i in left[c]],
                                  area=float(-signed_area(V))))
    k = len(out["shapes"])
    out["status"] = "reconstructed" if k == 1 else "ambiguous" if k > 1 else "unreconstructable"
    if not k:
        out["reason"] = (f"no right-angled clockwise outline closes within {100 * tol:.1f} cm (best misclosure "
                         f"{100 * err.min():.1f} cm): a tape slip, a wall not at a right angle or a missing wall"
                         if err.min() > tol else "every outline that closes crosses itself")
    return out


def outline_geometry(poly, names: list[str], door_walls=("W1",)) -> dict:
    """A reconstructed outline in the form of read_gt_geometry: {"poly", "walls": {name: {a, b, door}}}; W1 holds the
    main door by the naming rule, other door walls only when the GT file says so."""
    V = np.asarray(poly, float)
    n = len(V)
    return dict(poly=V, walls={nm: dict(a=V[k], b=V[(k + 1) % n], door=nm in door_walls) for k, nm in enumerate(names)})


def rigid_fit_error(A, B) -> float:
    """Largest distance between corresponding points after the best rigid fit of A onto B (rotation and translation,
    no mirror; least squares)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    a, b = A - A.mean(axis=0), B - B.mean(axis=0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = (U @ np.diag([1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    return float(np.max(np.linalg.norm(a @ R.T - b, axis=1)))


def _raster(poly, h: float):
    """Occupancy (cell centre inside) of the h-sized cells over the polygon's bounding box, and the box corner."""
    import shapely
    g = _polygon(poly)
    x0, y0, x1, y1 = g.bounds
    xs = x0 + h * (np.arange(int(np.ceil((x1 - x0) / h)) + 1) + 0.5)
    ys = y0 + h * (np.arange(int(np.ceil((y1 - y0) / h)) + 1) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    return shapely.contains_xy(g, X, Y).astype(float), np.array([x0, y0])


def iou_translation(Q, G, t_ref, h: float = 0.02) -> tuple[np.ndarray, float]:
    """The translation t of polygon Q that maximises IoU(Q + t, G), and that IoU.

    IoU grows with the intersection area, which is evaluated for every t on an h grid at once (cross-correlation of
    the two rasterised polygons by FFT), then refined on exact areas (shapely; pattern search down to 1 mm). Equal
    maxima (a room that fits inside the other slides freely) -> the one closest to t_ref (centroids aligned)."""
    from scipy.signal import fftconvolve
    from shapely import affinity
    mq, q0 = _raster(Q, h)
    mg, g0 = _raster(G, h)
    C = np.rint(fftconvolve(mg, mq[::-1, ::-1], mode="full"))
    iy, ix = np.nonzero(C >= C.max() - 0.5)
    T = g0 - q0 + h * np.stack([ix - (mq.shape[1] - 1), iy - (mq.shape[0] - 1)], axis=1)
    t = T[np.argmin(np.linalg.norm(T - np.asarray(t_ref, float), axis=1))].astype(float)
    pq, pg = _polygon(Q), _polygon(G)

    def f(x):
        m = affinity.translate(pq, float(x[0]), float(x[1]))
        u = m.union(pg).area
        return float(m.intersection(pg).area / u) if u > 0 else 0.0
    best, s = f(t), h
    while s >= 1e-3:
        cand = [t + s * np.array(v) for v in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))]
        vals = [f(x) for x in cand]
        if max(vals) > best + 1e-9:
            best, t = max(vals), cand[int(np.argmax(vals))]
        else:
            s /= 2
    return t, best


def register_outline(plan, room, gtg, names, mirror, openings, p=PARAMS) -> tuple[list[dict], str]:
    """Rigid registration of a predicted room onto a GT outline reconstructed from tape lengths, by overlap only.

    For each of the 4 rotations in 90 deg steps (no mirror) from the rooms' dominant wall directions: the translation
    of highest IoU (iou_translation), then the refinement the geometric pairing applies to a whole-plan fit
    (consensus_align: per-axis shift <= global_max_shift_m that puts the most predicted wall length on a GT wall
    line, ICP <= 5 deg). The registration is chosen by that highest IoU; rotations within tie_iou of the best are
    told apart by the door rule only: the GT door walls (W1, the main-door wall, plus any the GT file names) holding a
    predicted door or passage. Wall lengths play no part. Returns (the registrations to pair with, a note): one,
    or every tied one the door rule cannot split (the caller scores each and keeps the worst)."""
    ws, P, ppoly = plan_room_segs(plan, room, mirror)
    G = gt_room_segs(gtg, names)
    gpoly = np.asarray(gtg["poly"], float)
    gt_doors = [gtg["walls"][n].get("door", False) for n in names]
    base = _dominant_angle(G) - _dominant_angle(P)
    cands = []
    for k in range(4):
        R = _rot(base + k * np.pi / 2)
        Q = ppoly @ R.T
        t, v = iou_translation(Q, gpoly, _centroid(gpoly) - _centroid(Q), p["raster_m"])
        R2, t2 = consensus_align([(P, G)], R, t, p, max_shift=p["global_max_shift_m"])
        a = line_match(P.moved(R2, t2), G, p["offset_m"], p["angle_deg"])
        cands.append(dict(R=R2, t=t2, iou_reg=v, iou=iou(ppoly @ R2.T + t2, gpoly), frame=f"outline {90 * k}",
                          doors=_door_agreement(a, ws, openings, gt_doors), ws=ws, P=P, G=G,
                          rot_deg=float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0])))))
    top = max(c["iou_reg"] for c in cands)
    near = sorted((c for c in cands if c["iou_reg"] >= top - p["tie_iou"]), key=lambda c: -c["iou_reg"])
    keep = [c for c in near if c["doors"] == max(x["doors"] for x in near)]
    dw = "+".join(n for n, d in zip(names, gt_doors) if d)
    if len(near) == 1:
        note = f"best IoU {top:.3f} (next rotation {max([c['iou_reg'] for c in cands if c is not near[0]]):.3f})"
    elif len(keep) == 1:
        note = (f"{len(near)} rotations within {p['tie_iou']} IoU of the best ({top:.3f}); door rule: {dw} holds a "
                f"predicted door in {keep[0]['frame']} only")
    else:
        note = (f"{len(near)} rotations within {p['tie_iou']} IoU of the best ({top:.3f}) and the door rule cannot "
                f"split {len(keep)} of them ({dw} holds a predicted door in "
                f"{'all' if keep[0]['doors'] else 'none'} of them)")
    for c in keep:
        c["ambiguity"] = (f"weak overlap (IoU {c['iou_reg']:.2f} < {p['weak_iou']}): pairing by position is uncertain"
                          if c["iou_reg"] < p["weak_iou"] else None)
    return keep, note


# ------------------------------------------------------------------------------------------ order-merged pairing

def _same_line(wa: dict, wb: dict, p=PARAMS) -> bool:
    """Same direction (+-angle_deg, same sense), offset <= offset_m, and touching along the line (gap <= short_m)."""
    a0, a1, b0, b1 = (np.asarray(x, float) for x in (wa["p0"], wa["p1"], wb["p0"], wb["p1"]))
    da, db = a1 - a0, b1 - b0
    la, lb = np.linalg.norm(da), np.linalg.norm(db)
    if la < 1e-9 or lb < 1e-9 or (da @ db) / (la * lb) < np.cos(np.radians(p["angle_deg"])):
        return False
    u = da / la
    if abs(np.array([-u[1], u[0]]) @ (0.5 * (b0 + b1) - a0)) > p["offset_m"]:
        return False
    s0, s1 = sorted((float((b0 - a0) @ u), float((b1 - a0) @ u)))
    return max(s0 - la, -s1) <= p["short_m"]                     # gap between the two along the line


def _run_value(ws: list[dict], walls: list[int], s0, s1, c, p=PARAMS) -> tuple[list[int], float, float]:
    """Consecutive predicted walls taken as one wall: the walls on the run's main line (the line carrying the most
    wall length: same direction, offset <= offset_m), their merged length, and the length of the run's walls off
    that line (jogs, stubs, the bottom of a notch)."""
    q = dict(p, short_m=np.inf)
    L = {i: _val(ws[i]["length"])[0] or 0.0 for i in walls}
    lines = [[j for j in walls if j == i or _same_line(ws[i], ws[j], q)] for i in walls]
    on = sorted(max(lines, key=lambda ln: sum(L[j] for j in ln)))
    return on, piece_length(ws, on, s0, s1, c)[0], sum(L[j] for j in walls if j not in on)


def _align_runs(gl: np.ndarray, pl: np.ndarray, run, p=PARAMS) -> tuple[float, dict]:
    """Monotone alignment of GT walls gl to predicted walls pl (already rotated to a start). A GT wall takes a run
    of 1..max_run consecutive predicted walls: cost |run length - GT| + skip_pred_per_m * (run length off its main
    line). A predicted wall may be left out (skip_pred_per_m * its length: a stub is cheap, a real wall is not); a
    GT wall is left unpaired at skip_gt_m + skip_gt_rel * its length (nothing within ~50% of it). run(a, r) ->
    _run_value of predicted walls a..a+r-1. Returns (cost, {gt index: (first predicted wall, run length)})."""
    n, m = len(gl), len(pl)
    D = np.full((n + 1, m + 1), np.inf)
    back = {}
    D[0, 0] = 0.0
    for j in range(1, m + 1):
        D[0, j] = D[0, j - 1] + p["skip_pred_per_m"] * pl[j - 1]
    for i in range(1, n + 1):
        for j in range(m + 1):
            best, arg = D[i - 1, j] + p["skip_gt_m"] + p["skip_gt_rel"] * gl[i - 1], ("skip_gt",)
            if j > 0 and D[i, j - 1] + p["skip_pred_per_m"] * pl[j - 1] < best:
                best, arg = D[i, j - 1] + p["skip_pred_per_m"] * pl[j - 1], ("skip_p",)
            for r in range(1, min(p["max_run"], j) + 1):
                _, length, off = run(j - r, r)
                v = D[i - 1, j - r] + abs(length - gl[i - 1]) + p["skip_pred_per_m"] * off
                if v < best - 1e-12:
                    best, arg = v, ("run", r)
            D[i, j], back[i, j] = best, arg
    pairs, i, j = {}, n, m
    while i > 0:
        a = back[i, j]
        if a[0] == "skip_p":
            j -= 1
        elif a[0] == "skip_gt":
            i -= 1
        else:
            pairs[i - 1] = (j - a[1], a[1])
            i, j = i - 1, j - a[1]
    return float(D[n, m]), pairs


def order_merged_pairs(room: dict, ws_cw: list[dict], gt_len: list[float], openings: dict, p=PARAMS) -> dict:
    """Robust order-based pairing for a tape GT (lengths only), on the predicted walls clockwise from above: a cyclic,
    monotone alignment to W1..Wn in which a GT wall may take a run of consecutive predicted walls (collinear pieces
    split by jogs or door reveals, or a wall the plan breaks with a notch the GT lacks), short predicted walls are
    left out cheaply, and a door wall is preferred as W1 and the documented clockwise order over the reverse.
    Whether to merge is decided by the GT lengths, not a fixed threshold: a 0.2 m niche the tape measured stays a
    wall. Returns {gt index: predicted wall indices}, the order used and the corner model."""
    s0, s1, c = corner_model(room, ws_cw)
    out = dict(by_gt={}, direction=None, start=None)
    m = len(ws_cw)
    if not m or not gt_len:
        return out
    gl = np.asarray(gt_len, float)
    pl = np.array([_val(w["length"])[0] or 0.0 for w in ws_cw])
    door = [_has_door(w, openings) for w in ws_cw]
    memo: dict[frozenset, tuple] = {}

    def run(seq, a, r):
        key = frozenset(seq[a:a + r])
        if key not in memo:
            memo[key] = _run_value(ws_cw, sorted(key), s0, s1, c, p)
        return memo[key]
    best = None
    for direction in (1, -1):
        order = list(range(m)) if direction == 1 else list(reversed(range(m)))
        for s in range(m):
            seq = [order[(s + k) % m] for k in range(m)]
            cost, pairs = _align_runs(gl, pl[seq], lambda a, r: run(seq, a, r), p)
            w1 = pairs.get(0)
            cost += 0.0 if (w1 and any(door[k] for k in seq[w1[0]:w1[0] + w1[1]])) else p["door_bonus"]
            cost += p["reverse_penalty_m"] if direction == -1 else 0.0
            if best is None or cost < best[0] - 1e-12:
                best = (cost, {ig: run(seq, a, r)[0] for ig, (a, r) in pairs.items()}, direction, s)
    out.update(by_gt=best[1], direction=best[2], start=best[3])
    return out


# ------------------------------------------------------------------------------------- anchored alignment (tape GT)

def _anchored_dp(gl, r, seq, runv, skip_p, p=PARAMS) -> tuple[float, list]:
    """One monotone alignment of the GT walls rotated to start at index r with the predicted walls seq. Steps: a run
    of k GT walls (k odd) with a run of b consecutive predicted walls, cost |length - span| / span + off-line
    predicted length + merge_gt * (k - 1), span = the GT walls at even offsets of the run (parallel to the predicted
    wall: in a right-angled room consecutive walls turn by 90 deg, so a jog between two collinear walls, or an
    alcove's sides, sit at the odd offsets); a missed GT wall (miss_gt); a skipped predicted wall (skip_p).
    Returns (cost, [(GT indices, predicted indices on the run's main line)])."""
    n, m = len(gl), len(seq)
    D = np.full((n + 1, m + 1), np.inf)
    back = {}
    D[0, 0] = 0.0

    def relax(i, j, v, arg):
        if v < D[i, j] - 1e-12:
            D[i, j], back[i, j] = v, arg
    for i in range(n + 1):
        for j in range(m + 1):
            d = D[i, j]
            if not np.isfinite(d):
                continue
            if j < m:
                relax(i, j + 1, d + skip_p[seq[j]], ("skip_p",))
            if i < n:
                relax(i + 1, j, d + p["miss_gt"], ("miss",))
            for k in range(1, min(p["max_gt_run"], n - i) + 1, 2):
                span = sum(gl[(r + i + t) % n] for t in range(0, k, 2))
                for b in range(1, min(p["max_run"], m - j) + 1):
                    _, length, off = runv(j, b)
                    relax(i + k, j + b, d + abs(length - span) / span + off + p["merge_gt"] * (k - 1),
                          ("match", k, b))
    steps, i, j = [], n, m
    while i or j:
        a = back[i, j]
        if a[0] == "skip_p":
            j -= 1
        elif a[0] == "miss":
            i -= 1
        else:
            k, b = a[1], a[2]
            steps.append(([(r + i - k + t) % n for t in range(k)], runv(j - b, b)[0]))
            i, j = i - k, j - b
    return float(D[n, m]), steps[::-1]


def anchored_pairs(room: dict, ws_cw: list[dict], gt_len: list[float], openings: dict, p=PARAMS) -> dict:
    """Pairing for a tape GT (lengths only, W1..Wn clockwise from the wall with the room's main door) by anchored
    cyclic sequence alignment of the predicted walls clockwise from above (_anchored_dp), tried from every predicted
    start and every GT rotation that keeps W1 in the first GT run.

    Both sides may merge: a GT wall may take a run of consecutive predicted pieces (jogs, door reveals), and a
    predicted wall may span a run of GT walls a box model cannot hold (a jog, an alcove: 'merged with W5+W6'). In a
    merged GT run the predicted wall is compared with the longest GT wall parallel to it (the wall it lies on, as the
    geometric pairing does); the other GT walls of the run are reported as merged (not measured, they count as
    missed). Costs are in relative-error units, so long and short rooms weigh alike.

    Choice, made from what the tape GT supports (order, W1 = door wall, lengths), never from the score: the best
    alignment whose W1 run holds a predicted door or passage wall ('anchored at the door'), unless the best
    alignment of any start is cheaper by more than anchor_margin ('best start'). A runner-up with a different
    pairing within ambiguous_margin of the chosen cost is reported as an ambiguity."""
    s0, s1, c = corner_model(room, ws_cw)
    out = dict(by_gt={}, gt_runs={}, how=None, anchored=None, start=None, cost=None, ambiguity=None,
               door_walls=[], reverse_note=None)
    m, n = len(ws_cw), len(gt_len)
    if not m or not n:
        return out
    gl = [float(x) for x in gt_len]
    gnorm = float(np.median(gl)) or 1.0
    L = [_val(w["length"])[0] or 0.0 for w in ws_cw]
    door = [_has_door(w, openings) for w in ws_cw]
    out["door_walls"] = [ws_cw[i]["id"] for i in range(m) if door[i]]
    skip_p = [p["skip_pred_rel"] * x / gnorm for x in L]
    memo: dict[frozenset, tuple] = {}

    def make_runv(seq):
        def runv(a, b):
            key = frozenset(seq[a:a + b])
            if key not in memo:
                on, length, off = _run_value(ws_cw, seq[a:a + b], s0, s1, c, p)
                memo[key] = (on, length, p["skip_pred_rel"] * off / gnorm)
            return memo[key]
        return runv

    def align(order):
        res = []
        for s in range(m):
            if L[order[s]] < p["short_m"] and any(x >= p["short_m"] for x in L):
                continue                          # a run never needs to start at a stub (it is skipped at the end)
            seq = [order[(s + t) % m] for t in range(m)]
            runv = make_runv(seq)
            for r in sorted({0} | {(n - k) % n for k in range(1, min(p["max_gt_run"], n))}):
                cost, steps = _anchored_dp(gl, r, seq, runv, skip_p, p)
                w1 = next((pi for gi, pi in steps if 0 in gi), [])
                res.append(dict(cost=cost, steps=steps, start=seq[0], anchored=any(door[i] for i in w1),
                                w1=[ws_cw[i]["id"] for i in w1]))
        return res
    cands = align(list(range(m)))

    def signature(cd):                            # the reported pairing: GT wall compared -> predicted walls
        return tuple(sorted((max(gi[0::2], key=lambda g: gl[g]), tuple(sorted(pi))) for gi, pi in cd["steps"]))
    uniq = {}
    for cd in sorted(cands, key=lambda x: x["cost"]):
        uniq.setdefault(signature(cd), cd)
    cands = list(uniq.values())
    best_any = cands[0]
    anch = [cd for cd in cands if cd["anchored"]]
    if anch and anch[0]["cost"] <= best_any["cost"] + p["anchor_margin"]:
        best = anch[0]
        how = f"anchored at the door (W1 on {'+'.join(best['w1'])})"
    elif anch:
        best = best_any
        how = (f"best start (W1 on {'+'.join(best['w1']) or 'nothing'}; the door-anchored start costs "
               f"{anch[0]['cost'] - best_any['cost']:.2f} more)")
    else:
        best = best_any
        how = f"best start (no door wall in the plan's room; W1 on {'+'.join(best['w1']) or 'nothing'})"
    by_gt, gt_runs = {}, {}
    for gi, pi in best["steps"]:
        main = max(gi[0::2], key=lambda g: gl[g])
        by_gt[main] = list(pi)
        if len(gi) > 1:
            gt_runs[main] = gi
    pool = anch if best["anchored"] else cands          # the door anchor settles ties with unanchored starts
    rest = [cd for cd in pool if cd is not best and cd["cost"] <= best["cost"] + p["ambiguous_margin"]]
    amb = None
    if rest:
        alt = rest[0]
        alt_by = {max(gi[0::2], key=lambda g: gl[g]): pi for gi, pi in alt["steps"]}

        def vals(bg):
            return {g: round(piece_length(ws_cw, sorted(pi), s0, s1, c)[0], 3) for g, pi in bg.items()}
        same = vals(alt_by) == vals(by_gt)
        amb = (f"{len(rest)} other pairing(s) within {p['ambiguous_margin']:.2f} of the chosen cost; next: W1 on "
               f"{'+'.join(alt['w1']) or 'nothing'} (cost {alt['cost']:.3f} vs {best['cost']:.3f})"
               + ("; same lengths per GT wall (symmetric room), the score is unchanged" if same
                  else "; DIFFERENT lengths: the score depends on this choice"))
    rev = align(list(reversed(range(m))))
    if rev and min(x["cost"] for x in rev) < best["cost"] - p["anchor_margin"]:
        out["reverse_note"] = (f"the counter-clockwise order fits better (cost {min(x['cost'] for x in rev):.3f} vs "
                               f"{best['cost']:.3f}): check that the sketch's walls go clockwise")
    out.update(by_gt=by_gt, gt_runs=gt_runs, how=how, anchored=best["anchored"], start=ws_cw[best["start"]]["id"],
               cost=round(best["cost"], 4), ambiguity=amb)
    return out


# ---------------------------------------------------------------- constrained order (tape GT, no outline fits)

def _wall_axes(ws: list[dict]) -> list[int | None]:
    """Axis of each wall in the room's own Manhattan frame: 0 along the dominant wall direction, 1 across it, None
    more than 20 deg off both (or no length). Mirroring the plan does not change it."""
    S = Segs([w["p0"] for w in ws], [w["p1"] for w in ws], 1.0)
    c = np.abs(np.cos(np.arctan2(S.d[:, 1], S.d[:, 0]) - _dominant_angle(S)))
    lo, hi = np.sin(np.radians(20.0)), np.cos(np.radians(20.0))
    return [None if ln < 1e-6 else 0 if x >= hi else 1 if x <= lo else None for ln, x in zip(S.L, c)]


def _constrained_dp(g_rest: list[int], gl, axis_g: dict, seq: list[int], runv, skip_p, p=PARAMS) -> tuple[float, list]:
    """_anchored_dp on the GT walls g_rest (indices, in order, no wrap) and the predicted walls seq, with the axis as
    a hard constraint: a predicted run takes a GT run (k odd) only when its main line lies on the axis of the GT
    walls at the run's even offsets (axis_g: GT index -> axis). runv(a, b) -> (main-line walls, length, off-line
    cost, axis) of seq[a:a+b]. Returns (cost, [(GT indices, predicted main-line walls)])."""
    n, m = len(g_rest), len(seq)
    D = np.full((n + 1, m + 1), np.inf)
    back = {}
    D[0, 0] = 0.0

    def relax(i, j, v, arg):
        if v < D[i, j] - 1e-12:
            D[i, j], back[i, j] = v, arg
    for i in range(n + 1):
        for j in range(m + 1):
            d = D[i, j]
            if not np.isfinite(d):
                continue
            if j < m:
                relax(i, j + 1, d + skip_p[seq[j]], ("skip_p",))
            if i < n:
                relax(i + 1, j, d + p["miss_gt"], ("miss",))
            for k in range(1, min(p["max_gt_run"], n - i) + 1, 2):
                span = sum(gl[g_rest[i + t]] for t in range(0, k, 2))
                for b in range(1, min(p["max_run"], m - j) + 1):
                    _, length, off, ax = runv(j, b)
                    if ax == axis_g[g_rest[i]]:
                        relax(i + k, j + b, d + abs(length - span) / span + off + p["merge_gt"] * (k - 1),
                              ("match", k, b))
    steps, i, j = [], n, m
    while i or j:
        a = back[i, j]
        if a[0] == "skip_p":
            j -= 1
        elif a[0] == "miss":
            i -= 1
        else:
            k, b = a[1], a[2]
            steps.append((g_rest[i - k:i], runv(j - b, b)[0]))
            i, j = i - k, j - b
    return float(D[n, m]), steps[::-1]


def constrained_pairs(room: dict, ws_cw: list[dict], gt_len: list[float], openings: dict, p=PARAMS) -> dict:
    """Fallback pairing for a tape GT room whose outline cannot be rebuilt (reconstruct_outlines found none): an order
    alignment of the predicted walls clockwise from above with W1..Wn (costs as _anchored_dp) under HARD constraints:
      * axis: GT walls alternate two perpendicular axes, so the predicted run paired with W(k) must lie on W(k)'s
        axis of the room; both assignments of W1 to the room's two axes are tried;
      * door: when the plan's room has a door or passage wall, W1's run must hold one on its main line (every such
        run is tried as W1's, the other walls are aligned after it). Without one, W1 is free: WARNING.
    The cheapest alignment is kept. A runner-up with different lengths within fallback_margin of its cost is a
    WARNING (the score depends on a choice the tape cannot make). Returns dict(by_gt, gt_runs, how, cost, anchored,
    axis, door_walls, warnings)."""
    s0, s1, c = corner_model(room, ws_cw)
    out = dict(by_gt={}, gt_runs={}, how=None, cost=None, anchored=None, axis=None, door_walls=[], warnings=[])
    m, n = len(ws_cw), len(gt_len)
    if not m or not n:
        return out
    gl = [float(x) for x in gt_len]
    gnorm = float(np.median(gl)) or 1.0
    L = [_val(w["length"])[0] or 0.0 for w in ws_cw]
    door = [_has_door(w, openings) for w in ws_cw]
    out["door_walls"] = [ws_cw[i]["id"] for i in range(m) if door[i]]
    skip_p = [p["skip_pred_rel"] * x / gnorm for x in L]
    axes = _wall_axes(ws_cw)
    memo: dict[frozenset, tuple] = {}

    def run_of(walls):
        key = frozenset(walls)
        if key not in memo:
            on, length, off = _run_value(ws_cw, list(walls), s0, s1, c, p)
            memo[key] = (on, length, p["skip_pred_rel"] * off / gnorm, axes[max(on, key=lambda i: L[i])])
        return memo[key]

    def runv_for(seq):
        return lambda a, b: run_of(seq[a:a + b])
    cands = []
    for a0 in (0, 1):
        axis_g = {g: (a0 + g) % 2 for g in range(n)}
        for a in range(m):
            for b in range(1, min(p["max_run"], m) + 1):
                run = [(a + t) % m for t in range(b)]
                on, length, off, ax = run_of(run)
                # W1's run: on W1's axis, a door on its main line, starting and ending on that line (an off-line
                # piece at either end is skipped as cheaply after the run)
                if ax != a0 or not any(door[i] for i in on) or run[0] not in on or run[-1] not in on:
                    continue
                seq = [(a + b + t) % m for t in range(m - b)]
                for k in range(1, min(p["max_gt_run"], n) + 1, 2):
                    for o in range(0, k, 2):                 # W1 at an even offset of the GT run: on its main line
                        grun = [(t - o) % n for t in range(k)]
                        span = sum(gl[g] for g in grun[0::2])
                        cost, steps = _constrained_dp([(k - o + t) % n for t in range(n - k)], gl, axis_g, seq,
                                                      runv_for(seq), skip_p, p)
                        cands.append(dict(cost=cost + abs(length - span) / span + off + p["merge_gt"] * (k - 1),
                                          steps=[(grun, on)] + steps, axis=a0, anchored=True))
    if not cands:
        out["warnings"].append(
            "W1 is not anchored: " + ("the plan's room has no door or passage wall" if not any(door) else
                                      f"no run holding the door wall(s) {'+'.join(out['door_walls'])} lies on an axis"))
        for a0 in (0, 1):
            axis_g = {g: (a0 + g) % 2 for g in range(n)}
            for s in range(m):
                if L[s] < p["short_m"] and any(x >= p["short_m"] for x in L):
                    continue                          # a run never needs to start at a stub
                seq = [(s + t) % m for t in range(m)]
                cost, steps = _constrained_dp(list(range(n)), gl, axis_g, seq, runv_for(seq), skip_p, p)
                cands.append(dict(cost=cost, steps=steps, axis=a0, anchored=False))

    def by_gt_of(cd):
        bg, runs = {}, {}
        for gi, pi in cd["steps"]:
            main = max(gi[0::2], key=lambda g: gl[g])
            bg[main] = list(pi)
            if len(gi) > 1:
                runs[main] = list(gi)
        return bg, runs

    def vals(bg):
        return {g: round(piece_length(ws_cw, sorted(pi), s0, s1, c)[0], 3) for g, pi in bg.items()}
    cands.sort(key=lambda x: x["cost"])
    best = cands[0]
    bg, runs = by_gt_of(best)
    v0 = vals(bg)
    rivals = [cd for cd in cands[1:] if cd["cost"] <= best["cost"] + p["fallback_margin"]
              and vals(by_gt_of(cd)[0]) != v0]
    w1 = next((pi for gi, pi in best["steps"] if 0 in gi), [])
    if rivals:
        r = rivals[0]
        rw1 = next((pi for gi, pi in r["steps"] if 0 in gi), [])
        out["warnings"].append(
            f"{len(rivals)} pairing(s) with DIFFERENT lengths within {p['fallback_margin']} of the chosen cost; next: "
            f"W1 on {'+'.join(ws_cw[i]['id'] for i in rw1) or 'nothing'} (cost {r['cost']:.3f} vs "
            f"{best['cost']:.3f}): the score depends on this choice")
    how = (f"constrained order (W1 {'at the door, ' if best['anchored'] else ''}on "
           f"{'+'.join(ws_cw[i]['id'] for i in w1) or 'nothing'}; W1 on the room's {'first' if best['axis'] == 0 else 'second'} "
           f"axis; cost {best['cost']:.3f})")
    out.update(by_gt=bg, gt_runs=runs, how=how, cost=round(best["cost"], 4), anchored=best["anchored"], axis=best["axis"])
    return out


def _wnum(name: str) -> int:
    try:
        return int(str(name).lstrip("Ww"))
    except ValueError:
        return 0
