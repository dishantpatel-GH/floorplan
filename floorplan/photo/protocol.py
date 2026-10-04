"""Photo protocol v2 (decision D-017): spin groups, doorway pairs, and layout registration of a doorway photo.

The revised protocol gives two kinds of structure that feature matching alone does not exploit:

* SPIN GROUP. A room's non-doorway photos are taken turning on the spot near the middle, ~60 deg apart. So:
  (1) they share (almost) one camera centre (body sway + arm: prior sigma ~0.3 m);
  (2) their headings differ by about one step, in one turning direction, in capture-time order;
  (3) each photo's own wall normals give its yaw modulo 90 deg (Manhattan), measured, not assumed per step.
  (2)+(3) resolve the 90-deg ambiguity: the yaw step is the Manhattan-consistent value nearest to +-60 deg.
  With a 69 deg lens and 60 deg steps adjacent photos overlap only ~10%, so feature matches are often absent;
  these priors still give a panorama-like reconstruction around a common centre with per-photo metric depth.
  Feature edges, when present, are kept and the loop check removes a prior that contradicts them.
* DOORWAY PAIR. Two photos in DIFFERENT room folders whose EXIF capture times are seconds apart were taken from the
  same spot (the doorway), looking in roughly opposite directions. That is a shared-centre link (sigma ~0.2 m)
  with a relative yaw of ~180 deg snapped to the Manhattan axes of the two photos. It links rooms without any
  feature match.
* LAYOUT REGISTRATION (fallback inside a room). A doorway photo that shares no features with its room's spin group
  is placed by aligning its wall points to the spin group's walls (top view, Manhattan yaw hypotheses, FFT
  correlation over translation), accepted only if unambiguous and free-space consistent.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from floorplan.photo.geometry import manhattan_yaw
from floorplan.photo.link import Edge, _ry, free_space_violation


# ------------------------------------------------------------------------------------------------ doorway pairs
def shot_rhythm(photos) -> float | None:
    """Median time between consecutive photos of the SAME folder (the photographer's shooting rhythm)."""
    timed = sorted((p for p in photos if p.time_s is not None), key=lambda p: p.time_s)
    d = [y.time_s - x.time_s for x, y in zip(timed, timed[1:]) if x.room == y.room and y.time_s - x.time_s < 60]
    return float(np.median(d)) if d else None


def doorway_pairs(photos, max_dt_s: float, rhythm_factor: float = 2.0) -> list[dict]:
    """Pairs of photos in different folders, CONSECUTIVE in capture time and taken quickly one after the other.

    Why "consecutive": a photographer finishing a spin in room A and walking to the door produces a spin photo of A
    within 15 s of the doorway photo into B. Only photos with no other photo between them in time are a pair; each
    photo joins at most one pair (closest first).
    Why the rhythm limit: walking from A's last spin shot to B's first spin shot takes ~10-12 s, which a flat
    15 s limit accepts as a "pair" (measured on the simulated schedule). A doorway pair is shot at the same rhythm
    as the spin shots (turn, shoot), so the limit is min(max_dt_s, rhythm_factor x median same-room interval)."""
    timed = sorted((p for p in photos if p.time_s is not None), key=lambda p: (p.time_s, p.name))
    rhythm = shot_rhythm(photos)
    thr = max_dt_s if rhythm is None else min(max_dt_s, max(4.0, rhythm_factor * rhythm))
    cand = []
    for x, y in zip(timed, timed[1:]):
        dt = y.time_s - x.time_s
        if x.room != y.room and dt <= thr:
            cand.append((dt, x, y))
    used, out = set(), []
    for dt, x, y in sorted(cand, key=lambda c: c[0]):
        if x.name in used or y.name in used:
            continue
        used |= {x.name, y.name}
        out.append(dict(a=x.name, b=y.name, rooms=[x.room, y.room], dt_s=round(float(dt), 2),
                        limit_s=round(float(thr), 1)))
    return out


def spin_groups(photos, pairs: list[dict]) -> dict[str, list[str] | None]:
    """Room -> its non-doorway photos in capture-time order; None if the room has no capture times (no order,
    and no way to tell spin shots from doorway shots: no spin prior then, by design)."""
    in_pair = {n for p in pairs for n in (p["a"], p["b"])}
    out = {}
    for room in sorted({p.room for p in photos}):
        ph = [p for p in photos if p.room == room]
        if any(p.time_s is None for p in ph):
            out[room] = None
            continue
        out[room] = [p.name for p in sorted(ph, key=lambda p: (p.time_s, p.name)) if p.name not in in_pair]
    return out


def photo_pitch_deg(view) -> float:
    """Camera pitch in degrees (+ = looking up), from the photo's levelling rotation (camera -> +y-up frame)."""
    f = view.R_lev @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arcsin(np.clip(f[1], -1.0, 1.0))))


# ---------------------------------------------------------------------------------------- per-photo Manhattan
def photo_manhattan(view, stride: int = 4, min_pts: int = 300) -> float | None:
    """Dominant wall direction (radians, modulo 90 deg) of one photo in its LEVELLED frame, from MoGe-2 normals.

    A pose yaw theta maps a levelled direction at angle a (atan2(z, x)) to a - theta, so for a photo whose walls are
    on the room axes theta = m (mod 90 deg). None when the photo sees too little wall."""
    if view.normal_cam is None:
        return None
    _, idx = view.points_cam(stride)
    N = view.normal_cam.reshape(-1, 3)[idx] @ view.R_lev.T
    walls = N[np.abs(N[:, 1]) < 0.2]
    if len(walls) < min_pts:
        return None
    return manhattan_yaw(walls, min_pts=min_pts)


def _wrap(a: float) -> float:
    return float(np.angle(np.exp(1j * a)))


def snap_step(m_a: float | None, m_b: float | None, expected: float) -> tuple[float, float, bool]:
    """Yaw step b-a: the Manhattan-consistent value (m_b - m_a + k*90deg) nearest to `expected`.
    Returns (step, |residual to expected|, measured?) — unmeasured steps fall back to the expectation."""
    if m_a is None or m_b is None:
        return expected, 0.0, False
    base = m_b - m_a
    k = np.round((expected - base) / (np.pi / 2))
    step = base + k * np.pi / 2
    return float(step), abs(_wrap(step - expected)), True


def spin_edges(group: list[str], manh: dict, feat_edges: list[Edge], p) -> tuple[list[Edge], dict]:
    """Prior edges along a spin group (time order). The turning direction (sign) is chosen by the Manhattan
    residuals and by any feature edge inside the group, with a margin in favour of the protocol's direction
    (clockwise): at ~45 deg steps the two directions are indistinguishable modulo 90 deg (measured: R1 of
    floor_only/rotate chose counter-clockwise by 9 deg of cost and was 89 deg wrong)."""
    step = np.radians(p.spin_step_deg)
    inside = {(e.a, e.b): e for e in feat_edges if e.a in group and e.b in group}
    best, costs = None, {}
    for sign in (+1, -1):          # both are scored (reported); the protocol's direction is used unless "auto"
        steps = [snap_step(manh.get(a), manh.get(b), -sign * step) for a, b in zip(group, group[1:])]
        cost = sum(r for _, r, ok in steps if ok)
        th = np.concatenate([[0.0], np.cumsum([s for s, _, _ in steps])])
        pos = {n: th[i] for i, n in enumerate(group)}
        for (a, b), e in inside.items():                 # a feature edge measures theta_b - theta_a directly
            cost += abs(_wrap(pos[b] - pos[a] - e.yaw))
        costs[sign] = cost + (0.0 if sign == p.spin_preferred_sign else np.radians(p.spin_direction_margin_deg))
        if best is None or costs[sign] < best[0]:
            best = (costs[sign], sign, steps)
    cost, sign, steps = best
    if p.spin_direction in ("clockwise", "counter-clockwise"):
        sign = 1 if p.spin_direction == "clockwise" else -1
        steps = [snap_step(manh.get(a), manh.get(b), -sign * step) for a, b in zip(group, group[1:])]
        cost = costs[sign]
    edges = []
    for (a, b), (st, r, ok) in zip(zip(group, group[1:]), steps):
        sig_yaw = p.spin_sigma_yaw_deg if ok and np.degrees(r) < 35 else p.spin_sigma_yaw_unmeasured_deg
        edges.append(Edge(a, b, st, np.zeros(3), 0.0, 0, 0, 0.0, False, kind="spin_prior", sig_t=p.spin_sigma_t_m,
                          sig_yaw_deg=sig_yaw, sig_s=p.prior_sigma_log_s,
                          note=f"step {np.degrees(st):+.0f} deg, resid {np.degrees(r):.0f} deg, measured={ok}"))
    info = dict(photos=group, direction="clockwise" if sign > 0 else "counter-clockwise",
                steps_deg=[round(float(np.degrees(s)), 1) for s, _, _ in steps],
                resid_deg=[round(float(np.degrees(r)), 1) for _, r, _ in steps],
                measured=[bool(ok) for _, _, ok in steps], cost_deg=round(float(np.degrees(cost)), 1),
                other_direction_cost_deg=round(float(np.degrees(costs[-sign])), 1), mode=p.spin_direction,
                direction_warning=bool(costs[-sign] + np.radians(10) < costs[sign]))
    return edges, info


def pair_edge_prior(pair: dict, manh: dict, p) -> tuple[Edge, dict]:
    """Doorway pair: shared centre (sigma ~0.2 m), relative yaw ~180 deg snapped to the two photos' Manhattan axes."""
    a, b = pair["a"], pair["b"]
    st, r, ok = snap_step(manh.get(a), manh.get(b), np.pi)
    sig_yaw = p.pair_sigma_yaw_deg if ok and np.degrees(r) < 35 else p.pair_sigma_yaw_unmeasured_deg
    e = Edge(a, b, st, np.zeros(3), 0.0, 0, 0, 0.0, True, kind="doorway_pair", sig_t=p.pair_sigma_t_m,
             sig_yaw_deg=sig_yaw, sig_s=p.prior_sigma_log_s,
             note=f"dt {pair['dt_s']} s, yaw {np.degrees(st):+.0f} deg (resid to 180: {np.degrees(r):.0f}), measured={ok}")
    return e, dict(pair, yaw_deg=round(float(np.degrees(st)), 1), resid_deg=round(float(np.degrees(r)), 1),
                   manhattan_measured=ok)


# ------------------------------------------------------------------------------------ layout registration
def _wall_xz(view, pose, stride: int = 3) -> np.ndarray:
    """Top-view (x, z) of a photo's wall points (horizontal normals, between 1.0 m below and 0.5 m above the camera)."""
    th, t, ls = pose
    Pc, idx = view.points_cam(stride)
    R = _ry(th) @ view.R_lev
    P = np.exp(ls) * Pc @ R.T + t
    N = view.normal_cam.reshape(-1, 3)[idx] @ R.T
    keep = (np.abs(N[:, 1]) < 0.3) & (P[:, 1] - t[1] > -1.0) & (P[:, 1] - t[1] < 0.5)
    return P[keep][:, [0, 2]]


def register_to_room(view_d, spin_poses: dict, views: dict, m_d: float | None, m_room: float, p,
                     top_k: int = 0) -> dict | None:
    """Place doorway photo `view_d` in the frame of its room's spin group by wall alignment (top view).

    Hypotheses: yaw = m_d - m_room + k*90 deg (4), scale correction in +-8%; translation by FFT correlation of the
    photo's wall points against a blurred wall map of the spin group. The camera must land inside the spin group's
    wall extent. Accepted only if the best score is high, clearly better than the best hypothesis elsewhere
    (symmetric rooms and corridors are ambiguous: then nothing is returned) and the dense free-space check passes
    against every spin photo."""
    if m_d is None:
        return None
    res = p.layout_cell_m
    W = np.concatenate([_wall_xz(views[n], ps) for n, ps in spin_poses.items()])
    if len(W) < 500:
        return None
    lo = W.min(0) - 3.0
    shape = np.ceil((W.max(0) + 3.0 - lo) / res).astype(int) + 1
    occ = np.zeros(shape, bool)
    ij = np.floor((W - lo) / res).astype(int)
    occ[ij[:, 0], ij[:, 1]] = True
    dist_wall = distance_transform_edt(~occ) * res
    S = np.exp(-0.5 * (dist_wall / p.layout_sigma_m) ** 2)
    inner_lo, inner_hi = W.min(0), W.max(0)
    hyps = []
    for k in range(4):
        th = m_d - m_room + k * np.pi / 2
        for sc in (0.92, 0.96, 1.0, 1.04, 1.08):
            Q = _wall_xz(view_d, (th, np.zeros(3), np.log(sc)))
            if len(Q) < 200:
                return None
            q = np.floor(Q / res).astype(int)
            q -= q.min(0)
            if np.any(q.max(0) + 1 >= S.shape):
                continue
            src = np.zeros(q.max(0) + 1)
            np.add.at(src, (q[:, 0], q[:, 1]), 1.0)
            corr = fftconvolve(S, src[::-1, ::-1], mode="valid") / len(Q)
            # camera (origin) position for each placement: origin maps to lo + (offset - qmin) * res
            qmin = np.floor(Q / res).astype(int).min(0)
            cam = lo[None, None, :] + (np.stack(np.meshgrid(np.arange(corr.shape[0]), np.arange(corr.shape[1]),
                                                             indexing="ij"), -1) - qmin) * res
            ok = np.all((cam > inner_lo - 0.3) & (cam < inner_hi + 0.3), axis=-1)
            # a doorway photo is taken IN a doorway: the camera sits on the room's wall line (within 0.6 m of a
            # wall point of the spin group), not in the open room (sliding hypotheses are removed)
            ci = np.clip(np.floor((cam - lo) / res).astype(int), 0, np.array(shape) - 1)
            ok &= dist_wall[ci[..., 0], ci[..., 1]] < p.layout_door_max_wall_dist_m
            corr = np.where(ok, corr, 0.0)
            i, j = np.unravel_index(np.argmax(corr), corr.shape)
            hyps.append(dict(score=float(corr[i, j]), yaw=float(th), scale=sc, xz=cam[i, j].copy(), k=k, corr=corr,
                             cam=cam))
    if not hyps:
        return None
    hyps.sort(key=lambda h: -h["score"])
    best = hyps[0]
    if top_k:                                    # distinct candidate placements, for the pair bridge
        y = float(np.median([ps[1][1] for ps in spin_poses.values()]))
        out = []
        for h in hyps:
            corr = h["corr"].copy()
            for _ in range(3):                   # up to 3 local maxima per (yaw, scale), 0.5 m apart
                i, j = np.unravel_index(np.argmax(corr), corr.shape)
                if corr[i, j] < p.layout_min_score:
                    break
                xz = h["cam"][i, j].copy()
                if all(np.linalg.norm(xz - o["xz"]) > 0.5 or abs(o["yaw"] - h["yaw"]) > 0.1 for o in out):
                    out.append(dict(score=float(corr[i, j]), yaw=h["yaw"], scale=h["scale"], xz=xz,
                                    pose=(h["yaw"], np.array([xz[0], y, xz[1]]), float(np.log(h["scale"])))))
                corr[np.linalg.norm(h["cam"] - xz, axis=-1) < 0.5] = 0
        return dict(candidates=sorted(out, key=lambda o: -o["score"])[:top_k])
    # runner-up: best score of any hypothesis whose camera is > 0.5 m away or whose yaw differs (other k)
    rival = 0.0
    for h in hyps:
        far = np.linalg.norm(h["cam"] - best["xz"], axis=-1) > 0.5
        sc = float(h["corr"][far].max()) if far.any() else 0.0
        if h["k"] != best["k"]:
            sc = max(sc, h["score"])
        rival = max(rival, sc)
    out = dict(score=round(best["score"], 3), rival=round(rival, 3), yaw_deg=round(float(np.degrees(best["yaw"])), 1),
               scale=best["scale"], xz=best["xz"].round(3).tolist())
    if best["score"] < p.layout_min_score or rival > p.layout_max_rival_ratio * best["score"]:
        out["accepted"] = False
        out["reason"] = "low score" if best["score"] < p.layout_min_score else "ambiguous (rival hypothesis)"
        return out
    y = float(np.median([ps[1][1] for ps in spin_poses.values()]))      # same photographer: same camera height
    pose_d = (best["yaw"], np.array([best["xz"][0], y, best["xz"][1]]), float(np.log(best["scale"])))
    viol = 0.0
    for n, ps in spin_poses.items():
        th, s, t = _relative(ps, pose_d)
        v1 = free_space_violation(views[n], view_d, th, s, t)[0]
        th2, s2, t2 = _relative(pose_d, ps)
        v2 = free_space_violation(view_d, views[n], th2, s2, t2)[0]
        viol = max(viol, v1, v2)
    out["violation"] = round(viol, 3)
    out["accepted"] = viol <= p.edge_max_violation
    if not out["accepted"]:
        out["reason"] = "free-space contradiction"
    out["pose"] = pose_d
    return out


def _relative(pose_a, pose_b):
    """Edge parameters x_a = s R_y(th) x_b + t between two levelled photos with world poses (yaw, t, log_s)."""
    tha, ta, lsa = pose_a
    thb, tb, lsb = pose_b
    th = thb - tha
    s = float(np.exp(lsb - lsa))
    t = np.exp(-lsa) * _ry(-tha) @ (tb - ta)
    return th, s, t


def layout_edge(name_a: str, pose_a, name_d: str, pose_d, p, note: str) -> Edge:
    th, s, t = _relative(pose_a, pose_d)
    return Edge(name_a, name_d, th, t, float(np.log(s)), 0, 0, 0.0, False, kind="layout", sig_t=p.layout_sigma_t_m,
                sig_yaw_deg=p.layout_sigma_yaw_deg, sig_s=p.prior_sigma_log_s, note=note)


def pair_overlap_violation(edge: Edge, poses: dict, views: dict, margin_m: float = 1.5,
                           violate_rel: float = 0.3) -> float:
    """Post-solve check of a doorway pair: with the pair applied, do the two ROOMS contradict each other?

    A false pair (two photos that were not taken at one spot) pulls two rooms on top of each other; then the walls of
    one room stand inside the free space the other room's photos observed. Worst free-space violation over all photo
    pairs (one photo per room) under the solved poses; > edge_max_violation rejects the pair (no-overlap constraint).
    margin_m: a point only counts if it is > 30% AND > 1.5 m in front of the observed surface: this is a GROSS-error
    detector for prior-quality placements (sigma_t 0.3 m + 10% per-photo scale at 4 m + 8 deg yaw at 4 m ~ 1.3 m).
    Measured (floor_only/rotate, pair R3_07~R6_01, whose four Manhattan yaws were tried): with the pairwise 15%
    test the CORRECT yaw scored 0.94 (photos seeing the same surfaces through the door, 0.3 m / 16% scale off);
    margin 0.6 m: 0.72 vs 0.95-1.0 for wrong yaws; margin 1.0 m: 0.55 vs 0.84-0.97; margin 1.5 m: 0.002 vs
    0.50-0.93."""
    ra, rb = views[edge.a].room, views[edge.b].room
    A = [n for n in poses if views[n].room == ra]
    B = [n for n in poses if views[n].room == rb]
    worst = 0.0
    for a in A:
        for b in B:
            th, s, t = _relative(poses[a], poses[b])
            worst = max(worst, free_space_violation(views[a], views[b], th, s, t, stride=6, violate_rel=violate_rel,
                                                    violate_abs=margin_m)[0])
            th, s, t = _relative(poses[b], poses[a])
            worst = max(worst, free_space_violation(views[b], views[a], th, s, t, stride=6, violate_rel=violate_rel,
                                                    violate_abs=margin_m)[0])
    return worst
