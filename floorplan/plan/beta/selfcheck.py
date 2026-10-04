"""Repeatability self-check: the same apartment captured twice should give the same rooms and walls.

There is no ground truth in the sample data, but floor_only and with_ceiling are two captures of the same flat.
This module (1) registers capture B onto capture A with the shared rigid registration (floorplan/eval/register.py,
computed from the reconstructed geometry, never from the plans), (2) matches rooms by polygon overlap, and
(3) matches walls by orientation, position and corners, then compares their lengths against the Part 2 gate
(agree within 1 cm or 0.5%, whichever is larger).

A wall comparison is only meaningful when both plans bound that wall by the same corners. Walls whose endpoints
differ by more than `corner_tol` are listed separately as topology differences (one plan has a jog or a door
recess the other does not), not as measurement disagreement.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from floorplan.eval.register import apply_T2, register, scene_evidence


def _transform_plan(plan: dict, T: np.ndarray) -> dict:
    """Plan B expressed in A's frame: points get the full rigid transform, normals only the rotation."""
    R = T[:2, :2]
    rooms = [dict(r, polygon=apply_T2(T, np.asarray(r["polygon"])).tolist()) for r in plan["rooms"]]
    walls = [dict(w, p0=apply_T2(T, np.asarray([w["p0"]]))[0].tolist(), p1=apply_T2(T, np.asarray([w["p1"]]))[0].tolist(),
                  normal=(R @ np.asarray(w["normal"], float)).tolist()) for w in plan["walls"]]
    return dict(plan, rooms=rooms, walls=walls)


def match_rooms(a: dict, b: dict, min_iou: float = 0.3) -> list[tuple[str, str, float]]:
    """Greedy one-to-one room matching by polygon IoU (b already in a's frame)."""
    pairs = []
    for ra in a["rooms"]:
        pa = Polygon(ra["polygon"])
        for rb in b["rooms"]:
            pb = Polygon(rb["polygon"])
            iou = pa.intersection(pb).area / max(pa.union(pb).area, 1e-9)
            if iou >= min_iou:
                pairs.append((iou, ra["id"], rb["id"]))
    used_a, used_b, out = set(), set(), []
    for iou, ia, ib in sorted(pairs, reverse=True):
        if ia not in used_a and ib not in used_b:
            used_a.add(ia)
            used_b.add(ib)
            out.append((ia, ib, round(float(iou), 3)))
    return out


def match_walls(a: dict, b: dict, rooms: list[tuple[str, str, float]], offset_tol: float = 0.05,
                corner_tol: float = 0.05) -> list[dict]:
    """Same-room, parallel, same-side walls within offset_tol; flags whether both corners agree."""
    out = []
    by_room_b = {}
    for w in b["walls"]:
        by_room_b.setdefault(w["room_id"], []).append(w)
    pair = {ia: ib for ia, ib, _ in rooms}
    for wa in a["walls"]:
        if wa["room_id"] not in pair or wa["length"]["status"] != "measured":
            continue
        pa0, pa1 = np.asarray(wa["p0"]), np.asarray(wa["p1"])
        da = pa1 - pa0
        best = None
        for wb in by_room_b.get(pair[wa["room_id"]], []):
            pb0, pb1 = np.asarray(wb["p0"]), np.asarray(wb["p1"])
            db = pb1 - pb0
            if abs(np.dot(da, db)) < 0.9 * np.linalg.norm(da) * np.linalg.norm(db):
                continue                                         # not parallel
            n = np.array([-da[1], da[0]]) / max(np.linalg.norm(da), 1e-9)
            off = abs(np.dot((pb0 + pb1) / 2 - pa0, n))
            if off > offset_tol:
                continue
            ends = min(np.linalg.norm(pa0 - pb0) + np.linalg.norm(pa1 - pb1),
                       np.linalg.norm(pa0 - pb1) + np.linalg.norm(pa1 - pb0)) / 2
            if best is None or ends < best[0]:
                best = (ends, wb, off)
        if best is None:
            continue
        ends, wb, off = best
        la, lb = wa["length"]["value"], wb["length"]["value"]
        tol = max(0.01, 0.005 * la)
        out.append(dict(wall_a=wa["id"], wall_b=wb["id"], length_a=round(la, 4), length_b=round(lb, 4),
                        diff_m=round(lb - la, 4), gate_m=round(tol, 4), passes=bool(abs(lb - la) <= tol),
                        same_corners=bool(ends <= corner_tol), mean_corner_shift_m=round(float(ends), 4),
                        offset_diff_m=round(float(off), 4), b_measured=wb["length"]["status"] == "measured",
                        ci_a=round(wa["length"]["hi"] - wa["length"]["value"], 4),
                        ci_b=round(wb["length"]["hi"] - wb["length"]["value"], 4)))
    return out


def _wall_line(w: dict):
    p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
    d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
    return p0, p1, d, np.asarray(w["normal"], float)


def opposite_wall_distances(plan: dict, room_id: str, min_overlap: float = 0.3) -> list[dict]:
    """Distances between facing, parallel, measured walls of one room, taken at the middle of their overlap.

    This is what a laser measurer reads across a room, and it does not depend on where the plan put the
    corners, so it compares two captures even when their wall topologies differ (a jog in one, not the other).
    Each distance starts on the wall whose inward normal points to +u or +v, so the same measurement has the
    same start point in both captures."""
    walls = [w for w in plan["walls"] if w["room_id"] == room_id and w["fit_rmse_m"] is not None]
    out = []
    for i, w1 in enumerate(walls):
        for w2 in walls[i + 1:]:
            if np.dot(_wall_line(w1)[3], _wall_line(w2)[3]) > -0.95:
                continue                                         # not facing each other
            wa, wb = (w1, w2) if max(_wall_line(w1)[3], key=abs) > 0 else (w2, w1)
            a0, a1, da, na = _wall_line(wa)
            b0, b1, db, _ = _wall_line(wb)
            sa = sorted((np.dot(a0, da), np.dot(a1, da)))       # spans along the wall direction
            sb = sorted((np.dot(b0, da), np.dot(b1, da)))
            lo, hi = max(sa[0], sb[0]), min(sa[1], sb[1])
            if hi - lo < min_overlap:
                continue
            pa = a0 + da * ((lo + hi) / 2 - np.dot(a0, da))     # point on wall a at the overlap middle
            pb = b0 + db * np.dot(pa - b0, db)                   # foot point on wall b
            out.append(dict(walls=(wa["id"], wb["id"]), at=pa.tolist(), dist=float(np.dot(pb - pa, na))))
    return out


def compare_opposite_distances(a: dict, b_in_a: dict, rooms, pos_tol: float = 0.15) -> list[dict]:
    """Match opposite-wall distances of matched rooms by location and direction; report the differences."""
    rows = []
    for ia, ib, _ in rooms:
        da, db = opposite_wall_distances(a, ia), opposite_wall_distances(b_in_a, ib)
        for x in da:
            pa = np.asarray(x["at"])
            best = None
            for y in db:
                if abs(y["dist"] - x["dist"]) > 0.3:
                    continue
                dvec_x = np.asarray(_wall_line(next(w for w in a["walls"] if w["id"] == x["walls"][0]))[3])
                dvec_y = np.asarray(_wall_line(next(w for w in b_in_a["walls"] if w["id"] == y["walls"][0]))[3])
                if abs(np.dot(dvec_x, dvec_y)) < 0.95:
                    continue                                     # measured along a different direction
                # same measurement line: y's start point lies on x's line, close to x's start
                off = np.asarray(y["at"]) - pa
                along = off - dvec_x * np.dot(off, dvec_x)
                if np.linalg.norm(along) < 0.5 and abs(np.dot(off, dvec_x)) < pos_tol + 0.1:
                    if best is None or np.linalg.norm(off) < best[0]:
                        best = (np.linalg.norm(off), y)
            if best is not None:
                d1, d2 = x["dist"], best[1]["dist"]
                tol = max(0.01, 0.005 * d1)
                rows.append(dict(room_a=ia, room_b=ib, walls_a=x["walls"], walls_b=best[1]["walls"],
                                 dist_a=round(d1, 4), dist_b=round(d2, 4), diff_m=round(d2 - d1, 4),
                                 gate_m=round(tol, 4), passes=bool(abs(d2 - d1) <= tol)))
    return rows


def repeatability(scene_a, info_a, plan_a: dict, scene_b, info_b, plan_b: dict) -> dict:
    reg = register(scene_evidence(scene_a, info_a, plan_a["capture_id"]),
                   scene_evidence(scene_b, info_b, plan_b["capture_id"]))
    T = np.asarray(reg.T2)
    b_in_a = _transform_plan(plan_b, T)
    rooms = match_rooms(plan_a, b_in_a)
    walls = match_walls(plan_a, b_in_a, rooms)
    comparable = [w for w in walls if w["same_corners"] and w["b_measured"]]
    opp = compare_opposite_distances(plan_a, b_in_a, rooms)
    by_id_a = {r["id"]: r for r in plan_a["rooms"]}
    by_id_b = {r["id"]: r for r in plan_b["rooms"]}
    room_rows = []
    for ia, ib, iou in rooms:
        ra, rb = by_id_a[ia], by_id_b[ib]
        room_rows.append(dict(room_a=ia, room_b=ib, iou=iou,
                              area_a=round(ra["floor_area"]["value"], 3), area_b=round(rb["floor_area"]["value"], 3),
                              bbox_a=[round(d["value"], 3) for d in ra["bbox_dims"]],
                              bbox_b=[round(d["value"], 3) for d in rb["bbox_dims"]]))
    return dict(
        registration=dict(yaw_deg=reg.yaw_deg, t=reg.t, residual_rmse_m=reg.residual_rmse_m,
                          overlap_iou=reg.overlap_iou, distinctiveness=reg.distinctiveness),
        rooms_by_area=dict(a=_rooms_by_area(plan_a), b=_rooms_by_area(plan_b)),
        matched_rooms=room_rows, walls=walls, opposite_wall_distances=opp,
        summary=dict(matched_walls=len(walls), comparable_walls=len(comparable),
                     comparable_pass=sum(w["passes"] for w in comparable),
                     median_abs_diff_comparable_m=float(np.median([abs(w["diff_m"]) for w in comparable]))
                     if comparable else None,
                     within_ci=sum(abs(w["diff_m"]) <= np.hypot(w["ci_a"], w["ci_b"]) for w in comparable),
                     opposite_pairs=len(opp), opposite_pass=sum(r["passes"] for r in opp),
                     opposite_median_abs_diff_m=float(np.median([abs(r["diff_m"]) for r in opp])) if opp else None,
                     opposite_p90_abs_diff_m=float(np.percentile([abs(r["diff_m"]) for r in opp], 90))
                     if opp else None))


def _rooms_by_area(plan: dict) -> list[dict]:
    rows = [dict(id=r["id"], label=r["label"], area=round(r["floor_area"]["value"], 3),
                 bbox=[round(d["value"], 3) for d in r["bbox_dims"]]) for r in plan["rooms"]]
    return sorted(rows, key=lambda x: -x["area"])
