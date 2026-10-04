"""Score a plan against tape-measured ground truth (the candidate's own captures; decision D-015).

Ground truth comes from benchmark/ground_truth.csv (one row per measurement, metres). Walls are named W1..Wn
CLOCKWISE (seen from above) starting with the wall that holds the room's main door; see
TakeHome/HOUSE_CAPTURE_GUIDE.md step 8.

Matching (rooms -> walls -> openings) is done by geometry only, never by hand:
  * rooms: Hungarian assignment on a cost built from the sorted wall-length profile and perimeter, unless the plan
    room label equals the GT room id (photo tier: folder names). With GT polygons and a whole-plan fit of IoU >= 0.5,
    unlabelled rooms are paired by polygon overlap instead (a length profile can swap two similar rooms);
  * walls (floorplan/benchmark/wall_match.py): by GEOMETRY when GT polygons exist (sim GT: sim_gt.json next to the
    CSV): the predicted room is put in the GT room's frame by a rigid transform and each GT wall takes the predicted
    wall(s) on its line (same direction +-10 deg, offset <= 0.30 m; collinear pieces merge into one end-to-end
    length). A tape GT has lengths only (decision D-072, tape_method 'geometric', the default): the room's outline is
    rebuilt from the tape lengths (right-angled corners, clockwise from W1), the predicted room is registered onto it
    by IoU (door rule for near-ties; never by length error) and paired by geometry as above. When several outlines
    fit the tape, each is scored and the WORST is reported (WARNING); when none fits, a constrained order pairing
    (axis and door as hard constraints; NOTE issue I-001: plan (u, v) = (x, z) is mirror-reversed seen from above).
    The earlier order pairings stay selectable for comparison: tape_method 'anchored' and 'order-merged'. Every GT
    wall gets a row (MISSED if unpaired, MERGED if a predicted wall spans it with others: both count as failures),
    and every row says how it was paired; unmatched predicted walls are reported. The old pure cyclic-order pairing
    (which shifts after any extra short wall) is kept for comparability under the result key 'wall_order_based'
    (not gated);
  * openings: per room, Hungarian on width difference.

Reported per item: predicted value, interval, GT, error, |error| / GT, and whether GT lies inside the 95% interval
(calibration). Gate checks follow the case study's Part 2 thresholds.
"""
from __future__ import annotations

import csv
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from floorplan.benchmark import wall_match as wm

GATES = {
    "wall_rel_tol": {"photo": 0.08, "video": 0.03, "lidar": None},   # photo +-8%, video +-3% (case study Part 2)
    "opening_abs_tol_m": 0.02, "opening_pass_frac": 0.85,
    "ceiling_abs_tol_m": 0.015,
    "footprint_rel_tol_photo": 0.08,
}


@dataclass
class GTItem:
    room: str
    kind: str
    item: str
    value: float
    notes: str = ""
    description: str = ""


def read_gt(path) -> list[GTItem]:
    items = []
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    head = [h.strip() for h in rows[0]]
    for r in rows[1:]:
        d = dict(zip(head, [c.strip() for c in r]))
        if not d.get("value_m"):
            continue
        items.append(GTItem(d["room_id"], d["item_type"], d["item_id"], float(d["value_m"]), d.get("notes", ""),
                            d.get("description", "")))
    return items


def tape_door_walls(gt: list[GTItem], room: str) -> set[str]:
    """GT walls known to hold a door, for a tape GT: W1 (the naming rule: W1 has the main door), plus a wall a door
    row names explicitly ('on W3' in its description or notes). The guide records other door positions only in the
    sketch, so nothing else is assumed."""
    out = {"W1"}
    for it in gt:
        if it.room == room and it.kind in ("door_width", "door_height"):
            out |= {m.upper() for m in re.findall(r"\bon\s+(W\d+)\b", f"{it.description} {it.notes}", re.I)}
    return out


def _m(meas):
    """(value, lo, hi) from a serialised Measurement dict (None-safe).

    Accepts both serialisations: the dataclass dump (lo/hi) and the published plan.json schema (ci95: [lo, hi])."""
    if not meas:
        return None, None, None
    if "ci95" in meas and meas["ci95"] is not None:
        lo, hi = meas["ci95"]
        return meas.get("value"), lo, hi
    return meas.get("value"), meas.get("lo"), meas.get("hi")


def _signed_area(poly):
    p = np.asarray(poly, float)
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


def _wnum(name: str) -> int:
    return wm._wnum(name)


def read_gt_geometry(gt_csv, gt_json=None) -> dict | None:
    """GT room polygons with the wall names, when available: an explicit GT JSON, else <csv folder>/sim_gt.json (the
    simulator writes it; a tape GT has none -> None). Polygon in a top view (x right, y up); wall W(k) is the polygon
    segment from vertex 'edge' to the next. {room_id: {"poly": (N, 2), "walls": {"W1": {a, b, door}, ...}}}"""
    p = Path(gt_json) if gt_json else Path(gt_csv).with_name("sim_gt.json")
    if not p.is_file():
        if gt_json:
            raise FileNotFoundError(f"GT JSON not found: {p}")
        return None
    out = {}
    for r in json.loads(p.read_text()).get("rooms", []):
        poly = np.asarray(r.get("polygon") or [], float)
        if poly.ndim != 2 or len(poly) < 3 or not r.get("walls_cw"):
            continue
        n = len(poly)
        out[r["id"]] = dict(poly=poly, walls={
            w["name"]: dict(a=poly[int(w["edge"]) % n], b=poly[(int(w["edge"]) + 1) % n],
                            door=any(str(o).startswith("door") for o in w.get("openings", [])))
            for w in r["walls_cw"]})
    return out or None


def walls_clockwise_from_above(plan, room) -> list[dict]:
    """The room's walls ordered clockwise as seen from above (see the I-001 note in the module docstring)."""
    walls = {w["id"]: w for w in plan["walls"]}
    ws = [walls[i] for i in room["wall_ids"] if i in walls]
    poly = room["polygon"]
    # order walls by the polygon vertex they start from (nearest p0), following the polygon order
    pts = np.asarray(poly, float)
    def start_index(w):
        return int(np.argmin(np.linalg.norm(pts - np.asarray(w["p0"], float), axis=1)))
    ws.sort(key=start_index)
    # mirrored plan (I-001; coordinate_frame.view_from_above v = down, the default): polygon CCW in (u,v) ==
    # clockwise from above, so reverse a CW one; an unmirrored plan is the other way round
    if _signed_area(poly) * (1 if wm.plan_mirrored(plan) else -1) < 0:
        ws = ws[::-1]
    return ws


def match_walls(pred_walls, gt_lengths, door_flags):
    """Best cyclic alignment of predicted walls to GT W1..Wn. Returns (pairs, direction, start)."""
    n_p, n_g = len(pred_walls), len(gt_lengths)
    if n_p == 0 or n_g == 0:
        return [], None, None
    L = np.array([(_m(w["length"])[0] or 0.0) for w in pred_walls])
    best = None
    for direction in (1, -1):
        order = list(range(n_p)) if direction == 1 else list(reversed(range(n_p)))
        for s in range(n_p):
            seq = [order[(s + k) % n_p] for k in range(n_p)]
            m = min(n_p, n_g)
            cost = float(np.sum(np.abs(L[seq[:m]] - np.asarray(gt_lengths[:m]))))
            cost += 0.05 * abs(n_p - n_g)
            if not door_flags[seq[0]]:
                cost += 0.02                       # GT W1 is the wall with the main door
            if direction == -1:
                cost += 0.01                       # small preference for the documented clockwise order
            if best is None or cost < best[0]:
                best = (cost, seq, direction, s)
    _, seq, direction, s = best
    pairs = [(seq[k], k) for k in range(min(n_p, n_g))]
    return pairs, direction, s


def room_profile(lengths):
    v = np.sort(np.asarray(lengths, float))[::-1]
    return v


def _door_width_cost(gt_w: list[float], pred_w: list[float], cap=0.5) -> float:
    """Room-pairing evidence from door widths (a tape GT has them per room): Hungarian on |width difference|, each
    pair capped at `cap` m; doors without a width or without a partner add nothing."""
    if not gt_w or not pred_w:
        return 0.0
    C = np.minimum(np.abs(np.subtract.outer(np.asarray(gt_w, float), np.asarray(pred_w, float))), cap)
    ra, cb = linear_sum_assignment(C)
    return float(C[ra, cb].sum())


def match_rooms(plan, gt_rooms: dict[str, list[float]], gt_doors: dict[str, list[float]] | None = None):
    """GT room -> plan room index: exact label, else Hungarian on the wall-length profile (+ door widths when
    gt_doors is given: {GT room: [door widths]}; a profile alone can swap two rooms of similar size)."""
    pred = plan["rooms"]
    labels = {r.get("label", ""): i for i, r in enumerate(pred)}
    pairs, used_p, used_g = {}, set(), set()
    for g in gt_rooms:                                         # exact label match first (photo-tier folder names)
        if g in labels:
            pairs[g] = labels[g]; used_p.add(labels[g]); used_g.add(g)
    rest_g = [g for g in gt_rooms if g not in used_g]
    rest_p = [i for i in range(len(pred)) if i not in used_p]
    if rest_g and rest_p:
        walls = {w["id"]: w for w in plan["walls"]}
        C = np.zeros((len(rest_g), len(rest_p)))
        for a, g in enumerate(rest_g):
            pg = room_profile(gt_rooms[g])
            for b, i in enumerate(rest_p):
                pl = room_profile([(_m(walls[w]["length"])[0] or 0) for w in pred[i]["wall_ids"] if w in walls])
                k = min(len(pg), len(pl))
                C[a, b] = np.abs(pg[:k] - pl[:k]).sum() + 0.5 * abs(len(pg) - len(pl)) + abs(pg.sum() - pl.sum()) * 0.5
                if gt_doors is not None:
                    C[a, b] += _door_width_cost(gt_doors.get(g, []), [
                        _m(o.get("width"))[0] for o in plan.get("openings", [])
                        if pred[i]["id"] in o.get("room_ids", []) and o.get("kind") in ("door", "passage")
                        and _m(o.get("width"))[0] is not None])
        ra, cb = linear_sum_assignment(C)
        for a, b in zip(ra, cb):
            pairs[rest_g[a]] = rest_p[b]
    return pairs


def _wall_row(g_room, room, item, pred_id, v, lo, hi, gtv, note, match):
    return dict(room=g_room, pred_room=room["id"], item=item, kind="wall", pred_id=pred_id, value=v, lo=lo, hi=hi,
                gt=gtv, err=None if v is None else v - gtv, rel=None if v is None else abs(v - gtv) / gtv,
                inside=None if lo is None or v is None else bool(lo <= gtv <= hi), note=note, match=match)


def _legacy_order_rows(plan, room, g_room, gt_len, openings) -> list[dict]:
    """The old pairing (pure cyclic order, no merging of short walls); kept for comparability, not gated."""
    ws = walls_clockwise_from_above(plan, room)
    door_flags = [wm._has_door(w, openings) for w in ws]
    pairs, direction, start = match_walls(ws, gt_len, door_flags)
    return [_wall_row(g_room, room, f"W{k + 1}", ws[p]["id"], *_m(ws[p]["length"]), gt_len[k],
                      f"order {'clockwise' if direction == 1 else 'REVERSED'} start {start}", "order (legacy)")
            for p, k in pairs]


TAPE_METHODS = ("geometric", "anchored", "order-merged")
_MISS_GEO = f"MISSED (no predicted wall within {wm.PARAMS['offset_m']:.2f} m of this wall's line)"


def _room_score(rows: list[dict], tol: float) -> tuple[int, float]:
    """Order of the pairings of one room, to report the WORST: (GT walls within tol, -sum of min(rel. error, 1));
    MISSED and MERGED walls count as outside tol with error 1."""
    rel = [1.0 if r["rel"] is None else min(r["rel"], 1.0) for r in rows]
    return sum(x <= tol for x in rel), -float(sum(rel))


def _wall_rows(plan, room, g_room, names, gt_len, gtg, glob, openings, mirror, tape_method="geometric",
               outline=None, tol=0.08, door_walls=("W1",)) -> tuple[list[dict], dict]:
    """Default wall rows of one room pair: geometric pairing when the GT room has a polygon with these wall names,
    else a tape-GT pairing: tape_method 'geometric' (default, D-072): geometric on the outline rebuilt from the tape
    (outline: wall_match.reconstruct_outlines), or constrained order when no outline fits; 'anchored' or
    'order-merged': the earlier order pairings, for comparison. Every GT wall gets a row (MISSED when nothing pairs
    with it; a GT wall a predicted wall spans with others is MERGED and counts as missed). tol: the wall gate's
    relative tolerance, used to pick the worst of several pairings. Returns (rows, pairing summary)."""
    if (gtg is not None and all(n in gtg["walls"] for n in names) and len(room.get("polygon") or []) >= 3
            and room.get("wall_ids")):
        gp = wm.geometric_pairs(plan, room, gtg, names, mirror, glob, openings)
        al = gp["align"]
        how = f"geometric on the GT polygon ({al['frame']} frame, rot {al['rot_deg']:.1f} deg, IoU {al['iou']:.2f})"
        info = dict(method="geometric", frame=al["frame"], rot_deg=round(al["rot_deg"], 2), iou=round(al["iou"], 3),
                    ambiguity=al["ambiguity"])
        return _rows_from_pairs(room, g_room, names, gt_len, al["ws"], gp["by_gt"], gp["offsets"], {},
                                "geometric (GT polygon)", how, _MISS_GEO, info)
    if tape_method == "geometric" and room.get("wall_ids") and len(room.get("polygon") or []) >= 3:
        outline = outline or wm.reconstruct_outlines(gt_len, names)
        if outline["status"] != "unreconstructable":
            return _outline_rows(plan, room, g_room, names, gt_len, outline, openings, mirror, tol, door_walls)
    ws = walls_clockwise_from_above(plan, room)
    if tape_method == "geometric":
        om = wm.constrained_pairs(room, ws, gt_len, openings)
        reason = (outline or {}).get("reason") or "the plan's room has no polygon"
        warnings = [f"no outline: {reason}; walls paired by constrained order (axis and door as hard constraints)"]
        warnings += om["warnings"]
        info = dict(method="constrained order", how=om["how"], anchored=om["anchored"], cost=om["cost"],
                    door_walls=om["door_walls"], outline=_outline_summary(outline), warnings=warnings,
                    ambiguity="; ".join(om["warnings"]) or None)
        return _rows_from_pairs(room, g_room, names, gt_len, ws, om["by_gt"], {}, om["gt_runs"],
                                "constrained order (no outline)", f"{om['how']}; no outline: {reason}",
                                "MISSED (left out by the constrained order alignment)", info)
    if tape_method == "anchored":
        om = wm.anchored_pairs(room, ws, gt_len, openings)
        info = dict(method="anchored", how=om["how"], anchored=om["anchored"], start=om["start"], cost=om["cost"],
                    door_walls=om["door_walls"], ambiguity=om["ambiguity"], reverse_note=om["reverse_note"])
        return _rows_from_pairs(room, g_room, names, gt_len, ws, om["by_gt"], {}, om["gt_runs"], "anchored",
                                f"{om['how']}; no GT polygons",
                                "MISSED (left out by the alignment: nothing in order fits it better than a miss)", info)
    om = wm.order_merged_pairs(room, ws, gt_len, openings)
    info = dict(method="order-merged", direction=om["direction"], start=om["start"])
    return _rows_from_pairs(room, g_room, names, gt_len, ws, om["by_gt"], {}, {}, "order-merged",
                            f"order-merged ({'clockwise' if om['direction'] == 1 else 'REVERSED'} start {om['start']}; "
                            f"no GT polygons)", "MISSED (no predicted wall left in order within ~50% of its length)",
                            info)


def _outline_summary(outline: dict | None) -> dict | None:
    if not outline:
        return None
    return dict(status=outline["status"], n_outlines=len(outline["shapes"]),
                tol_m=None if outline["tol"] is None else round(outline["tol"], 4),
                misclosure_m=[round(s["misclosure"], 4) for s in outline["shapes"]],
                area_m2=[round(s["area"], 3) for s in outline["shapes"]],
                best_misclosure_m=None if outline["best_misclosure"] is None else round(outline["best_misclosure"], 4),
                reason=outline["reason"])


def _outline_rows(plan, room, g_room, names, gt_len, outline, openings, mirror, tol, door_walls):
    """Tape GT, default pairing: geometric on the outline(s) rebuilt from the tape (wall_match.register_outline, then
    pairs_on_lines). Several outlines, or registrations the door rule cannot split: every one is paired and scored,
    and the WORST (_room_score) is reported with a warning; never the best."""
    shapes, cands = outline["shapes"], []
    for si, sh in enumerate(shapes):
        gtg = wm.outline_geometry(sh["poly"], names, door_walls)
        regs, note = wm.register_outline(plan, room, gtg, names, mirror, openings)
        for al in regs:
            gp = wm.pairs_on_lines(al)
            tag = f"outline {si + 1} of {len(shapes)}, " if len(shapes) > 1 else ""
            how = (f"geometric on the reconstructed outline ({tag}{al['frame']}, rot {al['rot_deg']:.1f} deg, "
                   f"IoU {al['iou_reg']:.2f})")
            rows, _ = _rows_from_pairs(room, g_room, names, gt_len, al["ws"], gp["by_gt"], gp["offsets"], {},
                                       "", how, _MISS_GEO, {})
            cands.append(dict(rows=rows, gp=gp, how=how, al=al, note=note, shape=si, n_regs=len(regs),
                              score=_room_score(rows, tol)))
    worst = min(cands, key=lambda cd: cd["score"])
    vals = [tuple(None if r["value"] is None else round(r["value"], 3) for r in cd["rows"]) for cd in cands]
    differ = len(set(vals)) > 1
    warnings = []
    if len(shapes) > 1:
        areas = ", ".join(f"{s['area']:.2f}" for s in shapes)
        warnings.append(
            f"AMBIGUOUS outline: {len(shapes)} right-angled outlines fit the tape lengths (areas {areas} m2; e.g. a "
            f"bump that could be a notch); each was paired "
            + ("and the WORST is reported (walls within tol: "
               + ", ".join(f"{cd['score'][0]}/{len(names)}" for cd in cands) + ")" if differ
               else "and every one gives the same lengths"))
    ties = {cd["shape"]: cd["note"] for cd in cands if cd["n_regs"] > 1}
    if ties:
        warnings.append("registration tie: " + "; ".join(
            (f"outline {si + 1}: " if len(shapes) > 1 else "") + nt for si, nt in ties.items())
                        + ("; the WORST is reported" if differ else "; every one gives the same lengths"))
    if worst["al"].get("ambiguity"):
        warnings.append(worst["al"]["ambiguity"])
    method = "geometric (reconstructed outline" + ("; AMBIGUOUS, worst reported)" if differ else ")")
    al, gp = worst["al"], worst["gp"]
    rows, info = _rows_from_pairs(room, g_room, names, gt_len, al["ws"], gp["by_gt"], gp["offsets"], {}, method,
                                  worst["how"], _MISS_GEO, {})
    info = dict(info, method="geometric (reconstructed)", frame=al["frame"], rot_deg=round(al["rot_deg"], 2),
                iou=round(al["iou_reg"], 3), iou_refined=round(al["iou"], 3), registration=worst["note"],
                outline=_outline_summary(outline), outline_reported=worst["shape"] + 1, door_walls=sorted(door_walls),
                candidates=[dict(outline=cd["shape"] + 1, frame=cd["al"]["frame"], iou=round(cd["al"]["iou_reg"], 3),
                                 within_tol=cd["score"][0], err_sum=round(-cd["score"][1], 3)) for cd in cands],
                warnings=warnings, ambiguity="; ".join(warnings) or None)
    return rows, info


def _rows_from_pairs(room, g_room, names, gt_len, ws, groups, off, runs, method, how, miss, info):
    """Rows of one pairing ({GT index: predicted wall indices of ws}, optional offsets, GT runs {main: GT indices}),
    and info completed with the missed, merged and unmatched walls."""
    s0, s1, c = wm.corner_model(room, ws)
    pos = {wid: k for k, wid in enumerate(room.get("wall_ids", []))}
    merged_into = {g: main for main, gi in runs.items() for g in gi if g != main}
    rows = []
    for k, (name, gtv) in enumerate(zip(names, gt_len)):
        idx = sorted(groups.get(k, []), key=lambda i: pos.get(ws[i]["id"], i))
        if not idx and k in merged_into:
            main = merged_into[k]
            ids = "+".join(ws[i]["id"] for i in groups.get(main, []))
            rows.append(_wall_row(g_room, room, name, None, None, None, None, gtv,
                                  f"MERGED: {ids} spans {'+'.join(names[g] for g in runs[main])} and is compared "
                                  f"with {names[main]}; this wall is not measured by the plan", f"{method}: merged"))
            continue
        if not idx:
            rows.append(_wall_row(g_room, room, name, None, None, None, None, gtv, f"{miss}; {how}",
                                  f"{method}: missed"))
            continue
        v, lo, hi = wm.piece_length(ws, idx, s0, s1, c)
        ids = "+".join(ws[i]["id"] for i in idx)
        note = (f"{ids}: {how}"
                + (f"; {len(idx)} collinear pieces merged (end-to-end span)" if len(idx) > 1 else "")
                + (f"; spans {'+'.join(names[g] for g in runs[k])} (GT span "
                   f"{sum(gt_len[g] for g in runs[k][0::2]):.3f} m), compared with {name}, the longest GT wall on "
                   f"its line" if k in runs else ""))
        if off:
            note += f"; offset {max(off[i] for i in idx):.3f} m"
        rows.append(_wall_row(g_room, room, name, ids, v, lo, hi, gtv, note,
                              method + (f", {len(idx)} pieces merged" if len(idx) > 1 else "")
                              + (f", spans {'+'.join(names[g] for g in runs[k])}" if k in runs else "")))
    used = {i for idx in groups.values() for i in idx}
    um = [(w["id"], round(_m(w["length"])[0] or 0.0, 4)) for i, w in enumerate(ws) if i not in used]
    long_ = [x for x in um if x[1] >= wm.PARAMS["short_m"]]
    info.update(missed=[n for k, n in enumerate(names) if not groups.get(k)],
                merged={names[main]: [names[g] for g in gi] for main, gi in runs.items()}, unmatched_pred=um,
                unmatched_pred_n=len(um), unmatched_pred_len_m=round(sum(x[1] for x in um), 3),
                unmatched_pred_long_n=len(long_), unmatched_pred_long_len_m=round(sum(x[1] for x in long_), 3))
    return rows, info


def evaluate(plan: dict, gt: list[GTItem], tier: str, gt_geom: dict | None = None,
             tape_method: str = "geometric") -> dict:
    """Score one plan. gt_geom: GT room polygons (read_gt_geometry); without them (a tape GT) walls are paired by
    tape_method: 'geometric' (default: on the outline rebuilt from the tape, D-072), or the earlier order pairings
    'anchored' and 'order-merged', kept for comparison."""
    if tape_method not in TAPE_METHODS:
        raise ValueError(f"tape_method {tape_method!r}: one of {TAPE_METHODS}")
    gt_walls: dict[str, dict[str, float]] = {}
    for it in gt:
        if it.kind == "wall":
            gt_walls.setdefault(it.room, {})[it.item] = it.value
    gt_names = {r: sorted(ws, key=_wnum) for r, ws in gt_walls.items()}
    gt_rooms = {r: [gt_walls[r][n] for n in gt_names[r]] for r in gt_walls}
    gt_doors: dict[str, list[float]] = {}
    for it in gt:
        if it.kind == "door_width":
            gt_doors.setdefault(it.room, []).append(it.value)
    room_pairs = match_rooms(plan, gt_rooms, gt_doors)
    openings = {o["id"]: o for o in plan.get("openings", [])}
    mirror = wm.plan_mirrored(plan)
    glob = wm.global_alignment(plan, gt_geom, mirror) if gt_geom else None
    room_pairing, profile_pairs = "label, else wall-length profile + door widths (Hungarian)", None
    if glob is not None and glob["score"] >= wm.PARAMS["global_min_iou"]:
        # the whole plan sits on the whole GT: pair rooms by overlap (a length profile can swap similar rooms)
        labels = {r.get("label", ""): i for i, r in enumerate(plan["rooms"])}
        fixed = {g: labels[g] for g in gt_rooms if g in labels}
        geo = wm.pair_rooms_by_overlap(plan, {g: gt_geom[g] for g in gt_rooms if g in gt_geom}, glob, mirror, fixed)
        for g, i in room_pairs.items():                      # GT rooms without a polygon keep the profile pair
            if g not in gt_geom and i not in geo.values():
                geo[g] = i
        if geo != room_pairs:
            profile_pairs = {g: plan["rooms"][i]["id"] for g, i in room_pairs.items()}
        room_pairs, room_pairing = geo, "label, else polygon IoU under the global plan->GT transform (Hungarian)"
    # the old scorer in full (profile room pairs + pure cyclic order), for comparability only
    legacy = [r for g, i in match_rooms(plan, gt_rooms).items()
              for r in _legacy_order_rows(plan, plan["rooms"][i], g, gt_rooms[g], openings)]
    tol = GATES["wall_rel_tol"].get(str(tier).split("_")[0])
    # tape GT rooms (no polygon): the outline(s) the tape lengths allow
    outlines = {g: wm.reconstruct_outlines(gt_rooms[g], gt_names[g]) for g in gt_rooms
                if tape_method == "geometric" and g not in (gt_geom or {})}
    rows, pairing, candidates_phantom = [], {}, []
    for g_room, p_idx in room_pairs.items():
        room = plan["rooms"][p_idx]
        wr, pairing[g_room] = _wall_rows(plan, room, g_room, gt_names[g_room], gt_rooms[g_room],
                                         (gt_geom or {}).get(g_room), glob, openings, mirror, tape_method,
                                         outlines.get(g_room), tol or 0.08, tape_door_walls(gt, g_room))
        rows += wr
        # ceiling
        for it in gt:
            if it.room == g_room and it.kind == "ceiling_height":
                v, lo, hi = _m(room.get("ceiling_height"))
                rows.append(dict(room=g_room, pred_room=room["id"], item=it.item, kind="ceiling_height", pred_id=room["id"],
                                 value=v, lo=lo, hi=hi, gt=it.value, err=None if v is None else v - it.value,
                                 rel=None if v is None else abs(v - it.value) / it.value,
                                 inside=None if lo is None else bool(lo <= it.value <= hi), note="",
                                 match="room pair"))
        # openings (doors/passages and windows), Hungarian on width
        for kind_gt, kinds_pred in (("door_width", ("door", "passage")), ("window_width", ("window",))):
            g_items = [it for it in gt if it.room == g_room and it.kind == kind_gt]
            p_items = [o for o in plan.get("openings", []) if room["id"] in o.get("room_ids", []) and o.get("kind") in kinds_pred]
            if g_items and p_items:
                C = np.array([[abs((_m(o["width"])[0] or 9) - it.value) for o in p_items] for it in g_items])
                ra, cb = linear_sum_assignment(C)
            else:
                ra, cb = np.array([], int), np.array([], int)
            matched_g = set(ra.tolist())
            for a, b in zip(ra, cb):
                o, it = p_items[b], g_items[a]
                v, lo, hi = _m(o["width"])
                rows.append(dict(room=g_room, pred_room=room["id"], item=it.item, kind=kind_gt, pred_id=o["id"],
                                 value=v, lo=lo, hi=hi, gt=it.value, err=None if v is None else v - it.value,
                                 rel=None if v is None else abs(v - it.value) / it.value,
                                 inside=None if lo is None else bool(lo <= it.value <= hi), note="",
                                 match="Hungarian on width"))
            for a, it in enumerate(g_items):
                if a not in matched_g:
                    rows.append(dict(room=g_room, pred_room=room["id"], item=it.item, kind=kind_gt, pred_id=None, value=None,
                                     lo=None, hi=None, gt=it.value, err=None, rel=None, inside=None, note="MISSED (not detected)",
                                     match="missed"))
            for b, o in enumerate(p_items):
                if b not in set(cb.tolist()):
                    candidates_phantom.append((g_room, room["id"], kind_gt, o))
    # a door shared by two rooms may be listed in only one room's GT: it is phantom only if NO room matched it
    matched_ids = {r["pred_id"] for r in rows if r["kind"] in ("door_width", "window_width") and r["gt"] is not None}
    seen = set()
    for g_room, rid, kind_gt, o in candidates_phantom:
        if o["id"] in matched_ids or o["id"] in seen:
            continue
        seen.add(o["id"])
        rows.append(dict(room=g_room, pred_room=rid, item="-", kind=kind_gt, pred_id=o["id"], value=_m(o["width"])[0],
                         lo=None, hi=None, gt=None, err=None, rel=None, inside=None, note="PHANTOM (no GT opening)",
                         match="phantom"))
    pairs_ids = {g: plan["rooms"][i]["id"] for g, i in room_pairs.items()}
    rows += _area_rows(plan, gt, room_pairs)
    gates = gate_summary(rows, tier)
    gates["wall_pairing"] = _pairing_summary(pairing, glob)
    if outlines:
        gates["wall_pairing"]["tape_outlines"] = {s: sorted(g for g, o in outlines.items() if o["status"] == s)
                                                  for s in ("reconstructed", "ambiguous", "unreconstructable")}
    st = stitch_summary(plan, gt, pairs_ids, tier)
    if st:
        gates["stitch"] = st
    return dict(tier=tier, rows=rows, room_pairs=pairs_ids, room_pairing=room_pairing, tape_method=tape_method,
                tape_outlines={g: _outline_summary(o) for g, o in outlines.items()},
                room_pairs_by_profile_if_different=profile_pairs, gates=gates, wall_pairing=pairing,
                wall_order_based=dict(note="OLD pure cyclic-order wall pairing (shifts after any extra short wall); "
                                           "for comparability only, not gated",
                                      summary=_wall_stats(legacy, tol), rows=legacy),
                damage=evaluate_damage(plan, gt, pairs_ids))


def _pairing_summary(pairing: dict, glob) -> dict:
    out = dict(methods=sorted({p["method"] for p in pairing.values()}),
               missed_gt_walls=sum(len(p["missed"]) for p in pairing.values()),
               unmatched_pred_n=sum(p["unmatched_pred_n"] for p in pairing.values()),
               unmatched_pred_len_m=round(sum(p["unmatched_pred_len_m"] for p in pairing.values()), 3),
               unmatched_pred_long_n=sum(p["unmatched_pred_long_n"] for p in pairing.values()),
               unmatched_pred_long_len_m=round(sum(p["unmatched_pred_long_len_m"] for p in pairing.values()), 3),
               ambiguous_rooms={g: p["ambiguity"] for g, p in pairing.items() if p.get("ambiguity")},
               warnings=[f"{g}: {w}" for g, p in pairing.items() for w in p.get("warnings", [])],
               note=f"unmatched predicted walls: all / only those >= {wm.PARAMS['short_m']:.2f} m (short ones are "
                    f"jogs, door reveals, stubs)")
    if glob is not None:
        out["global_alignment"] = dict(score_iou=round(glob["score"], 3), used=glob["score"] >= wm.PARAMS["global_min_iou"],
                                       rot_deg=round(float(np.degrees(np.arctan2(glob["R"][1, 0], glob["R"][0, 0]))), 2))
    return out


def _area_rows(plan, gt, room_pairs):
    """Per-room floor area against GT 'area' rows (written by the simulator's ground truth; a tape GT may not have
    them). Reported, not gated per room; the footprint gate is in stitch_summary."""
    out = []
    for it in gt:
        if it.kind != "area" or it.room not in room_pairs:
            continue
        room = plan["rooms"][room_pairs[it.room]]
        v, lo, hi = _m(room.get("floor_area"))
        out.append(dict(room=it.room, pred_room=room["id"], item=it.item, kind="area", pred_id=room["id"], value=v, lo=lo,
                        hi=hi, gt=it.value, err=None if v is None else v - it.value,
                        rel=None if v is None else abs(v - it.value) / it.value,
                        inside=None if lo is None else bool(lo <= it.value <= hi), note="", match="room pair"))
    return out


def _overlaps(plan, min_frac=0.03):
    """Pairs of plan rooms whose polygons overlap by more than min_frac of the smaller room."""
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return None
    polys = []
    for r in plan["rooms"]:
        try:
            pg = Polygon(r.get("polygon") or [])
            polys.append((r["id"], pg.buffer(0) if pg.is_valid is False else pg))
        except Exception:
            polys.append((r["id"], None))
    out = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            a, b = polys[i][1], polys[j][1]
            if a is None or b is None or a.is_empty or b.is_empty:
                continue
            inter = a.intersection(b).area
            if inter > min_frac * min(a.area, b.area):
                out.append((polys[i][0], polys[j][0], round(float(inter), 3)))
    return out


def stitch_summary(plan, gt, pairs_ids, tier):
    """Whole-property stitch (case study Part 2): every GT room present, correct adjacency, no room overlaps,
    footprint within +-8% (photo tier; reported for every tier). GT adjacency comes from door rows whose notes name
    the two rooms ('rooms A+B', as the simulator's GT writes them); a tape GT without them skips that check."""
    gt_rooms = sorted({it.room for it in gt if it.kind == "wall"})
    gt_area = {it.room: it.value for it in gt if it.kind == "area"}
    if not gt_rooms:
        return None
    present = [g for g in gt_rooms if g in pairs_ids]
    out = dict(gt_rooms=len(gt_rooms), matched_rooms=len(present), plan_rooms=len(plan["rooms"]),
               missing_rooms=[g for g in gt_rooms if g not in pairs_ids])
    if gt_area and len(gt_area) == len(gt_rooms):
        fp = _m(plan.get("footprint_area"))
        tot = sum(gt_area.values())
        out["footprint"] = dict(gt=round(tot, 3), value=fp[0], rel_err=None if fp[0] is None else round((fp[0] - tot) / tot, 4),
                                inside=None if fp[1] is None else bool(fp[1] <= tot <= fp[2]))
    adj_gt = set()
    import re
    for it in gt:
        if it.kind == "door_width":
            m = re.search(r"rooms ([^;]+)", it.notes or "")
            if m and "+" in m.group(1):
                a, b = m.group(1).strip().split("+")[:2]
                adj_gt.add(frozenset((a.strip(), b.strip())))
    if adj_gt:
        inv = {v: k for k, v in pairs_ids.items()}
        adj_pred = set()
        for a in plan.get("adjacency", []):
            ga, gb = inv.get(a.get("room_a")), inv.get(a.get("room_b"))
            if ga and gb and ga != gb:
                adj_pred.add(frozenset((ga, gb)))
        tp = len(adj_gt & adj_pred)
        out["adjacency"] = dict(gt=len(adj_gt), found=tp, wrong=len(adj_pred - adj_gt),
                                missing=sorted("+".join(sorted(x)) for x in adj_gt - adj_pred),
                                extra=sorted("+".join(sorted(x)) for x in adj_pred - adj_gt))
    ov = _overlaps(plan)
    if ov is not None:
        out["overlaps"] = ov
    fp_ok = out.get("footprint", {}).get("rel_err")
    out["passed"] = bool(not out["missing_rooms"] and (fp_ok is not None and abs(fp_ok) <= GATES["footprint_rel_tol_photo"])
                         and (not adj_gt or (out["adjacency"]["found"] == len(adj_gt) and out["adjacency"]["wrong"] == 0))
                         and not ov)
    return out


def _damage_class(text: str) -> str | None:
    t = (text or "").lower()
    if "stain" in t or "water" in t:
        return "water_stain"
    if "crack" in t:
        return "crack"
    return None


def evaluate_damage(plan: dict, gt: list[GTItem], room_pairs: dict[str, str]) -> list[dict]:
    """Staged damage vs tape GT (D-015 extension): per GT decal, the best same-room, same-class detection by size.

    GT rows: item_type damage_width / damage_height (metres), item_id DMGn, and the class in the description or notes
    ("water stain" / "crack"). Detected extents are width (along the surface) and height (vertical). Confirmed
    detections with no GT decal are reported as false positives; review-tier ones are listed separately."""
    decals: dict[tuple[str, str], dict] = {}
    for it in gt:
        if it.kind in ("damage_width", "damage_height"):
            d = decals.setdefault((it.room, it.item), dict(room=it.room, item=it.item, cls=None))
            d[it.kind.split("_")[1]] = it.value
            d["cls"] = d["cls"] or _damage_class(it.notes) or _damage_class(it.item)
    dets = [d for d in plan.get("damage", []) if not d.get("concealed")]
    used, rows = set(), []
    for (room, item), g in decals.items():
        prid = room_pairs.get(room)
        cands = [d for d in dets if d.get("room_id") == prid and (g["cls"] is None or d.get("class") == g["cls"])
                 and d["id"] not in used]
        def size_err(d):
            e = d.get("extent", {})
            return sum(abs((e.get(k) or {}).get("value", 0) - g.get(k, 0)) for k in ("width", "height") if k in g)
        best = min(cands, key=size_err) if cands else None
        row = dict(room=room, item=item, cls=g["cls"], gt_width=g.get("width"), gt_height=g.get("height"),
                   detected=best is not None)
        if best is not None:
            used.add(best["id"])
            e = best.get("extent", {})
            for k in ("width", "height"):
                m = e.get(k) or {}
                v, lo, hi = m.get("value"), m.get("lo", (m.get("ci95") or [None, None])[0]), m.get("hi", (m.get("ci95") or [None, None])[1])
                row[f"pred_{k}"], row[f"err_{k}"] = v, (None if v is None or g.get(k) is None else v - g[k])
                row[f"inside_{k}"] = None if lo is None or g.get(k) is None else bool(lo <= g[k] <= hi)
            row.update(det_id=best["id"], status=best.get("status"), confidence=best.get("confidence"))
        rows.append(row)
    for d in dets:
        if d["id"] not in used:
            rows.append(dict(room=None, item="-", cls=d.get("class"), detected=True, det_id=d["id"],
                             status=d.get("status"), confidence=d.get("confidence"),
                             note="FALSE POSITIVE (no staged decal)" if d.get("status") == "confirmed" else "review-tier, no decal"))
    return rows


def _wall_stats(walls: list[dict], tol) -> dict:
    """Medians over the paired GT walls, reported with their coverage (paired / all GT walls); fractions over ALL GT
    walls (a MISSED or MERGED GT wall counts as a fail)."""
    ok = [r for r in walls if r["rel"] is not None]
    out = {}
    if ok:
        rel, ae = np.array([r["rel"] for r in ok]), np.array([abs(r["err"]) for r in ok])
        out = dict(wall_median_rel_err=float(np.median(rel)), wall_median_abs_err_m=float(np.median(ae)),
                   wall_coverage=len(ok) / len(walls),
                   wall_within_2cm_frac=float(np.sum(ae <= 0.02) / len(walls)), walls_gt=len(walls),
                   walls_missed=len(walls) - len(ok))
        if tol:
            out["wall_gate"] = dict(tol=tol, pass_frac=float(np.sum(rel <= tol) / len(walls)),
                                    passed=bool(len(ok) == len(walls) and np.all(rel <= tol)),
                                    n=len(walls), missed=len(walls) - len(ok))
    return out


def gate_summary(rows, tier):
    tol = GATES["wall_rel_tol"].get(str(tier).split("_")[0])   # "video_take2" -> video gate
    out = _wall_stats([r for r in rows if r["kind"] == "wall"], tol)
    ops = [r for r in rows if r["kind"] == "door_width"]
    if ops:
        ok = [r for r in ops if r["err"] is not None and abs(r["err"]) <= GATES["opening_abs_tol_m"]]
        frac = len(ok) / len(ops)                      # missed and phantom openings count as misses
        out["opening_gate"] = dict(tol_m=0.02, pass_frac=frac, passed=bool(frac >= GATES["opening_pass_frac"]),
                                   n=len(ops), missed=sum(1 for r in ops if "MISSED" in r["note"]),
                                   phantom=sum(1 for r in ops if "PHANTOM" in r["note"]))
    ceil = [r for r in rows if r["kind"] == "ceiling_height"]
    if ceil:
        ok = [r for r in ceil if r["err"] is not None and abs(r["err"]) <= GATES["ceiling_abs_tol_m"]]
        out["ceiling_gate"] = dict(tol_m=0.015, pass_frac=len(ok) / len(ceil), passed=len(ok) == len(ceil),
                                   not_observed=sum(1 for r in ceil if r["value"] is None))
    cal = [r for r in rows if r["inside"] is not None]
    if cal:
        out["calibration"] = dict(coverage_95=float(np.mean([r["inside"] for r in cal])), n=len(cal),
                                  note="fraction of GT values inside the reported 95% intervals; ~0.95 is calibrated, "
                                       "much lower = over-confident, ~1.0 with huge intervals = uninformative")
    return out


def to_markdown(res: dict) -> str:
    lines = [f"### Tier: {res['tier']}", "", "| Room | Item | Pred | 95% CI | GT | Error | Rel | In CI | Note |",
             "|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"
    for r in res["rows"]:
        ci = "—" if r["lo"] is None else f"[{r['lo']:.3f}, {r['hi']:.3f}]"
        rel = "—" if r["rel"] is None else f"{100 * r['rel']:.1f}%"
        lines.append(f"| {r['room']} | {r['item']} ({r['kind']}) | {f(r['value'])} | {ci} | {f(r['gt'])} | {f(r['err'])} | "
                     f"{rel} | {'—' if r['inside'] is None else ('yes' if r['inside'] else 'NO')} | {r['note']} |")
    gs = res.get("gates", {})
    if gs.get("walls_gt"):
        gate = gs.get("wall_gate")
        lines += ["", f"**Walls:** median rel. error {100 * gs['wall_median_rel_err']:.1f}% over "
                      f"{gs['walls_gt'] - gs['walls_missed']} of {gs['walls_gt']} GT walls (coverage "
                      f"{100 * gs['wall_coverage']:.0f}%)"
                      + (f"; within {100 * gate['tol']:.0f}%: {100 * gate['pass_frac']:.0f}% of all GT walls (MISSED "
                         f"and MERGED walls count as failures)" if gate else "")]
    if res.get("wall_pairing"):
        lines += ["", "**Wall pairing** (default; unmatched = predicted walls on no GT wall's line):", "",
                  "| Room | Method | Frame / order | IoU | Missed GT walls | Unmatched pred (n, m) | Unmatched >= 0.35 m |",
                  "|---|---|---|---|---|---|---|"]
        for g, p in res["wall_pairing"].items():
            fr = (f"{p['frame']}, rot {p['rot_deg']:.1f} deg" if str(p["method"]).startswith("geometric")
                  else f"{p['how']}, cost {p['cost']}" if p["method"] in ("anchored", "constrained order")
                  else f"{'clockwise' if p.get('direction') == 1 else 'REVERSED'} start {p.get('start')}")
            if p.get("outline"):
                o = p["outline"]
                fr += (f"; tape outline {o['status']}" + (f" ({o['n_outlines']})" if o["n_outlines"] > 1 else "")
                       + (f": {o['reason']}" if o.get("reason") else ""))
            if p.get("merged"):
                fr += "; merged " + ", ".join(f"{'+'.join(v)} -> {k}" for k, v in p["merged"].items())
            if p.get("ambiguity"):
                fr += f" (AMBIGUOUS: {p['ambiguity']})"
            lines.append(f"| {g} | {p['method']} | {fr} | {p.get('iou', '—')} | {', '.join(p['missed']) or '—'} | "
                         f"{p['unmatched_pred_n']}, {p['unmatched_pred_len_m']:.3f} | "
                         f"{', '.join(f'{i} ({v:.2f})' for i, v in p['unmatched_pred'] if v >= wm.PARAMS['short_m']) or '—'} |")
        warn = [f"WARNING: {g}: {w}" for g, p in res["wall_pairing"].items() for w in p.get("warnings", [])]
        warn += [f"WARNING: {g}: ambiguous wall pairing: {p['ambiguity']}" for g, p in res["wall_pairing"].items()
                 if p.get("ambiguity") and "warnings" not in p]
        warn += [f"WARNING: {g}: {p['reverse_note']}" for g, p in res["wall_pairing"].items() if p.get("reverse_note")]
        if warn:
            lines += [""] + warn
        ob = res.get("wall_order_based", {}).get("summary", {})
        if ob:
            lines += ["", f"Old order-based pairing (legacy, not gated): median |err| "
                          f"{100 * ob['wall_median_abs_err_m']:.1f} cm, median rel {100 * ob['wall_median_rel_err']:.1f}%, "
                          f"within 2 cm {100 * ob['wall_within_2cm_frac']:.0f}%"
                          + (f", within {100 * ob['wall_gate']['tol']:.0f}% {100 * ob['wall_gate']['pass_frac']:.0f}%"
                             if ob.get("wall_gate") else "")
                          + " (rows under 'wall_order_based' in the JSON)."]
    lines += ["", "**Gates:**", "```json", json.dumps(res["gates"], indent=1), "```"]
    if res.get("damage"):
        lines += ["", "**Staged damage vs tape:**", "", "| Room | Item | Class | Detected | Status | Conf | W pred/GT | H pred/GT | In CI (W,H) | Note |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for d in res["damage"]:
            fw = lambda a, b: "—" if a is None or b is None else f"{a:.3f}/{b:.3f}"  # noqa: E731
            lines.append(f"| {d.get('room')} | {d.get('item')} | {d.get('cls')} | {d.get('detected')} | {d.get('status')} | "
                         f"{'—' if d.get('confidence') is None else round(d['confidence'], 2)} | "
                         f"{fw(d.get('pred_width'), d.get('gt_width'))} | {fw(d.get('pred_height'), d.get('gt_height'))} | "
                         f"{d.get('inside_width')},{d.get('inside_height')} | {d.get('note', '')} |")
    return "\n".join(lines)
