"""Doors from priors where the door itself is not seen (decision D-085).

Why: the user's rule (5 Oct): "where we cannot identify the door itself we might need to use door priors, and based
on the gap assume that there must be a door". The segmenter finds a door only when its leaf is in frame. A doorway
photo and a walk-through stand IN the doorway and see no leaf, and the plan step's gaps become doors only where a
leaf was seen. But a room the photographer walked out of has a door there, whatever the pixels say.

Four rules. Every door they make or change is marked "source": "prior", says why in its evidence, and has a confidence
under 0.5 (dashed in the technical drawing, a door in the presentation drawing):
  a. A plan-step gap (a geometric passage) 0.55-1.15 m wide with wall on both sides is a door. Its width is the
     gap's, as measured. Wider gaps stay open passages; narrower ones are not doors.
  b. Photo tier: a doorway-pair photo stands on a threshold. The wall of its own room within 0.6 m of the camera,
     with the camera looking away from it into the room, gets a door centred at the camera's projection on it (a
     camera more than 0.2 m off that line keeps its own position: the plan lacks the jog or alcove the door is in).
     The pair's two doors (one in each room) on the two faces of one partition are one door. This replaces the plan
     step's doorway-pair door (centre = the two cameras' mean, no width).
  c. Video tier: where the camera path goes from one plan room into another (at least 0.5 m of path in each, no pose
     jump on the way), a door at the crossing.
  d. A door the segmenter saw cut by the image frame in every view (its width only a lower bound): the prior width,
     from the end that was seen, never narrower than what was seen.
Prior width, not measured: 0.80 m, 95% interval +-0.20 m (sigma 0.10; interior doors are 0.70-0.90 m wide); 0.70 m
for a bathroom or toilet by its room name. A measured door or passage on the same wall (or on the other face of the
partition) within 0.5 m wins: the prior only adds its room link and its evidence to it.
On by default: rule d only (run_capture.py --door-priors; the others found no door the plans lacked).
Measured: docs/modules/openings.md, "Door priors"; decision D-085.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from floorplan.model import Adjacency, Measurement, Opening, Plan


@dataclass
class PriorParams:
    door_m: float = 0.80               # interior door, not measured
    bath_m: float = 0.70               # bathroom / toilet door, by the room's name
    sigma_m: float = 0.10              # 95% interval +-0.20 m: wide, the width is not measured
    gap_door_m: tuple = (0.55, 1.15)   # (a) a gap this wide with wall on both sides is a door
    jamb_m: float = 0.10               # (a) wall on both sides: the host wall goes on this far past each end
    threshold_m: float = 0.6           # (b) a doorway photo's camera this close to its room's wall
    facing_cos: float = 0.5            # (b) looking into the room: view . wall normal >= this (60 deg)
    snap_m: float = 0.2                # (b) a camera farther than this from the wall line keeps its own position
    min_wall_m: float = 0.3            # (b, c) shorter walls (jogs) do not hold a door
    partition_m: float = 0.45          # walls of two rooms this close and parallel are the faces of one partition
    merge_m: float = 0.5               # a measured opening this close along the wall wins
    parallel_cos: float = 0.95
    path_gap_m: float = 1.5            # (c) last point in one room to first point in the next: at most this far
    dwell_m: float = 0.5               # (c) a stay in a room shorter than this much path is jitter, not a walk
    jump_m: float = 0.3                # (c) a step this long between two frames is a pose jump, not a walk
    path_wall_m: float = 0.5           # (c) the crossing within this of the room's wall
    cluster_m: float = 0.6             # (c) crossings of one wall this close are one door
    conf_gap: float = 0.45
    conf_photo: float = 0.4
    conf_path: float = 0.4
    conf_cut: float = 0.45

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


_BATH = re.compile(r"bath|toilet|\bwc\b|washroom|lavatory", re.I)


class _Line:
    def __init__(self, w):
        self.wall = w
        self.p0, p1 = np.asarray(w.p0, float), np.asarray(w.p1, float)
        self.L = float(np.linalg.norm(p1 - self.p0))
        self.e = (p1 - self.p0) / max(self.L, 1e-9)
        n = np.asarray(w.normal, float)
        self.n = n / max(float(np.linalg.norm(n)), 1e-9)

    def t(self, c) -> float:
        return float((np.asarray(c, float) - self.p0) @ self.e)

    def s(self, c) -> float:
        return float((np.asarray(c, float) - self.p0) @ self.n)

    def at(self, t: float) -> np.ndarray:
        return self.p0 + self.e * t


def _lines(plan: Plan) -> dict[str, _Line]:
    return {w.id: _Line(w) for w in plan.walls if np.linalg.norm(np.subtract(w.p1, w.p0)) > 1e-6}


def _room_text(plan: Plan, rid: str) -> str:
    r = next((r for r in plan.rooms if r.id == rid), None)
    return "" if r is None else " ".join(str(x) for x in (r.label, r.name, r.room_type) if x)


def prior_width(plan: Plan, room_ids, p: PriorParams, how: str) -> Measurement:
    """The standard-door width for a door of these rooms: 0.70 m into a bathroom or toilet, else 0.80 m."""
    bath = [r for r in room_ids if _BATH.search(_room_text(plan, r))]
    v = p.bath_m if bath else p.door_m
    why = (f"bathroom door prior {v:.2f} m ({_room_text(plan, bath[0])})" if bath
           else f"standard door prior {v:.2f} m")
    return Measurement.from_sigma(v, p.sigma_m, method=f"{why}, not measured: {how}", status="inferred")


def _clip_centre(ln: _Line, t: float, width: float) -> float:
    """Keep a door of this width inside its wall (5 cm from each end) when the wall is long enough."""
    lo, hi = width / 2 + 0.05, ln.L - width / 2 - 0.05
    return float(np.clip(t, lo, hi)) if hi >= lo else ln.L / 2


def _near_opening(plan: Plan, lines, c, ln: _Line, p: PriorParams, measured_only: bool = True):
    """A door or passage on this wall line, or on the other face of the partition, whose centre is within
    max(merge_m, half its width) of c along the wall: a measured one (measured_only), else a door from a prior."""
    for o in plan.openings:
        if o.kind not in ("door", "passage"):
            continue
        has_w = o.width is not None and o.width.value is not None
        if (not has_w or o.source == "prior") if measured_only else (o.source != "prior"):
            continue
        d = np.asarray(o.center, float) - np.asarray(c, float)
        if abs(float(d @ ln.n)) > p.partition_m:
            continue
        hosts = [lines[i] for i in o.wall_ids if i in lines]
        if hosts and not any(abs(float(h.e @ ln.e)) >= p.parallel_cos for h in hosts):
            continue
        if abs(float(d @ ln.e)) <= max(p.merge_m, (o.width.value / 2) if has_w else 0.0):
            return o
    return None


def _link(plan: Plan, o: Opening, room_ids, wall_ids, note: str) -> None:
    """A measured opening wins over a prior one: it gets the prior's rooms (and their adjacency) and a note."""
    rooms = {r.id for r in plan.rooms}
    for r in room_ids:
        if r not in o.room_ids and r in rooms:
            o.room_ids.append(r)
    walls = {w.id: w for w in plan.walls}
    for i in wall_ids:
        if i not in o.wall_ids and i in walls:
            o.wall_ids.append(i)
            if o.id not in walls[i].opening_ids:
                walls[i].opening_ids.append(o.id)
    _adjacency(plan, o)
    o.evidence = (o.evidence or "") + f"; {note}"


def _adjacency(plan: Plan, o: Opening) -> None:
    rs = [r for r in o.room_ids if r in {x.id for x in plan.rooms}]
    if len(rs) >= 2 and not any({a.room_a, a.room_b} == {rs[0], rs[1]} for a in plan.adjacency):
        plan.adjacency.append(Adjacency(rs[0], rs[1], o.id))


def _new_door(plan: Plan, centre, wall_ids, room_ids, width: Measurement, conf: float, evidence: str) -> Opening:
    ids = {o.id for o in plan.openings}
    k = 1
    while f"P{k}" in ids:
        k += 1
    o = Opening(id=f"P{k}", kind="door", wall_ids=list(wall_ids), room_ids=list(room_ids),
                center=(float(centre[0]), float(centre[1])), width=width, confidence=conf, evidence=evidence,
                source="prior")
    plan.openings.append(o)
    for w in plan.walls:
        if w.id in o.wall_ids and o.id not in w.opening_ids:
            w.opening_ids.append(o.id)
    _adjacency(plan, o)
    return o


def _partner(plan: Plan, lines, ln: _Line, t: float, width: float, room: str, p: PriorParams):
    """The wall of another room on the other face of the partition at this place, if any."""
    c = ln.at(t)
    best = None
    for other in lines.values():
        if other.wall.room_id == room or abs(float(other.e @ ln.e)) < p.parallel_cos or other.L < p.min_wall_m:
            continue
        if abs(other.s(c)) > p.partition_m:
            continue
        to = other.t(c)
        if -0.05 <= to <= other.L + 0.05 and (best is None or abs(other.s(c)) < abs(best.s(c))):
            best = other
    return best


# ------------------------------------------------------------------------------------------------ (a) gaps
def _walls_both_sides(lines, o: Opening, p: PriorParams) -> bool:
    w = o.width.value
    for i in o.wall_ids:
        ln = lines.get(i)
        if ln is None:
            continue
        tc = ln.t(o.center)
        if abs(ln.s(o.center)) <= 0.35 and tc - w / 2 >= p.jamb_m and tc + w / 2 <= ln.L - p.jamb_m:
            return True
    return False


def gap_doors(plan: Plan, p: PriorParams) -> list[dict]:
    """(a) Geometric passages 0.55-1.15 m wide with wall on both sides become doors (width = the gap)."""
    lines, out = _lines(plan), []
    for o in plan.openings:
        if o.kind != "passage" or (o.source or "geometry") != "geometry" or o.width is None or o.width.value is None:
            continue
        w = float(o.width.value)
        if not p.gap_door_m[0] <= w <= p.gap_door_m[1]:
            continue
        flag = "wall evidence on both sides: True" in (o.evidence or "")
        if not (flag or _walls_both_sides(lines, o, p)):
            out.append(dict(opening_id=o.id, rule="gap", action="kept as passage", width=round(w, 3),
                            why="no wall on both sides"))
            continue
        o.kind, o.source = "door", "prior"
        o.confidence = min(o.confidence, p.conf_gap)
        o.evidence = (o.evidence or "") + (f"; a door assumed (D-085): a gap of {w:.2f} m, within "
                                           f"{p.gap_door_m[0]}-{p.gap_door_m[1]} m, with wall on both sides; "
                                           f"width = the gap")
        out.append(dict(opening_id=o.id, rule="gap", action="passage -> door", width=round(w, 3),
                        rooms=list(o.room_ids)))
    return out


# ------------------------------------------------------------------------------------------------ (b) photos
def _threshold_wall(lines, room: str, C: np.ndarray, f: np.ndarray, p: PriorParams):
    best = None
    for ln in lines.values():
        if ln.wall.room_id != room or ln.L < p.min_wall_m:
            continue
        s, t, look = ln.s(C), ln.t(C), float(f @ ln.n)
        if abs(s) > p.threshold_m or not -0.15 <= t <= ln.L + 0.15 or look < p.facing_cos:
            continue
        key = (abs(s), -look)
        if best is None or key < best[0]:
            best = (key, ln, t, s, look)
    return best


def photo_threshold_doors(plan: Plan, scene: dict, info: dict, p: PriorParams) -> list[dict]:
    """(b) A door where each doorway-pair photo stands, on its own room's wall."""
    names = [str(x) for x in np.asarray(scene.get("cam_names", []))]
    pairs = (info.get("protocol_v2") or {}).get("doorway_pairs") or []
    if not names or not pairs:
        return []
    T = np.asarray(scene["T_wc"], float)
    pose = {}
    for i, n in enumerate(names):
        f = T[i][:3, 2][[0, 2]]
        pose[n] = (T[i][:3, 3][[0, 2]], f / max(float(np.linalg.norm(f)), 1e-9))
    room_of = {r.label: r.id for r in plan.rooms}
    lines, out = _lines(plan), []
    for pr in pairs:
        a, b = pr["a"], pr["b"]
        old = next((o for o in plan.openings if o.source != "prior" and (o.width is None or o.width.value is None)
                    and f"doorway pair {a} ~ {b}" in (o.evidence or "")), None)
        found = []
        for n, folder in zip((a, b), pr["rooms"]):
            rid = room_of.get(folder)
            if rid is None or n not in pose:
                continue
            hit = _threshold_wall(lines, rid, *pose[n], p)
            if hit is not None:
                found.append(dict(photo=n, room=rid, line=hit[1], t=hit[2], s=hit[3], look=hit[4], C=pose[n][0]))
        rooms = [room_of.get(f) for f in pr["rooms"] if room_of.get(f)]
        if not found:
            rec = dict(rule="photo", pair=f"{a} ~ {b}", action="no wall within "
                       f"{p.threshold_m} m of either camera that it looks away from")
            if old is not None:                  # keep the plan step's door, with the prior width
                old.width = prior_width(plan, old.room_ids, p, "doorway pair; centre = the two cameras' mean")
                old.source, old.confidence = "prior", min(old.confidence, p.conf_photo)
                old.evidence = (old.evidence or "") + "; width from the standard-door prior (D-085)"
                rec.update(action=rec["action"] + "; the plan step's pair door keeps its place, prior width",
                           opening_id=old.id)
            out.append(rec)
            continue
        # one door per partition: the pair's two doors on the two faces of one wall are the same door
        groups = [[found[0]]]
        for d in found[1:]:
            g0 = groups[0][0]
            same = (abs(float(d["line"].e @ g0["line"].e)) >= p.parallel_cos
                    and abs(g0["line"].s(d["line"].at(d["t"]))) <= p.partition_m
                    and abs(g0["line"].t(d["line"].at(d["t"])) - g0["t"]) <= p.merge_m)
            if same:
                groups[0].append(d)
            else:
                groups.append([d])
        new_ids = []
        for g in groups:
            ln = g[0]["line"]
            t = float(np.mean([ln.t(x["line"].at(x["t"])) for x in g]))
            room_ids = [x["room"] for x in g]
            wall_ids = [x["line"].wall.id for x in g]
            width = prior_width(plan, room_ids if len(g) > 1 else rooms, p,
                                "a doorway photo stands in the doorway and sees no door leaf")
            t = _clip_centre(ln, t, width.value)
            if len(g) == 1:                       # the other face of this partition, if the plan has one there
                other = _partner(plan, lines, ln, t, width.value, ln.wall.room_id, p)
                if other is not None:
                    wall_ids.append(other.wall.id)
                    room_ids.append(other.wall.room_id)
            far = max(abs(x["s"]) for x in g)
            if far > p.snap_m:
                # the camera stood in the doorway, but the plan's wall is not there (a jog or an alcove the plan
                # lacks, or the wall is off): projecting would move the door along the true wall; it keeps the
                # camera's place and is drawn on the wall
                c = np.mean([x["C"] for x in g], axis=0)
                how = f"centre = the camera position ({far:.2f} m from the wall line; drawn on the wall)"
            else:
                c = ln.at(t)
                how = "centre = the camera's projection on the wall"
            photos = ", ".join(x["photo"] for x in g)
            dists = ", ".join("%.2f" % abs(x["s"]) for x in g)
            hosts = ", ".join(x["line"].wall.id for x in g)
            ev = (f"door prior (D-085): doorway photo(s) {photos} stand on the threshold, {dists} m from {hosts}, "
                  f"looking into the room; {how}")
            meas = _near_opening(plan, lines, c, ln, p)
            if meas is not None:
                _link(plan, meas, room_ids, wall_ids, f"a doorway photo stands here ({g[0]['photo']}); the "
                                                       f"measured {meas.kind} wins over the door prior")
                out.append(dict(rule="photo", pair=f"{a} ~ {b}", action="measured opening wins", opening_id=meas.id,
                                photos=[x["photo"] for x in g]))
                new_ids.append(meas.id)
                continue
            prev = _near_opening(plan, lines, c, ln, p, measured_only=False)
            if prev is not None:
                _link(plan, prev, room_ids, wall_ids, f"also doorway photo {g[0]['photo']}")
                new_ids.append(prev.id)
                continue
            o = _new_door(plan, c, wall_ids, room_ids, width, p.conf_photo, ev)
            new_ids.append(o.id)
            out.append(dict(rule="photo", pair=f"{a} ~ {b}", action="added", opening_id=o.id, rooms=room_ids,
                            walls=wall_ids, width=width.value, photos=[x["photo"] for x in g],
                            dist_m=[round(abs(x["s"]), 3) for x in g]))
        if old is not None:                      # the plan step's widthless pair door is replaced
            _remove(plan, old, new_ids[0] if new_ids else None)
            out.append(dict(rule="photo", pair=f"{a} ~ {b}", action="replaced the plan step's pair door",
                            opening_id=old.id, by=new_ids))
    return out


def _remove(plan: Plan, o: Opening, via: str | None) -> None:
    plan.openings = [x for x in plan.openings if x is not o]
    for w in plan.walls:
        if o.id in w.opening_ids:
            w.opening_ids.remove(o.id)
    keep = []
    for adj in plan.adjacency:
        if adj.via == o.id:
            if via is None:
                adj.via = "shared_wall"
            else:
                adj.via = via
        keep.append(adj)
    plan.adjacency = keep


# ------------------------------------------------------------------------------------------------ (c) video path
def _room_index(plan: Plan, P: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as MplPath
    idx = np.full(len(P), -1, int)
    for k, r in enumerate(plan.rooms):
        if len(r.polygon) < 3:
            continue
        inside = MplPath(np.asarray(r.polygon, float)).contains_points(P) & (idx < 0)
        idx[inside] = k
    return idx


def path_crossing_doors(plan: Plan, scene: dict, p: PriorParams) -> list[dict]:
    """(c) A door where the camera path goes from one plan room into another."""
    traj = scene.get("traj")
    if traj is None or len(plan.rooms) < 2:
        return []
    P = np.asarray(traj, float)[:, [0, 2]]
    idx = _room_index(plan, P)
    inside = np.flatnonzero(idx >= 0)
    if len(inside) < 2:
        return []
    step = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    # stays: runs of frames in one room; a stay shorter than dwell_m of path is jitter on a room boundary (or a pose
    # jump) and is dropped, so only walks from one room well into the next count
    runs: list[list[int]] = []
    for k in inside:
        if runs and runs[-1][0] == idx[k]:
            runs[-1][2] = int(k)
        else:
            runs.append([int(idx[k]), int(k), int(k)])
    runs = [r for r in runs if step[r[2]] - step[r[1]] >= p.dwell_m]
    stays: list[list[int]] = []
    for r in runs:
        if stays and stays[-1][0] == r[0]:
            stays[-1][2] = r[2]
        else:
            stays.append(list(r))
    lines, cross = _lines(plan), []
    for r0, r1 in zip(stays[:-1], stays[1:]):
        i, j = r0[2], r1[1]
        if step[j] - step[i] > p.path_gap_m or float(np.max(np.diff(step[i:j + 1]), initial=0.0)) > p.jump_m:
            continue                              # too long a way between the rooms, or a pose jump on it
        ra, rb = plan.rooms[r0[0]].id, plan.rooms[r1[0]].id
        mid = (P[i] + P[j]) / 2
        best = None
        for ln in lines.values():                 # the wall of the room left behind, between the two points
            if ln.wall.room_id != ra or ln.L < p.min_wall_m:
                continue
            s, t = ln.s(mid), ln.t(mid)
            if abs(s) > p.path_wall_m or not -0.1 <= t <= ln.L + 0.1:
                continue
            if ln.s(P[i]) * ln.s(P[j]) > 0 and min(abs(ln.s(P[i])), abs(ln.s(P[j]))) > 0.15:
                continue                          # both points on one side of it: the path did not cross it
            if best is None or abs(s) < best[0]:
                best = (abs(s), ln, t)
        if best is not None:
            cross.append(dict(a=ra, b=rb, line=best[1], t=best[2], frame=int(j), at=best[1].at(best[2])))
    out, done = [], []
    for c in cross:                               # one door per room pair and place: crossings within cluster_m
        pair = frozenset((c["a"], c["b"]))
        grp = next((g for g in done if g["pair"] == pair
                    and float(np.linalg.norm(np.median(g["at"], axis=0) - c["at"])) <= p.cluster_m), None)
        if grp is None:
            done.append(dict(pair=pair, line=c["line"], t=[c["t"]], at=[c["at"]], frames=[c["frame"]],
                             rooms=[c["a"], c["b"]]))
        else:
            grp["at"].append(c["at"])
            grp["frames"].append(c["frame"])
            if grp["line"] is c["line"]:
                grp["t"].append(c["t"])
    for g in done:
        ln = g["line"]
        width = prior_width(plan, g["rooms"], p, "the camera walked through here; no door leaf was measured")
        t = _clip_centre(ln, float(np.median(g["t"])), width.value)
        c = ln.at(t)
        wall_ids = [ln.wall.id]
        other = _partner(plan, lines, ln, t, width.value, ln.wall.room_id, p)
        if other is not None and other.wall.room_id in g["rooms"]:
            wall_ids.append(other.wall.id)
        meas = _near_opening(plan, lines, c, ln, p)
        rec = dict(rule="path", rooms=g["rooms"], wall=ln.wall.id, crossings=len(g["frames"]),
                   frames=g["frames"][:6])
        if meas is not None:
            _link(plan, meas, g["rooms"], wall_ids, f"the camera path crosses here ({len(g['frames'])} time(s)); the "
                                                    f"measured {meas.kind} wins over the door prior")
            out.append(dict(rec, action="measured opening wins", opening_id=meas.id))
            continue
        prev = _near_opening(plan, lines, c, ln, p, measured_only=False)
        if prev is not None:
            _link(plan, prev, g["rooms"], wall_ids, "the camera path crosses here too")
            out.append(dict(rec, action="joined a prior door", opening_id=prev.id))
            continue
        ev = (f"door prior (D-085): the camera path crosses from {g['rooms'][0]} into {g['rooms'][1]} here "
              f"({len(g['frames'])} time(s), frames {', '.join(str(f) for f in g['frames'][:4])}); centre = the crossing")
        o = _new_door(plan, c, wall_ids, g["rooms"], width, p.conf_path, ev)
        out.append(dict(rec, action="added", opening_id=o.id, width=width.value))
    return out


# ------------------------------------------------------------------------------------------------ (d) cut doors
def cut_door_widths(plan: Plan, p: PriorParams) -> list[dict]:
    """(d) A segmenter door whose width is a lower bound (an end cut by the frame in every view): the prior width,
    from the end that was seen, never narrower than what was seen."""
    meta = {d.get("opening_id"): d for d in (plan.meta.get("semantic_openings") or {}).get("openings", [])}
    lines, out = _lines(plan), []
    for o in plan.openings:
        if (o.source not in ("segmentation", "see_through") or o.kind != "door" or o.width is None
                or o.width.value is None
                or o.width.status != "inferred"):
            continue
        rec = meta.get(o.id) or {}
        ln = lines.get(rec.get("host_id")) or next((lines[i] for i in o.wall_ids if i in lines), None)
        if ln is None:
            continue
        seen = float(o.width.value)
        prior = prior_width(plan, o.room_ids, p, "the segmenter saw this door cut by the image frame"
                            if o.source == "segmentation" else "seen through, one jamb never seen")
        if seen >= prior.value:
            out.append(dict(opening_id=o.id, rule="cut", action="kept: the seen part is as wide as the prior",
                            width=round(seen, 3)))
            continue
        if "t0" in rec and "t1" in rec:
            t0, t1 = float(rec["t0"]), float(rec["t1"])
        else:
            tc = ln.t(o.center)
            t0, t1 = tc - seen / 2, tc + seen / 2
        ends = rec.get("ends_seen") or [0, 0]
        W = prior.value
        if ends[0] > 0 and not ends[1] > 0:
            a, side = t0, "from its seen end at the lower end"
        elif ends[1] > 0 and not ends[0] > 0:
            a, side = t1 - W, "from its seen end at the upper end"
        else:
            a, side = (t0 + t1) / 2 - W / 2, "about the seen part's centre"
        a = float(np.clip(a, 0.0, max(ln.L - W, 0.0)))     # stays on its wall
        c = ln.at(a + W / 2)
        o.center = (float(c[0]), float(c[1]))
        o.width = Measurement(W, max(seen, prior.lo), prior.hi, "m",
                              prior.method + f"; at least the {seen:.2f} m seen", "inferred")
        o.source, o.confidence = "prior", min(o.confidence, p.conf_cut)
        o.evidence = (o.evidence or "") + (f"; width from the door prior (D-085): {W:.2f} m {side} "
                                           f"(the frame cut the other end; {seen:.2f} m seen)")
        out.append(dict(opening_id=o.id, rule="cut", action="prior width", seen=round(seen, 3), width=W,
                        ends_seen=list(ends), host=ln.wall.id))
    return out


# ------------------------------------------------------------------------------------------------ the step
def add_prior_doors(plan: Plan, scene: dict, info: dict, tier: str, log=print, p: PriorParams | None = None,
                    rules: str = "abcd") -> dict:
    """The whole step. Never raises: on a failure the plan is left as it was up to that rule, and the report says."""
    p = p or PriorParams()
    rep = dict(status="ok", rules=rules, params=p.to_dict(), actions=[])
    try:
        if "d" in rules and tier in ("photo", "video"):
            rep["actions"] += cut_door_widths(plan, p)
        if "a" in rules:
            rep["actions"] += gap_doors(plan, p)
        if "b" in rules and tier == "photo":
            rep["actions"] += photo_threshold_doors(plan, scene, info, p)
        if "c" in rules and tier == "video":
            rep["actions"] += path_crossing_doors(plan, scene, p)
    except Exception as e:                       # never lose the plan over an add-on
        rep["status"] = f"failed: {type(e).__name__}: {e}"
        log(f"[doors] WARNING: door priors failed: {type(e).__name__}: {e}")
    rep["prior_doors"] = sorted(o.id for o in plan.openings if o.source == "prior")
    plan.meta["door_priors"] = {k: v for k, v in rep.items() if k != "params"}
    n_add = sum(a.get("action") == "added" for a in rep["actions"])
    log(f"[doors] priors: {n_add} door(s) added, {len(rep['prior_doors'])} door(s) from priors in all "
        f"({', '.join(rep['prior_doors']) or 'none'})")
    return rep
