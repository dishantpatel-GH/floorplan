"""Match the elements of two plans of the same space: rooms, then walls inside matched rooms, then openings.

Why: every comparison the gates need is "this element in plan A versus the same element in plan B" (repeatability
pair, or video/photo tier versus the LiDAR reference). A wrong match makes a good plan look bad, or a bad plan look
good, so matching is explicit, conservative and auditable:

  1. Put B in A's frame with the registration transform (eval/register.py).
  2. Rooms: polygon IoU, one-to-one assignment by the Hungarian algorithm (maximises total IoU), keep IoU >= 0.3.
  3. Local correction per matched room. Real captures drift: after one global rigid fit, parts of the apartment
     are still 5-15 cm apart (eval.md, I-2). Each matched room pair is therefore re-aligned on its own walls (2D ICP)
     before its walls are matched. This only shifts and turns the room; it never changes a length.
  4. Walls, inside each matched room pair: same direction (<= 10 deg), facing the same way, offset between the two
     wall lines <= 15 cm, and >= 50% overlap along the wall. Hungarian on a cost that mixes offset and overlap.
  5. Openings: centre distance <= 0.5 m, Hungarian on distance. The kind (door/window/passage) is reported, not
     required to agree, because a kind error is a separate failure from a detection error.
Unmatched elements are returned explicitly: for the opening gate a missed or phantom opening counts as a miss.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from floorplan.eval.register import Evidence2D, RegisterParams, apply_T2, icp_point_to_line, plan_evidence


@dataclass
class MatchParams:
    min_room_iou: float = 0.3
    local_refine: bool = True
    max_wall_angle_deg: float = 10.0
    min_normal_cos: float = 0.5
    max_wall_offset_m: float = 0.15
    min_wall_overlap: float = 0.5
    max_opening_dist_m: float = 0.5
    min_local_points: int = 100


@dataclass
class RoomPair:
    a: str
    b: str
    iou: float                    # under the global transform
    iou_local: float              # after the per-room correction
    local_T: list                 # 3x3, B (already in A's frame) -> refined position


@dataclass
class WallPair:
    a: str
    b: str
    room_a: str
    room_b: str
    offset_m: float
    angle_deg: float
    overlap: float


@dataclass
class OpeningPair:
    a: str
    b: str
    distance_m: float
    kind_a: str
    kind_b: str


@dataclass
class MatchResult:
    rooms: list[RoomPair] = field(default_factory=list)
    walls: list[WallPair] = field(default_factory=list)
    openings: list[OpeningPair] = field(default_factory=list)
    unmatched: dict = field(default_factory=dict)   # {"rooms_a": [...], "rooms_b": [...], "walls_a": ..., ...}

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------------------------- plan helpers

def as_plan_dict(plan) -> dict:
    """Accept a floorplan.model.Plan, a plan.json dict, or a path to plan.json.

    Two interval encodings exist: the dataclass form {value, lo, hi} and the published-schema form
    {value, ci95: [lo, hi]} (export/json_export.py). Both are normalised to carry lo/hi."""
    if is_dataclass(plan):
        plan = json.loads(json.dumps(asdict(plan)))
    elif isinstance(plan, (str, Path)):
        plan = json.loads(Path(plan).read_text())
    _normalise_intervals(plan)
    return plan


def _normalise_intervals(obj) -> None:
    if isinstance(obj, dict):
        if "value" in obj and "ci95" in obj and "lo" not in obj:
            obj["lo"], obj["hi"] = obj["ci95"] if obj["ci95"] else (None, None)
        for v in obj.values():
            _normalise_intervals(v)
    elif isinstance(obj, list):
        for v in obj:
            _normalise_intervals(v)


def transform_plan(plan: dict, T: np.ndarray) -> dict:
    """Copy of the plan with every plan-coordinate moved by the 3x3 rigid transform T (lengths are unchanged)."""
    out = copy.deepcopy(plan)
    R = T[:2, :2]
    for r in out.get("rooms", []):
        r["polygon"] = apply_T2(T, r["polygon"]).tolist() if len(r["polygon"]) else []
    for w in out.get("walls", []):
        w["p0"], w["p1"] = apply_T2(T, w["p0"])[0].tolist(), apply_T2(T, w["p1"])[0].tolist()
        w["normal"] = (R @ np.asarray(w["normal"], float)).tolist()
    for o in out.get("openings", []):
        o["center"] = apply_T2(T, o["center"])[0].tolist()
    return out


def _poly(room: dict) -> Polygon:
    p = Polygon(room["polygon"]) if len(room.get("polygon", [])) >= 3 else Polygon()
    return p if p.is_valid else p.buffer(0)


def _iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    return float(inter / max(a.union(b).area, 1e-12))


def _hungarian(cost: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """Min-cost one-to-one assignment restricted to valid pairs."""
    if cost.size == 0 or not valid.any():
        return []
    big = cost[valid].max() + 1e6
    ri, ci = linear_sum_assignment(np.where(valid, cost, big))
    return [(int(i), int(j)) for i, j in zip(ri, ci) if valid[i, j]]


# ----------------------------------------------------------------------------------------------- rooms

def match_rooms(A: dict, B: dict, p: MatchParams) -> list[tuple[int, int, float]]:
    pa, pb = [_poly(r) for r in A["rooms"]], [_poly(r) for r in B["rooms"]]
    iou = np.array([[_iou(x, y) for y in pb] for x in pa]).reshape(len(pa), len(pb))
    pairs = _hungarian(-iou, iou >= p.min_room_iou)
    return [(i, j, float(iou[i, j])) for i, j in pairs]


def _room_walls(plan: dict, room_id: str) -> list[dict]:
    return [w for w in plan["walls"] if w.get("room_id") == room_id]


def local_room_transform(A: dict, B: dict, ra: dict, rb: dict, p: MatchParams) -> np.ndarray:
    """Small rigid correction aligning room rb's walls onto room ra's walls (both already in A's frame)."""
    ea = plan_evidence({"walls": _room_walls(A, ra["id"])})
    eb = plan_evidence({"walls": _room_walls(B, rb["id"])})
    if len(ea.wall_pts) < p.min_local_points or len(eb.wall_pts) < p.min_local_points:
        return np.eye(3)
    rp = RegisterParams(icp_max_dist_m=(0.3, 0.15, 0.08, 0.05, 0.03), icp_normal_cos=0.8, huber_m=0.01)
    T, r, _, n = icp_point_to_line(ea, Evidence2D("b", eb.wall_pts, eb.wall_normals, np.zeros((0, 2))),
                                   np.eye(3), rp)
    # accept the correction only if most of the room's walls found a partner (otherwise keep the global fit)
    return T if len(r) >= 0.5 * n else np.eye(3)


# ----------------------------------------------------------------------------------------------- walls

def wall_relation(wa: dict, wb: dict) -> tuple[float, float, float, float]:
    """(angle_deg, normal_cos, offset_m, overlap_fraction) between two wall segments in the same frame."""
    a0, a1 = np.asarray(wa["p0"], float), np.asarray(wa["p1"], float)
    b0, b1 = np.asarray(wb["p0"], float), np.asarray(wb["p1"], float)
    da, db = a1 - a0, b1 - b0
    la, lb = np.linalg.norm(da), np.linalg.norm(db)
    if la < 1e-9 or lb < 1e-9:
        return 180.0, -1.0, np.inf, 0.0
    da, db = da / la, db / lb
    angle = float(np.degrees(np.arccos(np.clip(abs(da @ db), 0, 1))))
    ncos = float(np.dot(wa["normal"], wb["normal"]))
    na = np.array([-da[1], da[0]])
    offset = float(abs(na @ ((b0 + b1) / 2 - a0)))
    s = sorted([(b0 - a0) @ da, (b1 - a0) @ da])
    overlap = max(0.0, min(la, s[1]) - max(0.0, s[0])) / min(la, lb)
    return angle, ncos, offset, float(overlap)


def match_walls(wa_list: list[dict], wb_list: list[dict], p: MatchParams) -> list[tuple[int, int, tuple]]:
    n, m = len(wa_list), len(wb_list)
    cost, valid = np.zeros((n, m)), np.zeros((n, m), bool)
    rel = {}
    for i, wa in enumerate(wa_list):
        for j, wb in enumerate(wb_list):
            ang, ncos, off, ov = wall_relation(wa, wb)
            rel[i, j] = (ang, ncos, off, ov)
            valid[i, j] = (ang <= p.max_wall_angle_deg and ncos >= p.min_normal_cos
                           and off <= p.max_wall_offset_m and ov >= p.min_wall_overlap)
            cost[i, j] = off / p.max_wall_offset_m + (1 - ov)
    return [(i, j, rel[i, j]) for i, j in _hungarian(cost, valid)]


# ----------------------------------------------------------------------------------------------- openings

def match_openings(A: dict, B: dict, p: MatchParams) -> list[tuple[int, int, float]]:
    ca = np.array([o["center"] for o in A["openings"]], float).reshape(-1, 2)
    cb = np.array([o["center"] for o in B["openings"]], float).reshape(-1, 2)
    d = np.linalg.norm(ca[:, None] - cb[None], axis=-1) if len(ca) and len(cb) else np.zeros((len(ca), len(cb)))
    return [(i, j, float(d[i, j])) for i, j in _hungarian(d, d <= p.max_opening_dist_m)]


# ----------------------------------------------------------------------------------------------- entry point

def _apply_local(B: dict, room_T: dict[str, np.ndarray]) -> dict:
    """Move each B room (polygon, walls, openings) by its own local correction."""
    out = copy.deepcopy(B)
    for r in out["rooms"]:
        if r["id"] in room_T:
            r["polygon"] = transform_plan({"rooms": [r]}, room_T[r["id"]])["rooms"][0]["polygon"]
    for i, w in enumerate(out["walls"]):
        if w.get("room_id") in room_T:
            out["walls"][i] = transform_plan({"walls": [w]}, room_T[w["room_id"]])["walls"][0]
    for i, o in enumerate(out["openings"]):
        rid = next((r for r in o.get("room_ids", []) if r in room_T), None)
        if rid is not None:
            out["openings"][i] = transform_plan({"openings": [o]}, room_T[rid])["openings"][0]
    return out


def aligned_plan_b(plan_b, T_ba: np.ndarray, match: MatchResult) -> dict:
    """Plan B as the matcher saw it: in A's frame, each matched room moved by its local correction (for figures)."""
    B = transform_plan(as_plan_dict(plan_b), np.asarray(T_ba, float))
    return _apply_local(B, {r.b: np.array(r.local_T) for r in match.rooms})


def match_plans(plan_a, plan_b, T_ba: np.ndarray | None = None, p: MatchParams = MatchParams()) -> MatchResult:
    """Match plan B to plan A. T_ba (3x3) maps B's plan coordinates into A's; identity if both share a frame."""
    A = as_plan_dict(plan_a)
    B = transform_plan(as_plan_dict(plan_b), np.eye(3) if T_ba is None else np.asarray(T_ba, float))
    room_pairs = match_rooms(A, B, p)
    room_T = {}
    for i, j, _ in room_pairs:
        ra, rb = A["rooms"][i], B["rooms"][j]
        room_T[rb["id"]] = local_room_transform(A, B, ra, rb, p) if p.local_refine else np.eye(3)
    BL = _apply_local(B, room_T)
    res = MatchResult()
    for i, j, iou in room_pairs:
        ra, rb = A["rooms"][i], BL["rooms"][j]
        res.rooms.append(RoomPair(ra["id"], rb["id"], iou, _iou(_poly(ra), _poly(rb)), room_T[rb["id"]].tolist()))
        wa, wb = _room_walls(A, ra["id"]), _room_walls(BL, rb["id"])
        for k, l, (ang, _, off, ov) in match_walls(wa, wb, p):
            res.walls.append(WallPair(wa[k]["id"], wb[l]["id"], ra["id"], rb["id"], off, ang, ov))
    for i, j, d in match_openings(A, BL, p):
        res.openings.append(OpeningPair(A["openings"][i]["id"], BL["openings"][j]["id"], d,
                                        A["openings"][i].get("kind", ""), BL["openings"][j].get("kind", "")))
    used = lambda pairs, side: {getattr(x, side) for x in pairs}   # noqa: E731
    res.unmatched = {
        "rooms_a": [r["id"] for r in A["rooms"] if r["id"] not in used(res.rooms, "a")],
        "rooms_b": [r["id"] for r in B["rooms"] if r["id"] not in used(res.rooms, "b")],
        "walls_a": [w["id"] for w in A["walls"] if w["id"] not in used(res.walls, "a")],
        "walls_b": [w["id"] for w in B["walls"] if w["id"] not in used(res.walls, "b")],
        "openings_a": [o["id"] for o in A["openings"] if o["id"] not in used(res.openings, "a")],
        "openings_b": [o["id"] for o in B["openings"] if o["id"] not in used(res.openings, "b")],
    }
    return res
