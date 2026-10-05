"""Open doorways from the depth seen through the walls (floorplan/openings/seethrough.py) on ray-cast synthetic views.

One room 5 x 4 m; the wall under test is z = 3 (plan wall R1-W3), 0.12 m thick, with a hole in it. Cameras stand
1.5 m above the floor, pitched 15 deg down. Depth and classes come from casting each pixel's ray against the
room's floor, the wall's faces, the hole's reveals and what lies beyond: the next room (floor and a far wall), the
outside (no depth), or a 0.8 m deep alcove."""
import json

import numpy as np

from floorplan.export.json_export import plan_from_json, plan_to_json, validate_plan_json
from floorplan.model import Measurement, Plan, Room, Wall
from floorplan.openings.semantic import SemView, _wall_lines
from floorplan.openings.seethrough import see_through_doors

LABELS = ["wall", "floor", "door", "windowpane"]
WALL, FLOOR = 0, 1
H, W, F = 300, 400, 300.0
FY = -1.5                           # floor height; the cameras are at y = 0
ZW, THICK, TOP = 3.0, 0.12, 2.6


def _m(v):
    return Measurement(v, v - 0.05, v + 0.05, "m", "test", "measured")


def _plan():
    pts = [(-2.0, -1.0), (3.0, -1.0), (3.0, 3.0), (-2.0, 3.0)]
    normals = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    walls = [Wall(f"R1-W{i + 1}", "R1", pts[i], pts[(i + 1) % 4],
                  _m(float(np.hypot(*np.subtract(pts[(i + 1) % 4], pts[i])))), normals[i]) for i in range(4)]
    room = Room("R1", "room", pts, [w.id for w in walls], _m(20.0), _m(18.0), _m(2.6))
    return Plan("synthetic", "photo", [room], walls, [], [], _m(20.0))


def _scene(x0, x1, ylo, yhi, beyond):
    """Rectangles (axis, value, range of the 1st other axis, range of the 2nd, class): the wall z = 3 with a hole
    x0..x1, ylo..yhi above the floor, its reveals, the room's floor and what lies beyond the wall."""
    zb = ZW + THICK
    R = [(1, FY, (-2.0, 3.0), (-1.0, ZW), FLOOR),                                  # the room's floor
         (2, ZW, (-2.0, x0), (FY, FY + TOP), WALL), (2, ZW, (x1, 3.0), (FY, FY + TOP), WALL),
         (2, ZW, (x0, x1), (FY + yhi, FY + TOP), WALL),                            # head
         (0, x0, (FY + ylo, FY + yhi), (ZW, zb), WALL), (0, x1, (FY + ylo, FY + yhi), (ZW, zb), WALL),
         (1, FY + yhi, (x0, x1), (ZW, zb), WALL)]
    if ylo > 0:                                                                    # sill
        R += [(2, ZW, (x0, x1), (FY, FY + ylo), WALL), (1, FY + ylo, (x0, x1), (ZW, zb), WALL)]
    if beyond == "room":
        R += [(1, FY, (-6.0, 7.0), (zb, 7.0), FLOOR), (2, 7.0, (-6.0, 7.0), (FY, FY + TOP), WALL)]
    elif beyond == "alcove":
        za = ZW + 0.8
        R += [(2, za, (x0, x1), (FY, FY + yhi), WALL), (1, FY, (x0, x1), (ZW, za), FLOOR),
              (0, x0, (FY, FY + yhi), (ZW, za), WALL), (0, x1, (FY, FY + yhi), (ZW, za), WALL),
              (1, FY + yhi, (x0, x1), (ZW, za), WALL)]
    return R


def _view(name, C, yaw_deg, rects, pitch_deg=15.0):
    K = np.array([[F, 0, W / 2 - 0.5], [0, F, H / 2 - 0.5], [0, 0, 1]])
    a, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])     # looks down by p
    T = np.eye(4)
    T[:3, :3] = Ry @ np.diag([-1.0, -1.0, 1.0]) @ Rx
    T[:3, 3] = C
    v, u = np.mgrid[0:H, 0:W]
    r = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u, float)], -1) @ T[:3, :3].T
    depth = np.full((H, W), np.inf)
    sem = np.zeros((H, W), np.uint8)
    others = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    for ax, val, ra, rb, cls in rects:
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (val - C[ax]) / r[..., ax]
        i, j = others[ax]
        A, B = C[i] + lam * r[..., i], C[j] + lam * r[..., j]
        hit = (lam > 1e-6) & (A >= ra[0]) & (A <= ra[1]) & (B >= rb[0]) & (B <= rb[1]) & (lam < depth)
        depth, sem = np.where(hit, lam, depth), np.where(hit, cls, sem)
    depth[~np.isfinite(depth)] = 0                    # nothing hit: beyond the depth range
    return SemView(name, sem, depth.astype(np.float32), K, T)


def _run(rects, plan=None, only_a=False):
    plan = plan or _plan()
    views = [_view("a", np.array([0.8, 0.0, 0.6]), 0.0, rects)]
    if not only_a:
        views.append(_view("b", np.array([-0.6, 0.0, 0.8]), 32.0, rects))
    walls = {wl.idx: wl for wl in _wall_lines(plan)}
    return plan, see_through_doors(views, LABELS, plan, FY, walls)


def test_open_doorway_is_a_door_of_its_width():
    plan, rep = _run(_scene(0.4, 1.2, 0.0, 2.05, "room"))
    assert len(rep["kept"]) == 1, rep["rejected"]
    k = rep["kept"][0]
    assert k["host_id"] == "R1-W3" and k["ends_seen"] == [2, 2] and k["whole_views"] == 2
    assert abs(k["width"] - 0.80) < 0.02             # jamb to jamb on the wall line, also from the oblique view
    door = [o for o in plan.openings if o.source == "see_through"]
    assert len(door) == 1 and door[0].kind == "door" and door[0].width.status == "measured"
    assert abs(door[0].center[0] - 0.8) < 0.02 and abs(door[0].center[1] - 3.0) < 1e-6
    assert "seen through" in door[0].evidence
    doc = plan_to_json(plan)
    assert validate_plan_json(doc) == []
    assert plan_from_json(json.loads(json.dumps(doc))).openings[0].source == "see_through"


def test_window_with_a_sill_is_not_a_door():
    for beyond in ("outside", "room"):               # the outside (no depth) or a room behind an inner window
        plan, rep = _run(_scene(-1.6, -0.4, 0.9, 2.1, beyond))
        assert not rep["kept"] and not plan.openings, beyond


def test_alcove_is_not_a_door():
    plan, rep = _run(_scene(0.4, 1.2, 0.0, 2.05, "alcove"))
    assert not rep["kept"] and not plan.openings


def test_one_jamb_alone_is_no_door():
    """Only view a, whose image border cuts the doorway (x 1.9-2.9 m; a sees up to x = 2.4 m): one jamb seen, so
    what was seen (0.5 m) is a lower bound with nothing at the other end. Logged, never a door or a width (k22's
    glossy shower tiles read as a 0.94 m one-jamb doorway)."""
    plan, rep = _run(_scene(1.9, 2.9, 0.0, 2.05, "room"), only_a=True)
    assert [s["cut0"] + s["cut1"] for s in rep["view_spans"]] == [1]
    assert not rep["kept"] and not plan.openings
    assert len(rep["rejected"]) == 1 and "both jambs" in rep["rejected"][0]["reason"]


def test_low_block_at_the_line_is_not_a_jamb():
    """A 1.6 m doorway (x 0-1.6 m) with a 0.9 m high block standing in it at the wall line (x 0-0.7 m), like k38's
    kitchen counter: the block's face is wall-like at the line, but over it the views look through the plane, so it
    is not a jamb and no 0.9 m door comes of it."""
    blk = [(2, ZW - 0.1, (0.0, 0.7), (FY, FY + 0.9), WALL), (1, FY + 0.9, (0.0, 0.7), (ZW - 0.1, ZW + THICK), WALL),
           (0, 0.7, (FY, FY + 0.9), (ZW - 0.1, ZW + THICK), WALL)]
    plan, rep = _run(blk + _scene(0.0, 1.6, 0.0, 2.05, "room"))
    assert not [o for o in plan.openings if o.width is not None and abs(o.width.value - 0.9) < 0.15]
    assert any(max(s["beside_thru"]) > 0.2 for s in rep["view_spans"] + rep["spans_left_out"] if "beside_thru" in s)
