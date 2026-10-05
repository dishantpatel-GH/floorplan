"""Doors and windows from segmenter pixels (floorplan/openings/semantic.py) on ray-cast synthetic views.

One room 5 x 4 m, cameras 1.5 m above the floor looking at the wall z = 3. The class map and depth come from casting
each pixel's ray against the floor and that wall, so the geometry is exact: widths must come back to within 1 cm."""
import json

import numpy as np

from floorplan.export.json_export import plan_from_json, plan_to_json, validate_plan_json
from floorplan.model import Measurement, Opening, Plan, Room, Wall
from floorplan.openings.semantic import SemParams, SemView, add_to_plan, detect_openings

LABELS = ["wall", "floor", "door", "windowpane", "cabinet"]
WALL, FLOOR, DOOR, WINDOW, CABINET = range(5)
H, W, F = 300, 400, 300.0
FLOOR_Y = -1.5                      # the cameras are at y = 0


def _m(v):
    return Measurement(v, v - 0.05, v + 0.05, "m", "test", "measured")


def _plan(openings=None):
    pts = [(-2.0, -1.0), (3.0, -1.0), (3.0, 3.0), (-2.0, 3.0)]
    normals = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    walls = [Wall(f"R1-W{i + 1}", "R1", pts[i], pts[(i + 1) % 4],
                  _m(float(np.hypot(*np.subtract(pts[(i + 1) % 4], pts[i])))), normals[i]) for i in range(4)]
    room = Room("R1", "room", pts, [w.id for w in walls], _m(20.0), _m(18.0), _m(2.6))
    return Plan("synthetic", "video", [room], walls, list(openings or []), [], _m(20.0))


def _view(name, cam_x, items):
    """Camera at (cam_x, 0, 0) looking along +z (level). items: (class, x0, x1, y0, y1, z) boxes; the nearest of
    floor (y = -1.5), wall (z = 3) and the boxes wins each pixel."""
    K = np.array([[F, 0, W / 2 - 0.5], [0, F, H / 2 - 0.5], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = np.diag([-1.0, -1.0, 1.0])          # camera x right = world -x, y down = world -y, z = world z
    T[:3, 3] = [cam_x, 0.0, 0.0]
    v, u = np.mgrid[0:H, 0:W]
    r = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u, float)], -1) @ T[:3, :3].T
    depth = np.full((H, W), np.inf)
    sem = np.zeros((H, W), np.uint8)
    lam_w = 3.0 / r[..., 2]                         # wall z = 3 (camera-frame depth = lam, since r_z = 1)
    depth, sem = np.where(lam_w < depth, lam_w, depth), np.where(lam_w < depth, WALL, sem)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_f = np.where(r[..., 1] < -1e-6, FLOOR_Y / r[..., 1], np.inf)
    hit = lam_f < depth
    depth, sem = np.where(hit, lam_f, depth), np.where(hit, FLOOR, sem)
    for cls, x0, x1, y0, y1, z in items:            # vertical rectangles facing the camera at depth z
        lam = z / r[..., 2]
        X = cam_x + lam * r[..., 0]
        Y = lam * r[..., 1]
        hit = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (lam < depth + 1e-9)
        depth, sem = np.where(hit, lam, depth), np.where(hit, cls, sem)
    depth[~np.isfinite(depth)] = 0
    return SemView(name, sem.astype(np.uint8), depth.astype(np.float32), K, T)


def _door(x0, x1):                                  # closed leaf on the wall, floor to 2.05 m
    return (DOOR, x0, x1, FLOOR_Y, FLOOR_Y + 2.05, 3.0)


def test_closed_door_and_window_widths():
    items = [_door(0.4, 1.2), (WINDOW, -1.6, -0.4, FLOOR_Y + 0.9, FLOOR_Y + 2.1, 3.0)]
    views = [_view("a", 0.0, items), _view("b", 0.3, items), _view("c", -0.2, items)]
    det = detect_openings(views, LABELS, _plan(), FLOOR_Y)
    got = {k["kind"]: k for k in det["kept"]}
    assert set(got) == {"door", "window"}
    assert abs(got["door"]["width"] - 0.80) < 0.01 and got["door"]["host_id"] == "R1-W3"
    assert abs(got["window"]["width"] - 1.20) < 0.01
    assert got["door"]["views"] == 3 and got["window"]["bottom"] > 0.8


def test_cupboard_door_is_rejected():
    # a kitchen cupboard door on the wall surface: reaches the floor but only 0.85 m high
    items = [(DOOR, 0.4, 1.0, FLOOR_Y, FLOOR_Y + 0.85, 3.0)]
    det = detect_openings([_view("a", 0.0, items), _view("b", 0.3, items)], LABELS, _plan(), FLOOR_Y)
    assert not det["kept"]
    assert any("too low" in str(r.get("reason")) for r in det["rejected"])


def test_wardrobe_front_is_not_on_the_wall():
    # wardrobe doors 0.6 m in front of the wall, full height: their pixels are not on the wall surface
    items = [(DOOR, 0.4, 1.4, FLOOR_Y, FLOOR_Y + 2.1, 2.4)]
    det = detect_openings([_view("a", 0.0, items), _view("b", 0.3, items)], LABELS, _plan(), FLOOR_Y)
    assert not [k for k in det["kept"] if k["kind"] == "door"]


def test_one_view_needs_both_ends_and_a_large_blob():
    det = detect_openings([_view("a", 0.0, [_door(0.4, 1.2)])], LABELS, _plan(), FLOOR_Y)
    assert len(det["kept"]) == 1                  # large, both ends seen
    p = SemParams(single_view_min_frac=0.5)
    det = detect_openings([_view("a", 0.0, [_door(0.4, 1.2)])], LABELS, _plan(), FLOOR_Y, p)
    assert not det["kept"] and "view" in det["rejected"][0]["reason"]


def test_geometry_wins_and_plan_json_round_trips():
    geo = Opening("O1", "passage", ["R1-W3"], ["R1"], (0.85, 3.0), _m(0.9), confidence=0.6, evidence="neck")
    plan = _plan([geo])
    plan.walls[2].opening_ids.append("O1")
    items = [_door(0.4, 1.2), (WINDOW, -1.6, -0.4, FLOOR_Y + 0.9, FLOOR_Y + 2.1, 3.0)]
    views = [_view("a", 0.0, items), _view("b", 0.3, items)]
    det = detect_openings(views, LABELS, plan, FLOOR_Y)
    log = add_to_plan(plan, det["kept"], det["walls"], "video", SemParams())
    by_id = {o.id: o for o in plan.openings}
    assert by_id["O1"].kind == "door" and by_id["O1"].width.value == 0.9 and by_id["O1"].source == "geometry"
    new = [o for o in plan.openings if o.source == "segmentation"]
    assert [o.kind for o in new] == ["window"] and abs(new[0].width.value - 1.2) < 0.02
    assert new[0].id in plan.walls[2].opening_ids
    assert {d["action"] for d in log} == {"matched geometric opening", "added"}
    doc = plan_to_json(plan)
    assert validate_plan_json(doc) == []
    back = plan_from_json(json.loads(json.dumps(doc)))
    assert {o.id: o.source for o in back.openings} == {"O1": "geometry", new[0].id: "segmentation"}
