"""Room pairing in floorplan.benchmark.gt_eval when several plan rooms carry the GT room's label.

The video tier labels most rooms "room", and my own tape GT names a room "room". The scorer used to keep the LAST plan
room with that label (on the own-house video plan: a 1.6 m2 room for an 11 m2 room). The label must not decide among
them: geometry does (overlap with the GT outline when GT polygons exist, else the wall-length profile).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from floorplan.benchmark.gt_eval import evaluate, read_gt, read_gt_geometry  # noqa: E402


def M(v, s=0.005):
    return dict(value=v, lo=v - 1.96 * s, hi=v + 1.96 * s, unit="m", method="", status="measured")


def make_plan(rooms):
    """rooms: [(id, label, polygon in plan (u, v))] -> plan dict, one wall per polygon edge."""
    walls, out = [], []
    for rid, label, poly in rooms:
        ids = []
        for k in range(len(poly)):
            p0, p1 = poly[k], poly[(k + 1) % len(poly)]
            ids.append(f"{rid}W{k}")
            walls.append(dict(id=ids[-1], room_id=rid, p0=p0, p1=p1, normal=(0, 0), opening_ids=[],
                              length=M(((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5)))
        out.append(dict(id=rid, label=label, polygon=poly, wall_ids=ids, ceiling_height=M(2.7)))
    return dict(rooms=out, walls=walls, openings=[])


def box(u0, v0, w, h):
    return [(u0, v0), (u0 + w, v0), (u0 + w, v0 + h), (u0, v0 + h)]


def gt_csv(rooms):
    """rooms: {GT room: [W1.. lengths]} -> CSV text."""
    rows = ["room_id,item_type,item_id,value_m,notes"]
    rows += [f"{g},wall,W{k + 1},{v},tape" for g, ls in rooms.items() for k, v in enumerate(ls)]
    return "\n".join(rows) + "\n"


def gt_json(polys):
    """polys: {GT room: polygon seen from above (x right, y up), clockwise from W1} -> sim_gt.json-like dict."""
    return dict(rooms=[dict(id=g, polygon=[list(p) for p in poly],
                            walls_cw=[dict(name=f"W{k + 1}", edge=k, openings=[]) for k in range(len(poly))])
                       for g, poly in polys.items()])


def score(tmp_path, plan, rooms, polys=None):
    (tmp_path / "gt.csv").write_text(gt_csv(rooms))
    geom = None
    if polys:
        (tmp_path / "gt.json").write_text(json.dumps(gt_json(polys)))
        geom = read_gt_geometry(tmp_path / "gt.csv", tmp_path / "gt.json")
    return evaluate(plan, read_gt(tmp_path / "gt.csv"), "video", geom)


def test_shared_label_tape_only_uses_profile(tmp_path):
    # three plan rooms called "room"; the tape GT has "room" (4 x 3) and "hall" (3 x 2.5). The 4 x 3 one is not last.
    plan = make_plan([("R1", "room", box(10, 0, 2, 2)), ("R2", "room", box(0, 0, 4, 3)),
                      ("R3", "room", box(4.1, 0, 3, 2.5))])
    res = score(tmp_path, plan, {"room": [3.0, 4.0, 3.0, 4.0], "hall": [2.5, 3.0, 2.5, 3.0]})
    assert res["room_pairs"] == {"room": "R2", "hall": "R3"}, res["room_pairs"]
    assert "wall-length profile" in res["room_shared_labels"]["room"]
    walls = [r for r in res["rows"] if r["kind"] == "wall"]
    assert len(walls) == 8 and all(abs(r["err"]) < 1e-6 for r in walls), walls


def test_shared_label_with_outlines_uses_overlap(tmp_path):
    # three identical 3 x 3 rooms called "room", side by side, and a 2 x 2 room; the GT "room" is the MIDDLE one.
    # Their wall-length profiles are equal, so only position (the whole-plan fit) can tell them apart.
    plan = make_plan([("R1", "room", box(0, 0, 3, 3)), ("R2", "room", box(3.1, 0, 3, 3)),
                      ("R3", "room", box(6.2, 0, 3, 3)), ("R4", "", box(9.3, 0, 2, 2))])
    def above(u0, w, h):            # the plan's v points down seen from above (I-001): y = -v; clockwise from W1
        return [(u0, 0), (u0 + w, 0), (u0 + w, -h), (u0, -h)]
    polys = {"hall": above(0, 3, 3), "room": above(3.1, 3, 3), "bath": above(6.2, 3, 3), "store": above(9.3, 2, 2)}
    rooms = {"hall": [3.0] * 4, "room": [3.0] * 4, "bath": [3.0] * 4, "store": [2.0] * 4}
    res = score(tmp_path, plan, rooms, polys)
    assert res["room_pairs"] == {"room": "R2", "hall": "R1", "bath": "R3", "store": "R4"}, res["room_pairs"]
    assert "overlap with the GT outline (IoU 1.00" in res["room_shared_labels"]["room"], res["room_shared_labels"]


def test_shared_label_room_alone_uses_its_shape(tmp_path):
    # one GT room, an L (10 m2); plan rooms called "room": a small square, the same L far away, a big box (last).
    # The whole plan does not fit the GT (IoU < 0.5), so each room is fitted on its own (rotation + translation).
    L = [(0, 3), (3, 3), (3, 1), (4, 1), (4, 0), (0, 0)]                     # seen from above, clockwise from W1
    plan = make_plan([("R1", "room", box(0, 0, 1.5, 1.5)), ("R2", "room", [(x + 20, -y) for x, y in L]),
                      ("R3", "room", box(-10, 0, 6, 5))])
    res = score(tmp_path, plan, {"room": [3.0, 2.0, 1.0, 1.0, 4.0, 3.0]}, {"room": L})
    assert res["room_pairs"] == {"room": "R2"}, res["room_pairs"]
    assert "overlap with the GT outline (IoU 1.00" in res["room_shared_labels"]["room"], res["room_shared_labels"]
    walls = [r for r in res["rows"] if r["kind"] == "wall"]
    assert all(r["err"] is not None and abs(r["err"]) < 1e-6 for r in walls), walls


def test_shared_label_warns_when_the_choice_looks_like_another_room(tmp_path):
    # the own-house video case in small: no plan room is the GT "room"; the best "room" candidate is hall-shaped
    plan = make_plan([("R1", "room", box(0, 0, 5, 3.2)), ("R2", "room", box(20, 0, 1, 1))])
    res = score(tmp_path, plan, {"room": [3.0, 4.0, 3.0, 4.0], "hall": [3.2, 5.0, 3.2, 5.0]})
    assert res["room_pairs"] == {"room": "R1", "hall": "R2"}, res["room_pairs"]
    assert "WARNING: R1 fits hall better" in res["room_shared_labels"]["room"], res["room_shared_labels"]
