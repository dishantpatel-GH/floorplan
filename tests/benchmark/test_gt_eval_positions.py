"""Openings paired by position (D-085): a GT that says where each opening is (sim_gt.json centres, or the wall that
holds it in gt_polygons.json) pairs predictions by place and type, not by width."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from floorplan.benchmark.gt_eval import evaluate, read_gt, read_gt_geometry  # noqa: E402

UP = {"view_from_above": {"u": "right", "v": "up"}}       # plan frame = GT frame (no mirror)
A = [(0, 0), (4.0, 0), (4.0, 3.0), (0, 3.0)]               # room A; B shares the partition x = 4.0-4.1
B = [(4.1, 0), (7.1, 0), (7.1, 3.0), (4.1, 3.0)]


def M(v, s=0.005):
    return dict(value=v, lo=v - 1.96 * s, hi=v + 1.96 * s, unit="m", method="", status="measured")


def plan(openings):
    walls, rooms = [], []
    for rid, poly in (("R1", A), ("R2", B)):
        ids = []
        for k in range(4):
            p0, p1 = poly[k], poly[(k + 1) % 4]
            L = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
            ids.append(f"{rid}W{k}")
            walls.append(dict(id=f"{rid}W{k}", room_id=rid, p0=p0, p1=p1, length=M(L), normal=(0, 0),
                              opening_ids=[]))
        rooms.append(dict(id=rid, label="", polygon=poly, wall_ids=ids, ceiling_height=M(2.7)))
    return dict(coordinate_frame=UP, rooms=rooms, walls=walls, openings=openings)


def sim_gt(d: Path) -> Path:
    """Two rooms, a 0.80 m door in the partition at y = 1.5 and a 1.20 m window in A's south wall at x = 2."""
    on = {("a", 0): ["window_0001"], ("a", 1): ["door_0001"], ("b", 3): ["door_0001"]}
    rooms = [dict(id=g, polygon=[list(p) for p in poly],
                  walls_cw=[dict(name=f"W{k + 1}", edge=k, openings=on.get((g, k), [])) for k in range(4)])
             for g, poly in (("a", A), ("b", B))]
    ops = [dict(id="door_0001", cls="door", center=[4.05, 1.5, 1.0], width=0.80, walls={"a": 1, "b": 3}),
           dict(id="window_0001", cls="window", center=[2.0, 0.0, 1.5], width=1.20, walls={"a": 0})]
    (d / "sim_gt.json").write_text(json.dumps(dict(rooms=rooms, openings=ops)))
    rows = ["room_id,item_type,item_id,value_m,notes"]
    for g, poly in (("a", A), ("b", B)):
        for k in range(4):
            p0, p1 = poly[k], poly[(k + 1) % 4]
            rows.append(f"{g},wall,W{k + 1},{((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5:.4f},")
    rows += ["a,door_width,door_0001,0.8000,rooms a+b", "b,door_width,door_0001,0.8000,rooms a+b",
             "a,window_width,window_0001,1.2000,"]
    (d / "ground_truth.csv").write_text("\n".join(rows) + "\n")
    return d / "ground_truth.csv"


def score(openings):
    with tempfile.TemporaryDirectory() as d:
        csv = sim_gt(Path(d))
        return evaluate(plan(openings), read_gt(csv), "lidar", read_gt_geometry(csv))


def test_found_by_place_not_width():
    # the door is in the right place with a width 0.25 m off: found (the width error is reported); a door of the
    # right width on another wall is a phantom; the window pairs with the window, not with the door
    res = score([dict(id="O1", kind="door", wall_ids=["R1W1"], room_ids=["R1", "R2"], center=(4.0, 1.6),
                      width=M(0.55)),
                 dict(id="O2", kind="door", wall_ids=["R1W2"], room_ids=["R1"], center=(2.0, 3.0), width=M(0.80)),
                 dict(id="O3", kind="window", wall_ids=["R1W0"], room_ids=["R1"], center=(2.1, 0.0),
                      width=M(1.10))])
    op = res["gates"]["openings"]
    assert op["pairing"].startswith("position"), op
    assert (op["door"]["found"], op["door"]["missed"], op["door"]["phantom"]) == (1, 0, 1), op["door"]
    assert op["door"]["width_err_cm"] == [-25.0] and op["door"]["pos_err_cm"] == [10.0], op["door"]
    assert (op["window"]["found"], op["window"]["phantom"]) == (1, 0), op["window"]
    rows = {r["pred_id"]: r for r in res["rows"] if r["kind"] == "door_width"}
    assert rows["O1"]["item"] == "door_0001" and "PHANTOM" in rows["O2"]["note"]
    # one row per true opening: the door listed in both rooms' GT is scored once
    assert sum(r["item"] == "door_0001" for r in res["rows"]) == 1


def test_wrong_place_is_missed_and_phantom():
    res = score([dict(id="O1", kind="door", wall_ids=["R1W1"], room_ids=["R1"], center=(4.0, 0.4), width=M(0.80))])
    d = res["gates"]["openings"]["door"]
    assert (d["found"], d["missed"], d["phantom"]) == (0, 1, 1), d      # 1.1 m from the true centre


def test_duplicate_counts_as_phantom():
    res = score([dict(id="O1", kind="door", wall_ids=["R1W1"], room_ids=["R1", "R2"], center=(4.0, 1.5),
                      width=M(0.80)),
                 dict(id="P1", kind="door", wall_ids=["R1W1"], room_ids=["R1", "R2"], center=(4.05, 1.7),
                      width=M(0.80), source="prior")])
    d = res["gates"]["openings"]["door"]
    assert (d["found"], d["phantom"], d["duplicate"]) == (1, 1, 1), d


def test_wall_level_gt_and_tape_fallback():
    # gt_polygons.json names the wall that holds a door but not where on it: a door on that wall pairs, width breaks
    # ties; without any outline the old per-room width pairing stays
    poly = {"rooms": [{"id": "a", "polygon": [list(p) for p in A],
                       "walls_cw": [{"name": f"W{k + 1}", "edge": k, "openings": ["door_x"] if k == 1 else []}
                                    for k in range(4)]}]}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "gt_polygons.json").write_text(json.dumps(poly))
        rows = ["room_id,item_type,item_id,value_m,notes"] + [
            f"a,wall,W{k + 1},{v},door on W2" if k == 1 else f"a,wall,W{k + 1},{v},"
            for k, v in enumerate((4.0, 3.0, 4.0, 3.0))] + ["a,door_width,D1,0.7800,door on W2"]
        (d / "gt.csv").write_text("\n".join(rows) + "\n")
        gt = read_gt(d / "gt.csv")
        p = plan([dict(id="O1", kind="door", wall_ids=["R1W1"], room_ids=["R1"], center=(4.0, 0.7), width=M(0.75))])
        res = evaluate(p, gt, "photo", read_gt_geometry(d / "gt.csv", d / "gt_polygons.json"))
        old = evaluate(p, gt, "photo")
    op = res["gates"]["openings"]
    assert op["pairing"].startswith("wall") and op["door"]["found"] == 1 and op["door"]["width_err_cm"] == [-3.0]
    assert old["gates"]["openings"]["pairing"].startswith("width")
