"""Room names (floorplan/plan/room_types.py, D-078): folder names, the class vote, the shape rules, one kitchen per
home, and the names in plan.json (schema 1.1) and on the drawing."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from make_synthetic_plan import synthetic_plan  # noqa: E402

from floorplan.export.json_export import plan_from_json, plan_to_json, validate_plan_json  # noqa: E402
from floorplan.export.render import room_title  # noqa: E402
from floorplan.model import Measurement, Opening, Plan, Room  # noqa: E402
from floorplan.plan import room_types as RT  # noqa: E402

NAMES = ["wall", "floor", "ceiling", "bed", "sofa", "stove", "sink", "toilet", "windowpane", "table"]


def _room(rid, x0, y0, x1, y1, label="room"):
    m = lambda v: Measurement(v, v - 0.01, v + 0.01)    # noqa: E731
    a, b = sorted((x1 - x0, y1 - y0), reverse=True)
    return Room(rid, label, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], [], m((x1 - x0) * (y1 - y0)),
                m(2 * (x1 - x0 + y1 - y0)), m(2.6), (m(a), m(b)))


def _plan(rooms, openings=()):
    return Plan("t", "video", list(rooms), [], list(openings), [], Measurement(1.0, 0.9, 1.1))


def _ev(**share):
    """Evidence with these class shares (of 10 000 pixels); the rest is wall."""
    c = np.zeros(len(NAMES), np.int64)
    for k, v in share.items():
        c[NAMES.index(k.replace("_", " "))] = int(round(v * 10000))
    c[0] += 10000 - c.sum()
    return dict(counts=c, names=NAMES, images=5, source="test")


def test_folder_names():
    assert RT.type_from_folder("01_living_room") == "living_room"
    assert RT.type_from_folder("03_bedroom2") == "bedroom"
    assert RT.type_from_folder("04-Bathroom") == "bathroom"
    assert RT.type_from_folder("05_balcony") == "balcony"
    assert RT.type_from_folder("room") is None
    assert RT.type_from_folder("hall") is None          # living room in India, entrance hall in Britain


def test_objects_name_rooms():
    p = _plan([_room("A", 0, 0, 4, 3.5), _room("B", 4, 0, 6.5, 3), _room("C", 0, 3.5, 4, 7.5)])
    RT.name_plan(p, {"A": _ev(bed=0.05), "B": _ev(stove=0.02, sink=0.01), "C": _ev(sofa=0.03, table=0.02)})
    assert [r.name for r in p.rooms] == ["Bedroom", "Kitchen", "Living room"]
    assert p.rooms[0].type_evidence["top_classes"][0] == ["bed", 0.05]


def test_shape_rules():
    door = Opening("O1", "door", [], ["F"], (0.0, 0.5), Measurement(0.9, 0.85, 0.95))
    p = _plan([_room("P", 0, 0, 4, 1.2), _room("F", 0, 2, 1.5, 3.5), _room("G", 3, 2, 4.5, 5),
               _room("N", 5, 0, 6, 1.1), _room("H", 7, 0, 8.06, 1.9)], [door])
    RT.name_plan(p, {"P": _ev(sofa=0.02), "G": _ev(windowpane=0.3)})
    by = {r.id: r for r in p.rooms}
    assert by["P"].room_type == "passage"        # a sofa seen through a door does not make a 1.2 m room a living room
    assert by["H"].room_type == "passage"        # 1.06 m wide, 1.8x as long: nobody lives there
    assert by["F"].room_type == "foyer"          # small, with the entrance door
    assert by["G"].room_type == "balcony"        # narrow and mostly glass
    assert by["N"].room_type == "room"           # no images, no telling shape: unknown


def test_one_kitchen_and_numbered_bedrooms():
    p = _plan([_room("A", 0, 0, 3, 3), _room("B", 4, 0, 7, 3), _room("C", 0, 3, 4, 7), _room("D", 4, 3, 7, 6),
               _room("E", 0, -2.2, 3, -0.1)])
    RT.name_plan(p, {"A": _ev(stove=0.03), "B": _ev(stove=0.01, toilet=0.009), "C": _ev(bed=0.04),
                     "D": _ev(bed=0.02), "E": _ev(stove=0.02)})
    by = {r.id: r for r in p.rooms}
    assert by["A"].name == "Kitchen" and by["B"].name == "Bathroom"       # a second kitchen elsewhere falls back
    assert by["E"].name == "Kitchen"                                      # next to the kitchen: the same, split
    assert {by["C"].name, by["D"].name} == {"Bedroom 1", "Bedroom 2"}
    assert by["C"].name == "Bedroom 1"                                    # numbered by area


def test_frames_vote_for_the_room_they_are_in():
    p = _plan([_room("A", 0, 0, 3, 3), _room("B", 3, 0, 6, 3)])
    k = RT.assign_frames(p, np.array([[1, 1], [4, 1], [3.1, -0.2], [9, 9]]))
    assert k.tolist() == [0, 1, 1, -1]
    assert RT.pick_frames(np.array([0, 0, 0, 0, 1, -1]), 2).tolist() == [0, 3, 4]


def test_names_in_plan_json_and_drawing():
    plan = synthetic_plan()
    assert room_title(plan.rooms[0]) == plan.rooms[0].label          # unnamed plans draw their labels as before
    assert "name" not in plan_to_json(plan)["rooms"][0]
    RT.name_plan(plan, {"bedroom": _ev(bed=0.05)})
    doc = json.loads(json.dumps(plan_to_json(plan)))
    assert validate_plan_json(doc) == []
    bed = next(r for r in doc["rooms"] if r["id"] == "bedroom")
    assert (bed["name"], bed["type"]) == ("Bedroom", "bedroom") and bed["type_evidence"]["images"] == 5
    assert plan_to_json(plan_from_json(doc)) == doc
    assert room_title(next(r for r in plan.rooms if r.id == "bedroom")) == "Bedroom"


def test_hall_with_a_cot_is_the_living_room():
    """No sofa anywhere: the largest room at the entrance is the living room, even with a cot in it, when another room
    shows a stronger bed (own home: hall and bedroom)."""
    door = Opening("O1", "passage", [], ["H"], (0.0, 1.0), Measurement(1.04, 1.0, 1.08))
    p = _plan([_room("H", 0, 0, 4.5, 2.9), _room("B", 0, 3.0, 3.0, 6.7)], [door])
    RT.name_plan(p, {"H": _ev(bed=0.015), "B": _ev(bed=0.05)})
    by = {r.id: r for r in p.rooms}
    assert (by["H"].name, by["B"].name) == ("Living room", "Bedroom")


def test_largest_room_with_a_weak_bed_is_the_living_room_without_an_entrance():
    p = _plan([_room("H", 0, 0, 6.7, 3.3), _room("B", 0, 3.4, 3.46, 6.6)])
    RT.name_plan(p, {"H": _ev(bed=0.017, sink=0.007), "B": _ev(bed=0.078)})
    by = {r.id: r for r in p.rooms}
    assert (by["H"].name, by["B"].name) == ("Living room", "Bedroom")
