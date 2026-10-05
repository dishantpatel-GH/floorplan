"""Doors from priors where the door itself is not seen (floorplan/openings/priors.py, D-085), on a two-room plan:
A (0..4 x 0..3) and B (4.1..7 x 0..3), a 10 cm partition at u = 4.0-4.1."""
import numpy as np

from floorplan.model import Measurement, Opening, Plan, Room, Wall
from floorplan.openings.priors import PriorParams, add_prior_doors, cut_door_widths, gap_doors

P = PriorParams()


def _m(v):
    return Measurement(v, v - 0.01, v + 0.01, "m", "test", "measured")


def _plan(labels=("room", "room"), openings=()):
    rooms, walls = [], []
    for rid, label, (x0, x1) in zip(("A", "B"), labels, ((0.0, 4.0), (4.1, 7.0))):
        pts = [(x0, 0.0), (x1, 0.0), (x1, 3.0), (x0, 3.0)]
        normals = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        ws = [Wall(f"{rid}-W{k + 1}", rid, pts[k], pts[(k + 1) % 4],
                   _m(float(np.hypot(*np.subtract(pts[(k + 1) % 4], pts[k])))), normals[k]) for k in range(4)]
        walls += ws
        rooms.append(Room(rid, label, pts, [w.id for w in ws], _m(12.0), _m(14.0), _m(2.6)))
    return Plan("t", "photo", rooms, walls, list(openings), [], _m(24.0))


def _pose(x, z, fx, fz):
    T = np.eye(4)
    f = np.array([fx, 0.0, fz]) / np.hypot(fx, fz)
    up = np.array([0.0, -1.0, 0.0])
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = np.cross(up, f), up, f, [x, 1.4, z]
    return T


def test_gap_with_jambs_is_a_door_wider_gap_stays_open():
    o1 = Opening("O1", "passage", ["A-W2", "B-W4"], ["A", "B"], (4.05, 1.5), _m(0.80),
                 evidence="free-space constriction 0.80 m wide separating A and B; wall evidence on both sides: True")
    o2 = Opening("O2", "passage", ["A-W2", "B-W4"], ["A", "B"], (4.05, 0.8), _m(1.40),
                 evidence="free-space constriction 1.40 m wide; wall evidence on both sides: True")
    plan = _plan(openings=[o1, o2])
    gap_doors(plan, P)
    assert (o1.kind, o1.source, o1.width.value) == ("door", "prior", 0.80) and o1.confidence < 0.5
    assert o2.kind == "passage" and o2.source == "geometry"


def _photo_case(extra=()):
    a, b = "01_living/IMG_1.jpg", "04_bathroom/IMG_2.jpg"
    po = Opening("PO1", "door", ["A-W2", "B-W4"], ["A", "B"], (4.05, 1.5),
                 Measurement.missing("m", "photo v3 doorway pair", "not measured"), confidence=0.6,
                 evidence=f"doorway pair {a} ~ {b} (dt 3.0 s)")
    plan = _plan(("01_living", "04_bathroom"), [po, *extra])
    scene = dict(cam_names=np.array([a, b]), T_wc=np.stack([_pose(4.05, 1.5, -1, 0), _pose(4.05, 1.5, 1, 0)]))
    info = dict(protocol_v2=dict(doorway_pairs=[dict(a=a, b=b, rooms=["01_living", "04_bathroom"])]))
    return plan, scene, info


def test_doorway_photos_give_one_door_with_the_bathroom_prior():
    plan, scene, info = _photo_case()
    add_prior_doors(plan, scene, info, "photo", log=lambda m: None, rules="b")
    doors = [o for o in plan.openings if o.kind == "door"]
    assert [o.id for o in doors] == ["P1"], doors                  # the widthless pair door is replaced
    d = doors[0]
    assert d.source == "prior" and sorted(d.room_ids) == ["A", "B"] and sorted(d.wall_ids) == ["A-W2", "B-W4"]
    assert np.allclose(d.center, (4.0, 1.5)) and d.width.value == 0.70 and d.width.status == "inferred"
    assert abs(d.width.hi - d.width.lo - 2 * 1.96 * P.sigma_m) < 1e-9
    assert any(a.via == "P1" for a in plan.adjacency) or not plan.adjacency


def test_measured_door_wins_over_the_prior():
    seen = Opening("S1", "door", ["A-W2"], ["A"], (4.0, 1.8), _m(0.76), source="segmentation")
    plan, scene, info = _photo_case([seen])
    add_prior_doors(plan, scene, info, "photo", log=lambda m: None, rules="b")
    assert [o.id for o in plan.openings if o.kind == "door"] == ["S1"]
    assert sorted(seen.room_ids) == ["A", "B"] and seen.width.value == 0.76


def test_walk_through_gives_a_door_jitter_does_not():
    plan = _plan()
    u = np.arange(2.0, 6.0, 0.05)
    scene = dict(traj=np.stack([u, np.full_like(u, 1.4), np.full_like(u, 1.5)], 1))
    add_prior_doors(plan, scene, {}, "video", log=lambda m: None, rules="c")
    doors = [o for o in plan.openings if o.kind == "door"]
    assert len(doors) == 1 and abs(doors[0].center[0] - 4.0) < 1e-6 and abs(doors[0].center[1] - 1.5) < 0.06
    assert sorted(doors[0].room_ids) == ["A", "B"] and doors[0].width.value == 0.80
    plan = _plan()                               # standing at the partition: a few cm back and forth
    u = np.r_[np.arange(2.0, 3.99, 0.05), np.tile([3.98, 4.12], 6), np.arange(3.95, 2.0, -0.05)]
    scene = dict(traj=np.stack([u, np.full_like(u, 1.4), np.full_like(u, 1.5)], 1))
    add_prior_doors(plan, scene, {}, "video", log=lambda m: None, rules="c")
    assert not [o for o in plan.openings if o.kind == "door"]


def test_door_cut_by_the_frame_gets_the_prior_width_from_its_seen_end():
    s1 = Opening("S1", "door", ["A-W1"], ["A"], (1.25, 0.0),
                 Measurement(0.5, 0.1, 0.9, "m", "lower bound", "inferred"), confidence=0.45, source="segmentation")
    plan = _plan(openings=[s1])
    plan.meta["semantic_openings"] = dict(openings=[dict(opening_id="S1", host_id="A-W1", t0=1.0, t1=1.5,
                                                         ends_seen=[1, 0])])
    cut_door_widths(plan, P)
    assert s1.source == "prior" and s1.width.value == 0.80 and s1.width.lo >= 0.5
    assert np.allclose(s1.center, (1.4, 0.0))     # from the seen end at t = 1.0 towards the cut one
