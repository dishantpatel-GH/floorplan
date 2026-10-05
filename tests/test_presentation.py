"""Tests for the presentation drawing (floorplan/export/presentation.py).

What they pin: the drawing is deterministic; it draws every opening of the plan (doors, windows, gaps); it never
prints an interval; it survives broken and empty plans; room names and "W x L" come out as on a real-estate plan;
a slit between two rooms' walls is drawn as one wall; the CLI writes it beside the technical drawing.
Run: env -u PYTHONPATH ../.venv/bin/python -m pytest tests/test_presentation.py -q
"""
import subprocess
import sys
from pathlib import Path

import pytest
from shapely import affinity
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from make_synthetic_plan import stress_plan, synthetic_plan  # noqa: E402

from floorplan.export.presentation import (PresentationStats, PresentationStyle,  # noqa: E402
                                           build_presentation_walls, render_presentation, room_display_name,
                                           room_size_wl, rotated_plan)
from floorplan.model import Measurement, Plan, Room  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_plan.json"


def _counts(plan):
    kinds = [o.kind for o in plan.openings]
    return kinds.count("door"), kinds.count("window"), len(kinds) - kinds.count("door") - kinds.count("window")


def test_presentation_is_deterministic_and_draws_every_opening(tmp_path):
    for name, plan in (("fixture", synthetic_plan()), ("stress", stress_plan(seed=1, n=5, angle_deg=17))):
        s1 = render_presentation(plan, tmp_path / f"{name}_a")
        render_presentation(plan, tmp_path / f"{name}_b")
        assert (tmp_path / f"{name}_a.svg").read_bytes() == (tmp_path / f"{name}_b.svg").read_bytes()
        assert (tmp_path / f"{name}_a.png").stat().st_size > 10_000
        assert s1.openings_unplaced == 0
        assert (s1.doors, s1.windows, s1.passages) == _counts(plan)
        assert s1.rooms == len(plan.rooms)
        assert not any("±" in t for t in s1.texts), "no intervals in the presentation drawing"
        assert any(t.startswith("Total: ") for t in s1.texts)


def test_presentation_labels_rooms_in_capitals_with_their_size(tmp_path):
    st = render_presentation(synthetic_plan(), tmp_path / "fixture")
    assert {"BEDROOM", "KITCHEN", "LIVING ROOM"} <= set(st.texts)
    assert st.labels_forced == 0
    assert any(t.endswith(" m") and " x " in t for t in st.texts)


def test_presentation_survives_degenerate_and_empty_plans(tmp_path):
    plan = synthetic_plan()
    plan.rooms[0].polygon = [(0, 0), (1, 1)]            # degenerate polygon
    plan.openings[0].width = Measurement.missing()
    st = render_presentation(plan, tmp_path / "degenerate")
    assert any("invalid polygon" in w for w in st.warnings)
    empty = synthetic_plan()
    empty.rooms, empty.walls, empty.openings, empty.adjacency, empty.damage = [], [], [], [], []
    render_presentation(empty, tmp_path / "empty")
    assert (tmp_path / "empty.png").stat().st_size > 1_000


@pytest.mark.parametrize("label,expected", [("02_bedroom", "BEDROOM"), ("living_room", "LIVING ROOM"),
                                            ("room", "ROOM"), ("R3", "ROOM"), ("", "ROOM"), ("Room 12", "ROOM 12"),
                                            ("bedroom2", "BEDROOM 2"),
                                            ("corridor (not entered; seen through an opening)", "CORRIDOR")])
def test_room_display_name_from_label(label, expected):
    room = synthetic_plan().rooms[0]
    room.label = label
    for attr in ("name", "room_type"):                  # names may come from the room classifier (D-078)
        if hasattr(room, attr):
            setattr(room, attr, None)
    assert room_display_name(room) == expected


def test_room_display_name_prefers_the_room_name():
    room = synthetic_plan().rooms[0]
    room.label = "room"
    room.name = "Bedroom"
    assert room_display_name(room) == "BEDROOM"


def test_room_size_is_the_box_in_the_rooms_own_wall_directions():
    rect = box(0, 0, 3.0, 2.0)
    assert room_size_wl(rect) == pytest.approx((3.0, 2.0))
    assert room_size_wl(affinity.rotate(rect, 30, origin=(0, 0))) == pytest.approx((3.0, 2.0))
    assert room_size_wl(affinity.rotate(rect, 90, origin=(0, 0))) == pytest.approx((2.0, 3.0))   # W runs across


def test_turning_the_drawing_swaps_w_and_l_and_keeps_every_opening(tmp_path):
    plan = synthetic_plan()
    turned = rotated_plan(plan, 90)
    assert plan.rooms[0].polygon == synthetic_plan().rooms[0].polygon      # the plan itself is not turned
    for r0, r1 in zip(plan.rooms, turned.rooms):
        w0, l0 = room_size_wl(Polygon(r0.polygon))
        w1, l1 = room_size_wl(Polygon(r1.polygon))
        assert (w1, l1) == pytest.approx((l0, w0))
    st0 = render_presentation(plan, tmp_path / "a")
    st1 = render_presentation(plan, tmp_path / "b", PresentationStyle(rotate_deg=90))
    assert (st1.doors, st1.windows, st1.passages) == (st0.doors, st0.windows, st0.passages)
    assert st1.openings_unplaced == 0


def _room(rid: str, poly: Polygon) -> Room:
    pts = [tuple(p) for p in poly.exterior.coords[:-1]]
    m = Measurement(poly.area, poly.area, poly.area, unit="m2")
    return Room(rid, "room", pts, [], m, Measurement(poly.length), Measurement(None, status="not_observed"))


def test_slit_between_two_rooms_is_drawn_as_one_wall():
    a, b = box(0, 0, 3, 3), box(3.45, 0, 6, 3)          # faces 0.45 m apart: 2 x 0.10 m nominal + a 0.25 m slit
    plan = Plan("t", "video", [_room("A", a), _room("B", b)], [], [], [], Measurement(a.area + b.area))
    stats = PresentationStats()
    solid, _, _, _ = build_presentation_walls(plan, PresentationStyle(), stats)
    between = box(3.0, 0.2, 3.45, 2.8)
    assert solid.intersection(between).area == pytest.approx(between.area, rel=1e-6)
    assert stats.gap_fill_m2 > 0.4                      # the slit left between the two rings, 3 m long
    far = Plan("t", "video", [_room("A", a), _room("B", box(4.0, 0, 6, 3))], [], [], [], Measurement(1.0))
    solid_far, _, _, _ = build_presentation_walls(far, PresentationStyle(), PresentationStats())
    assert solid_far.intersection(box(3.3, 0.2, 3.7, 2.8)).area < 0.05   # a real 0.6 m gap stays a gap


def test_cli_style_presentation_writes_only_the_presentation(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_plan.py"), str(FIXTURE), "--out",
                        str(tmp_path), "--style", "presentation"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["plan_presentation.png", "plan_presentation.svg", "render_report.json"]
