"""D-077 far-wall rule of the photo layout (floorplan/photo/layout.py, _far_wall): a wall seen BESIDE the surface the
side fit chose, farther out and past that surface's end, replaces it. Synthetic side points only (no model, no photo).

Run: env -u PYTHONPATH .venv/bin/python -m pytest tests/test_photo_far_wall.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan.photo.layout import _far_wall_in_room, _fit_side  # noqa: E402
from floorplan.photo.params import PhotoParams  # noqa: E402


def surface(dist, t0, t1, photo, step=0.02, seed=0):
    """One point per 2 cm of tangent (one image column each), at a distance `dist` from the spin centre."""
    rng = np.random.default_rng(seed)
    t = np.arange(t0, t1, step)
    return np.c_[dist + rng.normal(0, 0.005, len(t)), t, np.full(len(t), 0.3), np.full(len(t), photo)]


def fit(*parts, far_wall=True, **params):
    p = PhotoParams.from_dict(params)
    a = np.concatenate(parts)
    return _fit_side(a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(int), p, far_wall=far_wall)


def test_wardrobe_front_gives_way_to_the_wall_beside_it():
    # own Room, lit take: photo 0 sees a wardrobe front at 1.1 m; photo 1 sees the wall 0.65 m behind it, on the
    # other side of the door that ends the wardrobe
    s = fit(surface(1.1, -2.0, -0.9, 0), surface(1.75, 0.1, 1.0, 1))
    assert abs(s["offset"] - 1.75) < 0.03
    assert s["far_wall"]["from_m"] < 1.15 and s["far_wall"]["shadow"] == 0.0 and s["far_wall"]["beyond_end"] == 1.0


def test_switch_off_keeps_the_nearer_surface():
    s = fit(surface(1.1, -2.0, -0.9, 0), surface(1.75, 0.1, 1.0, 1), layout_far_wall=False)
    assert abs(s["offset"] - 1.1) < 0.03 and "far_wall" not in s
    s = fit(surface(1.1, -2.0, -0.9, 0), surface(1.75, 0.1, 1.0, 1), far_wall=False)   # D-060 geometry-only refit
    assert abs(s["offset"] - 1.1) < 0.03


def test_same_wall_at_another_photo_scale_stays():
    # sim k65 bedroom: a blank-wall close-up's depth 30% too deep puts the same wall 0.35 m farther (ratio 1.3)
    s = fit(surface(1.1, -1.6, 2.2, 0), surface(1.45, -0.3, 1.9, 1))
    assert abs(s["offset"] - 1.1) < 0.03 and "far_wall" not in s


def test_wall_behind_the_chosen_surface_stays():
    # the far peak's rays pass where the chosen surface was seen (shadow): one of the two depths is wrong
    s = fit(surface(1.0, -1.0, 1.0, 0), surface(1.7, -1.4, 1.4, 1))
    assert abs(s["offset"] - 1.0) < 0.03 and "far_wall" not in s


def test_patch_seen_through_an_opening_stays():
    # own lit, white-door side: a wall seen on both sides of the far patch is a wall with an opening
    s = fit(surface(1.1, -1.7, -0.9, 0), surface(1.1, 0.6, 1.2, 0, seed=1), surface(1.8, -1.2, 0.0, 1))
    assert abs(s["offset"] - 1.1) < 0.03 and "far_wall" not in s


def test_few_points_do_not_make_a_wall():
    # sim k65 living room: 8 points of one doorway photo, one per 10 cm, made a 0.8 m peak 0.55 m out
    s = fit(surface(1.67, -0.9, 1.1, 0), surface(2.23, -4.1, -3.3, 1, step=0.1))
    assert abs(s["offset"] - 1.67) < 0.03 and "far_wall" not in s


def test_far_wall_outside_the_room_is_the_next_room():
    # sim k38_s1 bathroom: past the end of the wall is the door, and through it the next room's wall 1.3 m farther,
    # beside the room's +z wall (0.61 m): no overlap with the room's extent, the side keeps 1.15 m
    p = PhotoParams()
    pts = np.concatenate([surface(1.15, -1.0, -0.2, 0), surface(2.43, 0.7, 1.6, 1, step=0.01)])
    side_pts = {"-x": (pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3].astype(int))}

    def sides(z_plus):
        s = {"-x": _fit_side(*side_pts["-x"], p, far_wall=True)}
        s.update({k: dict(status="measured", offset=v) for k, v in (("+x", 1.0), ("+z", z_plus), ("-z", 1.08))})
        return s
    s = sides(0.61)
    assert "far_wall" in s["-x"] and abs(s["-x"]["offset"] - 2.43) < 0.03     # moved by the side fit ...
    _far_wall_in_room(s, side_pts, p)
    assert abs(s["-x"]["offset"] - 1.15) < 0.03 and s["-x"]["far_wall_rejected"]["overlap_m"] < 0.3   # ... not kept
    s = sides(2.0)                                                            # a room that reaches past the patch
    _far_wall_in_room(s, side_pts, p)
    assert abs(s["-x"]["offset"] - 2.43) < 0.03 and "far_wall_rejected" not in s["-x"]


def test_never_moves_nearer_and_ignores_far_rooms():
    # rule B's failure (sim k38 bedroom): a 0.8 m piece of clutter at 0.35 m must not pull the side in from the
    # 5 m wall; and a wall 2 m beyond the chosen one (another room through a door) is out of reach
    s = fit(surface(0.35, 0.2, 1.0, 0), surface(2.7, -3.5, 0.7, 1))
    assert abs(s["offset"] - 2.7) < 0.03
    s = fit(surface(1.2, -2.0, -0.5, 0), surface(3.2, 0.5, 2.5, 1))
    assert abs(s["offset"] - 1.2) < 0.03 and "far_wall" not in s
