"""D-083 pillars and wall steps of the photo tier (floorplan/photo/pillars.py, polygon.cut_pillars). Synthetic depth
points of one photo at the spin centre looking at the +x wall (no model, no photo).

Run: env -u PYTHONPATH .venv/bin/python -m pytest tests/test_photo_pillars.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan.photo.params import PhotoParams  # noqa: E402
from floorplan.photo.pillars import merge, photo_candidates, profile_runs  # noqa: E402
from floorplan.photo.polygon import _shoelace, cut_pillars  # noqa: E402

WALL = 1.5            # +x wall, 1.5 m from the camera
SIDES = {"+x": WALL, "-x": 1.6, "+z": 1.6, "-z": 1.6}


def lay():
    return dict(ok=True, manhattan_yaw=0.0, cam_height_m=1.4,
                sides={s: dict(status="measured", offset=o, sigma=0.1) for s, o in SIDES.items()})


def photo(fronts=(), z0=-1.5, z1=1.6, h0=0.05, h1=1.8):
    """Points of the +x wall seen from the origin, one image column per 1 cm of wall; fronts: (z0, z1, depth, hlo,
    hhi) surfaces standing out of the wall by `depth` (they hide the wall behind them between hlo and hhi)."""
    P, col = [], []
    for c, z in enumerate(np.arange(z0, z1, 0.01)):
        for h in np.arange(h0, h1, 0.05):
            x = WALL
            for a, b, dep, lo, hi in fronts:
                if a <= z <= b and lo <= h <= hi:
                    x = WALL - dep
            P.append((x, h - 1.4, z))
            col.append(c)
    P = np.array(P)
    N = np.tile([-1.0, 0.0, 0.0], (len(P), 1))
    h = P[:, 1] + 1.4
    return P, N, h, np.array(col), np.ones(len(P), bool), np.ones(len(P), bool)


def find(*fronts, **kw):
    p = PhotoParams.from_dict(kw)
    P, N, h, col, wl, vert = photo(fronts)
    c = photo_candidates("ph", P, N, h, col, wl, vert, np.zeros(3), lay(), p)
    return merge(c, lay(), p)


def test_runs_split_the_profile_at_the_pillar():
    t = np.arange(-1.0, 1.0, 0.01)
    d = np.where((t > 0.2) & (t < 0.7), 1.38, 1.5)
    r = profile_runs(t, d, 0.05, 0.03)
    assert [round(x["level"], 2) for x in r] == [1.5, 1.38, 1.5]
    assert abs(r[1]["len"] - 0.5) <= 0.06


def test_pillar_in_the_middle_of_a_wall():
    # own Room W6: 0.55 m wide, 0.08 m out of the wall, floor to ceiling
    pil, rej = find((0.2, 0.75, 0.08, 0.0, 3.0))
    assert len(pil) == 1 and pil[0]["kind"] == "pillar" and pil[0]["side"] == "+x"
    assert abs(pil[0]["width_m"] - 0.55) < 0.07 and abs(pil[0]["depth_m"] - 0.08) < 0.015
    assert abs(pil[0]["face_m"] - (WALL - 0.08)) < 0.015


def test_step_runs_into_the_corner():
    # own Room W1/W3: the last 0.44 m of the wall is 0.126 m proud of it, up to the corner (+z side at 1.6 m)
    pil, _ = find((1.15, 1.6, 0.126, 0.0, 3.0))
    assert len(pil) == 1 and pil[0]["kind"] == "step"
    assert abs(pil[0]["t1"] - SIDES["+z"]) < 1e-6 and abs(pil[0]["depth_m"] - 0.126) < 0.015


def test_flat_wall_has_none():
    assert find()[0] == []


def test_wide_or_deep_furniture_is_not_a_pillar():
    assert find((-0.5, 0.6, 0.3, 0.0, 3.0))[0] == []          # 1.1 m wide: a wardrobe front, not a pillar
    assert find((0.0, 0.5, 0.9, 0.0, 3.0))[0] == []           # 0.9 m out of the wall (own wardrobe side)


def test_things_that_do_not_stand_on_the_floor_or_reach_the_top():
    assert find((0.0, 0.5, 0.2, 1.2, 3.0))[0] == []           # wall cabinet from 1.2 m up: wall below it
    assert find((0.0, 0.5, 0.2, 0.0, 1.3))[0] == []           # box on a stool: ends at 1.3 m, wall above
    assert find((0.0, 0.5, 0.2, 0.9, 3.0))[0] == []           # cabinet from 0.9 m up: fills the band, wall below


def test_far_wall_surface_is_furniture():
    p = PhotoParams()
    P, N, h, col, wl, vert = photo([(0.2, 0.75, 0.3, 0.0, 3.0)])
    L = lay()
    L["sides"]["+x"]["far_wall"] = dict(from_m=WALL - 0.3, to_m=WALL)
    c = photo_candidates("ph", P, N, h, col, wl, vert, np.zeros(3), L, p)
    pil, rej = merge(c, L, p)
    assert pil == [] and rej and "far-wall" in rej[0]["reason"]


def test_switch_off():
    from floorplan.photo.pillars import find_pillars
    assert find_pillars(lay(), {}, {}, {}, 1.0, PhotoParams.from_dict(dict(pillars=False)))["pillars"] == []


def rect_polygon():
    xl = [dict(offset=-SIDES["-x"], side="-x", kind="rectangle", status="measured", sigma=0.1),
          dict(offset=SIDES["+x"], side="+x", kind="rectangle", status="measured", sigma=0.1)]
    zl = [dict(offset=-SIDES["-z"], side="-z", kind="rectangle", status="measured", sigma=0.1),
          dict(offset=SIDES["+z"], side="+z", kind="rectangle", status="measured", sigma=0.1)]
    return dict(ok=True, used=False, x_lines=xl, z_lines=zl, inside=[[1]], changes=[],
                sides={s: dict(offset=o, status="measured", sigma=0.1) for s, o in SIDES.items()})


def test_cut_into_the_outline_keeps_the_sides():
    pl = rect_polygon()
    pil, _ = find((0.2, 0.75, 0.08, 0.0, 3.0), (1.15, 1.6, 0.126, 0.0, 3.0))
    cut_pillars(pl, pil, PhotoParams())
    assert pl["used"] and all(q["cut"] for q in pl["pillars"]) and pl["n_vertices"] == 10
    full = (SIDES["+x"] + SIDES["-x"]) * (SIDES["+z"] + SIDES["-z"])
    cut = sum(q["width_m"] * q["depth_m"] for q in pil)
    assert abs(pl["area_m2"] - (full - cut)) < 0.01
    assert abs(abs(_shoelace(np.array(pl["vertices"]))) - pl["area_m2"]) < 1e-3
    xs = [v[0] for v in pl["vertices"]]
    assert abs(max(xs) - SIDES["+x"]) < 1e-9 and abs(min(xs) + SIDES["-x"]) < 1e-9     # box sides unchanged
