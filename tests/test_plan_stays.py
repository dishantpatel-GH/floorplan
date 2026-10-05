"""Video plan step (D-079): camera stays split a region at a door or at a wide opening with jambs, never inside one
open room; a step into a space is not a visit. Synthetic rasters and camera paths, no scenes.

Run: env -u PYTHONPATH .venv/bin/python -m pytest tests/test_plan_stays.py -q
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapely.geometry import box  # noqa: E402

from floorplan.plan.beta.extract import dwell_s  # noqa: E402
from floorplan.plan.beta.grid import Grid  # noqa: E402
from floorplan.plan.beta.params import BetaParams  # noqa: E402
from floorplan.plan.beta.segment import camera_stays, distance_map, split_by_stays  # noqa: E402

P = replace(BetaParams(), tier="video", stay_split=True)
CELL = P.cell_m


def _walk(points_s):
    """Camera path through (u, v, seconds to stay there) with 1 s of walking between places, 30 poses per second."""
    uv, ts, t = [], [], 0.0
    rng = np.random.default_rng(0)
    for k, (u, v, stay) in enumerate(points_s):
        if k:
            pu, pv, _ = points_s[k - 1]
            for a in np.linspace(0, 1, 30, endpoint=False):
                uv.append((pu + a * (u - pu), pv + a * (v - pv)))
                ts.append(t)
                t += 1 / 30
        for _ in range(int(stay * 30)):
            uv.append((u + rng.normal(0, 0.1), v + rng.normal(0, 0.1)))
            ts.append(t)
            t += 1 / 30
    return np.array(uv), np.array(ts)


def _two_rooms(opening_m, jambs):
    """Two 3 x 3 m rooms side by side (u 0-3 and 3.1-6.1), joined by an opening in the 10 cm wall between them."""
    g = Grid(u0=-0.5, v0=-0.5, cell=CELL, rows=int(4.0 / CELL), cols=int(7.1 / CELL))
    free = np.zeros(g.shape, bool)
    barrier = np.zeros(g.shape, bool)
    c = lambda u: int(round((u - g.u0) / CELL))                                      # noqa: E731
    r = lambda v: int(round((v - g.v0) / CELL))                                      # noqa: E731
    free[r(0):r(3), c(0):c(6.1)] = True
    free[:, c(3.0):c(3.1)] = False                                                   # the wall ...
    lo, hi = 1.5 - opening_m / 2, 1.5 + opening_m / 2
    free[r(lo):r(hi), c(3.0):c(3.1)] = True                                          # ... with its opening
    barrier[r(0):r(3), c(3.0):c(3.1)] = True
    barrier[r(lo):r(hi), c(3.0):c(3.1)] = False
    if not jambs:                                                  # wall evidence missing near the opening ends
        barrier[r(lo - 0.4):r(hi + 0.4), c(3.0):c(3.1)] = False
    for outer in ((slice(r(0) - 3, r(0)), slice(None)), (slice(r(3), r(3) + 3), slice(None)),
                  (slice(None), slice(c(0) - 3, c(0))), (slice(None), slice(c(6.1), c(6.1) + 3))):
        barrier[outer] = True
    labels = free.astype(np.int32)                                 # the region step merged both rooms (wide neck)
    return g, free, barrier, labels


def test_stays_are_the_places_the_camera_stood():
    uv, ts = _walk([(1.5, 1.5, 6.0), (4.5, 1.5, 6.0), (4.6, 1.4, 0.5)])
    stays = camera_stays(uv, ts, P.stay_radius_m, P.stay_min_s)
    assert len(stays) == 2
    centres = [np.median(uv[a:b + 1], axis=0) for a, b in stays]
    assert np.allclose(centres[0], (1.5, 1.5), atol=0.2) and np.allclose(centres[1], (4.5, 1.5), atol=0.2)


def test_wide_opening_with_jambs_splits_between_stays():
    g, free, barrier, labels = _two_rooms(1.6, jambs=True)
    uv, ts = _walk([(1.5, 1.5, 6.0), (4.5, 1.5, 6.0)])
    out, rep = split_by_stays(labels, distance_map(free, P), barrier, uv, ts, g, P)
    assert out.max() == 2 and len(rep["split"]) == 1
    a1, a2 = rep["split"][0]["parts"]
    assert abs(a1 - a2) < 1.0                                        # cut at the opening, not somewhere in a room


def test_opening_without_jambs_stays_one_room():
    for width in (1.2, 1.6):                      # door-sized too: the cut must stand on walls
        g, free, barrier, labels = _two_rooms(width, jambs=False)
        uv, ts = _walk([(1.5, 1.5, 6.0), (4.5, 1.5, 6.0)])
        out, rep = split_by_stays(labels, distance_map(free, P), barrier, uv, ts, g, P)
        assert out.max() == 1 and not rep["split"], width


def test_opening_wider_than_stay_open_max_stays_one_room():
    g, free, barrier, labels = _two_rooms(2.4, jambs=True)
    uv, ts = _walk([(1.5, 1.5, 6.0), (4.5, 1.5, 6.0)])
    out, rep = split_by_stays(labels, distance_map(free, P), barrier, uv, ts, g, P)
    assert out.max() == 1 and not rep["split"]


def test_two_places_in_one_open_room_stay_one_room():
    g = Grid(u0=-0.5, v0=-0.5, cell=CELL, rows=int(4.0 / CELL), cols=int(7.0 / CELL))
    free = np.zeros(g.shape, bool)
    free[25:175, 25:325] = True                                      # one 6 x 3 m room
    barrier = ~free
    uv, ts = _walk([(1.0, 1.5, 6.0), (5.0, 1.5, 6.0)])
    out, rep = split_by_stays(free.astype(np.int32), distance_map(free, P), barrier, uv, ts, g, P)
    assert out.max() == 1 and not rep["split"]


def test_dwell_counts_seconds_inside():
    uv, ts = _walk([(1.5, 1.5, 6.0), (4.5, 1.5, 0.5)])
    scene = dict(traj=np.c_[uv[:, 0], np.zeros(len(uv)), uv[:, 1]], timestamps=ts)
    assert 5.5 < dwell_s(box(0, 0, 3, 3), scene) < 7.0
    assert dwell_s(box(4, 1, 5, 2), scene) < 1.5
