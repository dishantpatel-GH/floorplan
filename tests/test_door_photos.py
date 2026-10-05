"""sim/door_photos.py: the ray model of the planning views, door normals and the ranking rule (no scene files needed)."""
import math

import numpy as np

from sim import door_photos as DP


def test_rays_follow_the_photo_camera():
    # centre ray along the heading, tilted by the pitch; the left image edge is HFOV/2 to the left (counter-clockwise)
    o, d, dz = DP.SceneView.rays(None, [1.0, 2.0, 1.4, 90.0, -12.0, 0.0], nx=3, ny=3)
    c = d[4]
    assert np.allclose(o[4], [1.0, 2.0, 1.4])
    assert abs(math.degrees(math.atan2(c[1], c[0])) - 90.0) < 0.5
    assert abs(math.degrees(math.asin(c[2])) + 12.0) < 0.5
    o, d, dz = DP.SceneView.rays(None, [0.0, 0.0, 1.4, 0.0, 0.0, 0.0], nx=2, ny=1)
    left = math.degrees(math.atan2(d[0][1], d[0][0]))
    assert left > 0 and abs(left - DP.HFOV / 2 * 0.5) < 10      # ray at the first quarter of the width
    assert dz[0] < 1.0


def test_door_normal_points_into_the_room():
    rooms = {"a": dict(polygon=[[0, 0], [3, 0], [3, 3], [0, 3]]), "b": dict(polygon=[[0, 3.2], [3, 3.2], [3, 6], [0, 6]])}
    d = dict(center=[1.5, 3.1, 1.0], axis="x")
    _, n_a, _ = DP.door_frame(d, rooms, "a")
    _, n_b, _ = DP.door_frame(d, rooms, "b")
    assert np.allclose(n_a, [0, -1]) and np.allclose(n_b, [0, 1])


def _c(score, own, target=0.5):
    return dict(score=score, m=dict(tex_own=own, target=target))


def test_rank_objective_first_own_side_breaks_ties():
    a, b, c, d = _c(0.30, 0.0), _c(0.27, 0.10), _c(0.20, 0.5), _c(0.31, 0.0, target=0.05)
    r = DP.rank([a, b, c, d])
    assert r[0] is b                    # within 80% of the best and sees its own side
    assert r.index(c) > r.index(a)      # below 80% of the best: the own side does not lift it
    assert r[-1] is d or r.index(d) > r.index(b)   # not looking through the opening: after the good ones
