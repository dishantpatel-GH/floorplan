"""Video scale follow-up (D-076): the PnP self-check judges vote coverage, not the spread of the scale; the floor
levelling removes a vertical glitch of the camera path and ignores bed tops; the scale method is a parameter, and
"depth_agreement" is the default. Synthetic depth maps and poses, no images.

Run: env -u PYTHONPATH .venv/bin/python -m pytest tests/test_scale_followup.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan.video.frontend import floor_heights, level_to_floor, segment_quality  # noqa: E402
from floorplan.video.params import VideoParams  # noqa: E402
from floorplan.video.scale import estimate_scales  # noqa: E402

H, W = 192, 256
K = np.array([[200.0, 0, W / 2], [0, 200.0, H / 2], [0, 0, 1]])


def _pose(y, pitch_deg=-30.0, x=0.0):
    """Camera at height y in a +y-up world, looking along +z and down by pitch (camera +y points down in the image)."""
    a = np.radians(pitch_deg)
    R_wc = np.diag([-1.0, -1.0, 1.0]) @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_wc, [x, y, 0.0]
    return T


def _depth_of_plane(T, plane_y, top_y=None):
    """Depth map of the horizontal plane y = plane_y seen from pose T (0 where the ray goes up or past 6 m). With
    top_y, a bed top at that height hides the floor from every ray."""
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    d_w = rays @ T[:3, :3].T
    y0 = T[1, 3]
    t = (plane_y - y0) / np.where(np.abs(d_w[..., 1]) > 1e-9, d_w[..., 1], np.nan)
    if top_y is not None:
        t_top = (top_y - y0) / np.where(np.abs(d_w[..., 1]) > 1e-9, d_w[..., 1], np.nan)
        t = np.where(t_top > 0, t_top, t)
    z = np.where((t > 0) & np.isfinite(t), t, 0.0)            # rays have z = 1 in the camera frame, so depth = t
    return np.where(z < 6.0, z, 0.0)


def test_floor_heights_measures_the_camera_above_the_floor():
    T = np.stack([_pose(1.35), _pose(0.9, pitch_deg=-45), _pose(1.6, pitch_deg=20)])
    D = np.stack([_depth_of_plane(t, 0.0) for t in T])
    h = floor_heights(D, K, T, stride=2)
    assert abs(h[0] - 1.35) < 0.02 and abs(h[1] - 0.9) < 0.02
    assert np.isnan(h[2])                                      # looking up: no floor below


def test_level_to_floor_removes_a_vertical_glitch_and_ignores_bed_tops():
    n = 60
    y_true = 1.35 + 0.05 * np.sin(np.arange(n) / 5)            # real camera motion: kept
    drift = np.where(np.arange(n) >= 30, -1.2, 0.0)            # a DPVO glitch at kf 30: the path drops 1.2 m
    T_true = np.stack([_pose(y, x=0.1 * k) for k, y in enumerate(y_true)])
    D = np.stack([_depth_of_plane(t, 0.0) for t in T_true])
    for k in range(10, 16):                                    # the camera looks at a bed 0.6 m high: no floor seen
        D[k] = _depth_of_plane(T_true[k], 0.0, top_y=0.6)
    T_path = T_true.copy()
    T_path[:, 1, 3] += drift
    seg = np.zeros(n, int)
    dy, rep = level_to_floor(D, K, T_path, seg, VideoParams())
    assert rep["applied"] and rep["accepted"] == n - 6       # the bed keyframes are left out
    y_fixed = T_path[:, 1, 3] + dy
    off = y_fixed - y_true                                     # one constant offset everywhere (the gauge)
    assert np.ptp(off) < 0.03
    assert np.ptp(rep["floor_p10_p90_after_m"]) < 0.03 < np.ptp(rep["floor_p10_p90_before_m"])


def test_level_to_floor_needs_enough_floor():
    T = np.stack([_pose(1.35, pitch_deg=20) for _ in range(40)])
    D = np.stack([_depth_of_plane(t, 0.0) for t in T])         # looking up the whole time
    dy, rep = level_to_floor(D, K, T, np.zeros(40, int), VideoParams())
    assert not rep["applied"] and not dy.any()


def test_pnp_self_check_judges_votes_not_spread():
    n = 40
    T = np.stack([_pose(1.35, x=0.1 * k) for k in range(n)])
    D = np.stack([_depth_of_plane(t, 0.0) for t in T])
    seg = np.repeat([0, 1], 20)
    s_local = np.r_[np.geomspace(1.0, 30.0, 20), np.full(20, 5.0)]   # segment 0 follows a 30x drift
    p = VideoParams(scale_method="pnp")
    q = segment_quality(D, K, T, seg, s_local, p, votes={0: (0.9, 0.12), 1: (0.3, 0.12)})
    assert q[0]["scale_spread"] > p.trust_max_scale_spread and q[0]["trusted"]
    assert not q[1]["trusted"] and q[1]["reasons"][0].startswith("vote coverage")
    q = segment_quality(D, K, T, seg, s_local, p, votes={0: (0.9, 0.4), 1: (0.0, None)})
    assert q[0]["trusted"] and q[1]["reasons"] == ["vote coverage 0.00 < 0.5"]    # the vote residual is not judged
    p.scale_method = "depth_agreement"                         # the old check: the spread fails segment 0
    q = segment_quality(D, K, T, seg, s_local, p)
    assert not q[0]["trusted"] and q[0]["reasons"][0].startswith("local scale spread") and q[1]["trusted"]


def test_scale_method_is_checked():
    p = VideoParams()
    assert p.scale_method == "depth_agreement" and p.floor_level is None    # D-076: "pnp" is opt-in
    with pytest.raises(ValueError):
        estimate_scales(np.zeros((3, H, W)), K, np.tile(np.eye(4), (3, 1, 1)), [], 2, method="clamp")


def test_auto_takes_pnp_where_votes_are_dense_else_depth_agreement():
    from floorplan.video.scale import choose_segment_scales
    seg = np.repeat([0, 1, 2], 10)
    s_pnp, s_da = np.r_[np.full(10, 2.0), np.full(10, 9.0), np.full(10, 4.0)], np.full(30, 3.0)
    pnp = [dict(id=0, keyframes=[0, 9], votes=12, s=2.0, sigma_rel_stat=0.01, vote_coverage=0.8, vote_residual=0.02),
           dict(id=1, keyframes=[10, 19], votes=3, s=9.0, sigma_rel_stat=0.2, vote_coverage=0.3, vote_residual=0.1),
           dict(id=2, keyframes=[20, 29], votes=6, s=4.0, sigma_rel_stat=0.05, vote_coverage=0.5, vote_residual=0.03)]
    da = [dict(id=g, keyframes=[10 * g, 10 * g + 9], pairs=20, s=3.0, sigma_rel_stat=0.03) for g in range(3)]
    s, out = choose_segment_scales(seg, s_pnp, pnp, s_da, da, 0.5)
    assert [o["scale_method"] for o in out] == ["pnp", "depth_agreement", "pnp"]   # coverage 0.5 is enough
    assert np.allclose(s, np.r_[np.full(10, 2.0), np.full(10, 3.0), np.full(10, 4.0)])
    assert out[1]["pairs"] == 20 and out[1]["vote_coverage"] == 0.3 and out[1]["votes"] == 3   # its votes recorded


def test_auto_self_check_follows_each_segments_method():
    n = 40
    T = np.stack([_pose(1.35, x=0.1 * k) for k in range(n)])
    D = np.stack([_depth_of_plane(t, 0.0) for t in T])
    seg = np.repeat([0, 1], 20)
    s_local = np.r_[np.geomspace(1.0, 30.0, 20), np.geomspace(1.0, 30.0, 20)]   # both follow a 30x drift
    p = VideoParams(scale_method="auto")
    q = segment_quality(D, K, T, seg, s_local, p, votes={0: (0.9, 0.1), 1: (0.3, 0.1)},
                        methods={0: "pnp", 1: "depth_agreement"})
    assert q[0]["trusted"]                                     # PnP: the spread is not judged
    assert not q[1]["trusted"] and q[1]["reasons"][0].startswith("local scale spread")
