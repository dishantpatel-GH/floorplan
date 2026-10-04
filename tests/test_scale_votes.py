"""Video scale votes (floorplan/video/scale.py): the running median must follow a step of DPVO's scale and stay
inside its segment, and a vote must be metric step / DPVO step. Synthetic votes and poses, no images.

Run: env -u PYTHONPATH .venv/bin/python -m pytest tests/test_scale_votes.py -q
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan.video.scale import PNP_GAPS, pnp_scale_votes, running_median_scale  # noqa: E402


def _truth(k):
    # slow drift, then an 11x step at keyframe 60, like my own video (about 2.5 -> 33.5 m per DPVO unit at t 34.5 s)
    return np.where(k < 60, 2.5 * np.exp(0.003 * k), 33.5)


def test_running_median_follows_a_scale_step():
    rng = np.random.default_rng(0)
    n = 120
    mid = np.array([i + g / 2 for g in PNP_GAPS for i in range(n - g)])
    mid = mid[(mid < 15) | (mid > 40)]                       # no votes for 26 keyframes (a blank wall)
    vote = _truth(mid) * np.exp(rng.normal(0, 0.05, len(mid)))
    # 10% wrong votes, 8x off either way; none next to the gap, where a keyframe sees only 3 votes and 2 wrong ones
    # would win (min_votes=3 accepts that)
    bad = (rng.random(len(mid)) < 0.1) & (np.abs(mid - 27.5) > 17)
    vote[bad] *= rng.choice([1 / 8, 8], bad.sum())
    s, measured = running_median_scale(n, np.zeros(n, int), mid, vote, np.zeros(len(mid), int), half=8, min_votes=3)
    k = np.arange(n)
    err = np.abs(np.log(s / _truth(k)))
    assert not measured[23:33].any() and measured[:23].all() and measured[33:].all()
    # margins hold for 1000 seeds (worst 0.066 and 0.117); a running mean would put s[56] at 3.4-10.7x the truth
    assert err[(k <= 10) | (k >= 70)].max() < 0.08
    assert err[np.abs(k - 60) >= 4].max() < 0.15
    assert s[56] < 1.2 * _truth(56) and s[64] > 0.8 * _truth(64)     # the step stays sharp, not smeared over +-8
    gap = np.arange(22, 34)                                  # no keyframe measured in 23..32: log-interpolated
    assert np.allclose(np.log(s[gap]), np.interp(gap, [22, 33], np.log(s[[22, 33]])))


def test_votes_stay_inside_their_segment():
    rng = np.random.default_rng(1)
    n = 90
    seg = np.repeat([0, 1, 2], [40, 30, 20])                 # cuts at 40 and 70; segment 2 has no votes
    mid = np.array([i + g / 2 for g in PNP_GAPS for i in range(55 - g)], float)
    vseg = seg[np.floor(mid).astype(int)]
    vote = np.where(vseg == 0, 2.0, 20.0) * np.exp(rng.normal(0, 0.03, len(mid)))
    s, measured = running_median_scale(n, seg, mid, vote, vseg, half=8, min_votes=3)
    assert abs(np.log(s[39] / 2.0)) < 0.05 and abs(np.log(s[40] / 20.0)) < 0.05    # no leak across the cut
    last = np.where(measured & (seg == 1))[0].max()
    assert last < 69 and np.allclose(s[last:70], s[last])  # held after the segment's last measured keyframe
    assert np.isnan(s[70:]).all() and not measured[70:].any()


def test_vote_is_metric_step_over_dpvo_step():
    rng = np.random.default_rng(2)
    n, s_true = 12, 7.3                                      # metres per DPVO unit
    R = [Rotation.from_euler("yxz", [3.0 * k, -20 + rng.normal(0, 1), rng.normal(0, 1)], degrees=True).as_matrix()
         for k in range(n)]
    C = np.stack([0.15 * np.arange(n), 0.02 * np.sin(np.arange(n)), 0.1 * np.arange(n)], 1)    # metres
    G = np.eye(4)                                            # DPVO's world: any rigid frame, its own scale
    G[:3, :3] = Rotation.from_euler("xyz", [10, -40, 5], degrees=True).as_matrix()
    G[:3, 3] = [1.0, -2.0, 0.5]
    T = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        T[k, :3, :3], T[k, :3, 3] = R[k], C[k] / s_true
    T = G @ T
    steps = [dict(i=i, j=i + g, R=R[i + g].T @ R[i], c=R[i].T @ (C[i + g] - C[i]))
             for g in PNP_GAPS for i in range(n - g)]        # what PnP returns: X_j = R (X_i - c), metres
    I, J, v = pnp_scale_votes(steps, T, min_step_m=0.08, max_dir_deg=35.0, max_rot_deg=4.0)
    assert len(v) == len(steps) and np.allclose(v, s_true)
    steps[0]["c"] = Rotation.from_euler("y", 60, degrees=True).as_matrix() @ steps[0]["c"]     # wrong direction
    steps[1]["R"] = Rotation.from_euler("x", 10, degrees=True).as_matrix() @ steps[1]["R"]     # wrong rotation
    steps[2]["c"] = 0.05 * steps[2]["c"] / np.linalg.norm(steps[2]["c"])                       # 5 cm step
    I, J, v = pnp_scale_votes(steps, T, min_step_m=0.08, max_dir_deg=35.0, max_rot_deg=4.0)
    assert len(v) == len(steps) - 3 and np.allclose(v, s_true)
    assert not {(s["i"], s["j"]) for s in steps[:3]} & set(zip(I.tolist(), J.tolist()))
