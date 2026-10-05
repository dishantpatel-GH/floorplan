"""Steep keyframes (D-078): ceiling looks and straight-down looks leave the scale votes and the fusion; a look at the
end ends the walk where it starts, one in the middle cuts a scale segment after it; the pitch is measured against the
local up, so a slow drift of DPVO's rotation does not mark ordinary frames as steep. Synthetic pitches and rotations.

Run: env -u PYTHONPATH .venv/bin/python -m pytest tests/test_video_steep.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan.video.params import VideoParams  # noqa: E402
from floorplan.video.steep import local_pitch, steep_keyframes  # noqa: E402

P = VideoParams()


def _steep(pitch):
    return steep_keyframes(np.asarray(pitch, float), P.steep_up_deg, P.steep_down_deg, P.steep_min_kf,
                           P.steep_tail_kf)


def test_ceiling_look_at_the_end_ends_the_walk():
    # take1-like: 228 keyframes looking down ~23 deg, a ceiling look at kf 215-218 (+31..+38), then 9 more keyframes
    p = np.full(228, -23.0)
    p[211:215] = [1, 6, 16, 25]
    p[215:219] = [31, 34, 38, 33]
    p[219:222] = [21, 16, 7]
    r = _steep(p)
    assert r["stretches"] == [[215, 219, "up"]]
    assert r["walk_end"] == 215 and r["cuts"] == []
    assert r["exclude"].sum() == 4


def test_look_in_the_middle_cuts_a_segment_after_it():
    p = np.full(200, -25.0)
    p[100:106] = 40.0                        # a ceiling look in the middle
    p[150] = 35.0                            # one steep frame: left out, no cut
    r = _steep(p)
    assert r["walk_end"] == 200 and r["cuts"] == [106]
    assert r["exclude"][100:106].all() and r["exclude"][150] and r["exclude"].sum() == 7


def test_short_pieces_are_not_cut_and_walks_without_steep_looks_are_untouched():
    p = np.full(200, -30.0)
    p[10:14] = -60.0                         # straight down near the start: only 10 keyframes before it
    r = _steep(p)
    assert r["stretches"] == [[10, 14, "down"]] and r["cuts"] == [] and r["walk_end"] == 200
    p = np.full(164, -32.0)
    p[[40, 90]] = [-47.0, -49.0]             # single_room: inside both limits
    r = _steep(p)
    assert r["stretches"] == [] and not r["exclude"].any() and r["walk_end"] == 164


def _rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def test_local_up_follows_a_drifting_rotation():
    n = 120
    R_true = _rx(-25.0) @ np.diag([1.0, -1.0, -1.0])         # optical axis 25 deg below the horizon, +y up
    true = np.degrees(np.arcsin(R_true[1, 2]))
    drift = np.linspace(-10, 10, n)                           # DPVO's world tilts 20 deg over the walk (pitch axis)
    R = np.stack([_rx(d) @ R_true for d in drift])
    ups_kf = np.arange(0, n, 3)                               # GeoCalib's up, carried into DPVO's world
    ups = np.stack([_rx(d) @ np.array([0.0, 1.0, 0.0]) for d in drift[ups_kf]])
    glob = ups.mean(0) / np.linalg.norm(ups.mean(0))
    p_loc = local_pitch(R, ups_kf, ups, glob, half_kf=15)
    p_glob = np.degrees(np.arcsin(np.clip(R[:, :, 2] @ glob, -1, 1)))
    assert abs(true + 25.0) < 1e-6
    assert np.abs(p_loc - true).max() < 3.0                   # local up: within 3 deg everywhere
    assert np.abs(p_glob - true).max() > 8.0                  # one global up: ~10 deg off at both ends


def test_steep_frames_are_off_by_default():
    assert VideoParams().steep_frames is False
