"""Steep keyframes (D-084): looks at the ceiling and straight down, found from the camera pitch.

A walkthrough films the walls with the phone tilted down: the median camera pitch is -15 to -33 deg on every capture
we have. Turned to the ceiling, the camera sees a blank surface and DPVO loses track. On my own take1 the bedroom
ceiling look at the end (kf 211-222, t 110-115 s) puts the camera 2.0-13.6 m above the floor in the scene, against
1.1-1.6 m for the rest of the walk, on the nine cached DPVO runs. That look peaks at +33..+40 deg, inside GeoCalib's
+-45 deg range, so a 45 deg limit would not see it. Hence two limits:

  * up_deg (30): more than 30 deg above the horizon the view is mostly ceiling (take1: 3-4 keyframes of the look on
    every cached run; the sample walks reach it only on with_ceiling);
  * down_deg (50): straight down, beyond GeoCalib's range (gravity already ignores those frames).

Every keyframe beyond either limit is left out of the scale votes and the fusion. A run of at least min_kf such
keyframes is a steep look. If fewer than tail_kf keyframes follow the last look, the walk ends where the look starts:
the path after it is not reliable, and too short to be checked on its own. A look in the middle cuts a scale segment
after it when both sides keep at least tail_kf usable (not steep) keyframes; a shorter piece could not pass the
segment self-check.

The pitch is measured against the local "up": the mean of the GeoCalib up vectors of the keyframes within +-half_kf,
carried into DPVO's world by DPVO's rotations. DPVO's rotation drifts 15-20 deg on some runs (take1 after r1), and a
single global "up" then marks ordinary frames as steep."""
from __future__ import annotations

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[a, b) index ranges of the True runs of a boolean array."""
    d = np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def local_pitch(R_kf: np.ndarray, ups_kf: np.ndarray, ups: np.ndarray, up: np.ndarray, half_kf: int = 15,
                min_frames: int = 5) -> np.ndarray:
    """Pitch (deg, + = looking up) of each keyframe's optical axis (R_kf[:, :, 2], world) against the mean of the
    world up vectors `ups` of GeoCalib keyframes `ups_kf` within +-half_kf; the global `up` where fewer than
    min_frames are near."""
    out = np.empty(len(R_kf))
    for k in range(len(R_kf)):
        near = np.abs(ups_kf - k) <= half_kf
        u = ups[near].mean(0) if near.sum() >= min_frames else up
        out[k] = np.degrees(np.arcsin(np.clip(R_kf[k, :, 2] @ (u / np.linalg.norm(u)), -1, 1)))
    return out


def steep_keyframes(pitch: np.ndarray, up_deg: float, down_deg: float, min_kf: int, tail_kf: int) -> dict:
    """Steep looks of a walk, from the pitch (deg, + = up) of each keyframe.

    Returns exclude (bool per keyframe), stretches ([a, b, "up"/"down"]: keyframes a..b-1), walk_end (the walk keeps
    keyframes 0..walk_end-1) and cuts (keyframes where a new scale segment starts)."""
    pitch = np.asarray(pitch, float)
    n = len(pitch)
    st = [(a, b, "up") for a, b in _runs(pitch > up_deg) if b - a >= min_kf]
    st += [(a, b, "down") for a, b in _runs(pitch < -down_deg) if b - a >= min_kf]
    st.sort()
    exclude = (pitch > up_deg) | (pitch < -down_deg)
    end = n
    if st and n - st[-1][1] < tail_kf and st[-1][0] >= tail_kf:     # a walk shorter than that could not be checked
        end = st[-1][0]
    usable = ~exclude                                   # steep keyframes carry no depth and no votes
    cuts, last = [], 0
    for a, b, _ in st:
        if a > 0 and b < end and usable[last:b].sum() >= tail_kf and usable[b:end].sum() >= tail_kf:
            cuts.append(int(b))
            last = b
    return dict(exclude=exclude, stretches=[[int(a), int(b), k] for a, b, k in st], walk_end=int(end), cuts=cuts)
