#!/usr/bin/env python
"""Measure the LiDAR depth noise from the data itself (decision D-011, issue-free evidence for the error budget).

Idea: every surface is seen by many frames. Re-project frame i's depth into frame j (j = i + 5..60 frames, i.e.
0.1-1 s later) using the ARKit poses, and compare with frame j's own measured depth at that pixel. The difference
delta = d_j(measured) - d_j(predicted from i) contains the noise of BOTH frames plus a little pose error, so the
noise of a single depth sample is at most  sigma_single = 1.4826 * MAD(delta) / sqrt(2)  (MAD = robust std).

We bin by range and by ARKit confidence, which tells us:
  * how sigma grows with range        -> the range cap (D-006) and the sensor term of every interval
  * whether "medium" confidence is usable
  * which depth-intrinsics convention is right: plain scaling K*s, or the pixel-centre-correct
    c' = (c + 0.5) * s - 0.5 (a half-pixel shift); the right one gives the smaller residual
Usage: python scripts/noise_model.py <capture_dir> [<capture_dir> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.io.stray import DEPTH_H, DEPTH_W, load_stray  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "noise_model"
RANGE_BINS = np.array([0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])


def K_variant(cap, i, variant):
    s = DEPTH_W / cap.rgb_size[0]
    K = cap.K_rgb[i].copy()
    if variant == "plain":
        K[:2] *= s
    else:  # pixel-centre-correct scaling
        K[0, 0] *= s; K[1, 1] *= s
        K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K


def pair_residuals(cap, i, j, variant, stride=2):
    d_i = cap.depth(i, min_conf=0, max_m=5.0)
    c_i = cap.confidence(i)
    d_j = cap.depth(j, min_conf=0, max_m=5.0)
    c_j = cap.confidence(j)
    Ki, Kj = K_variant(cap, i, variant), K_variant(cap, j, variant)
    v, u = np.mgrid[0:DEPTH_H:stride, 0:DEPTH_W:stride]
    z = d_i[v, u]
    ok = z > 0
    u, v, z, ci = u[ok], v[ok], z[ok], c_i[v[ok], u[ok]]
    P = np.stack([(u - Ki[0, 2]) * z / Ki[0, 0], (v - Ki[1, 2]) * z / Ki[1, 1], z], 1)
    T = np.linalg.inv(cap.T_wc[j]) @ cap.T_wc[i]          # camera i -> camera j
    Q = P @ T[:3, :3].T + T[:3, 3]
    front = Q[:, 2] > 0.2
    uq = np.round(Kj[0, 0] * Q[:, 0] / np.where(front, Q[:, 2], 1) + Kj[0, 2]).astype(int)
    vq = np.round(Kj[1, 1] * Q[:, 1] / np.where(front, Q[:, 2], 1) + Kj[1, 2]).astype(int)
    inb = front & (uq >= 1) & (uq < DEPTH_W - 1) & (vq >= 1) & (vq < DEPTH_H - 1)
    meas = d_j[vq[inb], uq[inb]]
    cj = c_j[vq[inb], uq[inb]]
    pred = Q[inb, 2]
    good = meas > 0
    # reject occlusion / disocclusion outliers: > 10 cm disagreement is a different surface, not noise
    delta = meas[good] - pred[good]
    keep = np.abs(delta) < 0.10
    return dict(delta=delta[keep], range=pred[good][keep], conf=np.minimum(ci[inb][good][keep], cj[good][keep]))


def robust_sigma(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(2)) if len(x) > 50 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = {}
    for cdir in sys.argv[1:]:
        cap = load_stray(cdir)
        pairs = [(int(i), int(i + g)) for i, g in zip(rng.integers(0, cap.n - 61, 250), rng.integers(5, 61, 250))]
        per_variant = {}
        for variant in ("plain", "pixel_centre"):
            acc = [pair_residuals(cap, i, j, variant) for i, j in pairs]
            delta = np.concatenate([a["delta"] for a in acc])
            rnge = np.concatenate([a["range"] for a in acc])
            conf = np.concatenate([a["conf"] for a in acc])
            rows = []
            for c in (2, 1):
                for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
                    m = (conf == c) & (rnge >= lo) & (rnge < hi)
                    rows.append(dict(conf=c, r_lo=float(lo), r_hi=float(hi), n=int(m.sum()),
                                     sigma_single_mm=round(1000 * robust_sigma(delta[m]), 2),
                                     median_delta_mm=round(1000 * float(np.median(delta[m])), 2) if m.sum() > 50 else None))
            hi_all = conf == 2
            per_variant[variant] = dict(
                sigma_single_mm_conf2_all=round(1000 * robust_sigma(delta[hi_all]), 3),
                n_conf2=int(hi_all.sum()), bins=rows)
        results[cap.capture_id] = per_variant
        print(cap.capture_id, {v: per_variant[v]["sigma_single_mm_conf2_all"] for v in per_variant})
    (OUT / "noise_model.json").write_text(json.dumps(results, indent=1))
    # pooled table (plain variant unless pixel_centre is better everywhere)
    print("\nrange bin | conf2 sigma (mm) per capture | conf1 sigma (mm)")
    for b in range(len(RANGE_BINS) - 1):
        s2 = [results[c]["plain"]["bins"][b]["sigma_single_mm"] for c in results]
        s1 = [results[c]["plain"]["bins"][b + len(RANGE_BINS) - 1]["sigma_single_mm"] for c in results]
        n2 = [results[c]["plain"]["bins"][b]["n"] for c in results]
        print(f"{RANGE_BINS[b]:.1f}-{RANGE_BINS[b+1]:.1f} m | {s2} (n={n2}) | {s1}")


if __name__ == "__main__":
    main()
