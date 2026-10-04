"""Measurement on RAW LiDAR points (D-007): robust fits for wall faces, floors and ceilings.

The cell complex tells us WHERE each wall is to within a TSDF voxel (2 cm). The numbers we report must be better
than that, so every wall face is re-measured on the raw high-confidence LiDAR points near it:

  1. take raw points within +-fit_band of the face, away from corners and openings, at wall heights, and seen
     FROM THE ROOM SIDE (the viewing ray points into the wall). The side test separates the two faces of a thin
     partition, which are only ~10 cm apart and would otherwise contaminate each other;
  2. start at the MODE of the signed distances (the biggest flat surface, i.e. the wall, not a picture frame or
     skirting board) and refine with a Tukey biweight M-estimator (iteratively reweighted least squares), which
     gives zero weight to points more than ~4.7 robust sigmas away (furniture, points seen through a door);
  3. a wall face is fitted axis-aligned (an offset) by default, or as a tilted LINE d = offset + slope * (along -
     middle) if that makes it a clean plane (_choose_model): real walls can be 1-2 degrees off square, and forcing
     such a wall onto the axis moves its corners by L/2 * tan(tilt) (2.6 cm on a 3 m wall at 1 degree);
  4. standard error = robust sigma / sqrt(n_eff), where n_eff counts distinct 10 cm x 10 cm patches rather than
     points: neighbouring LiDAR points share the same frame's error, so counting points would claim an
     impossibly small error;
  5. surface inconsistency = sqrt(max(rmse^2 - point_noise^2, 0)): scatter the sensor alone cannot explain (two
     passes that disagree because of drift, a curtain, a wavy surface). The fit cannot tell which layer is the
     wall, so this goes into the interval as a bias term instead of being averaged away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floorplan.plan.alpha.params import AlphaParams

TUKEY_C = 4.685          # 95% efficiency under Gaussian noise
MAD_TO_SIGMA = 1.4826


@dataclass
class Fit:
    value: float          # fitted level at the reference position (offset), metres
    slope: float          # d(level)/d(along); 0 for a level fit
    se: float             # standard error of `value`
    rmse: float           # RMS of inlier residuals (how flat / consistent the surface is)
    n_inliers: int
    n_eff: int


def _mode(d: np.ndarray, band: float, bin_m: float = 0.002) -> float:
    edges = np.arange(-band, band + bin_m, bin_m)
    h, e = np.histogram(d, bins=edges)
    h = np.convolve(h, np.ones(5) / 5, mode="same")
    k = int(np.argmax(h))
    return float((e[k] + e[k + 1]) / 2)


def robust_fit(d: np.ndarray, patch_keys: np.ndarray, x: np.ndarray | None = None, band: float = 0.06,
               max_slope: float = 0.0, iters: int = 20) -> Fit | None:
    """Tukey-IRLS fit of d = value + slope * x (slope only if x is given and max_slope > 0). None if no support."""
    if len(d) < 10:
        return None
    use_slope = x is not None and max_slope > 0 and np.ptp(x) > 0.3
    X = np.c_[np.ones(len(d)), x] if use_slope else np.ones((len(d), 1))
    beta = np.zeros(X.shape[1])
    beta[0] = _mode(d, band)
    s = 0.003
    for _ in range(iters):
        r = d - X @ beta
        core = np.abs(r) < max(4 * s, 0.01)
        if core.sum() > 5:
            s = max(MAD_TO_SIGMA * float(np.median(np.abs(r[core] - np.median(r[core])))), 0.001)
        c = TUKEY_C * s
        w = np.where(np.abs(r) < c, (1 - (r / c) ** 2) ** 2, 0.0)
        if w.sum() <= 0:
            return None
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], d * sw, rcond=None)[0]
        if use_slope:
            new[1] = float(np.clip(new[1], -max_slope, max_slope))
        if np.max(np.abs(new - beta)) < 1e-6:
            beta = new
            break
        beta = new
    r = d - X @ beta
    inl = np.abs(r) < 2.5 * s
    if inl.sum() < 10:
        return None
    rmse = float(np.sqrt(np.mean(r[inl] ** 2)))
    n_eff = max(int(len(np.unique(patch_keys[inl]))), 1)
    return Fit(float(beta[0]), float(beta[1]) if use_slope else 0.0, rmse / np.sqrt(n_eff), rmse, int(inl.sum()), n_eff)


def inconsistency(rmse: float, p: AlphaParams) -> float:
    """Scatter beyond the sensor's own per-point noise (see module docstring, step 5)."""
    return float(np.sqrt(max(rmse ** 2 - p.point_noise_m ** 2, 0.0)))


@dataclass
class RawIndex:
    """Raw LiDAR points at wall heights, sorted along u and along v for fast band queries."""
    P: np.ndarray            # (N, 3) float64, aligned frame
    ray2: np.ndarray         # (N, 2) plan components of the viewing ray (camera -> point)
    h: np.ndarray            # height above the global floor
    order: dict[str, np.ndarray]
    keys: dict[str, np.ndarray]
    fit_lo: float            # wall faces are fitted on heights [fit_lo, fit_hi]; the index itself goes higher
    fit_hi: float            # so that door heads (lintels) can be found too


def build_raw_index(scene: dict, floor_y: float, fit_lo: float, fit_hi: float, top: float = 2.6) -> RawIndex:
    P = scene["raw_points"]
    h = P[:, 1] - floor_y
    m = (h > fit_lo) & (h < max(top, fit_hi))
    Pm = P[m].astype(np.float64)
    ray = scene["raw_ray"][m].astype(np.float32)
    order = {"x": np.argsort(P[m, 0]), "z": np.argsort(P[m, 2])}          # float32 sort: deterministic, 2x faster
    keys = {"x": Pm[order["x"], 0], "z": Pm[order["z"], 2]}
    return RawIndex(Pm, ray[:, [0, 2]], h[m], order, keys, fit_lo, fit_hi)


@dataclass
class FaceFit:
    offset: float            # wall position at the middle of the edge (u for family x, v for family z)
    slope: float             # d(offset)/d(along): the wall's tilt off the Manhattan axis
    mid: float               # along-coordinate where `offset` applies
    sigma: float             # 1-sigma of the offset: SE + LiDAR bias + surface inconsistency + half the drift budget
    se: float
    rmse: float | None
    n_points: int
    status: str              # measured | inferred


def _excluded(along: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    out = np.zeros(len(along), bool)
    for lo, hi in intervals:
        out |= (along > lo) & (along < hi)
    return out


def _choose_model(d: np.ndarray, keys: np.ndarray, x: np.ndarray, p: AlphaParams) -> Fit | None:
    """Axis-aligned by default. A tilted line is accepted only if it turns the face into a CLEAN plane: rmse at
    least 30% lower than the axis-aligned fit AND within 2x the per-point noise. Otherwise a tilt fitted to a
    smeared or cluttered wall is noise, and on these captures it skewed rooms into each other."""
    flat = robust_fit(d, keys, None, p.fit_band_m)
    tilt = robust_fit(d, keys, x, p.fit_band_m, float(np.tan(np.radians(p.max_tilt_deg))))
    if flat is None or tilt is None:
        return flat
    if tilt.rmse <= 0.7 * flat.rmse and tilt.rmse <= 2 * p.point_noise_m:
        return tilt
    return flat


def fit_face(idx: RawIndex, family: str, offset: float, inward: int, a: float, b: float,
             exclude: list[tuple[float, float]], p: AlphaParams) -> FaceFit:
    """Refit one wall face. family 'x': face is u = offset, runs along v in [a, b]; inward = +1 if the room is on
    the +u side. Falls back to the TSDF line ('inferred', wider sigma) when raw support is too thin."""
    ax, al = (0, 2) if family == "x" else (2, 0)
    mid = (a + b) / 2
    i0, i1 = np.searchsorted(idx.keys[family], [offset - p.fit_band_m, offset + p.fit_band_m])
    sel = idx.order[family][i0:i1]
    along = idx.P[sel, al]
    trim = min(p.corner_trim_m, 0.25 * abs(b - a))      # short walls keep their middle half
    ok = (along > min(a, b) + trim) & (along < max(a, b) - trim) & ~_excluded(along, exclude)
    ok &= (idx.h[sel] < idx.fit_hi) & (inward * idx.ray2[sel, 0 if family == "x" else 1] < -0.1)  # room side
    sel, along = sel[ok], along[ok]
    d = inward * (idx.P[sel, ax] - offset)                                 # + = into the room
    keys = np.floor(along / p.patch_m).astype(np.int64) * 100_000 + np.floor(idx.h[sel] / p.patch_m).astype(np.int64)
    fit = _choose_model(d, keys, along - mid, p) if len(sel) >= p.min_fit_points else None
    half_drift2 = p.drift_sigma_m ** 2 / 2
    if fit is None or fit.n_inliers < p.min_fit_points:
        return FaceFit(offset, 0.0, mid, float(np.sqrt(p.inferred_sigma_m ** 2 + half_drift2)), p.inferred_sigma_m,
                       None, 0 if fit is None else fit.n_inliers, "inferred")
    sigma = float(np.sqrt(fit.se ** 2 + p.lidar_sigma_m ** 2 + inconsistency(fit.rmse, p) ** 2 + half_drift2))
    return FaceFit(offset + inward * fit.value, inward * fit.slope, mid, sigma, fit.se, fit.rmse, fit.n_inliers,
                   "measured")
