"""MapAnything windows as a camera path (floorplan/video/path_mapanything.py): window layout, the robust fit and the
two chainings, on synthetic windows (no model, no GPU)."""
import numpy as np
from scipy.spatial.transform import Rotation

from floorplan.video import path_mapanything as PM

K = np.array([[200.0, 0, 64], [0, 200.0, 48], [0, 0, 1]])


def _walk(n, seed=0):
    """A camera walking along x and turning slowly, looking at a box room 3 m around it."""
    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        T[k, :3, :3] = Rotation.from_euler("y", 3.0 * k + rng.normal(0, 0.5), degrees=True).as_matrix()
        T[k, :3, 3] = [0.08 * k, 0.01 * np.sin(k), 0.02 * k]
    return T


def _depth(T_wc, h=96, w=128):
    """z-depth of a room [-2, 5.5] x [-1.4, 1.2] x [-2, 4] m seen from camera T_wc (ray cast on the 6 walls)."""
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    R, c = T_wc[:3, :3], T_wc[:3, 3]
    dw = d @ R.T
    best = np.full((h, w), np.inf)
    for ax, lo, hi in ((0, -2, 5.5), (1, -1.4, 1.2), (2, -2, 4)):
        for plane in (lo, hi):
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = (plane - c[ax]) / dw[..., ax]
            lam[~np.isfinite(lam) | (lam <= 0)] = np.inf
            best = np.minimum(best, lam)
    return best.astype(np.float32)          # lam is the z-depth: the ray's z component is 1


def _windows(T, bounds, noise_rot_deg=0.0, scale=None, seed=0):
    rng = np.random.default_rng(seed)
    wins = []
    for wi, (a, b) in enumerate(bounds):
        idx = np.arange(a, b)
        T0inv = np.linalg.inv(T[a])
        Tw = np.einsum("ij,njk->nik", T0inv, T[idx])
        if noise_rot_deg:
            Tw[:, :3, :3] = np.einsum("nij,njk->nik", Rotation.from_rotvec(
                rng.normal(0, np.radians(noise_rot_deg), (len(idx), 3))).as_matrix(), Tw[:, :3, :3])
        s = 1.0 if scale is None else scale[wi]
        Tw[:, :3, 3] *= s
        D = np.stack([_depth(T[k]) * s for k in idx])[:, ::PM.SUB, ::PM.SUB]
        wins.append(dict(idx=idx, T=Tw, K=np.repeat(K[None], len(idx), 0), depth=D.astype(np.float16),
                         conf=np.ones_like(D, np.float16)))
    return wins


def _relative(T):
    return np.einsum("ij,njk->nik", np.linalg.inv(T[0]), T)


def test_window_bounds_cover_and_overlap():
    for n, size, ov in ((228, 24, 8), (164, 16, 4), (30, 32, 8), (100, 32, 8)):
        b = PM.window_bounds(n, size, ov)
        assert b[0][0] == 0 and b[-1][1] == n
        assert all(bb - aa == min(size, n) for aa, bb in b)
        assert all(b[i + 1][0] <= b[i][1] - ov for i in range(len(b) - 1))


def test_robust_fit_recovers_similarity_with_outliers():
    rng = np.random.default_rng(1)
    src = rng.uniform(-2, 2, (2000, 3))
    R = Rotation.from_euler("xyz", [10, -25, 40], degrees=True).as_matrix()
    dst = 1.07 * src @ R.T + [0.3, -0.2, 1.5]
    dst[:300] += rng.uniform(-1, 1, (300, 3))                 # 15% outliers
    s, R_, t, r = PM.robust_fit(src, dst, np.ones(len(src)), with_scale=True)
    assert abs(s - 1.07) < 1e-3
    assert np.degrees(np.arccos((np.trace(R_.T @ R) - 1) / 2)) < 0.05
    assert np.median(r[300:]) < 1e-3
    s, *_ = PM.robust_fit(src[300:], dst[300:] / 1.07, np.ones(1700), with_scale=False)
    assert s == 1.0


def test_points_chain_recovers_the_walk():
    T = _walk(60)
    wins = _windows(T, PM.window_bounds(60, 16, 6))
    res = PM.chain_windows(wins)
    assert all(lk["linked"] for lk in res["links"])
    assert (res["segment"] == 0).all()
    err = np.linalg.norm(res["T"][:, :3, 3] - _relative(T)[:, :3, 3], axis=1)
    assert err.max() < 1e-3


def test_points_chain_sim3_follows_a_window_scale_change_se3_keeps_metres():
    T = _walk(40)
    bounds = PM.window_bounds(40, 16, 6)
    wins = _windows(T, bounds, scale=[1.0, 1.25, 1.25, 1.25][:len(bounds)])
    se3 = PM.chain_windows(wins, with_scale=False)
    sim3 = PM.chain_windows(wins, with_scale=True)
    assert abs(sim3["links"][0]["s"] - 0.8) < 0.01            # the window was 25% too large: Sim3 shrinks it
    assert se3["links"][0]["s"] == 1.0
    assert not se3["links"][0]["linked"] or se3["links"][0]["median_resid_m"] > 0.0


def test_points_chain_cuts_at_a_broken_window():
    T = _walk(60)
    bounds = PM.window_bounds(60, 16, 6)
    wins = _windows(T, bounds)
    bad = wins[2]
    bad["T"] = bad["T"].copy()
    bad["T"][:, :3, 3] *= 4.0                     # its camera steps no longer match its own depth
    res = PM.chain_windows(wins)
    assert [lk["linked"] for lk in res["links"]][1] is False
    assert len(np.unique(res["segment"])) >= 2


def test_anchored_chain_places_windows_by_view0():
    T = _walk(50)
    bounds = PM.window_bounds(50, 16, 6)
    wins = _windows(T, bounds)
    res = PM.chain_anchored(wins)
    assert all(lk["linked"] for lk in res["links"])
    err = np.linalg.norm(res["T"][:, :3, 3] - _relative(T)[:, :3, 3], axis=1)
    assert err.max() < 1e-6


def test_depth_on_model_grid_is_a_crop_and_scale():
    W0, H0 = 1024, 576
    D = np.tile(np.arange(504, dtype=np.float32), (284, 1))     # depth = column index at 504 px
    from floorplan.photo.recon import model_size
    Wm, Hm = model_size(W0, H0)
    a = Wm / Hm
    ox = round((W0 - H0 * a) / 2)
    s = Wm / (W0 - 2 * ox)
    out = PM.depth_on_model_grid(D, (W0, H0), (float(ox), 0.0, s), (Wm, Hm))
    assert out.shape == (Hm, Wm)
    u_full = (np.arange(Wm) + 0.5) / s + ox
    assert np.abs(out[0] - np.floor(u_full * 504 / W0)).max() <= 1


def test_fill_from_windows_places_missing_keyframes_on_the_global_path():
    T = _walk(40)
    wins = _windows(T, PM.window_bounds(40, 16, 6))
    G = T.copy()
    G[[12, 13, 14, 30]] = np.nan                       # keyframes the global path (SfM) did not register
    F, info = PM.fill_from_windows(G, wins)
    assert set(info) == {12, 13, 14, 30}
    assert np.abs(F[[12, 13, 14, 30]] - T[[12, 13, 14, 30]]).max() < 1e-6
    assert all(v["anchor_misfit_m"] < 1e-6 for v in info.values())
