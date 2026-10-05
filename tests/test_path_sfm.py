"""Camera path from global SfM (floorplan/video/path_sfm.py): pairs, MoGe-2 scale, and the joining of models."""
import numpy as np
from scipy.spatial.transform import Rotation

from floorplan.video import path_sfm as ps
from floorplan.video.evaluate import umeyama
from floorplan.video.params import VideoParams


def _walk(n=60):
    T = np.tile(np.eye(4), (n, 1, 1))
    k = np.arange(n)
    T[:, :3, :3] = Rotation.from_euler("y", 0.05 * k[:, None]).as_matrix()
    T[:, :3, 3] = np.stack([0.1 * k, 0.02 * np.sin(0.3 * k), 0.5 * np.sin(0.1 * k)], 1)
    return T


def _rigid(seed):
    rng = np.random.default_rng(seed)
    G = np.eye(4)
    G[:3, :3] = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
    G[:3, 3] = rng.normal(size=3) * 3
    return G


def _model(mid, T_true, ids, scale, seed):
    """Keyframes `ids` of the true walk in a model frame: an arbitrary rigid motion, positions in SfM units."""
    G = _rigid(seed)
    T = np.einsum("ij,njk->nik", G, T_true[ids])
    T[:, :3, 3] /= scale
    obs = [(np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0)) for _ in ids]
    return ps.Model(id=mid, kf=np.asarray(ids), T=T, obs=obs, n_points=0, focal=500.0, extra=[0.0])


def _dpvo(T_true, scale=0.3, jump_at=None, jump=1.0, seed=7):
    """DPVO's path: the true walk in its own frame and units; optionally its scale jumps after keyframe `jump_at`."""
    G = _rigid(seed)
    T = T_true.copy()
    C = T[:, :3, 3].copy()
    steps = np.diff(C, axis=0) * scale
    if jump_at is not None:
        steps[jump_at:] *= jump
    T[:, :3, 3] = np.vstack([C[:1] * scale, C[:1] * scale + np.cumsum(steps, axis=0)])
    return np.einsum("ij,njk->nik", G, T)


def test_pairs():
    w = ps.window_pairs(10, 3)
    assert (0, 3) in w and (0, 4) not in w and len(w) == 7 * 3 + 2 + 1
    s = ps.stride_pairs(20, 4, 3)
    assert (0, 19) in s and (0, 2) not in s and all(abs(i - j) > 3 for i, j in s)
    assert ps.stride_pairs(20, 0, 3) == set()
    G = np.eye(6)[[0, 1, 2, 0, 1, 2]].astype(float)          # keyframe 3 looks like 0, 4 like 1, 5 like 2
    r = ps.retrieval_pairs(G / np.linalg.norm(G, axis=1, keepdims=True), 1, 1)
    assert {(0, 3), (1, 4), (2, 5)} <= r


def test_depth_ratio_scale():
    n, w, h = 5, 100, 50
    D = np.zeros((n, h // 2, w // 2), np.float32)              # depth maps at half the SfM resolution
    rng = np.random.default_rng(0)
    obs = []
    for k in range(n):
        xy = rng.uniform([0, 0], [w, h], size=(40, 2))
        z = rng.uniform(1, 3, 40)
        D[k, np.floor(xy[:, 1] / 2).astype(int), np.floor(xy[:, 0] / 2).astype(int)] = 2.5 * z
        obs.append((xy, z, np.full(40, 0.5), np.full(40, 4)))
    m = ps.Model(id=0, kf=np.arange(n), T=np.tile(np.eye(4), (n, 1, 1)), obs=obs, n_points=200, focal=50.0,
                 extra=[0.0])
    r = ps.keyframe_depth_ratios(m, D, w)
    assert np.allclose(r, 2.5, rtol=1e-5)
    s = ps.model_scale(r * np.array([0.98, 1.0, 1.02, 1.0, 1.01]))
    assert abs(s["scale"] / 2.5 - 1) < 0.01 and s["keyframes"] == 5 and s["sigma_rel_stat"] >= 0
    obs[0] = (obs[0][0], obs[0][1], np.full(40, 5.0), obs[0][3])  # reprojection error too large: no ratio
    assert np.isnan(ps.keyframe_depth_ratios(m, D, w)[0])


def test_two_models_joined_through_a_consistent_link():
    T_true = _walk()
    A = _model(0, T_true, np.arange(0, 30), 2.0, 1)
    B = _model(1, T_true, np.arange(32, 60), 0.5, 2)
    scales = {0: dict(scale=2.0, keyframes=30), 1: dict(scale=0.5, keyframes=28)}
    ts = np.arange(60) * 0.5
    out = ps.assemble_path([A, B], scales, _dpvo(T_true), ts, VideoParams())
    s, R, t = umeyama(out["T_wc"][:, :3, 3], T_true[:, :3, 3], False)
    err = np.linalg.norm(out["T_wc"][:, :3, 3] @ R.T + t - T_true[:, :3, 3], axis=1)
    assert err.max() < 1e-6
    assert out["report"]["groups"] == 1 and (out["segment"] == 0).all()
    assert list(out["source"][30:32]) == ["fill", "fill"] and out["placed"].all()
    assert out["report"]["links"][0]["consistent"]


def test_link_across_a_dpvo_scale_jump_is_not_joined():
    T_true = _walk()
    A = _model(0, T_true, np.arange(0, 30), 2.0, 1)
    B = _model(1, T_true, np.arange(32, 60), 0.5, 2)
    scales = {0: dict(scale=2.0, keyframes=30), 1: dict(scale=0.5, keyframes=28)}
    out = ps.assemble_path([A, B], scales, _dpvo(T_true, jump_at=31, jump=4.0), np.arange(60) * 0.5, VideoParams())
    assert out["report"]["groups"] == 2
    assert (out["segment"][:30] == 0).all() and (out["segment"][32:] == 1).all()
    assert not out["placed"][30:32].any()                      # the gap between two groups stays out of the fusion
    link = out["report"]["links"][0]
    assert not link["consistent"] and any("scale ratio" in r for r in link["reasons"])
    # each group keeps its own shape: B alone matches the truth after a rigid fit
    s, R, t = umeyama(out["T_wc"][32:, :3, 3], T_true[32:, :3, 3], False)
    assert np.abs(out["T_wc"][32:, :3, 3] @ R.T + t - T_true[32:, :3, 3]).max() < 1e-6


def test_scale_report_matches_segments():
    T_true = _walk()
    A = _model(0, T_true, np.arange(0, 30), 2.0, 1)
    B = _model(1, T_true, np.arange(32, 60), 0.5, 2)
    scales = {0: dict(scale=2.0, keyframes=30, sigma_rel_stat=0.01), 1: dict(scale=0.5, keyframes=28,
                                                                             sigma_rel_stat=0.02)}
    out = ps.assemble_path([A, B], scales, _dpvo(T_true, jump_at=31, jump=4.0), np.arange(60) * 0.5, VideoParams())
    out.update(models=[A, B], scales=scales, stats=dict(pairs=100))
    sc = ps.scale_report(out, 60, VideoParams())
    assert np.all(sc["s_local"] == 1.0) and [s["id"] for s in sc["segments"]] == [0, 1]
    assert sc["segments"][0]["sigma_rel_stat"] == 0.01 and sc["segments"][1]["sigma_rel_stat"] == 0.02


def test_local_scale_check_finds_a_bent_stretch():
    r = np.full(80, 2.0) * np.exp(np.random.default_rng(1).normal(0, 0.05, 80))
    keep, rep = ps.local_scale_check(r, 2.0)
    assert keep.all() and rep["bent_keyframes"] == 0
    r[40:60] *= 2.85                                          # a stretch folded in at the wrong size
    keep, rep = ps.local_scale_check(r, float(np.median(r)))
    assert not keep[45:55].any() and keep[:30].all() and keep[70:].all() and rep["local_scale_max_dev"] > 2


def test_largest_coverage():
    T_true = _walk()
    A = _model(0, T_true, np.arange(0, 50), 1.0, 1)
    B = _model(1, T_true, np.arange(50, 60), 1.0, 2)
    assert abs(ps.largest_coverage([A, B], 60) - 50 / 60) < 1e-9
    assert ps.largest_coverage([], 60) == 0.0
    assert VideoParams().path_sfm_min_coverage == 0.8
