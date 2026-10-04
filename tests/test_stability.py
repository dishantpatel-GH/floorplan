"""Stability part: the drift ICP is bit-reproducible, and the plan extractor's raster-phase vote helpers behave.

Run: env -u PYTHONPATH .venv/bin/python -m pytest tests/test_stability.py -q
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from floorplan.model import Measurement, Plan, Wall
from floorplan.plan.beta.extract import vote_phases, wall_agreement
from floorplan.plan.beta.grid import Grid
from floorplan.recon import drift as D


def _room_cloud(seed: int, n: int = 60000) -> o3d.geometry.PointCloud:
    """A 4 x 3 x 2.5 m box room (floor + 4 walls), noisy, as one fragment would see it."""
    rng = np.random.default_rng(seed)
    faces = []
    for axis, val in ((0, 0.0), (0, 4.0), (2, 0.0), (2, 3.0), (1, 0.0)):
        P = rng.uniform([0, 0, 0], [4, 2.5, 3], (n // 5, 3))
        P[:, axis] = val + rng.normal(0, 0.005, n // 5)
        faces.append(P)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(faces)))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    pc.orient_normals_towards_camera_location(np.array([2.0, 1.2, 1.5]))
    return pc


def _frag(k: int, cloud) -> D.Fragment:
    return D.Fragment(k, np.array([k]), k, float(k), 1.0, np.eye(4), cloud)


def test_icp_bit_reproducible_in_parallel_pool():
    """register() inside deterministic_open3d gives identical transforms every time, also from Python threads."""
    from concurrent.futures import ThreadPoolExecutor
    p = D.DriftParams()
    a, b = _frag(0, _room_cloud(0)), _frag(1, _room_cloud(1))
    T0 = np.eye(4)
    T0[:3, 3] = [0.03, -0.01, 0.02]
    with D.deterministic_open3d():
        ref = D.register(a, b, T0, p, "local").T
        with ThreadPoolExecutor(4) as ex:
            outs = list(ex.map(lambda _: D.register(a, b, T0, p, "local").T, range(8)))
    for T in outs:
        assert np.array_equal(T, ref)
    assert np.abs(ref[:3, 3]).max() < 0.01          # converged back onto the same room


def test_deterministic_open3d_restores_thread_limit():
    before = o3d.utility.get_max_threads()
    with D.deterministic_open3d():
        assert o3d.utility.get_max_threads() == 1
    assert o3d.utility.get_max_threads() == before


def test_vote_phases():
    assert vote_phases(4) == [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
    ph = vote_phases(5)
    assert ph[0] == (0.0, 0.0) and len(set(ph)) == 5 and all(0 <= u < 1 and 0 <= v < 1 for u, v in ph)


def test_grid_phase_moves_cell_boundaries():
    uv = np.array([[0.0, 0.0], [3.0, 2.0]])
    g0 = Grid.around(uv, 0.02)
    g1 = Grid.around(uv, 0.02, phase=(0.5, 0.25))
    assert np.isclose(g0.u0 - g1.u0, 0.01) and np.isclose(g0.v0 - g1.v0, 0.005)


def _plan(walls):
    m = Measurement.from_sigma(1.0, 0.01, "m", "test")
    ws = [Wall(f"W{i}", "R1", p0, p1, m, n) for i, (p0, p1, n) in enumerate(walls)]
    return Plan("c", "lidar", [], ws, [], [], m)


def test_wall_agreement_gate():
    a = _plan([((0, 0), (4, 0), (0, 1)), ((4, 0), (4, 3), (-1, 0))])
    b = _plan([((0.004, 0), (4.006, 0), (0, 1)),       # same wall, ends within 1 cm
               ((4, 0), (4, 1.5), (-1, 0)),             # split wall: other topology
               ((4, 1.5), (4, 3), (-1, 0))])
    ia, ib = wall_agreement(a, b, 0.01)
    assert ia.tolist() == [True, False] and ib.tolist() == [True, False, False]
    ia, ib = wall_agreement(a, _plan([((4, 0), (0, 0), (0, -1))]), 0.01)   # opposite facing never agrees
    assert not ia.any() and not ib.any()
