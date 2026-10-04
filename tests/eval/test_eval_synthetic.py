"""Known-answer tests: perturb a synthetic plan by known amounts and check the metrics recover them exactly.

Run: PYTHONPATH=. .venv/bin/python -m pytest tests/eval -q
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from floorplan.eval.gates import (drift_gate, interval_calibration, opening_width_gate, photo_stitch_gate,
                                  reference_gates, room_overlaps, tier_wall_gate)
from floorplan.eval.match import match_plans
from floorplan.eval.register import RegisterParams, plan_evidence, register
from floorplan.eval.repeatability import ceiling_groups, classify_ceiling, repeatability, wall_tolerance
from synthetic import KNOWN_T, OpeningSpec, RoomSpec, base_specs, make_plan, moved, scale_plan


def _pair():
    """A = base plan. B = same rooms with known changes, expressed in a rotated + shifted frame.

    Changes: R1 wider by 7 mm (2 walls, within tolerance), R2 wider by 3 cm (2 walls, out of tolerance), R1 ceiling
    +5 mm (repeatable), R2 ceiling +2 cm (unrepeatable), d1 wider by 1.5 cm, window missing, one phantom opening."""
    rooms, ops, adj = base_specs()
    A = make_plan(rooms, ops, adj, capture_id="A")
    rb = copy.deepcopy(rooms)
    rb[0].x1 += 0.007
    rb[1].x1 += 0.03
    rb[0].ceiling += 0.005
    rb[1].ceiling += 0.02
    ob = [OpeningSpec("d1", "door", (2.0, 3.05), 0.815, ["R1", "C"]),
          OpeningSpec("d2", "door", (5.5, 3.05), 0.90, ["R2", "C"]),
          OpeningSpec("ghost", "door", (7.1, 1.5), 0.85, ["R2"])]
    B = moved(make_plan(rb, ob, adj, capture_id="B"))
    return A, B


def test_plan_registration_recovers_known_transform():
    A, B = _pair()
    reg = register(plan_evidence(A), plan_evidence(B), RegisterParams())
    T = np.array(reg.T2)
    assert abs(reg.yaw_deg - 91.3) < 0.05
    # translation is judged at the plan centre (a 7 mm / 3 cm geometry change shifts the best fit slightly)
    c = np.array([3.5, 2.0, 1.0])
    assert np.linalg.norm(T @ np.linalg.inv(KNOWN_T) @ c - c) < 0.02


def test_match_identity():
    A, _ = _pair()
    m = match_plans(A, A)
    assert len(m.rooms) == 3 and len(m.walls) == 12 and len(m.openings) == 3
    assert all(v == [] for v in m.unmatched.values())


def test_repeatability_recovers_known_deltas():
    A, B = _pair()
    m = match_plans(A, B, KNOWN_T)
    rep = repeatability(A, B, m, "A", "B", "lidar")
    by_wall = {w.wall_a: w for w in rep.walls}
    # R1 bottom/top walls (along x) +7 mm -> pass; R2 bottom/top +3 cm -> fail; all others unchanged
    for wid in ("R1_w0", "R1_w2"):
        assert by_wall[wid].delta_m == pytest.approx(0.007, abs=1e-9) and by_wall[wid].status == "pass"
    for wid in ("R2_w0", "R2_w2"):
        assert by_wall[wid].delta_m == pytest.approx(0.03, abs=1e-9) and by_wall[wid].status == "fail"
    others = [w for k, w in by_wall.items() if k not in ("R1_w0", "R1_w2", "R2_w0", "R2_w2")]
    assert len(others) == 8 and all(abs(w.delta_m) < 1e-9 for w in others)
    s = rep.summary()
    assert (s["walls_pass"], s["walls_fail"], s["walls_unmatched"], s["wall_gate"]) == (10, 2, 0, "fail")
    rooms = {r.room_a: r for r in rep.rooms}
    assert rooms["R1"].ceiling_spread_m == pytest.approx(0.005, abs=1e-9)
    assert rooms["R1"].ceiling_status == "repeatable_bias_unknown"
    assert rooms["R2"].ceiling_status == "unrepeatable"
    assert rooms["R2"].area_delta_m2 == pytest.approx(0.03 * 3.0, abs=1e-9)
    ops = {o.opening_a: o for o in rep.openings}
    assert ops["d1"].delta_m == pytest.approx(0.015, abs=1e-9)
    assert sorted(m.unmatched["openings_a"]) == ["win"] and m.unmatched["openings_b"] == ["ghost"]


def test_wall_tolerance_reading_of_or():
    assert wall_tolerance(1.0) == pytest.approx(0.01)       # short wall: 1 cm applies
    assert wall_tolerance(4.0) == pytest.approx(0.02)       # long wall: 0.5% applies


def test_matching_survives_local_drift():
    """Shift one room of B by 12 cm (as drift would) and check its walls are still matched and unchanged."""
    A, _ = _pair()
    B = copy.deepcopy(A)
    for w in B["walls"]:
        if w["room_id"] == "R2":
            w["p0"] = [w["p0"][0] + 0.12, w["p0"][1]]
            w["p1"] = [w["p1"][0] + 0.12, w["p1"][1]]
    B["rooms"][1]["polygon"] = [[x + 0.12, z] for x, z in B["rooms"][1]["polygon"]]
    m = match_plans(A, B)
    assert len(m.walls) == 12
    rep = repeatability(A, B, m)
    assert rep.summary()["wall_gate"] == "pass"


def test_ceiling_classification_and_groups():
    assert classify_ceiling([2.700, 2.708]) == "repeatable_bias_unknown"
    assert classify_ceiling([2.700, 2.708], gt_m=2.73) == "repeatable_but_biased"
    assert classify_ceiling([2.700, 2.708], gt_m=2.705) == "pass"
    assert classify_ceiling([2.700, 2.720]) == "unrepeatable"
    assert classify_ceiling([2.700, None]) == "not_verifiable"
    vals = {("a", "r1"): 2.70, ("b", "x"): 2.704, ("c", "k"): 2.712, ("a", "r2"): None}
    g = ceiling_groups(vals, [(("a", "r1"), ("b", "x")), (("b", "x"), ("c", "k"))])
    assert len(g) == 1 and g[0]["spread_m"] == pytest.approx(0.012) and g[0]["status"] == "unrepeatable"


def test_opening_gate_scores_detection():
    A, B = _pair()
    m = match_plans(A, B, KNOWN_T)
    g = opening_width_gate(A, B, m, "lidar_reference", "video", "x")
    # d1 off by 1.5 cm (hit), d2 exact (hit), window missed, ghost phantom -> 2 / 4
    assert g.value == pytest.approx(0.5) and g.n == 4 and g.status == "fail"


@pytest.mark.parametrize("scale,sigma,expected,why", [
    (1.02, 0.08, "pass", "within 3% and intervals cover the 14 cm error on the 7.1 m wall"),
    (1.02, 0.03, "fail", "within 3% but +-6 cm intervals miss a 14 cm error: confident garbage"),
    (1.04, 0.08, "fail", "4% scale error is outside the 3% gate"),
])
def test_video_wall_gate_detects_scale_error(scale, sigma, expected, why):
    rooms, ops, adj = base_specs()
    A = make_plan(rooms, ops, adj)
    V = scale_plan(make_plan(rooms, ops, adj, tier="video", sig_len=sigma), scale)
    g = tier_wall_gate(A, V, match_plans(A, V), "lidar_reference", "video", "x")
    assert g.status == expected, why


def test_calibration_detects_over_and_under_confidence():
    rng = np.random.default_rng(0)
    ref = rng.uniform(1, 5, 2000)
    sig = 0.01
    est = ref + rng.normal(0, sig, ref.size)
    for k, want in ((1.0, "calibrated"), (0.3, "overconfident"), (5.0, "underconfident")):
        c = interval_calibration(ref, est, est - 1.96 * k * sig, est + 1.96 * k * sig)
        assert c["status"] == want, (k, c)
    c = interval_calibration(ref, est, est - 1.96 * sig, est + 1.96 * sig)
    assert c["coverage_strict"] == pytest.approx(0.95, abs=0.015)


def test_photo_stitch_checks():
    rooms, ops, adj = base_specs()
    A = make_plan(rooms, ops, adj)
    P = make_plan(rooms, ops, adj, tier="photo")
    res = {g.gate: g for g in photo_stitch_gate(A, P, match_plans(A, P), "lidar_reference", "x")}
    assert all(g.status == "pass" for g in res.values()), res
    # overlapping rooms and a missing adjacency must fail their rows
    bad_rooms = [RoomSpec("R1", 0, 0, 4.5, 3.0), RoomSpec("R2", 4.1, 0, 7.1, 3.0), RoomSpec("C", 0, 3.1, 7.1, 4.3)]
    Pb = make_plan(bad_rooms, ops, adj[:2], tier="photo")
    assert room_overlaps(Pb)
    res = {g.gate: g for g in photo_stitch_gate(A, Pb, match_plans(A, Pb), "lidar_reference", "x")}
    assert res["photo_stitch.no_overlap"].status == "fail"
    assert res["photo_stitch.adjacency"].status == "fail"


def test_drift_gate_statuses():
    assert drift_gate({}, "lidar", "x").status == "not_verifiable"
    assert drift_gate({"drift": {"method": "none"}}, "lidar", "x").status == "fail"
    assert drift_gate({"drift": {"method": "pose_graph"}}, "lidar", "x").status == "fail"
    ok = {"drift": {"method": "pose_graph", "ablation": {"footprint_on_m2": 80.0, "footprint_off_m2": 82.0}}}
    assert drift_gate(ok, "lidar", "x").status == "pass"


def test_reference_gates_run_end_to_end():
    A, B = _pair()
    m = match_plans(A, B, KNOWN_T)
    out = reference_gates(A, B, m, "lidar_reference", "video", "x")
    assert {g.gate for g in out} >= {"opening_width", "ceiling_abs", "calibration", "wall_length_video"}
