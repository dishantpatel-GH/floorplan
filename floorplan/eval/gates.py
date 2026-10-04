"""Every Part 2 gate of the case study as data, plus the functions that decide pass / fail / not verifiable.

Why gates are code and not prose: the fix loop (Part 4) needs "the single worst-performing gate with the failing
number", regenerable before and after a fix. That is only possible when every gate is computed the same way every
run, from files on disk.

There is no ground truth (GT) for the sample data. Each result therefore states its *basis*:
  ground_truth     compared with tape/laser measurements (a GT plan.json in the same schema). None exist yet.
  lidar_reference  video/photo tier compared with the LiDAR-tier result of the same capture. This is NOT ground
                   truth: it can only show disagreement with LiDAR, and it inherits LiDAR's own error
                   (labelled everywhere so it is never presented as accuracy).
  repeat_pair      two captures of the same rooms at the same tier (needs no GT).
  declaration      the plan's own metadata (drift handling).
  none             nothing available -> status not_verifiable, with the reason.

Statuses: pass | fail | not_verifiable | not_applicable.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np
from shapely.geometry import Polygon

from floorplan.eval.match import MatchResult, as_plan_dict


@dataclass(frozen=True)
class GateSpec:
    id: str
    row: str                 # the Part 2 table row / sentence it encodes
    requirement: str
    tiers: tuple[str, ...]
    needs_gt: bool
    threshold: dict


GATES: tuple[GateSpec, ...] = (
    GateSpec("opening_width", "Opening widths",
             "|width error| <= 2 cm on >= 85% of openings; a missed and a phantom opening each count as a miss",
             ("lidar", "video", "photo"), True, {"abs_m": 0.02, "min_rate": 0.85}),
    GateSpec("ceiling_abs", "Ceiling height", "<= 1.5 cm per room", ("lidar", "video", "photo"), True,
             {"abs_m": 0.015}),
    GateSpec("ceiling_spread", "Ceiling height",
             "room captured more than once: spread across captures <= 1 cm; report says repeatable-but-biased or "
             "unrepeatable", ("lidar", "video", "photo"), False, {"spread_m": 0.01}),
    GateSpec("repeatability", "Repeatability",
             "two captures of the same room at the same tier agree within 1 cm or 0.5% per wall",
             ("lidar", "video", "photo"), False, {"abs_m": 0.01, "rel": 0.005}),
    GateSpec("drift", "Drift accountability",
             "state the drift handling and show an on/off ablation of the stitched footprint; 'poses used as-is' "
             "is an automatic fail", ("lidar", "video", "photo"), False, {}),
    GateSpec("photo_stitch", "Photo-tier whole-property stitch",
             "per-room folders -> one stitched plan, correct adjacency, no room overlaps, footprint within +-8% "
             "with calibrated intervals", ("photo",), True, {"footprint_rel": 0.08}),
    GateSpec("wall_length_photo", "Photo tier", "wall lengths within +-8% with calibrated intervals", ("photo",),
             True, {"rel": 0.08}),
    GateSpec("wall_length_video", "Video tier", "wall lengths within +-3% with calibrated intervals", ("video",),
             True, {"rel": 0.03}),
    GateSpec("calibration", "All tiers", "calibration is scored at every tier: 95% intervals must cover the truth "
             "~95% of the time without being uselessly wide", ("lidar", "video", "photo"), True, {"nominal": 0.95}),
    GateSpec("round1_lidar", "Round 1 gates", "Round 1 gates apply (thresholds not given in the Round 2 brief)",
             ("lidar",), True, {}),
)
GATE_BY_ID = {g.id: g for g in GATES}
TIER_WALL_GATE = {"photo": "wall_length_photo", "video": "wall_length_video"}
NO_DRIFT_METHODS = {"", "none", "as-is", "as_is", "poses_as_is", "off"}


@dataclass
class GateResult:
    gate: str
    tier: str
    capture: str
    status: str
    basis: str
    value: float | None = None
    threshold: str = ""
    n: int = 0
    detail: str = ""
    producer: str = ""
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        return d


# ----------------------------------------------------------------------------------------------- calibration

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than +-1.96 sqrt(p(1-p)/n) at small n or p ~ 1)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(c - h), float(c + h)


def interval_calibration(ref: np.ndarray, value: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                         ref_sigma: np.ndarray | None = None, nominal: float = 0.95) -> dict:
    """Empirical coverage of predicted 95% intervals.

    strict coverage       reference inside [lo, hi]; right when the reference is exact (GT).
    noise-aware coverage  |value - ref| <= 1.96 sqrt(sigma_pred^2 + sigma_ref^2); right when the reference has its
                          own error (the LiDAR reference, or the other capture of a repeat pair).
    status                overconfident  if the Wilson upper bound of coverage < nominal, or RMS z > 2 with
                                         n >= 10 (intervals more than 2x too narrow: "confident garbage", which
                                         the brief penalises hardest);
                          underconfident if RMS z < 0.5 with n >= 10 (intervals more than 2x too wide);
                          calibrated     otherwise; insufficient_n below 5 samples.
    """
    ref, value, lo, hi = (np.asarray(x, float) for x in (ref, value, lo, hi))
    n = len(ref)
    if n == 0:
        return dict(n=0, status="not_verifiable")
    sig_p = (hi - lo) / 2 / 1.96
    sig_r = np.zeros(n) if ref_sigma is None else np.asarray(ref_sigma, float)
    z = (value - ref) / np.sqrt(np.maximum(sig_p ** 2 + sig_r ** 2, 1e-18))
    k_strict = int(np.sum((ref >= lo) & (ref <= hi)))
    k_noise = int(np.sum(np.abs(z) <= 1.96))
    k = k_strict if ref_sigma is None else k_noise
    w_lo, w_hi = wilson(k, n)
    rms_z = float(np.sqrt(np.mean(z ** 2)))
    if n < 5:
        status = "insufficient_n"
    elif w_hi < nominal or (n >= 10 and rms_z > 2.0):
        status = "overconfident"
    elif n >= 10 and rms_z < 0.5:
        status = "underconfident"
    else:
        status = "calibrated"
    return dict(n=n, coverage_strict=k_strict / n, coverage_noise_aware=k_noise / n, coverage_used=k / n,
                wilson_lo=w_lo, wilson_hi=w_hi, rms_z=rms_z, mean_half_width=float(np.mean(hi - lo) / 2),
                mean_rel_half_width=float(np.mean((hi - lo) / 2 / np.maximum(np.abs(value), 1e-9))), status=status)


def _meas(m: dict | None) -> tuple[float | None, float | None, float | None]:
    if not m or m.get("value") is None:
        return None, None, None
    return float(m["value"]), m.get("lo"), m.get("hi")


def collect_pairs(ref: dict, test: dict, match: MatchResult) -> dict[str, list[tuple]]:
    """(ref Measurement, test Measurement) pairs for every matched element, by quantity."""
    rw, tw = {w["id"]: w for w in ref["walls"]}, {w["id"]: w for w in test["walls"]}
    rr, tr = {r["id"]: r for r in ref["rooms"]}, {r["id"]: r for r in test["rooms"]}
    ro, to = {o["id"]: o for o in ref["openings"]}, {o["id"]: o for o in test["openings"]}
    return {
        "wall_length": [(rw[p.a].get("length"), tw[p.b].get("length")) for p in match.walls],
        "floor_area": [(rr[p.a].get("floor_area"), tr[p.b].get("floor_area")) for p in match.rooms],
        "ceiling_height": [(rr[p.a].get("ceiling_height"), tr[p.b].get("ceiling_height")) for p in match.rooms],
        "opening_width": [(ro[p.a].get("width"), to[p.b].get("width")) for p in match.openings],
        "footprint": [(ref.get("footprint_area"), test.get("footprint_area"))],
    }


def calibration_from_pairs(pairs: list[tuple], ref_is_exact: bool) -> dict:
    rows = []
    for r, t in pairs:
        rv, rlo, rhi = _meas(r)
        tv, tlo, thi = _meas(t)
        if rv is None or tv is None or tlo is None or thi is None:
            continue
        rs = (rhi - rlo) / 2 / 1.96 if (rlo is not None and rhi is not None and not ref_is_exact) else 0.0
        rows.append((rv, tv, tlo, thi, rs))
    if not rows:
        return dict(n=0, status="not_verifiable")
    a = np.array(rows)
    return interval_calibration(a[:, 0], a[:, 1], a[:, 2], a[:, 3], None if ref_is_exact else a[:, 4])


# ----------------------------------------------------------------------------------------------- gates vs a reference

def opening_width_gate(ref: dict, test: dict, match: MatchResult, basis: str, tier: str, capture: str) -> GateResult:
    spec = GATE_BY_ID["opening_width"]
    ro, to = {o["id"]: o for o in ref["openings"]}, {o["id"]: o for o in test["openings"]}
    hits = 0
    for p in match.openings:
        a, b = _meas(ro[p.a].get("width"))[0], _meas(to[p.b].get("width"))[0]
        hits += int(a is not None and b is not None and abs(a - b) <= spec.threshold["abs_m"])
    missed, phantom = len(match.unmatched.get("openings_a", [])), len(match.unmatched.get("openings_b", []))
    n = len(match.openings) + missed + phantom
    thr = "|dw| <= 2 cm on >= 85%"
    if n == 0:
        return GateResult(spec.id, tier, capture, "not_verifiable", basis, None, thr, 0, "no openings in either plan")
    rate = hits / n
    return GateResult(spec.id, tier, capture, "pass" if rate >= spec.threshold["min_rate"] else "fail", basis, rate,
                      thr, n, f"{hits} within 2 cm, {len(match.openings) - hits} matched but off, {missed} missed, "
                          f"{phantom} phantom")


def ceiling_abs_gate(ref: dict, test: dict, match: MatchResult, basis: str, tier: str, capture: str) -> GateResult:
    spec = GATE_BY_ID["ceiling_abs"]
    rr, tr = {r["id"]: r for r in ref["rooms"]}, {r["id"]: r for r in test["rooms"]}
    ok, bad, unseen_test = 0, 0, 0
    for p in match.rooms:
        a, b = _meas(rr[p.a].get("ceiling_height"))[0], _meas(tr[p.b].get("ceiling_height"))[0]
        if a is None:
            continue
        if b is None:
            unseen_test += 1
        elif abs(a - b) <= spec.threshold["abs_m"]:
            ok += 1
        else:
            bad += 1
    n = ok + bad + unseen_test
    if n == 0:
        return GateResult(spec.id, tier, capture, "not_verifiable", basis, None, "<= 1.5 cm per room", 0,
                          "reference has no observed ceiling in any matched room")
    status = "pass" if bad == 0 and unseen_test == 0 else "fail"
    return GateResult(spec.id, tier, capture, status, basis, ok / n, "<= 1.5 cm per room", n,
                      f"{ok} within 1.5 cm, {bad} off, {unseen_test} not observed by the tested plan")


def tier_wall_gate(ref: dict, test: dict, match: MatchResult, basis: str, tier: str, capture: str) -> GateResult:
    """Photo +-8% / video +-3% wall-length gate. Pass = every matched wall within tolerance AND intervals not
    overconfident. Wall recall (matched / reference walls) is reported because a plan that drops hard walls
    could otherwise pass on the easy ones."""
    gid = TIER_WALL_GATE.get(tier)
    if gid is None:
        return GateResult("wall_length_tier", tier, capture, "not_applicable", basis, detail="no tier wall gate")
    spec = GATE_BY_ID[gid]
    rw, tw = {w["id"]: w for w in ref["walls"]}, {w["id"]: w for w in test["walls"]}
    rel = []
    for p in match.walls:
        a, b = _meas(rw[p.a].get("length"))[0], _meas(tw[p.b].get("length"))[0]
        if a and b is not None:
            rel.append(abs(b - a) / a)
    thr = f"+-{spec.threshold['rel'] * 100:.0f}% per wall, calibrated"
    if not rel:
        return GateResult(spec.id, tier, capture, "not_verifiable", basis, None, thr, 0, "no matched walls")
    cal = calibration_from_pairs(collect_pairs(ref, test, match)["wall_length"], basis == "ground_truth")
    within = float(np.mean(np.array(rel) <= spec.threshold["rel"]))
    recall = len(match.walls) / max(len(ref["walls"]), 1)
    status = "pass" if within == 1.0 and cal.get("status") != "overconfident" else "fail"
    return GateResult(spec.id, tier, capture, status, basis, within, thr, len(rel),
                      f"max rel err {max(rel) * 100:.1f}%, wall recall {recall:.2f}, calibration {cal.get('status')}",
                      extra=dict(calibration=cal))


def photo_stitch_gate(ref: dict | None, test: dict, match: MatchResult | None, basis: str,
                      capture: str) -> list[GateResult]:
    """Sub-checks of the photo stitch row. 'One plan' and 'no overlaps' need no reference; adjacency and footprint
    are compared with the reference (LiDAR tier or GT)."""
    spec, tier, out = GATE_BY_ID["photo_stitch"], "photo", []
    rooms = test.get("rooms", [])
    out.append(GateResult(f"{spec.id}.one_plan", tier, capture,
                          "pass" if len(rooms) >= 2 and _connected(test) else "fail", "self_check", len(rooms),
                          ">= 2 rooms, adjacency graph connected", len(rooms)))
    overl = room_overlaps(test)
    out.append(GateResult(f"{spec.id}.no_overlap", tier, capture, "pass" if not overl else "fail", "self_check",
                          len(overl), "0 overlapping room pairs (> 1% of the smaller room)", len(rooms),
                          "; ".join(f"{a}/{b}: {x:.2f} m2" for a, b, x in overl)))
    if ref is None or match is None:
        for sub in ("adjacency", "footprint"):
            out.append(GateResult(f"{spec.id}.{sub}", tier, capture, "not_verifiable", "none",
                                  detail="no reference plan for this capture"))
        return out
    prec, rec = adjacency_agreement(ref, test, match)
    out.append(GateResult(f"{spec.id}.adjacency", tier, capture, "pass" if prec == rec == 1.0 else "fail", basis,
                          min(prec, rec), "precision = recall = 1 over matched rooms", len(match.rooms),
                          f"precision {prec:.2f}, recall {rec:.2f}"))
    rv, (tv, tlo, thi) = _meas(ref.get("footprint_area"))[0], _meas(test.get("footprint_area"))
    if rv is None or tv is None:
        out.append(GateResult(f"{spec.id}.footprint", tier, capture, "not_verifiable", basis,
                              detail="footprint missing"))
    else:
        rel = abs(tv - rv) / rv
        covered = tlo is not None and thi is not None and tlo <= rv <= thi
        out.append(GateResult(f"{spec.id}.footprint", tier, capture,
                              "pass" if rel <= spec.threshold["footprint_rel"] and covered else "fail", basis, rel,
                              "+-8% and reference inside the 95% interval", 1,
                              f"test {tv:.2f} m2 [{tlo}, {thi}] vs reference {rv:.2f} m2; covered={covered}"))
    return out


def room_overlaps(plan: dict, rel_tol: float = 0.01) -> list[tuple[str, str, float]]:
    polys = [(r["id"], Polygon(r["polygon"]).buffer(0)) for r in plan.get("rooms", []) if len(r["polygon"]) >= 3]
    out = []
    for (ia, a), (ib, b) in itertools.combinations(polys, 2):
        x = a.intersection(b).area
        if x > rel_tol * min(a.area, b.area):
            out.append((ia, ib, float(x)))
    return out


def _connected(plan: dict) -> bool:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(r["id"] for r in plan.get("rooms", []))
    g.add_edges_from((a["room_a"], a["room_b"]) for a in plan.get("adjacency", []))
    return g.number_of_nodes() > 0 and nx.is_connected(g)


def adjacency_agreement(ref: dict, test: dict, match: MatchResult) -> tuple[float, float]:
    """Precision/recall of the test plan's room-adjacency edges against the reference, over matched rooms."""
    t2r = {p.b: p.a for p in match.rooms}
    matched_ref = set(t2r.values())
    ref_edges = {frozenset((a["room_a"], a["room_b"])) for a in ref.get("adjacency", [])
                 if a["room_a"] in matched_ref and a["room_b"] in matched_ref}
    test_edges = {frozenset((t2r[a["room_a"]], t2r[a["room_b"]])) for a in test.get("adjacency", [])
                  if a["room_a"] in t2r and a["room_b"] in t2r}
    tp = len(ref_edges & test_edges)
    return (tp / len(test_edges) if test_edges else 1.0), (tp / len(ref_edges) if ref_edges else 1.0)


def calibration_gate(ref: dict, test: dict, match: MatchResult, basis: str, tier: str, capture: str) -> GateResult:
    pairs = collect_pairs(ref, test, match)
    allp = [x for k in ("wall_length", "floor_area", "ceiling_height", "opening_width") for x in pairs[k]]
    cal = calibration_from_pairs(allp, basis == "ground_truth")
    status = {"calibrated": "pass", "overconfident": "fail", "underconfident": "fail"}.get(cal["status"],
                                                                                          "not_verifiable")
    per_kind = {k: calibration_from_pairs(v, basis == "ground_truth") for k, v in pairs.items()}
    return GateResult("calibration", tier, capture, status, basis, cal.get("coverage_used"),
                      "95% intervals cover ~95% (Wilson CI contains 0.95), 0.5 <= RMS z <= 2", cal.get("n", 0),
                      f"{cal['status']}; rms z {cal.get('rms_z', float('nan')):.2f}", extra=dict(per_kind=per_kind))


def reference_gates(ref: dict, test: dict, match: MatchResult, basis: str, tier: str, capture: str) -> list[GateResult]:
    """All gates that compare a tested plan with a reference (GT or LiDAR)."""
    ref, test = as_plan_dict(ref), as_plan_dict(test)
    out = [opening_width_gate(ref, test, match, basis, tier, capture),
           ceiling_abs_gate(ref, test, match, basis, tier, capture),
           calibration_gate(ref, test, match, basis, tier, capture)]
    if tier in TIER_WALL_GATE:
        out.append(tier_wall_gate(ref, test, match, basis, tier, capture))
    if tier == "photo":
        out += photo_stitch_gate(ref, test, match, basis, capture)
    return out


# ----------------------------------------------------------------------------------------------- gates, no reference

def not_verifiable_without_gt(tier: str, capture: str) -> list[GateResult]:
    """Rows that need ground truth and have none (the LiDAR tier is the reference, so nothing checks it)."""
    why = "no tape/laser ground truth for the sample data; LiDAR is the reference tier, so nothing checks it"
    out = [GateResult(g, tier, capture, "not_verifiable", "none", detail=why)
           for g in ("opening_width", "ceiling_abs", "calibration")]
    out.append(GateResult("round1_lidar", tier, capture, "not_verifiable", "none",
                          detail="Round 1 thresholds not given in the Round 2 brief, and no ground truth"))
    return out


def repeatability_gate(summary: dict, producer: str = "") -> GateResult:
    """From RepeatabilityReport.summary()."""
    cap = f"{summary['capture_a']} vs {summary['capture_b']}"
    return GateResult("repeatability", summary["tier"], cap, summary["wall_gate"], "repeat_pair",
                      summary["wall_pass_rate"], "every wall within max(1 cm, 0.5%)",
                      summary["walls_pass"] + summary["walls_fail"] + summary["walls_unmatched"],
                      f"{summary['walls_pass']} pass, {summary['walls_fail']} fail, {summary['walls_unmatched']} "
                      f"unmatched; median |d| {_cm(summary['wall_delta_median_m'])}, p90 "
                      f"{_cm(summary['wall_delta_p90_m'])}", producer)


def ceiling_spread_gate(groups: list[dict], tier: str, producer: str = "") -> GateResult:
    """From repeatability.ceiling_groups(): the worst group decides; the status names which failure it is."""
    judged = [g for g in groups if g["status"] != "not_verifiable"]
    if not judged:
        return GateResult("ceiling_spread", tier, "all captures", "not_verifiable", "repeat_pair", None, "<= 1 cm",
                          len(groups), "no room has an observed ceiling in two or more captures", producer)
    worst = max(judged, key=lambda g: g["spread_m"])
    bad = [g for g in judged if g["status"] == "unrepeatable"]
    status = "fail" if bad else "pass"
    detail = ("unrepeatable" if bad else "repeatable; bias not verifiable without GT") + \
             f"; worst group {worst['members']} spread {_cm(worst['spread_m'])}"
    return GateResult("ceiling_spread", tier, "all captures", status, "repeat_pair", worst["spread_m"], "<= 1 cm",
                      len(judged), detail, producer)


def drift_gate(meta: dict, tier: str, capture: str, producer: str = "",
               cycle_evidence: list[dict] | None = None) -> GateResult:
    """Pass needs (1) a declared drift method that is not 'poses as-is' and (2) an on/off ablation of the stitched
    footprint. Expected plan meta (requested convention): meta["drift"] = {"method": str,
    "ablation": {"footprint_on_m2": float, "footprint_off_m2": float, "figure": path}}.
    The capture-to-capture loop residual from registration is attached as independent evidence."""
    d = (meta or {}).get("drift")
    ev = {"cycle_residuals": cycle_evidence or []}
    if d is None:
        return GateResult("drift", tier, capture, "not_verifiable", "declaration", detail="plan meta has no 'drift'",
                          producer=producer, extra=ev)
    method = str(d.get("method", "") if isinstance(d, dict) else d).strip().lower()
    if method in NO_DRIFT_METHODS:
        return GateResult("drift", tier, capture, "fail", "declaration", detail=f"method '{method}' = poses as-is",
                          producer=producer, extra=ev)
    ab = d.get("ablation") if isinstance(d, dict) else None
    if not ab or ab.get("footprint_on_m2") is None or ab.get("footprint_off_m2") is None:
        return GateResult("drift", tier, capture, "fail", "declaration",
                          detail=f"method '{method}' declared but no on/off ablation", producer=producer, extra=ev)
    on, off = ab["footprint_on_m2"], ab["footprint_off_m2"]
    return GateResult("drift", tier, capture, "pass", "declaration", (on - off) / off if off else None,
                      "method declared + on/off ablation", 1,
                      f"{method}: footprint on {on:.2f} m2 vs off {off:.2f} m2", producer, extra=ev)


def _cm(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.2f} cm"
