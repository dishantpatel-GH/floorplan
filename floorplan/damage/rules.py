"""Concealed-damage rules: visible damage + where it is -> what may be hidden behind it.

Rules live in rules.json (data, not code) so a restoration expert can review and edit them without reading Python.
Each flag records the rule id, the measured evidence, and p_condition: the probability that the geometric condition
really holds given the measurement interval (a stain whose lower edge is 0.29 +- 0.02 m above the floor only
'probably' reaches the 0.30 m wicking zone). Flag confidence = detection confidence x p_condition.
"""
from __future__ import annotations

import json
from math import erf, sqrt
from pathlib import Path

import numpy as np

RULES_FILE = Path(__file__).with_name("rules.json")
WET_DEFAULT_SIGMA = 0.01


def load_rules(path: Path | None = None) -> list[dict]:
    return json.loads(Path(path or RULES_FILE).read_text())["rules"]


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _sigma(m: dict) -> float:
    return max((m["hi"] - m["value"]) / 1.96, 1e-4)


def _p_below(value: float, thr: float, sigma: float) -> float:
    return _phi((thr - value) / sigma)


def _check(rule: dict, d: dict, srf_meta: dict) -> tuple[float, dict] | None:
    """Return (p_condition, evidence) if the rule applies to region d, else None."""
    a = rule["applies_to"]
    if a.get("cls") and a["cls"] != d["cls"]:
        return None
    if a.get("surface_kind") and a["surface_kind"] != d["surface_kind"]:
        return None
    c = rule.get("condition", {})
    sig_pos = _sigma(d["center_t"])
    if not c:
        return 1.0, {}
    if "bottom_t_max_m" in c:
        p = _p_below(d["bottom_t"], c["bottom_t_max_m"], sig_pos)
        return (p, dict(bottom_above_floor_m=d["bottom_t"], threshold_m=c["bottom_t_max_m"])) if p > 0.05 else None
    if "below_opening_kind" in c:
        s0, _, s1, _ = d["bbox_st"]
        for o in srf_meta.get("openings", []):
            if o["kind"] != c["below_opening_kind"] or o.get("sill_t") is None:
                continue
            os0, os1 = o["s_center"] - o["half_width"], o["s_center"] + o["half_width"]
            overlap = min(s1, os1) - max(s0, os0)
            gap = o["sill_t"] - d["top_t"]
            if overlap > 0 and -0.05 <= gap <= c["max_gap_below_sill_m"]:
                return 1.0, dict(opening_id=o["id"], gap_below_sill_m=round(gap, 3), horizontal_overlap_m=round(overlap, 3))
        return None
    if "max_dist_to_opening_corner_m" in c:
        lo, hi = c["diagonal_deg"]
        ang = d["orientation_deg"] % 180
        ang = min(ang, 180 - ang)
        if not lo <= ang <= hi:
            return None
        best = None
        for o in srf_meta.get("openings", []):
            for cs in (o["s_center"] - o["half_width"], o["s_center"] + o["half_width"]):
                for ct in o["corners_t"]:
                    for e in d["endpoints_st"]:
                        dist = float(np.hypot(e[0] - cs, e[1] - ct))
                        if best is None or dist < best[0]:
                            best = (dist, o["id"])
        if best is None:
            return None
        p = _p_below(best[0], c["max_dist_to_opening_corner_m"], sig_pos * 1.4)
        return (p, dict(opening_id=best[1], dist_to_corner_m=round(best[0], 3), angle_from_axis_deg=round(ang, 1))) \
            if p > 0.05 else None
    if "min_area_m2" in c:
        p = 1.0 - _p_below(d["area"]["value"], c["min_area_m2"], _sigma(d["area"]))
        return (p, dict(area_m2=d["area"]["value"])) if p > 0.05 else None
    if "room_label_any" in c:
        lab = (srf_meta.get("room_label") or "").lower()
        hit = [k for k in c["room_label_any"] if k in lab]
        return (1.0, dict(room_label=lab)) if hit else None
    return None


def apply_rules(regions: list[dict], surfaces_meta: dict[str, dict], rules: list[dict] | None = None) -> list[dict]:
    """Evaluate every rule on every accepted region. Returns flags (one per rule firing on a region)."""
    rules = rules or load_rules()
    flags = []
    for i, d in enumerate(regions):
        for r in rules:
            res = _check(r, d, surfaces_meta.get(d["surface_id"], {}))
            if res is None:
                continue
            p, ev = res
            flags.append(dict(rule_id=r["id"], region_index=i, region_id=d.get("id"), surface_id=d["surface_id"],
                              flag=r["flag"], severity=r["severity"], actions=r["actions"],
                              p_condition=round(p, 3), confidence=round(p * d["confidence"], 3), evidence=ev,
                              rationale=r["rationale"], reference=r["reference"]))
    return flags
