"""Tier error budget: widen every interval by the tier's scale uncertainty (D-016).

Why: the plan extractor's intervals describe how well walls are fitted *within* the reconstruction (fit error,
sensor noise, residual drift). LiDAR depth is metric by construction. Video- and photo-tier reconstructions get
their metric scale from learned depth and priors (the paper-sheet cue is optional and off by default, D-067), so the
WHOLE reconstruction can be off by a relative scale error s (sigma_s of a few percent). A scale error multiplies every
length, so it is added in quadrature as a relative term:

    length:  sigma^2 = sigma_fit^2 + (sigma_s * L)^2
    area:    sigma^2 = sigma_fit^2 + (2 * sigma_s * A)^2      (area scales with s^2, so its relative error doubles)
    height:  sigma^2 = sigma_fit^2 + (sigma_s * H)^2

This is the single place where "intervals widen honestly as sensor data thins" (case study, Part 1) happens.
"""
from __future__ import annotations

import math

from floorplan.model import Measurement, Plan

Z95 = 1.96


def _widen(m: Measurement | None, rel_sigma: float, power: int = 1) -> None:
    if m is None or m.value is None or m.lo is None or m.hi is None:
        return
    s_fit = (m.hi - m.lo) / (2 * Z95)
    s_tot = math.sqrt(s_fit ** 2 + (power * rel_sigma * abs(m.value)) ** 2)
    m.lo, m.hi = m.value - Z95 * s_tot, m.value + Z95 * s_tot
    m.method = f"{m.method}; + tier scale term {100 * rel_sigma:.2f}% (x{power})".strip("; ")


def widen_plan(plan: Plan, rel_sigma: float, reason: str) -> Plan:
    if rel_sigma <= 0:
        return plan
    for w in plan.walls:
        _widen(w.length, rel_sigma)
        _widen(w.thickness, rel_sigma)
    for r in plan.rooms:
        _widen(r.floor_area, rel_sigma, power=2)
        _widen(r.perimeter, rel_sigma)
        _widen(r.ceiling_height, rel_sigma)
        if r.bbox_dims:
            for d in r.bbox_dims:
                _widen(d, rel_sigma)
    for o in plan.openings:
        _widen(o.width, rel_sigma)
        _widen(o.height, rel_sigma)
        _widen(o.sill_height, rel_sigma)
    _widen(plan.footprint_area, rel_sigma, power=2)
    plan.meta.setdefault("uncertainty", {})["tier_scale_sigma_rel"] = rel_sigma
    plan.meta["uncertainty"]["tier_scale_reason"] = reason
    return plan
