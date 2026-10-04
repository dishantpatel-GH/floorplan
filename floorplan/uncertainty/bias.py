"""LiDAR-tier systematic bias: every LiDAR surface stands ~7 mm INTO the room (docs/modules/lidar_bias.md).

What was measured (scripts/investigate_lidar_bias.py, outputs/lidar_bias/): on ARKitScenes rooms scanned by both an
iPad Pro (LiDAR + ARKit poses) and a Faro laser, our raw-point plane fits put walls, floor and ceiling a few mm in
front of the laser surface, so every interior distance (wall to wall, floor to ceiling) reads short by about twice
that. It is NOT a depth-scale or VIO-scale error (flat with range, error-vs-distance slope inconsistent between rooms),
NOT the registration (distances between facing planes do not move when the laser is shifted 1 cm or replaced by an
independent transform), and NOT the choice of plane estimator (every estimator, including a KDE mode and a
1.0-1.5 m band, shows it): the whole point distribution of a surface is shifted, not just a one-sided tail.

Why a constant per-surface offset (and not a scale): the error of a facing pair is "-2b" whatever its length, the
leave-one-room-out values of b agree (0.57-0.76 cm), and it was fixed on three development rooms and then tested on
two hold-out rooms that were downloaded and analysed only after the value was written down
(outputs/lidar_bias/preregistration.json).

How it is applied: a surface whose unit normal n points INTO the room satisfies n . p = d; the corrected surface is
n . p = d - b (moved away from the room). Consequences for plan quantities measured between such surfaces:

    interior distance (room width, wall length between corners, floor-to-ceiling)   + 2b
    wall thickness (between the faces of one wall seen from two rooms)              - 2b
    rectilinear room polygon: perimeter + 8b, area + b * P + 4 b^2  (exact offset of an orthogonal polygon)

Opening widths are not corrected (jamb geometry was not validated); they get the systematic sigma only.

The residual systematic uncertainty (room-to-room spread of the bias, plus an allowance when the device class was
not the validated one) is added in quadrature to every affected interval, because a bias does not average down with
more points: the plane fit's own standard error (< 0.1 mm) is ~100x too small.

Switch: LidarBiasConfig(enabled=False) or environment variable FLOORPLAN_LIDAR_BIAS=off reports raw values and
instead widens the intervals one-sidedly (toward longer) by the uncorrected bias.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from floorplan.model import Measurement, Plan

Z95 = 1.96


@dataclass
class LidarBiasConfig:
    enabled: bool = True
    # per-surface inward bias b (m). Default = the value pre-registered from the 3 development rooms
    # (outputs/lidar_bias/preregistration.json), then tested on 2 hold-out rooms (docs/modules/lidar_bias.md section 5).
    inward_bias_m: float = 0.0068
    # 1-sigma residual error of one interior distance AFTER correction: spread of the room-box errors over all 5 rooms
    # (std of 14 room-box errors = 0.81 cm; development rooms alone 0.71 cm, hold-out after correction 1.05 cm).
    sigma_distance_m: float = 0.0081
    # the bias was measured on iPad Pro LiDAR (ARKitScenes). For another device class (e.g. iPhone via Stray Scanner)
    # it is plausible but unverified: add an extra 1-sigma term per distance equal to the full correction 2b/2 = b.
    device_validated: bool = False

    @staticmethod
    def load(path: str | Path | None = None, **overrides) -> "LidarBiasConfig":
        cfg = LidarBiasConfig()
        if path:
            for k, v in json.loads(Path(path).read_text()).items():
                setattr(cfg, k, v)
        env = os.environ.get("FLOORPLAN_LIDAR_BIAS", "").strip().lower()
        if env in ("0", "off", "false", "no"):
            cfg.enabled = False
        names = {f.name for f in fields(LidarBiasConfig)}
        for k, v in overrides.items():
            if k not in names:
                raise KeyError(f"unknown lidar-bias key {k}")
            setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def sigma_per_distance(self) -> float:
        """1-sigma systematic error of one interior distance (two surfaces), after correction if enabled."""
        s = self.sigma_distance_m
        if not self.device_validated:
            s = math.hypot(s, self.inward_bias_m)
        return s


# ------------------------------------------------------------------------------------------------ primitives
def correct_plane_offset(d: float, cfg: LidarBiasConfig) -> float:
    """Plane n . p = d with n pointing INTO the room -> the corrected plane (moved away from the room by b)."""
    return d - cfg.inward_bias_m if cfg.enabled else d


def correct_interior_distance(dist_m: float, cfg: LidarBiasConfig, faces: int = 2) -> float:
    """Distance between `faces` LiDAR surfaces that face each other across a space (2 for a room width)."""
    return dist_m + faces * cfg.inward_bias_m if cfg.enabled else dist_m


def _shift_and_widen(m: Measurement | None, delta: float, sigma: float, cfg: LidarBiasConfig, what: str) -> None:
    """Add `delta` to the value (if correcting) and the systematic `sigma` to the 95% interval in quadrature.
    When the correction is OFF, the interval is instead stretched toward the true side by |delta| (one-sided), so
    it still covers the truth without moving the reported value."""
    if m is None or m.value is None:
        return
    lo = m.lo if m.lo is not None else m.value
    hi = m.hi if m.hi is not None else m.value
    s_lo, s_hi = (m.value - lo) / Z95, (hi - m.value) / Z95
    if cfg.enabled:
        m.value += delta
        m.lo = m.value - Z95 * math.hypot(s_lo, sigma)
        m.hi = m.value + Z95 * math.hypot(s_hi, sigma)
        m.method = f"{m.method}; LiDAR inward-bias correction {100 * delta:+.2f} cm ({what})".strip("; ")
    else:
        m.lo = m.value - Z95 * math.hypot(s_lo, sigma) + min(delta, 0.0)
        m.hi = m.value + Z95 * math.hypot(s_hi, sigma) + max(delta, 0.0)
        m.method = f"{m.method}; uncorrected LiDAR bias {100 * delta:+.2f} cm folded into interval ({what})".strip("; ")


def correct_measurement(m: Measurement | None, cfg: LidarBiasConfig, faces: int = 2, sign: int = +1,
                        what: str = "interior distance") -> Measurement | None:
    """Generic: a length bounded by `faces` LiDAR surfaces. sign=+1 for interior distances (they read short),
    sign=-1 for wall thickness (read long)."""
    sigma = cfg.sigma_per_distance() * faces / 2.0
    _shift_and_widen(m, sign * faces * cfg.inward_bias_m, sigma, cfg, what)
    return m


# ------------------------------------------------------------------------------------------------ whole plan
def _corner_signs(plan: Plan) -> dict[str, tuple[int, int]]:
    """Per wall: +1 if the corner at that end is convex (interior angle < 180 deg), -1 if reflex.

    Why (D-027, found by the ARKitScenes plan-level benchmark): moving every face outward by b lengthens a wall by b
    at a convex corner but SHORTENS it by b at a reflex corner, so the length correction is b * (s_start + s_end),
    not a blanket 2b. Measured offline: walls within 1.5 cm of the laser 82% -> 94%."""
    import numpy as np
    out: dict[str, tuple[int, int]] = {}
    rooms = {r.id: r for r in plan.rooms}
    for w in plan.walls:
        r = rooms.get(w.room_id)
        if r is None or len(r.polygon) < 3:
            out[w.id] = (1, 1)
            continue
        P = np.asarray(r.polygon, float)
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        area2 = float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
        orient = 1.0 if area2 > 0 else -1.0

        def sign_at(pt) -> int:
            i = int(np.argmin(np.linalg.norm(P - np.asarray(pt, float), axis=1)))
            a, v, c = P[i - 1], P[i], P[(i + 1) % len(P)]
            cross = (v[0] - a[0]) * (c[1] - v[1]) - (v[1] - a[1]) * (c[0] - v[0])
            return 1 if cross * orient > 0 else -1
        out[w.id] = (sign_at(w.p0), sign_at(w.p1))
    return out


def apply_to_plan(plan: Plan, cfg: LidarBiasConfig | None = None) -> Plan:
    """Correct (or, if disabled, only widen) every LiDAR-tier plan quantity bounded by fitted surfaces.
    Call once, on LiDAR-tier plans only, before tier_budget.widen_plan."""
    cfg = cfg or LidarBiasConfig.load()
    b = cfg.inward_bias_m
    sig = cfg.sigma_per_distance()
    signs = _corner_signs(plan)
    for w in plan.walls:
        s0, s1 = signs.get(w.id, (1, 1))
        _shift_and_widen(w.length, b * (s0 + s1), sig, cfg,
                         f"wall length between corners; corner signs {s0:+d}/{s1:+d} (convex +1, reflex -1)")
        if w.thickness is not None:
            correct_measurement(w.thickness, cfg, 2, -1, "wall thickness")
    for r in plan.rooms:
        perim = r.perimeter.value if r.perimeter is not None else None
        if r.floor_area is not None and r.floor_area.value is not None and perim is not None:
            # exact offset of an orthogonal polygon by b; sigma propagated with the perimeter
            _shift_and_widen(r.floor_area, b * perim + 4 * b * b, perim * sig / 2.0, cfg, "area of room offset by b")
        _shift_and_widen(r.perimeter, 8 * b, 4 * sig, cfg, "perimeter of room offset by b")
        correct_measurement(r.ceiling_height, cfg, 2, +1, "floor to ceiling")
        for d in (r.bbox_dims or []):
            correct_measurement(d, cfg, 2, +1, "room box")
    for o in plan.openings:
        _shift_and_widen(o.width, 0.0, sig, cfg, "opening width: systematic sigma only")
    plan.meta.setdefault("uncertainty", {})["lidar_bias"] = dict(cfg.to_dict(), sigma_per_distance_m=sig)
    return plan
