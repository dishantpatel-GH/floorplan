"""Synthetic plans with exactly known differences, for testing match / repeatability / gates without real plans.

A plan is built from axis-aligned rectangular rooms (interior faces), so every wall length, area and opening width
is known exactly. Perturbing a room's size by d changes exactly two wall lengths by d, which the metrics must
recover to floating-point precision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from floorplan.eval.register import make_T2
from floorplan.eval.match import transform_plan
from floorplan.model import Adjacency, Measurement, Opening, Plan, Room, Wall

SIGMA_LEN = 0.004
SIGMA_CEIL = 0.003


@dataclass
class RoomSpec:
    id: str
    x0: float
    z0: float
    x1: float
    z1: float
    ceiling: float | None = 2.70


@dataclass
class OpeningSpec:
    id: str
    kind: str
    center: tuple[float, float]
    width: float
    rooms: list[str] = field(default_factory=list)


def _walls(r: RoomSpec, sig: float) -> list[Wall]:
    corners = [(r.x0, r.z0), (r.x1, r.z0), (r.x1, r.z1), (r.x0, r.z1)]       # counter-clockwise
    out = []
    for k in range(4):
        p0, p1 = np.array(corners[k], float), np.array(corners[(k + 1) % 4], float)
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = (-d[1], d[0])                                                    # left of a CCW edge = inside
        L = float(np.linalg.norm(p1 - p0))
        out.append(Wall(f"{r.id}_w{k}", r.id, tuple(map(float, p0)), tuple(map(float, p1)),
                        Measurement.from_sigma(L, sig),
                        tuple(map(float, n))))
    return out


def make_plan(rooms: list[RoomSpec], openings: list[OpeningSpec], adjacency: list[tuple[str, str, str]],
              tier: str = "lidar", capture_id: str = "synthetic", sig_len: float = SIGMA_LEN,
              footprint: float | None = None) -> dict:
    rs, ws = [], []
    for r in rooms:
        w = _walls(r, sig_len)
        ws += w
        area = (r.x1 - r.x0) * (r.z1 - r.z0)
        ceil = (Measurement.from_sigma(r.ceiling, SIGMA_CEIL) if r.ceiling is not None
                else Measurement.missing(reason="ceiling not observed"))
        rs.append(Room(r.id, r.id, [(r.x0, r.z0), (r.x1, r.z0), (r.x1, r.z1), (r.x0, r.z1)], [x.id for x in w],
                       Measurement.from_sigma(area, 0.02), Measurement.from_sigma(2 * (r.x1 - r.x0 + r.z1 - r.z0),
                                                                                   0.01), ceil))
    ops = [Opening(o.id, o.kind, [], o.rooms, o.center, Measurement.from_sigma(o.width, 0.005)) for o in openings]
    fp = footprint if footprint is not None else sum((r.x1 - r.x0) * (r.z1 - r.z0) for r in rooms)
    plan = Plan(capture_id, tier, rs, ws, ops, [Adjacency(a, b, v) for a, b, v in adjacency],
                Measurement.from_sigma(fp, 0.05 * fp / 1.96 if tier == "photo" else 0.05))
    return json.loads(json.dumps(plan.to_dict()))     # exactly what a plan.json round trip gives


def base_specs() -> tuple[list[RoomSpec], list[OpeningSpec], list[tuple[str, str, str]]]:
    """Two rooms side by side and a corridor above them; 10 cm partition walls."""
    rooms = [RoomSpec("R1", 0.0, 0.0, 4.0, 3.0, 2.70), RoomSpec("R2", 4.1, 0.0, 7.1, 3.0, 2.70),
             RoomSpec("C", 0.0, 3.1, 7.1, 4.3, 2.70)]
    openings = [OpeningSpec("d1", "door", (2.0, 3.05), 0.80, ["R1", "C"]),
                OpeningSpec("d2", "door", (5.5, 3.05), 0.90, ["R2", "C"]),
                OpeningSpec("win", "window", (0.0, 1.5), 1.20, ["R1"])]
    adjacency = [("R1", "C", "d1"), ("R2", "C", "d2"), ("R1", "R2", "shared_wall")]
    return rooms, openings, adjacency


def scale_plan(plan: dict, s: float) -> dict:
    """Uniform scale about the origin, the error mode of a video/photo tier with wrong metric scale."""
    out = transform_plan(plan, np.diag([s, s, 1.0]))
    for w in out["walls"]:
        w["normal"] = (np.asarray(w["normal"]) / s).tolist()
        for k in ("value", "lo", "hi"):
            w["length"][k] *= s
    for r in out["rooms"]:
        for k in ("value", "lo", "hi"):
            r["floor_area"][k] *= s * s
    for o in out["openings"]:
        for k in ("value", "lo", "hi"):
            o["width"][k] *= s
    for k in ("value", "lo", "hi"):
        out["footprint_area"][k] *= s * s
    return out


KNOWN_T = make_T2(np.radians(91.3), np.array([5.0, -2.0]))


def moved(plan: dict, T: np.ndarray = KNOWN_T) -> dict:
    """The same plan expressed in another capture's frame (B = T^-1 applied), so T maps B back onto A."""
    return transform_plan(plan, np.linalg.inv(T))

