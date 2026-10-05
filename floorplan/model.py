"""Output data model. Every number carries a 95% interval (Part 2: "a confidence interval on every measurement").

Plan coordinates: metres, 2D (u, v) = (x, z) of the gravity- and Manhattan-aligned world (see plan/align.py).
Heights are metres above the room's own floor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Measurement:
    value: Optional[float]                 # None = not observed (we never invent a number)
    lo: Optional[float] = None             # 95% interval lower bound
    hi: Optional[float] = None             # 95% interval upper bound
    unit: str = "m"
    method: str = ""                       # how it was measured (auditable)
    status: str = "measured"               # measured | inferred | not_observed

    @staticmethod
    def from_sigma(value: float, sigma: float, unit: str = "m", method: str = "", status: str = "measured"):
        return Measurement(value, value - 1.96 * sigma, value + 1.96 * sigma, unit, method, status)

    @staticmethod
    def missing(unit: str = "m", method: str = "", reason: str = "not observed"):
        return Measurement(None, None, None, unit, f"{method}; {reason}".strip("; "), "not_observed")

    @property
    def half_width(self) -> Optional[float]:
        return None if self.value is None else (self.hi - self.lo) / 2


@dataclass
class Opening:
    id: str
    kind: str                              # door | window | passage (doorless opening)
    wall_ids: list[str]
    room_ids: list[str]                    # rooms it connects (doors/passages) or belongs to (windows)
    center: tuple[float, float]            # plan (u, v)
    width: Measurement
    height: Optional[Measurement] = None
    sill_height: Optional[Measurement] = None
    confidence: float = 1.0                # detection confidence 0..1
    evidence: str = ""
    source: str = "geometry"               # geometry (gaps in the points) | segmentation (door/window pixels) | prior


@dataclass
class Wall:
    id: str
    room_id: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    length: Measurement
    normal: tuple[float, float]            # unit normal pointing into the room
    thickness: Optional[Measurement] = None
    opening_ids: list[str] = field(default_factory=list)
    support_points: int = 0                # how many LiDAR points support this wall plane
    fit_rmse_m: Optional[float] = None


@dataclass
class Room:
    id: str
    label: str
    polygon: list[tuple[float, float]]     # interior polygon, counter-clockwise, plan metres
    wall_ids: list[str]
    floor_area: Measurement
    perimeter: Measurement
    ceiling_height: Measurement
    bbox_dims: tuple[Measurement, Measurement] | None = None   # length x width of the aligned bounding box
    floor_level: float = 0.0               # floor height in the aligned world (m)
    name: Optional[str] = None             # floor-plan name (Bedroom, Kitchen, ...); None = not named
    room_type: Optional[str] = None        # plan/room_types.py TYPES
    type_evidence: Optional[dict] = None   # classes seen in the room, scores, the rule that decided


@dataclass
class Adjacency:
    room_a: str
    room_b: str
    via: str                               # opening id, or "shared_wall"


@dataclass
class Plan:
    capture_id: str
    tier: str                              # lidar | video | photo
    rooms: list[Room]
    walls: list[Wall]
    openings: list[Opening]
    adjacency: list[Adjacency]
    footprint_area: Measurement
    damage: list[dict] = field(default_factory=list)
    scope_items: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
