"""Plan (floorplan.model) <-> JSON that conforms to schema/plan.schema.json, plus validation.

Why a separate, explicit JSON layer instead of `json.dumps(plan.to_dict())`:
  * The output contract (Part 2) is "JSON to the published schema". A schema is a promise to downstream consumers
    (estimating software, the scope engine, a web viewer). The dataclasses are an internal detail that will change;
    the JSON keys must not change silently. An explicit mapping makes every key a deliberate choice.
  * Every measurement must carry a 95% interval. `asdict` would emit `lo`/`hi` fields next to each other with no
    rule tying them to `value` or `status`. Here each measurement becomes one uniform object
    `{value, ci95: [lo, hi], unit, method, status}` and the schema enforces that a not-observed value has no number
    and no interval (we never publish an invented number).
  * Validation has two layers: JSON Schema (structure, types, enums) and semantic checks that JSON Schema cannot
    express (lo <= value <= hi, every id referenced actually exists, polygons have >= 3 vertices). A plan that
    fails either layer is a bug in the pipeline, and we want to find it before a homeowner or estimator does.
  * Floats are rounded to 0.1 mm (4 decimals). That is 100x finer than the tightest gate (1 cm), keeps files
    readable and makes repeated runs byte-identical, which matters for regenerable before/after runs (Part 4).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

from floorplan.model import Adjacency, Measurement, Opening, Plan, Room, Wall

SCHEMA_ID = "floorplan.plan"
SCHEMA_VERSION = "1.1.0"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "plan.schema.json"
DECIMALS = 4

COORDINATE_FRAME = {
    "name": "aligned_plan",
    "description": (
        "Metric 2-D plan coordinates (u, v) = (x, z) of the capture's gravity- and Manhattan-aligned world frame. "
        "The 3-D frame is right-handed with +y up, so when the plan is viewed from above, +u points right and +v "
        "points DOWN the page (drawing +v upward mirrors the plan). The origin and heading are arbitrary per "
        "capture (wherever the phone started, rotated so the dominant walls run along u and v)."
    ),
    "plan_axes": ["u = x (m)", "v = z (m)"],
    "up_axis": "+y",
    "handedness": "right",
    "view_from_above": {"u": "right", "v": "down"},
    "heights": "metres above the room's own floor; floor_level is the floor height in the aligned frame",
}
UNITS = {"length": "m", "area": "m2", "angle": "deg", "height": "m"}
_NUM = r"-?[0-9.]+(?:[eE][-+]?[0-9]+)?|null"
_PAIR = re.compile(rf"\[\s+({_NUM}),\s+({_NUM})\s+\]")


# ----------------------------------------------------------------------------------------------- Plan -> JSON
def _num(x: Optional[float]) -> Optional[float]:
    """Round a float to 0.1 mm; map NaN/inf to None so the JSON stays valid (json.dumps would write NaN)."""
    if x is None:
        return None
    x = float(x)
    return round(x, DECIMALS) if math.isfinite(x) else None


def _pt(p) -> list[float]:
    return [_num(p[0]), _num(p[1])]


def measurement_to_json(m: Optional[Measurement]) -> Optional[dict]:
    """One uniform measurement object. A measurement whose value is missing is published as not_observed with
    no interval, whatever status the producer set: a number-less 'measured' value would be a contradiction."""
    if m is None:
        return None
    value = _num(m.value)
    if value is None:
        return {"value": None, "ci95": None, "unit": m.unit, "method": m.method, "status": "not_observed"}
    lo, hi = _num(m.lo), _num(m.hi)
    ci = [lo, hi] if lo is not None and hi is not None else None
    return {"value": value, "ci95": ci, "unit": m.unit, "method": m.method, "status": m.status}


def _wall_to_json(w: Wall) -> dict:
    return {
        "id": w.id, "room_id": w.room_id, "p0": _pt(w.p0), "p1": _pt(w.p1),
        "length": measurement_to_json(w.length), "normal": _pt(w.normal),
        "thickness": measurement_to_json(w.thickness), "opening_ids": list(w.opening_ids),
        "support_points": int(w.support_points), "fit_rmse_m": _num(w.fit_rmse_m),
    }


def _room_to_json(r: Room) -> dict:
    bbox = None
    if r.bbox_dims is not None:
        bbox = {"length": measurement_to_json(r.bbox_dims[0]), "width": measurement_to_json(r.bbox_dims[1])}
    out = {
        "id": r.id, "label": r.label, "polygon": [_pt(p) for p in r.polygon], "wall_ids": list(r.wall_ids),
        "floor_area": measurement_to_json(r.floor_area), "perimeter": measurement_to_json(r.perimeter),
        "ceiling_height": measurement_to_json(r.ceiling_height), "bbox_dims": bbox,
        "floor_level": _num(r.floor_level),
    }
    if r.name is not None:                     # schema 1.1: room names (plan/room_types.py), optional
        out.update(name=r.name, type=r.room_type or "room", type_evidence=_free_to_json(r.type_evidence or {}))
    return out


def _opening_to_json(o: Opening) -> dict:
    return {
        "id": o.id, "kind": o.kind, "wall_ids": list(o.wall_ids), "room_ids": list(o.room_ids),
        "center": _pt(o.center), "width": measurement_to_json(o.width),
        "height": measurement_to_json(o.height), "sill_height": measurement_to_json(o.sill_height),
        "confidence": _num(o.confidence), "evidence": o.evidence, "source": o.source,
    }


def _free_to_json(x: Any) -> Any:
    """Damage regions, scope items and meta are open dicts produced by other modules. Convert any Measurement
    (or a dict that looks like an un-converted Measurement from `asdict`) into the uniform measurement object,
    and make everything else JSON-safe."""
    if isinstance(x, Measurement):
        return measurement_to_json(x)
    if isinstance(x, dict):
        if {"value", "lo", "hi", "status"} <= set(x):
            return measurement_to_json(Measurement(x["value"], x["lo"], x["hi"], x.get("unit", "m"),
                                                   x.get("method", ""), x["status"]))
        return {str(k): _free_to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_free_to_json(v) for v in x]
    if isinstance(x, float):
        return _num(x)
    if hasattr(x, "item"):                        # numpy scalars
        return _free_to_json(x.item())
    return x


def plan_to_json(plan: Plan) -> dict:
    """Plan -> JSON-ready dict in schema order (header first, so a human opening the file sees what it is)."""
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "units": dict(UNITS),
        "coordinate_frame": dict(COORDINATE_FRAME),
        "capture_id": plan.capture_id,
        "tier": plan.tier,
        "footprint_area": measurement_to_json(plan.footprint_area),
        "rooms": [_room_to_json(r) for r in plan.rooms],
        "walls": [_wall_to_json(w) for w in plan.walls],
        "openings": [_opening_to_json(o) for o in plan.openings],
        "adjacency": [{"room_a": a.room_a, "room_b": a.room_b, "via": a.via} for a in plan.adjacency],
        "damage": _free_to_json(list(plan.damage)),
        "scope_items": _free_to_json(list(plan.scope_items)),
        "meta": _free_to_json(dict(plan.meta)),
    }


def dumps(doc: dict) -> str:
    """Indented JSON, but with number pairs (points, intervals) kept on one line so a polygon reads as a list of
    points rather than a column of numbers."""
    return _PAIR.sub(r"[\1, \2]", json.dumps(doc, indent=2, ensure_ascii=False)) + "\n"


def save_plan_json(plan: Plan, path: str | Path, validate: bool = True) -> list[str]:
    """Write the plan; return validation errors (empty list = valid). Writes even when invalid so the broken
    output can be inspected, but the caller is told."""
    doc = plan_to_json(plan)
    errors = validate_plan_json(doc) if validate else []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(doc))
    return errors


# ----------------------------------------------------------------------------------------------- JSON -> Plan
def measurement_from_json(d: Optional[dict]) -> Optional[Measurement]:
    if d is None:
        return None
    lo, hi = (d["ci95"] if d.get("ci95") else (None, None))
    return Measurement(d["value"], lo, hi, d.get("unit", "m"), d.get("method", ""), d.get("status", "measured"))


def plan_from_json(doc: dict) -> Plan:
    """Inverse of plan_to_json (up to the 0.1 mm rounding). Used by the renderer CLI and by round-trip tests."""
    m = measurement_from_json
    rooms = [Room(
        id=r["id"], label=r["label"], polygon=[tuple(p) for p in r["polygon"]], wall_ids=list(r["wall_ids"]),
        floor_area=m(r["floor_area"]), perimeter=m(r["perimeter"]), ceiling_height=m(r["ceiling_height"]),
        bbox_dims=(m(r["bbox_dims"]["length"]), m(r["bbox_dims"]["width"])) if r.get("bbox_dims") else None,
        floor_level=r.get("floor_level") or 0.0, name=r.get("name"), room_type=r.get("type"),
        type_evidence=r.get("type_evidence")) for r in doc["rooms"]]
    walls = [Wall(
        id=w["id"], room_id=w["room_id"], p0=tuple(w["p0"]), p1=tuple(w["p1"]), length=m(w["length"]),
        normal=tuple(w["normal"]), thickness=m(w.get("thickness")), opening_ids=list(w.get("opening_ids", [])),
        support_points=w.get("support_points", 0), fit_rmse_m=w.get("fit_rmse_m")) for w in doc["walls"]]
    openings = [Opening(
        id=o["id"], kind=o["kind"], wall_ids=list(o["wall_ids"]), room_ids=list(o["room_ids"]),
        center=tuple(o["center"]), width=m(o["width"]), height=m(o.get("height")),
        sill_height=m(o.get("sill_height")), confidence=o.get("confidence", 1.0),
        evidence=o.get("evidence", ""), source=o.get("source", "geometry")) for o in doc["openings"]]
    adjacency = [Adjacency(a["room_a"], a["room_b"], a["via"]) for a in doc["adjacency"]]
    return Plan(capture_id=doc["capture_id"], tier=doc["tier"], rooms=rooms, walls=walls, openings=openings,
                adjacency=adjacency, footprint_area=m(doc["footprint_area"]), damage=list(doc.get("damage", [])),
                scope_items=list(doc.get("scope_items", [])), meta=dict(doc.get("meta", {})))


def load_plan_json(path: str | Path) -> Plan:
    return plan_from_json(json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------------------------------- validation
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def schema_errors(doc: dict) -> list[str]:
    """Structural validation against schema/plan.schema.json (JSON Schema draft 2020-12)."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(load_schema())
    errs = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    return [f"schema: /{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in errs]


def _iter_measurements(x: Any, path: str):
    """Yield (path, measurement-dict) for every measurement object anywhere in the document."""
    if isinstance(x, dict):
        if {"value", "ci95", "status"} <= set(x):
            yield path, x
            return
        for k, v in x.items():
            yield from _iter_measurements(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _iter_measurements(v, f"{path}/{i}")


def _interval_errors(doc: dict) -> list[str]:
    out = []
    for path, m in _iter_measurements(doc, ""):
        v, ci = m["value"], m["ci95"]          # "a number always has an interval" is enforced by the schema
        if v is not None and ci is not None and not (ci[0] <= v <= ci[1]):
            out.append(f"interval: {path}: value {v} outside its interval {ci}")
    return out


def _reference_errors(doc: dict) -> list[str]:
    rooms = {r["id"] for r in doc["rooms"]}
    walls = {w["id"] for w in doc["walls"]}
    openings = {o["id"] for o in doc["openings"]}
    out = []

    def need(ids, known, where):
        out.extend(f"reference: {where} -> unknown id '{i}'" for i in ids if i not in known)

    for r in doc["rooms"]:
        need(r["wall_ids"], walls, f"room {r['id']}.wall_ids")
    for w in doc["walls"]:
        need([w["room_id"]], rooms, f"wall {w['id']}.room_id")
        need(w["opening_ids"], openings, f"wall {w['id']}.opening_ids")
    for o in doc["openings"]:
        need(o["wall_ids"], walls, f"opening {o['id']}.wall_ids")
        need(o["room_ids"], rooms, f"opening {o['id']}.room_ids")
    for a in doc["adjacency"]:
        need([a["room_a"], a["room_b"]], rooms, "adjacency")
        if a["via"] != "shared_wall":
            need([a["via"]], openings, "adjacency.via")
    for kind in ("rooms", "walls", "openings"):
        ids = [x["id"] for x in doc[kind]]
        out.extend(f"duplicate: {kind} id '{i}'" for i in sorted({i for i in ids if ids.count(i) > 1}))
    return out


def semantic_errors(doc: dict) -> list[str]:
    """Rules JSON Schema cannot express: an interval contains its value, and every id reference resolves."""
    return _interval_errors(doc) + _reference_errors(doc)


def validate_plan_json(doc: dict) -> list[str]:
    """All validation errors (empty = valid). Semantic checks run only on structurally valid documents, because
    they assume the structure."""
    errs = schema_errors(doc)
    return errs if errs else semantic_errors(doc)
