"""Scope of work: detected regions + fired rules -> line items keyed to surface ids, with quantity intervals.

Quantities come from the plan's measured surfaces (wall length x room height, room floor area for ceilings) and from
the damage extents, so every quantity carries a 95% interval propagated from its inputs:
  product a*b      : relative sigmas add in quadrature
  region + margin  : (w + 2m)(h + 2m) with the width/height intervals
Items that would repeat on the same surface (two stains on one wall -> one repaint) are merged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CATALOG_FILE = Path(__file__).with_name("scope_catalog.json")


def load_catalog(path: Path | None = None) -> list[dict]:
    return json.loads(Path(path or CATALOG_FILE).read_text())["items"]


def _sig(m: dict) -> float:
    return max((m["hi"] - m["lo"]) / (2 * 1.96), 0.0)


def _q(value: float, sigma: float, unit: str, method: str) -> dict:
    return dict(value=round(value, 3), lo=round(max(value - 1.96 * sigma, 0.0), 3), hi=round(value + 1.96 * sigma, 3),
                unit=unit, method=method)


def _product(a: dict, b: dict, unit: str, method: str) -> dict:
    v = a["value"] * b["value"]
    rel = np.hypot(_sig(a) / max(a["value"], 1e-9), _sig(b) / max(b["value"], 1e-9))
    return _q(v, v * rel, unit, method)


def _matches(trigger: dict, d: dict, fired: set[str]) -> bool:
    if "rule" in trigger:
        return trigger["rule"] in fired
    return all(d.get("cls" if k == "cls" else k) == v for k, v in trigger.items())


def _quantity(item: dict, d: dict, surf: dict) -> dict | None:
    per = item["per"]
    if per == "each":
        return _q(1.0, 0.0, "ea", "one per finding")
    if per == "surface_area":
        return surf.get("area")
    if per == "length":
        return d.get("length")
    if per == "wall_length":
        return surf.get("length")
    if per == "wall_length_x_height":
        if not surf.get("length"):
            return None
        h = dict(value=item["height_m"], lo=item["height_m"], hi=item["height_m"])
        return _product(surf["length"], h, "m2", f"wall length x {item['height_m']} m cut height")
    if per == "region_area_margin":
        m = item["margin_m"]
        w = dict(d["width"], value=d["width"]["value"] + 2 * m)
        h = dict(d["height"], value=d["height"]["value"] + 2 * m)
        return _product(w, h, "m2", f"(width + 2 x {m}) x (height + 2 x {m}) of the stain")
    return None


def build_scope(regions: list[dict], flags: list[dict], surfaces: dict[str, dict],
                catalog: list[dict] | None = None) -> list[dict]:
    """surfaces: id -> {'kind', 'area': interval dict, 'length': interval dict (walls)}."""
    catalog = catalog or load_catalog()
    items: dict[tuple, dict] = {}
    for i, d in enumerate(regions):
        fired = {f["rule_id"] for f in flags if f["region_index"] == i}
        surf = surfaces.get(d["surface_id"], {})
        for it in catalog:
            if not _matches(it["trigger"], d, fired):
                continue
            q = _quantity(it, d, surf)
            if q is None:
                continue
            key = (d["surface_id"], it["id"])
            if key in items:
                prev = items[key]
                prev["region_ids"].append(d.get("id"))
                if it["per"] in ("length", "region_area_margin", "each"):          # additive items
                    for k in ("value", "lo", "hi"):
                        prev["quantity"][k] = round(prev["quantity"][k] + q[k], 3)
                continue
            items[key] = dict(surface_id=d["surface_id"], item_id=it["id"],
                              line=f"{d['surface_id']}: {it['text']} {q['value']:.2f} {q['unit']} "
                                   f"[{q['lo']:.2f}-{q['hi']:.2f}]",
                              text=it["text"], quantity=q, why=it["why"], region_ids=[d.get("id")],
                              confidence=d["confidence"])
    # one repaint per wall: the stain-block repaint already covers the repaint after a crack repair
    for (sid, iid) in list(items):
        if iid == "WALL_PAINT_AFTER_CRACK" and (sid, "WALL_STAINBLOCK_PAINT") in items:
            items[(sid, "WALL_STAINBLOCK_PAINT")]["region_ids"] += items.pop((sid, iid))["region_ids"]
    out = list(items.values())
    for x in out:                                   # refresh the line text after merges
        q = x["quantity"]
        x["line"] = f"{x['surface_id']}: {x['text']} {q['value']:.2f} {q['unit']} [{q['lo']:.2f}-{q['hi']:.2f}]"
    return out
