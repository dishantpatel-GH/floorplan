"""Plan -> DXF (AutoCAD R2010) for architects, contractors and estimators who work in CAD.

Why DXF in addition to SVG/PNG: magicplan and poly.cam both export DXF, because the people who act on a floor plan
(contractors, insurers' estimators) redraw or annotate it in CAD. A DXF only helps them if it is CAD-native:
  * real units (metres, $INSUNITS = 6), so a distance measured in the CAD tool equals the real distance;
  * real DIMENSION entities, not lines plus text, so the CAD tool knows they are dimensions (they re-scale and can be
    restyled). Each one shows OUR measured value and interval as override text, because the measured length can
    differ by a millimetre from the drawn endpoint distance and the interval is the point of the product;
  * one layer per element type (WALLS, ROOMS, DIMENSIONS, OPENINGS, TEXT), so a user can hide or restyle them.

The geometry is the SAME as the rendered plan (walls, opening cuts, swing side) because it comes from
render.build_wall_geometry; only the drawing primitives differ. Coordinates use the same top-down view
transform (X = u, Y = -v): DXF is a y-up drawing space, so without the flip the CAD plan would be mirrored.
"""
from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import numpy as np
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from shapely.ops import polylabel

from floorplan.export.render import (MIN_DIM_LENGTH_M, NOMINAL_THICKNESS_M, OpeningPlacement, build_wall_geometry,
                                     exterior_side, fmt_area, fmt_ceiling, fmt_length, iter_polygons, room_polygon,
                                     room_title, to_view, view_angle, wall_thickness)
from floorplan.model import Plan

LAYERS = {  # name: (AutoCAD colour index, description)
    "WALLS": (7, "wall solids; SOLID hatch = measured thickness, ANSI31 hatch = nominal thickness"),
    "ROOMS": (8, "interior wall-face polygon of each room"),
    "DIMENSIONS": (1, "aligned DIMENSION per wall, text = measured length and 95% interval"),
    "OPENINGS": (5, "doors (leaf + swing arc), windows (glazing lines), passages (dashed threshold)"),
    "TEXT": (3, "room labels, opening widths, title"),
}
DIMSTYLE = "PLAN_M"
TEXT_H = 0.12           # metres: legible at 1:50, the usual scale of a residential plan
DIM_OFFSET_M = 0.30     # dimension line distance beyond the wall face (CAD is zoomable, so a fixed metric offset)


def _xy(p) -> tuple[float, float]:
    v = to_view(p)
    return float(v[0]), float(v[1])


def _new_document() -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2010", setup=True)            # setup=True adds standard linetypes (DASHED) and text styles
    doc.units = units.M
    doc.header["$MEASUREMENT"] = 1                  # metric
    doc.header["$LUNITS"] = 2                       # decimal
    doc.header["$LUPREC"] = 3
    for name, (color, desc) in LAYERS.items():
        doc.layers.add(name, color=color).description = desc
    doc.dimstyles.new(DIMSTYLE, dxfattribs={
        "dimtxt": TEXT_H * 0.9, "dimasz": 0.0, "dimtsz": 0.06,      # architectural ticks instead of arrows
        "dimexe": 0.05, "dimexo": 0.03, "dimgap": 0.03, "dimtad": 1,  # text above the dimension line
        "dimdec": 2, "dimlunit": 2, "dimclrd": 1, "dimclre": 1, "dimclrt": 1,
    })
    return doc


def _add_walls(msp, geo) -> None:
    measured = geo.solid.difference(geo.nominal)
    for geom, pattern in ((measured, "SOLID"), (geo.nominal, "ANSI31")):
        for poly in iter_polygons(geom):
            rings = [poly.exterior, *poly.interiors]
            for ring in rings:
                msp.add_lwpolyline([_xy(p) for p in ring.coords[:-1]], close=True, dxfattribs={"layer": "WALLS"})
            hatch = msp.add_hatch(color=7 if pattern == "SOLID" else 8, dxfattribs={"layer": "WALLS"})
            if pattern != "SOLID":
                hatch.set_pattern_fill(pattern, scale=0.02)
            for k, ring in enumerate(rings):
                hatch.paths.add_polyline_path([_xy(p) for p in ring.coords[:-1]], is_closed=True,
                                              flags=1 if k == 0 else 0)


def _add_rooms(msp, plan: Plan) -> None:
    for room in plan.rooms:
        poly = room_polygon(room)
        if poly is None:
            continue
        msp.add_lwpolyline([_xy(p) for p in poly.exterior.coords[:-1]], close=True,
                           dxfattribs={"layer": "ROOMS"})
        lines = [room_title(room), fmt_area(room.floor_area), fmt_ceiling(room.ceiling_height)]
        x, y = _xy(polylabel(poly, tolerance=0.02).coords[0])
        mt = msp.add_mtext("\\P".join(lines), dxfattribs={"layer": "TEXT", "char_height": TEXT_H})
        mt.set_location((x, y), attachment_point=5)        # 5 = middle centre


def _add_dimensions(msp, plan: Plan) -> int:
    rooms = [p for r in plan.rooms if (p := room_polygon(r)) is not None]
    n = 0
    for w in plan.walls:
        p0, p1 = np.asarray(w.p0, float), np.asarray(w.p1, float)
        if np.linalg.norm(p1 - p0) < MIN_DIM_LENGTH_M:
            continue
        t = wall_thickness(w)[0]
        outside = exterior_side(w, rooms, t + DIM_OFFSET_M)
        side = -np.asarray(w.normal, float) if outside else np.asarray(w.normal, float)
        offset = (t if outside else 0.0) + DIM_OFFSET_M
        a, b = to_view(p0), to_view(p1)
        if b[0] < a[0] - 1e-9 or (abs(b[0] - a[0]) <= 1e-9 and b[1] < a[1]):
            a, b = b, a                                 # dimension text follows p1->p2: keep it readable
        d = b - a
        left = np.array([-d[1], d[0]])
        sign = 1.0 if float(left @ to_view(side)) > 0 else -1.0     # ezdxf: positive distance = left of p1->p2
        dim = msp.add_aligned_dim(p1=tuple(a), p2=tuple(b), distance=sign * offset,
                                  text=fmt_length(w.length).replace("\n", " "), dimstyle=DIMSTYLE,
                                  dxfattribs={"layer": "DIMENSIONS"})
        dim.render()                                   # builds the anonymous block CAD viewers display
        n += 1
    return n


def _arc_angles(pl: OpeningPlacement) -> tuple[float, float]:
    """Start/end angles (deg, counter-clockwise in DXF space) of the 90-degree swing from leaf to jamb."""
    leaf, jamb = to_view(pl.into), to_view(pl.b - pl.a)
    a0 = math.degrees(math.atan2(leaf[1], leaf[0])) % 360
    a1 = math.degrees(math.atan2(jamb[1], jamb[0])) % 360
    return (a0, a1) if (a1 - a0) % 360 <= 180 else (a1, a0)


def _add_openings(msp, geo) -> None:
    attrs = {"layer": "OPENINGS"}
    for pl in geo.placements:
        kind = pl.opening.kind
        if kind == "door":
            msp.add_line(_xy(pl.a), _xy(pl.a + pl.into * pl.width), dxfattribs=attrs)
            s, e = _arc_angles(pl)
            msp.add_arc(_xy(pl.a), pl.width, s, e, dxfattribs=attrs)
        elif kind == "window":
            t = wall_thickness(pl.host)[0]
            for f in (0.0, 1 / 3, 2 / 3, 1.0):
                msp.add_line(_xy(pl.a - pl.into * t * f), _xy(pl.b - pl.into * t * f), dxfattribs=attrs)
        else:
            for depth in (0.0, pl.depth):
                msp.add_line(_xy(pl.a - pl.into * depth), _xy(pl.b - pl.into * depth),
                             dxfattribs={**attrs, "linetype": "DASHED"})
        w = pl.opening.width
        label = fmt_length(w) if pl.width_observed else "width n/o"
        mid = (pl.a + pl.b) / 2 + pl.into * (TEXT_H * 0.8)
        text = msp.add_text(f"{kind} {label}", height=TEXT_H * 0.6, rotation=view_angle(pl.b - pl.a),
                            dxfattribs={"layer": "TEXT"})
        text.set_placement(_xy(mid), align=TextEntityAlignment.MIDDLE_CENTER)


def _add_title(msp, plan: Plan, geo) -> None:
    bounds = [p.bounds for p in iter_polygons(geo.solid)]
    if not bounds:
        return
    b = np.array(bounds)
    x, y = _xy((b[:, 0].min(), b[:, 1].min()))
    text = (f"{plan.capture_id} | {plan.tier.upper()} tier | footprint {fmt_area(plan.footprint_area)}\\P"
            f"Units: metres. Dimensions = interior wall-face length and 95% interval. "
            f"Hatched walls: thickness not observed (nominal {NOMINAL_THICKNESS_M:.2f} m).")
    mt = msp.add_mtext(text, dxfattribs={"layer": "TEXT", "char_height": TEXT_H * 1.2})
    mt.set_location((x, y + 1.0), attachment_point=7)       # 7 = bottom left, above the plan


def export_dxf(plan: Plan, path: str | Path) -> dict:
    """Write the DXF; return a small summary (entity counts per layer, audit errors) as evidence."""
    geo = build_wall_geometry(plan)
    doc = _new_document()
    msp = doc.modelspace()
    _add_walls(msp, geo)
    _add_rooms(msp, plan)
    n_dims = _add_dimensions(msp, plan)
    _add_openings(msp, geo)
    _add_title(msp, plan, geo)
    auditor = doc.audit()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    counts: dict[str, int] = {}
    for e in msp:
        counts[e.dxf.layer] = counts.get(e.dxf.layer, 0) + 1
    return {"path": str(path), "dimensions": n_dims, "entities_per_layer": counts,
            "audit_errors": len(auditor.errors), "insunits": doc.header["$INSUNITS"]}


def preview_dxf(dxf_path: str | Path, png_path: str | Path) -> None:
    """Rasterise the DXF with ezdxf's own drawing add-on: an independent check that the file reads back as a
    sensible plan (it does not share any code with render.py's matplotlib drawing)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    doc = ezdxf.readfile(dxf_path)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes((0, 0, 1, 1))
    Frontend(RenderContext(doc), MatplotlibBackend(ax)).draw_layout(doc.modelspace(), finalize=True)
    fig.savefig(png_path, dpi=150, facecolor="white", metadata={"Software": None})
    plt.close(fig)


__all__ = ["export_dxf", "preview_dxf", "LAYERS"]
