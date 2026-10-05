"""Plan -> SVG + PNG floor plan that a homeowner would recognise from poly.cam / magicplan.

What the drawing must show (Part 2, "the stitched plan is the product surface"): every room filled and labelled
with its area and ceiling height, walls with thickness, doors and windows, and a dimension on every wall with its
95% interval. Design choices (details and evidence in docs/modules/export.md):

  * One drawing, two files. Matplotlib draws the figure once and saves it as SVG (vector, text stays text,
    zoomable) and PNG (for reports and quick viewing). No second rendering path that could disagree.
  * True top-down view. The aligned frame is right-handed with +y up, so seen from above +z points DOWN the page.
    Drawing v = z upward would mirror the apartment, which a homeowner notices at once (their kitchen would be on
    the wrong side). Every point is drawn at (u, -v); see `to_view`.
  * Fixed scale in inches per metre, chosen from the plan size. Because the scale is known, font sizes in points
    convert exactly to metres, so we can test whether a label fits inside a room or along a wall before drawing it.
  * Walls are built per room as a mitred ring: each interior-face edge of the room polygon is pushed outward by
    that wall's thickness and consecutive offset lines are intersected. This gives clean corners at any angle
    (non-Manhattan walls included). Thickness = measured when the extractor measured it, else a nominal 0.10 m
    drawn hatched so nobody mistakes it for a measurement.
  * Openings are cut out of the wall solid; doors get a conventional leaf + swing arc, windows thin double lines,
    passages a dashed threshold. The swing direction is NOT measured, so it is drawn by convention (into the room
    with fewer doors, i.e. away from the corridor) and the legend says the symbol is conventional.
  * Labels are placed by a small greedy collision checker (`Labeler`): each label has a list of candidate
    positions / sizes / wordings, from best to most compact; the first that fits and overlaps nothing wins.
    Dimension text has priority over room labels, room labels over opening widths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Arc, Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from shapely.geometry import LineString, Point, Polygon  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402
from shapely.ops import polylabel, unary_union  # noqa: E402

from floorplan.model import Measurement, Opening, Plan, Room, Wall  # noqa: E402

NOMINAL_THICKNESS_M = 0.10      # typical interior partition (plasterboard on studs is 0.075-0.125 m)
EDGE_MATCH_TOL_M = 0.05         # a wall belongs to a polygon edge if it lies within 5 cm of it and is parallel
MIN_DIM_LENGTH_M = 0.25         # shorter walls (jogs, stubs) get no dimension line: unreadable, still in the JSON
MITRE_LIMIT = 4.0               # bevel a corner whose mitre would stick out more than 4 wall thicknesses


# ======================================================================================= formatting helpers
def fmt_interval(m: Optional[Measurement], unit_cm: bool = True) -> str:
    """'±1.2 cm' (or '±12 cm' once the interval is wider than 10 cm); '' when there is no interval."""
    if m is None or m.value is None or m.lo is None or m.hi is None:
        return ""
    hw = (m.hi - m.lo) / 2
    if not unit_cm:
        return f"±{hw:.2f}"
    return f"±{hw * 100:.1f} cm" if hw < 0.1 else f"±{hw * 100:.0f} cm"


def fmt_length(m: Optional[Measurement], compact: bool = False) -> str:
    """'3.42 m ±1.2 cm'. Inferred values get '≈' so a reader never confuses them with measured ones."""
    if m is None or m.value is None:
        return "n/o"
    approx = "≈" if m.status == "inferred" else ""
    ci = fmt_interval(m)
    sep = "\n" if compact else " "
    return f"{approx}{m.value:.2f} m{sep}{ci}".strip()


def fmt_area(m: Optional[Measurement]) -> str:
    if m is None or m.value is None:
        return "area not observed"
    approx = "≈" if m.status == "inferred" else ""
    ci = fmt_interval(m, unit_cm=False)
    return f"{approx}{m.value:.2f} m² {ci}".strip()


def fmt_ceiling(m: Optional[Measurement], short: bool = False) -> str:
    if m is None or m.value is None:
        return "ceiling n/o" if short else "ceiling not observed"
    approx = "≈" if m.status == "inferred" else ""
    return f"{'h' if short else 'ceiling'} {approx}{m.value:.2f} m {fmt_interval(m)}".strip()


def room_title(room: Room) -> str:
    """The room's floor-plan name (Bedroom, Kitchen, ...; plan/room_types.py) when it has one, else its label."""
    return room.name or room.label


def text_style(m: Optional[Measurement]) -> dict:
    if m is None or m.value is None or m.status == "not_observed":
        return {"color": "#8a8a8a", "style": "italic"}
    if m.status == "inferred":
        return {"color": "#6b6b6b", "style": "italic"}
    return {"color": "#1a1a1a", "style": "normal"}


# ======================================================================================= plan geometry
def to_view(p) -> np.ndarray:
    """Plan (u, v) -> drawing (x, y) for a true top-down view: y = -v (see module docstring)."""
    p = np.asarray(p, float)
    return p * np.array([1.0, -1.0])


def _unit(d: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-12 else np.array([1.0, 0.0])


def _left(d: np.ndarray) -> np.ndarray:
    return np.array([-d[1], d[0]])


def wall_thickness(w: Optional[Wall]) -> tuple[float, bool]:
    """(thickness to draw, is_nominal)."""
    if w is not None and w.thickness is not None and w.thickness.value is not None and w.thickness.value > 0:
        return float(w.thickness.value), False
    return NOMINAL_THICKNESS_M, True


def room_polygon(room: Room) -> Optional[Polygon]:
    if len(room.polygon) < 3:
        return None
    poly = Polygon(room.polygon)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    return orient(poly, sign=1.0)


def _match_wall(p0: np.ndarray, p1: np.ndarray, walls: Iterable[Wall]) -> Optional[Wall]:
    """The wall lying on polygon edge p0->p1: parallel, midpoint within EDGE_MATCH_TOL_M of the edge."""
    d = _unit(p1 - p0)
    edge = LineString([p0, p1])
    best, best_len = None, 0.0
    for w in walls:
        wd = np.subtract(w.p1, w.p0)
        L = float(np.linalg.norm(wd))
        if L < 1e-6 or abs(float(d @ (wd / L))) < 0.99:
            continue
        if edge.distance(Point(np.add(w.p0, w.p1) / 2)) < EDGE_MATCH_TOL_M and L > best_len:
            best, best_len = w, L
    return best


def _offset_vertex(p, d_prev, t_prev, d_next, t_next) -> list[np.ndarray]:
    """Outer corner for vertex p between an incoming edge (direction d_prev, thickness t_prev) and an outgoing one.
    Outward = right of a CCW edge. Mitre (line intersection) when well conditioned, else two bevel points."""
    o_prev, o_next = -_left(d_prev), -_left(d_next)
    a, b = p + o_prev * t_prev, p + o_next * t_next
    cross = d_prev[0] * d_next[1] - d_prev[1] * d_next[0]
    if abs(cross) < 0.1:                                    # (nearly) collinear: no corner to mitre
        return [a, b] if np.linalg.norm(a - b) > 1e-6 else [a]
    s = ((b - a)[0] * d_next[1] - (b - a)[1] * d_next[0]) / cross
    x = a + s * d_prev
    if np.linalg.norm(x - p) > MITRE_LIMIT * max(t_prev, t_next):
        return [a, b]
    return [x]


@dataclass
class EdgeBand:
    room_id: str
    wall: Optional[Wall]
    p0: np.ndarray
    p1: np.ndarray
    thickness: float
    nominal: bool

    def quad(self, extend: float = 0.0) -> Polygon:
        """The band between the interior face and the outer face, optionally extended past both ends."""
        d = _unit(self.p1 - self.p0)
        o = -_left(d) * self.thickness
        a, b = self.p0 - d * extend, self.p1 + d * extend
        return Polygon([a, b, b + o, a + o])


def room_ring(room: Room, walls: list[Wall]) -> tuple[Optional[Polygon], list[EdgeBand]]:
    """Mitred wall ring around one room (outer offset polygon minus the room) and its per-edge bands."""
    poly = room_polygon(room)
    if poly is None:
        return None, []
    pts = np.asarray(poly.exterior.coords[:-1], float)
    room_walls = [w for w in walls if w.room_id == room.id]
    bands = []
    for i in range(len(pts)):
        p0, p1 = pts[i], pts[(i + 1) % len(pts)]
        w = _match_wall(p0, p1, room_walls)
        t, nominal = wall_thickness(w)
        bands.append(EdgeBand(room.id, w, p0, p1, t, nominal))
    outer = []
    for i in range(len(pts)):
        prev, nxt = bands[i - 1], bands[i]
        outer += _offset_vertex(pts[i], _unit(prev.p1 - prev.p0), prev.thickness,
                                _unit(nxt.p1 - nxt.p0), nxt.thickness)
    outer_poly = Polygon(outer)
    if not outer_poly.is_valid:
        outer_poly = outer_poly.buffer(0)
    return outer_poly.union(poly).difference(poly), bands


def _seg_distance(p, w: Wall) -> float:
    return LineString([w.p0, w.p1]).distance(Point(p))


@dataclass
class OpeningPlacement:
    """Where an opening sits on the drawing: jamb points a, b on the face of the wall it is drawn from, the
    direction into the room the door swings into (`into`), and the wall depth to cut through (`depth`)."""
    opening: Opening
    host: Wall
    a: np.ndarray
    b: np.ndarray
    into: np.ndarray
    depth: float
    width: float
    width_observed: bool
    swing_room: Optional[str] = None


def _door_count(plan: Plan) -> dict[str, int]:
    count: dict[str, int] = {}
    for o in plan.openings:
        if o.kind in ("door", "passage"):
            for r in o.room_ids:
                count[r] = count.get(r, 0) + 1
    return count


def _swing_room(o: Opening, door_count: dict[str, int]) -> Optional[str]:
    """Swing side is not measured; convention: doors open into the room with fewer doors (bedroom, not corridor)."""
    if not o.room_ids:
        return None
    return min(o.room_ids, key=lambda r: (door_count.get(r, 0), o.room_ids.index(r)))


def place_opening(o: Opening, plan: Plan, walls_by_id: dict[str, Wall],
                  door_count: dict[str, int]) -> Optional[OpeningPlacement]:
    """Resolve an opening to drawable geometry. Robust to missing wall links (falls back to the nearest wall) and
    to a missing width (drawn at a nominal 0.8 m but flagged)."""
    linked = [walls_by_id[i] for i in o.wall_ids if i in walls_by_id]
    if not linked and plan.walls:
        linked = [min(plan.walls, key=lambda w: _seg_distance(o.center, w))]
    if not linked:
        return None
    swing = _swing_room(o, door_count) if o.kind == "door" else (o.room_ids[0] if o.room_ids else None)
    host = next((w for w in linked if w.room_id == swing), linked[0])
    d = _unit(np.subtract(host.p1, host.p0))
    n = np.asarray(host.normal, float)
    c = np.asarray(host.p0, float) + d * float((np.asarray(o.center) - np.asarray(host.p0)) @ d)
    observed = o.width is not None and o.width.value is not None and o.width.value > 0
    width = float(o.width.value) if observed else 0.8
    t_host, _ = wall_thickness(host)
    sep = max([abs(float((np.asarray(w.p0) - c) @ n)) + wall_thickness(w)[0] for w in linked] + [t_host])
    return OpeningPlacement(o, host, c - d * width / 2, c + d * width / 2, n, sep, width, observed, swing)


def opening_cut(pl: OpeningPlacement, margin: float = 0.02) -> Polygon:
    """Rectangle that removes the wall where the opening is: from just inside the host face, through the
    whole partition (both rooms' wall rings)."""
    n = pl.into
    return Polygon([pl.a + n * margin, pl.b + n * margin, pl.b - n * (pl.depth + margin),
                    pl.a - n * (pl.depth + margin)])


@dataclass
class WallGeometry:
    solid: object                       # shapely geometry of all walls, openings cut out
    nominal: object                     # the part of `solid` whose thickness is nominal (drawn hatched)
    bands: list[EdgeBand]
    placements: list[OpeningPlacement]


def build_wall_geometry(plan: Plan) -> WallGeometry:
    """All wall solids for the plan. Rooms' rings are unioned, then every room interior is subtracted (a ring
    must never intrude into a neighbouring room), then openings are cut out."""
    rings, bands = [], []
    room_ids = {r.id for r in plan.rooms}
    for room in plan.rooms:
        ring, b = room_ring(room, plan.walls)
        if ring is not None:
            rings.append(ring)
            bands += b
    for w in plan.walls:                                   # walls not attached to any drawn room
        if w.room_id not in room_ids:
            t, nominal = wall_thickness(w)
            band = EdgeBand(w.room_id, w, np.asarray(w.p0, float), np.asarray(w.p1, float), t, nominal)
            rings.append(band.quad())
            bands.append(band)
    interiors = unary_union([p for r in plan.rooms if (p := room_polygon(r)) is not None])
    walls_by_id = {w.id: w for w in plan.walls}
    counts = _door_count(plan)
    placements = [p for o in plan.openings if (p := place_opening(o, plan, walls_by_id, counts)) is not None]
    cuts = unary_union([opening_cut(p) for p in placements])
    solid = unary_union(rings).difference(interiors).difference(cuts)
    # Nominal = everything not backed by a measured thickness: solid minus the measured bands, each extended by
    # its thickness so that the corner and junction squares next to a measured wall count as measured, while a
    # corner between two nominal walls stays hatched.
    nominal = solid.difference(unary_union([b.quad(extend=b.thickness) for b in bands if not b.nominal]))
    return WallGeometry(solid, nominal, bands, placements)


def iter_polygons(geom) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from iter_polygons(g)


def exterior_side(w: Wall, rooms: list[Polygon], probe_m: float) -> bool:
    """True when the outward side of wall w is outside every room (envelope wall): its dimension goes outside the
    building. For a partition the outward side is a neighbouring room, so the dimension stays in w's own room
    (otherwise both rooms' dimension lines for the same partition would pile up in one room)."""
    mid = np.add(w.p0, w.p1) / 2
    probe = Point(mid - np.asarray(w.normal) * probe_m)
    return not any(r.contains(probe) for r in rooms)


# ======================================================================================= drawing
@dataclass(frozen=True)
class Style:
    target_size_in: float = 14.0       # longest side of the plan area
    min_scale: float = 0.35            # inches per metre
    max_scale: float = 1.4
    dpi: int = 200
    font_pt: float = 7.5               # dimension text
    room_font_pt: float = 9.0
    opening_font_pt: float = 5.5
    min_font_pt: float = 5.0
    dim_gap_pt: float = 9.0            # wall face (or outer face) to dimension line
    dim_step_pt: float = 8.0           # extra offset tried when a dimension label collides
    tick_pt: float = 3.5
    wall_color: str = "#2b2b2b"
    nominal_color: str = "#8c8c8c"
    door_color: str = "#1f4e79"
    window_color: str = "#2a7fb8"
    lowconf_color: str = "#d9822b"
    damage_color: str = "#c0392b"
    palette: tuple = ("#f6e7c1", "#d8ead3", "#d4e4f2", "#f4d7d7", "#e6dcf0", "#fbe3c9", "#d7efe9",
                      "#ece7d6", "#f2d9e6", "#dfe8c9", "#d9dff2", "#f0e0d0")


@dataclass
class RenderStats:
    walls: int = 0
    dims_drawn: int = 0
    dims_skipped_short: int = 0
    dims_forced: int = 0               # placed although no collision-free candidate existed
    room_labels_full: int = 0
    room_labels_reduced: int = 0
    room_labels_forced: int = 0
    opening_labels: int = 0
    opening_labels_dropped: int = 0
    nominal_walls: int = 0
    openings_unplaced: int = 0
    warnings: list[str] = field(default_factory=list)


class Labeler:
    """Greedy collision-free text placement in display space (pixels), using the real rendered extents."""

    def __init__(self, ax):
        self.ax = ax
        self.renderer = ax.figure.canvas.get_renderer()
        self.boxes: list[Bbox] = []          # hard obstacles: placed text
        self.soft: list[Bbox] = []           # soft obstacles: door swings (text may cover them if nothing else fits)

    def extent(self, text: str, size: float, style: str = "normal") -> tuple[float, float]:
        """Unrotated (width, height) of a possibly multi-line text, in points."""
        t = self.ax.text(0, 0, text, fontsize=size, style=style, linespacing=1.1)
        bb = t.get_window_extent(self.renderer)
        t.remove()
        px_per_pt = self.ax.figure.dpi / 72.0
        return bb.width / px_per_pt, bb.height / px_per_pt

    def try_place(self, candidates: list[dict], fits: Optional[Callable[[Bbox], bool]] = None, force: bool = False,
                  allow_soft: bool = False):
        """candidates: kwargs for ax.text (x, y, s, ...). Returns (text, index) or (None, -1)."""
        obstacles = self.boxes if allow_soft else self.boxes + self.soft
        for i, c in enumerate(candidates):
            t = self.ax.text(**c)
            bb = t.get_window_extent(self.renderer).expanded(1.04, 1.08)
            if not any(bb.overlaps(b) for b in obstacles) and (fits is None or fits(bb)):
                self.boxes.append(bb)
                return t, i
            t.remove()
        if force and candidates:
            t = self.ax.text(**candidates[-1])
            self.boxes.append(t.get_window_extent(self.renderer))
            return t, len(candidates) - 1
        return None, -1

    def reserve_points(self, xy: np.ndarray) -> None:
        """Mark the display-space bounding box of some drawing points (e.g. a door swing) as occupied."""
        d = self.ax.transData.transform(np.asarray(xy, float))
        self.soft.append(Bbox([d.min(0), d.max(0)]))


def view_angle(d: np.ndarray) -> float:
    """Text rotation (deg) for a plan direction d, kept upright (-90, 90]."""
    v = to_view(d)
    a = math.degrees(math.atan2(v[1], v[0]))
    if a > 90:
        a -= 180
    elif a <= -90:
        a += 180
    return a


class PlanRenderer:
    def __init__(self, plan: Plan, style: Style = Style()):
        self.plan, self.s = plan, style
        self.stats = RenderStats()
        self.geo = build_wall_geometry(plan)
        self.room_polys = {r.id: p for r in plan.rooms if (p := room_polygon(r)) is not None}

    # ------------------------------------------------------------------------------------------- layout
    def _bounds(self) -> tuple[float, float, float, float]:
        geoms = [g for g in iter_polygons(self.geo.solid)] + list(self.room_polys.values())
        if not geoms:
            return -1.0, -1.0, 1.0, 1.0
        return unary_union(geoms).bounds

    def _setup(self):
        u0, v0, u1, v1 = self._bounds()
        span = max(u1 - u0, v1 - v0, 1.0)
        self.scale = float(np.clip(self.s.target_size_in / span, self.s.min_scale, self.s.max_scale))
        self.m_per_pt = 1.0 / (72.0 * self.scale)
        margin = (self.s.dim_gap_pt + 3 * self.s.dim_step_pt + 2.5 * self.s.font_pt) * self.m_per_pt + 0.1
        x0, x1 = u0 - margin, u1 + margin
        y0, y1 = -v1 - margin, -v0 + margin                  # view y = -v
        plan_w, plan_h = (x1 - x0) * self.scale, (y1 - y0) * self.scale
        head, foot = 0.75, 1.15
        fig_w = max(plan_w, 9.0)
        fig = plt.figure(figsize=(fig_w, plan_h + head + foot), dpi=self.s.dpi)
        left = (fig_w - plan_w) / 2 / fig_w
        ax = fig.add_axes((left, foot / (plan_h + head + foot), plan_w / fig_w, plan_h / (plan_h + head + foot)))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.axis("off")
        self.fig, self.ax, self.fig_w, self.fig_h, self.foot = fig, ax, fig_w, plan_h + head + foot, foot
        self.lab = Labeler(ax)

    def pt(self, x: float) -> float:
        """Points -> metres at the drawing scale."""
        return x * self.m_per_pt

    # ------------------------------------------------------------------------------------------- primitives
    def _poly_patch(self, poly: Polygon, **kw):
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            xy = to_view(np.asarray(ring.coords))
            verts += list(xy)
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY]
        self.ax.add_patch(PathPatch(MplPath(verts, codes), **kw))

    def _line(self, pts, **kw):
        xy = to_view(np.asarray(pts, float))
        self.ax.plot(xy[:, 0], xy[:, 1], **kw)

    # ------------------------------------------------------------------------------------------- layers
    def draw_rooms(self):
        for i, room in enumerate(self.plan.rooms):
            poly = self.room_polys.get(room.id)
            if poly is None:
                self.stats.warnings.append(f"room {room.id}: invalid polygon, not drawn")
                continue
            self._poly_patch(poly, facecolor=self.s.palette[i % len(self.s.palette)], edgecolor="none", zorder=1)

    def draw_walls(self):
        for poly in iter_polygons(self.geo.solid):
            self._poly_patch(poly, facecolor=self.s.wall_color, edgecolor=self.s.wall_color, lw=0.3, zorder=3)
        for poly in iter_polygons(self.geo.nominal):
            self._poly_patch(poly, facecolor="white", edgecolor=self.s.nominal_color, lw=0.4, hatch="/////",
                             zorder=4)
        self.stats.walls = len(self.plan.walls)
        self.stats.nominal_walls = sum(1 for b in self.geo.bands if b.nominal)

    def draw_damage(self):
        for d in self.plan.damage:
            poly = d.get("plan_polygon")
            if poly and len(poly) >= 3:
                self._poly_patch(Polygon(poly), facecolor=self.s.damage_color, alpha=0.18,
                                 edgecolor=self.s.damage_color, lw=0.8, hatch="xx", zorder=2)
                c = to_view(np.asarray(Polygon(poly).centroid.coords[0]))
                self.ax.text(c[0], c[1], d.get("class", "damage").replace("_", " "), fontsize=self.s.min_font_pt,
                             color=self.s.damage_color, ha="center", va="center", zorder=9)

    def draw_openings(self):
        for pl in self.geo.placements:
            prior = pl.opening.source == "prior"           # a door assumed, not seen (D-085): always dashed
            color = self.s.lowconf_color if (pl.opening.confidence < 0.5 or prior) else None
            ls = "--" if (pl.opening.confidence < 0.5 or not pl.width_observed or prior) else "-"
            kind = pl.opening.kind
            if kind == "door":
                self._door(pl, color or self.s.door_color, ls)
            elif kind == "window":
                self._window(pl, color or self.s.window_color, ls)
            else:
                self._passage(pl, color or self.s.door_color)
        self.stats.openings_unplaced = len(self.plan.openings) - len(self.geo.placements)

    def _door(self, pl: OpeningPlacement, color: str, ls: str):
        leaf_end = pl.a + pl.into * pl.width
        self.lab.reserve_points(to_view([pl.a, pl.b, leaf_end, pl.b + pl.into * pl.width]))   # keep text off swings
        self._line([pl.a, leaf_end], color=color, lw=1.1, ls=ls, zorder=6)
        c = to_view(pl.a)
        v_leaf, v_jamb = to_view(leaf_end - pl.a), to_view(pl.b - pl.a)
        a0, a1 = sorted([math.degrees(math.atan2(v_leaf[1], v_leaf[0])), math.degrees(math.atan2(v_jamb[1], v_jamb[0]))])
        if a1 - a0 > 180:
            a0, a1 = a1, a0 + 360
        self.ax.add_patch(Arc(c, 2 * pl.width, 2 * pl.width, theta1=a0, theta2=a1, color=color, lw=0.6, ls=ls,
                              zorder=6))
        self._threshold(pl, color)

    def _threshold(self, pl: OpeningPlacement, color: str):
        """Thin line across the gap at the wall face, so the opening reads as part of the wall line."""
        self._line([pl.a, pl.b], color=color, lw=0.4, ls=(0, (2, 2)), zorder=6)

    def _passage(self, pl: OpeningPlacement, color: str):
        self._threshold(pl, color)
        self._line([pl.a - pl.into * pl.depth, pl.b - pl.into * pl.depth], color=color, lw=0.4, ls=(0, (2, 2)),
                   zorder=6)

    def _window(self, pl: OpeningPlacement, color: str, ls: str):
        t = wall_thickness(pl.host)[0]
        for f in (0.0, 1 / 3, 2 / 3, 1.0):
            lw = 0.5 if f in (0.0, 1.0) else 0.9
            self._line([pl.a - pl.into * t * f, pl.b - pl.into * t * f], color=color, lw=lw, ls=ls, zorder=6)
        for p in (pl.a, pl.b):
            self._line([p, p - pl.into * t], color=color, lw=0.5, zorder=6)

    # ------------------------------------------------------------------------------------------- dimensions
    def draw_dimensions(self):
        rooms = list(self.room_polys.values())
        walls = sorted(self.plan.walls, key=lambda w: -(w.length.value or 0))   # long walls claim space first
        for w in walls:
            L = float(np.linalg.norm(np.subtract(w.p1, w.p0)))
            if L < MIN_DIM_LENGTH_M:
                self.stats.dims_skipped_short += 1
                continue
            self._dimension(w, L, rooms)

    def _dim_candidates(self, w: Wall, L: float, outside: bool) -> list[tuple[float, dict]]:
        """(offset from interior face, text kwargs) from preferred to most compact."""
        d = _unit(np.subtract(w.p1, w.p0))
        n = np.asarray(w.normal, float)
        side = -n if outside else n
        t = wall_thickness(w)[0] if outside else 0.0
        st = text_style(w.length)
        rot = view_angle(d)
        out = []
        for k in range(3):
            off = t + self.pt(self.s.dim_gap_pt + k * self.s.dim_step_pt)
            for size in (self.s.font_pt, 0.85 * self.s.font_pt, self.s.min_font_pt):
                for compact in (False, True):
                    s = fmt_length(w.length, compact=compact)
                    wpt, hpt = self.lab.extent(s, size, st["style"])
                    if wpt * self.m_per_pt > 0.98 * L:
                        continue
                    mid = np.add(w.p0, w.p1) / 2 + side * (off + self.pt(1.5) + self.pt(hpt) / 2)
                    xy = to_view(mid)
                    out.append((off, dict(x=xy[0], y=xy[1], s=s, fontsize=size, rotation=rot, ha="center",
                                          va="center", rotation_mode="anchor", linespacing=1.1, zorder=8, **st)))
        return out

    def _dimension(self, w: Wall, L: float, rooms: list[Polygon]):
        outside = exterior_side(w, rooms, wall_thickness(w)[0] + self.pt(self.s.dim_gap_pt))
        # Order of preference: own side avoiding everything; own side allowed over a door swing; the other side
        # (inside the neighbouring room or outside) only as a last resort, because a dimension that sits next to
        # the wrong wall is worse than one drawn over a door arc.
        own = [(o, c, outside) for o, c in self._dim_candidates(w, L, outside)]
        other = [(o, c, not outside) for o, c in self._dim_candidates(w, L, not outside)]
        cands = own + own + other
        if not cands:                                   # text longer than the wall at every size: short label
            st = text_style(w.length)
            s = "n/o" if w.length.value is None else f"{w.length.value:.2f}"
            off = (wall_thickness(w)[0] if outside else 0.0) + self.pt(self.s.dim_gap_pt)
            side = -np.asarray(w.normal) if outside else np.asarray(w.normal)
            xy = to_view(np.add(w.p0, w.p1) / 2 + side * (off + self.pt(1.5 + self.s.min_font_pt / 2)))
            cands = [(off, dict(x=xy[0], y=xy[1], s=s, fontsize=self.s.min_font_pt, rotation=view_angle(
                np.subtract(w.p1, w.p0)), ha="center", va="center", rotation_mode="anchor", zorder=8, **st),
                outside)]
        idx = self._place_in_stages([c for _, c, _ in cands], len(own))
        if idx < 0:
            self.stats.dims_forced += 1
            idx = 0
            self.lab.try_place([cands[0][1]], force=True)
        self._dim_line(w, cands[idx][0], cands[idx][2])
        self.stats.dims_drawn += 1

    def _place_in_stages(self, cands: list[dict], n_own: int) -> int:
        """Stage 1: own side, no overlaps at all. Stage 2: own side, door swings allowed. Stage 3: other side."""
        stages = [(0, n_own, False), (n_own, 2 * n_own, True), (2 * n_own, len(cands), False)]
        for lo, hi, soft in stages:
            if hi > lo:
                _, i = self.lab.try_place(cands[lo:hi], allow_soft=soft)
                if i >= 0:
                    return lo + i
        return -1

    def _dim_line(self, w: Wall, off: float, outside: bool):
        p0, p1 = np.asarray(w.p0, float), np.asarray(w.p1, float)
        d = _unit(p1 - p0)
        side = -np.asarray(w.normal, float) if outside else np.asarray(w.normal, float)
        q0, q1 = p0 + side * off, p1 + side * off
        color = text_style(w.length)["color"]
        self._line([q0, q1], color=color, lw=0.45, zorder=7)
        start = (wall_thickness(w)[0] if outside else 0.0) + self.pt(1.5)
        for p, q in ((p0, q0), (p1, q1)):
            self._line([p + side * start, q + side * self.pt(2.5)], color=color, lw=0.3, zorder=7)
            tick = (d + side) / math.sqrt(2) * self.pt(self.s.tick_pt) / 2      # 45-degree architectural tick
            self._line([q - tick, q + tick], color=color, lw=0.9, zorder=7)

    # ------------------------------------------------------------------------------------------- room labels
    def draw_room_labels(self):
        for room in self.plan.rooms:
            poly = self.room_polys.get(room.id)
            if poly is not None:
                self._room_label(room, poly)

    @staticmethod
    def _label_anchors(poly: Polygon, n: int = 12) -> list[Point]:
        """Pole of inaccessibility (the interior point farthest from all walls: better than the centroid for
        L-shapes), then grid points ordered by distance to the walls, as fallbacks when the best spot is taken."""
        best = polylabel(poly, tolerance=0.02)
        x0, y0, x1, y1 = poly.bounds
        step = max(x1 - x0, y1 - y0) / 8
        grid = [Point(x, y) for x in np.arange(x0 + step / 2, x1, step) for y in np.arange(y0 + step / 2, y1, step)]
        grid = sorted((p for p in grid if poly.contains(p)), key=lambda p: -poly.exterior.distance(p))
        return [best] + grid[:n]

    def _room_label_candidates(self, room: Room, poly: Polygon) -> list[dict]:
        anchors = self._label_anchors(poly)
        title = room_title(room)
        variants = [
            [title, fmt_area(room.floor_area), fmt_ceiling(room.ceiling_height)],
            [title, fmt_area(room.floor_area), fmt_ceiling(room.ceiling_height, short=True)],
            [title, fmt_area(room.floor_area)],
            [title],
        ]
        sizes = (self.s.room_font_pt, 0.8 * self.s.room_font_pt, self.s.min_font_pt)
        out = []
        for lines in variants:
            for size in sizes:
                for a in anchors:
                    xy = to_view(np.asarray(a.coords[0]))
                    out.append(dict(x=xy[0], y=xy[1], s="\n".join(lines), fontsize=size, ha="center",
                                    va="center", linespacing=1.15, zorder=8, color="#1a1a1a"))
        return out

    def _room_label(self, room: Room, poly: Polygon):
        inner = poly.buffer(-self.pt(2))
        inv = self.ax.transData.inverted()

        def fits(bb: Bbox) -> bool:
            corners = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
            return not inner.is_empty and inner.contains(Polygon(corners * np.array([1.0, -1.0])))

        cands = self._room_label_candidates(room, poly)
        text, idx = self.lab.try_place(cands, fits)
        if text is None:
            text, idx = self.lab.try_place(cands, fits, allow_soft=True)
        if text is None:
            self.stats.room_labels_forced += 1
            text, idx = self.lab.try_place([cands[-1]], force=True)
        elif cands[idx]["s"] == cands[0]["s"] and cands[idx]["fontsize"] == cands[0]["fontsize"]:
            self.stats.room_labels_full += 1
        else:
            self.stats.room_labels_reduced += 1
        text.set_fontweight("normal")
        self._bold_first_line(text)

    def _bold_first_line(self, text):
        """Room name in bold above the numbers: draw the name as a separate bold text at the same spot."""
        lines = text.get_text().split("\n")
        if len(lines) == 1:
            text.set_fontweight("bold")
            return
        text.set_text("\n" + "\n".join(lines[1:]))
        x, y = text.get_position()
        self.ax.text(x, y, "\n".join([lines[0]] + [""] * (len(lines) - 1)), fontsize=text.get_fontsize(),
                     ha="center", va="center", linespacing=1.15, fontweight="bold", zorder=8, color="#1a1a1a")

    # ------------------------------------------------------------------------------------------- opening labels
    def draw_opening_labels(self):
        for pl in self.geo.placements:
            w = pl.opening.width
            s = (f"{pl.width:.2f} {fmt_interval(w)}" if pl.width_observed else "width n/o").strip()
            st = text_style(w)
            rot = view_angle(pl.b - pl.a)
            mid = (pl.a + pl.b) / 2
            size = self.s.opening_font_pt
            hpt = self.lab.extent(s, size)[1]
            # doors/passages: inside the empty gap first; windows: never on top of the glazing lines
            spots = [] if pl.opening.kind == "window" else [mid - pl.into * pl.depth / 2]
            spots += [mid + pl.into * (self.pt(hpt) / 2 + self.pt(2))]
            spots += [mid - pl.into * (pl.depth + self.pt(hpt) / 2 + self.pt(2))]
            cands = [dict(x=to_view(p)[0], y=to_view(p)[1], s=s, fontsize=size, rotation=rot, ha="center",
                          va="center", rotation_mode="anchor", zorder=9, color=st["color"], style=st["style"])
                     for p in spots]
            text, _ = self.lab.try_place(cands)
            if text is None:
                self.stats.opening_labels_dropped += 1
            else:
                self.stats.opening_labels += 1

    # ------------------------------------------------------------------------------------------- frame
    def draw_frame(self):
        p = self.plan
        fp = p.footprint_area
        title = f"{p.capture_id}  ·  {p.tier.upper()} tier  ·  {len(p.rooms)} rooms  ·  footprint {fmt_area(fp)}"
        if p.meta.get("synthetic"):
            title += "  ·  SYNTHETIC TEST FIXTURE"
        sub = ("Top-down view.  Dimensions: interior wall-face length ± half-width of the 95% interval.  "
               "≈ grey italic = inferred, n/o = not observed.  Door swing is drawn by convention (not measured).")
        self.fig.text(0.5, 1 - 0.28 / self.fig_h, title, ha="center", va="center", fontsize=12, fontweight="bold")
        self.fig.text(0.5, 1 - 0.55 / self.fig_h, sub, ha="center", va="center", fontsize=7.5, color="#444")
        self._scale_bar()
        self._legend()

    def _scale_bar(self):
        span_m = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
        L = next(x for x in (1, 2, 5, 10, 20, 50, 100) if x >= span_m * 0.15) if span_m * 0.15 > 1 else 1
        n = 5 if L in (5, 50) else (4 if L in (2, 20) else 2 if L == 1 else 5)
        w_in = L * self.scale
        x0_in, y0_in = 0.4, 0.55
        ax = self.fig.add_axes((x0_in / self.fig_w, y0_in / self.fig_h, w_in / self.fig_w, 0.08 / self.fig_h))
        ax.set_xlim(0, L)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for i in range(n):
            ax.add_patch(plt.Rectangle((i * L / n, 0), L / n, 1, facecolor="black" if i % 2 == 0 else "white",
                                       edgecolor="black", lw=0.6))
        for x in (0, L):
            ax.text(x, -0.6, f"{x:g} m", ha="center", va="top", fontsize=7)
        ax.text(0, 1.6, "scale", ha="left", va="bottom", fontsize=7)

    def _legend(self):
        s = self.s
        items = [
            Patch(facecolor=s.wall_color, edgecolor=s.wall_color, label="wall, measured thickness"),
            Patch(facecolor="white", edgecolor=s.nominal_color, hatch="/////",
                  label=f"wall, thickness not observed (nominal {NOMINAL_THICKNESS_M:.2f} m)"),
            Line2D([], [], color=s.door_color, lw=1.1, label="door (leaf + swing)"),
            Line2D([], [], color=s.door_color, lw=0.6, ls=(0, (2, 2)), label="passage (no door)"),
            Line2D([], [], color=s.window_color, lw=0.9, label="window"),
            Line2D([], [], color=s.lowconf_color, lw=0.9, ls="--", label="opening, low confidence or assumed (prior)"),
        ]
        if any(d.get("plan_polygon") for d in self.plan.damage):
            items.append(Patch(facecolor=s.damage_color, alpha=0.3, edgecolor=s.damage_color, hatch="xx",
                               label="damage region (floor)"))
        self.fig.legend(handles=items, loc="lower right", ncol=2, fontsize=7, frameon=False,
                        bbox_to_anchor=(1 - 0.3 / self.fig_w, 0.12 / self.fig_h))

    # ------------------------------------------------------------------------------------------- entry
    def render(self):
        self._setup()
        self.draw_rooms()
        self.draw_damage()
        self.draw_walls()
        self.draw_openings()
        self.draw_dimensions()
        self.draw_room_labels()
        self.draw_opening_labels()
        self.draw_frame()
        return self.fig


def render_plan(plan: Plan, out_stem: str | Path, style: Style = Style()) -> RenderStats:
    """Render `plan` to <out_stem>.svg and <out_stem>.png. Deterministic: fixed SVG id salt, no timestamps."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "floorplan", "hatch.linewidth": 0.4,
                         "font.family": "DejaVu Sans"}):
        r = PlanRenderer(plan, style)
        fig = r.render()
        fig.savefig(out_stem.with_suffix(".svg"), metadata={"Date": None})
        fig.savefig(out_stem.with_suffix(".png"), dpi=style.dpi, metadata={"Software": None})
        plt.close(fig)
    return r.stats


__all__ = ["render_plan", "room_title", "build_wall_geometry", "WallGeometry", "OpeningPlacement", "Style", "RenderStats",
           "to_view", "view_angle", "room_polygon", "wall_thickness", "exterior_side", "iter_polygons", "fmt_length", "fmt_area", "fmt_ceiling",
           "NOMINAL_THICKNESS_M"]
