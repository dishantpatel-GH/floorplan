"""Plan -> a clean presentation drawing (plan_presentation.svg / .png), written beside the technical plan.png.

Why a second drawing. The technical drawing (render.py) is for checking a plan: every wall has its length and 95%
interval, walls of unknown thickness are hatched, low-confidence openings are orange. Shown to a homeowner, that is
noise. A real-estate plan (our reference is CubiCasa's plan of the own house, cubicasa/NearMaheshwariBhawan_dim_0.jpg)
shows the same plan with much less ink:

  * dark grey solid walls, a light warm fill per room, the room name in capitals and its size "W x L m" under it;
  * doors as a gap with a leaf and a swing arc, windows as double thin lines in the wall, doorless openings as gaps;
  * a footer with the total area and a small scale bar. No intervals, no dimension chains.

The geometry is the plan's. Room outlines (with their notches and pillars), wall positions, opening positions and
widths come from plan.json unchanged; nothing here measures or moves anything. What is a drawing convention:
  * A wall whose thickness was not observed is drawn at a nominal thickness (0.15 m on the building's outside,
    0.10 m between rooms). A measured thickness is drawn as measured, kept within 0.08-0.35 m so one bad fit does not
    draw a hairline or a slab.
  * A gap narrower than CLOSE_GAP_M between two rooms' walls is drawn as wall. The space between two rooms' wall
    faces is the partition, so the drawing shows one solid wall instead of two thin walls with a slit between them.
    A hole smaller than HOLE_FILL_M2 that walls close in on all sides is drawn as wall too.
  * Slits and spikes narrower than SLIVER_M in a room outline are drawn closed (a 3 cm slit is extractor noise, a
    pillar is 20 cm or more). Room sizes and the total are computed from the plan's own outlines.
  * Door swing side and angle are not measured. Same rule as the technical drawing: the door opens into the room with
    fewer doors, hinged on one jamb.
  * "W x L" is the room's bounding box in its own wall directions, rounded to cm, as on a real-estate plan (an
    L-shaped room has less area than W x L). The first number is the side that runs across the page.
  * Text is drawn as outlines (the SVG needs no fonts) in a geometric sans when the machine has one.
"""
from __future__ import annotations

import math
import re
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Arc, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.textpath import TextPath, text_to_path  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from shapely import affinity  # noqa: E402
from shapely.geometry import LineString, Point, Polygon  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402
from shapely.ops import polylabel, unary_union  # noqa: E402

from floorplan.export.render import (OpeningPlacement, _door_count, _match_wall, _offset_vertex, _unit,  # noqa: E402
                                     iter_polygons, place_opening, room_polygon, to_view)
from floorplan.model import Plan, Room, Wall  # noqa: E402

CLOSE_GAP_M = 0.30           # a slit narrower than this between two rooms' walls is drawn as wall
DOUBLE_DOOR_M = 1.2          # a door wider than this is drawn as two leaves
HOLE_FILL_M2 = 0.6           # a hole this small, closed in by walls on all sides, is drawn as wall (duct, thick corner)
SLIVER_M = 0.06              # outline slits and spikes narrower than this are drawing noise, not pillars
PARTITION_TOUCH_M = 0.05     # two room outlines this close share a partition that neither ring can show
GENERIC_NAMES = {"", "room", "unknown", "space", "area"}


# ======================================================================================= style
@dataclass(frozen=True)
class PresentationStyle:
    page_landscape_in: tuple = (12.0, 9.0)
    margin_in: float = 0.7
    head_in: float = 0.5
    foot_in: float = 1.45
    max_scale: float = 2.0             # inches per metre (a single small room must not fill a poster)
    dpi: int = 200
    wall_color: str = "#767676"
    line_color: str = "#767676"        # door leaves, arcs, window lines
    text_color: str = "#6b6b6b"
    footer_color: str = "#1f1f1f"
    window_fill: str = "#ffffff"
    tints: tuple = ("#f4ede7", "#efe3d6", "#f3e7dd", "#ebe4da", "#f6ebe0", "#ede2dc", "#f1e8dd", "#e9e0d3")
    nominal_exterior_m: float = 0.15   # building outside, thickness not observed (drawing convention)
    nominal_partition_m: float = 0.10  # between rooms, thickness not observed
    min_wall_m: float = 0.08
    max_wall_m: float = 0.35
    door_open_deg: float = 90.0        # leaf drawn open at this angle, arc from the closed position
    rotate_deg: float = 0.0            # turn the drawing counter-clockwise on the page (the heading is arbitrary)
    line_pt: float = 0.8
    name_cap_m: float = 0.15           # room-name letter height in plan metres (CubiCasa's is about 0.16 m)
    max_font_pt: float = 20.0
    min_font_pt: float = 6.0
    tracking_em: float = 0.08          # extra letter spacing of room names
    label_fonts: tuple = ("URW Gothic", "Josefin Sans", "Century Gothic", "Lato", "DejaVu Sans")
    footer_fonts: tuple = ("Open Sans", "Lato", "DejaVu Sans")


@dataclass
class PresentationStats:
    rooms: int = 0
    labels_full: int = 0               # name and size at the preferred size
    labels_reduced: int = 0            # smaller, rotated or name only
    labels_forced: int = 0             # nothing fitted; drawn at the smallest size anyway
    doors: int = 0
    windows: int = 0
    passages: int = 0
    openings_unplaced: int = 0
    nominal_edges: int = 0
    gap_fill_m2: float = 0.0           # wall area added by closing slits between rooms
    partition_m: float = 0.0           # length of partition drawn between rooms that touch
    texts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _first_font(families: tuple, weight: str = "normal") -> font_manager.FontProperties:
    for fam in families:
        try:
            font_manager.findfont(font_manager.FontProperties(family=fam, weight=weight), fallback_to_default=False)
            return font_manager.FontProperties(family=fam, weight=weight)
        except ValueError:
            continue
    return font_manager.FontProperties(family="DejaVu Sans", weight=weight)


# ======================================================================================= names and sizes
def room_display_name(room: Room) -> str:
    """Name in capitals: a name/type field when the room has one, else its label; generic labels become 'ROOM'.
    '02_bedroom' -> 'BEDROOM' (photo folders are numbered for capture order), 'bedroom2' -> 'BEDROOM 2', 'R3' -> 'ROOM'.
    A note in brackets ('corridor (not entered; seen through an opening)') stays in the technical drawing."""
    for attr in ("name", "room_type", "type", "label"):
        v = getattr(room, attr, None)
        if isinstance(v, str) and v.strip():
            s = re.sub(r"\(.*?\)", "", v).strip()
            s = re.sub(r"^\d+[_\-\s]+", "", s)
            s = re.sub(r"[_\-]+", " ", s)
            s = re.sub(r"(?<=[A-Za-z])(?=\d+$)", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            if s.lower() in GENERIC_NAMES or re.fullmatch(r"[rR] ?\d+", s):
                continue
            return s.upper()
    return "ROOM"


def _dominant_angle(poly: Polygon) -> float:
    """Wall direction of a room in radians, in (-pi/4, pi/4]: the length-weighted circular mean of its edge angles
    mod 90 deg."""
    pts = np.asarray(poly.exterior.coords, float)
    d = np.diff(pts, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    a = np.arctan2(d[:, 1], d[:, 0]) * 4.0               # mod 90 deg -> full circle
    c, s = float((L * np.cos(a)).sum()), float((L * np.sin(a)).sum())
    return math.atan2(s, c) / 4.0 if (c or s) else 0.0


def room_size_wl(poly: Polygon) -> tuple[float, float]:
    """(W, L): the room's bounding box in its own wall directions. W is the side closer to the page's horizontal."""
    th = _dominant_angle(poly)
    pts = np.asarray(poly.exterior.coords, float)
    e1 = np.array([math.cos(th), math.sin(th)])
    e2 = np.array([-math.sin(th), math.cos(th)])
    s1, s2 = pts @ e1, pts @ e2
    a, b = float(s1.max() - s1.min()), float(s2.max() - s2.min())
    # e1 is within 45 deg of +u, and +u is the page's horizontal (to_view keeps u, flips v)
    return a, b


def _upright(deg: float) -> float:
    """A text angle in (-90, 90], so the label never reads upside down."""
    deg = (deg + 180.0) % 360.0 - 180.0
    return deg - 180.0 if deg > 90.0 else (deg + 180.0 if deg <= -90.0 else deg)


def _label_rotations(poly: Polygon, W: float, L: float) -> list[float]:
    """Text angles to try, in order: horizontal; along the room's walls when the room is turned; across it when
    the room is long and narrow that way (a corridor drawn tall gets its name written upwards)."""
    a = -math.degrees(_dominant_angle(poly))             # plan angle -> page angle (the page flips v)
    rots = [0.0]
    if abs(a) > 2.0:
        rots.append(_upright(a))
    if L > 1.3 * W:
        rots.append(_upright(a + 90.0))
    return rots


# ======================================================================================= wall geometry
def _display_thickness(w: Optional[Wall], exterior: bool, s: PresentationStyle) -> tuple[float, bool]:
    if w is not None and w.thickness is not None and w.thickness.value is not None and w.thickness.value > 0:
        return float(np.clip(w.thickness.value, s.min_wall_m, s.max_wall_m)), False
    return (s.nominal_exterior_m if exterior else s.nominal_partition_m), True


def _ring(poly: Polygon, walls: list[Wall], others, s: PresentationStyle, stats: PresentationStats) -> Polygon:
    """Mitred wall ring outside one room polygon, each edge at its display thickness."""
    pts = np.asarray(poly.exterior.coords[:-1], float)
    n = len(pts)
    th = []
    for i in range(n):
        p0, p1 = pts[i], pts[(i + 1) % n]
        w = _match_wall(p0, p1, walls)
        d = _unit(p1 - p0)
        probe = Point((p0 + p1) / 2 + np.array([d[1], -d[0]]) * (s.nominal_partition_m + 0.05))   # outward = right
        exterior = others is None or not others.contains(probe)
        t, nominal = _display_thickness(w, exterior, s)
        stats.nominal_edges += nominal
        th.append(t)
    outer = []
    for i in range(n):
        outer += _offset_vertex(pts[i], _unit(pts[i] - pts[i - 1]), th[i - 1], _unit(pts[(i + 1) % n] - pts[i]), th[i])
    ring = Polygon(outer)
    if not ring.is_valid:
        ring = ring.buffer(0)
    return ring.union(poly)


@dataclass
class OpeningGap:
    pl: OpeningPlacement
    depth: float                       # host face to the far side of the wall (other room's face or outside)
    gap: object                        # the wall area removed (plan coordinates)
    tint: str
    s_in: float = 0.0                  # the removed wall reaches this far into the host room (a partition band)
    s_out: float = 0.0                 # ... and this far behind the host face

    def frame(self) -> list[np.ndarray]:
        """Outline of the opening's place in the wall: jamb to jamb, across the whole removed depth."""
        a, b, n = self.pl.a, self.pl.b, self.pl.into
        return [a + n * self.s_in, b + n * self.s_in, b - n * self.s_out, a - n * self.s_out, a + n * self.s_in]


def _depth_through(solid, pl: OpeningPlacement, polys: dict, max_depth: float = 0.8) -> float:
    """How deep to cut the wall behind an opening. Between two rooms: up to the other room's face, so the doorway
    opens the whole partition. Otherwise: the run of wall solid from the host face outward, median over three
    lines across the opening (one line may run along a wall that meets this one at a right angle)."""
    m = (pl.a + pl.b) / 2
    ray = LineString([m + pl.into * 0.01, m - pl.into * max_depth])
    behind = []
    for rid in pl.opening.room_ids:
        if rid != pl.host.room_id and rid in polys:
            hit = ray.intersection(polys[rid])
            if not hit.is_empty:
                behind.append(min(float((m - np.asarray(c)) @ pl.into) for g in getattr(hit, "geoms", [hit])
                                  for c in g.coords))
    if behind:
        return float(np.clip(min(behind), 0.02, max_depth))
    reaches = []
    for f in (0.25, 0.5, 0.75):
        p = pl.a + (pl.b - pl.a) * f
        inter = LineString([p + pl.into * 0.01, p - pl.into * max_depth]).intersection(solid)
        reach = 0.0
        pieces = sorted((g for g in getattr(inter, "geoms", [inter]) if g.geom_type == "LineString" and not g.is_empty),
                        key=lambda g: min(float((p - np.asarray(c)) @ pl.into) for c in g.coords))
        for g in pieces:
            ds = [float((p - np.asarray(c)) @ pl.into) for c in g.coords]
            if min(ds) > reach + 0.04:                   # a separate wall further out: stop
                break
            reach = max(reach, max(ds))
        reaches.append(reach)
    reach = float(np.median(reaches))
    return reach if reach > 0.02 else pl.depth


def _tidy(poly: Polygon) -> Polygon:
    """Close slits and drop spikes narrower than SLIVER_M (closing, then opening, with square corners). Keeps the
    outline when the result is not one polygon or changes the area by more than 3%."""
    e = SLIVER_M / 2
    q = poly.buffer(e, join_style=2, mitre_limit=5.0).buffer(-2 * e, join_style=2, mitre_limit=5.0)
    q = q.buffer(e, join_style=2, mitre_limit=5.0)
    if q.is_empty or q.geom_type != "Polygon" or abs(q.area - poly.area) > 0.03 * poly.area:
        return poly
    return orient(q.simplify(0.002), sign=1.0)


def _partition_bands(polys: dict, s: PresentationStyle):
    """Walls between rooms that touch (or nearly: within PARTITION_TOUCH_M). Their outlines share a line, so neither
    room's ring can show the partition (each ring is cut away by the other room). Draw a band of the nominal partition
    thickness centred on the shared line; it covers the touching edge of both rooms by half the thickness."""
    half = s.nominal_partition_m / 2
    ids = list(polys)
    lines = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pa, pb = polys[a], polys[b]
            if pa.distance(pb) > PARTITION_TOUCH_M:
                continue
            for p, q in ((pa, pb), (pb, pa)):
                shared = p.exterior.intersection(q.buffer(PARTITION_TOUCH_M, join_style=2))
                lines += [g for g in getattr(shared, "geoms", [shared])
                          if g.geom_type in ("LineString", "MultiLineString") and g.length > 0.15]
    if not lines:
        return Polygon()
    return unary_union([ln.buffer(half, cap_style=2, join_style=2) for ln in lines])


def build_presentation_walls(plan: Plan, s: PresentationStyle, stats: PresentationStats):
    """(wall solid with openings cut, list of OpeningGap, room polygons by id, interiors union)."""
    polys = {r.id: _tidy(p) for r in plan.rooms if (p := room_polygon(r)) is not None}
    for r in plan.rooms:
        if r.id not in polys:
            stats.warnings.append(f"room {r.id}: invalid polygon, not drawn")
    interiors = unary_union(list(polys.values())) if polys else Polygon()
    pieces = []
    for rid, poly in polys.items():
        others = unary_union([p for k, p in polys.items() if k != rid]) if len(polys) > 1 else None
        pieces.append(_ring(poly, [w for w in plan.walls if w.room_id == rid], others, s, stats))
    for w in plan.walls:                                  # walls not attached to a drawn room
        if w.room_id not in polys:
            d = _unit(np.subtract(w.p1, w.p0))
            t, _ = _display_thickness(w, True, s)
            o = -np.array([-d[1], d[0]]) * t
            p0, p1 = np.asarray(w.p0, float), np.asarray(w.p1, float)
            pieces.append(Polygon([p0, p1, p1 + o, p0 + o]))
    if not pieces:
        return Polygon(), [], polys, interiors
    whole = unary_union(pieces)
    g = CLOSE_GAP_M / 2
    closed = whole.buffer(g, join_style=2, mitre_limit=3.0).buffer(-g, join_style=2, mitre_limit=3.0)
    closed = closed.union(whole)                          # closing never removes anything
    holes = [Polygon(h) for q in iter_polygons(closed) for h in q.interiors if Polygon(h).area < HOLE_FILL_M2]
    if holes:
        closed = unary_union([closed] + holes)
    stats.gap_fill_m2 = round(float(closed.area - whole.area), 3)
    bands = _partition_bands(polys, s)
    stats.partition_m = round(float(bands.area / s.nominal_partition_m), 2) if not bands.is_empty else 0.0
    solid = closed.difference(interiors).union(bands)

    walls_by_id = {w.id: w for w in plan.walls}
    counts = _door_count(plan)
    gaps, cuts, tint_of = [], [], room_tints(plan, polys, s)
    reach = s.nominal_partition_m / 2 + 0.01            # how far a partition band can stick out of a face
    for o in plan.openings:
        pl = place_opening(o, plan, walls_by_id, counts)
        if pl is None:
            stats.openings_unplaced += 1
            continue
        a, b, n = pl.a, pl.b, pl.into
        depth = _depth_through(solid, pl, polys)
        s_in, s_out = 0.0, depth                         # the cut, measured from the host face along n
        if not bands.is_empty:                           # widen it to take out a partition band at the opening
            strip = Polygon([a + n * reach, b + n * reach, b - n * (depth + reach), a - n * (depth + reach)])
            sv = [float((np.asarray(c) - a) @ n) for q in iter_polygons(strip.intersection(bands))
                  for c in q.exterior.coords]
            if sv:
                s_in, s_out = max(0.0, max(sv)), max(depth, -min(sv))
        e = 0.01
        cut = Polygon([a + n * (s_in + e), b + n * (s_in + e), b - n * (s_out + e), a - n * (s_out + e)])
        room = pl.swing_room if pl.swing_room in tint_of else next((r for r in o.room_ids if r in tint_of), None)
        gaps.append(OpeningGap(pl, depth, cut.intersection(solid), tint_of.get(room, s.tints[0]), s_in, s_out))
        cuts.append(cut)
    if cuts:
        solid = solid.difference(unary_union(cuts))
    return solid, gaps, polys, interiors


def room_tints(plan: Plan, polys: dict, s: PresentationStyle) -> dict[str, str]:
    """A tint per room; rooms within 0.6 m of each other get different tints (greedy colouring)."""
    out: dict[str, str] = {}
    for rid, p in polys.items():
        near = {out[k] for k, q in polys.items() if k in out and p.distance(q) < 0.6}
        out[rid] = next((t for t in s.tints if t not in near), s.tints[len(out) % len(s.tints)])
    return out


# ======================================================================================= text as outlines
def text_outline(s: str, fp: font_manager.FontProperties, size: float, tracking_em: float = 0.0) -> MplPath:
    """Outline of a one-line text in points: letters spaced by tracking_em * size, baseline at y = 0, from x = 0."""
    fp = fp.copy()
    fp.set_size(size)
    if tracking_em <= 0:
        return TextPath((0, 0), s, prop=fp)
    verts, codes, x = [], [], 0.0
    for ch in s:
        if ch != " ":
            tp = TextPath((x, 0), ch, prop=fp)
            if len(tp.vertices):
                verts.append(tp.vertices)
                codes.append(tp.codes)
        x += text_to_path.get_text_width_height_descent(ch, fp, ismath=False)[0] + tracking_em * size
    if not verts:
        return MplPath(np.zeros((1, 2)), [MplPath.MOVETO])
    return MplPath(np.concatenate(verts), np.concatenate(codes))


def _centered(paths: list[MplPath], gaps_pt: list[float]) -> MplPath:
    """Stack one-line outlines top to bottom (each centred horizontally), then centre the block's ink on (0, 0)."""
    out_v, out_c, y = [], [], 0.0
    for i, p in enumerate(paths):
        ext = p.get_extents()
        dy = y - ext.y1                                   # this line's top at y
        out_v.append(p.vertices + np.array([-(ext.x0 + ext.x1) / 2, dy]))
        out_c.append(p.codes)
        y = y - ext.height - (gaps_pt[i] if i < len(gaps_pt) else 0.0)
    v = np.concatenate(out_v)
    v = v - (v.min(0) + v.max(0)) / 2
    return MplPath(v, np.concatenate(out_c))


# ======================================================================================= drawing
class PresentationRenderer:
    def __init__(self, plan: Plan, style: PresentationStyle = PresentationStyle()):
        self.plan, self.s = plan, style
        self.stats = PresentationStats()
        self.solid, self.gaps, self.polys, self.interiors = build_presentation_walls(plan, style, self.stats)
        self.tints = room_tints(plan, self.polys, style)
        self.label_fp = _first_font(style.label_fonts)
        self.footer_fp = _first_font(style.footer_fonts)
        self.footer_bold = _first_font(style.footer_fonts, "bold")
        self.taken: list[Polygon] = []                   # placed labels (view coordinates)
        self.swings: list[Polygon] = []                  # door swing areas (view coordinates)

    # ------------------------------------------------------------------------------------------- page
    def _setup(self):
        geoms = list(iter_polygons(self.solid)) + list(self.polys.values())
        u0, v0, u1, v1 = unary_union(geoms).bounds if geoms else (-1.0, -1.0, 1.0, 1.0)
        w_m, h_m = max(u1 - u0, 0.5), max(v1 - v0, 0.5)
        W, H = self.s.page_landscape_in
        if h_m > 1.15 * w_m:
            W, H = H, W                                  # portrait page for a tall plan
        avail_w, avail_h = W - 2 * self.s.margin_in, H - self.s.head_in - self.s.foot_in
        self.scale = min(avail_w / w_m, avail_h / h_m, self.s.max_scale)          # inches per metre
        self.m_per_pt = 1.0 / (72.0 * self.scale)
        fig = plt.figure(figsize=(W, H), dpi=self.s.dpi)
        fig.patch.set_facecolor("white")
        bottom = self.s.foot_in / H
        ax = fig.add_axes((0, bottom, 1, avail_h / H))
        cx, cy = (u0 + u1) / 2, -(v0 + v1) / 2
        half_w, half_h = W / 2 / self.scale, avail_h / 2 / self.scale
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)
        ax.set_aspect("equal")
        ax.axis("off")
        self.fig, self.ax, self.W, self.H = fig, ax, W, H
        cap_pt = self.s.name_cap_m * self.scale * 72.0
        self.font_pt = float(np.clip(cap_pt / 0.72, self.s.min_font_pt * 1.6, self.s.max_font_pt))

    # ------------------------------------------------------------------------------------------- primitives
    def _poly_patch(self, poly: Polygon, **kw):
        verts, codes = [], []
        for ring in [poly.exterior, *poly.interiors]:
            xy = to_view(np.asarray(ring.coords))
            verts += list(xy)
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY]
        self.ax.add_patch(PathPatch(MplPath(verts, codes), **kw))

    def _line(self, pts, lw: Optional[float] = None, **kw):
        xy = to_view(np.asarray(pts, float))
        self.ax.plot(xy[:, 0], xy[:, 1], color=kw.pop("color", self.s.line_color), lw=lw or self.s.line_pt,
                     solid_capstyle="butt", **kw)

    # ------------------------------------------------------------------------------------------- layers
    def draw_rooms(self):
        for rid, poly in self.polys.items():
            self._poly_patch(poly, facecolor=self.tints[rid], edgecolor="none", zorder=1)
        self.stats.rooms = len(self.polys)

    def draw_walls(self):
        for poly in iter_polygons(self.solid):
            self._poly_patch(poly, facecolor=self.s.wall_color, edgecolor=self.s.wall_color, lw=0.25, zorder=3)

    def draw_openings(self):
        for g in self.gaps:
            for poly in iter_polygons(g.gap):
                self._poly_patch(poly, facecolor=self.s.window_fill if g.pl.opening.kind == "window" else g.tint,
                                 edgecolor="none", zorder=2)
            kind = g.pl.opening.kind
            if kind == "door":
                self._door(g)
                self.stats.doors += 1
            elif kind == "window":
                self._window(g)
                self.stats.windows += 1
            else:
                self.stats.passages += 1                 # a plain gap: the cut wall ends say it all

    def _door(self, g: OpeningGap):
        pl, s = g.pl, self.s
        lw = s.line_pt
        a, b, n = pl.a, pl.b, pl.into
        self._line(g.frame(), lw=0.6 * lw, zorder=5)    # the door's place in the wall, outlined like a threshold
        u = _unit(b - a)
        if pl.width > DOUBLE_DOOR_M:                     # a wide door is drawn as a pair of leaves
            self._leaf(a, u, n, pl.width / 2)
            self._leaf(b, -u, n, pl.width / 2)
        else:
            self._leaf(a, u, n, pl.width)

    def _leaf(self, hinge: np.ndarray, u: np.ndarray, n: np.ndarray, width: float):
        """One door leaf hinged at `hinge`, closed along u, drawn open by door_open_deg towards n, with its arc."""
        s = self.s
        th = math.radians(s.door_open_deg)
        tip = hinge + width * (math.cos(th) * u + math.sin(th) * n)
        self._line([hinge, tip], lw=s.line_pt, zorder=6)
        c = to_view(hinge)
        v_closed, v_open = to_view(hinge + u * width) - c, to_view(tip) - c
        a0 = math.degrees(math.atan2(v_closed[1], v_closed[0]))
        a1 = math.degrees(math.atan2(v_open[1], v_open[0]))
        if (a1 - a0) % 360 > 180:                        # draw the short way round
            a0, a1 = a1, a0
        self.ax.add_patch(Arc(c, 2 * width, 2 * width, theta1=a0, theta2=a1, color=s.line_color,
                              lw=0.75 * s.line_pt, zorder=6))
        k = max(8, int(s.door_open_deg / 8))
        fan = [hinge] + [hinge + width * (math.cos(t) * u + math.sin(t) * n) for t in np.linspace(0, th, k)]
        self.swings.append(Polygon(to_view(np.asarray(fan))).buffer(0.02))

    def _window(self, g: OpeningGap):
        pl, lw = g.pl, self.s.line_pt
        a, b, n = pl.a, pl.b, pl.into
        self._line(g.frame(), lw=0.6 * lw, zorder=5)
        for f in (0.42, 0.58):                           # the double glazing line, mid-wall
            t = g.s_in - (g.s_in + g.s_out) * f
            self._line([a + n * t, b + n * t], lw=0.6 * lw, zorder=5)

    # ------------------------------------------------------------------------------------------- labels
    def _label_path(self, lines: list[str], size: float) -> MplPath:
        paths = [text_outline(lines[0], self.label_fp, size, self.s.tracking_em)]
        paths += [text_outline(t, self.label_fp, 0.86 * size, 0.03) for t in lines[1:]]
        return _centered(paths, [0.55 * size] * (len(paths) - 1))

    def _anchors(self, vpoly: Polygon, n: int = 14) -> list[Point]:
        best = polylabel(vpoly, tolerance=0.02)
        x0, y0, x1, y1 = vpoly.bounds
        step = max(x1 - x0, y1 - y0) / 10
        grid = [Point(x, y) for x in np.arange(x0 + step / 2, x1, step) for y in np.arange(y0 + step / 2, y1, step)]
        grid = sorted((p for p in grid if vpoly.contains(p)), key=lambda p: -vpoly.exterior.distance(p))
        return [best] + grid[:n]

    def _try_label(self, vpoly: Polygon, inner: Polygon, lines: list[str], sizes, rotations, anchors,
                   avoid_swings: bool):
        """First fit over (size, rotation), preferring a horizontal label down to 72% size before turning it."""
        order = [(z, r) for part in (sizes[:3], sizes[3:]) for r in rotations for z in part]
        paths = {}
        for size, rot in order:
            if size not in paths:
                paths[size] = self._label_path(lines, size)
            flat = paths[size].transformed(Affine2D().scale(self.m_per_pt))
            ext, pad = flat.get_extents(), 0.25 * size * self.m_per_pt
            pth = flat.transformed(Affine2D().rotate_deg(rot))
            box0 = affinity.rotate(Polygon([(ext.x0 - pad, ext.y0 - pad), (ext.x1 + pad, ext.y0 - pad),
                                            (ext.x1 + pad, ext.y1 + pad), (ext.x0 - pad, ext.y1 + pad)]),
                                   rot, origin=(0, 0))      # the label's own (turned) box, not its upright bounds
            for p in anchors:
                x, y = p.x, p.y
                box = affinity.translate(box0, x, y)
                if not inner.contains(box) or any(box.intersects(t) for t in self.taken):
                    continue
                if avoid_swings and any(box.intersects(sw) for sw in self.swings):
                    continue
                return pth.transformed(Affine2D().translate(x, y)), box, size, rot
        return None

    def draw_room_labels(self):
        for room in self.plan.rooms:
            poly = self.polys.get(room.id)
            if poly is None:
                continue
            vpoly = Polygon(to_view(np.asarray(poly.exterior.coords)))
            name = room_display_name(room)
            W, L = room_size_wl(room_polygon(room) or poly)
            dims = f"{W:.2f} x {L:.2f} m"
            inner = vpoly.buffer(-0.06)
            if inner.is_empty:
                inner = vpoly
            anchors = self._anchors(vpoly)
            f = self.font_pt
            sizes = [s for s in (f, 0.85 * f, 0.72 * f, 0.6 * f, 0.5 * f) if s >= self.s.min_font_pt] or [
                self.s.min_font_pt]
            rots = _label_rotations(poly, W, L)
            hit, full, lines = None, False, [name, dims]
            for lines, avoid in (([name, dims], True), ([name], True), ([name, dims], False), ([name], False)):
                hit = self._try_label(vpoly, inner, lines, sizes, rots, anchors, avoid)
                if hit:
                    full = len(lines) == 2 and hit[2] == sizes[0] and hit[3] == 0
                    break
            if hit is None:                              # nothing fits: smallest size at the best spot, over lines
                lines = [name, dims]
                pth = self._label_path(lines, sizes[-1]).transformed(
                    Affine2D().scale(self.m_per_pt).translate(anchors[0].x, anchors[0].y))
                hit = (pth, None, sizes[-1], 0)
                self.stats.labels_forced += 1
            elif full:
                self.stats.labels_full += 1
            else:
                self.stats.labels_reduced += 1
            if hit[1] is not None:
                self.taken.append(hit[1])
            self.ax.add_patch(PathPatch(hit[0], facecolor=self.s.text_color, edgecolor="none", zorder=8))
            self.stats.texts += lines

    # ------------------------------------------------------------------------------------------- footer
    def _fig_text(self, x_in: float, y_in: float, s: str, fp, size: float, color: str, ha: str = "center",
                  tracking: float = 0.0):
        """Outline text placed in page inches (x centred or left, y baseline)."""
        p = text_outline(s, fp, size, tracking)
        ext = p.get_extents()
        dx = -(ext.x0 + ext.x1) / 2 if ha == "center" else -ext.x0
        tr = Affine2D().translate(dx, 0).scale(1 / 72.0).translate(x_in, y_in).scale(1 / self.W, 1 / self.H)
        self.fig.add_artist(PathPatch(p.transformed(tr), facecolor=color, edgecolor="none",
                                      transform=self.fig.transFigure))
        self.stats.texts.append(s)

    def draw_footer(self):
        p, s = self.plan, self.s
        total = p.footprint_area.value if (p.footprint_area is not None and p.footprint_area.value) else None
        if total is None and self.polys:
            total = sum(q.area for q in self.polys.values())
        y0 = s.foot_in
        if total is not None:
            num = f"{total:.0f}" if total >= 100 else f"{total:.2f}"     # the same digits as plan.json/plan.png
            self._fig_text(self.W / 2, y0 - 0.55, f"Total: {num} m²", self.footer_bold, 13, s.footer_color)
        n = len(self.polys)
        sub = f"{n} room{'s' if n != 1 else ''}  ·  {p.tier} capture  ·  {p.capture_id}"
        if p.meta.get("synthetic"):
            sub += "  ·  synthetic test fixture"
        self._fig_text(self.W / 2, y0 - 0.80, sub, self.footer_fp, 9.5, "#3a3a3a")
        fine = ("FLOOR PLAN DRAWN FROM A PHONE CAPTURE. ROOM SIZES ARE INTERIOR, WALL FACE TO WALL FACE. "
                "INTERVALS AND WALL-BY-WALL DIMENSIONS: SEE THE TECHNICAL DRAWING (PLAN.PNG).")
        self._fig_text(self.W / 2, y0 - 1.15, fine, self.footer_fp, 5.6, "#555555")
        self._scale_bar()

    def _scale_bar(self):
        span_m = (self.W - 2 * self.s.margin_in) / self.scale
        L = next((x for x in (1, 2, 5, 10, 20) if x >= 0.12 * span_m), 20)
        n = {1: 2, 2: 2, 5: 5, 10: 5, 20: 4}[L]
        x0, y0, h = self.s.margin_in, self.s.foot_in - 0.62, 0.055
        seg = L / n * self.scale
        for i in range(n):
            self.fig.add_artist(plt.Rectangle(((x0 + i * seg) / self.W, y0 / self.H), seg / self.W, h / self.H,
                                              transform=self.fig.transFigure, lw=0.6, edgecolor=self.s.wall_color,
                                              facecolor=self.s.wall_color if i % 2 == 0 else "white"))
        for i in (0, n):
            lab = f"{L * i / n:g}" + (" m" if i == n else "")
            self._fig_text(x0 + i * seg, y0 - 0.17, lab, self.footer_fp, 7, "#555555")

    # ------------------------------------------------------------------------------------------- entry
    def render(self):
        self._setup()
        self.draw_rooms()
        self.draw_walls()
        self.draw_openings()
        self.draw_room_labels()
        self.draw_footer()
        return self.fig


def rotated_plan(plan: Plan, deg: float) -> Plan:
    """A copy of the plan turned by deg counter-clockwise as seen from above. The plan's heading is wherever the
    phone started, so turning it is free; it only helps compare with a sketch drawn the other way round. Seen from
    above v points down the page, so a counter-clockwise turn on the page is a clockwise one in (u, v)."""
    if not deg % 360:
        return plan
    t = math.radians(deg)
    c, s_ = math.cos(t), math.sin(t)

    def r(p):
        return (p[0] * c + p[1] * s_, -p[0] * s_ + p[1] * c)

    out = copy.deepcopy(plan)
    for room in out.rooms:
        room.polygon = [r(p) for p in room.polygon]
    for w in out.walls:
        w.p0, w.p1, w.normal = r(w.p0), r(w.p1), r(w.normal)
    for o in out.openings:
        o.center = r(o.center)
    return out


def render_presentation(plan: Plan, out_stem: str | Path,
                        style: PresentationStyle = PresentationStyle()) -> PresentationStats:
    """Render `plan` to <out_stem>.svg and <out_stem>.png. Deterministic: fixed SVG id salt, no timestamps."""
    plan = rotated_plan(plan, style.rotate_deg)
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.fonttype": "path", "svg.hashsalt": "floorplan-presentation"}):
        r = PresentationRenderer(plan, style)
        fig = r.render()
        fig.savefig(out_stem.with_suffix(".svg"), metadata={"Date": None}, facecolor="white")
        fig.savefig(out_stem.with_suffix(".png"), dpi=style.dpi, metadata={"Software": None}, facecolor="white")
        plt.close(fig)
    return r.stats


__all__ = ["render_presentation", "PresentationStyle", "PresentationStats", "room_display_name", "room_size_wl",
           "build_presentation_walls", "rotated_plan", "CLOSE_GAP_M"]
