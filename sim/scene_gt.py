#!/usr/bin/env python
"""Ground truth of a simulated house, measured on the visible surfaces the way a tape would (decision D-036).

Why not just use rooms.json? InteriorAgent's rooms.json gives the designer's room outlines. The surfaces a camera
(or a tape) sees can sit a few cm away: wall panels, a TV wall cladding (3.4 cm in kujiale_0065's living room),
tray ceilings. A cm-level benchmark needs the visible surface, so every number here comes from casting rays against
the scene's own triangles (scene_geom.py):

  walls     each rooms.json edge is moved to the median wall surface found by horizontal rays shot outward from
            0.25 m inside the room, at 0.8-1.7 m height, every 5 cm (rays that pass through a doorway are ignored).
            Corners are the intersections of the moved edges; wall length = corner to corner (what a tape along
            the wall reads). The spread of the surface along each wall is kept in the notes.
  ceiling   upward rays on a 10 cm grid; the GT value is the median near the room's most open point ("near the
            middle of the room", as in HOUSE_CAPTURE_GUIDE step 8); distinct ceiling levels are listed in notes.
  doors     clear width between the frames at ~1 m height, by rays along the wall through the opening (walls + the
            visible door frame meshes); height = floor to the underside of the head.
  windows   the hole in the wall (walls only, glass ignored): width, height and sill height.

Outputs (in <out>/):
  ground_truth.csv  the same format as the tape GT (floorplan/benchmark/gt_eval.py): room_id,item_type,item_id,
                    value_m,notes. Walls W1..Wn clockwise seen from above, W1 = the wall with the room's main door.
  sim_gt.json       everything above with coordinates (USD world, Z up), plus A4 sheet spots and the room graph.
  walkable.npz      2 cm grid: where a person holding the phone can stand (for driving and auto paths).
  gt_topview.png    picture of it all, for checking by eye.

Usage: python sim/scene_gt.py <scene.usda> [--out <dir>]   (default <scene dir>/.simcache/gt)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import scene_geom as sg  # noqa: E402

ROOM_SLUG = {"living room": "living_room", "dining room": "dining_room", "study room": "study"}
WALL_RAY_START = 0.25          # m inside the room
WALL_OFFSET_WINDOW = 0.12      # accept surfaces within +-12 cm of the rooms.json edge
GRID = 0.02                    # walkable grid (m)
JOG_MIN_M = 0.10               # wall steps / outline edges shorter than this are merged: a tape ignores them


# ----------------------------------------------------------------------------------------------- polygon helpers
def signed_area(p):
    p = np.asarray(p)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def merge_collinear(poly, tol_m=0.005):
    p = [np.asarray(q, float) for q in poly]
    changed = True
    while changed and len(p) > 3:
        changed = False
        for i in range(len(p)):
            a, b, c = p[i - 1], p[i], p[(i + 1) % len(p)]
            d = c - a
            if np.linalg.norm(d) < 1e-9:
                continue
            dist = abs(d[0] * (b - a)[1] - d[1] * (b - a)[0]) / np.linalg.norm(d)
            if dist < tol_m and np.dot(b - a, c - b) > 0:
                p.pop(i)
                changed = True
                break
    return np.array(p)


def line_intersect(p0, d0, p1, d1):
    A = np.array([d0, -d1]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s = np.linalg.solve(A, p1 - p0)
    return p0 + s[0] * d0


def point_in_poly(pts, poly):
    from matplotlib.path import Path as MPath
    return MPath(np.asarray(poly)).contains_points(np.atleast_2d(pts))


# ----------------------------------------------------------------------------------------------------- walls
def drop_tiny_edges(poly, min_len=JOG_MIN_M):
    """Replace each edge shorter than min_len by its midpoint (a 2 cm step in an outline is not a wall)."""
    P = [np.asarray(q, float) for q in poly]
    changed = True
    while changed and len(P) > 3:
        changed = False
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            if np.linalg.norm(b - a) < min_len:
                P[i] = (a + b) / 2
                P.pop((i + 1) % len(P))
                changed = True
                break
    return np.array(P)


def refine_room(poly, wall_scene, log):
    """Move each edge of a CCW polygon to the visible wall surface. Returns (polygon, per-edge info)."""
    P = merge_collinear(drop_tiny_edges(merge_collinear(poly)), tol_m=0.02)
    if signed_area(P) < 0:
        P = P[::-1]
    n = len(P)
    edges = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        L = float(np.linalg.norm(b - a))
        d = (b - a) / L
        nrm = np.array([d[1], -d[0]])                      # outward for a CCW polygon
        s = np.arange(0.10, L - 0.10 + 1e-9, 0.05) if L > 0.3 else np.array([L / 2])
        zs = np.array([0.8, 1.1, 1.4, 1.7])
        S, Z = np.meshgrid(s, zs)
        xy = a[None] + S.reshape(-1, 1) * d[None] - WALL_RAY_START * nrm[None]
        org = np.concatenate([xy, Z.reshape(-1, 1)], 1)
        dirs = np.tile(np.r_[nrm, 0.0], (len(org), 1))
        t, _, _ = sg.cast(wall_scene, org, dirs)
        off = t - WALL_RAY_START
        ok = np.isfinite(off) & (np.abs(off) < WALL_OFFSET_WINDOW)
        cover = float(ok.mean()) if len(ok) else 0.0
        if ok.sum() < 3:                                   # the whole edge is an opening: use the wall above/below it
            S, Z = np.meshgrid(s, np.array([0.05, 0.12, 2.35, 2.45]))
            xy = a[None] + S.reshape(-1, 1) * d[None] - WALL_RAY_START * nrm[None]
            org = np.concatenate([xy, Z.reshape(-1, 1)], 1)
            t, _, _ = sg.cast(wall_scene, org, np.tile(np.r_[nrm, 0.0], (len(org), 1)))
            off = t - WALL_RAY_START
            ok = np.isfinite(off) & (np.abs(off) < WALL_OFFSET_WINDOW)
        if ok.sum() >= 3:
            o = off[ok]
            edges.append(dict(offset=float(np.median(o)), spread=float(np.percentile(o, 95) - np.percentile(o, 5)),
                              coverage=cover, virtual=False))
        else:
            edges.append(dict(offset=0.0, spread=0.0, coverage=cover, virtual=True))
        edges[-1].update(a=a.tolist(), b=b.tolist(), dir=d.tolist(), normal=nrm.tolist(), length_json=L)
    # shifted lines -> new corners. Parallel neighbours (a step in the outline) give a short "jog" wall between them;
    # steps under 5 cm are merged (a tape ignores them), larger ones become real walls with their own record.
    lines = [(np.asarray(e["a"]) + e["offset"] * np.asarray(e["normal"]), np.asarray(e["dir"])) for e in edges]
    Q, E = [], []
    for i in range(n):
        q = line_intersect(*lines[i - 1], *lines[i])
        if q is None:
            a_i = np.asarray(edges[i]["a"])
            q1 = lines[i - 1][0] + np.dot(a_i - lines[i - 1][0], lines[i - 1][1]) * lines[i - 1][1]
            q2 = lines[i][0] + np.dot(a_i - lines[i][0], lines[i][1]) * lines[i][1]
            if np.linalg.norm(q2 - q1) < JOG_MIN_M:
                Q.append((q1 + q2) / 2); E.append(edges[i])
            else:
                dj = (q2 - q1) / np.linalg.norm(q2 - q1)
                Q.append(q1)
                E.append(dict(offset=0.0, spread=0.0, coverage=1.0, virtual=False, a=q1.tolist(), b=q2.tolist(),
                              dir=dj.tolist(), normal=[dj[1], -dj[0]], length_json=float(np.linalg.norm(q2 - q1)),
                              jog=True))
                Q.append(q2); E.append(edges[i])
        else:
            Q.append(q); E.append(edges[i])
    return np.array(Q), E


# ------------------------------------------------------------------------------------------------- openings
def opening_measure(box, cls, frame_scene, wall_scene, log):
    """Clear width / height (doors) or hole width / height / sill (windows) of one door or window object."""
    mn, mx = np.asarray(box["min"]), np.asarray(box["max"])
    ext = mx - mn
    ax = 0 if ext[0] >= ext[1] else 1                       # long horizontal axis = along the wall
    nax = 1 - ax
    c = (mn + mx) / 2
    along = np.zeros(3); along[ax] = 1
    res = dict(cls=cls, center=c.tolist(), axis="x" if ax == 0 else "y", box_min=mn.tolist(), box_max=mx.tolist())
    if cls == "door":
        zc = [0.9, 1.0, 1.1]
        scene = frame_scene
    else:
        zc = [c[2] - 0.1, c[2], c[2] + 0.1]
        scene = wall_scene
    widths = []
    for dn in np.linspace(-0.4, 0.4, 9) * ext[nax]:
        for z in zc:
            o = c.copy(); o[nax] += dn; o[2] = z
            t, _, _ = sg.cast(scene, np.array([o, o]), np.array([along, -along]))
            if np.all(np.isfinite(t)) and np.all(t > 0.05) and t.sum() < 4.0:
                widths.append(float(t.sum()))
    res["width"] = float(np.median(widths)) if widths else None
    res["width_spread"] = float(np.ptp(widths)) if len(widths) > 1 else None
    if cls == "door" and widths:
        # Across the wall thickness the width changes: wall hole, door frame (jamb lining), sliding-door track, and a
        # closed leaf or its lock. The tape GT is "between the two side frames", i.e. the narrowest width that a
        # large share of rays agree on: cluster the widths (8 mm), keep clusters with >= 20% of the rays, take the
        # narrowest. Rays that start inside a closed leaf or hit the lock form small clusters and are ignored.
        w = np.sort(np.asarray(widths))
        clusters, cur = [], [w[0]]
        for v in w[1:]:
            if v - cur[-1] <= 0.008:
                cur.append(v)
            else:
                clusters.append(cur); cur = [v]
        clusters.append(cur)
        big = [c for c in clusters if len(c) >= 0.2 * len(w)] or [max(clusters, key=len)]
        res["width"] = float(np.median(big[0]))
        res["width_candidates"] = [(round(float(np.median(c)), 4), len(c)) for c in clusters]
        tw, _, _ = sg.cast(wall_scene, np.array([c, c]) + [0, 0, 1.0 - c[2]], np.array([along, -along]))
        res["hole_width"] = float(tw.sum()) if np.all(np.isfinite(tw)) else None
        o = c.copy(); o[2] = 0.05
        t, _, _ = sg.cast(frame_scene, o[None], np.array([[0, 0, 1.0]]))
        res["height"] = float(o[2] + t[0]) if np.isfinite(t[0]) else None
        res["sill"] = 0.0
    else:
        o = c.copy()
        tu, _, _ = sg.cast(wall_scene, o[None], np.array([[0, 0, 1.0]]))
        td, _, _ = sg.cast(wall_scene, o[None], np.array([[0, 0, -1.0]]))
        res["sill"] = float(o[2] - td[0]) if np.isfinite(td[0]) else None
        res["height"] = float(tu[0] + td[0]) if np.isfinite(tu[0]) and np.isfinite(td[0]) else None
    return res


def attach_openings(openings, rooms):
    """Find which room walls each opening sits on (within 0.35 m of the wall line, overlapping along it)."""
    for op in openings:
        c = np.asarray(op["center"][:2])
        ax = 0 if op["axis"] == "x" else 1
        half = (op["box_max"][ax] - op["box_min"][ax]) / 2
        hits = []
        for r in rooms:
            for k, e in enumerate(r["edges"]):
                d = np.asarray(e["dir"])
                if abs(d[ax]) < 0.95:
                    continue
                a = np.asarray(r["polygon"][k]); b = np.asarray(r["polygon"][(k + 1) % len(r["polygon"])])
                dist = abs(np.dot(c - a, np.asarray(e["normal"])))
                s = np.dot(c - a, d); L = np.linalg.norm(b - a)
                if dist < 0.35 and -half < s < L + half:
                    hits.append((r["id"], k, float(dist)))
        best = {}
        for rid, k, dist in hits:
            if rid not in best or dist < best[rid][1]:
                best[rid] = (k, dist)
        op["walls"] = {rid: k for rid, (k, _) in best.items()}
        op["room_ids"] = sorted(best)


# --------------------------------------------------------------------------------------------------- ceiling
def ceiling_levels(poly, ceil_scene, floor_scene):
    P = np.asarray(poly)
    mn, mx = P.min(0), P.max(0)
    xs, ys = np.meshgrid(np.arange(mn[0] + 0.05, mx[0], 0.10), np.arange(mn[1] + 0.05, mx[1], 0.10))
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    pts = pts[point_in_poly(pts, P)]
    # distance to the boundary (to find the most open point and drop the perimeter strip)
    from matplotlib.path import Path as MPath
    def dist_to_edges(q):
        dm = np.full(len(q), np.inf)
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            ab = b - a; t = np.clip(((q - a) @ ab) / (ab @ ab), 0, 1)
            dm = np.minimum(dm, np.linalg.norm(q - (a + t[:, None] * ab), axis=1))
        return dm
    dist = dist_to_edges(pts)
    inner = pts[dist > 0.3]
    o = np.c_[inner, np.full(len(inner), 0.05)]
    tu, _, _ = sg.cast(ceil_scene, o, np.tile([0, 0, 1.0], (len(o), 1)))
    of = np.c_[inner, np.full(len(inner), 1.0)]
    tf, _, _ = sg.cast(floor_scene, of, np.tile([0, 0, -1.0], (len(of), 1)))
    floor_z = float(np.median(1.0 - tf[np.isfinite(tf)])) if np.isfinite(tf).any() else 0.0
    h = 0.05 + tu - floor_z
    ok = np.isfinite(h)
    center = pts[np.argmax(dist)]
    near = ok & (np.linalg.norm(inner - center, axis=1) < 0.6)
    hist_levels = []
    if ok.any():
        r = np.round(h[ok], 2)
        vals, cnt = np.unique(r, return_counts=True)
        for v, c in sorted(zip(vals, cnt), key=lambda x: -x[1]):
            if c / ok.sum() >= 0.08:
                hist_levels.append(dict(height=float(v), area_frac=round(float(c / ok.sum()), 3)))
    return dict(floor_z=floor_z, center_xy=center.tolist(),
                at_center=float(np.median(h[near])) if near.any() else (float(np.median(h[ok])) if ok.any() else None),
                median=float(np.median(h[ok])) if ok.any() else None, levels=hist_levels)


# -------------------------------------------------------------------------------------------------- walkable
def walkable_grid(geom, rooms, sill_boxes, margin_wall=0.12, body=0.25, omap=None):
    import cv2
    allp = np.concatenate([np.asarray(r["polygon"]) for r in rooms])
    x0, y0 = allp.min(0) - 0.5
    x1, y1 = allp.max(0) + 0.5
    W, H = int(np.ceil((x1 - x0) / GRID)), int(np.ceil((y1 - y0) / GRID))
    to_px = lambda p: np.round((np.asarray(p) - [x0, y0]) / GRID).astype(np.int32)
    inside = np.zeros((H, W), np.uint8)
    for r in rooms:
        cv2.fillPoly(inside, [to_px(r["polygon"])], 1)
    k = int(round(margin_wall / GRID)) * 2 + 1
    inside = cv2.erode(inside, np.ones((k, k), np.uint8))
    for sb in sill_boxes:                                  # door passages: the sill rectangle, pushed into both rooms
        mn, mx = np.asarray(sb["min"][:2]), np.asarray(sb["max"][:2])
        ext = mx - mn
        nax = 0 if ext[0] < ext[1] else 1
        mn = mn.copy(); mx = mx.copy()
        mn[nax] -= 0.3; mx[nax] += 0.3
        ax = 1 - nax
        mn[ax] += 0.08; mx[ax] -= 0.08
        a, b = to_px(mn), to_px(mx)
        inside[a[1]:b[1] + 1, a[0]:b[0] + 1] = 1
    soft = inside.copy()                                    # walls only (teleop: furniture is a warning)
    # Obstacles at body and phone height (0.10-1.90 m), from the real triangles of EVERY non-structural object
    # (furniture, curtains, range hoods, pendant lamps: the v1 map skipped curtains and the camera walked through
    # them, D-051), united with Isaac Sim's own occupancy map (PhysX colliders, sim/isaac_omap.py) when it exists.
    skip = {"wall", "floor", "ceiling", "window", "doorsill"}     # doors ARE obstacles (stacked sliding panels, open
    cls_ok = np.array([m["cls"] not in skip for m in geom["meshes"]])   # leaves); the passages are re-opened below
    F = geom["F"][cls_ok[geom["tri_mesh"]]]
    Z = geom["V"][F][:, :, 2]
    F = F[(Z.max(1) > 0.10) & (Z.min(1) < 1.90)]
    tri = ((geom["V"][F][:, :, :2] - [x0, y0]) / GRID).round().astype(np.int32)
    furn = np.zeros_like(inside)
    # one triangle at a time: a single fillPoly over all triangles uses the even-odd rule, so the top and bottom faces
    # of a box (which overlap exactly seen from above) cancel and a fridge becomes a hole (D-051 bug, found in k22)
    for t3 in tri:
        cv2.fillConvexPoly(furn, t3, 1)
    if omap is not None:                                     # Isaac occupancy (x stored reversed: flip columns)
        occ = omap["occupied"][:, ::-1]
        mn, cell = omap["min_bound"], float(omap["cell"])
        jj, ii = np.meshgrid(np.arange(W), np.arange(H))
        oi = np.floor((y0 + ii * GRID - mn[1]) / cell).astype(int)
        oj = np.floor((x0 + jj * GRID - mn[0]) / cell).astype(int)
        ok = (oi >= 0) & (oi < occ.shape[0]) & (oj >= 0) & (oj < occ.shape[1])
        o2 = np.zeros_like(furn)
        o2[ok] = occ[oi[ok], oj[ok]]
        furn |= o2
    passage = np.zeros_like(inside)
    for sb in sill_boxes:                                    # door passages stay open (door frames are colliders)
        mn_, mx_ = np.asarray(sb["min"][:2]), np.asarray(sb["max"][:2])
        a, b = to_px(mn_ - 0.05), to_px(mx_ + 0.05)
        passage[max(a[1], 0):b[1] + 1, max(a[0], 0):b[0] + 1] = 1
    kb = int(round(body / GRID)) * 2 + 1
    furn = cv2.dilate(furn, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))) & (1 - passage)
    strict = inside & (1 - furn)
    return dict(origin=np.array([x0, y0]), res=GRID, soft=soft.astype(bool), strict=strict.astype(bool),
                furniture=furn.astype(bool))


# ------------------------------------------------------------------------------------------------- the build
def build(usd: Path, out: Path, log=print) -> dict:
    geom = sg.load(usd, log=log)
    rj = sg.rooms_json(usd)
    boxes = sg.object_boxes(geom)
    wall_scene, _ = sg.raycaster(geom, sg.mesh_mask(geom, lambda m: m["cls"] == "wall"))
    frame_scene, _ = sg.raycaster(geom, sg.mesh_mask(geom, lambda m: m["cls"] in ("wall", "door")))
    ceil_scene, _ = sg.raycaster(geom, sg.mesh_mask(geom, lambda m: m["cls"] == "ceiling"))
    floor_scene, _ = sg.raycaster(geom, sg.mesh_mask(geom, lambda m: m["cls"] == "floor"))

    # rooms, numbered by size (largest first) so ids are stable: 01_living_room, 02_bedroom, ...
    order = sorted(range(len(rj)), key=lambda i: -abs(signed_area(rj[i]["polygon"])))
    rooms, counts = [], {}
    for k, i in enumerate(order):
        t = rj[i]["room_type"].strip().lower()
        slug = ROOM_SLUG.get(t, t.replace(" ", "_"))
        counts[slug] = counts.get(slug, 0) + 1
        rid = f"{k + 1:02d}_{slug}" + (f"{counts[slug]}" if counts[slug] > 1 else "")
        poly, edges = refine_room(rj[i]["polygon"], wall_scene, log)
        rooms.append(dict(id=rid, type=t, polygon_json=rj[i]["polygon"], polygon=poly.tolist(), edges=edges,
                          area=abs(signed_area(poly)), area_json=abs(signed_area(rj[i]["polygon"]))))
    # openings
    openings = []
    for g, b in sorted(boxes.items()):
        if b["cls"] in ("door", "window"):
            m = opening_measure(b, b["cls"], frame_scene, wall_scene, log)
            m["id"] = g
            openings.append(m)
    attach_openings(openings, rooms)
    # a door that touches no room wall (e.g. a shower screen inside a room) is not an opening
    openings = [o for o in openings if o["room_ids"]]
    # ceilings
    for r in rooms:
        r["ceiling"] = ceiling_levels(r["polygon"], ceil_scene, floor_scene)
    # main door per room: breadth-first from the entrance (a door with only one room = exterior door)
    doors = [o for o in openings if o["cls"] == "door"]
    ext = [o for o in doors if len(o["room_ids"]) == 1]
    main = {}
    if ext:
        entrance = max(ext, key=lambda o: o["width"] or 0)
        main[entrance["room_ids"][0]] = entrance["id"]
        frontier = [entrance["room_ids"][0]]
        while frontier:
            nxt = []
            for rid in frontier:
                for o in doors:
                    if rid in o["room_ids"]:
                        for other in o["room_ids"]:
                            if other not in main:
                                main[other] = o["id"]
                                nxt.append(other)
            frontier = nxt
    for r in rooms:
        r["main_door"] = main.get(r["id"])
    # walls clockwise seen from above, W1 = wall holding the main door
    rows = []
    for r in rooms:
        P = np.asarray(r["polygon"]); n = len(P)
        lengths = [float(np.linalg.norm(P[(k + 1) % n] - P[k])) for k in range(n)]
        md = next((o for o in openings if o["id"] == r["main_door"]), None)
        start = md["walls"][r["id"]] if md and r["id"] in md["walls"] else int(np.argmax(lengths))
        cw = [(start - j) % n for j in range(n)]            # CCW polygon -> walk it backwards = clockwise
        # the CCW edge k runs P[k]->P[k+1]; walking clockwise visits edges start, start-1, ...
        r["walls_cw"] = []
        for j, k in enumerate(cw):
            e = r["edges"][k]
            ops = [o["id"] for o in openings if o["walls"].get(r["id"]) == k]
            r["walls_cw"].append(dict(name=f"W{j + 1}", edge=k, length=lengths[k], virtual=e["virtual"],
                                      offset_from_json=e["offset"], surface_spread=e["spread"], openings=ops))
            note = f"edge {k}; surface {e['offset'] * 100:+.1f} cm from rooms.json, spread {e['spread'] * 100:.1f} cm"
            if e["coverage"] < 0.2:
                note += f"; mostly opening (wall surface on {e['coverage'] * 100:.0f}% at 0.8-1.7 m; length = corner to corner)"
            if ops:
                note += "; openings " + " ".join(ops)
            rows.append([r["id"], "wall", f"W{j + 1}", f"{lengths[k]:.4f}", note])
        c = r["ceiling"]
        lv = ", ".join(f"{l['height']:.2f} m ({l['area_frac'] * 100:.0f}%)" for l in c["levels"])
        rows.append([r["id"], "ceiling_height", "C1", f"{c['at_center']:.4f}", f"near the middle; levels: {lv}"])
        rows.append([r["id"], "area", "A1", f"{r['area']:.4f}", f"visible-surface polygon; rooms.json {r['area_json']:.3f} m2"])
    for o in openings:
        kind = "door_width" if o["cls"] == "door" else "window_width"
        for rid in o["room_ids"]:
            if o["width"] is not None:
                rows.append([rid, kind, o["id"], f"{o['width']:.4f}",
                             (f"wall hole {o['hole_width']:.3f} m; " if o.get("hole_width") else "") + (f"height {o['height']:.3f} m" if o['height'] is not None else "height: no head (full-height passage)") + (f", sill {o['sill']:.3f} m" if o["cls"] == "window" and o['sill'] is not None else "")
                             + ("; rooms " + "+".join(o["room_ids"]))])
            if o["cls"] == "window" and o["sill"] is not None:
                rows.append([rid, "window_sill", o["id"], f"{o['sill']:.4f}", "floor to the bottom of the wall opening"])
    # A4 sheet spots: the most open floor point of each room, nudged 0.7 m off the centre (not under furniture)
    # door passages = sills of doors that join two rooms (an exterior door is closed in these scenes)
    inner_doors = [o for o in openings if o["cls"] == "door" and len(o["room_ids"]) == 2]
    def _near(b, o):
        return all(b["min"][i] - 0.2 < o["center"][i] < b["max"][i] + 0.2 for i in (0, 1))
    sills = [b for g, b in boxes.items() if b["cls"] == "doorsill" and any(_near(b, o) for o in inner_doors)]
    omap_f = out / "isaac_omap.npz"
    wk = walkable_grid(geom, rooms, sills, omap=dict(np.load(omap_f)) if omap_f.exists() else None)
    sheets = place_sheets(rooms, wk)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["room_id", "item_type", "item_id", "value_m", "notes"])
        f.write(f"# simulated house {usd.name}: ground truth from scene geometry (sim/scene_gt.py)\n")
        w.writerows(rows)
    gt = dict(scene=str(usd), frame="USD world, Z up, metres", rooms=rooms, openings=openings, sheets=sheets,
              walkable=dict(origin=wk["origin"].tolist(), res=wk["res"], shape=list(wk["soft"].shape)))
    (out / "sim_gt.json").write_text(json.dumps(gt, indent=1))
    np.savez_compressed(out / "walkable.npz", origin=wk["origin"], res=wk["res"], soft=wk["soft"],
                        strict=wk["strict"], furniture=wk["furniture"])
    topview(gt, wk, out / "gt_topview.png")
    log(f"[gt] {len(rooms)} rooms, {sum(len(r['walls_cw']) for r in rooms)} walls, {len(openings)} openings -> {out}")
    return gt


def place_sheets(rooms, wk):
    import cv2
    sheets = []
    for r in rooms:
        H, W = wk["strict"].shape
        m = np.zeros((H, W), np.uint8)
        px = np.round((np.asarray(r["polygon"]) - wk["origin"]) / wk["res"]).astype(np.int32)
        cv2.fillPoly(m, [px], 1)
        free = (m & wk["strict"]).astype(np.uint8)
        if free.sum() < 50:
            free = (m & wk["soft"]).astype(np.uint8)
        dt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        iy, ix = np.unravel_index(np.argmax(dt), dt.shape)
        c = wk["origin"] + np.array([ix, iy]) * wk["res"]
        # the most open spot is where the photographer stands; put the sheet 0.7 m away from it when the floor there
        # is still open (>= 0.3 m from walls and furniture), else on the open spot itself
        best = c
        if dt.max() * wk["res"] > 0.6:
            for ang in np.radians([0, 90, 180, 270, 45, 135, 225, 315]):
                q = c + 0.7 * np.array([np.cos(ang), np.sin(ang)])
                j = np.round((q - wk["origin"]) / wk["res"]).astype(int)
                if 0 <= j[1] < H and 0 <= j[0] < W and dt[j[1], j[0]] * wk["res"] > 0.3:
                    best = q
                    break
        sheets.append(dict(room=r["id"], xy=best.tolist(), yaw_deg=15.0, size=[0.297, 0.210]))
    return sheets


def topview(gt, wk, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 12))
    o, res = wk["origin"], wk["res"]
    H, W = wk["soft"].shape
    img = np.zeros((H, W, 3)) + 1.0
    img[wk["soft"]] = [0.85, 0.93, 0.85]
    img[wk["strict"]] = [0.70, 0.88, 0.70]
    ax.imshow(img, origin="lower", extent=[o[0], o[0] + W * res, o[1], o[1] + H * res])
    for r in gt["rooms"]:
        pj = np.asarray(r["polygon_json"] + [r["polygon_json"][0]])
        ax.plot(pj[:, 0], pj[:, 1], ":", color="gray", lw=1)
        P = np.asarray(r["polygon"] + [r["polygon"][0]])
        ax.plot(P[:, 0], P[:, 1], "-", color="k", lw=1.5)
        for wl in r["walls_cw"]:
            k = wl["edge"]
            a, b = P[k], P[k + 1]
            m = (a + b) / 2 - 0.18 * np.asarray(r["edges"][k]["normal"])
            ax.text(m[0], m[1], f"{wl['name']}\n{wl['length']:.3f}", fontsize=6, ha="center", va="center",
                    color="gray" if wl["virtual"] else "k")
        c = r["ceiling"]
        ax.text(*c["center_xy"], f"{r['id']}\n{r['area']:.2f} m2\nh {c['at_center']:.3f}", fontsize=8, ha="center",
                color="navy", weight="bold")
    for op in gt["openings"]:
        mn, mx = np.asarray(op["box_min"]), np.asarray(op["box_max"])
        ax.add_patch(plt.Rectangle(mn[:2], *(mx - mn)[:2], fill=False, color="tab:red" if op["cls"] == "door" else "tab:blue"))
        ax.text(op["center"][0], op["center"][1], f"{op['id']}\n{op['width']:.3f}" if op["width"] else op["id"],
                fontsize=6, color="tab:red" if op["cls"] == "door" else "tab:blue", ha="center")
    for s in gt["sheets"]:
        ax.plot(*s["xy"], "s", color="orange", ms=6)
    ax.set_aspect("equal"); ax.set_title(Path(gt["scene"]).name + " ground truth (USD world, Z up; view from above)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or a.usd.parent / ".simcache" / "gt"
    build(a.usd.resolve(), out)


if __name__ == "__main__":
    main()
