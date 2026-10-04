"""Scripted capture person v2: the protocol designed ONCE from published practice + the user's review (D-051).

Sources (docs/CAPTURE_PRACTICES_RESEARCH.md): photo spins from one open spot (Matterport, ZInD, magicplan, HorizonNet);
small rooms shot from the doorway (CubiCasa, Hover, ZInD); every corner incl. ceiling corners in view (DocuSketch,
HorizonNet); doorway links from both sides of the threshold (Matterport, Metashape); video: walk forward slowly, never
sideways, 1-3 m from walls, aimed at the baseboards, no panning from the middle of a room (CubiCasa), pause and pan
in corners (Hover); separate sequences for floor and ceiling (ARKitScenes; the provided sample's floor_only /
with_ceiling captures). User review (4 Oct 12:45): no camera inside furniture (Isaac occupancy map + 3D clearance),
small rooms need 2-3 photos (from the entrance, from the far end looking back, one with the ceiling), one ceiling
photo per room, floor pass then ceiling pass, less jitter; some blank walls and close-ups are fine (humans make them).

Photos per room (max 8; v2.1 = D-055, user review 4 Oct 14:20: "photos from each room looking out and in so they can
be patched together", "right-facing as well as left-facing in the kitchen and bathroom", "the balcony from the hall"):
  doorway pair at EVERY door between two rooms (also doors the route does not walk through): standing on the
               threshold, one photo back into the room being left, one into the next room (each room seen IN from
               each of its doors)
  normal room  spin from the open middle, the FIRST photo facing the door you came in through (the room seen OUT
               through its door): N by size (4 under 10 m2, 5 up to 18, 6 above; <= 7 - doors), about a quarter
               overlap (varying), tilted ~12 deg down; plus ONE ceiling photo (tilted ~40 deg up), which closes the
               room's series
  small room   (area < 6.5 m2 or narrower than 2 m): the doorway pair's photo into it is turned LEFT of the room's
               axis; from the far end two photos back at the door (turned left of it, then right; door in both = out);
               ONE ceiling photo; then, leaving, from the doorway one photo turned RIGHT (after the ceiling photo)
  shallow room (< 2.2 m deep from its door: balcony): also one photo of it from ~2 m back in the next room, through
               the door, taken last
Walk (one continuous take; video and LiDAR use the same route):
  pass 1 (floor)   every room: a loop along its walls (inset 0.5-1.3 m), walking forward, phone ~25 deg down,
                   a short pause + look into each corner, 2 s pause in every doorway (until D-067 the A4 sheet was
                   also looked at once; LOOK_AT_SHEETS switches that back on)
  pass 2 (ceiling) the same route again with the phone ~25 deg up
"""
from __future__ import annotations

import math

import numpy as np

from sim.rig import humanize, humanize_photo

FLOOR_PITCH, CEIL_PITCH = -25.0, 25.0
PHOTO_PITCH, CEIL_PHOTO_PITCH = -12.0, 40.0
CORNER_TURN_W = 30.0
WALK_MPS = 0.8            # photo tier: walking pace between photo spots (sets the EXIF capture times)
SHALLOW_M = 2.2           # a room this shallow seen from its door (a balcony) is also photographed from the next room


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def room_is_small(room) -> bool:
    P = np.asarray(room["polygon"])
    ext = P.max(0) - P.min(0)
    return room["area"] < 6.5 or ext.min() < 2.0


def room_loop(room, planner, view) -> list[np.ndarray]:
    """Clockwise loop inside the room, inset from the walls, as walkable waypoints ~0.6 m apart."""
    from shapely.geometry import Polygon
    P = Polygon(room["polygon"])
    ext = np.asarray(room["polygon"]).max(0) - np.asarray(room["polygon"]).min(0)
    inset = float(np.clip(0.35 * ext.min(), 0.5, 1.3))
    for _ in range(4):
        L = P.buffer(-inset)
        if not L.is_empty and L.geom_type == "Polygon" and L.area > 0.3:
            break
        inset *= 0.7
    else:
        return []
    if L.exterior.is_ccw:
        coords = np.asarray(L.exterior.coords)[::-1]
    else:
        coords = np.asarray(L.exterior.coords)
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    ss = np.arange(0, s[-1], 0.6)
    pts = np.stack([np.interp(ss, s, coords[:, 0]), np.interp(ss, s, coords[:, 1])], 1)
    out = []
    for q in pts:
        q = planner.snap(q)
        if not out or np.linalg.norm(q - out[-1]) > 0.2:
            out.append(q)
    return out


def corner_targets(room) -> np.ndarray:
    return np.asarray(room["polygon"])


LOOK_INTO_ROOM = -30.0    # v2.2: on the (clockwise) wall loop the phone looks 30 deg into the room, not straight ahead,
                          # so one loop sweeps every wall from across the room while walking (no turning on the spot,
                          # which monocular tracking cannot handle)
LOOK_AT_SHEETS = False    # D-067: no reference object in the protocol, so the scripted person no longer looks at the
                          # A4 sheet props (they stay in the scenes; walks scripted before 4 Oct 21:10 still look)


def build_walk(pb, gt, planner, rooms_obj, rooms, order, door_geometry, stand, rng, pitch, look_sheets=True,
               tilts=None):
    """One pass over the route (DFS order). In each room: one continuous loop along its walls (walking forward), split at
    the room's real corners; in the floor pass a short look into each corner (Hover) and, only if LOOK_AT_SHEETS, one
    look at the A4 sheet. A 2 s pause in every doorway, facing into the next room."""
    sheets = {s["room"]: s for s in gt["sheets"]}
    corner_looks = look_sheets                     # the floor pass looks into corners; the ceiling pass just walks
    look_sheets = look_sheets and LOOK_AT_SHEETS   # set AFTER corner_looks: the corner glances do not depend on D-067
    for step, rid, info in order:
        if step == "enter" and room_is_small(rooms[rid]):
            # small room (bathroom, small kitchen, balcony): step in, pan slowly across it, step out (no loop: in a
            # 1.7 m wide bathroom a loop squeezes past the basin; CubiCasa/Hover/ZInD treat small rooms from the door)
            # v2.2 (D-064, user 4 Oct 18:50: "for all smaller rooms videos are taken from the door step, never from
            # inside"): walk IN to the far end (clear view back), look across the room both ways and back at the door,
            # then walk out again. The 18 cm body clearance (was 25) leaves room for this in a 1.7 m bathroom.
            room = rooms[rid]
            here = np.asarray(pb.last[1:3])
            far = planner.snap(farthest_from(planner, room, here))
            pb.walk(planner.path(here, far))
            back = math.degrees(math.atan2(here[1] - far[1], here[0] - far[0]))
            tl = tilts or (pitch,)
            for dy in (-70.0, 70.0, 0.0):                       # floor line, across the room both ways
                pb.turn_to(back + dy, pitch=tl[0], rate=20.0)
                pb.hold(0.4)
            if look_sheets and rid in sheets:
                sh = np.asarray(sheets[rid]["xy"])
                pb.look_at(sh[0], sh[1], 0.0, hold=1.5)
            for pt in tl[1:]:                                   # then the ceiling line, the other way round
                for dy in (70.0, -70.0, 0.0):
                    pb.turn_to(back + dy, pitch=pt, rate=20.0)
                    pb.hold(0.4)
            pb.walk(planner.path(np.asarray(pb.last[1:3]), here))
            pb.turn_to(pitch=tl[0], rate=CORNER_TURN_W)
            continue
        if step == "enter":
            room = rooms[rid]
            loop = room_loop(room, planner, pb.view)
            if not loop:
                loop = [planner.snap(stand[rid])]
            here = np.asarray(pb.last[1:3])
            k0 = int(np.argmin([np.linalg.norm(q - here) for q in loop]))
            loop = loop[k0:] + loop[:k0] + [loop[k0]]
            corners = corner_targets(room)
            # waypoints nearest to each real corner split the loop into legs
            near = sorted({int(np.argmin([np.linalg.norm(q - c) for q in loop])) for c in corners})
            legs, a = [], 0
            for b in near + [len(loop) - 1]:
                if b > a:
                    legs.append((loop[a:b + 1], b))
                    a = b
            sheet_done = not look_sheets or rid not in sheets
            for li, pt in enumerate(tilts or (pitch,)):
              pb.turn_to(pitch=pt, rate=CORNER_TURN_W)
              looks = 0 if li == 0 else 2                       # corner glances on the floor loop only
              for leg, b in legs:
                pts = [np.asarray(pb.last[1:3])]
                for q in leg:
                    seg = planner.path(pts[-1], q)
                    pts.extend(list(seg[1:]) if len(seg) > 1 else [q])
                pb.walk(np.asarray(pts), look_offset=LOOK_INTO_ROOM)
                q = loop[b]
                if corner_looks and looks < 2:
                    dists = np.linalg.norm(corners - q, axis=1)
                    c = corners[int(np.argmin(dists))]
                    yaw_c = math.degrees(math.atan2(c[1] - q[1], c[0] - q[0]))
                    if abs(wrap(yaw_c - pb.last[4])) <= 70:     # a glance, not a turn-around
                        pb.turn_to(yaw_c, pitch=pt, rate=CORNER_TURN_W)
                        pb.hold(float(rng.uniform(0.5, 0.9)))
                        looks += 1
                if not sheet_done:
                    sh = np.asarray(sheets[rid]["xy"])
                    if np.linalg.norm(sh - q) < 2.5:
                        pb.look_at(sh[0], sh[1], 0.0, hold=1.5)
                        sheet_done = True
                pb.turn_to(pitch=pt, rate=CORNER_TURN_W)
            if not sheet_done:                                   # never came close: step towards it and look
                sh = np.asarray(sheets[rid]["xy"])
                cur = np.asarray(pb.last[1:3])
                q = planner.snap(sh + 1.3 * (cur - sh) / max(np.linalg.norm(cur - sh), 1e-6))
                pb.walk(planner.path(cur, q))
                pb.look_at(sh[0], sh[1], 0.0, hold=1.5)
                pb.turn_to(pitch=pitch, rate=CORNER_TURN_W)
        else:
            other, d = info
            c, h = door_geometry(d, rooms_obj, other)
            approach = planner.snap(c - 0.7 * np.array([math.cos(math.radians(h)), math.sin(math.radians(h))]))
            pb.walk(planner.path(pb.last[1:3], approach))
            pb.turn_to(h, pitch=pitch, rate=CORNER_TURN_W)
            pb.walk(np.array([pb.last[1:3], c]))
            pb.hold(2.0)                                         # pause in the doorway, facing in


def _line_of_sight(planner, a, b, margin=0.4) -> bool:
    """Clear straight view from a to within `margin` of b on the walkable map (walls, furniture, glass block)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.linalg.norm(b - a))
    if d <= margin + 0.1:
        return True
    ts = np.arange(0.1, d - margin, 0.04)
    q = a + (b - a)[None] * (ts / d)[:, None]
    wk = planner.wk
    i = np.round((q[:, 1] - wk.origin[1]) / wk.res).astype(int)
    j = np.round((q[:, 0] - wk.origin[0]) / wk.res).astype(int)
    H, W = wk.soft.shape
    if ((i < 0) | (i >= H) | (j < 0) | (j >= W)).any():
        return False
    return bool(wk.soft[i, j].all())


def farthest_from(planner, room, p0, visible=None) -> np.ndarray:
    """Walkable point of the room farthest from p0 (>= ~0.4 m from walls) WITH a clear view of p0: the 'far end' of a
    small room. Without the view test the k22 bathroom's far end was inside the glass shower enclosure (a close-up
    of the black glass; nobody stands there to photograph the door)."""
    from matplotlib.path import Path as MPath
    P = MPath(np.asarray(room["polygon"]))
    ii, jj = np.nonzero(planner.clear > 0.12)
    pts = np.stack([planner.wk.origin[0] + jj * planner.wk.res, planner.wk.origin[1] + ii * planner.wk.res], 1)
    pts = pts[P.contains_points(pts)]
    if not len(pts):
        return np.asarray(room["ceiling"]["center_xy"])
    order = np.argsort(-np.linalg.norm(pts - p0, axis=1))
    seen = visible or (lambda a, b: _line_of_sight(planner, a, b))
    for k in order[:600]:
        if seen(pts[k], p0):
            return pts[k]
    return pts[order[0]]


def _heading(a, b) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _free_dist(P: np.ndarray, xy, yaw: float) -> float:
    """Distance from xy along heading yaw to the room outline P (n, 2)."""
    o = np.asarray(xy, float)
    d = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
    best = np.inf
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        e = b - a
        den = d[0] * (-e[1]) + d[1] * e[0]
        if abs(den) < 1e-9:
            continue
        w = a - o
        t = (w[0] * (-e[1]) + w[1] * e[0]) / den          # along the ray
        u = (d[0] * w[1] - d[1] * w[0]) / den             # along the edge
        if t > 1e-6 and 0.0 <= u <= 1.0:
            best = min(best, t)
    return float(best)


def _depth_from_door(room, c, h) -> float:
    """How far the room reaches from its door along the heading into it (a balcony: ~1.4 m)."""
    u = np.array([math.cos(math.radians(h)), math.sin(math.radians(h))])
    return float(((np.asarray(room["polygon"]) - c) @ u).max())


def build_photos(add, gt, planner, rooms_obj, rooms, order, nbr, door_geometry, stand, rng, hfov, repeat_room,
                 visible=None, seer=None):
    """add(folder, x, y, yaw, z, pitch, dt) records one photo (dt = seconds since the previous one). Follows the module
    docstring (v2.1, D-055). Capture times come from the walking distance between spots plus aiming time, so a
    doorway pair (a turn on the spot) is always quicker than walking to the next spot, as with a person."""
    doors_done, entered_by, pair_into, turn = set(), {}, set(), {}
    route_doors = {info[1]["id"] for step, _, info in order if step == "go"}
    for step, rid, info in order:
        if step == "enter" and info is not None:
            entered_by[rid] = info
    ext = [o for o in gt["openings"] if o["cls"] == "door" and len(o["room_ids"]) == 1]
    last = {"xy": None}

    shots = []                                     # (folder, x, y, z, yaw, pitch) of every photo, for the coverage pass

    def shoot(folder, xy, yaw, z, pitch, aim_s, role=""):
        xy = np.asarray(xy, float)
        walk = 0.0 if last["xy"] is None else float(np.linalg.norm(xy - last["xy"])) / WALK_MPS
        add(folder, float(xy[0]), float(xy[1]), wrap(yaw), float(z), float(pitch), round(walk + aim_s, 2), role=role)
        shots.append((folder, float(xy[0]), float(xy[1]), float(z), float(wrap(yaw)), float(pitch)))
        last["xy"] = xy

    def open_aim(xy, yaw, need=1.5):
        """Turn away from a view blocked within `need` m (a person does not photograph a wall at 40 cm)."""
        if seer is None:
            return yaw
        for d in (0, -15, 15, -30, 30, -45, 45, -60, 60):
            if seer.free(xy[0], xy[1], yaw + d) >= need:
                return yaw + d
        return yaw

    def spin_headings(c, yaw0, n):
        hs, h = [], yaw0
        for _ in range(n):
            for d in (0, -15, -30, -45):              # keep turning right past a blocked view
                if seer.free(c[0], c[1], h + d) >= 1.2:
                    h = h + d
                    break
            hs.append(wrap(h))
            h = h - 0.7 * hfov
        return hs

    spin_plan = {}

    def plan_spin(rid, n):
        """Turning spot chosen by what the photos would show: most of the room's walls (1-2 m band) in view, views
        clear for >= 1.2 m. None: no spot where the turn keeps walls >= 1 m away (a bed fills the middle): the room
        is photographed like a small room instead."""
        if seer is None:
            return None
        if rid in spin_plan:
            return spin_plan[rid]
        cands = _room_cells(planner, rooms[rid])
        if len(cands) > 80:
            cands = cands[np.linspace(0, len(cands) - 1, 80).astype(int)]
        face = entry_door_xy(rid)
        best, scored = None, []
        for c in cands:
            yaw0 = _heading(c, face) if face is not None else 0.0
            vis = np.zeros(len(seer.samples[rid][0]), bool)
            fr = []
            for h in spin_headings(c, yaw0, n):
                vis |= seer.seen(c[0], c[1], 1.4, h, PHOTO_PITCH, rid)
                fr.append(seer.free(c[0], c[1], h))
            # v2.3 (k38 review: the living room's spot sat at the south end, its north half 5-8 m away): a wall seen
            # from beyond 4.5 m counts half (a 0.5x pixel there is ~2.5 cm), plus a mild pull to the room's middle
            dist = np.linalg.norm(seer.samples[rid][0][:, :2] - c, axis=1)
            w = np.where(dist > 4.5, 0.5, 1.0)
            cen = np.asarray(rooms[rid]["ceiling"]["center_xy"])
            score = float((vis * w).sum() / w.sum()) + 0.05 * min(min(fr), 2.5) - 0.02 * float(np.linalg.norm(c - cen))
            scored.append((score, c, float(min(fr))))
            if best is None or score > best[0]:
                best = (score, c, float(min(fr)), float(vis.mean()))
        ranked = [c for sc_, c, mf in sorted(scored, key=lambda t: -t[0]) if mf >= 1.0]
        spin_plan[rid] = None if best is None or best[2] < 1.0 else dict(xy=best[1], min_free=best[2], cover=best[3],
                                                                         ranked=ranked)
        return spin_plan[rid]

    def entry_door_xy(rid):
        d = entered_by.get(rid)
        if d is not None:
            return np.asarray(d["center"][:2])
        own = [o for o in ext if o["room_ids"][0] == rid]
        return np.asarray(own[0]["center"][:2]) if own else None

    def spin(rid, folder, base_xy):
        # bigger rooms get more turning photos (user, 14:00): ~4 under 10 m2, 5 up to 18 m2, 6 above; capped so the
        # room's folder stays within 8 (spin + 1 ceiling photo + one doorway photo per door)
        n_doors = len(nbr[rid])
        area = rooms[rid]["area"]
        target = 4 if area < 10 else (5 if area < 18 else 6)
        n = int(np.clip(min(target, 7 - n_doors), 3, 6))
        plan = plan_spin(rid, n)
        take2 = folder.endswith("_take2")
        if plan is not None:                       # v2.2: the spot that shows the room (Seer)
            base_xy = plan["xy"]
            if take2:                              # repeat take: another good spot 0.6-1.2 m away (k38 review: the
                alt = [c for c in plan.get("ranked", [])  # old take 2 stood within 25 cm and duplicated take 1)
                       if 0.5 <= float(np.linalg.norm(c - plan["xy"])) <= 1.5]
                if alt:
                    base_xy = alt[0]
                else:                              # v2.3: no ranked spot that far: 0.8 m towards the room's middle
                    cen = np.asarray(rooms[rid]["ceiling"]["center_xy"])
                    v_ = cen - plan["xy"]
                    base_xy = plan["xy"] + 0.8 * v_ / max(float(np.linalg.norm(v_)), 1e-6)
        x, y = planner.snap(base_xy + rng.normal(0, 0.12, 2))
        face = entry_door_xy(rid)                  # the first photo looks OUT through the door you came in by
        yaw = _heading((x, y), face) + rng.normal(0, 5) if face is not None else float(rng.uniform(-180, 180))
        if take2 and seer is not None:
            yaw = wrap(yaw - 0.5 * hfov)           # and starts half a frame further round
        z = 1.40 + rng.normal(0, 0.05)
        pitch = PHOTO_PITCH + rng.normal(0, 2.0)
        for k in range(n):
            if seer is not None:                   # keep turning right past a view blocked within 1.2 m
                for d in (0, -15, -30, -45):
                    if seer.free(x, y, yaw + d) >= 1.2:
                        yaw = wrap(yaw + d)
                        break
            shoot(folder, (x, y), yaw, z, pitch, float(rng.uniform(2.0, 3.0)) if k else 2.0,
                  role=f"turn {k + 1}/{n}" + (" (faces entry door)" if k == 0 and face is not None else ""))
            yaw = wrap(yaw - hfov * (1 - rng.uniform(0.10, 0.50)))
            z = float(np.clip(z + rng.normal(0, 0.04), 1.2, 1.6))
            pitch = float(np.clip(pitch + rng.normal(0, 2.5), -25, 0))
            x, y = np.array([x, y]) + rng.normal(0, 0.05, 2)
        ceiling_photo(rid, folder, (x, y), yaw)

    def ceiling_photo(rid, folder, xy, yaw):
        # aim the ceiling photo along the room's long side, so two ceiling-wall lines are in view; towards the side
        # with room in front (from a small room's far end: back towards the door). User, 4 Oct 15:00: the bathroom
        # ceiling photo aimed at the window wall 0.4 m away was a close-up, not a ceiling photo.
        P = np.asarray(rooms[rid]["polygon"])
        ext_ = P.max(0) - P.min(0)
        yaw_l = 0.0 if ext_[0] >= ext_[1] else 90.0
        flip = bool(rng.random() < 0.5)
        free = (_free_dist(P, xy, yaw_l), _free_dist(P, xy, yaw_l + 180.0))
        if min(free) < 1.2:
            flip = free[1] > free[0]
        yaw_l += 180.0 * flip
        shoot(folder, xy, yaw_l + rng.normal(0, 8), 1.40 + rng.normal(0, 0.05), CEIL_PHOTO_PITCH + rng.normal(0, 4),
              float(rng.uniform(2.5, 3.5)), role="ceiling")

    def small_axis(rid, d):
        """Threshold point, heading into the small room, and the room's axis seen from the door (towards its middle
        unless that is far off the door normal)."""
        c, h_in = door_geometry(d, rooms_obj, rid)
        h_c = _heading(c, rooms[rid]["ceiling"]["center_xy"])
        return c, h_in, (h_c if abs(wrap(h_c - h_in)) < 50 else h_in)

    def shallow(rid, d):
        """Shallow from its door AND long along the door wall (a balcony); a small square room is not (v2.3: the k38
        bathroom got +-60 deg doorway turns into a wall 0.4 m away)."""
        c, h_in, _ = small_axis(rid, d)
        depth = _depth_from_door(rooms[rid], c, h_in)
        t = np.array([-math.sin(math.radians(h_in)), math.cos(math.radians(h_in))])
        along = float(np.ptp((np.asarray(rooms[rid]["polygon"]) - c) @ t))
        return depth < SHALLOW_M and along >= 1.5 * depth

    def small(rid, folder, door, have_door_shot):
        c, h_in, base = small_axis(rid, door)
        sh = shallow(rid, door)
        # v2.2: a shallow room (balcony) is photographed ALONG its length; its doorway photos turn ~60 deg so each end
        # corner is in frame (k38 review: "far end looking back at the door only shows the living room")
        dl = turn.setdefault(rid, float(rng.uniform(55, 62)) if sh else float(rng.uniform(18, 26)))
        u = np.array([math.cos(math.radians(h_in)), math.sin(math.radians(h_in))])
        if not have_door_shot:                     # in: from the doorway, turned LEFT (the pair photo, when there is one)
            shoot(folder, c + 0.1 * u + rng.normal(0, 0.04, 2), base + dl + rng.normal(0, 4),
                  1.40 + rng.normal(0, 0.05), -10 + rng.normal(0, 3), 2.5, role="doorway, turned left (in)")
        z = 1.40 + rng.normal(0, 0.05)
        if sh:
            # out/along: from each end of the room, looking along it towards the other end
            P = np.asarray(rooms[rid]["polygon"])
            ax = 0 if np.ptp(P[:, 0]) >= np.ptp(P[:, 1]) else 1
            cells = (_room_cells(planner, rooms[rid], min_clear=0.2, min_edge=0.25) if seer is not None
                     else np.zeros((0, 2)))   # v2.3: the ends of a furnished balcony are only ~0.2 m clear
            if not len(cells):
                cells = np.array([farthest_from(planner, rooms[rid], c, visible)])
            a_end, b_end = cells[np.argmin(cells[:, ax])], cells[np.argmax(cells[:, ax])]
            h_ab = _heading(a_end, b_end)
            shoot(folder, a_end, h_ab + rng.normal(0, 4), z, -10 + rng.normal(0, 3), 2.5, role="one end, along the room")
            shoot(folder, b_end, h_ab + 180 + rng.normal(0, 4), z + rng.normal(0, 0.03), -10 + rng.normal(0, 3),
                  float(rng.uniform(2.0, 3.0)), role="other end, along the room")
            far, yaw_b = b_end, h_ab + 180
        else:
            # out: from the far end, two photos back at the door (turned left of it, then right), the door in both
            far = farthest_from(planner, rooms[rid], c, visible)
            if seer is not None:                   # v2.3: not wedged against a wall (k38 kitchen: 65% tile at 0.4 m)
                cells = _room_cells(planner, rooms[rid], min_clear=0.2, min_edge=0.25)
                if len(cells):
                    dd = np.linalg.norm(cells - c, axis=1)
                    top = cells[dd >= np.percentile(dd, 70)]
                    if len(top):
                        def _open(q):
                            hb = _heading(q, c)
                            return min(seer.free(q[0], q[1], hb + o) for o in (-30, 0, 30))
                        far = max(top, key=_open)
            yaw_b = _heading(far, c)
            half = math.degrees(math.atan2(door["width"] / 2, max(float(np.linalg.norm(far - c)), 0.5)))
            df = float(np.clip(hfov / 2 - half - 4, 18, 25)) * float(rng.uniform(0.9, 1.1))   # wide door: one jamb each
            shoot(folder, far, yaw_b + df + rng.normal(0, 3), z, -10 + rng.normal(0, 3), 2.5,
                  role="far end, left of door (out)")
            shoot(folder, far + rng.normal(0, 0.04, 2), yaw_b - df + rng.normal(0, 3), z + rng.normal(0, 0.03),
                  -10 + rng.normal(0, 3), float(rng.uniform(2.0, 3.0)), role="far end, right of door (out)")
        ceiling_photo(rid, folder, far + rng.normal(0, 0.05, 2), yaw_b)
        # in again, leaving: from the doorway, turned RIGHT (after the ceiling photo, so never part of a spin)
        shoot(folder, c + 0.1 * u + rng.normal(0, 0.04, 2), base - dl + rng.normal(0, 4), 1.40 + rng.normal(0, 0.05),
              -10 + rng.normal(0, 3), 2.0, role="doorway, turned right (in)")
        # shallow room (balcony): one photo of it from ~2 m back in the next room, through the door
        if _depth_from_door(rooms[rid], c, h_in) < SHALLOW_M:
            q = planner.snap(c - float(rng.uniform(1.6, 2.2)) * u)
            shoot(folder, q, _heading(q, rooms[rid]["ceiling"]["center_xy"]) + rng.normal(0, 4),
                  1.40 + rng.normal(0, 0.05), -8 + rng.normal(0, 3), 2.0, role="from next room, through door (in)")

    def pair(rid, other, d):
        """Doorway pair on the threshold of d: back into rid, then into other (turned left when other is small)."""
        c, _ = door_geometry(d, rooms_obj, other)
        xy = c + rng.normal(0, 0.06, 2)
        z = 1.40 + rng.normal(0, 0.05)
        if room_is_small(rooms[other]):
            h_next = small_axis(other, d)[2] + turn.setdefault(
                other, float(rng.uniform(55, 62)) if shallow(other, d) else float(rng.uniform(18, 26)))
        else:
            h_next = _heading(xy, stand[other])
        h_back = _heading(xy, (spin_plan.get(rid) or {}).get("xy", stand[rid]))
        h_back, h_next = open_aim(xy, h_back), open_aim(xy, h_next)
        # v2.3 (k38 review: the bedroom-door "back into the living room" photo had turned into the bedroom): each photo
        # must look INTO its own room, within 70 deg of the door's inward normal
        _, n_rid = door_geometry(d, rooms_obj, rid)
        _, n_oth = door_geometry(d, rooms_obj, other)
        h_back = n_rid + float(np.clip(wrap(h_back - n_rid), -70, 70))
        h_next = n_oth + float(np.clip(wrap(h_next - n_oth), -70, 70))
        shoot(rid, xy, h_back + rng.normal(0, 6), z, -12 + rng.normal(0, 3), 2.0,
              role=f"doorway pair: back from {other[3:]} door (in)")
        shoot(other, xy, h_next + rng.normal(0, 6), z + rng.normal(0, 0.02), -12 + rng.normal(0, 3),
              float(rng.uniform(2.3, 3.2)), role="doorway pair: into room" + (", turned left" if room_is_small(rooms[other]) else "") + " (in)")
        doors_done.add(d["id"])
        pair_into.add((other, d["id"]))

    def cramped(rid):
        """No turning spot that keeps the photos' views >= 1 m (v2.2): photograph it like a small room."""
        if seer is None or room_is_small(rooms[rid]):
            return False
        n_doors = len(nbr[rid])
        area = rooms[rid]["area"]
        n = int(np.clip(min(4 if area < 10 else (5 if area < 18 else 6), 7 - n_doors), 3, 6))
        return plan_spin(rid, n) is None

    for step, rid, info in order:
        if step == "enter":
            door = entered_by.get(rid) or (nbr[rid][0][1] if nbr[rid] else None)
            if (room_is_small(rooms[rid]) or cramped(rid)) and door is not None:
                small(rid, rid, door, (rid, door["id"]) in pair_into)
            else:
                spin(rid, rid, stand[rid])
            for other, d in nbr[rid]:              # a door the route never walks through still gets its pair
                if d["id"] not in route_doors and d["id"] not in doors_done:
                    pair(rid, other, d)
        elif step == "go":
            other, d = info
            if d["id"] not in doors_done:
                pair(rid, other, d)
    # v2.2 coverage pass (protocol: "check every wall appears in at least one photo of its room; if not, go back and
    # take one"): walls seen < 30% in the 1-2 m band get one photo each from the spot that shows most of them, within
    # the 8-photo cap; taken last, so they come after every room's ceiling photo
    if seer is not None:
        for rid in [r for st, r, _ in order if st == "enter"]:
            mine = [x for x in shots if x[0] == rid]
            vis = np.zeros(len(seer.samples[rid][0]), bool)
            for _, x, y, z, yaw, pitch in mine:
                vis |= seer.seen(x, y, z, yaw, pitch, rid)
            P, wid = seer.samples[rid]
            poly = np.asarray(rooms[rid]["polygon"])
            lens = {k: float(np.linalg.norm(poly[(k + 1) % len(poly)] - poly[k])) for k in np.unique(wid)}
            weak = sorted((k for k in np.unique(wid) if vis[wid == k].mean() < 0.3 and lens[k] >= 0.8),
                          key=lambda k: -lens[k])
            cells = _room_cells(planner, rooms[rid], min_edge=0.3)
            for k in weak:
                if len([x for x in shots if x[0] == rid]) >= 8 or not len(cells):
                    break
                best = None
                for c in cells[np.linspace(0, len(cells) - 1, min(len(cells), 60)).astype(int)]:
                    for h in np.arange(-180, 180, 20):
                        if seer.free(c[0], c[1], h) < 1.0:
                            continue
                        v = seer.seen(c[0], c[1], 1.4, h, PHOTO_PITCH, rid)
                        gain = float((v & (wid == k) & ~vis).sum()) / max(int((wid == k).sum()), 1)
                        if best is None or gain > best[0]:
                            best = (gain, c, h, v)
                if best is None or best[0] < 0.15:              # adds >= 15% of that wall (furniture hides the rest)
                    continue
                shoot(rid, best[1], best[2] + rng.normal(0, 3), 1.40 + rng.normal(0, 0.05), PHOTO_PITCH + rng.normal(0, 2),
                      2.5, role=f"coverage: wall {k + 1} (in)")
                vis |= best[3]
    if repeat_room:
        spin(repeat_room, repeat_room + "_take2", stand[repeat_room] + rng.normal(0, 0.3, 2))



# ------------------------------------------------------------------------------------- what the person sees (v2.2)
class Seer:
    """What a person holding the phone sees: room-wall points in the 4:3 photo frame, unoccluded (glass is see-through,
    as for the eye), and how far the view is clear straight ahead. v2.2 (D-064, user review 4 Oct 18:40: "turns 2/4
    and 3/4 are useless; photos should be captured properly from inside the room; check the data first"): the
    scripted person picks the turning spot, aims and adds photos by what the camera would actually show."""

    W, H, F = 4032, 3024, 3028.4

    def __init__(self, geom, gt, F=None):
        if F:
            self.F = float(F)
        from sim import scene_geom as sg
        self.sg = sg
        self.rc, _ = sg.raycaster(geom, sg.mesh_mask(geom, lambda m: not m["glass"]))
        self.samples = {}
        for r in gt["rooms"]:
            self.samples[r["id"]] = _wall_samples(np.asarray(r["polygon"]))

    def _cast(self, o, d):
        th, _, _ = self.sg.cast(self.rc, np.repeat(np.asarray(o, float)[None], len(d), 0), d)
        return th

    def free(self, x, y, yaw, z=1.4):
        a = math.radians(yaw)
        th = self._cast((x, y, z), np.array([[math.cos(a), math.sin(a), 0.0]]))
        return float(th[0]) if np.isfinite(th[0]) else 20.0

    def seen(self, x, y, z, yaw, pitch, rid):
        P, wid = self.samples[rid]
        a, b = math.radians(yaw), math.radians(pitch)
        f = np.array([math.cos(b) * math.cos(a), math.cos(b) * math.sin(a), math.sin(b)])
        r = np.array([math.sin(a), -math.cos(a), 0.0])
        R = np.stack([r, np.cross(f, r), f], 1)                 # columns: camera x (right), y (down), z (forward)
        o = np.array([x, y, z])
        pc = (P - o) @ R
        ok = pc[:, 2] > 0.2
        u = self.F * pc[:, 0] / np.where(ok, pc[:, 2], 1) + self.W / 2
        v = self.F * pc[:, 1] / np.where(ok, pc[:, 2], 1) + self.H / 2
        ok &= (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        vis = np.zeros(len(P), bool)
        if ok.any():
            dd = P[ok] - o
            L = np.linalg.norm(dd, axis=1)
            th = self._cast(o, dd / L[:, None])
            vis[np.flatnonzero(ok)] = np.abs(th - L) < 0.06
        return vis

    def wall_coverage(self, rid, vis):
        _, wid = self.samples[rid]
        return {int(k): float(vis[wid == k].mean()) for k in np.unique(wid)}


def _wall_samples(poly, step=0.10, heights=(1.0, 1.25, 1.5, 1.75, 2.0)):
    pts, wid = [], []
    ccw = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) > 0
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        L = float(np.linalg.norm(b - a))
        if L < 0.05:
            continue
        d = (b - a) / L
        n_in = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])
        for t in np.arange(0.05, L, step):
            q = a + t * d + 0.02 * n_in
            for zz in heights:
                pts.append([q[0], q[1], zz])
                wid.append(k)
    return np.asarray(pts), np.asarray(wid)


def _room_cells(planner, room, step=0.25, min_clear=0.35, min_edge=0.6):
    """Walkable standing spots inside the room (a 25 cm grid), clear of furniture and away from the walls."""
    from matplotlib.path import Path as MPath
    Pg = np.asarray(room["polygon"])
    ii, jj = np.nonzero(planner.clear >= min_clear)
    pts = np.stack([planner.wk.origin[0] + jj * planner.wk.res, planner.wk.origin[1] + ii * planner.wk.res], 1)
    pts = pts[MPath(Pg).contains_points(pts)]
    if not len(pts):
        return pts
    edge = np.full(len(pts), np.inf)
    for k in range(len(Pg)):
        a, b = Pg[k], Pg[(k + 1) % len(Pg)]
        ab = b - a
        t = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
        edge = np.minimum(edge, np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
    pts = pts[edge >= min(min_edge, 0.8 * edge.max())]
    key = np.round(pts / step).astype(int)
    _, first = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(first)]

# ------------------------------------------------------------------------------------- humanise + clearance
class Clearance:
    """3D distance from the camera to the nearest scene surface (all visible triangles). Poses closer than min_d are
    pushed back along the path's nominal position (the planned path is clear by construction)."""

    def __init__(self, geom, min_d=0.25):
        import open3d as o3d
        self.o3d = o3d
        self.sc = o3d.t.geometry.RaycastingScene()
        tm = o3d.t.geometry.TriangleMesh()
        tm.vertex.positions = o3d.core.Tensor(geom["V"])
        tm.triangle.indices = o3d.core.Tensor(geom["F"])
        self.sc.add_triangles(tm)
        self.min_d = min_d

    def visible(self, a_xy, b_xy, z: float = 1.4, margin: float = 0.4) -> bool:
        """Clear 3-D line of sight at phone height from a to within `margin` of b (glass and furniture block)."""
        a = np.array([a_xy[0], a_xy[1], z], np.float32)
        d = np.array([b_xy[0], b_xy[1], z], np.float32) - a
        L = float(np.linalg.norm(d))
        if L <= margin:
            return True
        r = self.sc.cast_rays(self.o3d.core.Tensor(np.concatenate([a, d / L])[None].astype(np.float32)))
        return bool(r["t_hit"].numpy()[0] >= L - margin)

    def dist(self, xyz: np.ndarray) -> np.ndarray:
        q = self.sc.compute_closest_points(self.o3d.core.Tensor(np.asarray(xyz, np.float32)))
        return np.linalg.norm(q["points"].numpy() - xyz, axis=1)

    def fix(self, P: np.ndarray, nominal: np.ndarray) -> tuple[np.ndarray, dict]:
        """P, nominal: (n, 7) rows [t, x, y, z, yaw, pitch, roll]. Blend bad poses towards the nominal position."""
        P = P.copy()
        d = self.dist(P[:, 1:4])
        bad0 = int((d < self.min_d).sum())
        for w in (0.5, 1.0):
            bad = d < self.min_d
            if not bad.any():
                break
            P[bad, 1:4] = (1 - w) * P[bad, 1:4] + w * nominal[bad, 1:4]
            d = self.dist(P[:, 1:4])
        return P, dict(too_close_before=bad0, too_close_after=int((d < self.min_d).sum()),
                       min_dist_m=round(float(d.min()), 3), p05_dist_m=round(float(np.percentile(d, 5)), 3))


def humanize_walk(samples: list, level: float, seed: int, clear: Clearance, rate=30.0) -> tuple[list, dict]:
    from sim.rig import resample
    nominal = resample(samples, rate)
    P = humanize(nominal, level=level, seed=seed)
    P, rep = clear.fix(P, nominal)
    return P.tolist(), rep


def humanize_photos(photos: list, level: float, rng, clear: Clearance) -> dict:
    bad = 0
    for p in photos:
        nom = list(p["pose"])
        hp = humanize_photo(nom, level=level, rng=rng)
        d = clear.dist(np.array([hp[:3]]))[0]
        if d < clear.min_d:
            hp[:3] = nom[:3]
            bad += 1
        p["pose"] = [round(float(v), 5) for v in hp]
        p["humanized"] = True
        p["clearance_m"] = round(float(clear.dist(np.array([p["pose"][:3]]))[0]), 3)
    return dict(photos=len(photos), reset_to_nominal=bad,
                min_clearance_m=min((p["clearance_m"] for p in photos), default=None))
