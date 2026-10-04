#!/usr/bin/env python
"""Script a capture session that follows docs/CAPTURE_PROTOCOL.md literally, like a person would (decision D-040).

Why scripted sessions: the walk-in test (30% of the score) is a cold run on a capture of an unseen space made by
someone else following our page. Scripted sessions let us rehearse that on many houses and many "people" (seeds)
without anyone driving, and A/B-test protocol variants with exact ground truth. The paths are the INTENDED paths;
render.py adds the slow drift of height / tilt / roll / aim and hand tremor (rig.humanize), so no two runs are alike.

Photos (protocol tier 1), per room in visiting order:
  spin    stand at the most open spot near the middle, phone landscape at chest height, tilted ~12 deg down, turn
          clockwise taking N photos, N = 8 - (doors of the room), at least 4 (7 for a one-door room, as the page says).
          Consecutive photos overlap by a VARYING amount (10-50% of the view, drawn per step): people do not turn
          exactly an eighth each time. Height and tilt drift from shot to shot (correlated), and the feet shuffle.
  corners (--photo-protocol corners, the alternative under test) stand ~0.6 m out of each free corner and aim
          across the room at the opposite corner.
  doorway pair: first time through each door, on the threshold: one photo back into the room being left, turn round,
          one photo into the next room, ~3-5 s apart (the EXIF times pair them).
  repeat: one room (--repeat-room, default the bedroom-like second room) shot again into <room>_take2.
Walk (video tier 2 and LiDAR tier 3 share one route; the LiDAR page says "walk the same route as the video"):
  start in the entrance room; in every room walk to its open middle, turn a full circle slowly (<= 20 deg/s) with the
  floor line in view, sweep the ceiling (tilt up) and the floor (tilt down), look at the floor sheet for ~2 s from two
  spots 1-2 m away; pause 2 s in every doorway facing into the next room; finish where you started and film the first
  view again for 3 s. take2 repeats the walk (another person-seed), lowlight repeats it with the lights dimmed.

Usage: python sim/auto_capture.py --scene <usd> --session outputs/sim/<name> [--seed 0] [--photo-protocol spin]
                                  [--takes take1,take2,lowlight] [--no-photos]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.rig import Rooms, Session, Walkable  # noqa: E402

RATE = 30.0                 # nominal path samples per second
WALK_V = 0.50               # m/s
TURN_W = 20.0               # deg/s standing (protocol limit)
WALK_TURN_W = 35.0          # deg/s while walking
PITCH = -14.0               # floor line in the lower third
PHOTO_LENS_F = 3028.4       # D-068: 1x main lens (D-065 tried the 0.5x ultra-wide, 1514.2 px: reverted)
HFOV_PHOTO = 67.3           # 1x main camera (isaac_common.PHOTO_F); the 0.5x ultra-wide is 106.2 deg


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class PathBuilder:
    def __init__(self, x, y, yaw, z=1.40, pitch=PITCH):
        self.s = [[0.0, x, y, z, yaw, pitch, 0.0]]

    @property
    def last(self):
        return self.s[-1]

    def _push(self, dt, x, y, z, yaw, pitch, roll=0.0):
        self.s.append([self.last[0] + dt, x, y, z, yaw, pitch, roll])

    def hold(self, sec):
        n = max(1, int(round(sec * RATE)))
        for _ in range(n):
            self._push(1 / RATE, *self.last[1:])

    def turn_to(self, yaw=None, pitch=None, rate=TURN_W, direction=0):
        t, x, y, z, y0, p0, r = self.last
        y1 = y0 if yaw is None else y0 + (wrap(yaw - y0) if direction == 0 else
                                           (((yaw - y0) % 360) if direction > 0 else -((y0 - yaw) % 360)))
        p1 = p0 if pitch is None else pitch
        ang = max(abs(y1 - y0), abs(p1 - p0))
        n = max(1, int(math.ceil(ang / rate * RATE)))
        for k in range(1, n + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * k / n)           # ease in / out
            self._push(1 / RATE, x, y, z, y0 + a * (y1 - y0), p0 + a * (p1 - p0))

    def spin(self, degrees=360.0, rate=TURN_W):
        y0 = self.last[4]
        steps = int(math.ceil(abs(degrees) / 30))
        for k in range(1, steps + 1):
            self.turn_to(y0 - degrees * k / steps, rate=rate, direction=-1 if degrees > 0 else 1)

    view = None                      # ViewChecker (set by build): lets the walker look toward open space

    def walk(self, pts: np.ndarray, face_forward=True, look_offset=0.0):
        """Follow a polyline at walking speed; heading follows the path with a turn-rate limit. When looking along
        the path would fill the picture with a near wall (< 1.5 m), the person looks toward the most open direction
        within 60 deg instead (protocol: "never fill the picture with a blank wall")."""
        if len(pts) < 2:
            return
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        n = max(2, int(s[-1] / WALK_V * RATE))
        ss = np.linspace(0, s[-1], n)
        xs, ys = np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])
        # first turn towards the path, standing
        h0 = math.degrees(math.atan2(ys[min(5, n - 1)] - ys[0], xs[min(5, n - 1)] - xs[0]))
        if face_forward and abs(wrap(h0 - self.last[4])) > 120:
            self.turn_to(h0, rate=35.0)            # only a reversal is a standing turn; smaller turns happen while walking
        
        yaw = self.last[4]
        for k in range(1, n):
            j = min(k + 8, n - 1)
            if face_forward and (xs[j] != xs[k] or ys[j] != ys[k]):
                tgt = math.degrees(math.atan2(ys[j] - ys[k], xs[j] - xs[k])) + look_offset
                if self.view is not None:
                    tgt = self.view.open_heading(xs[k], ys[k], tgt)
                d = wrap(tgt - yaw)
                yaw += float(np.clip(d, -WALK_TURN_W / RATE, WALK_TURN_W / RATE))
            self._push(1 / RATE, xs[k], ys[k], self.last[3], yaw, self.last[5])

    def look_at(self, px, py, pz=0.0, hold=2.0):
        t, x, y, z, yaw, p, r = self.last
        yaw1 = math.degrees(math.atan2(py - y, px - x))
        pitch1 = math.degrees(math.atan2(pz - z, math.hypot(px - x, py - y)))
        self.turn_to(yaw1, pitch=max(pitch1, -70))
        self.hold(hold)
        self.turn_to(pitch=p)                       # back to the tilt the person was holding


class Planner:
    """Shortest paths on the walkable grid that keep away from walls (cost grows near obstacles)."""

    def __init__(self, wk: Walkable):
        import cv2
        from skimage.graph import MCP_Geometric
        self.wk = wk
        free = wk.strict.astype(np.uint8)
        self.clear = cv2.distanceTransform(free, cv2.DIST_L2, 5) * wk.res
        cost = np.where(free > 0, 1.0 + 2.5 / (self.clear + 0.15), np.inf)
        self.cost = cost                      # strict: clear of furniture at body/phone height (D-051)
        soft_cost = cost.copy()
        soft_cost[wk.soft & ~wk.strict] = 60.0
        self.soft_cost = soft_cost            # fallback only when a target is unreachable in the strict map
        self.MCP = MCP_Geometric

    def ij(self, p):
        return (int(round((p[1] - self.wk.origin[1]) / self.wk.res)), int(round((p[0] - self.wk.origin[0]) / self.wk.res)))

    def xy(self, ij):
        return np.array([self.wk.origin[0] + ij[1] * self.wk.res, self.wk.origin[1] + ij[0] * self.wk.res])

    def snap(self, p):
        i, j = self.ij(p)
        fin = np.isfinite(self.cost)
        if 0 <= i < fin.shape[0] and 0 <= j < fin.shape[1] and fin[i, j]:
            return p
        ii, jj = np.nonzero(fin)
        k = np.argmin((ii - i) ** 2 + (jj - j) ** 2)
        return self.xy((ii[k], jj[k]))

    def path(self, a, b):
        a, b = self.snap(np.asarray(a, float)), self.snap(np.asarray(b, float))
        m = self.MCP(self.cost, fully_connected=True)
        costs, _ = m.find_costs([self.ij(a)], [self.ij(b)])
        if not np.isfinite(costs[self.ij(b)]):                     # unreachable in the strict map: soft fallback
            m = self.MCP(self.soft_cost, fully_connected=True)
            m.find_costs([self.ij(a)], [self.ij(b)])
        tr = np.array(m.traceback(self.ij(b)))
        pts = np.array([self.xy(q) for q in tr])
        if len(pts) > 5:                         # smooth the staircase
            k = 9
            pad = np.r_[np.repeat(pts[:1], k, 0), pts, np.repeat(pts[-1:], k, 0)]
            ker = np.ones(2 * k + 1) / (2 * k + 1)
            sm = np.stack([np.convolve(pad[:, c], ker, mode="same")[k:-k] for c in (0, 1)], 1)
            sm[0], sm[-1] = pts[0], pts[-1]
            pts = sm
        return pts[::3] if len(pts) > 6 else pts


class ViewChecker:
    """How far the camera can see along a heading, from the walkable map (walls block; 2 cm ray marching)."""

    def __init__(self, wk: Walkable, max_m=3.0):
        self.wk, self.max_m = wk, max_m
        self.steps = np.arange(0.1, max_m, 0.04)

    def free(self, x, y, headings_deg):
        h = np.radians(np.atleast_1d(headings_deg))[:, None]
        px = x + np.cos(h) * self.steps[None]
        py = y + np.sin(h) * self.steps[None]
        i = np.round((py - self.wk.origin[1]) / self.wk.res).astype(int)
        j = np.round((px - self.wk.origin[0]) / self.wk.res).astype(int)
        H, W = self.wk.soft.shape
        inside = (i >= 0) & (i < H) & (j >= 0) & (j < W)
        ok = np.zeros_like(inside)
        ok[inside] = self.wk.soft[i[inside], j[inside]]
        blocked = ~ok
        first = np.where(blocked.any(1), blocked.argmax(1), len(self.steps) - 1)
        return self.steps[first] + 0.12          # + the wall margin of the walkable map

    def open_heading(self, x, y, path_heading, need=1.5):
        if self.free(x, y, path_heading)[0] >= need:
            return path_heading
        cand = path_heading + np.arange(-60, 61, 10)
        f = np.minimum(self.free(x, y, cand), 2.5)
        return float(cand[int(np.argmax(f - 0.004 * np.abs(cand - path_heading)))])


def scene_graph(gt):
    rooms = {r["id"]: r for r in gt["rooms"]}
    doors = [o for o in gt["openings"] if o["cls"] == "door" and len(o["room_ids"]) == 2]
    ext = [o for o in gt["openings"] if o["cls"] == "door" and len(o["room_ids"]) == 1]
    start = ext[0]["room_ids"][0] if ext else max(rooms, key=lambda k: rooms[k]["area"])
    nbr = {k: [] for k in rooms}
    for d in doors:
        a, b = d["room_ids"]
        nbr[a].append((b, d))
        nbr[b].append((a, d))
    return rooms, doors, start, nbr


def door_geometry(d, rooms_obj: Rooms, into: str):
    """Threshold point and the heading that faces into room `into`."""
    c = np.asarray(d["center"][:2])
    n = np.array([0.0, 1.0]) if d["axis"] == "x" else np.array([1.0, 0.0])
    for sgn in (1, -1):
        if rooms_obj.at(*(c + sgn * 0.6 * n)) == into:
            return c, math.degrees(math.atan2(sgn * n[1], sgn * n[0]))
    return c, 0.0


def standing_point(room, planner: Planner):
    """Most open walkable spot inside the room, at least ~0.6 m inside its outline (not in a doorway)."""
    from matplotlib.path import Path as MPath
    Pg = np.asarray(room["polygon"])
    P = MPath(Pg)
    ii, jj = np.nonzero(planner.clear > 0)
    pts = np.stack([planner.wk.origin[0] + jj * planner.wk.res, planner.wk.origin[1] + ii * planner.wk.res], 1)
    inside = P.contains_points(pts)
    if not inside.any():
        return np.asarray(room["ceiling"]["center_xy"])
    pts, cl = pts[inside], planner.clear[ii[inside], jj[inside]]
    edge = np.full(len(pts), np.inf)
    for k in range(len(Pg)):
        a, b = Pg[k], Pg[(k + 1) % len(Pg)]
        ab = b - a
        t = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
        edge = np.minimum(edge, np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
    need = min(0.6, 0.8 * edge.max())
    ok = (edge >= need) & (cl >= min(0.35, 0.8 * cl.max()))
    if not ok.any():
        ok = edge >= need
    # "Near the middle of the room", as a person picks it: far from the walls (up to ~1.6 m; a spot 0.8 m from a wall
    # gives close-up spin photos), clear of furniture, then nearest the centroid (in a long hub room the most open
    # spot can be at one end; in an L-shaped room the centroid can sit next to a wall).
    from shapely.geometry import Polygon
    c = np.asarray(Polygon(Pg).centroid.coords[0])
    d = np.linalg.norm(pts - c, axis=1)
    score = np.minimum(edge, 1.6) + 0.5 * np.minimum(cl, 0.8) - 0.12 * d
    return pts[int(np.argmax(np.where(ok, score, -np.inf)))]


def build(gt_dir: Path, scene: str, session_dir: Path, seed: int, photo_protocol: str, takes: list[str],
          photos: bool, repeat_room: str | None, human_level: float, photos_only: bool = False,
          walks_only: bool = False):
    gt = json.loads((gt_dir / "sim_gt.json").read_text())
    wk = Walkable.load(gt_dir / "walkable.npz")
    rooms_obj = Rooms.load(gt_dir / "sim_gt.json")
    planner = Planner(wk)
    PathBuilder.view = ViewChecker(wk)
    rooms, doors, start, nbr = scene_graph(gt)
    rng = np.random.default_rng(seed)
    stand = {k: standing_point(r, planner) for k, r in rooms.items()}
    # depth-first visiting order with the doors used
    order, seen = [], set()

    def dfs(rid, via):
        seen.add(rid)
        order.append(("enter", rid, via))
        for other, d in sorted(nbr[rid], key=lambda od: od[0]):
            if other not in seen:
                order.append(("go", rid, (other, d)))
                dfs(other, d)
                order.append(("back", other, (rid, d)))
    dfs(start, None)

    sess_path = session_dir / "session.json"
    if walks_only and sess_path.exists():
        sess = Session(sess_path, scene)
        sess.d["takes"] = []
        sess.d.setdefault("meta", {}).update(walks_regenerated="view-aware heading (D-045)")
        photos = False
    elif photos_only and sess_path.exists():
        sess = Session(sess_path, scene)
        sess.d["photos"] = []
        sess.d.setdefault("meta", {}).update(photo_protocol=photo_protocol, photos_regenerated=True)
        takes = []
    else:
        if sess_path.exists():
            sess_path.unlink()
        sess = Session(sess_path, scene, meta=dict(generator="auto_capture", seed=seed, photo_protocol=photo_protocol,
                                                   human_level=human_level, start_room=start))
    if photo_protocol == "v2":                                # D-051: the protocol designed once (protocol_v2.py)
        return _build_v2(gt, scene, sess, planner, rooms_obj, rooms, order, nbr, stand, start, seed, takes, photos,
                         repeat_room, human_level)
    # ------------------------------------------------------------------------------------------- walk takes
    for take in takes:
        x, y = stand[start]
        trng = np.random.default_rng(seed * 10 + len(sess.d["takes"]))
        pb = PathBuilder(x, y, yaw=float(trng.uniform(-180, 180)))
        first_view = list(pb.last)
        for step, rid, info in order:
            if step == "enter":
                sh = next((s for s in gt["sheets"] if s["room"] == rid), None)
                ang0 = math.degrees(math.atan2(pb.last[2] - sh["xy"][1], pb.last[1] - sh["xy"][0])) if sh else 0.0

                def sheet_view(ang):
                    q = planner.snap(np.asarray(sh["xy"]) + 1.5 * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))]))
                    pb.walk(planner.path(pb.last[1:3], q))
                    pb.look_at(*sh["xy"], 0.0, hold=1.2)
                if sh:                                                       # sheet in view, first spot (on the way in)
                    sheet_view(ang0 + trng.uniform(-30, 30))
                p = planner.snap(stand[rid] + trng.normal(0, 0.15, 2))
                pb.walk(planner.path(pb.last[1:3], p))
                # one slow full turn; the tilt swings between the floor line and the ceiling line meanwhile
                # (covers the LiDAR page's ceiling and floor sweeps without extra time)
                y0, deg = pb.last[4], 360.0 + trng.uniform(-10, 30)
                n = int(deg / TURN_W * RATE)
                ph = trng.uniform(0, 6.28)
                for k in range(1, n + 1):
                    a = k / n
                    pitch = -14.0 + 28.0 * math.sin(2 * math.pi * 2.0 * a + ph)
                    pb._push(1 / RATE, pb.last[1], pb.last[2], pb.last[3], y0 - deg * a, float(np.clip(pitch, -45, 32)))
                pb.turn_to(pitch=PITCH)
                if sh:                                                       # second spot, about 100 deg around
                    sheet_view(ang0 + 100 + trng.uniform(-20, 20))
            else:                                                            # go / back through a door
                other, d = info
                c, h = door_geometry(d, rooms_obj, other)
                approach = planner.snap(c - 0.7 * np.array([math.cos(math.radians(h)), math.sin(math.radians(h))]))
                pb.walk(planner.path(pb.last[1:3], approach))
                pb.turn_to(h, pitch=PITCH)
                pb.walk(np.array([pb.last[1:3], c]))
                pb.turn_to(h, pitch=PITCH)
                pb.hold(2.0)                                                 # pause in the doorway, facing in
        pb.walk(planner.path(pb.last[1:3], first_view[1:3]))
        pb.turn_to(first_view[4], pitch=first_view[5])
        pb.hold(3.0)                                                         # film the first view again
        sess.start_take("dim" if take == "lowlight" else "day")
        sess.take["name"] = take
        sess.take["kind"] = "walk"
        for s in pb.s:
            sess.add_sample(*([s[0]] + [s[1:]]))
        sess.stop_take()
    # ---------------------------------------------------------------------------------------------- photos
    if photos:
        t = 0.0
        hfov = HFOV_PHOTO
        doors_done = set()

        def shoot(folder, x, y, yaw, z, pitch, dt):
            nonlocal t
            t += dt
            sess.add_photo(t, [x, y, z, yaw, pitch, rng.normal(0, 1.0)], folder, "day")

        def spin_set(rid, folder, base_xy):
            n_doors = sum(1 for _, d in nbr[rid])
            n = int(np.clip(8 - n_doors, 4, 7))
            x, y = planner.snap(base_xy + rng.normal(0, 0.15, 2))
            # start facing the room's longest open direction, then turn clockwise with varying overlap
            yaw = float(rng.uniform(-180, 180))
            z = 1.40 + rng.normal(0, 0.05)
            pitch = -12.0 + rng.normal(0, 2.0)
            for k in range(n):
                shoot(folder, x, y, yaw, z, pitch, dt=2.5 if k else 8.0)
                if photo_protocol == "spin_even":                           # variant: N photos spread over the circle
                    yaw = wrap(yaw - 360.0 / n + rng.normal(0, 8.0))
                else:                                                       # page: overlap the previous by ~a quarter
                    overlap = rng.uniform(0.10, 0.50)                       # varies shot to shot
                    yaw = wrap(yaw - hfov * (1 - overlap))
                z = float(np.clip(z + rng.normal(0, 0.04), 1.2, 1.6))       # height drifts (correlated)
                pitch = float(np.clip(pitch + rng.normal(0, 2.5), -25, 0))  # tilt drifts (correlated)
                x, y = np.array([x, y]) + rng.normal(0, 0.06, 2)            # feet shuffle

        def corner_set(rid, folder):
            room = rooms[rid]
            P = np.asarray(room["polygon"])
            c = P.mean(0)
            cand = []
            for v in P:
                q = v + 0.6 * (c - v) / max(np.linalg.norm(c - v), 1e-6)
                q = planner.snap(q)
                cand.append(q)
            n_doors = sum(1 for _, d in nbr[rid])
            n = int(np.clip(8 - n_doors, 2, 6))
            idx = np.argsort([-np.linalg.norm(q - c) for q in cand])[:n]
            for k in idx:
                q = cand[k]
                yaw = math.degrees(math.atan2(c[1] - q[1], c[0] - q[0])) + rng.normal(0, 6)
                shoot(folder, q[0], q[1], yaw, 1.40 + rng.normal(0, 0.06), -12 + rng.normal(0, 3), dt=6.0)

        for step, rid, info in order:
            if step == "enter":
                if photo_protocol == "corners":
                    corner_set(rid, rid)
                else:
                    spin_set(rid, rid, stand[rid])
            elif step == "go":
                other, d = info
                if d["id"] in doors_done:
                    continue
                doors_done.add(d["id"])
                c, h_in = door_geometry(d, rooms_obj, other)
                xy = c + rng.normal(0, 0.08, 2)
                z = 1.40 + rng.normal(0, 0.05)
                t += 6.0
                # a person aims "back into the room" at its open middle, not straight across a narrow hallway
                h_back = math.degrees(math.atan2(stand[rid][1] - xy[1], stand[rid][0] - xy[0]))
                h_next = math.degrees(math.atan2(stand[other][1] - xy[1], stand[other][0] - xy[0]))
                shoot(rid, xy[0], xy[1], wrap(h_back + rng.normal(0, 8)), z, -12 + rng.normal(0, 3), dt=0.0)
                shoot(other, xy[0], xy[1], wrap(h_next + rng.normal(0, 8)), z + rng.normal(0, 0.02),
                      -12 + rng.normal(0, 3), dt=float(rng.uniform(2.5, 4.5)))
        rep = repeat_room or (sorted(rooms)[1] if len(rooms) > 1 else start)
        t += 30.0
        if photo_protocol == "corners":
            corner_set(rep, rep + "_take2")
        else:
            spin_set(rep, rep + "_take2", stand[rep] + rng.normal(0, 0.3, 2))
    sess.save()
    return sess



def _build_v2(gt, scene, sess, planner, rooms_obj, rooms, order, nbr, stand, start, seed, takes, photos,
              repeat_room, human_level):
    """D-051: perimeter-loop walk (floor pass, then ceiling pass) and the v2 photo set, humanised here and checked for
    3D clearance (no camera inside furniture), so render.py renders exactly what was checked."""
    from sim import protocol_v2 as V
    from sim import scene_geom as sg
    geom = sg.load(scene)
    clear = V.Clearance(geom, min_d=0.15)           # v2.2 (user: "25 cm is too much"): camera >= 15 cm from surfaces
    seer = V.Seer(geom, gt, F=PHOTO_LENS_F)
    rng = np.random.default_rng(seed)
    reports = {}
    for ti, take in enumerate(takes):
        trng = np.random.default_rng(seed * 10 + ti)
        x, y = planner.snap(stand[start])
        pb = PathBuilder(x, y, yaw=float(trng.uniform(-180, 180)), pitch=V.FLOOR_PITCH)
        first = list(pb.last)
        # v2.2 (D-064, user 4 Oct 19:00): each room's floor line, then its ceiling line, before moving on (was: the whole
        # house at the floor, then the whole house again at the ceiling). A room's two looks are seconds apart, so
        # drift between them is small and the tracker never has to re-find a room it left minutes ago.
        V.build_walk(pb, gt, planner, rooms_obj, rooms, order, door_geometry, stand, trng, V.FLOOR_PITCH,
                     tilts=(V.FLOOR_PITCH, V.CEIL_PITCH))
        pb.walk(planner.path(pb.last[1:3], first[1:3]))
        pb.turn_to(first[4], pitch=first[5], rate=V.CORNER_TURN_W)
        pb.hold(3.0)                                                          # film the first view again
        hs, rep = V.humanize_walk(pb.s, human_level, seed * 1000 + ti, clear)
        sess.start_take("dim" if take == "lowlight" else "day")
        sess.take.update(name=take, kind="walk", humanized=True, protocol="v2", clearance=rep)
        for r in hs:
            sess.add_sample(r[0], r[1:])
        sess.stop_take()
        reports[take] = rep
    if photos:
        t = [0.0]

        def add(folder, x, y, yaw, z, pitch, dt, role=""):
            t[0] += dt
            sess.d["photos"].append(dict(index=len(sess.d["photos"]) + 1, t=round(t[0], 3), folder=folder,
                                         lighting="day", pose=[x, y, z, yaw, pitch, float(rng.normal(0, 1.0))],
                                         role=role, f_px=PHOTO_LENS_F, lens="0.5x" if PHOTO_LENS_F < 2000 else "1x"))
        rep_room = repeat_room or next((r for r in sorted(rooms) if r != start and not V.room_is_small(rooms[r])),
                                       start)
        V.build_photos(add, gt, planner, rooms_obj, rooms, order, nbr, door_geometry, stand, rng, HFOV_PHOTO, rep_room,
                       visible=clear.visible, seer=seer)
        reports["photos"] = V.humanize_photos(sess.d["photos"], human_level, rng, clear)
    sess.d.setdefault("meta", {}).update(protocol="v2", clearance=reports)
    sess.save()
    return sess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--photo-protocol", default="v2", choices=["v2", "spin", "spin_even", "corners"],
                    help="v2 = walk + photos designed from published practice and my review (D-051)")
    ap.add_argument("--takes", default="take1,take2")
    ap.add_argument("--no-photos", action="store_true")
    ap.add_argument("--repeat-room", default=None)
    ap.add_argument("--human", type=float, default=1.0)
    ap.add_argument("--photos-only", action="store_true", help="keep the session's walks, regenerate its photos")
    ap.add_argument("--walks-only", action="store_true", help="keep the session's photos, regenerate its walks")
    a = ap.parse_args()
    scene = str(Path(a.scene).resolve())
    gt_dir = Path(scene).parent / ".simcache" / "gt"
    s = build(gt_dir, scene, a.session, a.seed, a.photo_protocol, [t for t in a.takes.split(",") if t],
              not a.no_photos, a.repeat_room, a.human, a.photos_only, a.walks_only)
    for tk in s.d["takes"]:
        S = np.asarray(tk["samples"])
        dist = np.sum(np.hypot(np.diff(S[:, 1]), np.diff(S[:, 2])))
        print(f"[auto] {tk['name']}: {S[-1, 0] - S[0, 0]:.0f} s, {dist:.1f} m, {len(S)} samples, lighting {tk['lighting']}")
    from collections import Counter
    print(f"[auto] photos: {len(s.d['photos'])} in folders {dict(Counter(p['folder'] for p in s.d['photos']))}")
    print(f"[auto] -> {a.session / 'session.json'}")


if __name__ == "__main__":
    main()

