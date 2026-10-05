#!/usr/bin/env python
"""Door photos for a scripted photo session: one photo OUT through each door between two rooms, from 1.0-1.5 m inside
the smaller room, and one IN through it from 1.5-2.5 m back in the neighbour, or 0.4-1.2 m when nothing of the room's
furniture can be seen from that band (a door off a 1 m alcove, sim k65). CAPTURE_PROTOCOL 3, "keep the door leaf out
of the picture"; sim k65, 5 Oct: the bedroom and bathroom were not stitched because their door photos showed jambs, a
passage wall and a bare living-room wall, so nothing matched the next room (the scene's door leaves are hidden: the
"leaf" in those photos is the jamb, 0.35 m from a camera standing inside the door frame).

The spot and the heading are chosen by what the camera would show (ray-cast labels of the scene's visible meshes,
as the renderer draws them; glass is see-through): the textured objects (furniture, frames, appliances; not wall,
floor, ceiling, door frame, window frame or mirror) of the room on the other side of the opening, weighted by their
depth spread, with door frames and close-ups (< 0.6 m) kept small. Then the person's aim error (rig.humanize_photo,
seeded) is added and the photo is checked again, as one would check the screen and retake a bad shot.

  plan      CPU: plans the photos, writes <out>/render_session/session.json (only the new photos, humanised, so
            render.py renders exactly those poses) and <out>/door_plan.json (scores, candidates)
  render    GPU (not here): flock outputs/.gpu.lock envs/isaacsim_np126/bin/python sim/render.py
            <out>/render_session --no-takes
  assemble  CPU: emulates the new renders like the old photos (emulate.write_photos, iphone15 profile), copies the
            old photo folders, writes <out>/photos/<room>/IMG_NNNN.JPG, <out>/session.json (old + new, roles),
            <out>/render_photos/photos.json (true poses of all) and a contact sheet of the new photos

Usage: python sim/door_photos.py plan --session outputs/sim/k65v2/old_v21 --out outputs/sim/k65v2/doorphotos
       python sim/door_photos.py assemble --session outputs/sim/k65v2/old_v21 --out outputs/sim/k65v2/doorphotos \
              --sheet ../MyHouse_Dataset/k65_new_door_photos.jpg
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import scene_geom as sg  # noqa: E402
from sim.rig import Walkable, camera_to_world, humanize_photo  # noqa: E402

W, H, F = 4032, 3024, 3028.4            # iPhone 15 main camera, 1x (isaac_common.PHOTO_*)
HFOV = math.degrees(2 * math.atan(W / 2 / F))
PITCH, Z_CAM = -12.0, 1.40              # protocol_v2 turning photos: chest height, ~12 deg down
OUT_BACK, IN_BACK = (1.0, 1.5), (1.5, 2.5)
OUT_MAX_OFF = 35.0                      # outward heading within this of the door normal
MIN_CLEAR = 0.15                        # camera to any surface (protocol_v2 Clearance, D-064)
NEAR_M = 0.6                            # a close-up pixel
NX, NY = 96, 72                         # ray grid of the planning views (42 px per ray)
# textured classes, weighted by how well they match between photos (glossy black TV, glass lamps, curtain folds less)
TEX_W = dict(television=0.5, chandelier=0.5, spot_light=0.2, curtain=0.3)
NOT_TEX = {"wall", "floor", "ceiling", "door", "doorsill", "window", "mirror"}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class SceneView:
    """Ray-cast label images of the scene: class, range, depth and room of every pixel of a (coarse) photo."""

    def __init__(self, scene: str, gt: dict, gt_dir: Path):
        from shapely.geometry import Polygon
        self.geom = geom = sg.load(scene)
        self.gt = gt
        self.rc, tri_ids = sg.raycaster(geom, sg.mesh_mask(geom, lambda m: not m["glass"]))
        names = sorted({m["cls"] for m in geom["meshes"]})
        self.cls_names = names
        code = np.array([names.index(m["cls"]) for m in geom["meshes"]])
        self.tri_cls = code[geom["tri_mesh"][tri_ids]]
        self.cls_w = np.array([0.0 if n in NOT_TEX else TEX_W.get(n, 1.0) for n in names])
        self.door_codes = np.array([names.index(n) for n in ("door", "doorsill") if n in names])
        self.rooms = {r["id"]: r for r in gt["rooms"]}
        self.room_ids = list(self.rooms)
        # room raster (2 cm): which room a hit point lies in (outline + 5 cm: a hit on a wall face belongs to the
        # room whose face it is; the other room's face is a wall thickness away)
        allp = np.concatenate([np.asarray(r["polygon"]) for r in gt["rooms"]])
        self.o = allp.min(0) - 1.0
        self.res = 0.02
        n = np.ceil((allp.max(0) + 1.0 - self.o) / self.res).astype(int)
        gx, gy = np.meshgrid(self.o[0] + np.arange(n[0]) * self.res, self.o[1] + np.arange(n[1]) * self.res)
        from matplotlib.path import Path as MPath
        self.room_r = np.full(gx.shape, -1, np.int16)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        for k, rid in enumerate(self.room_ids):
            P = Polygon(self.rooms[rid]["polygon"]).buffer(0.05, join_style=2)
            m = MPath(np.asarray(P.exterior.coords)).contains_points(pts).reshape(gx.shape)
            self.room_r[m & (self.room_r < 0)] = k
        self.wk = Walkable.load(gt_dir / "walkable.npz")
        from sim.protocol_v2 import Clearance
        self.clear = Clearance(geom, min_d=MIN_CLEAR)

    def room_of(self, xy: np.ndarray) -> np.ndarray:
        ij = np.round((np.asarray(xy) - self.o) / self.res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.room_r.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.room_r.shape[0])
        out = np.full(len(ij), -1, int)
        out[ok] = self.room_r[ij[ok, 1], ij[ok, 0]]
        return out

    def rays(self, pose, nx=NX, ny=NY):
        T = camera_to_world(*pose)
        u = (np.arange(nx) + 0.5) * W / nx - 0.5
        v = (np.arange(ny) + 0.5) * H / ny - 0.5
        uu, vv = np.meshgrid(u, v)
        d = np.stack([(uu - (W / 2 - 0.5)) / F, -(vv - (H / 2 - 0.5)) / F, -np.ones_like(uu)], -1).reshape(-1, 3)
        dz = 1.0 / np.linalg.norm(d, axis=1)                     # depth = range * cos(angle to the optical axis)
        dw = d @ T[:3, :3].T
        dw /= np.linalg.norm(dw, axis=1, keepdims=True)
        return np.repeat(T[:3, 3][None], len(dw), 0), dw, dz

    def view(self, pose, nx=NX, ny=NY) -> dict:
        o, d, dz = self.rays(pose, nx, ny)
        t, pid, _ = sg.cast(self.rc, o, d)
        hit = np.isfinite(t)
        cls = np.full(len(t), -1, int)
        cls[hit] = self.tri_cls[pid[hit]]
        P = o + d * np.where(hit, t, 0.0)[:, None]
        room = np.where(hit, self.room_of(P[:, :2]), -1)
        return dict(t=t, depth=np.where(hit, t * dz, np.inf), cls=cls, room=room, P=P, hit=hit, nx=nx, ny=ny)

    def metrics(self, v: dict, stand: str, target: str) -> dict:
        n = len(v["t"])
        hit = v["hit"]
        w = np.where(hit, self.cls_w[np.maximum(v["cls"], 0)], 0.0)
        it, ist = self.room_ids.index(target), self.room_ids.index(stand)
        tgt = hit & (v["room"] == it)
        own = hit & (v["room"] == ist)
        tex_t = tgt & (w > 0)
        dep = v["depth"][tex_t]
        spread = float(np.percentile(dep, 90) - np.percentile(dep, 10)) if tex_t.sum() >= 20 else 0.0
        door = hit & np.isin(v["cls"], self.door_codes)
        near = hit & (v["t"] < NEAR_M)
        other = hit & (v["room"] != it) & (v["room"] != ist) & (v["room"] >= 0)
        return dict(tex_target=float(w[tgt].sum() / n), tex_own=float(w[own].sum() / n),
                    tex_other=float(w[other].sum() / n), target=float(tgt.mean()), spread_m=round(spread, 2),
                    door=float(door.mean()), near=float(near.mean()), outdoors=float((~hit).mean()),
                    median_depth_target=float(np.median(v["depth"][tgt])) if tgt.any() else None)


def score(m: dict) -> float:
    """The task's objective: textured surfaces of the room on the other side of the opening, more when they spread in
    depth (better-conditioned matches and PnP); door frames beyond 12% of the frame, close-ups and the outdoors
    (pure-rotation matches through windows, D-081) count against it."""
    return (m["tex_target"] * (1.0 + min(m["spread_m"], 3.0) / 3.0) - 1.0 * max(0.0, m["door"] - 0.12)
            - 1.0 * max(0.0, m["near"] - 0.03) - 0.3 * m["outdoors"])


def door_frame(d: dict, gt_rooms: dict, into: str):
    """Door centre, unit normal pointing INTO room `into`, unit tangent (from the room outlines, as door_geometry)."""
    from matplotlib.path import Path as MPath
    c = np.asarray(d["center"][:2], float)
    n = np.array([0.0, 1.0]) if d["axis"] == "x" else np.array([1.0, 0.0])
    P = MPath(np.asarray(gt_rooms[into]["polygon"]))
    if not P.contains_point(c + 0.6 * n):
        n = -n
    return c, n, np.array([-n[1], n[0]])


def standing_spots(sv: SceneView, room: str, step=0.10) -> np.ndarray:
    """Spots in `room` where a person stands: strict walkable cells (clear of furniture at body height, D-064), on a
    10 cm grid, with the camera >= MIN_CLEAR + 5 cm from every surface at chest height (the aim error moves it ~5 cm)."""
    wk = sv.wk
    ii, jj = np.nonzero(wk.strict)
    pts = np.stack([wk.origin[0] + jj * wk.res, wk.origin[1] + ii * wk.res], 1)
    _, first = np.unique(np.round(pts / step).astype(int), axis=0, return_index=True)
    pts = pts[first]
    pts = pts[sv.room_of(pts) == sv.room_ids.index(room)]
    return pts[sv.clear.dist(np.c_[pts, np.full(len(pts), Z_CAM)]) >= MIN_CLEAR + 0.05]


def rank(cands: list[dict]) -> list[dict]:
    """Best first: the task's objective (score) decides; among views within 80% of the best one that look through
    the opening (>= 15% of the frame in the room behind it), the photo's own side of the door breaks the tie
    (textured objects of the room it stands in). Why: a door photo goes in the folder of the room it looks into, so
    door_stitch can only link it to the room it stands in through what it sees of that room (its PnP points must lie
    in the other room, D-081)."""
    if not cands:
        return cands
    through = [c for c in cands if c["m"]["target"] >= 0.15] or cands
    top = max(c["score"] for c in through)
    good = [c for c in through if c["score"] >= 0.8 * top - 1e-9]
    rest = [c for c in cands if not any(c is g for g in good)]
    good.sort(key=lambda c: -(c["score"] + 2.0 * top * c["m"]["tex_own"]))   # 5% own side ~ 10% of the top score
    rest.sort(key=lambda c: -c["score"])
    return good + rest


def plan_view(sv: SceneView, d: dict, stand: str, target: str, kind: str, band: tuple, log=print) -> list[dict]:
    """Scored candidates for one photo through door d, best first. kind 'out': `band` = metres back from the door
    plane, heading within OUT_MAX_OFF of the door normal. kind 'in': `band` = metres from the door centre (in front
    of the door), heading within 30 deg of the door centre (the opening stays in the 67 deg frame)."""
    c, n_t, t_ = door_frame(d, sv.rooms, target)          # n_t points into the target room
    pts = standing_spots(sv, stand)
    rel = pts - c
    back, lat = -(rel @ n_t), rel @ t_
    if kind == "out":
        sel = (back >= band[0]) & (back <= band[1]) & (np.abs(lat) <= d["width"] / 2 + 0.4)
    else:
        dist = np.hypot(back, lat)
        sel = (dist >= band[0]) & (dist <= band[1]) & (back > 0.3)
    h_n = math.degrees(math.atan2(n_t[1], n_t[0]))
    out = []
    for p, b_ in zip(pts[sel], back[sel]):
        if kind == "out":
            heads = h_n + np.arange(-OUT_MAX_OFF, OUT_MAX_OFF + 0.1, 2.5)
        else:
            heads = math.degrees(math.atan2(c[1] - p[1], c[0] - p[0])) + np.arange(-30.0, 30.1, 2.5)
        for h in heads:
            pose = [float(p[0]), float(p[1]), Z_CAM, float(wrap(h)), PITCH, 0.0]
            m = sv.metrics(sv.view(pose), stand, target)
            out.append(dict(pose=pose, score=score(m), m=m, off_normal=float(wrap(h - h_n)), back_m=float(b_),
                            dist_door_m=float(np.linalg.norm(p - c))))
    out = rank(out)
    if out:
        b = out[0]
        log(f"[door] {d['id']} {kind} {stand} -> {target} band {band}: {int(sel.sum())} spots, {len(out)} views; best "
            f"xy ({b['pose'][0]:.2f}, {b['pose'][1]:.2f}) yaw {b['pose'][3]:.0f} (normal {b['off_normal']:+.0f}), "
            f"textured {b['m']['tex_target']:.3f} of the frame, spread {b['m']['spread_m']} m, own side "
            f"{b['m']['tex_own']:.3f}, door frame {b['m']['door']:.2f}")
    else:
        log(f"[door] {d['id']} {kind} {stand} -> {target} band {band}: no standing spot")
    return out


def humanise(sv: SceneView, cand: dict, stand: str, target: str, rng, tries=20) -> dict:
    """The person's aim error (rig.humanize_photo, level 1: 5 cm, 4 deg yaw/tilt, 1.5 deg roll) on the planned pose,
    checked like humanize_photos (camera >= MIN_CLEAR from surfaces, else the planned spot) and by what it shows:
    a shot that loses > 30% of the textured view, or gains door frame or close-ups, is retaken (a person checks the
    screen). Returns the kept pose, its metrics and the number of shots taken."""
    x, y, _, yaw, _, _ = cand["pose"]
    nom_pose = [x, y, Z_CAM + float(rng.normal(0, 0.05)), yaw, PITCH + float(rng.normal(0, 2.0)),
                float(rng.normal(0, 1.0))]                 # height and tilt vary shot to shot (protocol_v2.spin)
    m0 = cand["m"]
    best = None
    for k in range(1, tries + 1):
        hp = humanize_photo(nom_pose, level=1.0, rng=rng)
        if sv.clear.dist(np.array([hp[:3]]))[0] < MIN_CLEAR:
            hp[:3] = nom_pose[:3]
        m = sv.metrics(sv.view(hp), stand, target)
        ok = (m["tex_target"] >= 0.7 * m0["tex_target"] and m["near"] <= max(0.03, m0["near"] + 0.01)
              and m["door"] <= max(0.15, m0["door"] + 0.03) and abs(hp[2] - Z_CAM) <= 0.15)
        if best is None or score(m) > score(best[1]):
            best = (hp, m)
        if ok:
            return dict(pose=[round(float(v), 5) for v in hp], m=m, shots=k, nominal=[round(float(v), 5) for v in nom_pose])
    return dict(pose=[round(float(v), 5) for v in best[0]], m=best[1], shots=tries, nominal=[round(float(v), 5) for v in nom_pose])


# ------------------------------------------------------------------------------------------------------- plan
SPEC = dict(out=OUT_BACK, inn=IN_BACK)
NEAR_IN = (0.4, 1.2)            # inward spot when the spec band cannot see the room's textured objects (an alcove)
MIN_TEX_IN = 0.05               # ... i.e. when its best view has < 5% of the frame on them


def _load(session_dir: Path):
    sess = json.loads((session_dir / "session.json").read_text())
    gt_dir = Path(sess["scene"]).parent / ".simcache" / "gt"
    gt = json.loads((gt_dir / "sim_gt.json").read_text())
    return sess, gt, gt_dir


def short(rid: str) -> str:
    return rid[3:] if rid[:2].isdigit() and rid[2] == "_" else rid


def plan(a):
    sess, gt, gt_dir = _load(a.session)
    sv = SceneView(sess["scene"], gt, gt_dir)
    rooms = sv.rooms
    rng = np.random.default_rng(a.seed)
    doors = [o for o in gt["openings"] if o["cls"] == "door" and len(o["room_ids"]) == 2]
    shots, report = [], []
    for d in doors:
        r0, r1 = d["room_ids"]
        small, big = (r0, r1) if rooms[r0]["area"] < rooms[r1]["area"] else (r1, r0)
        rec = dict(door=d["id"], small=small, big=big, width_m=d["width"])
        out = plan_view(sv, d, small, big, "out", SPEC["out"])
        inn = plan_view(sv, d, big, small, "in", SPEC["inn"])
        rec["out_top"] = [_brief(c) for c in out[:5]]
        rec["in_spec_top"] = [_brief(c) for c in inn[:5]]
        use_in, ref_in = (inn[0] if inn else None), None
        if not inn or inn[0]["m"]["tex_target"] < MIN_TEX_IN:
            near = plan_view(sv, d, big, small, "in", NEAR_IN)
            rec["in_near_top"] = [_brief(c) for c in near[:5]]
            if near and (not inn or near[0]["m"]["tex_target"] > inn[0]["m"]["tex_target"]):
                use_in, ref_in = near[0], (inn[0] if inn else None)
        if out:
            shots.append(dict(d=d, kind="out", stand=small, target=big, cand=out[0], band=SPEC["out"], in_set=True))
        if use_in is not None:
            shots.append(dict(d=d, kind="in", stand=big, target=small, cand=use_in,
                              band=SPEC["inn"] if use_in is (inn[0] if inn else None) else NEAR_IN, in_set=True))
        if ref_in is not None:
            shots.append(dict(d=d, kind="in", stand=big, target=small, cand=ref_in, band=SPEC["inn"], in_set=False))
        report.append(rec)
    # route: from the last photo of the session, the nearest door next; each door's photos out, then in
    last = max(sess["photos"], key=lambda p: p["t"])
    here, t = np.asarray(last["pose"][:2]), float(last["t"])
    idx = max(p["index"] for p in sess["photos"])
    by_door = {}
    for s in shots:
        by_door.setdefault(s["d"]["id"], []).append(s)
    todo, new = list(by_door), []
    while todo:
        did = min(todo, key=lambda k: np.linalg.norm(np.asarray(by_door[k][0]["cand"]["pose"][:2]) - here))
        todo.remove(did)
        for s in [x for x in by_door[did] if x["in_set"]]:
            xy = np.asarray(s["cand"]["pose"][:2])
            walk = float(np.linalg.norm(xy - here))
            if s["kind"] == "out":                  # walk to the spot, aim, check the screen: never paired with the
                t += max(16.0, walk / 0.8 + 6.0)    # previous photo (frontend pairs <= 15 s)
            else:                                   # out, then straight through the door and turn round: a pair
                t += float(np.clip(walk / 1.0 + 1.0, 2.8, 3.8))
            here = xy
            new.append((s, round(t, 2)))
    for s in [x for x in shots if not x["in_set"]]:      # spec-band references: after everything, never paired
        t += 30.0
        new.append((s, round(t, 2)))
    photos = []
    for s, t_ in new:
        idx += 1
        h = humanise(sv, s["cand"], s["stand"], s["target"], rng)
        role = f"door out: {short(s['target'])}" if s["kind"] == "out" else f"door in: from {short(s['stand'])}"
        photos.append(dict(index=idx, t=t_, folder=s["target"], lighting="day", pose=h["pose"], role=role,
                           humanized=True, clearance_m=round(float(sv.clear.dist(np.array([h["pose"][:3]]))[0]), 3),
                           f_px=F, lens="1x", door=s["d"]["id"], stand=s["stand"], in_set=s["in_set"],
                           plan=dict(band_m=list(s["band"]), back_m=round(s["cand"]["back_m"], 2),
                                     dist_door_m=round(s["cand"]["dist_door_m"], 2),
                                     off_normal_deg=round(s["cand"]["off_normal"], 1), planned_pose=s["cand"]["pose"],
                                     nominal_pose=h["nominal"], shots=h["shots"],
                                     planned=_r(s["cand"]["m"]), kept=_r(h["m"]))))
        m = h["m"]
        print(f"[door] IMG_{idx:04d} t={t_:7.2f} {photos[-1]['folder']:15s} {role:30s} {s['d']['id']} "
              f"{'set' if s['in_set'] else 'ref'} xy ({h['pose'][0]:.2f}, {h['pose'][1]:.2f}) z {h['pose'][2]:.2f} "
              f"yaw {h['pose'][3]:.1f} pitch {h['pose'][4]:.1f}  textured {m['tex_target']:.3f} own {m['tex_own']:.3f} "
              f"door {m['door']:.2f} near {m['near']:.2f} shots {h['shots']}")
    rs = a.out / "render_session"
    rs.mkdir(parents=True, exist_ok=True)
    (rs / "session.json").write_text(json.dumps(dict(version=sess.get("version", 1), scene=sess["scene"],
                                                     created=sess.get("created"), takes=[], photos=photos,
                                                     meta=dict(human_level=1.0, generator="sim/door_photos.py",
                                                               base_session=str(a.session), seed=a.seed)), indent=1))
    (a.out / "door_plan.json").write_text(json.dumps(dict(base_session=str(a.session), seed=a.seed, doors=report,
                                                          photos=photos), indent=1))
    print(f"[door] {len(photos)} photos -> {rs / 'session.json'}")


def _r(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def _brief(c: dict) -> dict:
    return dict(pose=[round(v, 3) for v in c["pose"]], score=round(c["score"], 4), back_m=round(c["back_m"], 2),
                dist_door_m=round(c["dist_door_m"], 2), off_normal_deg=round(c["off_normal"], 1), m=_r(c["m"]))



# --------------------------------------------------------------------------------------------------- assemble
def assemble(a):
    from sim import emulate as E
    sess, gt, gt_dir = _load(a.session)
    cap = a.capture or a.session / "capture_photos"
    rend = a.renders or a.session / "render_photos"
    rs = a.out / "render_session"
    rdir = rs / "render"
    plan_s = json.loads((rs / "session.json").read_text())
    new_meta = json.loads((rdir / "photos" / "photos.json").read_text())
    missing = {f"{p['index']:04d}.jpg" for p in plan_s["photos"]} - {m["file"] for m in new_meta["photos"]}
    if missing:
        raise SystemExit(f"not rendered: {sorted(missing)}")
    # 1. phone files exactly as for the old photos (emulate.write_photos, iphone15: ISP noise, JPEG q92, EXIF with
    #    DateTimeOriginal = 2026-10-04 11:00:00 + t)
    emu = rs / "capture"
    if emu.exists():
        shutil.rmtree(emu)
    E.write_photos(rdir, emu, E.PROFILES["iphone15"], np.random.default_rng(a.seed), print)
    # 2. photos/: the old room folders as used today (k65_inputs_1x: the *_take2 repeat set left out) + the new set;
    #    reference_photos/: spec-band views that were not used
    ph, ref = a.out / "photos", a.out / "reference_photos"
    for d in (ph, ref):
        if d.exists():
            shutil.rmtree(d)
    for room in sorted(x for x in cap.iterdir() if x.is_dir() and not x.name.endswith("_take2")):
        (ph / room.name).mkdir(parents=True)
        for f in sorted(room.glob("*.JPG")):
            _link(f, ph / room.name / f.name)
    for p in plan_s["photos"]:
        src = emu / "photos" / p["folder"] / f"IMG_{p['index']:04d}.JPG"
        dst = (ph if p["in_set"] else ref) / p["folder"]
        dst.mkdir(parents=True, exist_ok=True)
        _link(src, dst / src.name)
    # 3. session.json: the old session (walk and photos) + the new photos with their roles and true poses
    out_s = dict(sess)
    out_s["photos"] = sess["photos"] + plan_s["photos"]
    out_s["meta"] = dict(sess.get("meta", {}), door_photos=dict(
        generator="sim/door_photos.py", base_session=str(a.session), added=[p["index"] for p in plan_s["photos"]],
        in_photos_folder=[p["index"] for p in plan_s["photos"] if p["in_set"]],
        note="new photos: humanised true poses; in_set False = reference views kept out of photos/"))
    (a.out / "session.json").write_text(json.dumps(out_s))
    # 4. render_photos/: true poses of every photo (old renders linked, new ones copied)
    rp = a.out / "render_photos"
    if rp.exists():
        shutil.rmtree(rp)
    rp.mkdir()
    old_meta = json.loads((rend / "photos.json").read_text())
    for m in old_meta["photos"]:
        (rp / m["file"]).symlink_to((rend / m["file"]).resolve())
    for m in new_meta["photos"]:
        _link(rdir / "photos" / m["file"], rp / m["file"])
    (rp / "photos.json").write_text(json.dumps(dict(old_meta, photos=old_meta["photos"] + new_meta["photos"]), indent=1))
    if (a.out / "truth").exists():
        shutil.rmtree(a.out / "truth")
    shutil.copytree(emu / "truth", a.out / "truth")
    # 5. what the photo tier will make of the times: doorway pairs as frontend.doorway_pairs forms them
    _check_pairs(ph, a.out / "door_pairs.json")
    if a.sheet:
        contact_sheet(a, plan_s["photos"], ph, ref, gt)


def _link(src: Path, dst: Path):
    """Hard link (same bytes, no extra disk: the files are never edited in place), a copy across file systems."""
    import os
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _check_pairs(ph: Path, out: Path):
    from types import SimpleNamespace
    from PIL import Image
    from floorplan.photo.images import capture_time
    from floorplan.photo.params import PhotoParams
    from floorplan.photo.protocol import doorway_pairs, shot_rhythm
    photos = []
    for f in sorted(ph.glob("*/*.JPG")):
        photos.append(SimpleNamespace(name=f"{f.parent.name}/{f.name}", room=f.parent.name,
                                      time_s=capture_time(Image.open(f).getexif())))
    p = PhotoParams()
    pairs = doorway_pairs(photos, p.doorway_pair_max_dt_s, p.doorway_pair_rhythm)
    r = dict(rhythm_s=shot_rhythm(photos), pairs=pairs)
    out.write_text(json.dumps(r, indent=1))
    print(f"[door] photo tier's doorway pairs on {ph}: rhythm {r['rhythm_s']:.2f} s, "
          + "; ".join(f"{q['a']}~{q['b']} ({q['dt_s']} s <= {q['limit_s']} s)" for q in pairs))


def contact_sheet(a, photos: list, ph: Path, ref: Path, gt: dict, tw=500, th=375, hdr=58):
    from PIL import Image, ImageDraw, ImageFont
    fnt = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)  # noqa: E731
    fb = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)  # noqa: E731
    order = [p for p in photos if p["in_set"]] + [p for p in photos if not p["in_set"]]
    cols, rows = 4, (len(order) + 3) // 4
    mapw = 520
    sheet = Image.new("RGB", (cols * tw + mapw, rows * (th + hdr)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, p in enumerate(order):
        f = (ph if p["in_set"] else ref) / p["folder"] / f"IMG_{p['index']:04d}.JPG"
        im = Image.open(f).convert("RGB").resize((tw, th))
        x0, y0 = (k % cols) * tw, (k // cols) * (th + hdr)
        sheet.paste(im, (x0, y0 + hdr))
        dr.rectangle([x0, y0, x0 + tw - 1, y0 + hdr - 1], fill=(0, 0, 0))
        m, pl = p["plan"]["kept"], p["plan"]
        tag = "" if p["in_set"] else "  [reference, not in photos/]"
        dr.text((x0 + 4, y0 + 2), f"{p['folder']}/IMG_{p['index']:04d}.JPG{tag}", fill=(255, 215, 0), font=fb(13))
        door = next(o for o in gt["openings"] if o["id"] == p["door"])
        area = {r["id"]: r["area"] for r in gt["rooms"]}
        small = min(door["room_ids"], key=lambda r: area[r])
        dr.text((x0 + 4, y0 + 20), f"{p['role']} ({short(small)} door), t={p['t']:.1f} s",
                fill=(120, 220, 255), font=fnt(12))
        x, y, z, yaw, pitch, _ = p["pose"]
        dr.text((x0 + 4, y0 + 37), f"xy {x:.2f},{y:.2f} z {z:.2f} yaw {yaw:.0f} pitch {pitch:.0f} | {pl['dist_door_m']:.1f} m "
                f"from door | textured {100 * m['tex_target']:.0f}%, own side {100 * m['tex_own']:.0f}%",
                fill=(255, 255, 255), font=fnt(11))
    _map(sheet, cols * tw, 0, mapw, rows * (th + hdr), photos, gt, fnt, fb)
    a.sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.sheet, quality=90)
    print(f"[door] contact sheet -> {a.sheet}")


def _map(sheet, X0, Y0, Wm, Hm, photos, gt, fnt, fb):
    """Top view: room outlines, doors, the new cameras with their 67 deg view wedges (north up)."""
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    allp = np.concatenate([np.asarray(r["polygon"]) for r in gt["rooms"]])
    lo, hi = allp.min(0) - 0.3, allp.max(0) + 0.3
    s = min((Wm - 20) / (hi[0] - lo[0]), (Hm - 60) / (hi[1] - lo[1]))
    ox, oy = X0 + 10, Y0 + 40
    P = lambda q: (ox + (q[0] - lo[0]) * s, oy + (hi[1] - q[1]) * s)  # noqa: E731
    dr.text((X0 + 8, Y0 + 8), "new photos, top view (north up)", fill="black", font=fb(15))
    for r in gt["rooms"]:
        pts = [P(q) for q in r["polygon"]]
        dr.polygon(pts, outline=(60, 60, 60), fill=(245, 245, 240))
        c = np.asarray(r["polygon"]).mean(0)
        dr.text(P(c), short(r["id"]), fill=(90, 90, 90), font=fnt(12), anchor="mm")
    for o in gt["openings"]:
        lo_, hi_ = np.asarray(o["box_min"][:2]), np.asarray(o["box_max"][:2])
        col = (200, 0, 0) if o["cls"] == "door" else (80, 160, 255)
        dr.rectangle([P((lo_[0], hi_[1])), P((hi_[0], lo_[1]))], fill=col)
    for p in photos:
        x, y, _, yaw, _, _ = p["pose"]
        col = ((0, 120, 0) if p["role"].startswith("door out") else (180, 0, 160)) if p["in_set"] else (150, 150, 150)
        rng_ = 1.6
        a0, a1 = math.radians(yaw - HFOV / 2), math.radians(yaw + HFOV / 2)
        wedge = [P((x, y)), P((x + rng_ * math.cos(a0), y + rng_ * math.sin(a0))),
                 P((x + rng_ * math.cos(a1), y + rng_ * math.sin(a1)))]
        dr.polygon(wedge, outline=col)
        cx, cy = P((x, y))
        dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=col)
        dr.text((cx + 6, cy - 6), f"{p['index']}", fill=col, font=fb(12))
    yb = Y0 + Hm - 18
    dr.text((X0 + 8, yb), "green: door out   magenta: door in   grey: reference", fill="black", font=fnt(12))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "assemble"):
        s = sub.add_parser(name)
        s.add_argument("--session", type=Path, required=True, help="the photo session the door photos are added to")
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--seed", type=int, default=65)
        if name == "assemble":
            s.add_argument("--capture", type=Path, help="the session's emulated photos (default <session>/capture_photos)")
            s.add_argument("--renders", type=Path, help="the session's renders (default <session>/render_photos)")
            s.add_argument("--sheet", type=Path, help="contact sheet of the new photos (JPEG)")
    a = ap.parse_args()
    {"plan": plan, "assemble": assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
