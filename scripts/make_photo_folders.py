#!/usr/bin/env python
"""Derive photo-tier inputs (one folder of 2-8 stills per room) from a Stray Scanner capture (D-014, D-017).

Why: the case study's photo tier is "2 to 8 stills per room, no depth, no poses". The sample data has no native photos,
and the recruiters confirmed they have none. To run the photo tier on the SAME rooms as the LiDAR tier (so the LiDAR
plan can serve as a reference), we simulate a person following docs/CAPTURE_PROTOCOL.md by picking video frames.

Two protocols (--protocol):
  * corners (v1, D-014): per room a greedy wall set-cover of frames + one "looking out" frame per doorway.
  * rotate  (v2, D-017, default): per room 4-6 frames from the MOST CENTRAL cluster of camera positions with headings
    spread ~60 deg (a person turning on the spot), plus DOORWAY PAIRS: two frames near a door centre, one looking into
    each of the two rooms, each filed under the room it looks into. A walk-through video rarely contains a true spin
    or a true doorway pair, so the simulator takes the closest available frames and RECORDS how far from the ideal they
    are (distance of the spin cluster from the room centre, spread of the cluster, true distance between the two
    frames of a pair, their real time gap).

EXIF written into every JPEG (as a phone would): FocalLengthIn35mmFilm, Make, DateTimeOriginal + SubSecTimeOriginal.
  * corners: capture times from the video timestamps.
  * rotate: the times follow the PROTOCOL order (spin shots 4 s apart in clockwise heading order, then the doorway
    pairs 3 s apart), because the frames were picked from different moments of the video. This is a simulation of the
    protocol's timing and is DISCLOSED in the truth file (`exif_time_simulated: true`, real gaps listed).
The simulator uses the LiDAR plan (room polygons, doors) and the true camera poses ONLY to choose frames. Poses, depth
and intrinsics go to a separate truth file for evaluation and are never read by the photo-tier pipeline.

Usage: python scripts/make_photo_folders.py <capture_dir> <plan.json> <scene_dir> [--protocol rotate|corners]
                                            [--out outputs/photo_inputs]
Output: <out>/<capture>__<protocol>/{photos/<room>/*.jpg, truth/photo_truth.json}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.io.stray import load_stray  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402

MAX_PER_ROOM, MIN_PER_ROOM = 8, 2
HFOV_MARGIN_DEG = 5.0
SPIN_STEP_DEG = 60.0            # protocol: about a sixth of a turn between photos
SPIN_RADIUS_M = 0.40            # frames within this radius count as "the same spot" (body sway + arm length)
SPIN_MIN, SPIN_MAX = 4, 6
DOOR_RADIUS_M = 0.8             # doorway-pair frames: within this distance of the door centre
DOOR_LOOK_DEG = 50.0            # "looking into room X": forward within 50 deg of the direction to X's centre
T0 = dt.datetime(2026, 10, 3, 8, 0, 0)


# ----------------------------------------------------------------------------------------------- frame geometry
def frame_geometry(cap, T_align):
    """Camera centre (u, v), forward direction in plan, pitch (deg) and the 90-deg rotation that makes images upright."""
    R_al = T_align[:3, :3]
    T = cap.T_wc
    C = T[:, :3, 3] @ R_al.T + T_align[:3, 3]
    fwd = (T[:, :3, :3] @ np.array([0, 0, 1.0])) @ R_al.T          # camera +z (OpenCV) in aligned world
    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 1], -1, 1)))
    f2 = fwd[:, [0, 2]]
    f2 /= np.maximum(np.linalg.norm(f2, axis=1, keepdims=True), 1e-9)
    up_cam = np.einsum("nji,j->ni", T[:, :3, :3], np.array([0, 1.0, 0]))   # world up expressed in camera axes
    # image "up" is -y in OpenCV pixel axes; the angle of world-up in the image plane tells the needed rotation
    ang = np.degrees(np.arctan2(up_cam[:, 0], -up_cam[:, 1]))
    rot90 = (np.round(ang / 90.0).astype(int)) % 4
    return C[:, [0, 2]], f2, pitch, rot90


def sharpness(bgr):
    g = cv2.cvtColor(cv2.resize(bgr, (480, 360)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def wall_samples(poly: Polygon, step=0.25):
    pts, wid = [], []
    coords = list(poly.exterior.coords)
    for k in range(len(coords) - 1):
        a, b = np.array(coords[k]), np.array(coords[k + 1])
        n = max(2, int(np.linalg.norm(b - a) / step))
        for t in np.linspace(0.05, 0.95, n):
            pts.append(a + t * (b - a)); wid.append(k)
    return np.array(pts), np.array(wid)


def visible_walls(c, f, hfov_deg, wp, wid, max_range=6.0):
    d = wp - c
    dist = np.linalg.norm(d, axis=1)
    cosang = (d @ f) / np.maximum(dist, 1e-9)
    ok = (cosang > np.cos(np.radians(hfov_deg / 2 - HFOV_MARGIN_DEG))) & (dist < max_range) & (dist > 0.3)
    return set(wid[ok].tolist())


def heading_deg(f2: np.ndarray) -> np.ndarray:
    """Heading in the plan, atan2(v, u). Drawn with v DOWN (I-001, the true top view), increasing = clockwise."""
    return np.degrees(np.arctan2(f2[..., 1], f2[..., 0])) % 360


def ang_diff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def room_candidates(poly: Polygon, C, pitch, cap_n: int = 600) -> np.ndarray:
    inside = np.array([poly.contains(Point(c)) for c in C])
    cand = np.where(inside & (np.abs(pitch) < 30))[0]                 # protocol: chest height, roughly level
    return cand[:: max(1, len(cand) // cap_n)]


# ------------------------------------------------------------------------------------------ protocol: corners
def select_corners(poly, cand, C, F, hfov, sharp, times, doors, rid):
    """v1 protocol: greedy wall set cover (prefer sharp frames, 1 s apart) + one 'looking out' frame per door."""
    s_med = np.median([sharp[i] for i in cand])
    wp, wid = wall_samples(poly)
    vis = {i: visible_walls(C[i], F[i], hfov[i], wp, wid) for i in cand}
    chosen, covered, all_walls = [], set(), set(wid.tolist())
    while covered != all_walls and len(chosen) < MAX_PER_ROOM - 1:
        best, best_gain = None, 0.0
        for i in cand:
            if any(abs(times[i] - times[j]) < 1.0 for j in chosen):
                continue
            gain = len(vis[i] - covered) * min(1.5, sharp[i] / s_med)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            break
        chosen.append(best); covered |= vis[best]
    for d in doors:
        if rid not in d.get("room_ids", []) or len(chosen) >= MAX_PER_ROOM:
            continue
        dc = np.array(d["center"])
        near = [i for i in cand if np.linalg.norm(C[i] - dc) < 1.2
                and np.dot(F[i], (dc - C[i]) / max(np.linalg.norm(dc - C[i]), 1e-6)) > np.cos(np.radians(35))
                and all(abs(times[i] - times[j]) >= 1.0 for j in chosen)]
        if near:
            chosen.append(max(near, key=lambda i: sharp[i]))
    for i in sorted(cand, key=lambda i: -sharp[i]):
        if len(chosen) >= MIN_PER_ROOM:
            break
        if all(abs(times[i] - times[j]) >= 1.0 for j in chosen):
            chosen.append(i)
    return dict(frames=sorted(int(i) for i in chosen), walls_covered=f"{len(covered)}/{len(all_walls)}",
                roles={int(i): "corner_or_doorway_lookout" for i in chosen})


# ------------------------------------------------------------------------------------------- protocol: rotate
def pick_headings(members, H, sharp, n_max):
    """From frames at one spot, pick up to n_max with headings ~SPIN_STEP_DEG apart, as ONE contiguous turn.

    For 12 start headings, fills the six 60-deg targets with the sharpest member within +-20 deg, then keeps the
    longest cyclically CONTIGUOUS run of filled targets (a real spin has no skipped step: a skipped sector would
    make a 120-deg step, which the protocol never produces). Returned in clockwise order (increasing heading)."""
    best, best_key = [], (-1, 0.0)
    for h0 in np.arange(0, 60, 5.0):
        slot = []
        for k in range(6):
            tgt = (h0 + k * SPIN_STEP_DEG) % 360
            ok = [i for i in members if ang_diff(H[i], tgt) < 20]
            slot.append(max(ok, key=lambda i: sharp.get(i, 0.0) - 50 * ang_diff(H[i], tgt)) if ok else None)
        filled = [s is not None for s in slot]
        if all(filled):
            run = list(range(6))
        else:
            run = []
            for st in range(6):                       # longest run starting right after an empty slot
                if filled[st] and not filled[st - 1]:
                    r = []
                    k = st
                    while filled[k % 6] and len(r) < 6:
                        r.append(k % 6); k += 1
                    run = r if len(r) > len(run) else run
        pick = [slot[k] for k in run]
        err = sum(float(ang_diff(H[i], (h0 + k * SPIN_STEP_DEG) % 360)) for i, k in zip(pick, run))
        key = (len(pick), -err)
        if key > best_key:
            best, best_key = pick, key
    return best[:n_max]


def select_spin(poly, cand, C, H, sharp, n_max):
    """Most central cluster of camera positions that still offers >= SPIN_MIN headings ~60 deg apart.

    A walk-through rarely stands in the exact centre; we take the cluster that minimises
    distance-to-centre + 0.6 m per missing heading sector, and record that distance."""
    centre = np.array(poly.centroid.coords[0]) if poly.contains(poly.centroid) else \
        np.array(poly.representative_point().coords[0])
    seeds = cand[:: max(1, len(cand) // 200)]
    best, best_cost = None, np.inf
    for s in seeds:
        members = [i for i in cand if np.linalg.norm(C[i] - C[s]) < SPIN_RADIUS_M]
        pick = pick_headings(members, H, sharp, n_max)
        if len(pick) < 2:
            continue
        cost = np.linalg.norm(C[pick].mean(0) - centre) + 0.6 * (6 - len(pick))
        if cost < best_cost:
            best, best_cost = pick, cost
    if best is None:
        return [], {}
    P = C[best]
    stats = dict(n=len(best), dist_from_room_centre_m=round(float(np.linalg.norm(P.mean(0) - centre)), 2),
                 spread_max_m=round(float(max(np.linalg.norm(P[a] - P[b]) for a in range(len(P))
                                              for b in range(len(P)))), 2) if len(P) > 1 else 0.0,
                 headings_deg=[round(float(H[i]), 1) for i in best], room_centre=centre.round(2).tolist())
    return best, stats


def select_doorway_pair(d, polys, cand_all, C, H, sharp):
    """Two frames near the door centre: one looking into each room; minimise the distance between them."""
    a, b = d["room_ids"][:2]
    dc = np.array(d["center"])
    near = cand_all[np.linalg.norm(C[cand_all] - dc, axis=1) < DOOR_RADIUS_M]
    looks = {}
    for r in (a, b):
        tgt = np.array(polys[r].representative_point().coords[0])
        if polys[r].contains(polys[r].centroid):
            tgt = np.array(polys[r].centroid.coords[0])
        dirs = (tgt - C[near]) / np.maximum(np.linalg.norm(tgt - C[near], axis=1, keepdims=True), 1e-6)
        hd = heading_deg(dirs)
        looks[r] = near[ang_diff(H[near], hd) < DOOR_LOOK_DEG]
    if not len(looks[a]) or not len(looks[b]):
        return None
    D = np.linalg.norm(C[looks[a]][:, None] - C[looks[b]][None], axis=2)
    # "turn round": the two photos must look roughly opposite ways (and be two different frames)
    opp = ang_diff(H[looks[a]][:, None], H[looks[b]][None]) > 120
    D = np.where(opp & (looks[a][:, None] != looks[b][None]), D, np.inf)
    if not np.isfinite(D).any():
        return None
    score = D - 0.0005 * np.minimum(np.add.outer([sharp.get(i, 0.0) for i in looks[a]], [sharp.get(j, 0.0) for j in looks[b]]), 400)
    ia, ib = np.unravel_index(np.argmin(score), score.shape)
    return {a: int(looks[a][ia]), b: int(looks[b][ib]), "dist_m": round(float(D[ia, ib]), 2)}


def select_rotate(plan, C, F, pitch, sharp_of, times):
    """Per room: spin cluster (4-6 frames) + doorway pairs, max 8 photos per room folder."""
    polys = {r["id"]: Polygon(r["polygon"]) for r in plan["rooms"]}
    polys = {k: p for k, p in polys.items() if p.is_valid and p.area >= 1.0}
    H = heading_deg(F)
    doors = [o for o in plan.get("openings", []) if o.get("kind") in ("door", "passage")
             and len(o.get("room_ids", [])) == 2 and all(r in polys for r in o["room_ids"])]
    level = np.where(np.abs(pitch) < 30)[0]
    cand_all = level[:: max(1, len(level) // 4000)]
    cands = {rid: room_candidates(p, C, pitch) for rid, p in polys.items()}
    sharp = sharp_of(sorted(set(np.concatenate([cand_all, *cands.values()]).tolist())))
    pairs = []
    for d in doors:
        pr = select_doorway_pair(d, polys, cand_all, C, H, sharp)
        if pr is not None:
            pairs.append(dict(door=d["id"], **pr))
    sel = {}
    for rid, poly in polys.items():
        n_door = sum(rid in p for p in pairs)
        spin, stats = select_spin(poly, cands[rid], C, H, sharp, max(SPIN_MIN, min(SPIN_MAX, MAX_PER_ROOM - n_door)))
        roles = {int(i): "spin" for i in spin}
        sel[rid] = dict(frames=[int(i) for i in spin], spin=stats, roles=roles)
    for p in pairs:                       # doorway photos, within the 8-per-room budget (spin keeps >= SPIN_MIN)
        rooms = [k for k in p if k in sel]
        if any(len(sel[r]["frames"]) >= MAX_PER_ROOM for r in rooms):
            p["skipped"] = "room folder full"
            continue
        if any(p[r] in sel[r]["frames"] for r in rooms):
            p["skipped"] = "frame already used in this room"
            continue
        for r in rooms:
            sel[r]["frames"].append(p[r]); sel[r]["roles"][p[r]] = f"doorway_pair:{p['door']}"
    return {r: s for r, s in sel.items() if len(s["frames"]) >= MIN_PER_ROOM}, pairs


def protocol_times(sel, pairs, real_times):
    """EXIF capture times in PROTOCOL order (simulated, disclosed): per room the spin shots 4 s apart in clockwise
    heading order; when leaving a room, its doorway pairs: photo back into this room, 3 s later the photo into the
    next room; 8 s walk between steps. Returns {frame_index_in_room_key: seconds since T0}."""
    t, out = 0.0, {}
    done = set()
    for rid in sel:
        spin = [i for i in sel[rid]["frames"] if sel[rid]["roles"][i] == "spin"]
        for i in spin:                                   # already in clockwise heading order (pick_headings)
            out[(rid, i)] = t; t += 4.0
        for p in pairs:
            if "skipped" in p or p["door"] in done or rid not in p:
                continue
            other = next(k for k in p if k not in ("door", "dist_m", rid))
            t += 8.0
            out[(rid, p[rid])] = t
            out[(other, p[other])] = t + 3.0
            done.add(p["door"]); t += 3.0
        t += 8.0
    return out


# --------------------------------------------------------------------------------------------------- writing
def exif_for(f35: int, t_sec: float) -> Image.Exif:
    exif = Image.Exif()
    exif[0xA405] = f35                                   # FocalLengthIn35mmFilm (also in the Exif IFD below)
    exif[0x010F] = "simulated-from-stray-scanner"        # Make: makes the provenance explicit
    ifd = exif.get_ifd(0x8769)
    ts = T0 + dt.timedelta(seconds=float(t_sec))
    ifd[0x9003] = ts.strftime("%Y:%m:%d %H:%M:%S")       # DateTimeOriginal
    ifd[0x9291] = f"{ts.microsecond // 10000:02d}"      # SubSecTimeOriginal (centiseconds)
    ifd[0xA405] = f35
    return exif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("plan", type=Path)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--protocol", choices=["rotate", "corners"], default="rotate")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "photo_inputs")
    a = ap.parse_args()

    cap = load_stray(a.capture)
    scene, _ = load_scene(a.scene_dir)
    plan = json.loads(a.plan.read_text())
    C, F, pitch, rot90 = frame_geometry(cap, scene["T_align"])
    W, H = cap.rgb_size
    times = np.asarray(cap.timestamps, float)
    cap_name = f"{a.capture.parent.name}__{a.capture.name}"
    out = a.out / f"{cap_name}__{a.protocol}"
    photos_dir, truth_dir = out / "photos", out / "truth"
    photos_dir.mkdir(parents=True, exist_ok=True); truth_dir.mkdir(parents=True, exist_ok=True)
    hfov_of = lambda i: float(np.degrees(2 * np.arctan(min(W, H) / (2 * cap.K_rgb[i][0, 0]))))  # upright: short side

    def sharp_of(idx):
        return {i: sharpness(bgr) for i, bgr in cap.iter_rgb(idx)}

    pairs = []
    if a.protocol == "corners":
        doors = [o for o in plan.get("openings", []) if o.get("kind") in ("door", "passage")]
        selection = {}
        for room in plan["rooms"]:
            poly = Polygon(room["polygon"])
            if not poly.is_valid or poly.area < 1.0:
                continue
            cand = room_candidates(poly, C, pitch, 400)
            if len(cand) < MIN_PER_ROOM:
                continue
            sharp = sharp_of(cand)
            selection[room["id"]] = select_corners(poly, cand, C, F, {i: hfov_of(i) for i in cand}, sharp, times,
                                                   doors, room["id"])
        t_exif = {(r, i): times[i] - times[0] for r, s in selection.items() for i in s["frames"]}
    else:
        selection, pairs = select_rotate(plan, C, F, pitch, sharp_of, times)
        t_exif = protocol_times(selection, pairs, times)
        for p in pairs:
            ks = [k for k in p if k not in ("door", "dist_m", "skipped")]
            p["real_time_gap_s"] = round(float(abs(times[p[ks[0]]] - times[p[ks[1]]])), 1)

    wanted = sorted({i for s in selection.values() for i in s["frames"]})
    frames = dict(cap.iter_rgb(wanted))
    truth = {}
    for rid, s in selection.items():
        rdir = photos_dir / rid
        rdir.mkdir(exist_ok=True)
        order = sorted(s["frames"], key=lambda i: t_exif[(rid, i)])
        for k, i in enumerate(order):
            img = frames[i]
            r = int(rot90[i])
            if r:
                img = np.rot90(img, k=r).copy()          # counter-clockwise quarter turns
            fx = cap.K_rgb[i][0, 0]
            # 35 mm equivalent focal as phones write it: the standard DIAGONAL definition (43.27 mm), integer (D-048).
            # v1 wrote fx / long side * 36 mm, which matched the v1 reader's rule and hid issue I-009.
            f35 = int(round(fx * 43.27 / float(np.hypot(W, H))))
            name = f"{rid}_{k + 1:02d}.jpg"
            Image.fromarray(img[:, :, ::-1]).save(rdir / name, quality=95, exif=exif_for(f35, t_exif[(rid, i)]))
            truth[f"{rid}/{name}"] = dict(frame=int(i), rot90=r, T_wc=cap.T_wc[i].tolist(),
                                          K_rgb=cap.K_rgb[i].tolist(), f35_written=f35,
                                          role=s["roles"].get(i, "unknown"),
                                          exif_time_s=round(float(t_exif[(rid, i)]), 2),
                                          video_time_s=round(float(times[i] - times[0]), 2))
    for s in selection.values():
        s["roles"] = {str(k): v for k, v in s["roles"].items()}
    meta = dict(capture=str(a.capture), capture_name=cap_name, plan=str(a.plan), protocol=a.protocol,
                selection=selection, doorway_pairs=pairs, photos=truth,
                exif_time_simulated=a.protocol == "rotate",
                disclosure=("rotate: frames come from a walk-through video, not a real spin or doorway pair. EXIF "
                            "times are written in protocol order (simulated); the real time gap and the true distance "
                            "between the two frames of every doorway pair are in doorway_pairs; the distance of each "
                            "spin cluster from the room centre is in selection[room].spin.")
                if a.protocol == "rotate" else "corners: EXIF times are the real video timestamps.")
    (truth_dir / "photo_truth.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(dict(selection={r: dict(n=len(s["frames"]), spin=s.get("spin")) for r, s in selection.items()},
                          doorway_pairs=pairs), indent=1))
    print(f"wrote {len(truth)} photos in {len(selection)} room folders to {photos_dir}")


if __name__ == "__main__":
    main()
