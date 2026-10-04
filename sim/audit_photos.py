#!/usr/bin/env python
"""Audit a simulated photo set against exact ground truth BEFORE any pipeline runs on it (user, 4 Oct 18:40: "first
check if the data we have is accurate").

For every photo (true pose, true intrinsics) rays are cast against the scene mesh:
  - median depth and close-up share (pixels nearer than 0.8 m): a turning photo of a blank wall at 0.4 m is useless;
  - wall coverage: sample points on the photo's own room walls (10 cm along each wall, heights 1.0-2.0 m, the band
    the photo tier fits walls in) count as SEEN when they project into the frame and the ray reaches them unoccluded.
Per room: share of each GT wall seen by at least one photo, the turning spot's clearance, and a list of problems.

Usage: python sim/audit_photos.py outputs/sim/k65v2 [outputs/sim/k22v2 ...]   -> <dataset>/verify/photo_audit.md/.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import scene_geom as sg  # noqa: E402

W, H = 4032, 3024
STEP_PX = 48                       # ray grid for depth statistics (84 x 63 rays per photo)


def wall_samples(poly: np.ndarray, step=0.10, heights=(1.0, 1.25, 1.5, 1.75, 2.0)):
    """Points on each wall (polygon edge) of a room, slightly inside the room; returns (pts (n,3), wall id (n,))."""
    pts, wid = [], []
    ccw = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) > 0
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        L = float(np.linalg.norm(b - a))
        if L < 0.05:
            continue
        d = (b - a) / L
        n_in = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])
        for s in np.arange(step / 2, L, step):
            q = a + s * d + 0.02 * n_in
            for z in heights:
                pts.append([q[0], q[1], z])
                wid.append(k)
    return np.asarray(pts), np.asarray(wid)


def main(datasets):
    for ds in map(Path, datasets):
        cap = ds / "capture_iphone15"
        gt = json.load(open(cap / "gt" / "sim_gt.json"))
        scene = json.load(open(cap / "capture.json"))["scene"]
        geom = sg.load(scene)
        rc, ids = sg.raycaster(geom, sg.mesh_mask(geom, lambda m: not m["glass"]))
        truth = {p["file"]: p for p in json.load(open(cap / "truth" / "photos_true.json"))}
        sess = json.load(open(ds / "session.json"))
        role = {f"{p['folder']}/IMG_{p['index']:04d}.JPG": p.get("role", "") for p in sess["photos"]}
        rooms = {r["id"]: r for r in gt["rooms"]}
        samp = {rid: wall_samples(np.asarray(r["polygon"])) for rid, r in rooms.items()}
        seen = {rid: np.zeros(len(samp[rid][0]), bool) for rid in rooms}
        rows = []
        vv, uu = np.mgrid[STEP_PX // 2:H:STEP_PX, STEP_PX // 2:W:STEP_PX]
        for f, t in sorted(truth.items()):
            folder = f.split("/")[0]
            rid = folder.replace("_take2", "")
            T = np.asarray(t["T_usd"]); R = T[:3, :3] @ np.diag([1, -1, -1]); o = T[:3, 3]
            fpx = float(t["f_px_true"])
            d_cam = np.stack([(uu - (W / 2 - 0.5)) / fpx, (vv - (H / 2 - 0.5)) / fpx, np.ones_like(uu, float)], -1)
            d_cam = d_cam.reshape(-1, 3)
            dirs = d_cam @ R.T
            n = np.linalg.norm(dirs, axis=1, keepdims=True)
            th, _, _ = sg.cast(rc, np.repeat(o[None], len(dirs), 0), dirs / n)
            z = th / n[:, 0]
            zf = z[np.isfinite(z)]
            med = float(np.median(zf)) if len(zf) else float("nan")
            near = float(np.mean(zf < 0.8)) if len(zf) else 1.0
            # wall coverage of the photo's own room
            P, wid = samp[rid]
            pc = (P - o) @ R
            front = pc[:, 2] > 0.1
            u = fpx * pc[:, 0] / np.where(front, pc[:, 2], 1) + W / 2 - 0.5
            v = fpx * pc[:, 1] / np.where(front, pc[:, 2], 1) + H / 2 - 0.5
            inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            vis = np.zeros(len(P), bool)
            if inside.any():
                dd = P[inside] - o
                L = np.linalg.norm(dd, axis=1)
                th2, _, _ = sg.cast(rc, np.repeat(o[None], inside.sum(), 0), dd / L[:, None])
                vis[np.flatnonzero(inside)] = np.abs(th2 - L) < 0.06
            if folder == rid:
                seen[rid] |= vis
            walls_seen = sorted({int(w) for w in wid[vis]})
            rows.append(dict(photo=f, room=rid, role=role.get(f, ""), median_depth_m=round(med, 2),
                             closeup_share=round(near, 2), wall_points_seen=int(vis.sum()),
                             walls_seen=walls_seen, pitch_deg=round(float(np.degrees(np.arcsin(
                                 np.clip((R @ [0, 0, 1.0])[2], -1, 1)))), 1)))
        # per-room summary
        report = []
        for rid, r in rooms.items():
            P, wid = samp[rid]
            nw = len(np.asarray(r["polygon"]))
            per_wall = []
            for k in range(nw):
                m = wid == k
                if m.any():
                    per_wall.append(round(float(seen[rid][m].mean()), 2))
            ph = [x for x in rows if x["photo"].split("/")[0] == rid]
            problems = []
            for x in ph:
                if "ceiling" in x["role"]:
                    continue
                if x["median_depth_m"] < 1.0 or x["closeup_share"] > 0.5:
                    problems.append(f"{x['photo'].split('/')[1]} close-up ({x['role']}: median {x['median_depth_m']} m,"
                                    f" {int(100 * x['closeup_share'])}% < 0.8 m)")
                elif x["wall_points_seen"] < 20 and "doorway pair: back" not in x["role"]:
                    problems.append(f"{x['photo'].split('/')[1]} sees almost none of its room's walls ({x['role']})")
            weak = [k + 1 for k, c in enumerate(per_wall) if c < 0.3]
            if weak:
                problems.append(f"walls {weak} (of {nw}) less than 30% seen in the 1-2 m band")
            report.append(dict(room=rid, area_m2=round(r["area"], 1), photos=len(ph),
                               wall_coverage=round(float(seen[rid].mean()), 2), per_wall=per_wall,
                               problems=problems))
        out = ds / "verify"
        out.mkdir(exist_ok=True)
        (out / "photo_audit.json").write_text(json.dumps(dict(rooms=report, photos=rows), indent=1))
        lines = [f"# Photo audit: {ds.name} (ray-cast against the scene, true poses)", "",
                 "| Room | m² | Photos | Wall coverage (1-2 m band) | Per wall | Problems |", "|---|---|---|---|---|---|"]
        for x in report:
            lines.append(f"| {x['room']} | {x['area_m2']} | {x['photos']} | {int(100 * x['wall_coverage'])}% | "
                         f"{' '.join(str(int(100 * c)) for c in x['per_wall'])} | {'; '.join(x['problems']) or 'ok'} |")
        (out / "photo_audit.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
