"""Score door-anchored stitching (D-081) on a simulated flat: plan placement vs the exact ground truth, and the doors
found in each room's own frame vs the true openings.

    python scripts/eval_door_stitch.py --run outputs/door_stitch/k65_after \
        --gt outputs/sim/k65v2/capture_iphone15/gt/ground_truth.csv --session outputs/sim/k65v2/old_v21/session.json

Plan: whole-plan IoU with the GT rooms under one rigid transform (floorplan.benchmark.wall_match.global_alignment),
per-room offset = distance between the plan room's centroid (under that transform) and the GT room's centroid.
Doors (needs the run's scene_info door_stitch report): each room's frame is put on the GT with the simulator's true
camera poses of the photos in it (orthogonal Procrustes on headings and positions); each door centre then goes to
the GT, and is paired with the nearest true opening of that room within 0.75 m (along-wall error, width error).
Unpaired detections are phantoms; true doors of the room that no detection reaches are misses. The overlay PNG
shows the plan rooms (red, dashed) over the GT rooms (filled), as in MyHouse_Dataset/k65_photo_plan_vs_gt.png.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.benchmark import wall_match as wm  # noqa: E402
from floorplan.benchmark.gt_eval import read_gt_geometry  # noqa: E402


def _true_poses(session: Path) -> dict:
    d = json.loads(session.read_text())
    return {(p["folder"], int(p["index"])): p["pose"] for p in d["photos"]}


def _frame_to_gt(photos: dict, truth: dict):
    """2-D orthogonal map (rotation or reflection) + shift from a room frame to the GT plan, from its photos."""
    hF, hG, cF, cG = [], [], [], []
    for n, (yaw, x, z, _) in photos.items():
        room, f = n.split("/")
        m = re.search(r"(\d+)", f)
        tp = truth.get((room, int(m.group(1)))) if m else None
        if tp is None:
            continue
        hF.append([np.sin(yaw), np.cos(yaw)])
        hG.append([np.cos(np.radians(tp[3])), np.sin(np.radians(tp[3]))])
        cF.append([x, z])
        cG.append([tp[0], tp[1]])
    if not hF:
        return None
    hF, hG, cF, cG = map(np.array, (hF, hG, cF, cG))
    H = hF.T @ hG + (cF - cF.mean(0)).T @ (cG - cG.mean(0))
    U, _, Vt = np.linalg.svd(H)
    # a room frame (x left of the first photo, z ahead, seen from above) is a mirror image of the GT plan (x east,
    # y north): force det(Q) = -1, which one photo's heading alone does not fix
    Q = Vt.T @ np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T)) * -1.0]) @ U.T
    t = cG.mean(0) - Q @ cF.mean(0)
    resid = float(np.degrees(np.median(np.arccos(np.clip(np.sum((hF @ Q.T) * hG, 1), -1, 1)))))
    return Q, t, resid, len(hF)


def door_table(info: dict, sim_gt: dict, truth: dict) -> list[dict]:
    ds = info.get("door_stitch") or {}
    rows = []
    gt_open = sim_gt["openings"]
    for room, doors in (ds.get("doors") or {}).items():
        fr = (ds.get("frames") or {}).get(room)
        if not fr:
            continue
        m = _frame_to_gt(fr["photos"], truth)
        gts = [o for o in gt_open if room in o["room_ids"] and o["cls"] == "door"]
        hit = set()
        for d in doors:
            if d["source"] == "door pixels":
                continue
            ax = 0 if d["side"][1] == "x" else 1
            x = np.zeros(2)
            x[ax], x[1 - ax] = d["face_m"], d["centre_m"]
            row = dict(room=room, door=d["id"], source=d["source"], width=d["width_m"], pairs=len(d["pairs"]))
            if m is None:
                rows.append(dict(row, result="no true poses"))
                continue
            Q, t, resid, n = m
            g = Q @ x + t
            best = None
            for o in gt_open:
                if room not in o["room_ids"]:
                    continue
                c = np.array(o["center"][:2])
                along = abs((g - c)[0 if o["axis"] == "x" else 1])
                dist = float(np.linalg.norm(g - c))
                if dist < 0.75 + o["width"] / 2 and (best is None or dist < best[0]):
                    best = (dist, along, o)
            if best is None:
                rows.append(dict(row, result="phantom", frame_fit_deg=round(resid, 1)))
                continue
            dist, along, o = best
            if o["cls"] == "door":
                hit.add(o["id"])
            rows.append(dict(row, result="window (phantom door)" if o["cls"] != "door" else "found", gt=o["id"],
                             along_err_m=round(along, 3), dist_m=round(dist, 3), gt_width=round(o["width"], 3),
                             width_err_m=None if d["width_m"] is None else round(d["width_m"] - o["width"], 3),
                             frame_fit_deg=round(resid, 1)))
        for o in gts:
            if o["id"] not in hit:
                rows.append(dict(room=room, door=None, result="missed", gt=o["id"], gt_width=round(o["width"], 3)))
    return rows


_DIR = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+z": (0.0, 1.0), "-z": (0.0, -1.0)}


def placement_errors(info: dict, truth: dict, ref: str = "01_living_room") -> dict:
    """Where each anchored room's frame sits relative to the reference room's, plan vs truth (m, deg): independent
    of the rooms' sizes and of the whole-plan alignment. Frames are put on the GT by their fitted photos' true poses
    (layout fit_poses), placements in the plan come from the layout anchors."""
    lays = info.get("room_layouts") or {}
    fits, plc = {}, {}
    for r, lay in lays.items():
        fp, a = lay.get("fit_poses"), lay.get("anchor") or {}
        if not lay.get("ok") or not fp or not a.get("side_to_plan"):
            continue
        m = float(lay["manhattan_yaw"])
        ph = {}
        for n, (phi, ox, oy, oz) in fp.items():
            c = np.array([[np.cos(m), 0, np.sin(m)], [0, 1, 0], [-np.sin(m), 0, np.cos(m)]]) @ np.array([ox, oy, oz])
            ph[n] = (phi + m, c[0], c[2], "fit")
        f = _frame_to_gt(ph, truth)
        if f is None:
            continue
        fits[r] = f
        M = np.stack([np.array(_DIR[a["side_to_plan"]["+x"]]), np.array(_DIR[a["side_to_plan"]["+z"]])], 1)
        plc[r] = (M, np.asarray(a["centre_uv"], float))
    if ref not in fits:
        return {}
    Qr, tr = fits[ref][0], fits[ref][1]
    Mr, cr = plc[ref]
    out = {}
    for r in fits:
        if r == ref:
            continue
        dp = Mr.T @ (plc[r][1] - cr)
        dt = Qr.T @ (fits[r][1] - tr)
        Rp, Rt = Mr.T @ plc[r][0], Qr.T @ fits[r][0]
        out[r] = dict(err_m=round(float(np.linalg.norm(dp - dt)), 2),
                      rot_ok=bool(np.allclose(np.round(Rp), np.round(Rt))))
    return out


def plan_metrics(plan: dict, gt_geom: dict) -> dict:
    mirror = wm.plan_mirrored(plan)
    glob = wm.global_alignment(plan, gt_geom, mirror)
    out = dict(iou=round(glob["score"], 3) if glob else None, rooms={})
    if not glob:
        return out
    M = np.diag([1.0, -1.0]) if mirror else np.eye(2)
    for r in plan["rooms"]:
        g = gt_geom.get(r.get("label"))
        if g is None or len(r.get("polygon") or []) < 3:
            continue
        P = np.asarray(r["polygon"], float) @ M.T @ glob["R"].T + glob["t"]
        cp = np.array(wm._polygon(P).centroid.coords[0])
        cg = np.array(wm._polygon(np.asarray(g["poly"], float)).centroid.coords[0])
        out["rooms"][r["label"]] = dict(offset_m=round(float(np.linalg.norm(cp - cg)), 2),
                                        iou=round(wm.iou(P, np.asarray(g["poly"], float)), 3))
    out["glob"] = glob
    out["mirror"] = mirror
    return out


def overlay(plan: dict, gt_geom: dict, pm: dict, title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8.5))
    cols = ["#d4dcec", "#e8d8bc", "#d4e6cf", "#ecd6dc", "#e6e2c8", "#d8d0e8"]
    for i, (name, g) in enumerate(sorted(gt_geom.items())):
        P = np.asarray(g["poly"], float)
        ax.fill(P[:, 0], P[:, 1], color=cols[i % len(cols)], zorder=1)
        ax.plot(*np.vstack([P, P[:1]]).T, color="#3c8a3c", lw=2.5, zorder=2)
        c = P.mean(0)
        ax.text(c[0], c[1], name.split("_", 1)[-1].replace("_", " ").upper(), ha="center", va="center",
                fontsize=9, fontweight="bold", color="#222", zorder=5)
    glob, M = pm.get("glob"), np.diag([1.0, -1.0]) if pm.get("mirror") else np.eye(2)
    for r in plan["rooms"]:
        if glob is None or len(r.get("polygon") or []) < 3:
            continue
        P = np.asarray(r["polygon"], float) @ M.T @ glob["R"].T + glob["t"]
        ax.plot(*np.vstack([P, P[:1]]).T, color="#cc2222", lw=2.2, ls="--", zorder=4)
    for o in plan.get("openings", []):
        ctr = o.get("position") or o.get("center")
        if glob is None or ctr is None:
            continue
        q = np.asarray(ctr, float) @ M.T @ glob["R"].T + glob["t"]
        ax.plot(q[0], q[1], "o", color="#cc2222", ms=6, zorder=6)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True, help="ground_truth.csv (sim_gt.json beside it)")
    ap.add_argument("--session", type=Path, help="simulator session with the photos' true poses (door table)")
    ap.add_argument("--png", type=Path)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    plan = json.loads((a.run / "plan.json").read_text())
    info = json.loads((a.run / "scene" / "scene_info.json").read_text())
    gt_geom = read_gt_geometry(a.gt, None)
    sim_gt = json.loads(a.gt.with_name("sim_gt.json").read_text())
    pm = plan_metrics(plan, gt_geom)
    res = dict(iou=pm["iou"], rooms=pm["rooms"])
    if a.session:
        res["doors"] = door_table(info, sim_gt, _true_poses(a.session))
        res["placement_vs_living"] = placement_errors(info, _true_poses(a.session))
    res["stitch"] = {k: v for k, v in (info.get("door_stitch") or {}).items() if k in ("result", "rooms",
                                                                                        "reference_room")}
    if a.png:
        overlay(plan, gt_geom, pm, a.title or f"{a.run.name}: plan (red dashed) over GT (filled); overlap IoU "
                f"{pm['iou']}", a.png)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
