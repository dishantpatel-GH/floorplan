"""Evaluation run + figures for simulated photo folders (truth used only here, after the pipeline has run)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

from floorplan.photo.evaluate import align_rotation_first, load_truth, relative_rotation_errors, truth_T_wc
from floorplan.pipeline.scene import load_scene


def _lidar_depth(capture: Path, t: dict, hw) -> np.ndarray:
    d = cv2.imread(str(capture / "depth" / f"{t['frame']:06d}.png"), -1).astype(np.float32) / 1000
    c = cv2.imread(str(capture / "confidence" / f"{t['frame']:06d}.png"), -1)
    d[c < 2] = 0
    d = np.rot90(d, k=t["rot90"]).copy()
    return cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


def _extent(P2: np.ndarray) -> np.ndarray:
    return np.percentile(P2, 98, axis=0) - np.percentile(P2, 2, axis=0)


def evaluate_run(photo_root: Path, scene: dict, info: dict, out: Path, repo: Path) -> dict | None:
    truth = load_truth(photo_root)
    if truth is None:
        return None
    capture = Path(truth["capture"])
    cap_name = truth.get("capture_name") or Path(photo_root).resolve().parent.name
    lidar, linfo = load_scene(repo / "outputs" / cap_name)
    plan = json.loads(Path(truth["plan"]).read_text()) if Path(truth["plan"]).exists() else None
    Ta = lidar["T_align"]
    names = [str(n) for n in scene["cam_names"]]
    placement = [str(x) for x in scene["cam_placement"]]
    T_est = scene["T_wc"]
    T_ref = np.array([Ta @ truth_T_wc(truth["photos"][n]) for n in names])
    linked = np.array([p == "linked" for p in placement])
    if linked.sum() < 2:          # nothing was linked by image evidence: placement cannot be scored
        ev = dict(summary=dict(photos=len(names), placed_linked=int(linked.sum()), rooms=len(info["rooms"]),
                               all_rooms_one_frame=info["all_rooms_one_frame"],
                               note="fewer than 2 photos linked by image evidence; placement not scorable"))
        (out / "evaluation.json").write_text(json.dumps(ev, indent=1))
        return ev

    # 1. whole-property placement: one similarity for ALL linked photos (the stitch), then its residuals
    s, R, t = align_rotation_first(T_est[linked], T_ref[linked])
    C = s * T_est[:, :3, 3] @ R.T + t
    cerr = np.linalg.norm(C - T_ref[:, :3, 3], axis=1)
    rel = relative_rotation_errors(T_est[linked], T_ref[linked])
    rooms = sorted({n.split("/")[0] for n in names})
    per_room = {}
    for r in rooms:
        m = np.array([n.startswith(r + "/") for n in names])
        per_room[r] = dict(photos=int(m.sum()), placement=sorted({placement[i] for i in np.where(m)[0]}),
                           centre_err_median_m=float(np.median(cerr[m])), centre_err_max_m=float(cerr[m].max()))

    # 2. points vs the LiDAR surface, rigid alignment at the photo tier's OWN scale (honest) and after similarity
    tree = cKDTree(lidar["points"])
    Pr = scene["raw_points"][linked_points(scene, names, linked)]
    sub = Pr[:: max(1, len(Pr) // 200000)]
    s1, R1, t1 = align_rotation_first(T_est[linked], T_ref[linked], with_scale=False)
    d_rigid = tree.query(sub @ R1.T + t1)[0]
    d_sim = tree.query(s * sub @ R.T + t)[0]

    # 3. per-photo depth scale vs LiDAR after the pipeline's corrections
    ratios = {}
    z = np.load(out / "work" / "depth_final.npz") if (out / "work" / "depth_final.npz").exists() else None
    if z is not None:
        for n in names:
            key = n.replace("/", "__")
            if key in z.files:
                D = z[key]
                L = _lidar_depth(capture, truth["photos"][n], D.shape)
                ok = (L > 0.2) & (L < 4) & (D > 0)
                if ok.sum() > 100:
                    ratios[n] = float(np.median(D[ok] / L[ok]))

    # 4. per-room extents inside the LiDAR plan polygons (2-98 percentile, 0.5-2.0 m above the floor)
    room_dims = {}
    if plan is not None:
        P_al = scene["raw_points"] @ R1.T + t1
        fy = linfo["floor_y"]
        for room in plan["rooms"]:
            poly = Polygon(room["polygon"]).buffer(0.05)
            if not poly.is_valid or poly.area < 1:
                continue
            res = {}
            for tag, P in (("photo", P_al), ("lidar", lidar["raw_points"][::20])):
                band = P[(P[:, 1] > fy + 0.5) & (P[:, 1] < fy + 2.0)]
                bx = poly.bounds
                pre = band[(band[:, 0] > bx[0]) & (band[:, 0] < bx[2]) & (band[:, 2] > bx[1]) & (band[:, 2] < bx[3])]
                inside = np.array([poly.contains(Point(u, v)) for u, v in pre[:: max(1, len(pre) // 20000), [0, 2]]])
                Q = pre[:: max(1, len(pre) // 20000)][inside] if len(pre) else pre
                res[tag] = _extent(Q[:, [0, 2]]).tolist() if len(Q) > 200 else None
            if res["photo"] and res["lidar"]:
                err = [100 * (a / b - 1) for a, b in zip(res["photo"], res["lidar"])]
                room_dims[room["id"]] = dict(photo_m=np.round(res["photo"], 3).tolist(),
                                             lidar_m=np.round(res["lidar"], 3).tolist(),
                                             err_pct=np.round(err, 1).tolist())

    errs = [abs(e) for r in room_dims.values() for e in r["err_pct"]]
    summary = dict(
        photos=len(names), placed_linked=int(linked.sum()), rooms=len(rooms),
        rooms_linked=sum(1 for r in info["rooms"].values() if r["placement"] == "linked"),
        all_rooms_one_frame=info["all_rooms_one_frame"],
        global_scale_error_pct=100 * (1 / s - 1), scale_sigma_pct=100 * info["scale"]["sigma_rel"],
        camera_centre_err_median_m=float(np.median(cerr[linked])), camera_centre_err_p90_m=float(np.percentile(cerr[linked], 90)),
        rel_rotation_err_median_deg=float(np.median(rel)), rel_rotation_err_p90_deg=float(np.percentile(rel, 90)),
        point_to_lidar_median_m_rigid=float(np.median(d_rigid)), point_to_lidar_p90_m_rigid=float(np.percentile(d_rigid, 90)),
        point_to_lidar_median_m_sim3=float(np.median(d_sim)), point_to_lidar_p90_m_sim3=float(np.percentile(d_sim, 90)),
        photo_depth_ratio_median=float(np.median(list(ratios.values()))) if ratios else None,
        photo_depth_ratio_p10_p90=[float(np.percentile(list(ratios.values()), q)) for q in (10, 90)] if ratios else None,
        room_extent_abs_err_pct_median=float(np.median(errs)) if errs else None,
        room_extent_abs_err_pct_p90=float(np.percentile(errs, 90)) if errs else None)
    ev = dict(summary=summary, per_room_cameras=per_room, room_extents=room_dims, photo_depth_ratio=ratios,
              note="Truth = ARKit poses + LiDAR of the frames the photos were simulated from; used only here.")
    (out / "evaluation.json").write_text(json.dumps(ev, indent=1))
    _figure(scene, lidar, linfo, R1, t1, T_ref, names, placement, out / "photo_vs_lidar_topview.png", summary)
    return ev


def linked_points(scene: dict, names: list[str], linked: np.ndarray) -> np.ndarray:
    """Mask of raw points that belong to rooms placed by image evidence (fallback rooms are a guess by design)."""
    rooms = [str(r) for r in scene["room_names"]]
    good = {rooms.index(n.split("/")[0]) for n, l in zip(names, linked) if l}
    return np.isin(scene["raw_point_room"], list(good))


def _figure(scene, lidar, linfo, R1, t1, T_ref, names, placement, path: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fy = linfo["floor_y"]
    L = lidar["points"]
    L = L[(L[:, 1] > fy + 0.3) & (L[:, 1] < fy + 2.0)]
    P = scene["raw_points"] @ R1.T + t1
    keep = (P[:, 1] > fy + 0.3) & (P[:, 1] < fy + 2.0)
    P, lab = P[keep], scene["raw_point_room"][keep]
    fig, ax = plt.subplots(1, 1, figsize=(11, 11))
    ax.scatter(L[::5, 0], L[::5, 2], s=0.2, c="0.6", label="LiDAR surface (reference)")
    cmap = plt.get_cmap("tab10")
    rooms = [str(r) for r in scene["room_names"]]
    sel = np.arange(0, len(P), max(1, len(P) // 120000))
    ax.scatter(P[sel, 0], P[sel, 2], s=0.3, c=[cmap(i % 10) for i in lab[sel]], alpha=0.5)
    C = scene["T_wc"][:, :3, 3] @ R1.T + t1
    F = scene["T_wc"][:, :3, 2] @ R1.T
    for k, n in enumerate(names):
        col = cmap(rooms.index(n.split("/")[0]) % 10)
        ax.plot(C[k, 0], C[k, 2], "o", color=col, mec="k", ms=6)
        ax.arrow(C[k, 0], C[k, 2], 0.5 * F[k, 0], 0.5 * F[k, 2], color=col, width=0.01)
        ax.plot([C[k, 0], T_ref[k, 0, 3]], [C[k, 2], T_ref[k, 2, 3]], "-", color="r", lw=0.8)
        ax.text(C[k, 0], C[k, 2], n.split("/")[1].replace(".jpg", ""), fontsize=6)
    for i, r in enumerate(rooms):
        ax.plot([], [], "o", color=cmap(i % 10), label=r)
    ax.plot([], [], "-", color="r", label="photo camera -> true position")
    ax.set_aspect("equal")
    ax.invert_yaxis()            # I-001: (x, z) seen from above needs v pointing down
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    ax.set_title(f"Photo tier vs LiDAR (rigid alignment at the photo tier's own scale)\n"
                 f"scale err {summary['global_scale_error_pct']:+.1f}%, cam err median "
                 f"{summary['camera_centre_err_median_m']:.2f} m, pts->LiDAR median "
                 f"{100 * summary['point_to_lidar_median_m_rigid']:.1f} cm; rooms linked "
                 f"{summary['rooms_linked']}/{summary['rooms']}", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m), drawn downward")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
