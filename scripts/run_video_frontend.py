#!/usr/bin/env python
"""Video tier: build the metric scene from rgb.mp4 only, save it, and (if the capture also has LiDAR) score it.

Usage:
  python scripts/run_video_frontend.py <capture_dir_or_mp4> [--out outputs/video_tier] [--depth-model moge2]
                                       [--lidar-scene outputs/<name>] [--loop-closure]

The scene is written as <out>/<name>/scene.npz + scene_info.json (same format as the LiDAR tier). When the input is
a Stray Scanner folder and a LiDAR scene exists (--lidar-scene, or outputs/<name>), evaluation.json and figures
compare the two. The LiDAR data is read ONLY for this evaluation, never by the video tier itself.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floorplan.pipeline.scene import load_scene, save_scene  # noqa: E402
from floorplan.video import VideoParams, build_scene_from_video  # noqa: E402
from floorplan.video.evaluate import evaluate_against_lidar  # noqa: E402


def capture_name(p: Path) -> str:
    p = p.parent if p.suffix.lower() in (".mp4", ".mov") else p
    return f"{p.parent.name}__{p.name}"


def _slice(P: np.ndarray, floor: float) -> np.ndarray:
    return (P[:, 1] > floor + 0.9) & (P[:, 1] < floor + 1.5)


def overlay_figure(scene, lidar, ev, info, out: Path) -> None:
    """Top view at 0.9-1.5 m above the LiDAR floor: LiDAR TSDF (grey) vs video TSDF after the rigid (SE3)
    alignment (red), plus both camera paths."""
    M = np.array(ev["transforms"]["se3"])
    V = scene["points"] @ M[:3, :3].T + M[:3, 3]
    L = lidar["points"]
    floor = np.percentile(L[:, 1], 5)
    tv = scene["traj"] @ M[:3, :3].T + M[:3, 3]
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    for a, show_video in zip(ax, (False, True)):
        a.scatter(L[_slice(L, floor), 0], L[_slice(L, floor), 2], s=0.2, c="0.6", label="LiDAR slice 0.9-1.5 m")
        if show_video:
            a.scatter(V[_slice(V, floor), 0], V[_slice(V, floor), 2], s=0.2, c="r", label="video slice")
            a.plot(tv[:, 0], tv[:, 2], "r-", lw=0.6, label="video camera path")
        a.plot(lidar["traj"][:, 0], lidar["traj"][:, 2], "k-", lw=0.6, label="ARKit camera path")
        a.set_title("video tier, rigid (SE3) alignment" if show_video else "LiDAR tier (reference)")
        a.set_aspect("equal")
        a.grid(True, lw=0.3)
        a.legend(loc="upper right", markerscale=20)
    tr = ev["trajectory"]
    fig.suptitle(f"{Path(info['source']).parent.name}: scale error {tr['scale_error_pct']:+.2f}% "
                 f"(claimed 95% CI +-{100 * info['scale']['ci95_rel_total']:.1f}%), "
                 f"SE3 ATE {100 * tr['se3']['ate_rmse_m']:.1f} cm, C2C median {100 * ev['c2c_points_se3']['median_m']:.1f} cm")
    fig.tight_layout()
    fig.savefig(out / "video_vs_lidar_topview.png", dpi=100)
    plt.close(fig)


def window_figure(ev, info, out: Path) -> None:
    """Scale error per 30-keyframe window after drift correction, against the claimed 95% interval."""
    w = ev["windows"]
    x = [d["start"] for d in w]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, [d["scale_error_pct"] for d in w], width=20, color="tab:red", label="scale error per window", zorder=3)
    ci = 100 * info["scale"]["ci95_rel_total"]
    ax.axhspan(-ci, ci, color="0.85", zorder=0, label=f"claimed 95% interval +-{ci:.1f}%")
    ax.axhline(3, ls="--", c="k", lw=0.8)
    ax.axhline(-3, ls="--", c="k", lw=0.8, label="video gate +-3%")
    ax.set_xlabel("keyframe index (window start)")
    ax.set_ylabel("scale error (%)")
    ax.set_title(f"{Path(info['source']).parent.name}: local scale error after drift correction (vs ARKit)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scale_error_windows.png", dpi=100)
    plt.close(fig)


def sheet_oracle(scene, info, cap_dir, lidar, linfo) -> dict:
    """What a paper sheet (opt-in cue, D-067) WOULD do to the scale (the sample captures have no sheet).

    A sheet measures the camera's height above the floor in metres (floorplan.scale; 0.55% median error on rendered
    sheets). The video tier divides it by the same camera's height above its OWN reconstructed floor. Here the
    metric height comes from ARKit + the LiDAR floor instead, so this isolates the reconstruction side of the cue:
    per keyframe ratio = true height / video height. Its median is the correction a sheet seen everywhere would
    apply; its scatter is the error of a sheet seen in ONE keyframe (before the sheet's own 0.55%)."""
    from floorplan.video.evaluate import arkit_reference
    kf = scene["kf"]
    ref = arkit_reference(cap_dir, scene["timestamps"], kf, info["rotation"]["final_rotation"])
    h_true = (ref["T_wc"][:, :3, 3] @ lidar["T_align"][:3, :3].T + lidar["T_align"][:3, 3])[:, 1] - linfo["floor_y"]
    h_vid = scene["traj"][kf, 1] - info["floor_y"]
    r = h_true / h_vid
    ok = np.isfinite(r) & (h_true > 0.5) & (h_vid > 0.3)
    seg = np.asarray(scene.get("kf_segment", np.zeros(len(kf), int)))
    out = dict(median_ratio=float(np.median(r[ok])), mad_sigma_rel=float(1.4826 * np.median(np.abs(np.log(r[ok] / np.median(r[ok]))))),
               median_h_true_m=float(np.median(h_true[ok])), median_h_video_m=float(np.median(h_vid[ok])), per_segment=[])
    for g in np.unique(seg):
        m = ok & (seg == g)
        if m.sum() >= 10:
            out["per_segment"].append(dict(segment=int(g), median_ratio=float(np.median(r[m])), n=int(m.sum())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/video_tier")
    ap.add_argument("--depth-model", default=None, choices=["moge2", "da3metric"])
    ap.add_argument("--lidar-scene", type=Path, default=None)
    ap.add_argument("--loop-closure", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-only", action="store_true", help="re-score a saved scene without rebuilding it")
    ap.add_argument("--eval-capture", type=Path, default=None,
                    help="Stray capture whose ARKit poses score this video (for re-encoded copies of its rgb.mp4)")
    ap.add_argument("--no-graph", action="store_true", help="ablation: v1 chaining instead of the pose graph")
    ap.add_argument("--sheet", action="store_true", help="opt in to the paper-sheet scale cue (off by default, D-067)")
    a = ap.parse_args()
    params = VideoParams()
    if a.depth_model:
        params.depth_model = a.depth_model
    params.dpvo_loop_closure = a.loop_closure
    params.use_pose_graph = not a.no_graph
    params.use_sheet = a.sheet
    name = a.capture.stem if (a.capture.suffix.lower() == ".mp4" and a.eval_capture) else capture_name(a.capture)
    out = a.out / (name + a.tag)
    if a.eval_only:
        scene, info = load_scene(out)
    else:
        scene, info = build_scene_from_video(a.capture, params, work_dir=out / "work")
        save_scene(scene, info, out)
    cap_dir = a.eval_capture or (a.capture.parent if a.capture.suffix.lower() == ".mp4" else a.capture)
    lidar_dir = a.lidar_scene or ROOT / "outputs" / capture_name(cap_dir)
    if (lidar_dir / "scene.npz").exists() and (cap_dir / "odometry.csv").exists():
        lidar, linfo = load_scene(lidar_dir)
        ev = evaluate_against_lidar(scene, info, cap_dir, lidar)
        if info.get("floor_y") is not None:
            ev["sheet_oracle"] = sheet_oracle(scene, info, cap_dir, lidar, linfo)
            so = ev["sheet_oracle"]
            corrected = (1 + ev["trajectory"]["scale_error_pct"] / 100) * so["median_ratio"] - 1
            print(f"[eval] sheet oracle: true/video camera height {so['median_ratio']:.4f} (single-sighting scatter "
                  f"{100 * so['mad_sigma_rel']:.1f}%); whole-path scale error after it {100 * corrected:+.2f}%")
        (out / "evaluation.json").write_text(json.dumps(ev, indent=2, default=float))
        overlay_figure(scene, lidar, ev, info, out)
        window_figure(ev, info, out)
        tr = ev["trajectory"]
        print(f"[eval] scale error {tr['scale_error_pct']:+.2f}%  SE3 ATE {tr['se3']['ate_rmse_m']:.3f} m  "
              f"Sim3 ATE {tr['sim3']['ate_rmse_m']:.3f} m  rot {tr['rot_err_median_deg']:.2f} deg  "
              f"C2C(TSDF, SE3) median {ev['c2c_points_se3']['median_m']:.3f} m  "
              f"C2C(raw, SE3) median {ev['c2c_raw_points_se3']['median_m']:.3f} m")
        if "trajectory_trusted" in ev:
            tt = ev["trajectory_trusted"]
            print(f"[eval] trusted keyframes only ({tt['frames']}): scale error {tt['scale_error_pct']:+.2f}%  SE3 ATE "
                  f"{tt['se3']['ate_rmse_m']:.3f} m  Sim3 ATE {tt['sim3']['ate_rmse_m']:.3f} m  path {tt['path_length_m']:.1f} m")
        for sg in ev.get("segments", []):
            q = next((x for x in info["scale"]["segments"] if x["id"] == sg["segment"]), {})
            print(f"[eval] segment {sg['segment']} ({'trusted' if q.get('trusted', True) else 'UNTRUSTED'}) "
                  f"kf {sg['keyframes']}: path {sg['path_m']:.1f} m, scale error "
                  f"{sg['scale_error_pct']:+.2f}% (claimed 1-sigma {sg['claimed_sigma_pct']:.1f}%, covered "
                  f"{sg['covered_95']}), Sim3 ATE {100 * sg['sim3_ate_m']:.1f} cm")
        g = info.get("pose_graph") or {}
        print(f"[eval] restarts {info['scale']['vo_restarts']}, whole_scene_consistent "
              f"{info['scale']['whole_scene_consistent']}, connectivity {g.get('connectivity')}, "
              f"bridges {g.get('bridges', {}).get('accepted')}/{g.get('bridges', {}).get('tried')}, "
              f"loops {g.get('loops')}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
