"""Structure from motion on the keyframes: hloc (ALIKED features + LightGlue matching) and pycolmap incremental
mapping with self-calibration.

Why SfM and not a feed-forward model for poses: bundle adjustment minimises reprojection error over hundreds of
images jointly, so relative poses are accurate to millimetres-to-centimetres over a room. Feed-forward models
(MapAnything, DA3) predict each window independently, and chaining windows accumulates error (decision V-3 in
docs/modules/video_tier.md compares them on this data). SfM gives no metric scale; scale.py adds it.

Here SfM has one job in the main pipeline: SELF-CALIBRATION of the focal length on the first keyframes (bundle
adjustment refines one shared focal over many images; measured 0.6% from ARKit's value on single_room versus 3.9% for
GeoCalib). It is NOT used for the trajectory: on these LiDAR-style sweeps past blank walls it breaks into fragments
(decision V-3). Pairs are sequential: each keyframe against the next `seq_overlap` keyframes, which finds almost
every overlapping pair of an ordered video at a fraction of the cost of exhaustive matching.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap


@dataclass
class SfmResult:
    names: list[str]                 # image names in keyframe order (registered or not)
    registered: np.ndarray           # (N,) bool
    T_wc: np.ndarray                 # (N,4,4) camera-to-world (OpenCV axes), SfM units; NaN if not registered
    K: np.ndarray                    # (3,3) shared intrinsics at the SfM image resolution
    dist: np.ndarray                 # distortion parameters of the camera model
    image_size: tuple[int, int]      # (width, height) of the SfM images
    points: np.ndarray               # (M,3) triangulated points, SfM units
    point_err: np.ndarray            # (M,) mean reprojection error (px)
    point_track: np.ndarray          # (M,) number of images observing each point
    obs: list[tuple[np.ndarray, np.ndarray]]   # per keyframe: (point indices, 2-D pixel positions)
    stats: dict


def _write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{a} {b}\n" for a, b in pairs))


def sequential_pairs(names: list[str], overlap: int) -> list[tuple[str, str]]:
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, min(i + 1 + overlap, len(names)))]


def _extract_and_match(image_dir: Path, names: list[str], pairs_file: Path, work: Path, max_kp: int):
    from hloc import extract_features, match_features
    fconf = copy.deepcopy(extract_features.confs["aliked-n16"])
    fconf["model"]["max_num_keypoints"] = max_kp
    mconf = copy.deepcopy(match_features.confs["aliked+lightglue"])
    mconf["model"]["mp"] = True
    feats = extract_features.main(fconf, image_dir, work, image_list=names, as_half=False)  # fp32 keypoints
    matches = work / "matches.h5"
    match_features.main(mconf, pairs_file, features=feats, matches=matches)   # skips pairs already matched
    return feats, matches


def _run_mapper(image_dir, names, pairs_file, feats, matches, out: Path, f_prior: float, cam_model: str, seed: int):
    from hloc import reconstruction
    w, h = _image_size(image_dir / names[0])
    params = {"SIMPLE_RADIAL": f"{f_prior},{w / 2},{h / 2},0", "PINHOLE": f"{f_prior},{f_prior},{w / 2},{h / 2}",
              "SIMPLE_PINHOLE": f"{f_prior},{w / 2},{h / 2}", "RADIAL": f"{f_prior},{w / 2},{h / 2},0,0"}[cam_model]
    return reconstruction.main(out, image_dir, pairs_file, feats, matches, camera_mode=pycolmap.CameraMode.SINGLE,
                               image_list=names, image_options=dict(camera_model=cam_model, camera_params=params),
                               mapper_options=dict(random_seed=seed, num_threads=1))   # 1 thread: deterministic BA


def _image_size(path: Path) -> tuple[int, int]:
    import cv2
    h, w = cv2.imread(str(path)).shape[:2]
    return w, h


def _to_result(rec: pycolmap.Reconstruction, names: list[str], stats: dict) -> SfmResult:
    pid_list = sorted(rec.points3D.keys())
    pid_index = {p: k for k, p in enumerate(pid_list)}
    pts = np.array([rec.points3D[p].xyz for p in pid_list]) if pid_list else np.zeros((0, 3))
    err = np.array([rec.points3D[p].error for p in pid_list])
    trk = np.array([rec.points3D[p].track.length() for p in pid_list])
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    T = np.full((len(names), 4, 4), np.nan)
    reg = np.zeros(len(names), bool)
    obs = []
    for k, n in enumerate(names):
        im = by_name.get(n)
        if im is None:
            obs.append((np.zeros(0, int), np.zeros((0, 2))))
            continue
        cw = im.cam_from_world().matrix()                       # 3x4 world -> camera
        Tcw = np.eye(4)
        Tcw[:3] = cw
        T[k] = np.linalg.inv(Tcw)
        reg[k] = True
        p2d = [(pid_index[p.point3D_id], p.xy) for p in im.points2D if p.has_point3D()]
        obs.append((np.array([a for a, _ in p2d], int), np.array([b for _, b in p2d]).reshape(-1, 2)))
    cam = next(iter(rec.cameras.values()))
    stats.update(registered=int(reg.sum()), images=len(names), points=len(pts),
                 mean_reproj_px=float(np.mean(err)) if len(err) else None,
                 camera_model=cam.model.name, camera_params=[float(x) for x in cam.params])
    return SfmResult(names=names, registered=reg, T_wc=T, K=cam.calibration_matrix(),
                     dist=np.asarray(cam.params[3:]), image_size=(cam.width, cam.height), points=pts,
                     point_err=err, point_track=trk, obs=obs, stats=stats)


def run_sfm(image_dir: Path, names: list[str], work: Path, f_prior: float, params, log=print) -> SfmResult:
    """SfM with sequential pairs; returns the largest reconstructed model."""
    work.mkdir(parents=True, exist_ok=True)
    pairs = sequential_pairs(names, params.seq_overlap)
    pairs_file = work / "pairs_seq.txt"
    _write_pairs(pairs_file, pairs)
    feats, matches = _extract_and_match(image_dir, names, pairs_file, work, params.max_keypoints)
    rec = _run_mapper(image_dir, names, pairs_file, feats, matches, work / "sfm", f_prior,
                      params.camera_model, params.seed)
    if rec is None:
        raise RuntimeError("SfM failed: no model could be initialised from the sequential pairs")
    res = _to_result(rec, names, dict(pairs=len(pairs)))
    log(f"[sfm] {len(pairs)} sequential pairs: {res.stats['registered']}/{len(names)} registered, "
        f"{res.stats['points']} points, reproj {res.stats['mean_reproj_px']:.2f} px")
    return res
