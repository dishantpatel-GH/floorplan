"""Joint structure from motion over ALL photos of ALL rooms (decision P-1).

Why joint (not per room): the capture protocol asks for a "looking out" photo from every doorway. Such a photo shares
features with the neighbouring room's photos, so a single SfM over every image can register two rooms into ONE frame
through real image evidence. That is the strongest possible stitch: adjacency and relative placement are measured,
not guessed. Rooms that no photo links come out as separate models ("components"); stitch.py places those.

Why exhaustive pairs: 2-8 photos per room means at most a few hundred photos per property, so matching every pair
(N^2/2) costs seconds to minutes and cannot miss a link, unlike retrieval or sequential pairing.

Why min_model_size = 2: COLMAP discards models with fewer than 10 images by default; a room has 2-8.
SfM gives poses and sparse points up to an unknown scale per component; scale.py makes them metric.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap


@dataclass
class Component:
    """One SfM model: images registered together, in an arbitrary-scale frame."""
    names: list[str]                       # registered image names
    T_wc: dict[str, np.ndarray]            # name -> 4x4 camera-to-world (OpenCV axes), SfM units
    K: dict[str, np.ndarray]               # name -> 3x3 intrinsics at the SfM resolution (refined by BA)
    dist: dict[str, np.ndarray]            # name -> distortion parameters
    points: np.ndarray                     # (M,3) triangulated points
    point_err: np.ndarray                  # (M,) reprojection error, px
    obs: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # name -> (point idx, (k,2) px)


def _write_pairs(path: Path, names: list[str]) -> int:
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    path.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    return len(pairs)


def _component(rec: pycolmap.Reconstruction) -> Component:
    pid = sorted(rec.points3D.keys())
    index = {p: k for k, p in enumerate(pid)}
    pts = np.array([rec.points3D[p].xyz for p in pid]).reshape(-1, 3)
    err = np.array([rec.points3D[p].error for p in pid])
    comp = Component(names=[], T_wc={}, K={}, dist={}, points=pts, point_err=err)
    for im in rec.images.values():
        if not im.has_pose:
            continue
        Tcw = np.eye(4)
        Tcw[:3] = im.cam_from_world().matrix()
        cam = rec.cameras[im.camera_id]
        comp.names.append(im.name)
        comp.T_wc[im.name] = np.linalg.inv(Tcw)
        comp.K[im.name] = cam.calibration_matrix()
        comp.dist[im.name] = np.asarray(cam.params[3:])
        p2d = [(index[p.point3D_id], p.xy) for p in im.points2D if p.has_point3D()]
        comp.obs[im.name] = (np.array([a for a, _ in p2d], int), np.array([b for _, b in p2d]).reshape(-1, 2))
    comp.names.sort()
    return comp


def verified_pair_inliers(database: Path) -> dict[tuple[str, str], int]:
    """Number of geometrically verified matches per image pair (the evidence behind every link)."""
    out = {}
    max_id = 2147483647                       # COLMAP pair_id = id1 * 2^31-1 + id2
    with pycolmap.Database.open(database) as db:
        names = {im.image_id: im.name for im in db.read_all_images()}
        for pid, n in zip(*db.read_two_view_geometry_num_inliers()):
            i, j = int(pid) // max_id, int(pid) % max_id
            if i in names and j in names and n > 0:
                out[(names[i], names[j])] = int(n)
    return out


def run_joint_sfm(image_dir: Path, names: list[str], work: Path, f_prior: float | None, params,
                  log=print) -> tuple[list[Component], dict]:
    """Exhaustive ALIKED+LightGlue matching and incremental mapping; returns ALL models, largest first."""
    from hloc import extract_features, match_features, reconstruction
    work.mkdir(parents=True, exist_ok=True)
    pairs_file = work / "pairs_exhaustive.txt"
    n_pairs = _write_pairs(pairs_file, names)
    fconf = copy.deepcopy(extract_features.confs["aliked-n16"])
    fconf["model"]["max_num_keypoints"] = params.max_keypoints
    fconf["preprocessing"]["resize_max"] = params.sfm_long_side
    mconf = copy.deepcopy(match_features.confs["aliked+lightglue"])
    mconf["model"]["mp"] = True
    feats = extract_features.main(fconf, image_dir, work, image_list=names, as_half=False)
    matches = work / "matches.h5"
    match_features.main(mconf, pairs_file, features=feats, matches=matches)
    image_options, mode = {}, pycolmap.CameraMode.AUTO
    if f_prior is not None:   # one phone, one image size: share one camera so BA refines one focal from all photos
        from PIL import Image
        w, h = Image.open(image_dir / names[0]).size
        mode = pycolmap.CameraMode.SINGLE
        image_options = dict(camera_model=params.camera_model,
                             camera_params=f"{f_prior},{w / 2},{h / 2},0" if params.camera_model == "SIMPLE_RADIAL"
                             else f"{f_prior},{w / 2},{h / 2}")
    sfm_dir = work / "sfm"
    mapper = dict(min_model_size=params.min_model_size, min_num_matches=params.min_inliers_pair)
    mapper["mapper"] = dict(init_min_num_inliers=max(params.min_inliers_pair, 50))
    try:
        reconstruction.main(sfm_dir, image_dir, pairs_file, feats, matches, camera_mode=mode, image_list=names,
                            image_options=image_options, mapper_options=_mapper_options(mapper, params.seed))
    except Exception as e:     # pragma: no cover - logged, then handled as "no components"
        log(f"[photo/sfm] mapper failed: {e}")
    comps = _load_models(sfm_dir)
    comps.sort(key=lambda c: -len(c.names))
    stats = dict(images=len(names), pairs=n_pairs, components=[len(c.names) for c in comps],
                 registered=sum(len(c.names) for c in comps))
    log(f"[photo/sfm] {n_pairs} pairs -> {len(comps)} model(s) with {stats['components']} images "
        f"({stats['registered']}/{len(names)} registered)")
    return comps, stats


def _mapper_options(d: dict, seed: int) -> dict:
    out = dict(d)
    out["random_seed"] = seed
    return out


def _load_models(sfm_dir: Path) -> list[Component]:
    comps = []
    roots = [sfm_dir] + sorted((sfm_dir / "models").glob("*")) if (sfm_dir / "models").exists() else [sfm_dir]
    for r in roots:
        if (r / "images.bin").exists():
            comps.append(_component(pycolmap.Reconstruction(str(r))))
    return [c for c in comps if len(c.names) >= 2]


def verified_matches(database: Path, min_inliers: int) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Geometrically verified (epipolar RANSAC) feature matches per image pair, as pixel coordinates.

    Returns {(name_a, name_b): (xy_a (k,2), xy_b (k,2))} in the SfM images' pixels (pixel-centre convention: hloc adds
    0.5 when it writes COLMAP keypoints, we remove it). These matches are the raw evidence for cross-room links."""
    out = {}
    max_id = 2147483647
    with pycolmap.Database.open(database) as db:
        names = {im.image_id: im.name for im in db.read_all_images()}
        kps = {i: db.read_keypoints(i)[:, :2] - 0.5 for i in names}
        for pid, g in zip(*db.read_two_view_geometries()):
            i, j = int(pid) // max_id, int(pid) % max_id
            m = np.asarray(g.inlier_matches)
            if len(m) < min_inliers or i not in names or j not in names:
                continue
            out[(names[i], names[j])] = (kps[i][m[:, 0]], kps[j][m[:, 1]])
    return out
