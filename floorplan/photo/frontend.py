"""Photo tier front-end: per-room photo folders -> ONE metric, aligned scene in the LiDAR scene format.

Pipeline (docs/modules/photo_tier.md section 2):
  1. load photos (EXIF orientation, HEIC), focal length prior from EXIF;
  2. ALIKED + LightGlue on ALL photo pairs of ALL rooms, epipolar verification (COLMAP), diagnostic joint SfM;
  3. MoGe-2 metric depth + normals per photo; GeoCalib gravity per photo, refined by the floor plane -> levelled
     metric point cloud per photo;
  4. metric photo-photo edges (gravity-constrained similarity from lifted matches) -> robust pose graph ->
     every photo placed in one frame; cross-room edges are the room links;
  5. global scale from every cue (learned depth, camera-height prior; the paper sheet only if use_sheet=True, D-067);
  6. rooms that no photo links: fallback placement next to the linked block, no overlap, flagged and widened;
     v2 (D-017): spin-group priors and doorway-pair links are added to the graph in step 4, an unmatched doorway
     photo is placed in its room by wall alignment, and a door-matching fallback runs before "beside the block";
  7. gravity/Manhattan alignment with the SAME code as the LiDAR tier (floorplan.plan.align), voxel fusion.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from floorplan.config import Config
from floorplan.photo import images as I
from floorplan.photo import link as L
from floorplan.photo import protocol as PR
from floorplan.photo import scale as SC
from floorplan.photo.depth import MoGeRunner, clean_depth
from floorplan.photo.geometry import manhattan_yaw, yaw_matrix
from floorplan.photo.params import PhotoParams
from floorplan.photo.sfm import run_joint_sfm, verified_matches
from floorplan.photo.views import PhotoView, build_view, gravity_geocalib
from floorplan.plan.align import align_scene


def _prepare(photos, p: PhotoParams, work: Path):
    """SfM-resolution copies on disk (for hloc) and depth-resolution arrays + intrinsics in memory."""
    img_dir = work / "images"
    small, Ks, f_src = {}, {}, {}
    for ph in photos:
        (img_dir / ph.room).mkdir(parents=True, exist_ok=True)
        sfm_img = I.resize_long(ph.rgb, p.sfm_long_side)
        Image.fromarray(sfm_img).save(img_dir / ph.name, quality=95)
        d = I.resize_long(ph.rgb, p.depth_long_side)
        f, src = I.focal_px(ph, max(d.shape[:2]), p.default_hfov_deg, p.f35_rule)
        h, w = d.shape[:2]
        small[ph.name] = (d, d.shape[1] / sfm_img.shape[1])
        Ks[ph.name] = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1]])
        f_src[ph.name] = src
    return img_dir, small, Ks, f_src


def _views(photos, small, Ks, p: PhotoParams, log, cache: Path | None = None) -> dict[str, PhotoView]:
    """Per-photo levelled metric views. Cached in the work dir (keyed by photo names, focal and depth settings):
    re-running the graph stages must not recompute MoGe-2/GeoCalib (minutes on a shared GPU)."""
    import pickle
    key = (tuple(ph.name for ph in photos), tuple(round(float(Ks[ph.name][0, 0]), 3) for ph in photos),
           p.depth_long_side, p.depth_max_m, p.depth_edge_rel, p.seed)
    if cache is not None and cache.exists():
        try:
            k, views = pickle.loads(cache.read_bytes())
            if k == key:
                log(f"[photo] depth + gravity for {len(views)} photos loaded from cache")
                return views
        except Exception:
            pass
    views = _compute_views(photos, small, Ks, p, log)
    if cache is not None:
        cache.write_bytes(pickle.dumps((key, views)))
    return views


def _compute_views(photos, small, Ks, p: PhotoParams, log) -> dict[str, PhotoView]:
    moge = MoGeRunner()
    preds = {ph.name: moge.infer(small[ph.name][0], Ks[ph.name][0, 0]) for ph in photos}
    moge.close()
    ups = gravity_geocalib([small[ph.name][0] for ph in photos], [Ks[ph.name][0, 0] for ph in photos], p.seed)
    views = {}
    for ph, up in zip(photos, ups):
        r = preds[ph.name]
        d = clean_depth(r["depth"], p.depth_max_m, p.depth_edge_rel)
        src = "geocalib" if up is not None else "camera_prior"
        up = up if up is not None else np.array([0.0, -1.0, 0.0])    # image "up" = -y in OpenCV camera axes
        views[ph.name] = build_view(ph.name, ph.room, small[ph.name][0], Ks[ph.name], d, r["normal"], up, src,
                                    small[ph.name][1])
    n_floor = sum(v.up_source == "floor" for v in views.values())
    log(f"[photo] depth + gravity for {len(views)} photos ({n_floor} levelled by their floor plane)")
    return views


def _pose_T(pose) -> np.ndarray:
    th, t, ls = pose
    T = np.eye(4)
    T[:3, :3] = L._ry(th)
    T[:3, 3] = t
    return T


def _photo_points(v: PhotoView, pose, stride: int):
    """World points, normals (unit), colours and ranges of one photo under pose (yaw, t, log_s)."""
    th, t, ls = pose
    Pc, idx = v.points_cam(stride)
    R = L._ry(th) @ v.R_lev
    P = np.exp(ls) * Pc @ R.T + t
    return P, R, idx, np.exp(ls) * np.linalg.norm(Pc, axis=1)


def _place_components(comps, views, edges, log):
    """Solve each component; decide which are placed by image evidence and which need the fallback."""
    placed, fallback, dropped = [], [], []
    rooms_done: set[str] = set()
    # anchor = the component spanning the most rooms (then most photos): a big single-room spin group must not
    # displace a multi-room block as the reference
    comps = sorted(comps, key=lambda c: (-len({views[n].room for n in c}), -len(c)))
    for c in comps:
        ce = [e for e in edges if e.a in c and e.b in c]
        poses = L.solve_component(c, ce) if len(c) > 1 else {c[0]: (0.0, np.zeros(3), 0.0)}
        rooms = {views[n].room for n in c}
        if not placed:
            placed.append((c, poses))
            rooms_done |= rooms
        elif rooms - rooms_done:
            fallback.append((c, poses))
            rooms_done |= rooms
        else:
            dropped.append(c)       # a fragment of rooms already placed with no link to them: cannot be placed
    log(f"[photo] components: {[len(c) for c in comps]} -> 1 linked block, {len(fallback)} fallback, "
        f"{len(dropped)} unplaceable fragment(s)")
    return placed, fallback, dropped


def _cloud(views, poses, stride):
    P, N, C, rng, lab = [], [], [], [], []
    for n, pose in poses.items():
        v = views[n]
        Pw, R, idx, r = _photo_points(v, pose, stride)
        nc = v.normal_cam.reshape(-1, 3)[idx] if getattr(v, "normal_cam", None) is not None else None
        P.append(Pw); rng.append(r); lab += [n] * len(Pw)
        C.append(v.rgb.reshape(-1, 3)[idx])
        N.append(nc @ R.T if nc is not None else np.zeros_like(Pw))
    return (np.concatenate(P), np.concatenate(N), np.concatenate(C), np.concatenate(rng), np.array(lab))


def _rays(P, lab, poses, Ra) -> np.ndarray:
    """Unit ray camera -> point in the aligned frame (the LiDAR scene's raw_ray; used for free-space carving)."""
    C = np.array([poses[n][1] for n in lab])
    d = (P - C) @ Ra.T
    return (d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)).astype(np.float16)


def _fallback_place(block_P, comp_P, comp_yaw, block_yaw, gap):
    """Rotate a component onto the block's Manhattan axes and put it beside the block (+x), floors level.
    Returns (yaw, shift) to apply. Position is a GUESS (flagged), sizes are still measured."""
    dyaw = block_yaw - comp_yaw
    Q = comp_P @ L._ry(-dyaw).T            # yaw_matrix convention: rotate about +y by -(-dyaw)
    shift = np.array([block_P[:, 0].max() + gap - Q[:, 0].min(),
                      np.percentile(block_P[:, 1], 2) - np.percentile(Q[:, 1], 2),
                      block_P[:, 2].min() - Q[:, 2].min()])
    return -dyaw, shift


def _voxel_mean(P, N, C, voxel):
    key = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    def mean(X):
        out = np.zeros((len(cnt), X.shape[1]))
        np.add.at(out, inv, X)
        return out / cnt[:, None]
    Nm = mean(N)
    Nm /= np.maximum(np.linalg.norm(Nm, axis=1, keepdims=True), 1e-9)
    return mean(P), Nm, mean(C.astype(float)).astype(np.uint8), inv


def build_scene_from_photos(photo_root, params: dict | None = None, work_dir: Path | None = None, log=print):
    """Photo folders (one per room) -> (scene, info) in the LiDAR scene format plus photo-specific keys."""
    t0 = time.time()
    p = PhotoParams.from_dict(params)
    photo_root = Path(photo_root)
    # one work/cache dir per input folder: runs on different photo sets under one output folder (main set, repeat
    # takes) used to share and overwrite one cache, so every rerun recomputed its depth maps on the GPU
    work = Path(work_dir) if work_dir else photo_root.parent / f"photo_work__{photo_root.name}"
    work.mkdir(parents=True, exist_ok=True)
    photos = I.load_all(photo_root)
    rooms = sorted({ph.room for ph in photos})
    log(f"[photo] {len(photos)} photos in {len(rooms)} room folders")
    img_dir, small, Ks, f_src = _prepare(photos, p, work)
    names = [ph.name for ph in photos]
    sizes = {small[n][0].shape for n in names}
    f_prior = None
    if len(sizes) == 1 and all(ph.f35 for ph in photos):
        f_prior = I.focal_px(photos[0], p.sfm_long_side, p.default_hfov_deg, p.f35_rule)[0]
    _, sfm_stats = run_joint_sfm(img_dir, names, work, f_prior, p, log)
    matches = verified_matches(work / "sfm" / "database.db", p.min_inliers_pair)
    views = _views(photos, small, Ks, p, log, cache=work / "views_cache.pkl")
    from floorplan.photo.semantic import attach_semantics
    sem_info = attach_semantics(photos, views, work, p, log)          # D-060: wall masks (cached, optional)
    edges = L.build_edges(views, matches, p, log)
    intra_stats = None
    if p.intra_room_proposals:
        from floorplan.photo.intra import mapanything_edges
        extra, intra_stats = mapanything_edges(views, edges, p, log)
        edges = edges + extra
    proto = _protocol_edges(photos, views, edges, p, log)
    edges = edges + proto["edges"]
    feat_status = {}                     # BEFORE loop pruning: a false link would otherwise win against a true pair
    edges = _features_need_pair(views, edges, p, feat_status)
    edges, pruned = L.prune_inconsistent(names, edges, p.prune_max_dt_m, p.prune_max_dt_rel, p.prune_max_dyaw_deg, log)
    edges, bridges = L.drop_weak_bridges(edges, p.bridge_min_inliers, log)
    pruned += bridges
    edges, proto["pair_checks"] = _verify_pairs(names, views, edges, p, log)
    proto["pair_checks"] = [dict(a=a, b=b, kind=k, violation=v, kept=not v.startswith("rejected"))
                            for (a, b, k), v in feat_status.items()] + proto["pair_checks"]
    if p.layout_registration:
        edges, proto["layout"] = _layout_register(names, views, edges, proto, p, log)
        edges, more = _verify_pairs(names, views, edges, p, log)
        proto["pair_checks"] += more
        edges, proto["pair_bridge"] = _pair_bridge(names, views, edges, proto, p, log)
    comps = L.components(names, edges)
    placed, fallback, dropped = _place_components(comps, views, edges, log)

    # --- one frame: linked block first, then fallback components beside it ---
    poses = dict(placed[0][1])
    P, N, C, R_, lab = _cloud(views, poses, p.raw_stride * 2)
    block_yaw = manhattan_yaw(N)
    # a one-photo anchor is not "linked" by any evidence: label it so its intervals widen like a fallback
    placement = {n: ("linked" if len(poses) > 1 else "single_photo_anchor") for n in poses}
    door_match_log = []
    for c, cp in fallback:
        Pc, Nc, _, _, _ = _cloud(views, cp, p.raw_stride * 2)
        dm = _door_match(views, poses, cp, P, lab, Pc, block_yaw, manhattan_yaw(Nc), proto["pairs"], p)
        door_match_log.append(dm["log"])
        if dm["ok"]:
            dyaw, shift, how = dm["dyaw"], dm["shift"], "door_match"
        else:
            (dyaw, shift), how = _fallback_place(P, Pc, manhattan_yaw(Nc), block_yaw, p.fallback_gap_m), \
                "fallback_beside_block"
        for n, (th, t, ls) in cp.items():
            poses[n] = (th + dyaw, L._ry(dyaw) @ t + shift, ls)
            placement[n] = how
        P, N, C, R_, lab = _cloud(views, poses, p.raw_stride * 2)

    # --- scale cues ---
    cam_h = [(-views[n].floor_h * np.exp(poses[n][2])) for n in poses if views[n].floor_h is not None]
    if p.use_sheet is True:              # opt-in (D-067): --photo-param use_sheet=true
        sheet, sheet_status = SC.sheet_cues(views, poses, photos={ph.name: ph for ph in photos})
    else:
        sheet, sheet_status = [], "off (no reference object in the protocol, D-067)"
    log(f"[photo] paper sheet: {sheet_status}" + "".join(f"\n    {c['image']}: scale x{c['scale']:.3f} "
                                                        f"+-{100 * c['sigma_rel']:.1f}%" for c in sheet))
    # v2: a guessed field of view (no EXIF focal) is a scale error of the same order as the FOV error: measured on
    # phone_like/photos_no_exif, ceiling 3.59 m vs 2.93 m with EXIF (+22%) while v1 still reported sigma 3.8%
    no_focal = any(not str(f_src[n]).startswith("exif") for n in poses)   # exif_f35 or exif_focal_sensor
    cues = [dict(cue="learned_depth_moge2", scale=1.0,
                 sigma_rel=p.scale_sigma_no_focal if no_focal else p.scale_model_sigma, focal_guessed=no_focal),
            SC.camera_height_cue(cam_h), *sheet]
    sc = SC.fuse([c for c in cues if c])
    k = sc["scale"]
    poses = {n: (th, k * t, ls + np.log(k)) for n, (th, t, ls) in poses.items()}

    # --- dense points in the common frame, then the LiDAR tier's gravity + Manhattan alignment ---
    P, N, C, rng, lab = _cloud(views, poses, p.raw_stride)
    cams = np.array([poses[n][1] for n in names if n in poses])
    cfg = Config()
    Ps, Ns, Cs, inv = _voxel_mean(P, N, C, p.voxel_m)
    T_align, Pa, Na, ainfo = align_scene(Ps, Ns, cams, cfg)
    Ra, ta = T_align[:3, :3], T_align[:3, 3]
    raw_key = np.floor(P / p.raw_voxel_m).astype(np.int64)
    _, keep = np.unique(raw_key, axis=0, return_index=True)
    room_idx = {r: i for i, r in enumerate(rooms)}
    point_room_full = np.array([room_idx[views[n].room] for n in lab], np.int16)
    surf_room = np.zeros(len(Ps), np.int16)
    surf_room[inv] = point_room_full          # last writer wins: any photo's room that saw the voxel

    order = [n for n in names if n in poses]
    T_wc = []
    for n in order:
        T = np.eye(4)
        T[:3, :3] = Ra @ L._ry(poses[n][0]) @ views[n].R_lev
        T[:3, 3] = Ra @ poses[n][1] + ta
        T_wc.append(T)
    scene = dict(points=Pa.astype(np.float32), normals=Na.astype(np.float32), colors=Cs,
                 raw_points=(P[keep] @ Ra.T + ta).astype(np.float32), raw_range=rng[keep].astype(np.float32),
                 raw_ray=_rays(P[keep], lab[keep], poses, Ra),
                 T_align=T_align, traj=np.array([T[:3, 3] for T in T_wc], np.float32),
                 kf=np.arange(len(order)), T_wc=np.array(T_wc), cam_names=np.array(order),
                 cam_room=np.array([views[n].room for n in order]), room_names=np.array(rooms),
                 point_room=surf_room, raw_point_room=point_room_full[keep],
                 cam_placement=np.array([placement[n] for n in order]))
    # final per-photo metric depth (after link + scale corrections): kept for evaluation and for damage mapping
    np.savez_compressed(work / "depth_final.npz", **{n.replace("/", "__"): (views[n].depth * np.exp(poses[n][2]))
                                                    .astype(np.float32) for n in order})
    info = _info(rooms, names, views, poses, placement, edges, comps, dropped, sc, sheet_status, ainfo, sfm_stats,
                 f_src, p, t0)
    info["pruned_edges"] = pruned
    info["semantic"] = sem_info
    info["protocol_v2"] = dict(doorway_pairs=proto["pair_info"], spin_groups=proto["spin_info"],
                               pair_checks=proto["pair_checks"],
                               layout_registration=proto.get("layout", []), door_match=door_match_log,
                               pair_bridge=proto.get("pair_bridge", []),
                               photo_link_kinds=_photo_link_kinds(edges, poses))
    info["room_links_v2"] = _room_links(edges, views, poses)
    if p.room_layouts and p.spin_prior:            # v3: each room's own layout from its spin (layout.py)
        from floorplan.photo.layout import room_layouts
        try:
            into = {}
            for pr in proto.get("pairs", []):          # the pair photo that looks INTO room X was taken on X's door
                for n in (pr["a"], pr["b"]):
                    if n in views:
                        into.setdefault(views[n].room, n)
            pair_photos = {n for pr in proto.get("pairs", []) for n in (pr["a"], pr["b"])}
            from floorplan.photo.layout import ceiling_photo_scales
            csc = ceiling_photo_scales(views, matches, proto.get("ceiling", {}))
            log("[photo/v3] ceiling photos linked to their room by matches (D-057): " + (", ".join(
                f"{c}: x{v['scale']:.3f} from {v['refs']} photo(s), {v['matches']} matches" for c, v in csc.items())
                or "none"))
            info["ceiling_photo_scales"] = csc
            info["room_layouts"] = room_layouts(views, proto["spin_info"], poses, k, Ra, ta, p, log,
                                                ceiling_photos=proto.get("ceiling", {}), doorway_into=into,
                                                all_pair_photos=pair_photos, extra_photos=proto.get("extras", {}),
                                                ceiling_scale=csc)
        except Exception as e:                     # never lose the scene over the layout step
            info["room_layouts_error"] = f"{type(e).__name__}: {e}"
    info["intra_room_proposals"] = intra_stats
    info["edges"] = [dict(a=e.a, b=e.b, source=e.kind if e.kind != "features" else
                          ("features" if e.matches else "mapanything+icp"), inliers=e.inliers, matches=e.matches,
                          scale_ratio=float(np.exp(e.log_s)), yaw_deg=float(np.degrees(e.yaw)),
                          rmse_m=e.rmse_m, violation=e.violation, agreement=e.agreement, cross_room=e.cross_room,
                          note=e.note) for e in edges]
    log(f"[photo] scene: {len(Pa)} surface / {len(keep)} raw points, scale x{k:.3f} +- {100 * sc['sigma_rel']:.1f}%, "
        f"floor {ainfo['floor_y']}, ceiling {ainfo['ceiling_y']} ({time.time() - t0:.0f}s)")
    return scene, info


def _info(rooms, names, views, poses, placement, edges, comps, dropped, sc, sheet_status, ainfo, sfm_stats, f_src,
          p, t0) -> dict:
    links = {}
    for e in edges:
        if not e.cross_room:
            continue
        key = "|".join(sorted((views[e.a].room, views[e.b].room)))
        d = links.setdefault(key, dict(edges=0, inliers=0, photos=[]))
        d["edges"] += 1
        d["inliers"] += e.inliers
        d["photos"].append(f"{e.a}~{e.b}")
    resid = L.edge_residuals(poses, edges)
    room_status = {}
    for r in rooms:
        rn = [n for n in names if views[n].room == r]
        pl = [placement.get(n) for n in rn if n in poses]
        status = "linked" if "linked" in pl else ("door_match" if "door_match" in pl else
                                                  ("fallback" if pl else "not_placed"))
        widen = dict(linked=p.widen_linked, door_match=p.widen_door_match).get(status, p.widen_unlinked)
        room_status[r] = dict(photos=len(rn), placed=len(pl), placement=status, widen=widen)
    any_fallback = any(s["placement"] != "linked" for s in room_status.values())
    return dict(
        tier="photo", floor_y=ainfo["floor_y"], ceiling_y=ainfo["ceiling_y"],
        ceiling_height_global=ainfo["ceiling_height_global"], floor_tilt_before_deg=ainfo["floor_tilt_before_deg"],
        manhattan_yaw_deg=ainfo["manhattan_yaw_deg"], manhattan_score=ainfo["manhattan_score"],
        scale=dict(factor=sc["scale"], sigma_rel=sc["sigma_rel"], cues=sc["cues"], sheet_status=sheet_status),
        rooms=room_status, links=links, all_rooms_one_frame=not any_fallback,
        components=[[n for n in c] for c in comps], unplaceable_fragments=dropped,
        edge_residuals=dict(n=len(resid), median_dt_m=float(np.median([r["dt_m"] for r in resid])) if resid else None,
                            max_dt_m=float(max([r["dt_m"] for r in resid])) if resid else None,
                            median_dyaw_deg=float(np.median([r["dyaw_deg"] for r in resid])) if resid else None),
        per_photo_scale_correction={n: float(np.exp(poses[n][2] - np.log(sc["scale"]))) for n in poses},
        up_source={n: views[n].up_source for n in views}, focal_source=f_src, sfm=sfm_stats,
        interval_widen_factor=p.widen_unlinked if any_fallback else p.widen_linked,
        scale_sigma_rel=sc["sigma_rel"], params=p.to_dict(), runtime_s=round(time.time() - t0, 1))


# ------------------------------------------------------------------------------------- protocol v2 (D-017)
def _protocol_edges(photos, views, feat_edges, p, log) -> dict:
    """Spin-group priors and doorway-pair links (floorplan/photo/protocol.py) as extra pose-graph edges."""
    pairs = PR.doorway_pairs(photos, p.doorway_pair_max_dt_s, p.doorway_pair_rhythm) \
        if p.doorway_pair_max_dt_s > 0 else []
    groups = PR.spin_groups(photos, pairs)
    # protocol v2 (D-052): a photo tilted UP is the room's ceiling photo, not part of the turning spin. The protocol
    # takes it last, and a real phone's ceiling shot can be tilted up only 16-17 deg (own home), so the last photo of
    # the series counts from 10 deg; photos mid-series keep 18 deg (turning frames reached 16.9 deg)
    ceiling = {}
    for room, g in list(groups.items()):
        if g is None:
            continue
        up = [n for n in g if n in views and (PR.photo_pitch_deg(views[n]) > p.ceiling_photo_min_pitch_deg or (
            n == g[-1] and PR.photo_pitch_deg(views[n]) > p.ceiling_photo_last_min_pitch_deg))]
        if up:
            ceiling[room] = up
            groups[room] = [n for n in g if n not in up]
    # protocol v2.1 (D-055): the ceiling photo closes the room's turning series. The 1-2 photos taken after it come
    # from other spots (a small room's second doorway photo, a balcony seen from the next room): they are not spin
    # photos. Only when the series before it has >= 2 photos and at most 2 follow (else the person simply took the
    # ceiling photo mid-series, and nothing changes).
    t_of = {ph.name: ph.time_s for ph in photos}
    extras = {}
    for room, up in ceiling.items():
        g = groups.get(room) or []
        t_c = min((t_of[n] for n in up if t_of.get(n) is not None), default=None)
        if t_c is None or any(t_of.get(n) is None for n in g):
            continue
        before, after = [n for n in g if t_of[n] < t_c], [n for n in g if t_of[n] > t_c]
        if after and len(before) >= 2 and len(after) <= p.extra_photos_max:
            extras[room] = after
            groups[room] = before
    manh = {n: PR.photo_manhattan(v) for n, v in views.items()}
    out, spin_info, pair_info = [], {}, []
    for room, g in groups.items():
        if not p.spin_prior:
            spin_info[room] = dict(photos=g, note="spin prior disabled (photos not declared as a spin)")
        elif g is None:
            spin_info[room] = dict(photos=None, note="no EXIF capture times: spin prior not applied")
        elif len(g) < 2:
            spin_info[room] = dict(photos=g, note="fewer than 2 spin photos")
        else:
            es, si = PR.spin_edges(g, manh, feat_edges, p)
            out += es
            spin_info[room] = si
    for pr in pairs:
        e, pi = PR.pair_edge_prior(pr, manh, p)
        out.append(e)
        pair_info.append(pi)
    n_time = sum(ph.time_s is not None for ph in photos)
    log(f"[photo/v2] {n_time}/{len(photos)} photos with capture time; {len(pairs)} doorway pair(s); "
        f"{sum(e.kind == 'spin_prior' for e in out)} spin prior edges in "
        f"{sum(1 for v in spin_info.values() if 'direction' in v)} room(s); "
        f"{sum(m is not None for m in manh.values())}/{len(manh)} photos with a Manhattan yaw")
    if ceiling:
        log(f"[photo/v2] ceiling photos (tilted up > {p.ceiling_photo_min_pitch_deg:.0f} deg, or > "
            f"{p.ceiling_photo_last_min_pitch_deg:.0f} deg as the last photo of a series; kept out of the spins): "
            + ", ".join(f"{r}: {len(v)}" for r, v in sorted(ceiling.items())))
    if extras:
        log("[photo/v2] photos after the ceiling photo (other spots, kept out of the spins): "
            + ", ".join(f"{r}: {len(v)}" for r, v in sorted(extras.items())))
    return dict(edges=out, pairs=pairs, spin_info=spin_info, pair_info=pair_info, manh=manh, groups=groups,
                ceiling=ceiling, extras=extras)


def _layout_register(names, views, edges, proto, p, log):
    """A doorway photo whose component holds none of its room's spin photos is aligned to that room's walls."""
    report = []
    for pr in proto["pairs"]:
        for d in (pr["a"], pr["b"]):
            room = views[d].room
            comps = L.components(names, edges)
            comp_of = {n: i for i, c in enumerate(comps) for n in c}
            spin = [n for n in (proto["groups"].get(room) or []) if n != d]
            if not spin or any(comp_of[n] == comp_of[d] for n in spin):
                continue
            c = comps[comp_of[spin[0]]]
            poses = L.solve_component(c, [e for e in edges if e.a in c and e.b in c]) if len(c) > 1 else \
                {c[0]: (0.0, np.zeros(3), 0.0)}
            room_poses = {n: poses[n] for n in c if views[n].room == room}
            Pc, Nc, *_ = _cloud(views, room_poses, 4)
            res = PR.register_to_room(views[d], room_poses, views, proto["manh"].get(d), manhattan_yaw(Nc), p)
            entry = dict(photo=d, room=room, result=None if res is None else
                         {k: v for k, v in res.items() if k != "pose"})
            if res and res.get("accepted"):
                anchor = spin[0]
                edges = edges + [PR.layout_edge(anchor, room_poses[anchor], d, res["pose"], p,
                                                note=f"wall alignment score {res['score']} rival {res['rival']}")]
            report.append(entry)
    log(f"[photo/v2] layout registration: {sum(1 for r in report if r['result'] and r['result'].get('accepted'))}"
        f"/{len(report)} unmatched doorway photo(s) placed in their room")
    return edges, report


def _door_match(views, poses_block, comp_poses, P_block, lab_block, P_comp, block_yaw, comp_yaw, pairs, p) -> dict:
    """Door-matching fallback for a component no evidence links: doorway photos stand IN a door, so their camera
    centres are door positions. Try every (door of the component, unmatched door of the block, Manhattan rotation
    0/90/180/270) placement; keep the one with no overlap between the component and the block (top-view 10 cm
    occupancy of wall/floor points). Accept only a unique, non-overlapping solution; else 'beside the block'."""
    partner = {pr["a"]: pr["b"] for pr in pairs} | {pr["b"]: pr["a"] for pr in pairs}
    dc = [n for n in comp_poses if n in partner and partner[n] not in comp_poses]
    db = [n for n in poses_block if n in partner and partner[n] not in poses_block]
    log = dict(component=sorted(comp_poses), doors_component=dc, doors_block=db)
    if not dc or not db:
        return dict(ok=False, log=dict(log, result="no door positions to match"))
    cell = 0.10
    occ_b = {tuple(k) for k in np.floor(P_block[:, [0, 2]] / cell).astype(int)}
    cands = []
    for a in dc:
        for b in db:
            for k in range(4):
                dyaw = block_yaw - comp_yaw + k * np.pi / 2
                ca = L._ry(dyaw) @ comp_poses[a][1]
                shift = poses_block[b][1] - ca
                Q = P_comp @ L._ry(dyaw).T + shift
                keys = {tuple(x) for x in np.floor(Q[:, [0, 2]] / cell).astype(int)}
                ov = len(keys & occ_b) / max(len(keys), 1)
                cands.append((ov, a, b, k, dyaw, shift))
    cands.sort(key=lambda c: c[0])
    best = cands[0]
    second = cands[1][0] if len(cands) > 1 else 1.0
    out = dict(log, best_overlap=round(best[0], 3), second_overlap=round(second, 3),
               door_pair=[best[1], best[2]], rotation_deg=90 * best[3])
    ok = best[0] < 0.05 and second > 2 * max(best[0], 0.02)
    return dict(ok=ok, dyaw=best[4], shift=best[5],
                log=dict(out, result="placed by door match" if ok else "ambiguous or overlapping: beside block"))


def _photo_link_kinds(edges, poses) -> dict:
    out = {}
    for e in edges:
        for n in (e.a, e.b):
            if n in poses:
                out.setdefault(n, set()).add(e.kind)
    return {n: sorted(k) for n, k in out.items()}


def _room_links(edges, views, poses) -> dict:
    """Room-level links by kind of evidence (features across rooms, doorway pairs) among placed photos."""
    links = {}
    for e in edges:
        ra, rb = views[e.a].room, views[e.b].room
        if ra == rb or e.a not in poses or e.b not in poses:
            continue
        key = "|".join(sorted((ra, rb)))
        links.setdefault(key, {}).setdefault(e.kind, 0)
        links[key][e.kind] += 1
    return links


def _verify_pairs(names, views, edges, p, log):
    """Drop cross-room edges (doorway pairs AND feature links) whose solved placement makes the two ROOMS contradict
    each other (no-overlap constraint, checked with every photo of both rooms, worst first). Why also features:
    measured on floor_only/rotate, two mutually consistent false feature edges (repeated structure, 90 deg off,
    6.6-7.2 m long) passed the pairwise checks and made loop pruning remove a CORRECT 48-inlier edge.

    A doorway pair whose Manhattan-snapped yaw makes the rooms overlap first gets its other three Manhattan yaws
    tried (the photographer may not have turned a full 180 deg): the no-overlap constraint picks the yaw."""
    status = {}
    edges = _verify_phase(names, views, edges, p, "doorway_pair", status)
    report = [dict(a=a, b=b, kind=k, violation=v, kept=not (isinstance(v, str) and v.startswith("rejected")))
              for (a, b, k), v in status.items()]
    log(f"[photo/v2] cross-room links checked for room overlap: {sum(not r['kept'] for r in report)} rejected "
        f"of {len(report)}")
    return edges, report


def _features_need_pair(views, edges, p, status):
    """Under the D-017 protocol every opening the photographer walked through has a doorway pair. A cross-room
    FEATURE edge between two rooms that no doorway pair joins is therefore either seen through a room in between or
    a repeated structure. Measured on both rotate captures (13 cross-room feature edges, truth-labelled,
    room_check_discrimination.json): all 4 false edges joined rooms without a pair, all 9 true edges joined paired
    rooms; the dense room-overlap test did NOT separate them (false 0.06-1.0, true 0.0-1.0). Without any pair in the
    capture (no EXIF times, corner-style folders) feature edges are left as in v1."""
    pairs = {frozenset((views[e.a].room, views[e.b].room)) for e in edges if e.kind == "doorway_pair"}
    if not pairs or not p.features_need_pair:
        return edges
    keep = []
    for e in edges:
        if e.kind == "features" and e.cross_room and frozenset((views[e.a].room, views[e.b].room)) not in pairs:
            status[(e.a, e.b, e.kind)] = "rejected (no doorway pair between these rooms)"
            continue
        if e.kind == "features" and e.cross_room:
            status[(e.a, e.b, e.kind)] = "kept (rooms joined by a doorway pair)"
        keep.append(e)
    return keep


def _isolated(e, edges, views):
    """The link e with ONLY the intra-room edges of its two rooms: each cross-room link is judged on its own, so one
    false link cannot make a correct one look contradictory (measured: with the whole component solved, 6 correct
    cross-room feature edges, one with 285 inliers, were rejected on with_ceiling/rotate)."""
    rooms = {views[e.a].room, views[e.b].room}
    sub = [x for x in edges if not x.cross_room and views[x.a].room in rooms] + [e]
    c = next(c for c in L.components(sorted({n for x in sub for n in (x.a, x.b)}), sub) if e.a in c)
    return c, [x for x in sub if x.a in c and x.b in c]


def _verify_phase(names, views, edges, p, kind, status):
    while True:
        worst = None
        for c in L.components(names, edges):
            ce = [e for e in edges if e.a in c and e.b in c]
            check = [e for e in ce if e.cross_room and e.kind == kind]
            if not check:
                continue
            thr = p.pair_max_overlap_violation if kind == "doorway_pair" else p.edge_max_violation
            for e in check:
                c, ce = _isolated(e, edges, views)
                poses = L.solve_component(c, ce)
                v = _room_violation(e, poses, views)
                if v > thr and e.kind == "doorway_pair" and "yaw by no-overlap" not in e.note:
                    v = _repair_pair_yaw(e, c, ce, views, v, p)
                status[(e.a, e.b, e.kind)] = round(v, 3)
                if v > thr and (worst is None or v > worst[1]):
                    worst = (e, v)
        if worst is None:
            break
        edges = [e for e in edges if e is not worst[0]]
        status[(worst[0].a, worst[0].b, worst[0].kind)] = f"rejected ({worst[1]:.3f})"
    return edges


def _room_violation(e, poses, views) -> float:
    """Room-level contradiction of a cross-room link. A FEATURE edge claims cm-level geometry, so it gets the strict
    pairwise test (15%, no margin; caught the two false 90-deg R5~R6 edges at 0.50/0.34 while correct cross-room
    feature edges scored 0.0). A DOORWAY PAIR claims only ~0.2 m, so it gets the gross-error test (30% and 1.5 m)."""
    if e.kind == "features":
        return PR.pair_overlap_violation(e, poses, views, margin_m=0.0, violate_rel=0.15)
    return PR.pair_overlap_violation(e, poses, views)


def _repair_pair_yaw(e, comp, comp_edges, views, v0, p) -> float:
    """Try the pair's yaw + 90/180/270 deg; keep the one with the least room overlap (if it passes)."""
    best = (v0, e.yaw)
    y0 = e.yaw
    for k in (1, 2, 3):
        e.yaw = y0 + k * np.pi / 2
        poses = L.solve_component(comp, comp_edges)
        v = _room_violation(e, poses, views)
        if v < best[0]:
            best = (v, e.yaw)
    e.yaw = best[1]
    if best[1] != y0:
        e.note += f"; yaw by no-overlap: {np.degrees(best[1]):+.0f} deg (snap gave {np.degrees(y0):+.0f})"
        e.sig_yaw_deg = p.pair_sigma_yaw_deg
    else:
        e.note += "; yaw by no-overlap: unchanged"
    return best[0]


def _pair_bridge(names, views, edges, proto, p, log):
    """Door matching through an isolated doorway pair (the D-017 fallback that uses doors, not features).

    Case: a doorway pair (a in room A, b in room B) forms its own little component because neither photo shares
    features with its room's spin group, while A's and B's spin groups sit in two DIFFERENT components. Each
    photo's wall-layout registration alone is ambiguous (several placements along a wall score alike), but the pair
    couples them: for every candidate placement of a in A and of b in B, the pair fixes where B's whole component
    goes relative to A's. Rooms must not overlap, so the joint hypothesis with no room overlap is far more
    constrained than either side. Accepted only if exactly one joint hypothesis passes; added as two 'layout' edges
    (placement flagged and widened as layout-registered)."""
    report = []
    for pr in proto["pairs"]:
        a, b = pr["a"], pr["b"]
        comps = L.components(names, edges)
        cid = {n: i for i, c in enumerate(comps) for n in c}
        ga = [n for n in (proto["groups"].get(views[a].room) or []) if n in cid]
        gb = [n for n in (proto["groups"].get(views[b].room) or []) if n in cid]
        if not ga or not gb or cid[a] != cid[b] or len(comps[cid[a]]) != 2:
            continue
        ca, cb = comps[cid[ga[0]]], comps[cid[gb[0]]]
        if cid[ga[0]] == cid[gb[0]]:
            continue
        pe = next((e for e in edges if {e.a, e.b} == {a, b} and e.kind == "doorway_pair"), None)
        if pe is None:
            continue
        sol = {}
        for side, comp, d in (("a", ca, a), ("b", cb, b)):
            poses = L.solve_component(comp, [e for e in edges if e.a in comp and e.b in comp]) if len(comp) > 1 \
                else {comp[0]: (0.0, np.zeros(3), 0.0)}
            rp = {n: poses[n] for n in comp if views[n].room == views[d].room}
            _, Nc, *_ = _cloud(views, rp, 4)
            r = PR.register_to_room(views[d], rp, views, proto["manh"].get(d), manhattan_yaw(Nc), p, top_k=6)
            sol[side] = (poses, r["candidates"] if r else [])
        entry = dict(pair=[a, b], candidates=[len(sol["a"][1]), len(sol["b"][1])], joint=[])
        ok = []
        Pa = _cloud(views, sol["a"][0], 6)[0]
        occ_a = {tuple(k) for k in np.floor(Pa[:, [0, 2]] / 0.1).astype(int)}
        for ha in sol["a"][1]:
            # b's pose in A's frame from the pair edge: theta_b = theta_a + yaw, shared centre
            th_b = ha["pose"][0] + (pe.yaw if pe.a == a else -pe.yaw)
            for hb in sol["b"][1]:
                dyaw = th_b - hb["pose"][0]
                shift = ha["pose"][1] - L._ry(dyaw) @ hb["pose"][1]
                moved = {n: (th + dyaw, L._ry(dyaw) @ t + shift, ls) for n, (th, t, ls) in sol["b"][0].items()}
                Pb = _cloud(views, moved, 6)[0]
                kb = {tuple(k) for k in np.floor(Pb[:, [0, 2]] / 0.1).astype(int)}
                ov = len(kb & occ_a) / max(len(kb), 1)
                entry["joint"].append(dict(score=round(ha["score"] + hb["score"], 3), overlap=round(ov, 3)))
                if ov < p.bridge_max_overlap:
                    ok.append((ha["score"] + hb["score"], ha, hb, ov))
        entry["passing"] = len(ok)
        if len(ok) == 1 or (len(ok) > 1 and sorted(o[0] for o in ok)[-2] < p.layout_max_rival_ratio * max(o[0] for o in ok)):
            _, ha, hb, ov = max(ok, key=lambda o: o[0])
            pa_anchor, pb_anchor = ga[0], gb[0]
            edges = edges + [
                PR.layout_edge(pa_anchor, sol["a"][0][pa_anchor], a, ha["pose"], p, note=f"pair bridge, overlap {ov:.3f}"),
                PR.layout_edge(pb_anchor, sol["b"][0][pb_anchor], b, hb["pose"], p, note=f"pair bridge, overlap {ov:.3f}")]
            entry["accepted"] = True
        else:
            entry["accepted"] = False
        entry["joint"] = sorted(entry["joint"], key=lambda j: (j["overlap"], -j["score"]))[:6]
        report.append(entry)
    log(f"[photo/v2] pair bridge: {sum(r['accepted'] for r in report)}/{len(report)} isolated doorway pair(s) used")
    return edges, report
