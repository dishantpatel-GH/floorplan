"""Geometry for the damage module: surfaces, posed views, occlusion, and rectified surface textures (orthophotos).

Why orthophotos: damage has to be reported in METRES on a named SURFACE ("R2-W3, 0.42 m2, 0.35 m above the floor").
If every keyframe that sees a wall is warped onto that wall's plane at a fixed pixel size (5 mm by default), then
  * one pixel = 25 mm2 everywhere, so area and extent are pixel counts times a constant;
  * the same physical spot lands on the same ortho pixel in every view, so a per-pixel MEDIAN over views removes
    things that are not on the wall (specular highlights, people, a reflection that moves with the camera);
  * the per-view stack lets us check multi-view consistency: real damage is seen in most views, glare is not.

Conventions (shared with the rest of the repo):
  * aligned world frame, +y up, plan coordinates (u, v) = (x, z);
  * poses are camera-to-world with OpenCV camera axes (x right, y down, z forward);
  * a Views object is tier-agnostic: LiDAR, video and photo front-ends all reduce to images + K + T_wc (aligned).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from matplotlib.path import Path as MplPath


# ----------------------------------------------------------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Surface:
    """A planar rectangle (optionally masked by a polygon) in the aligned world.

    A point with surface coordinates (s, t) in metres is origin + s * ax_s + t * ax_t.
    For walls ax_t is +y, so t is the height above the room's floor; for ceilings/floors (s, t) = (x, z)."""
    id: str
    kind: str                         # wall | ceiling | floor
    room_id: str
    origin: np.ndarray                # (3,)
    ax_s: np.ndarray                  # (3,) unit
    ax_t: np.ndarray                  # (3,) unit
    normal: np.ndarray                # (3,) unit, pointing INTO the room (towards the camera side)
    size: tuple[float, float]         # (S, T) metres
    polygon_st: np.ndarray | None = None   # (k,2) valid region in (s,t); None = whole rectangle
    holes_st: list = field(default_factory=list)   # list of (s0, t0, s1, t1) rectangles to ignore (openings)
    meta: dict = field(default_factory=dict)

    def to_world(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.origin + s[..., None] * self.ax_s + t[..., None] * self.ax_t

    def area(self) -> float:
        if self.polygon_st is None:
            return self.size[0] * self.size[1]
        x, y = self.polygon_st[:, 0], self.polygon_st[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def grid_mask(self, res: float) -> np.ndarray:
        """Boolean (H, W) mask of ortho pixels that belong to the surface (polygon minus openings).
        Ortho row 0 is the TOP of a wall (largest t) so figures read like a photo of the wall."""
        W, H = int(np.ceil(self.size[0] / res)), int(np.ceil(self.size[1] / res))
        s, t = self.pixel_st(np.arange(W)[None, :].repeat(H, 0), np.arange(H)[:, None].repeat(W, 1), res)
        m = np.ones((H, W), bool)
        if self.polygon_st is not None:
            m = MplPath(self.polygon_st).contains_points(np.c_[s.ravel(), t.ravel()]).reshape(H, W)
        for s0, t0, s1, t1 in self.holes_st:
            m &= ~((s >= s0) & (s <= s1) & (t >= t0) & (t <= t1))
        return m

    def st_to_pixel(self, s, t, res: float):
        """Inverse of pixel_st: surface (s, t) metres -> fractional ortho (col, row)."""
        col = np.asarray(s, float) / res - 0.5
        if self.kind == "wall":
            row = (self.size[1] - np.asarray(t, float)) / res - 0.5
        else:
            row = np.asarray(t, float) / res - 0.5
        return col, row

    def pixel_st(self, col, row, res: float):
        """Ortho pixel (col,row) -> surface (s,t). Walls: row 0 = top. Horizontal surfaces: row 0 = t=0."""
        s = (np.asarray(col, float) + 0.5) * res
        if self.kind == "wall":
            t = self.size[1] - (np.asarray(row, float) + 0.5) * res
        else:
            t = (np.asarray(row, float) + 0.5) * res
        return s, t


def _wall_surface(w: dict, floor_y: float, height: float) -> Surface | None:
    p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 0.25:                      # jogs of a few cm carry no usable texture
        return None
    d = (p1 - p0) / L
    return Surface(id=w["id"], kind="wall", room_id=w["room_id"],
                   origin=np.array([p0[0], floor_y, p0[1]]), ax_s=np.array([d[0], 0.0, d[1]]),
                   ax_t=np.array([0.0, 1.0, 0.0]),
                   normal=np.array([w["normal"][0], 0.0, w["normal"][1]], float), size=(L, height))


def surfaces_from_plan(plan: dict, scene: dict, include_floor: bool = False,
                       default_height: float = 2.4) -> list[Surface]:
    """Walls, ceilings (only where the plan says the ceiling was measured) and optionally floors of every room.

    Openings (doors/windows) are cut out of walls: a closed wooden door lies IN the wall plane and is brown; without
    the cut it would look exactly like a giant water stain."""
    rooms = {r["id"]: r for r in plan["rooms"]}
    openings = {o["id"]: o for o in plan.get("openings", [])}
    out: list[Surface] = []
    for w in plan["walls"]:
        r = rooms.get(w["room_id"])
        if r is None:
            continue
        ch = r["ceiling_height"]
        H = ch["value"] if ch.get("value") else _observed_height(scene, r["floor_level"], default_height)
        srf = _wall_surface(w, r["floor_level"], H)
        if srf is None:
            continue
        srf.meta.update(ceiling_observed=bool(ch.get("value")), room_label=r.get("label", ""))
        for oid in w.get("opening_ids", []):
            o = openings.get(oid)
            if o is None:
                continue
            c = np.array([o["center"][0], 0.0, o["center"][1]]) - srf.origin
            sc = float(c @ srf.ax_s)
            half = 0.5 * (o["width"]["value"] or 0.9) + 0.05
            if o["kind"] == "window":
                sill = (o.get("sill_height") or {}).get("value") or 0.8
                hh = (o.get("height") or {}).get("value") or 1.2
                srf.holes_st.append((sc - half, sill - 0.05, sc + half, sill + hh + 0.05))
            else:
                hh = (o.get("height") or {}).get("value") or 2.1
                srf.holes_st.append((sc - half, -1.0, sc + half, hh + 0.05))
            if o["kind"] == "window":
                corners_t = [sill, sill + hh]
            else:
                corners_t = [hh]
            srf.meta.setdefault("openings", []).append(dict(id=oid, kind=o["kind"], s_center=sc, half_width=half - 0.05,
                                                            corners_t=corners_t,
                                                            sill_t=sill if o["kind"] == "window" else None))
        out.append(srf)
    for r in plan["rooms"]:
        poly = np.asarray(r["polygon"], float)
        if len(poly) < 3:
            continue
        lo = poly.min(0)
        size = tuple((poly.max(0) - lo).tolist())
        ch = r["ceiling_height"]
        if ch.get("value"):
            out.append(Surface(id=f"{r['id']}-C", kind="ceiling", room_id=r["id"],
                               origin=np.array([lo[0], r["floor_level"] + ch["value"], lo[1]]),
                               ax_s=np.array([1.0, 0, 0]), ax_t=np.array([0, 0, 1.0]), normal=np.array([0, -1.0, 0]),
                               size=size, polygon_st=poly - lo, meta=dict(room_label=r.get("label", ""))))
        if include_floor:
            out.append(Surface(id=f"{r['id']}-F", kind="floor", room_id=r["id"],
                               origin=np.array([lo[0], r["floor_level"], lo[1]]),
                               ax_s=np.array([1.0, 0, 0]), ax_t=np.array([0, 0, 1.0]), normal=np.array([0, 1.0, 0]),
                               size=size, polygon_st=poly - lo, meta=dict(room_label=r.get("label", ""))))
    return out


def _observed_height(scene: dict, floor_y: float, default: float) -> float:
    """Wall height when the ceiling was not measured: 98th percentile of surface points above the floor (the
    texture above that was never seen anyway)."""
    y = scene["points"][:, 1] - floor_y
    y = y[(y > 0.1) & (y < 4.0)]
    return float(np.percentile(y, 98)) if len(y) > 100 else default


def surfaces_from_scene(scene: dict, info: dict, min_len: float = 0.6, min_pts: int = 600,
                        max_gap: float = 0.3) -> list[Surface]:
    """Fallback when no plan exists yet: axis-aligned wall planes straight from the aligned TSDF surface.

    The scene is Manhattan-aligned (walls along x or z), so a wall is a peak in the histogram of x (or z) of the
    points whose normal is +-x (or +-z). Segments separated by more than `max_gap` are split. Ids are P1..Pn and are
    re-keyed to plan wall ids later (nearest wall) when a plan becomes available."""
    P, N = scene["points"], scene["normals"]
    floor_y = info.get("floor_y")
    floor_y = float(np.percentile(P[:, 1], 1)) if floor_y is None else float(floor_y)
    ceil_y = info.get("ceiling_y")
    H = (float(ceil_y) - floor_y) if ceil_y else _observed_height(scene, floor_y, 2.4)
    band = (P[:, 1] > floor_y + 0.15) & (P[:, 1] < floor_y + min(H, 2.2))
    out: list[Surface] = []
    k = 0
    for axis, other in ((0, 2), (2, 0)):
        for sign in (1.0, -1.0):
            sel = band & (N[:, axis] * sign > 0.9)
            c = P[sel, axis]
            if len(c) < min_pts:
                continue
            bins = np.arange(c.min() - 0.02, c.max() + 0.04, 0.02)
            hist, edges = np.histogram(c, bins)
            for i in np.argsort(-hist):
                if hist[i] < min_pts:
                    break
                if i > 0 and hist[i - 1] > hist[i] or i + 1 < len(hist) and hist[i + 1] > hist[i]:
                    continue                      # not a local maximum
                m = sel & (np.abs(P[:, axis] - 0.5 * (edges[i] + edges[i + 1])) < 0.03)
                pos = float(np.median(P[m, axis]))
                o = np.sort(P[m, other])
                splits = np.where(np.diff(o) > max_gap)[0]
                for seg in np.split(o, splits + 1):
                    if len(seg) < min_pts // 3 or seg[-1] - seg[0] < min_len:
                        continue
                    k += 1
                    a0, a1 = float(seg[0]), float(seg[-1])
                    org = np.zeros(3)
                    org[axis], org[other], org[1] = pos, a0, floor_y
                    ax = np.zeros(3)
                    ax[other] = 1.0
                    nrm = np.zeros(3)
                    nrm[axis] = sign
                    out.append(Surface(id=f"P{k}", kind="wall", room_id="S", origin=org, ax_s=ax,
                                       ax_t=np.array([0, 1.0, 0]), normal=nrm, size=(a1 - a0, H),
                                       meta=dict(source="scene_fallback", ceiling_observed=bool(ceil_y))))
    if ceil_y:
        sel = (N[:, 1] < -0.9) & (np.abs(P[:, 1] - ceil_y) < 0.05)
        if sel.sum() > min_pts:
            q = P[sel][:, [0, 2]]
            lo, hi = q.min(0), q.max(0)
            out.append(Surface(id="S-C", kind="ceiling", room_id="S", origin=np.array([lo[0], ceil_y, lo[1]]),
                               ax_s=np.array([1.0, 0, 0]), ax_t=np.array([0, 0, 1.0]), normal=np.array([0, -1.0, 0]),
                               size=tuple((hi - lo).tolist()), meta=dict(source="scene_fallback")))
    return out


def tile_surface(srf: Surface, max_side: float = 4.0) -> list[Surface]:
    """Split very large surfaces (corridor walls, whole-apartment ceilings) into <= max_side tiles to bound memory.
    Tiles keep the parent id in meta so regions can be reported against the parent surface."""
    ns = max(1, int(np.ceil(srf.size[0] / max_side)))
    nt = max(1, int(np.ceil(srf.size[1] / max_side))) if srf.kind != "wall" else 1
    if ns == nt == 1:
        return [srf]
    out = []
    ds, dt = srf.size[0] / ns, srf.size[1] / nt
    for i in range(ns):
        for j in range(nt):
            t0 = j * dt
            org = srf.origin + i * ds * srf.ax_s + t0 * srf.ax_t
            poly = None if srf.polygon_st is None else srf.polygon_st - np.array([i * ds, t0])
            holes = [(a - i * ds, b - t0, c - i * ds, d - t0) for a, b, c, d in srf.holes_st]
            out.append(Surface(id=srf.id, kind=srf.kind, room_id=srf.room_id, origin=org, ax_s=srf.ax_s,
                               ax_t=srf.ax_t, normal=srf.normal, size=(ds, dt if srf.kind != "wall" else srf.size[1]),
                               polygon_st=poly, holes_st=holes,
                               meta=dict(srf.meta, tile=(i, j), tile_offset=(i * ds, t0), parent_size=srf.size)))
    return out


def sub_surface(srf: Surface, s0: float, t0: float, s1: float, t1: float) -> Surface:
    """The (s0..s1, t0..t1) window of a surface as its own Surface (v2: used for the fine crack orthophoto).

    Axes and normal are shared, the origin moves to (s0, t0), polygon and openings are shifted, and
    meta['tile_offset'] accumulates so that describe() still reports positions in the PARENT surface's (s, t)."""
    s0, t0 = max(s0, 0.0), max(t0, 0.0)
    s1, t1 = min(s1, srf.size[0]), min(t1, srf.size[1])
    off = srf.meta.get("tile_offset", (0.0, 0.0))
    poly = None if srf.polygon_st is None else srf.polygon_st - np.array([s0, t0])
    holes = [(a - s0, b - t0, c - s0, d - t0) for a, b, c, d in srf.holes_st]
    return Surface(id=srf.id, kind=srf.kind, room_id=srf.room_id, origin=srf.origin + s0 * srf.ax_s + t0 * srf.ax_t,
                   ax_s=srf.ax_s, ax_t=srf.ax_t, normal=srf.normal, size=(s1 - s0, t1 - t0), polygon_st=poly,
                   holes_st=holes, meta=dict(srf.meta, tile_offset=(off[0] + s0, off[1] + t0), window=True))


# ----------------------------------------------------------------------------------------------------------------------
# Views (tier-agnostic posed images)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Views:
    images: list[np.ndarray]          # BGR uint8, possibly different sizes
    K: np.ndarray                     # (N,3,3) intrinsics at each image's own resolution
    T_wc: np.ndarray                  # (N,4,4) camera-to-ALIGNED-world, OpenCV axes
    names: list[str]
    tier: str = "lidar"
    pose_sigma_m: float = 0.005       # typical registration error between views (sets the position interval)
    scale_sigma_rel: float = 0.0      # 1-sigma relative scale error of the scene (0 for LiDAR; video/photo > 0)

    @property
    def n(self) -> int:
        return len(self.images)

    def save(self, out_dir: Path, quality: int = 95) -> None:
        out_dir = Path(out_dir)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        for name, img in zip(self.names, self.images):
            cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        meta = dict(tier=self.tier, pose_sigma_m=self.pose_sigma_m, scale_sigma_rel=self.scale_sigma_rel,
                    views=[dict(name=n, K=k.tolist(), T_wc=t.tolist()) for n, k, t in zip(self.names, self.K, self.T_wc)])
        (out_dir / "views.json").write_text(json.dumps(meta))

    @staticmethod
    def load(views_dir: Path) -> "Views":
        views_dir = Path(views_dir)
        meta = json.loads((views_dir / "views.json").read_text())
        names = [v["name"] for v in meta["views"]]
        imgs = [cv2.imread(str(views_dir / "images" / f"{n}.jpg")) for n in names]
        return Views(images=imgs, K=np.array([v["K"] for v in meta["views"]]),
                     T_wc=np.array([v["T_wc"] for v in meta["views"]]), names=names, tier=meta.get("tier", "lidar"),
                     pose_sigma_m=meta.get("pose_sigma_m", 0.005), scale_sigma_rel=meta.get("scale_sigma_rel", 0.0))


def views_from_stray(capture_dir: Path, scene: dict, max_views: int = 160, width: int = 960) -> Views:
    """LiDAR tier: keyframes of the Stray capture, poses moved into the aligned frame with the scene's T_align.

    Frames are kept in the stored (sensor) orientation: K refers to that orientation, so no rotation is needed for
    projection. Images are downscaled to `width` (2.5 mm/px at 2 m, finer than the 5 mm ortho grid)."""
    from floorplan.io.stray import load_stray
    cap = load_stray(capture_dir)
    kf = np.asarray(scene["kf"])
    if len(kf) > max_views:
        kf = kf[np.linspace(0, len(kf) - 1, max_views).round().astype(int)]
    T = np.einsum("ij,njk->nik", scene["T_align"], scene["T_wc"][kf])
    s = width / cap.rgb_size[0]
    imgs, Ks, names = [], [], []
    for i, bgr in cap.iter_rgb(kf):
        imgs.append(cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        K = cap.K_rgb[i].copy()
        K[:2] *= s
        Ks.append(K)
        names.append(f"{i:06d}")
    keep = np.searchsorted(kf, [int(n) for n in names])
    return Views(images=imgs, K=np.array(Ks), T_wc=T[keep], names=names, tier="lidar", pose_sigma_m=0.005)


def views_from_video(video_out_dir: Path, scene: dict, info: dict, max_views: int = 160) -> Views:
    """Video tier (floorplan.video): the upright keyframes it exported, with its focal length and metric poses.

    The relative scale uncertainty of the video scene (info['scale']['sigma_rel_total']) is carried into every
    damage extent interval: a 3% scale error is a 6% area error."""
    video_out_dir = Path(video_out_dir)
    rot = info.get("rotation", {})
    sub = "frames_flipped" if rot.get("flipped_by_pitch_check") else "frames"
    files = sorted((video_out_dir / "work" / sub / "sfm").glob("*.jpg"))
    kf = np.asarray(scene["kf"])
    if len(files) != len(kf):
        raise ValueError(f"{len(files)} keyframe images but {len(kf)} keyframes in the video scene")
    idx = np.arange(len(kf))
    if len(idx) > max_views:
        idx = idx[np.linspace(0, len(idx) - 1, max_views).round().astype(int)]
    cal = info["calibration"]
    up_w = cal["upright_size"][0]
    imgs, Ks = [], []
    for i in idx:
        img = cv2.imread(str(files[i]))
        h, w = img.shape[:2]
        f = cal["f_full_px"] * w / up_w
        imgs.append(img)
        Ks.append(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]))
    T = np.einsum("ij,njk->nik", scene["T_align"], scene["T_wc"][kf[idx]])
    sig = float(info.get("scale", {}).get("sigma_rel_total", 0.03))
    return Views(images=imgs, K=np.array(Ks), T_wc=T, names=[files[i].stem for i in idx], tier="video",
                 pose_sigma_m=0.02, scale_sigma_rel=sig)


def views_from_video_file(video: Path, scene: dict, info: dict, max_views: int = 160,
                          long_side: int = 1280) -> Views:
    """Video tier (v2), straight from the video file: no dependence on the front-end's work directory.

    The video scene's `kf` are frame indices of the video and its poses refer to the UPRIGHT frame
    (info['rotation']['final_rotation'], which already includes the 180-degree pitch-check flip). Decoding the
    keyframes again at 1280 px (instead of the 1024 px SfM copies) also gives the crack search finer pixels.
    The focal length is the front-end's full-resolution one scaled to the decoded size; principal point at centre
    (same model as views_from_video)."""
    from floorplan.video.frames import iter_frames, resolve_video, upright
    video = resolve_video(video)
    rot = int(info.get("rotation", {}).get("final_rotation", info.get("rotation", {}).get("rotation", 0)) or 0)
    cal = info["calibration"]
    up_w = cal["upright_size"][0]
    kf = np.asarray(scene["kf"])
    idx = np.arange(len(kf))
    if len(idx) > max_views:
        idx = idx[np.linspace(0, len(idx) - 1, max_views).round().astype(int)]
    want = {int(kf[i]): i for i in idx}
    imgs, Ks, names, keep = [], [], [], []
    for fi, bgr in iter_frames(video, sorted(want)):
        up = upright(bgr, rot)
        h, w = up.shape[:2]
        s = min(1.0, long_side / max(h, w))
        if s < 1:
            up = cv2.resize(up, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = up.shape[:2]
        f = cal["f_full_px"] * w / up_w
        imgs.append(up)
        Ks.append(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]))
        names.append(f"{fi:06d}")
        keep.append(want[fi])
    T = np.einsum("ij,njk->nik", scene["T_align"], scene["T_wc"][kf[np.array(keep, int)]])
    sig = float(info.get("scale", {}).get("sigma_rel_total", info.get("scale_sigma_rel", 0.03)) or 0.03)
    return Views(images=imgs, K=np.array(Ks).reshape(-1, 3, 3), T_wc=T.reshape(-1, 4, 4), names=names, tier="video",
                 pose_sigma_m=0.02, scale_sigma_rel=sig)


def views_from_photos(photo_root: Path, scene: dict, info: dict, long_side: int = 2048) -> Views:
    """Photo tier adapter (v2): the original photos + the photo front-end's poses.

    * Poses: the photo scene's `T_wc` are ALREADY in the aligned frame (floorplan/photo/frontend.py composes the
      Manhattan alignment into them), unlike the LiDAR/video scenes whose T_wc must be multiplied by T_align.
    * Images: re-loaded from the photo folders (EXIF-upright, HEIC ok) at up to 2048 px. The front-end only kept
      518 px depth copies; 2048 px of a phone photo is ~1.5 mm/px at 3 m, fine enough for crack search.
    * K: the front-end's own model (EXIF 35 mm focal on the long side, else a 70 deg default; principal point at the
      centre), recomputed at the loaded size so it matches the poses exactly.
    * Photos the front-end could not place have no pose and are skipped (recorded in the run log by the caller).
    * pose_sigma: 3 cm for photos linked by feature matches; the placement of unlinked rooms is a guess, but the plan
      walls of such a room come from the SAME photo's depth, so the projection onto them stays self-consistent."""
    from floorplan.photo.images import focal_px, list_rooms, load_photo, resize_long
    names = [str(n) for n in scene.get("cam_names", [])]
    files = {f"{room}/{f.stem}.jpg": (room, f) for room, fs in list_rooms(Path(photo_root)).items() for f in fs}
    hfov = float(info.get("params", {}).get("default_hfov_deg", 70.0))
    imgs, Ks, Ts, keep = [], [], [], []
    for i, n in enumerate(names):
        if n not in files:
            continue
        ph = load_photo(files[n][1], files[n][0])
        rgb = resize_long(ph.rgb, long_side) if max(ph.rgb.shape[:2]) > long_side else ph.rgb
        h, w = rgb.shape[:2]
        f, _ = focal_px(ph, max(h, w), hfov)
        imgs.append(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
        Ks.append(np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1.0]]))
        Ts.append(np.asarray(scene["T_wc"][i], float))
        keep.append(n.replace("/", "__").replace(".jpg", ""))
    sig = float(info.get("scale_sigma_rel", info.get("scale", {}).get("sigma_rel", 0.05)) or 0.05)
    return Views(images=imgs, K=np.array(Ks).reshape(-1, 3, 3), T_wc=np.array(Ts).reshape(-1, 4, 4), names=keep,
                 tier="photo", pose_sigma_m=0.03, scale_sigma_rel=sig)


# ----------------------------------------------------------------------------------------------------------------------
# Occlusion: a z-buffer of the fused scene surface, per view
# ----------------------------------------------------------------------------------------------------------------------
def zbuffer(points: np.ndarray, K: np.ndarray, T_wc: np.ndarray, shape: tuple[int, int], down: int = 8) -> np.ndarray:
    """Depth of the nearest scene surface along each pixel, at 1/down resolution (inf = nothing known).

    Uses the fused TSDF points of ANY tier (LiDAR, video, photo), so occlusion handling does not depend on a depth
    sensor. At 1/8 resolution a 2 cm voxel covers about one pixel at 2 m; a 3x3 min-filter closes the remaining gaps.
    Both choices err towards 'occluded', which only costs a few edge pixels of texture, never a sofa's texture
    leaking onto the wall behind it (seen in the first test: speckled sofa fabric on the wall orthophoto)."""
    h, w = shape[0] // down, shape[1] // down
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    pc = (points - t) @ R
    z = pc[:, 2]
    m = z > 0.1
    pc, z = pc[m], z[m]
    u = (K[0, 0] * pc[:, 0] / z + K[0, 2]) / down
    v = (K[1, 1] * pc[:, 1] / z + K[1, 2]) / down
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    m = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    zb = np.full(h * w, np.inf, np.float32)
    np.minimum.at(zb, vi[m] * w + ui[m], z[m].astype(np.float32))
    zb = zb.reshape(h, w)
    finite = np.where(np.isfinite(zb), zb, 1e6).astype(np.float32)
    closed = cv2.erode(finite, np.ones((3, 3), np.uint8))
    closed[closed >= 1e6] = np.inf
    return closed


# ----------------------------------------------------------------------------------------------------------------------
# Orthophotos
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Ortho:
    surface: Surface
    res: float
    median: np.ndarray                # (H,W,3) float32 BGR in 0..1, NaN where unseen
    count: np.ndarray                 # (H,W) number of views that saw each pixel
    stack: np.ndarray                 # (V,H,W,3) float16 per-view samples (NaN = not seen), gain-normalised
    view_ids: list[int]
    mask: np.ndarray                  # (H,W) pixel belongs to the surface


def _view_scores(srf: Surface, views: Views, max_range: float, n_probe: int = 12) -> np.ndarray:
    """Cheap pre-selection: fraction of a coarse probe grid that is in front of, inside, and close to each view,
    times how frontal the view is. Avoids projecting a million pixels into hundreds of irrelevant views."""
    s = np.linspace(0.05, 0.95, n_probe) * srf.size[0]
    t = np.linspace(0.05, 0.95, n_probe) * srf.size[1]
    S, Tt = np.meshgrid(s, t)
    X = srf.to_world(S.ravel(), Tt.ravel())
    scores = np.zeros(views.n)
    for i in range(views.n):
        T, K = views.T_wc[i], views.K[i]
        h, w = views.images[i].shape[:2]
        pc = (X - T[:3, 3]) @ T[:3, :3]
        z = pc[:, 2]
        cam_dir = T[:3, 3] - X
        dist = np.linalg.norm(cam_dir, axis=1)
        facing = (cam_dir @ srf.normal) / np.maximum(dist, 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]
            v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        ok = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (dist < max_range) & (facing > 0.25)
        scores[i] = ok.mean() * (np.median(facing[ok]) if ok.any() else 0) / max(np.median(dist[ok]) if ok.any() else 1, 0.5)
    return scores


def build_ortho(srf: Surface, views: Views, scene_points: np.ndarray, res: float = 0.005, max_views: int = 16,
                max_range: float = 4.0, occl_tol: float = 0.05, occl_rel: float = 0.03,
                zcache: dict | None = None) -> Ortho | None:
    """Warp the best views onto the surface plane and fuse them by a per-pixel median.

    Per view and pixel the sample is rejected when (a) the surface point is behind the camera or outside the image,
    (b) the scene z-buffer says something is in FRONT of the surface (furniture, a person, a door leaf), (c) the
    ray grazes the surface (< ~15 deg) or the point is beyond `max_range`. Exposure differences between video
    frames are removed with one multiplicative gain per view (robust ratio to the first-pass median)."""
    mask = srf.grid_mask(res)
    H, W = mask.shape
    if mask.sum() < 400:
        return None
    sc = _view_scores(srf, views, max_range)
    order = [i for i in np.argsort(-sc) if sc[i] > 0.02][:max_views]
    if len(order) < 2:
        return None
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    s, t = srf.pixel_st(cols, rows, res)
    X = srf.to_world(s, t).reshape(-1, 3)
    stack = np.full((len(order), H, W, 3), np.nan, np.float32)
    zcache = {} if zcache is None else zcache
    for k, i in enumerate(order):
        T, K, img = views.T_wc[i], views.K[i], views.images[i]
        h, w = img.shape[:2]
        pc = (X - T[:3, 3]) @ T[:3, :3]
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]
            v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        cam_dir = T[:3, 3] - X
        dist = np.linalg.norm(cam_dir, axis=1)
        facing = (cam_dir @ srf.normal) / np.maximum(dist, 1e-6)
        ok = (z > 0.2) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1) & (dist < max_range) & (facing > 0.25)
        if i not in zcache:
            zcache[i] = zbuffer(scene_points, K, T, (h, w))
        zb = zcache[i]
        d = zb.shape[0] / h
        zi = zb[np.clip((v * d).astype(int), 0, zb.shape[0] - 1), np.clip((u * d).astype(int), 0, zb.shape[1] - 1)]
        # unknown depth (inf) counts as occluded: the surface itself is in the fused scene, so a hole means the
        # TSDF never confirmed it from this side
        ok &= np.isfinite(zi) & (zi >= z - (occl_tol + occl_rel * z))
        ok &= mask.ravel()
        if ok.sum() < 200:
            continue
        samp = cv2.remap(img, u.reshape(H, W).astype(np.float32), v.reshape(H, W).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
        samp[~ok.reshape(H, W)] = np.nan
        stack[k] = samp
    keep = [k for k in range(len(order)) if np.isfinite(stack[k, ..., 0]).any()]
    if len(keep) < 2:
        return None
    stack = stack[keep]
    order = [order[k] for k in keep]
    med = np.nanmedian(stack, axis=0)
    # one gain per view: robust ratio of the view to the consensus over pixels both see
    for k in range(len(stack)):
        lum_v, lum_m = stack[k].mean(-1), med.mean(-1)
        both = np.isfinite(lum_v) & np.isfinite(lum_m) & (lum_v > 0.02)
        if both.sum() > 200:
            stack[k] *= np.clip(np.median(lum_m[both] / lum_v[both]), 0.5, 2.0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    count = np.isfinite(stack[..., 0]).sum(0)
    return Ortho(surface=srf, res=res, median=med, count=count, stack=stack.astype(np.float16), view_ids=order,
                 mask=mask)
