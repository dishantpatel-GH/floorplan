#!/usr/bin/env python
"""Turn rendered frames into the files a phone would hand over (decision D-039). Main .venv, no Isaac Sim.

Input: <session>/render/ (render.py). Output: <out>/ in exactly the layout of a real capture
(HOUSE_CAPTURE_GUIDE step 0, CAPTURE_PROTOCOL "Hand the files over"), so the same scripts run on sim and real data:

  photos/<room>/IMG_NNNN.JPG   JPEG + EXIF as the phone writes it: FocalLength, FocalLengthIn35mmFilm (integer,
                               diagonal definition), DateTimeOriginal + SubSecTimeOriginal, Make/Model (marked
                               simulated), ExposureTime, ISO
  video/<take>.MOV             16:9 centre crop of the 4:3 frames (what phones do for 1080p), H.264, the QuickTime
                               35 mm-equivalent focal key an iPhone writes (iphone15 profile; the Android profile
                               writes none)
  lidar/<take>/                Stray Scanner folder: rgb.mp4 (HEVC 1920x1440), depth/NNNNNN.png (uint16 mm, 256x192),
                               confidence/NNNNNN.png (0/1/2), odometry.csv (ARKit-like poses WITH drift, per-frame
                               intrinsics), camera_matrix.csv, imu.csv
  gt/ground_truth.csv          the scene's ground truth (sim/scene_gt.py), same format as the tape GT
  truth/                       what the phone does NOT know (true poses, frame transforms): diagnostics only

What is emulated, and from what evidence:
  * LiDAR depth: rays cast against the scene's triangles (identical to Isaac's depth, checked to 0.00 mm), 3x3
    supersampled per 256x192 pixel and averaged, so depth edges blend like the real sensor's. Noise = the measured
    sigma(range) of the real sample (floorplan/uncertainty/lidar_noise.json), half white, the rest a smooth field
    (ARKit depth errors are spatially correlated), x4 for medium confidence.
  * Confidence: 2 near and head-on; 1 at range (fading 2.5 -> 4.5 m), at grazing angles and depth edges; 0 beyond
    ~4.5-5 m or with no return. Glass: the pulse goes through (depth of what is behind, medium confidence).
    Mirrors: the pulse is reflected, so the depth is the path length into the "room behind the mirror", with HIGH
    confidence, which is how a real iPhone gets fooled.
  * Poses: ARKit-like world (Y up, origin at the first pose, first view along -Z), OpenCV camera axes (as verified
    on the real sample, DATA_NOTES), plus visual-inertial drift: yaw random walk 0.25 deg/sqrt(m), position random
    walk 0.6 cm/sqrt(m), scale error 0.4% (gives ~10-15 cm revisit gaps on a whole-flat walk, like the sample's 14 cm).
  * Images: auto exposure (dim scenes are brightened up to 8x, with the matching shot/read noise), sensor noise, motion
    blur from the camera's rotation during a 1/60 s (1/30 s when dim) exposure, JPEG / H.264 / HEVC compression.

Usage: python sim/emulate.py <session> [--profile iphone15|nord] [--tiers photo,video,lidar] [--drift arkit|none|bad]
                             [--seed 0] [--out <dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim import scene_geom as sg  # noqa: E402

PROFILES = {
    "iphone15": dict(make="SimPhone (Isaac Sim render)", model="iphone15-like", lens_mm=6.24, video_ext=".MOV",
                     video_f35_key=True, photo_quality=92, video_crf=20),
    "nord": dict(make="SimPhone (Isaac Sim render)", model="oneplus-nord-like", lens_mm=4.74, video_ext=".mp4",
                 video_f35_key=False, photo_quality=92, video_crf=21),
}
DRIFT = {"none": dict(yaw=0.0, kappa=0.0, pos=0.0, scale=0.0),
         "arkit": dict(yaw=math.radians(0.25), kappa=0.3, pos=0.006, scale=0.004),
         "bad": dict(yaw=math.radians(0.75), kappa=0.3, pos=0.018, scale=0.012)}
M_ZUP_TO_YUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])     # (x, y, z)_usd -> (x, z, -y): Y up
USD_TO_CV = np.diag([1.0, -1.0, -1.0])                                  # camera axes: USD (x, y, -z fwd) -> OpenCV
DEPTH_W, DEPTH_H, SS = 256, 192, 3
WORKERS = 6                     # depth worker processes (each casts ~440k rays per frame with its own ray caster)
WORKER_THREADS = 3              # ray-casting threads per worker: unbounded, 6 workers x 22 threads hit load 120 (D-043)
BASE_TIME = datetime(2026, 10, 4, 11, 0, 0)


# ------------------------------------------------------------------------------------------------ image side
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_LUT_S2L = None
_LUT_L2S = None


def _luts():
    global _LUT_S2L, _LUT_L2S
    if _LUT_S2L is None:
        _LUT_S2L = srgb_to_lin(np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)
        _LUT_L2S = (lin_to_srgb(np.linspace(0, 1, 4096)) * 255 + 0.5).astype(np.uint8)
    return _LUT_S2L, _LUT_L2S


class PhoneISP:
    """Auto exposure + sensor noise + motion blur. Gain is smoothed over time for video (phones adapt slowly).
    Speed: sRGB <-> linear through lookup tables, noise from a pre-drawn bank (random bank entry and offset per
    frame), ~0.1 s per 1920x1440 frame."""

    BANK = 4
    PAD = 32

    def __init__(self, rng, video: bool):
        self.rng, self.video, self.gain = rng, video, None
        self.bank = None

    def _noise(self, shape):
        if self.bank is None or self.bank[0].shape[0] < shape[0] + self.PAD or self.bank[0].shape[1] < shape[1] + self.PAD:
            self.bank = [self.rng.standard_normal((shape[0] + self.PAD, shape[1] + self.PAD, 3), dtype=np.float32)
                         for _ in range(self.BANK)]
        b = self.bank[int(self.rng.integers(self.BANK))]
        dy, dx = (int(v) for v in self.rng.integers(0, self.PAD, 2))
        return b[dy:dy + shape[0], dx:dx + shape[1]]

    def __call__(self, rgb8: np.ndarray, blur_px: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, dict]:
        s2l, l2s = _luts()
        x = s2l[rgb8]
        h, w = x.shape[:2]
        c = float(x[h // 4: 3 * h // 4: 4, w // 4: 3 * w // 4: 4].mean())
        g = float(np.clip(0.18 / max(c, 1e-4), 1.0, 8.0))
        if self.video and self.gain is not None:
            g = self.gain + 0.15 * (g - self.gain)
        self.gain = g
        shot, read = 0.0006 * g, 0.0015 * g
        if g != 1.0:
            x *= g
        sd = np.sqrt(np.maximum(x, 0.0) * shot + read * read)
        x += self._noise(x.shape) * sd
        bx, by = blur_px
        L = math.hypot(bx, by)
        if L >= 1.0:
            k = int(math.ceil(L)) | 1
            ker = np.zeros((k, k), np.float32)
            c0 = k // 2
            cv2.line(ker, (int(round(c0 - bx / 2)), int(round(c0 - by / 2))), (int(round(c0 + bx / 2)), int(round(c0 + by / 2))), 1.0, 1)
            ker /= ker.sum()
            x = cv2.filter2D(x, -1, ker)
        idx = np.clip(x * 4095.0 + 0.5, 0, 4095).astype(np.int16)
        out = l2s[idx]
        iso = int(round(50 * g / 5) * 5)
        return out, dict(gain=g, iso=iso, exposure_s=(1 / 60 if g < 2 else 1 / 30))


def blur_vector(T0, T1, K, dt, exposure_s):
    """Image-centre motion (px) during the exposure, from the camera's rotation between two frames."""
    if T0 is None:
        return (0.0, 0.0)
    R = T0[:3, :3].T @ T1[:3, :3]                 # rotation in camera (USD axes) coordinates
    rv = Rotation.from_matrix(R).as_rotvec() * (exposure_s / dt)
    f = K[0, 0]
    # small rotation about camera y (up) moves the image horizontally; about x (right) vertically
    return (float(-rv[1] * f), float(rv[0] * f))


# ---------------------------------------------------------------------------------------------- pose frames
def to_arkit(Ts_usd: np.ndarray) -> tuple[np.ndarray, dict]:
    """USD-world camera poses (USD camera axes) -> ARKit-like world (Y up, origin = first camera, first view -Z)
    with OpenCV camera axes. Returns (T (n,4,4), the transform used)."""
    R = M_ZUP_TO_YUP[None] @ Ts_usd[:, :3, :3] @ USD_TO_CV[None]
    c = (M_ZUP_TO_YUP @ Ts_usd[:, :3, 3].T).T
    f = R[0][:, 2]
    a = math.atan2(f[0], -f[2])
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    T = np.tile(np.eye(4), (len(R), 1, 1))
    T[:, :3, :3] = Ry[None] @ R
    T[:, :3, 3] = (Ry @ (c - c[0]).T).T
    return T, dict(R_yup_from_usd=(Ry @ M_ZUP_TO_YUP).tolist(), origin_usd=Ts_usd[0, :3, 3].tolist(), yaw_rad=a)


def add_drift(T: np.ndarray, kind: str, rng) -> np.ndarray:
    p = DRIFT[kind]
    if not p["yaw"] and not p["pos"]:
        return T.copy()
    out = T.copy()
    psi = 0.0
    s = 1.0 + rng.normal(0, p["scale"])
    pe = T[0, :3, 3].copy()
    for k in range(1, len(T)):
        dp = T[k, :3, 3] - T[k - 1, :3, 3]
        d = float(np.linalg.norm(dp))
        dth = float(np.linalg.norm(Rotation.from_matrix(T[k - 1, :3, :3].T @ T[k, :3, :3]).as_rotvec()))
        psi += rng.normal(0, p["yaw"] * math.sqrt(d + p["kappa"] * dth + 1e-12))
        Ry = Rotation.from_rotvec([0, psi, 0]).as_matrix()
        noise = rng.normal(0, p["pos"] * math.sqrt(d + 1e-12), 3) * np.array([1.0, 0.3, 1.0])
        pe = pe + s * (Ry @ dp) + noise
        out[k, :3, :3] = Ry @ T[k, :3, :3]
        out[k, :3, 3] = pe
    return out


# ------------------------------------------------------------------------------------------------ LiDAR side
class DepthSensor:
    def __init__(self, scene_usd: str, K_rgb: np.ndarray, W: int, H: int, seed: int):
        self.geom = sg.load(scene_usd)
        glass = sg.mesh_mask(self.geom, lambda m: m["glass"])
        self.full, self.full_ids = sg.raycaster(self.geom)
        self.noglass, self.ng_ids = sg.raycaster(self.geom, ~glass)
        T = self.geom["tri_mesh"]
        self.tri_glass = glass[T]
        self.tri_mirror = sg.mesh_mask(self.geom, lambda m: m["mirror"])[T]
        sx, sy = DEPTH_W / W, DEPTH_H / H
        self.K = np.array([[K_rgb[0, 0] * sx, 0, (K_rgb[0, 2] + 0.5) * sx - 0.5],
                           [0, K_rgb[1, 1] * sy, (K_rgb[1, 2] + 0.5) * sy - 0.5], [0, 0, 1.0]])
        off = (np.arange(SS) - (SS - 1) / 2) / SS
        u = (np.arange(DEPTH_W)[None, :, None, None] + off[None, None, None, :])
        v = (np.arange(DEPTH_H)[:, None, None, None] + off[None, None, :, None])
        u, v = np.broadcast_arrays(u, v)                          # (H, W, SS, SS)
        x = (u - self.K[0, 2]) / self.K[0, 0]
        y = (v - self.K[1, 2]) / self.K[1, 1]
        self.dirs_cam = np.stack([x, -y, -np.ones_like(x)], -1).reshape(-1, 3)   # USD camera axes, z-length 1
        self.dnorm = np.linalg.norm(self.dirs_cam, axis=1)
        tab = json.loads((ROOT / "floorplan/uncertainty/lidar_noise.json").read_text())["table"]
        self.sig_r = np.array([r["r_mid"] for r in tab])
        self.sig_v = np.array([r["sigma_mm"] for r in tab]) / 1000.0
        self.rng = np.random.default_rng(seed)
        self.nthreads = 0

    def _smooth(self, gh=6, gw=8):
        g = self.rng.normal(0, 1, (gh, gw)).astype(np.float32)
        return cv2.resize(g, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_CUBIC) / 0.75

    def frame(self, T_usd: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        R, o = T_usd[:3, :3], T_usd[:3, 3]
        d = self.dirs_cam @ R.T
        O = np.broadcast_to(o, d.shape)
        t, pid, nrm = sg.cast(self.full, O, d, self.nthreads)
        hit = pid >= 0
        tri = np.where(hit, self.full_ids[np.maximum(pid, 0)], -1)
        is_glass = hit & self.tri_glass[np.maximum(tri, 0)]
        is_mirror = hit & self.tri_mirror[np.maximum(tri, 0)]
        z = t.astype(np.float64).copy()
        stats = dict(glass=int(is_glass.sum()), mirror=int(is_mirror.sum()))
        if is_glass.any():                                   # the pulse goes through glass
            t2, p2, _ = sg.cast(self.noglass, O[is_glass], d[is_glass], self.nthreads)
            z[is_glass] = t2
        if is_mirror.any():                                  # ... and is reflected by mirrors
            dm = d[is_mirror] / self.dnorm[is_mirror, None]
            n = nrm[is_mirror].astype(np.float64)
            n *= -np.sign(np.sum(n * dm, axis=1, keepdims=True))
            p = O[is_mirror] + t[is_mirror, None] * d[is_mirror]
            r = dm - 2 * np.sum(dm * n, axis=1, keepdims=True) * n
            t3, p3, _ = sg.cast(self.full, p + 1e-3 * r, r, self.nthreads)
            z[is_mirror] = (t[is_mirror] * self.dnorm[is_mirror] + t3) / self.dnorm[is_mirror]
        z = z.astype(np.float32).reshape(DEPTH_H, DEPTH_W, SS * SS)
        valid = np.isfinite(z)
        n_ok = valid.sum(-1)
        zs = np.where(valid, z, 0.0).sum(-1)
        mean = np.where(n_ok > 0, zs / np.maximum(n_ok, 1), np.inf).astype(np.float32)
        zmax = np.where(valid, z, -np.inf).max(-1)
        zmin = np.where(valid, z, np.inf).min(-1)
        spread = np.where(n_ok > 0, (zmax - zmin) / np.maximum(mean, 1e-3), np.inf)
        c = SS * SS // 2
        nc = nrm.reshape(DEPTH_H, DEPTH_W, SS * SS, 3)[:, :, c]
        dc = d.reshape(DEPTH_H, DEPTH_W, SS * SS, 3)[:, :, c]
        cos_inc = np.abs(np.sum(nc * dc, -1)) / np.linalg.norm(dc, axis=-1)
        glass_px = is_glass.reshape(DEPTH_H, DEPTH_W, -1).any(-1)
        # confidence
        conf = np.full((DEPTH_H, DEPTH_W), 2, np.uint8)
        # P(not high) vs range, fitted to the real sample (TakeHome/Dataset, every 40th frame of the three captures):
        # high-confidence share 0.93-0.95 below 1.5 m, 0.87-0.91 at 1.5-2.5 m, 0.67-0.73 at 2.5-3.5 m, 0.56-0.64 at
        # 3.5-4.5 m. Model: 0.05 + 0.45 * clip((r - 1.8) / 2.4, 0, 1) -> high share 0.95 / 0.91 / 0.73 / 0.54 at
        # 1 / 2 / 3 / 4 m (sim check after the change: 0.96 / 0.91 / ~0.75 / ~0.6 per range bin)
        fade = 0.05 + 0.45 * np.clip((mean - 1.8) / 2.4, 0, 1)
        field = cv2.resize(self.rng.random((12, 16)).astype(np.float32), (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_LINEAR)
        conf[(field < fade) | (cos_inc < 0.2) | (spread > 0.05) | (n_ok < SS * SS) | glass_px] = 1
        conf[(mean > 4.8) | (spread > 0.15) | (n_ok < 3) | ~np.isfinite(mean)] = 0
        # noise
        sig = np.interp(np.where(np.isfinite(mean), mean, 0.0), self.sig_r, self.sig_v).astype(np.float32)
        sig = sig * np.where(conf == 2, 1.0, np.where(conf == 1, 4.0, 8.0))
        noise = sig * (0.5 * self.rng.standard_normal(sig.shape, dtype=np.float32) + 0.85 * self._smooth()) + self.rng.normal(0, 0.002)
        # ARKit depth maps are dense: pixels with no return (sky through a window) or beyond the sensor's range still
        # get a value, flagged confidence 0 (the real sample has no zero pixels). Use 6-7.5 m for those.
        far = ~np.isfinite(mean) | (mean >= 7.5)
        depth = np.where(far, 6.0 + 1.5 * field, mean + noise)
        mm = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
        stats.update(mean_conf2=float((conf == 2).mean()))
        return mm, conf, stats


# ------------------------------------------------------------- parallel depth (each worker: its own ray caster)
_SENSOR = None


def _depth_init(scene, K, W, H):
    global _SENSOR
    import open3d as o3d
    o3d.utility.set_max_threads(WORKER_THREADS)
    _SENSOR = DepthSensor(scene, K, W, H, seed=0)
    _SENSOR.nthreads = WORKER_THREADS


def _depth_job(job):
    i, T, seed, ddir, cdir = job
    _SENSOR.rng = np.random.default_rng(seed)        # per-frame seed: same output whatever the worker count
    mm, conf, st = _SENSOR.frame(T)
    cv2.imwrite(f"{ddir}/{i:06d}.png", mm)
    cv2.imwrite(f"{cdir}/{i:06d}.png", conf)
    return i, st


# ------------------------------------------------------------------------------------------------- writers
def decode_frames(mp4: Path):
    cap = cv2.VideoCapture(str(mp4))
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield bgr[:, :, ::-1]
    cap.release()


NVENC = {"libx264": ("h264_nvenc", "-cq"), "libx265": ("hevc_nvenc", "-cq")}
USE_NVENC = True                 # phones encode in hardware too; frees the CPU for the depth workers


def ffmpeg_writer(path: Path, W, H, fps, codec, crf, extra=()):
    head = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}",
            "-i", "-"]
    if USE_NVENC and codec in NVENC:
        enc, q = NVENC[codec]
        body = ["-c:v", enc, "-preset", "p4", q, str(crf), "-pix_fmt", "yuv420p"]
    else:
        body = ["-c:v", codec, "-crf", str(crf), "-preset", "faster", "-pix_fmt", "yuv420p"]
        if codec == "libx265":
            body += ["-x265-params", "log-level=error"]
    return subprocess.Popen(head + body + list(extra) + [str(path)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def exif_bytes(prof, f_px, W, H, when: datetime, info: dict):
    ex = Image.Exif()
    ex[0x010F] = prof["make"]
    ex[0x0110] = prof["model"]
    ex[0x0112] = 1
    ifd = ex.get_ifd(0x8769)
    f35 = int(round(f_px * 43.27 / math.hypot(W, H)))           # what a phone writes: integer, diagonal definition
    ifd[0xA405] = f35
    ifd[0x920A] = (int(round((prof["lens_mm"] if f_px > 2000 else prof.get("lens_mm_uw", 1.54)) * 100)), 100)  # main 6.24 mm, ultra-wide 1.54 mm (iPhone 15)
    ifd[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
    ifd[0x9291] = f"{when.microsecond // 1000:03d}"
    ifd[0x829A] = (1, int(round(1 / info["exposure_s"])))
    ifd[0x8827] = info["iso"]
    return ex, f35


def write_photos(rdir: Path, out: Path, prof, rng, log):
    meta = json.loads((rdir / "photos" / "photos.json").read_text())
    K = np.asarray(meta["K"])
    isp = PhoneISP(rng, video=False)
    truth = []
    K0 = K
    for p in meta["photos"]:
        K = np.asarray(p["K"]) if "K" in p else K0        # v2.2 (D-065): per-photo lens
        rgb = np.asarray(Image.open(rdir / "photos" / p["file"]).convert("RGB"))
        img, info = isp(rgb)
        when = BASE_TIME + timedelta(seconds=float(p["t"]))
        ex, f35 = exif_bytes(prof, K[0, 0], rgb.shape[1], rgb.shape[0], when, info)
        fd = out / "photos" / p["folder"]
        fd.mkdir(parents=True, exist_ok=True)
        name = f"IMG_{int(p['file'][:4]):04d}.JPG"
        Image.fromarray(img).save(fd / name, quality=prof["photo_quality"], exif=ex)
        truth.append(dict(file=f"{p['folder']}/{name}", T_usd=p["T"], f_px_true=float(K[0, 0]), f35_written=f35,
                          f_px_from_f35_diag=f35 * math.hypot(rgb.shape[1], rgb.shape[0]) / 43.27,
                          f_px_from_f35_36mm=f35 / 36.0 * max(rgb.shape[:2]), iso=info["iso"], nominal_pose=p["nominal_pose"]))
    (out / "truth").mkdir(parents=True, exist_ok=True)
    (out / "truth" / "photos_true.json").write_text(json.dumps(truth, indent=1))
    log(f"[emulate] {len(truth)} photos in {len({t['file'].split('/')[0] for t in truth})} folders")


def write_take(rdir: Path, take: str, out: Path, prof, tiers, drift, scene, seed, log):
    z = np.load(rdir / take / "poses.npz")
    Ts, t, K, fps = z["T"], z["t"], z["K"], float(z["fps"])
    W, H = int(z["W"]), int(z["H"])
    rng = np.random.default_rng(seed)
    writers = {}
    if "video" in tiers:
        (out / "video").mkdir(parents=True, exist_ok=True)
        vh = W * 9 // 16
        extra = ["-movflags", "use_metadata_tags", "-metadata", f"creation_time={BASE_TIME.isoformat()}Z"]
        f35v = int(round(K[0, 0] * 43.27 / math.hypot(W, vh)))
        if prof["video_f35_key"]:
            extra += ["-metadata", f"com.apple.quicktime.camera.focal_length.35mm_equivalent={f35v}",
                      "-metadata", f"com.apple.quicktime.make={prof['make']}", "-metadata", f"com.apple.quicktime.model={prof['model']}"]
        writers["video"] = ffmpeg_writer(out / "video" / f"{take}{prof['video_ext']}", W, vh, fps, "libx264", prof["video_crf"], extra)
        isp_v = PhoneISP(np.random.default_rng(seed + 1), video=True)
    if "lidar" in tiers:
        L = out / "lidar" / take
        if L.exists():
            shutil.rmtree(L)
        (L / "depth").mkdir(parents=True)
        (L / "confidence").mkdir()
        writers["lidar"] = ffmpeg_writer(L / "rgb.mp4", W, H, fps, "libx265", 26, ["-tag:v", "hvc1"])
        isp_l = PhoneISP(np.random.default_rng(seed + 2), video=True)
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(WORKERS, initializer=_depth_init, initargs=(scene, K, W, H))
        jobs = [(i, Ts[i], seed * 100003 + i, str(L / "depth"), str(L / "confidence")) for i in range(len(Ts))]
        depth_async = pool.map_async(_depth_job, jobs, chunksize=16)
    t0 = time.time()
    n = 0
    prevT = None
    stats = []
    for i, rgb in enumerate(decode_frames(rdir / take / "frames.mp4")):
        if i >= len(Ts):
            break
        T = Ts[i]
        exp = 1 / 60
        isp = isp_l if "lidar" in writers else isp_v
        blur = blur_vector(prevT, T, K, 1 / fps, 1 / 60 if (isp.gain or 1) < 2 else 1 / 30)
        full, _ = isp(np.ascontiguousarray(rgb), blur)
        if "video" in writers:
            y0 = (H - vh) // 2
            writers["video"].stdin.write(np.ascontiguousarray(full[y0:y0 + vh]).tobytes())
        if "lidar" in writers:
            writers["lidar"].stdin.write(full.tobytes())
        prevT = T
        n += 1
        if i % 300 == 0:
            log(f"[emulate] {take}: frame {i}/{len(Ts)}  {(time.time() - t0) / max(i, 1):.3f} s/frame")
    for w in writers.values():
        w.stdin.close()
        w.wait()
    if "lidar" in writers:
        res = depth_async.get()
        pool.close()
        pool.join()
        stats = [st for i, st in sorted(res) if i < n]
        for i in range(n, len(Ts)):                                   # frames the decoder did not deliver
            for d in ("depth", "confidence"):
                (L / d / f"{i:06d}.png").unlink(missing_ok=True)
    Ts = Ts[:n]
    t = t[:n]
    if "lidar" in writers:
        T_true, frame_info = to_arkit(Ts)
        T_est = add_drift(T_true, drift, np.random.default_rng(seed + 4))
        boot = 1000.0 + 37.0 * (seed % 10)
        with open(L / "odometry.csv", "w") as f:
            f.write("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y\n")
            for i in range(n):
                q = Rotation.from_matrix(T_est[i, :3, :3]).as_quat()
                x, y, zz = T_est[i, :3, 3]
                f.write(f"{boot + t[i]:.9f}, {i:06d}, {x:.8f}, {y:.8f}, {zz:.8f}, {q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, "
                        f"{q[3]:.8f}, {K[0, 0]:.4f}, {K[1, 1]:.4f}, {K[0, 2]:.4f}, {K[1, 2]:.4f}, , \n")
        (L / "camera_matrix.csv").write_text("\n".join(", ".join(f"{v:.4f}" for v in row) for row in K))
        with open(L / "imu.csv", "w") as f:
            f.write("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n")
            g = np.array([0, -1.0, 0])
            for i in range(n):
                R = T_true[i, :3, :3]
                a = R.T @ g + np.random.default_rng(i).normal(0, 0.01, 3)
                w = (Rotation.from_matrix(T_true[i - 1, :3, :3].T @ R).as_rotvec() * fps) if i else np.zeros(3)
                f.write(f"{boot + t[i]:.9f}, {a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f}, {w[0]:.6f}, {w[1]:.6f}, {w[2]:.6f}\n")
        tr = out / "truth"
        tr.mkdir(parents=True, exist_ok=True)
        np.savez(tr / f"lidar_{take}_poses.npz", T_true=T_true, T_est=T_est, T_usd=Ts, t=t)
        (tr / f"lidar_{take}_frame.json").write_text(json.dumps(dict(frame_info, drift=drift, n=n,
                                                                    conf2_mean=float(np.mean([s['mean_conf2'] for s in stats])),
                                                                    mirror_px=int(sum(s['mirror'] for s in stats)),
                                                                    glass_px=int(sum(s['glass'] for s in stats))), indent=1))
        gap = np.linalg.norm(T_est[-1, :3, 3] - T_true[-1, :3, 3])
        log(f"[emulate] lidar {take}: {n} frames, drift at the end {gap * 100:.1f} cm, "
            f"high-confidence share {np.mean([s['mean_conf2'] for s in stats]):.2f}")
    log(f"[emulate] {take}: {n} frames in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--profile", default="iphone15", choices=sorted(PROFILES))
    ap.add_argument("--tiers", default="photo,video,lidar")
    ap.add_argument("--drift", default="arkit", choices=sorted(DRIFT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--takes", default="")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rdir = a.session / "render"
    rinfo = json.loads((rdir / "render.json").read_text())
    scene = rinfo["scene"]
    out = a.out or a.session / f"capture_{a.profile}"
    out.mkdir(parents=True, exist_ok=True)
    tiers = set(a.tiers.split(","))
    prof = PROFILES[a.profile]
    log = lambda m: print(m, flush=True)
    if "photo" in tiers and (rdir / "photos" / "photos.json").exists():
        if (out / "photos").exists():
            shutil.rmtree(out / "photos")
        write_photos(rdir, out, prof, np.random.default_rng(a.seed), log)
    takes = [d.name for d in sorted(rdir.iterdir()) if (d / "poses.npz").exists()]
    if a.takes:
        takes = [t for t in takes if t in a.takes.split(",")]
    for k, take in enumerate(takes):
        if tiers & {"video", "lidar"}:
            write_take(rdir, take, out, prof, tiers, a.drift, scene, a.seed * 100 + k, log)
    gt_dir = Path(scene).parent / ".simcache" / "gt"
    (out / "gt").mkdir(exist_ok=True)
    for f in ("ground_truth.csv", "sim_gt.json", "gt_topview.png"):
        shutil.copy(gt_dir / f, out / "gt" / f)
    (out / "capture.json").write_text(json.dumps(dict(session=str(a.session), scene=scene, profile=a.profile,
                                                     drift=a.drift, seed=a.seed, tiers=sorted(tiers),
                                                     render=rinfo, made=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    log(f"[emulate] capture -> {out}")


if __name__ == "__main__":
    main()
