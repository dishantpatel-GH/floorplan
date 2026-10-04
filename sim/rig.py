"""The simulated phone: where it is, how the keys move it, what gets recorded. Pure Python (no Isaac Sim), so the
same logic drives the interactive tool (teleop.py), the scripted protocol paths (auto_capture.py) and the tests.

Pose convention (USD world of the scene: Z up, metres):
  x, y   position on the floor plan
  z      camera height above the scene origin (floors are at z = 0 in InteriorAgent scenes)
  yaw    degrees, 0 = looking along +X, counter-clockwise seen from above
  pitch  degrees, negative = looking down
  roll   degrees about the viewing axis: 0 = landscape, 90 = portrait
camera_to_world(...) gives the 4x4 matrix with USD camera axes (x right, y up, looking down -z), which is what the
renderer needs. emulate.py converts it to the OpenCV camera / Y-up world convention of the phone formats.

Session file (session.json): everything needed to re-render a capture later, at any quality:
  takes   continuous recordings (video / LiDAR walks): samples [t, x, y, z, yaw, pitch, roll] at the UI frame rate
  photos  single shots: time, pose, photo folder (one folder per room, as on the phone)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SESSION_VERSION = 1


def camera_to_world(x, y, z, yaw, pitch, roll=0.0) -> np.ndarray:
    yw, p, r = math.radians(yaw), math.radians(pitch), math.radians(roll)
    fwd = np.array([math.cos(p) * math.cos(yw), math.cos(p) * math.sin(yw), math.sin(p)])
    right0 = np.cross(fwd, [0.0, 0.0, 1.0])
    right0 /= np.linalg.norm(right0)
    up0 = np.cross(right0, fwd)
    right = math.cos(r) * right0 + math.sin(r) * up0
    up = -math.sin(r) * right0 + math.cos(r) * up0
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, -fwd, (x, y, z)
    return T


@dataclass
class Walkable:
    """2 cm grid from scene_gt.py: 'soft' = inside rooms and door passages (walls block), 'strict' = also clear of
    furniture at body height."""
    origin: np.ndarray
    res: float
    soft: np.ndarray
    strict: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "Walkable":
        z = np.load(path)
        return cls(z["origin"], float(z["res"]), z["soft"], z["strict"])

    def _ij(self, x, y):
        return int(round((y - self.origin[1]) / self.res)), int(round((x - self.origin[0]) / self.res))

    def ok(self, x, y, strict=False) -> bool:
        i, j = self._ij(x, y)
        g = self.strict if strict else self.soft
        return 0 <= i < g.shape[0] and 0 <= j < g.shape[1] and bool(g[i, j])


@dataclass
class Rooms:
    """Room polygons (visible surfaces) and ids from sim_gt.json, for naming photo folders and the HUD."""
    ids: list
    polys: list

    @classmethod
    def load(cls, gt_json: Path) -> "Rooms":
        g = json.loads(Path(gt_json).read_text())
        return cls([r["id"] for r in g["rooms"]], [np.asarray(r["polygon"]) for r in g["rooms"]])

    def at(self, x, y) -> str | None:
        from matplotlib.path import Path as MPath
        for rid, P in zip(self.ids, self.polys):
            if MPath(P).contains_point((x, y)):
                return rid
        return None

    def looked_at(self, x, y, yaw) -> str | None:
        """The room in front of the camera (1.2 m ahead), else the room it stands in. This puts both shots of a
        doorway pair in the right folders: the back-shot into the room being left, the forward shot into the next."""
        for d in (1.2, 0.8, 0.4):
            r = self.at(x + d * math.cos(math.radians(yaw)), y + d * math.sin(math.radians(yaw)))
            if r:
                return r
        return self.at(x, y)


@dataclass
class Rig:
    x: float
    y: float
    z: float = 1.40          # chest height (HOUSE_CAPTURE_GUIDE: phone at chest height)
    yaw: float = 0.0
    pitch: float = -12.0     # slightly down: floor line in view (capture protocol)
    roll: float = 0.0
    walk_speed: float = 0.5        # m/s (slow indoor walk)
    turn_rate: float = 20.0        # deg/s: protocol limit for video ("at most about 20 deg per second")
    tilt_rate: float = 20.0
    lift_speed: float = 0.3
    fast: float = 3.0              # Shift multiplier (repositioning, not while recording)
    smooth_s: float = 0.25         # velocity smoothing time constant (no instant start/stop)
    walkable: Walkable | None = None
    _v: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def pose(self):
        return [self.x, self.y, self.z, self.yaw, self.pitch, self.roll]

    def matrix(self):
        return camera_to_world(*self.pose())

    def step(self, dt: float, cmd: dict) -> list[str]:
        """cmd: forward, strafe, lift, turn, tilt in -1..1, fast bool. Returns warnings."""
        k = self.fast if cmd.get("fast") else 1.0
        target = np.array([cmd.get("forward", 0) * self.walk_speed * k, cmd.get("strafe", 0) * self.walk_speed * 0.8 * k,
                           cmd.get("lift", 0) * self.lift_speed, cmd.get("turn", 0) * self.turn_rate * k,
                           cmd.get("tilt", 0) * self.tilt_rate, 0.0])
        a = 1.0 - math.exp(-dt / self.smooth_s) if self.smooth_s > 0 else 1.0
        self._v += a * (target - self._v)
        vf, vs, vz, vyaw, vpitch, _ = self._v
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        dx = (vf * c + vs * s) * dt          # strafe right = (sin, -cos) ... right of heading
        dy = (vf * s - vs * c) * dt
        warn = []
        nx, ny = self.x + dx, self.y + dy
        if self.walkable is None or self.walkable.ok(nx, ny):
            self.x, self.y = nx, ny
        elif self.walkable.ok(nx, self.y):    # slide along the wall
            self.x = nx; warn.append("wall")
        elif self.walkable.ok(self.x, ny):
            self.y = ny; warn.append("wall")
        else:
            warn.append("wall")
            self._v[:2] = 0
        self.z = float(np.clip(self.z + vz * dt, 0.3, 2.2))
        self.yaw = (self.yaw + vyaw * dt + 180.0) % 360.0 - 180.0
        self.pitch = float(np.clip(self.pitch + vpitch * dt, -85.0, 85.0))
        if self.walkable is not None and not self.walkable.ok(self.x, self.y, strict=True):
            warn.append("furniture")
        if abs(vyaw) > 25:
            warn.append("turning fast")
        return warn


class Session:
    """Recorder. Saved after every change so a crash or a closed window loses nothing."""

    def __init__(self, path: Path, scene: str, meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.d = json.loads(self.path.read_text())
        else:
            self.d = dict(version=SESSION_VERSION, scene=str(scene), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                          meta=meta or {}, takes=[], photos=[])
        self.take = None
        self.t0 = time.time()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d))
        tmp.replace(self.path)

    def start_take(self, lighting: str, kind: str = "walk") -> str:
        name = f"take{sum(1 for t in self.d['takes'] if t['name'].startswith('take')) + 1}"
        self.take = dict(name=name, kind=kind, lighting=lighting, started=time.strftime("%H:%M:%S"), samples=[])
        self.d["takes"].append(self.take)
        return name

    def add_sample(self, t: float, pose):
        if self.take is not None:
            self.take["samples"].append([round(t, 4)] + [round(float(v), 5) for v in pose])

    def stop_take(self):
        tk, self.take = self.take, None
        if tk is not None and len(tk["samples"]) < 10:
            self.d["takes"].remove(tk)       # an accidental double press
            tk = None
        self.save()
        return tk

    def add_photo(self, t: float, pose, folder: str, lighting: str):
        n = len(self.d["photos"]) + 1
        self.d["photos"].append(dict(index=n, t=round(t, 4), folder=folder, lighting=lighting,
                                     pose=[round(float(v), 5) for v in pose]))
        self.save()
        return n


def resample(samples: list, fps: float) -> np.ndarray:
    """Take samples [t, x, y, z, yaw, pitch, roll] (irregular UI rate) -> poses at a fixed frame rate (angles are
    unwrapped before interpolating). Returns (n, 7) with t starting at 0."""
    S = np.asarray(samples, float)
    t = S[:, 0] - S[0, 0]
    keep = np.r_[True, np.diff(t) > 1e-6]
    S, t = S[keep], t[keep]
    for c in (4, 5, 6):
        S[:, c] = np.degrees(np.unwrap(np.radians(S[:, c])))
    tf = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)
    out = np.empty((len(tf), 7))
    out[:, 0] = tf
    for c in range(1, 7):
        out[:, c] = np.interp(tf, t, S[:, c])
    return out


# ------------------------------------------------------------------------------------------- human imperfection
def _ou(n, dt, sigma, tau, rng):
    """Ornstein-Uhlenbeck noise (mean-reverting random walk), stationary from the first sample."""
    a = math.exp(-dt / tau)
    b = sigma * math.sqrt(1 - a * a)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    e = rng.normal(0, 1, n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + b * e[k]
    return x


HUMAN = dict(          # level 1.0 = an ordinary person following the protocol; 2.0 = sloppy
    height_sigma=0.07, height_tau=6.0,       # m, s: the phone drifts between ~1.25 and ~1.55 m (slow)
    bob_amp=0.005, bob_hz=1.8,               # walking bob (only while moving); was 1.2 cm, user: "less jitter"
    pitch_sigma=5.0, pitch_tau=4.0,          # deg: "slightly down" is not held exactly (slow)
    roll_sigma=1.5, roll_tau=4.0,            # deg: the phone is not held level (slow)
    yaw_sigma=1.0, yaw_tau=2.0,              # deg: aiming wobble on top of the path heading
    sway_sigma=0.02, sway_tau=3.0,           # m: side-to-side
    shake_deg=0.10, shake_tau=0.15,          # deg: hand tremor; was 0.35 deg (D-051: less jitter)
)


def humanize(poses: np.ndarray, level: float = 1.0, seed: int = 0, params: dict | None = None) -> np.ndarray:
    """Add the slow, human drift of height, tilt, roll and aim (and hand tremor) to a nominal path.
    poses: (n, 7) [t, x, y, z, yaw, pitch, roll] at a fixed rate. Returns a new array. level 0 = robot."""
    if level <= 0 or len(poses) < 2:
        return poses.copy()
    P = dict(HUMAN, **(params or {}))
    rng = np.random.default_rng(seed)
    out = poses.copy()
    n = len(poses)
    dt = float(np.median(np.diff(poses[:, 0])))
    v = np.r_[0.0, np.hypot(np.diff(poses[:, 1]), np.diff(poses[:, 2])) / dt]
    moving = np.clip(np.convolve(v, np.ones(9) / 9, mode="same") / 0.4, 0, 1)
    t = poses[:, 0]
    L = level
    out[:, 3] += _ou(n, dt, P["height_sigma"] * L, P["height_tau"], rng)
    out[:, 3] += P["bob_amp"] * L * moving * np.sin(2 * np.pi * P["bob_hz"] * t + rng.uniform(0, 6.28))
    out[:, 5] += _ou(n, dt, P["pitch_sigma"] * L, P["pitch_tau"], rng) + _ou(n, dt, P["shake_deg"], P["shake_tau"], rng)
    out[:, 6] += _ou(n, dt, P["roll_sigma"] * L, P["roll_tau"], rng) + _ou(n, dt, P["shake_deg"], P["shake_tau"], rng)
    out[:, 4] += _ou(n, dt, P["yaw_sigma"] * L, P["yaw_tau"], rng) + _ou(n, dt, P["shake_deg"], P["shake_tau"], rng)
    sway = _ou(n, dt, P["sway_sigma"] * L, P["sway_tau"], rng)
    yaw = np.radians(poses[:, 4])
    out[:, 1] += sway * np.sin(yaw)
    out[:, 2] -= sway * np.cos(yaw)
    out[:, 3] = np.clip(out[:, 3], 0.9, 1.9)
    out[:, 5] = np.clip(out[:, 5], -85, 85)
    return out


def humanize_photo(pose, level: float = 1.0, rng=None) -> list:
    """One still: the person does not stand, aim or tilt exactly as intended."""
    if level <= 0:
        return list(pose)
    rng = rng or np.random.default_rng()
    x, y, z, yaw, pitch, roll = pose
    return [x + rng.normal(0, 0.05 * level), y + rng.normal(0, 0.05 * level), z + rng.normal(0, 0.05 * level),
            yaw + rng.normal(0, 4.0 * level), pitch + rng.normal(0, 4.0 * level), roll + rng.normal(0, 1.5 * level)]
