"""Room names from what is in each room: Bedroom, Kitchen, Living room, Bathroom, Foyer, Passage, Balcony; Room when
the evidence says nothing (D-078).

Evidence is the ADE20K class map (SegFormer-B5, the segmenter the photo tier already runs for its wall masks, D-060)
of the images of each room:
  photo  the photos of the room's folder; a folder already named after a type (01_living_room) keeps that name
  video  keyframes (work/frames_r*/sfm): each pixel votes for the room its 3D point lies in (MoGe-2 depth + pose)
  lidar  the same with frames decoded from the Stray Scanner rgb.mp4 and the LiDAR depth
Pixel projection counts what stands in a room from wherever it was seen, and not what is seen through its doorways.
Self-check: the pixels called floor must land on the plan's floor; if not (depth and poses disagree), each frame
votes for the room its camera is in. Glass and sky have no depth, so the balcony test uses the camera-in-room view.

A type scores the weighted pixel share of its indicator classes (bed -> bedroom; stove, oven, hood, sink, counter,
refrigerator -> kitchen; sofa, coffee table, television -> living room; toilet, bathtub, shower -> bathroom). Shape
decides what the classes cannot: a room under 2 m wide holds no sofa or double bed, so there living room and bedroom
evidence must be three times stronger (it is usually seen through a doorway); a narrow room that is mostly glass or
sky is a balcony, a long narrow one a passage, a small one at the entrance a foyer. A home has one kitchen and one
living room: a second claim next to the first is the same room split by the plan, elsewhere it falls to its next
type. With no sofa, TV or coffee table anywhere, the largest room (at the entrance if one is) is the living
room.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

DISPLAY = {"bedroom": "Bedroom", "kitchen": "Kitchen", "living_room": "Living room", "bathroom": "Bathroom",
           "foyer": "Foyer", "passage": "Passage", "balcony": "Balcony", "room": "Room"}
TYPES = tuple(DISPLAY)

# Indicator classes (ADE20K names) and weights: 1.0 = the object alone names the room.
CLASS_WEIGHTS = {
    "bedroom": {"bed": 1.0, "blanket": 0.5, "cradle": 0.5, "pillow": 0.3, "wardrobe": 0.15, "chest of drawers": 0.15},
    "kitchen": {"stove": 1.0, "oven": 1.0, "hood": 1.0, "dishwasher": 1.0, "microwave": 0.8, "kitchen island": 0.8,
                "refrigerator": 0.6, "countertop": 0.4, "counter": 0.3, "pot": 0.3, "plate": 0.2},
    "living_room": {"sofa": 1.0, "coffee table": 1.0, "television receiver": 0.8, "armchair": 0.8, "fireplace": 0.8,
                    "ottoman": 0.5, "cushion": 0.3},
    "bathroom": {"toilet": 1.0, "bathtub": 1.0, "shower": 1.0, "screen door": 0.5, "towel": 0.4, "mirror": 0.25,
                 "countertop": 0.2},
}
SINK_WEIGHT = 0.6          # a sink is the bathroom's when a toilet, bathtub or shower is seen too, else the kitchen's
BATH_FIXTURES = ("toilet", "bathtub", "shower")
BATH_FIXTURE_MIN = 0.005
GLAZING = ("windowpane", "sky", "railing", "tree", "plant", "building", "palm")    # balcony: glass and outdoors
STRUCTURE = {"wall", "floor", "ceiling", "door", "windowpane", "column", "beam", "stairs", "step", "light",
             "painting", "curtain", "rug", "sconce", "chandelier"}            # not reported as top classes

MIN_SCORE = 0.008          # weighted share a type needs (a bed on 0.8% of the pixels)
NARROW_M = 2.0             # narrower: no living room or bedroom unless the evidence is NARROW_FACTOR times stronger
NARROW_FACTOR = 3.0
BALCONY_GLAZING = 0.10     # share of glass + outdoor pixels that makes a narrow room a balcony ...
BALCONY_OVER_OBJECTS = 3.0  # ... when it is this many times the best object score and no kitchen or bathroom
PASSAGE_ASPECT = 1.8       # length / width of a passage ...
PASSAGE_MAX_WIDTH_M = 1.2  # ... or this narrow at most and 1.5x as long (no room to live in)
FOYER_MAX_M2 = 4.0
LIVING_MIN_M2 = 9.0        # the largest room at the entrance is the living room when no room shows one
WEAK_BED = 0.4             # a bed this much weaker than another room's (a cot in the hall) does not make a bedroom
UNIQUE = ("kitchen", "living_room")
FRAGMENT_GAP_M = 0.5       # a second kitchen (living room) claim this close to the first is the same room, split
PROJECTION_MAX_FLOOR_OFF_M = 0.25   # projected floor pixels further from the floor: depth and poses disagree

# folder names that state a type (photo tier); 'hall' is not here: in India it is the living room, in Britain the
# entrance hall, so a folder called hall is named from its photos
FOLDER_WORDS = {
    "bedroom": ("bedroom", "bed room", "master bedroom", "guest room", "kids room", "nursery"),
    "kitchen": ("kitchen", "kitchenette", "pantry"),
    "living_room": ("living room", "living", "lounge", "family room", "sitting room", "drawing room"),
    "bathroom": ("bathroom", "bath", "toilet", "wc", "washroom", "restroom", "shower room", "powder room"),
    "foyer": ("foyer", "entrance", "entry", "entryway", "vestibule"),
    "passage": ("passage", "corridor", "hallway"),
    "balcony": ("balcony", "terrace", "veranda", "verandah", "loggia"),
}


def type_from_folder(name: str) -> str | None:
    """'01_living_room' -> 'living_room'; 'bedroom2' -> 'bedroom'; 'room' or 'hall' -> None."""
    s = re.sub(r"[_\-.]+", " ", str(name).lower())
    s = re.sub(r"\d+", " ", s)
    s = " ".join(s.split())
    for t, words in FOLDER_WORDS.items():
        if s in words:
            return t
    return None


# ------------------------------------------------------------------------------------------------ evidence

def class_shares(counts: np.ndarray) -> np.ndarray:
    tot = float(counts.sum())
    return counts / tot if tot > 0 else counts.astype(float)


def type_scores(shares: np.ndarray, names: list[str]) -> dict[str, float]:
    idx = {n: i for i, n in enumerate(names)}
    share = lambda c: float(shares[idx[c]]) if c in idx else 0.0      # noqa: E731
    out = {t: float(sum(w * share(c) for c, w in cw.items())) for t, cw in CLASS_WEIGHTS.items()}
    bath = sum(share(c) for c in BATH_FIXTURES) >= BATH_FIXTURE_MIN
    out["bathroom" if bath else "kitchen"] += SINK_WEIGHT * share("sink")
    out["balcony"] = float(sum(share(c) for c in GLAZING))
    return out


def top_classes(shares: np.ndarray, names: list[str], k: int = 6) -> list[list]:
    order = [i for i in np.argsort(-shares) if names[i] not in STRUCTURE and shares[i] > 0]
    return [[names[i], round(float(shares[i]), 4)] for i in order[:k]]


# ------------------------------------------------------------------------------------------------ shape

def room_shape(room) -> dict:
    """Length, width (aligned bounding box, else the polygon's) and area of a plan room (model.Room)."""
    poly = Polygon(room.polygon)
    if room.bbox_dims is not None and room.bbox_dims[0].value is not None and room.bbox_dims[1].value is not None:
        a, b = room.bbox_dims[0].value, room.bbox_dims[1].value
    else:
        x0, y0, x1, y1 = poly.bounds
        a, b = x1 - x0, y1 - y0
    area = room.floor_area.value if room.floor_area is not None and room.floor_area.value is not None else poly.area
    return dict(length=float(max(a, b)), width=float(min(a, b)), area=float(area))


def entrance_rooms(plan) -> set[str]:
    """Rooms with a door or passage to the outside of the plan (the entrance, or space that was not mapped): the
    opening has one room, or the point 30 cm behind its wall lies in no room."""
    polys = {r.id: Polygon(r.polygon).buffer(0) for r in plan.rooms if len(r.polygon) >= 3}
    walls = {w.id: w for w in plan.walls}
    out = set()
    for o in plan.openings:
        if o.kind not in ("door", "passage") or not o.room_ids:
            continue
        if len(o.room_ids) == 1:
            out.add(o.room_ids[0])
            continue
        for wid in o.wall_ids:
            w = walls.get(wid)
            if w is None:
                continue
            probe = Point(np.asarray(o.center) - 0.3 * np.asarray(w.normal))
            if not any(p.contains(probe) for p in polys.values()):
                out.add(w.room_id)
    return out


# ------------------------------------------------------------------------------------------------ decision

def decide(scores: dict[str, float] | None, shape: dict, entrance: bool, exclude: tuple = ()):
    """One room's type from its class scores and shape. Returns (type, rule, ranked object candidates)."""
    narrow = shape["width"] < NARROW_M
    cands = []
    if scores:
        for t in ("bedroom", "kitchen", "living_room", "bathroom"):
            need = MIN_SCORE * (NARROW_FACTOR if narrow and t in ("bedroom", "living_room") else 1.0)
            if t not in exclude and scores.get(t, 0.0) >= need:
                cands.append((t, scores[t]))
        cands.sort(key=lambda x: -x[1])
    glazing = scores.get("balcony", 0.0) if scores else 0.0
    if narrow and glazing >= BALCONY_GLAZING and not any(t in ("kitchen", "bathroom") for t, _ in cands) and \
            glazing >= BALCONY_OVER_OBJECTS * max([v for _, v in cands], default=0.0):
        return "balcony", f"under {NARROW_M:g} m wide and {100 * glazing:.0f}% glass or outdoors", cands
    if cands:
        return cands[0][0], "objects", cands
    aspect = shape["length"] / max(shape["width"], 1e-6)
    if narrow and (aspect >= PASSAGE_ASPECT or (shape["width"] <= PASSAGE_MAX_WIDTH_M and aspect >= 1.5)):
        return "passage", f"{shape['width']:.2f} m wide, {aspect:.1f}x as long", []
    if entrance and shape["area"] <= FOYER_MAX_M2:
        return "foyer", f"{shape['area']:.1f} m2 at the entrance", []
    return "room", "no indicator objects" if scores else "no images inside the room", []


def name_plan(plan, evidence: dict[str, dict], folder_types: dict[str, str] | None = None) -> list[dict]:
    """Set room.name, room.room_type and room.type_evidence on every room of `plan` (in place).

    evidence: {room id: dict(counts=(C,) pixel counts, names=[C class names], images=n, source=str)}.
    folder_types: {room id: type} from photo folder names; they win, and what the images say is kept beside them.
    Returns one summary row per room."""
    entrance = entrance_rooms(plan)
    folder_types = folder_types or {}
    state = {}
    for r in plan.rooms:
        ev = evidence.get(r.id)
        shape = room_shape(r)
        scores, shares, names = None, None, None
        if ev is not None and ev.get("images", 0) > 0 and float(np.sum(ev["counts"])) > 0:
            names = list(ev["names"])
            shares = class_shares(np.asarray(ev["counts"], float))
            scores = type_scores(shares, names)
            if ev.get("view_counts") is not None and float(np.sum(ev["view_counts"])) > 0:
                # glass and sky have no depth (or lie outside the room): a balcony is judged from the view inside it
                scores["balcony"] = type_scores(class_shares(np.asarray(ev["view_counts"], float)), names)["balcony"]
        t, rule, cands = decide(scores, shape, r.id in entrance)
        state[r.id] = dict(type=t, rule=rule, cands=cands, scores=scores, shares=shares, names=names, shape=shape,
                           ev=ev or {}, entrance=r.id in entrance, poly=Polygon(r.polygon).buffer(0))
    _unique(state)
    _living_room_at_entrance(state, plan)
    for rid, s in state.items():
        s["images_say"] = s["type"]
        if rid in folder_types:
            s["type"], s["rule"] = folder_types[rid], "folder name"
    count, used, rows = {}, {}, []
    order = sorted(plan.rooms, key=lambda r: -(r.floor_area.value or 0.0))
    for r in order:
        count[state[r.id]["type"]] = count.get(state[r.id]["type"], 0) + 1
    for r in order:
        s = state[r.id]
        t = s["type"]
        used[t] = used.get(t, 0) + 1
        name = DISPLAY[t] if (count[t] == 1 or t in ("room",) + UNIQUE) else f"{DISPLAY[t]} {used[t]}"
        te = dict(rule=s["rule"], images=int(s["ev"].get("images", 0)), source=s["ev"].get("source", "none"),
                  view_images=int(s["ev"].get("view_images", 0)),
                  shape=dict(length_m=round(s["shape"]["length"], 2), width_m=round(s["shape"]["width"], 2),
                             area_m2=round(s["shape"]["area"], 2), entrance=s["entrance"]))
        if s["rule"] == "folder name":
            te["images_say"] = s["images_say"]
        if s["scores"] is not None:
            te["top_classes"] = top_classes(s["shares"], s["names"])
            te["scores"] = {k: round(v, 4) for k, v in sorted(s["scores"].items(), key=lambda kv: -kv[1])}
        r.name, r.room_type, r.type_evidence = name, t, te
        rows.append(dict(room=r.id, label=r.label, type=t, name=name, rule=s["rule"], images=te["images"],
                         images_say=s["images_say"], top=te.get("top_classes", [])[:3]))
    return rows


def _unique(state: dict) -> None:
    """One kitchen and one living room per home: the strongest claim keeps it (a room under 2 m wide only when no
    wider room claims it), the others take their next type."""
    for t in UNIQUE:
        claim = sorted((rid for rid, s in state.items() if s["type"] == t and s["rule"].startswith("objects")),
                       key=lambda rid: (state[rid]["shape"]["width"] < NARROW_M, -state[rid]["scores"][t]))
        for rid in claim[1:]:
            s = state[rid]
            if s["poly"].distance(state[claim[0]]["poly"]) <= FRAGMENT_GAP_M:
                s["rule"] = f"objects; the same {DISPLAY[t].lower()} as {claim[0]} (the plan splits it)"
                continue
            taken = tuple(u for u in UNIQUE if any(o["type"] == u for k, o in state.items() if k != rid))
            nt, rule, cands = decide(s["scores"], s["shape"], s["entrance"], exclude=(t,) + taken)
            s.update(type=nt, rule=f"{rule}; {DISPLAY[t].lower()} is in {claim[0]}", cands=cands)


def _living_room_at_entrance(state: dict, plan) -> None:
    """A home has a living room. When no room shows a sofa, TV or coffee table, the largest room (>= 9 m2) with the
    entrance is the living room (the Indian 'hall', often with a cot or diwan in it), else the largest room. A room
    with a bed qualifies only when another room shows a bed at least 2.5 times stronger (that is the bedroom). Plans
    of one room are left alone."""
    if len(plan.rooms) < 2 or any(s["type"] == "living_room" for s in state.values()):
        return
    bed = {rid: (s["scores"] or {}).get("bedroom", 0.0) for rid, s in state.items()}

    def ok(rid):
        s = state[rid]
        if s["shape"]["area"] < LIVING_MIN_M2 or s["type"] not in ("room", "bedroom"):
            return False
        return s["type"] == "room" or any(bed[rid] < WEAK_BED * v for k, v in bed.items() if k != rid)
    cands = [r.id for r in sorted(plan.rooms, key=lambda r: -(r.floor_area.value or 0.0)) if ok(r.id)]
    if not cands:
        return
    at_door = [rid for rid in cands if state[rid]["entrance"]]
    rid = (at_door or cands)[0]
    s = state[rid]
    why = "; its bed is weaker than another room's" if s["type"] == "bedroom" else ""
    where = "largest room at the entrance" if at_door else "largest room (no entrance found)"
    s.update(type="living_room", rule=f"{where}, no sofa, TV or coffee table in the plan{why}")


# ------------------------------------------------------------------------------------------------ images per room

def assign_frames(plan, xy: np.ndarray, snap_m: float = 0.3) -> np.ndarray:
    """Room index (into plan.rooms) of each camera position (N, 2), -1 when it is in no room and more than snap_m
    from every room."""
    polys = [Polygon(r.polygon).buffer(0) for r in plan.rooms]
    out = np.full(len(xy), -1, int)
    for i, (u, v) in enumerate(np.asarray(xy, float)):
        p = Point(u, v)
        inside = [k for k, g in enumerate(polys) if g.contains(p)]
        if inside:
            out[i] = min(inside, key=lambda k: polys[k].area)
            continue
        d = [g.distance(p) for g in polys]
        if d and min(d) <= snap_m:
            out[i] = int(np.argmin(d))
    return out


def pick_frames(room_of: np.ndarray, per_room: int) -> np.ndarray:
    """Up to per_room frames per room, evenly spread over the room's frames (indices into room_of)."""
    keep = []
    for k in np.unique(room_of[room_of >= 0]):
        idx = np.flatnonzero(room_of == k)
        if len(idx) > per_room:
            idx = idx[np.round(np.linspace(0, len(idx) - 1, per_room)).astype(int)]
        keep += idx.tolist()
    return np.array(sorted(keep), int)


def evidence_from_maps(plan, room_of_image: dict[str, int], maps: dict[str, np.ndarray], names: list[str],
                       source: str) -> dict[str, dict]:
    """Sum the class pixel counts of each room's images."""
    C = len(names)
    out = {}
    for key, k in room_of_image.items():
        m = maps.get(key)
        if m is None or k < 0:
            continue
        rid = plan.rooms[k].id
        e = out.setdefault(rid, dict(counts=np.zeros(C, np.int64), names=names, images=0, source=source))
        e["counts"] += np.bincount(m.ravel(), minlength=C)[:C]
        e["images"] += 1
    return out


# ------------------------------------------------------------------------------------------------ segmenter

def load_class_maps(npz: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    z = np.load(npz)
    names = [str(x).strip() for x in z["names"]]
    return names, {str(k): z[f"m{i}"] for i, k in enumerate(z["keys"])}


def segment_images(items: list[tuple[str, Path]], out_npz: Path, gpu_lock: Path | None = None,
                   long_side: int = 640, timeout_s: int = 1800, log=print) -> tuple[list[str], dict[str, np.ndarray]]:
    """ADE20K class maps for (key, image path) items with scripts/seg_walls.py in envs/seg (cached in out_npz).
    gpu_lock: run the segmenter under `flock gpu_lock` (when the caller does not hold the GPU lock itself)."""
    from floorplan.photo.semantic import _seg_python, seg_hf_home
    out_npz = Path(out_npz)
    if out_npz.exists():
        names, maps = load_class_maps(out_npz)
        if {k for k, _ in items} <= set(maps):
            return names, maps
    py = _seg_python()
    script = Path(__file__).resolve().parents[2] / "scripts" / "seg_walls.py"
    if py is None or not script.exists():
        raise RuntimeError("no segmentation env (envs/seg; build it with setup/seg_env.sh)")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    lst = out_npz.with_suffix(".list.txt")
    lst.write_text("".join(f"{k}\t{Path(p).resolve()}\n" for k, p in items))
    env = dict(os.environ)
    env["HF_HOME"] = str(seg_hf_home(py))
    env.pop("PYTHONPATH", None)
    cmd = [str(py), str(script), "--list", str(lst), "--out", str(out_npz), "--long-side", str(long_side)]
    if gpu_lock is not None:
        cmd = ["flock", str(gpu_lock)] + cmd
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0 or not out_npz.exists():
        raise RuntimeError(f"segmentation failed (rc {r.returncode}): {r.stderr[-400:]}")
    log(f"[names] segmented {len(items)} images in {time.time() - t0:.0f} s")
    return load_class_maps(out_npz)


# ------------------------------------------------------------------------------------------------ per tier

def photo_evidence(plan, work_dir: Path) -> tuple[dict, dict, dict]:
    """Photo tier: rooms are named after their folders (room.label). Evidence = the class maps the photo front-end
    saved (semantic_raw.npz, keys 'folder/file'). Returns (evidence, folder types, report)."""
    folder_types = {r.id: t for r in plan.rooms if (t := type_from_folder(r.label))}
    raw = Path(work_dir) / "semantic_raw.npz"
    if not raw.exists():
        return {}, folder_types, dict(status=f"no class maps ({raw.name} missing)")
    names, maps = load_class_maps(raw)
    rid_of = {r.label: k for k, r in enumerate(plan.rooms)}
    room_of = {key: rid_of.get(key.split("/")[0], -1) for key in maps}
    ev = evidence_from_maps(plan, room_of, maps, names, "photos of the room's folder")
    return ev, folder_types, dict(status="ok", photos=len(maps), class_maps=str(raw))


def video_frames(scene: dict, info: dict, work_dir: Path) -> list[tuple]:
    """Video tier keyframes on disk (work/frames_r<rot>/sfm/<frame>.jpg, already upright): (key, path, frame, 0)."""
    rot = int((info.get("rotation") or {}).get("final_rotation", (info.get("rotation") or {}).get("rotation", 0)) or 0)
    d = Path(work_dir) / f"frames_r{rot}" / "sfm"
    if not d.is_dir():
        cands = sorted(Path(work_dir).glob("frames_r*/sfm"))
        if not cands:
            return []
        d = cands[0]
    out = []
    for p in sorted(d.glob("*.jpg")):
        try:
            out.append((p.stem, p, int(p.stem), 0))
        except ValueError:
            continue
    return out


def video_depth(scene: dict, info: dict, work_dir: Path):
    """depth(frame) -> (metric depth, K, camera -> aligned 4x4) from the MoGe-2 cache (work/depth_moge2.npz), times
    the segment's scale correction (as fused, video/frontend.py); None when the cache is missing."""
    f = Path(work_dir) / "depth_moge2.npz"
    if not f.exists():
        return None
    z = np.load(f)
    D, K = z["depth"], np.asarray(z["K"], float)
    row = {int(Path(str(n)).stem): i for i, n in enumerate(z["names"])}
    kf = [int(x) for x in scene["kf"]]
    seg = dict(zip(kf, (int(x) for x in scene["kf_segment"]))) if "kf_segment" in scene else {}
    segs = (info.get("scale") or {}).get("segments", [])
    corr = {int(g["id"]): float(g.get("scale_correction") or 1.0) for g in segs}
    T_align, T_wc = np.asarray(scene["T_align"], float), np.asarray(scene["T_wc"], float)

    def get(frame: int):
        if frame not in row:
            return None
        d = D[row[frame]].astype(np.float32) * corr.get(seg.get(frame, -1), 1.0)
        return d, K, T_align @ T_wc[frame]
    return get


def lidar_depth(scene: dict, capture_dir: Path):
    """depth(frame) -> (LiDAR depth: confidence >= 2, 0.2-4 m; its K; camera -> aligned 4x4)."""
    from floorplan.io.stray import load_stray
    cap = load_stray(capture_dir)
    T_align, T_wc = np.asarray(scene["T_align"], float), np.asarray(scene["T_wc"], float)

    def get(frame: int):
        return cap.depth(frame), cap.K_depth(frame), T_align @ T_wc[frame]
    return get


def projected_evidence(plan, frames: list[tuple], maps: dict, names: list[str], depth_of, floor_y: float | None,
                       stride: int = 2, min_pixels: int = 300) -> tuple[dict[str, dict], dict]:
    """Each pixel votes for the room its 3D point lies in (depth + pose -> plan position): what stands in a room
    counts for that room from wherever it was seen, and what is seen through its doorways does not.
    Self-check: the height of the pixels the segmenter calls floor against the plan's floor level (a depth map that
    does not fit its pose puts them elsewhere). Returns (evidence, check)."""
    from shapely import contains_xy
    polys = [Polygon(r.polygon).buffer(0) for r in plan.rooms]
    C = len(names)
    counts = np.zeros((len(polys), C), np.int64)
    images = np.zeros(len(polys), int)
    floor_cls = names.index("floor") if "floor" in names else -1
    floor_h = []
    for key, _, frame, rot in frames:
        sem = maps.get(key)
        got = depth_of(frame) if sem is not None else None
        if got is None:
            continue
        d, K, T = got
        sem = np.rot90(sem, -rot) if rot else sem              # back to the depth map's orientation
        h, w = d.shape
        yi = np.minimum((np.arange(0, h, stride) * sem.shape[0] / h).astype(int), sem.shape[0] - 1)
        xi = np.minimum((np.arange(0, w, stride) * sem.shape[1] / w).astype(int), sem.shape[1] - 1)
        cls = sem[yi][:, xi]
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        dd = d[::stride, ::stride]
        ok = np.isfinite(dd) & (dd > 0.1) & (dd < 8.0)
        x = (uu[ok] - K[0, 2]) / K[0, 0] * dd[ok]
        y = (vv[ok] - K[1, 2]) / K[1, 1] * dd[ok]
        P = np.stack([x, y, dd[ok]], 1) @ T[:3, :3].T + T[:3, 3]
        c = cls[ok]
        if floor_cls >= 0 and (c == floor_cls).sum() >= 50:
            floor_h.append(float(np.median(P[c == floor_cls, 1])))
        for k, poly in enumerate(polys):
            inside = contains_xy(poly, P[:, 0], P[:, 2])
            n = int(inside.sum())
            if n:
                counts[k] += np.bincount(c[inside], minlength=C)[:C]
                images[k] += n >= min_pixels
    ev = {plan.rooms[k].id: dict(counts=counts[k], names=names, images=int(images[k]),
                                 source="pixels whose 3D point lies in the room (depth + pose)")
          for k in range(len(polys)) if counts[k].sum() > 0}
    off = float(np.median(floor_h) - floor_y) if floor_h and floor_y is not None else None
    return ev, dict(floor_pixels_above_floor_m=None if off is None else round(off, 3), frames_with_floor=len(floor_h))


def choose_frames(plan, scene: dict, frames: list[tuple], per_room: int, spread: int,
                  trusted: np.ndarray | None = None):
    """Frames to segment: up to per_room with the camera inside each room, plus `spread` evenly over the capture
    (they also see rooms the camera never entered). Returns (chosen frames, room of each, report)."""
    traj = np.asarray(scene["traj"], float)
    fidx = np.array([fr[2] for fr in frames], int)
    ok = (fidx >= 0) & (fidx < len(traj))
    if trusted is not None and trusted.any():
        ok &= trusted
    frames = [fr for fr, k in zip(frames, ok) if k]
    if not frames:
        return [], np.zeros(0, int), dict(frames=0)
    room_of = assign_frames(plan, traj[[fr[2] for fr in frames]][:, [0, 2]])
    keep = set(pick_frames(room_of, per_room).tolist())
    if spread:
        keep |= set(np.round(np.linspace(0, len(frames) - 1, min(spread, len(frames)))).astype(int).tolist())
    keep = sorted(keep)
    per = {plan.rooms[k].id: int((room_of == k).sum()) for k in range(len(plan.rooms))}
    return [frames[i] for i in keep], room_of[keep], dict(frames=len(frames), outside_rooms=int((room_of < 0).sum()),
                                                          frames_per_room=per, segmented=len(keep))


def lidar_frames(scene: dict, capture_dir: Path, out_dir: Path, indices, long_side: int = 640) -> list[tuple]:
    """Decode Stray Scanner frames (rgb.mp4), turn them upright from the camera pose (OpenCV axes, world +y up) and
    write them as JPEG: (key, path, frame, quarter turns counter-clockwise applied)."""
    import cv2
    from floorplan.io.stray import load_stray
    cap = load_stray(capture_dir)
    T = np.asarray(scene["T_wc"], float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out, want = [], {}
    for i in sorted(set(int(x) for x in indices)):
        up = T[i, :3, :3].T @ np.array([0.0, 1.0, 0.0])            # world up in camera axes (x right, y down)
        k = int(np.round(np.degrees(np.arctan2(up[0], -up[1])) / 90.0)) % 4    # 1: up points right -> turn CCW
        p = out_dir / f"{i:06d}.jpg"
        out.append((p.stem, p, i, k))
        if not p.exists():
            want[i] = (k, p)
    for i, bgr in cap.iter_rgb(sorted(want)):
        k, p = want[i]
        img = np.ascontiguousarray(np.rot90(bgr, k))
        s = long_side / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return [fr for fr in out if fr[1].exists()]


def name_rooms(plan, scene: dict, info: dict, tier: str, capture: Path | None, work_dir: Path,
               cache_dir: Path | None = None, gpu_lock: Path | None = None, per_room: int = 60, spread: int = 200,
               vote: str = "project", log=print) -> dict:
    """Name every room of `plan` (in place) from the tier's images; returns a report (also put in plan.meta).
    work_dir: photo -> the photo work dir (semantic_raw.npz); video -> the run's work dir (keyframes, depth cache).
    cache_dir (default work_dir): the class-map cache, and the decoded LiDAR frames. vote: 'project' (each pixel votes
    for the room its 3D point lies in; needs depth) or 'camera' (a frame votes for the room its camera is in)."""
    t0 = time.time()
    work_dir = Path(work_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else work_dir
    folder_types, rep = {}, {}
    try:
        if tier == "photo":
            ev, folder_types, rep = photo_evidence(plan, work_dir)
        elif tier in ("video", "lidar"):
            trusted = None
            if tier == "video":
                frames = video_frames(scene, info, work_dir)
                if "kf_trusted" in scene and len(scene["kf"]) == len(scene["kf_trusted"]):
                    tk = {int(f) for f, t in zip(scene["kf"], scene["kf_trusted"]) if t}
                    trusted = np.array([fr[2] in tk for fr in frames]) if tk else None
                depth_of = video_depth(scene, info, work_dir) if vote == "project" else None
            else:
                frames = [(f"{i:06d}", None, int(i), 0) for i in np.asarray(scene["kf"], int)]
                depth_of = lidar_depth(scene, Path(capture)) if vote == "project" else None
            chosen, room_of, rep = choose_frames(plan, scene, frames, per_room, spread, trusted)
            if tier == "lidar":
                done = {fr[2]: fr for fr in lidar_frames(scene, Path(capture), cache_dir / "frames",
                                                         [fr[2] for fr in chosen])}
                room_of = np.array([k for fr, k in zip(chosen, room_of) if fr[2] in done], int)
                chosen = [done[fr[2]] for fr in chosen if fr[2] in done]
            names, maps = segment_images([(fr[0], fr[1]) for fr in chosen], cache_dir / "room_names_classes.npz",
                                         gpu_lock, log=log)
            if depth_of is not None:
                ev, rep["projection_check"] = projected_evidence(plan, chosen, maps, names, depth_of,
                                                                 info.get("floor_y"))
                off = rep["projection_check"]["floor_pixels_above_floor_m"]
                if off is None or abs(off) > PROJECTION_MAX_FLOOR_OFF_M:
                    log(f"[names] depth does not fit the poses (floor pixels {off} m off the floor): camera vote")
                    depth_of = None
            view = evidence_from_maps(plan, {fr[0]: int(k) for fr, k in zip(chosen, room_of)}, maps, names,
                                      "keyframes with the camera inside the room")
            if depth_of is None:
                ev = view
            else:
                for rid, e in ev.items():
                    e["view_counts"] = view[rid]["counts"] if rid in view else None
                    e["view_images"] = view[rid]["images"] if rid in view else 0
                for rid in set(view) - set(ev):            # seen from inside but no depth landed in it
                    ev[rid] = view[rid]
            rep.update(status="ok", vote="project" if depth_of is not None else "camera",
                       class_maps=str(cache_dir / "room_names_classes.npz"))
        else:
            raise ValueError(tier)
    except Exception as e:                       # names are an add-on: never lose the plan over them
        ev, rep = {}, dict(status=f"failed: {type(e).__name__}: {e}")
        log(f"[names] WARNING: room names from shape only ({rep['status']})")
    rows = name_plan(plan, ev, folder_types)
    rep.update(rooms=rows, seconds=round(time.time() - t0, 1),
               method="ADE20K classes (SegFormer-B5) of the images of each room, then shape; "
                      "floorplan/plan/room_types.py")
    plan.meta["room_names"] = rep
    log("[names] " + ", ".join(f"{r['room']} {r['name']}" for r in rows))
    return rep
