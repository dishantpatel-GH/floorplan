"""Scene geometry of a simulated house, read straight from its USD (pxr only, no Isaac Sim needed).

Why a separate geometry module (decision D-035):
  The renderer (Isaac Sim) is only needed for colour images. Everything geometric is computed here from the same
  USD: the ground truth (scene_gt.py), the walkable area for driving the phone (occupancy), and the LiDAR depth
  maps (emulate.py casts rays against these triangles). One source of geometry means the depth, the ground truth and
  the rendered pixels cannot disagree, and the expensive renderer runs once per path, not once per noise setting.
  sim/check_render_depth.py verifies the agreement with Isaac Sim's own depth output.

World frame here = the USD world: Z up, metres (InteriorAgent scenes have metersPerUnit = 1).
Every mesh gets a class from its InteriorAgent semantic label (wall, floor, ceiling, door, window, mirror, ...),
a 'glass' flag from its material (translucent / OmniGlass), and is skipped if invisible (open door leaves are
hidden in these scenes).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

CACHE_VERSION = 2
STRUCTURE = ("wall", "floor", "ceiling", "door", "window", "doorsill")
_GROUP_RE = re.compile(r"^([a-z_]+?)_\d{4}$")


def _semantic_class(prim) -> str | None:
    p = prim
    while p and p.GetPath().pathString != "/":
        a = p.GetAttribute("semantic:Semantics:params:semanticData")
        if a and a.HasAuthoredValue() and a.Get():
            return str(a.Get()).lower()
        p = p.GetParent()
    return None


def _group(path: str) -> str:
    """'/Root/Meshes/wall/wall_0003/Meshes/wall_0003/mesh_0001' -> 'wall_0003' (the placed object)."""
    for part in path.split("/"):
        if _GROUP_RE.match(part):
            return part
    return path.rsplit("/", 1)[-1]


def _material_source(prim) -> str:
    from pxr import UsdShade
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return ""
    sh = mat.ComputeSurfaceSource("mdl")[0] or mat.ComputeSurfaceSource()[0]
    if not sh:
        return ""
    a = sh.GetPrim().GetAttribute("info:mdl:sourceAsset")
    return str(a.Get().path) if a and a.Get() else ""


def load(usd_path: str | Path, use_cache: bool = True, log=print) -> dict:
    """World-space triangles of every visible mesh plus a per-mesh table.

    Returns {'V': (N,3) float32, 'F': (M,3) int32, 'tri_mesh': (M,) int32, 'meshes': [ {path, group, cls, glass,
    mirror} ], 'usd': str}. Cached next to the scene in .simcache/geom.npz (rebuilt when the USD is newer)."""
    usd_path = Path(usd_path).resolve()
    cache = usd_path.parent / ".simcache" / "geom.npz"
    if use_cache and cache.exists() and cache.stat().st_mtime > usd_path.stat().st_mtime:
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("version") == CACHE_VERSION:
            return dict(V=z["V"], F=z["F"], tri_mesh=z["tri_mesh"], meshes=meta["meshes"], usd=str(usd_path))
    from pxr import Usd, UsdGeom
    t0 = time.time()
    stage = Usd.Stage.Open(str(usd_path))
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    Vs, Fs, Ts, meshes = [], [], [], []
    nv = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        fvc = m.GetFaceVertexCountsAttr().Get()
        fvi = m.GetFaceVertexIndicesAttr().Get()
        if not pts or not fvc or not fvi:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        M = np.asarray(xc.GetLocalToWorldTransform(prim), dtype=np.float64)    # row-vector convention
        pw = pts @ M[:3, :3] + M[3, :3]
        fvc = np.asarray(fvc); fvi = np.asarray(fvi)
        starts = np.concatenate([[0], np.cumsum(fvc)[:-1]])
        tris = []
        for k in range(1, int(fvc.max()) - 1):                                 # fan triangulation
            sel = fvc > k + 1
            s = starts[sel]
            tris.append(np.stack([fvi[s], fvi[s + k], fvi[s + k + 1]], axis=1))
        tri = np.concatenate(tris) if tris else np.zeros((0, 3), int)
        path = prim.GetPath().pathString
        src = _material_source(prim)
        cls = _semantic_class(prim) or _group(path).rsplit("_", 1)[0]
        meshes.append(dict(path=path, group=_group(path), cls=cls,
                           glass=bool(re.search(r"translucent|glass", src, re.I)),
                           mirror=cls == "mirror"))
        Vs.append(pw.astype(np.float32)); Fs.append((tri + nv).astype(np.int32))
        Ts.append(np.full(len(tri), len(meshes) - 1, np.int32))
        nv += len(pw)
    V, F, T = np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ts)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, V=V, F=F, tri_mesh=T,
                        meta=json.dumps(dict(version=CACHE_VERSION, meshes=meshes)))
    log(f"[geom] {len(meshes)} visible meshes, {len(F)} triangles from {usd_path.name} in {time.time() - t0:.1f}s")
    return dict(V=V, F=F, tri_mesh=T, meshes=meshes, usd=str(usd_path))


def mesh_mask(geom: dict, pred) -> np.ndarray:
    """Boolean per mesh."""
    return np.array([bool(pred(m)) for m in geom["meshes"]])


def raycaster(geom: dict, mesh_sel: np.ndarray | None = None):
    """Open3D RaycastingScene over the selected meshes. Returns (scene, tri_ids) where tri_ids maps the scene's
    primitive ids back to rows of geom['F'] (so hits can be traced to a mesh and its class)."""
    import open3d as o3d
    F, T = geom["F"], geom["tri_mesh"]
    tri_sel = np.ones(len(F), bool) if mesh_sel is None else mesh_sel[T]
    tri_ids = np.nonzero(tri_sel)[0]
    sc = o3d.t.geometry.RaycastingScene()
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(geom["V"])
    tm.triangle.indices = o3d.core.Tensor(F[tri_ids])
    sc.add_triangles(tm)
    return sc, tri_ids


def cast(scene, origins: np.ndarray, dirs: np.ndarray, nthreads: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t_hit, primitive_id, normal) for rays; t_hit = inf where nothing is hit, primitive_id = -1 there.
    nthreads 0 = all cores (set it in worker processes, or they oversubscribe the machine)."""
    import open3d as o3d
    rays = np.concatenate([origins, dirs], axis=-1).astype(np.float32).reshape(-1, 6)
    r = scene.cast_rays(o3d.core.Tensor(rays), nthreads=nthreads)
    t = r["t_hit"].numpy()
    pid = r["primitive_ids"].numpy().astype(np.int64)
    pid[~np.isfinite(t)] = -1
    return t, pid, r["primitive_normals"].numpy()


def object_boxes(geom: dict) -> dict[str, dict]:
    """Axis-aligned world box per placed object (group): {'wall_0003': {'cls', 'min', 'max'}}."""
    V, F, T = geom["V"], geom["F"], geom["tri_mesh"]
    out = {}
    groups = {}
    for i, m in enumerate(geom["meshes"]):
        groups.setdefault(m["group"], []).append(i)
    for g, ids in groups.items():
        tri = F[np.isin(T, ids)]
        if not len(tri):
            continue
        p = V[np.unique(tri)]
        out[g] = dict(cls=geom["meshes"][ids[0]]["cls"], min=p.min(0).tolist(), max=p.max(0).tolist())
    return out


def rooms_json(usd_path: str | Path) -> list[dict]:
    """InteriorAgent rooms.json: [{'room_type', 'polygon': [[x, y], ...]}] in the USD world (Z up)."""
    return json.loads((Path(usd_path).parent / "rooms.json").read_text())
