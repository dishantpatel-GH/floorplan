"""Isaac Sim helpers shared by the interactive tool (teleop.py) and the renderer (render.py).

Run these with the Isaac Sim interpreter that has numpy 1.26 (envs/isaacsim_np126/bin/python, see sim/README.md):
Isaac Sim 5.1 needs numpy==1.26.0 and my env_isaaclab has numpy 2.4.2, which breaks Replicator's annotators
("TypeError: Unable to write from unknown dtype, kind=f, size=0"). Issue I-008.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

# Camera for walks (video and LiDAR) and the preview: the iPhone ARKit video format of the real Stray Scanner sample,
# 1920x1440 with fx = fy = 1598 px (62.0 deg horizontal field of view; TakeHome/Dataset odometry.csv). A phone video
# is the 16:9 centre crop of the same frames (emulate.py), as phones crop their 4:3 sensors for 1080p video.
WALK_W, WALK_H, WALK_F = 1920, 1440, 1598.0
# Photos: iPhone 15 main camera at its default 12 MP 4:3 size (4032x3024, landscape), EXIF 26 mm (35 mm equivalent).
# With the diagonal definition of the 35 mm equivalent, f = 26 * 5040 / 43.27 = 3028 px (67.3 deg horizontal).
# (Photos are 4:3 on phones by default; 16:9 is the VIDEO format, made by emulate.py from the walk frames.)
PHOTO_W, PHOTO_H, PHOTO_F = 4032, 3024, 3028.4


def start_app(headless: bool, width=1600, height=1000):
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": headless, "renderer": "RaytracedLighting", "width": width, "height": height,
                         "anti_aliasing": 3})
    return app


def open_stage(app, usd: str, log=print):
    import omni.usd
    ctx = omni.usd.get_context()
    ctx.open_stage(str(usd))
    t = time.time()
    while True:
        app.update()
        _, loaded, total = ctx.get_stage_loading_status()
        if total == 0 and time.time() - t > 1.0:
            break
    log(f"[isaac] stage loaded in {time.time() - t:.1f}s")
    return ctx.get_stage()


def add_sheets(stage, sheets: list):
    """White A4 sheets flat on the floor: legacy props from the withdrawn scale-sheet protocol (D-013, D-067)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    mat = UsdShade.Material.Define(stage, "/World/SimProps/PaperMat")
    sh = UsdShade.Shader.Define(stage, "/World/SimProps/PaperMat/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.88, 0.88, 0.86))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    for k, s in enumerate(sheets):
        cube = UsdGeom.Cube.Define(stage, f"/World/SimProps/Sheet_{k}")
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(s["xy"][0], s["xy"][1], 0.0004))
        xf.AddRotateZOp().Set(float(s.get("yaw_deg", 0.0)))
        xf.AddScaleOp().Set(Gf.Vec3f(s["size"][0], s["size"][1], 0.0006))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)


def make_camera(stage, path: str, W: int, H: int, f_px: float):
    """USD pinhole camera whose rendered image has fx = fy = f_px and principal point at the image centre
    (cx = W/2 - 0.5 in OpenCV pixel convention: checked to 0.00 mm against ray casting, sim/README.md)."""
    from pxr import Gf, UsdGeom
    cam = UsdGeom.Camera.Define(stage, path)
    focal = 18.0
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(focal * W / f_px)
    cam.GetVerticalApertureAttr().Set(focal * H / f_px)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 200.0))
    op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
    return cam, op


def set_pose(op, T: np.ndarray):
    from pxr import Gf
    op.Set(Gf.Matrix4d(np.asarray(T, float).T.tolist()))


_LIGHT_BASE: dict = {}


def set_lighting(stage, preset: str):
    """'day' = as authored; 'dim' = every light at 20% (the brief's low-light case; the phone's auto exposure is
    emulated afterwards by emulate.py, which brightens and adds the matching sensor noise)."""
    from pxr import UsdLux
    scale = {"day": 1.0, "dim": 0.2, "dark": 0.05}[preset]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI) or prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase):
            a = UsdLux.LightAPI(prim).GetIntensityAttr()
            key = str(prim.GetPath())
            if key not in _LIGHT_BASE:
                _LIGHT_BASE[key] = float(a.Get() or 0.0)
            a.Set(_LIGHT_BASE[key] * scale)


def load_scene_assets(usd: str):
    """Paths of the GT products made by scene_gt.py (built on first use)."""
    usd = Path(usd).resolve()
    gt_dir = usd.parent / ".simcache" / "gt"
    if not (gt_dir / "sim_gt.json").exists():
        raise SystemExit(f"run first (main .venv): python sim/scene_gt.py {usd}")
    return gt_dir, json.loads((gt_dir / "sim_gt.json").read_text())
