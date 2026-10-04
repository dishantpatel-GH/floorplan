#!/usr/bin/env python
"""Walk through the simulated house with a phone and record a capture session (decision D-037).

What it records is only what the phone would give, later (render.py + emulate.py turn the recorded path into phone
files). Here we store the path and the shots; nothing about the house leaks into the capture.

Run (Isaac Sim interpreter with numpy 1.26, see sim/README.md):
    envs/isaacsim_np126/bin/python floorplan-capture/sim/teleop.py --session floorplan-capture/outputs/sim/my_session

Keys (click the viewport once so it has keyboard focus):
    W / S        walk forward / back            A / D     step left / right
    Q / E        lower / raise the phone         arrows    turn left/right, tilt down/up
    Shift        3x faster (to reposition; do not hold while recording)
    R            start / stop a recording (video + LiDAR walk). The HUD shows REC.
    P            take a photo. It goes to the folder of the room in front of the camera (the doorway rule of the
                 capture guide), or to the room picked with 1-9 (0 = automatic again).
    T            toggle "_take2" photo folders (the repeat set of one room)
    Backspace    delete the last photo
    O            portrait / landscape          L   lighting day / dim (low-light takes)
    H            back to the start point
Everything is saved to <session>/session.json after each take and photo. Close the window to finish.
Then: envs/isaacsim_np126/bin/python floorplan-capture/sim/render.py <session>   (see sim/README.md)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default=str(Path(__file__).resolve().parents[2] / "data/sim_scenes/kujiale_0065/kujiale_0065.usda"))
ap.add_argument("--session", required=True, type=Path)
ap.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW"), help="start pose (default: living room)")
ap.add_argument("--selftest", action="store_true", help="drive by a scripted key sequence, then quit (no window needed)")
ap.add_argument("--autoclose", type=float, default=0, help="close after N seconds (GUI smoke test)")
args = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import isaac_common as ic  # noqa: E402

app = ic.start_app(headless=args.selftest, width=1700, height=1050)

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.timeline  # noqa: E402
from sim.rig import Rig, Rooms, Session, Walkable  # noqa: E402

gt_dir, gt = ic.load_scene_assets(args.scene)
stage = ic.open_stage(app, args.scene)
ic.add_sheets(stage, gt["sheets"])
cam, cam_op = ic.make_camera(stage, "/World/Phone", ic.WALK_W, ic.WALK_H, ic.WALK_F)
walk = Walkable.load(gt_dir / "walkable.npz")
rooms = Rooms.load(gt_dir / "sim_gt.json")
if args.start:
    x0, y0, yaw0 = args.start
else:
    first = gt["rooms"][0]
    x0, y0 = first["ceiling"]["center_xy"]
    yaw0 = 90.0
rig = Rig(x0, y0, yaw=yaw0, walkable=walk)
sess = Session(args.session / "session.json", args.scene,
               meta=dict(camera=dict(W=ic.WALK_W, H=ic.WALK_H, f=ic.WALK_F),
                         photo_camera=dict(W=ic.PHOTO_W, H=ic.PHOTO_H, f=ic.PHOTO_F)))
lighting = "day"
ic.set_lighting(stage, lighting)
timeline = omni.timeline.get_timeline_interface()

if not args.selftest:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    vp.camera_path = "/World/Phone"

# ---------------------------------------------------------------------------------------------------- keyboard
K = carb.input.KeyboardInput
pressed: set = set()
events: list = []


def on_key(e, *a):
    if e.type == carb.input.KeyboardEventType.KEY_PRESS:
        pressed.add(e.input)
        events.append(e.input)
    elif e.type == carb.input.KeyboardEventType.KEY_RELEASE:
        pressed.discard(e.input)
    return True


if not args.selftest:
    inp = carb.input.acquire_input_interface()
    kb_sub = inp.subscribe_to_keyboard_events(omni.appwindow.get_default_app_window().get_keyboard(), on_key)

NUM = {getattr(K, f"KEY_{i}"): i for i in range(10)}
state = dict(folder_override=None, take2=False, msg="", msg_t=0.0, warn=[])


def say(m):
    state["msg"], state["msg_t"] = m, time.time()
    print("[teleop]", m, flush=True)


def photo_folder():
    if state["folder_override"] is not None:
        rid = rooms.ids[state["folder_override"]]
    else:
        rid = rooms.looked_at(rig.x, rig.y, rig.yaw) or "00_unknown"
    return rid + ("_take2" if state["take2"] else "")


def handle(ev):
    global lighting
    if ev == K.R:
        if sess.take is None:
            say(f"REC started: {sess.start_take(lighting)}")
        else:
            tk = sess.stop_take()
            say(f"REC stopped: {tk['name']} ({len(tk['samples'])} samples, "
                f"{tk['samples'][-1][0] - tk['samples'][0][0]:.0f} s)" if tk else "REC discarded (too short)")
    elif ev == K.P:
        f = photo_folder()
        n = sess.add_photo(time.time() - sess.t0, rig.pose(), f, lighting)
        say(f"photo {n} -> {f}")
    elif ev == K.BACKSPACE and sess.d["photos"]:
        p = sess.d["photos"].pop()
        sess.save()
        say(f"deleted photo {p['index']} ({p['folder']})")
    elif ev in NUM:
        i = NUM[ev]
        state["folder_override"] = None if i == 0 or i > len(rooms.ids) else i - 1
        say("photo folder: " + ("automatic (room in front)" if state["folder_override"] is None else rooms.ids[i - 1]))
    elif ev == K.T:
        state["take2"] = not state["take2"]
        say(f"repeat set (_take2): {'ON' if state['take2'] else 'off'}")
    elif ev == K.O:
        rig.roll = 0.0 if rig.roll else 90.0
        say("portrait" if rig.roll else "landscape")
    elif ev == K.L:
        lighting = "dim" if lighting == "day" else "day"
        ic.set_lighting(stage, lighting)
        say(f"lighting: {lighting}")
    elif ev == K.H:
        rig.x, rig.y, rig.yaw, rig.pitch = x0, y0, yaw0, -12.0
        say("back to start")


def command():
    def ax(pos, neg):
        return (1.0 if pos in pressed else 0.0) - (1.0 if neg in pressed else 0.0)
    return dict(forward=ax(K.W, K.S), strafe=ax(K.D, K.A), lift=ax(K.E, K.Q), turn=ax(K.LEFT, K.RIGHT),
                tilt=ax(K.UP, K.DOWN), fast=(K.LEFT_SHIFT in pressed or K.RIGHT_SHIFT in pressed) and sess.take is None)


# --------------------------------------------------------------------------------------------------------- HUD
hud = None
if not args.selftest:
    import omni.ui as ui

    class Hud:
        def __init__(self):
            self.win = ui.Window("Phone capture", width=430, height=820)
            self.base, self.scale = self._base_map()
            self.provider = ui.ByteImageProvider()
            with self.win.frame:
                with ui.VStack(spacing=4):
                    self.rec = ui.Label("", height=28, style={"font_size": 22, "color": 0xFF2020FF})
                    self.status = ui.Label("", height=150, word_wrap=True, style={"font_size": 15})
                    self.msg = ui.Label("", height=40, word_wrap=True, style={"font_size": 15, "color": 0xFF40C0FF})
                    h, w = self.base.shape[:2]
                    ui.ImageWithProvider(self.provider, width=w, height=h)
                    ui.Label("W/S walk  A/D step  Q/E height  arrows turn/tilt  Shift fast\n"
                             "R record  P photo  1-9/0 photo folder  T _take2  Backspace undo\n"
                             "O portrait  L lighting  H home", height=60, word_wrap=True,
                             style={"font_size": 13, "color": 0xFFAAAAAA})
            self.last = 0.0

        def _base_map(self):
            g = walk.soft[::-1]                               # image rows go down; plan y goes up
            s = 3                                              # 2 cm grid -> 6 cm per minimap pixel
            g = g[::s, ::s]
            img = np.zeros(g.shape + (4,), np.uint8)
            img[..., 3] = 255
            img[g] = (70, 90, 70, 255)
            st = walk.strict[::-1][::s, ::s]
            img[st] = (90, 130, 90, 255)
            return img, walk.res * s

        def _px(self, x, y):
            H = self.base.shape[0]
            return int((x - walk.origin[0]) / self.scale), H - 1 - int((y - walk.origin[1]) / self.scale)

        def draw(self):
            img = self.base.copy()
            H, W = img.shape[:2]
            if sess.take is not None:
                for s in sess.take["samples"][::5]:
                    u, v = self._px(s[1], s[2])
                    if 0 <= u < W and 0 <= v < H:
                        img[v, u] = (255, 80, 80, 255)
            for p in sess.d["photos"]:
                u, v = self._px(p["pose"][0], p["pose"][1])
                img[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2] = (255, 200, 0, 255)
            u, v = self._px(rig.x, rig.y)
            for k in range(12):
                a = math.radians(rig.yaw)
                uu, vv = int(u + k * math.cos(a)), int(v - k * math.sin(a))
                if 0 <= uu < W and 0 <= vv < H:
                    img[vv, uu] = (255, 255, 255, 255)
            img[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3] = (60, 160, 255, 255)
            self.provider.set_data_array(np.ascontiguousarray(img), [W, H])

        def update(self, warn):
            now = time.time()
            if now - self.last < 0.2:
                return
            self.last = now
            if sess.take is not None:
                d = sess.take["samples"][-1][0] - sess.take["samples"][0][0] if sess.take["samples"] else 0
                self.rec.text = f"● REC {sess.take['name']}  {d:5.1f} s"
            else:
                self.rec.text = "  (not recording)"
            room = rooms.at(rig.x, rig.y) or "-"
            nph = len(sess.d["photos"])
            self.status.text = (f"position  x {rig.x:6.2f}  y {rig.y:6.2f} m   height {rig.z:4.2f} m\n"
                                f"heading {rig.yaw:6.1f} deg   tilt {rig.pitch:5.1f} deg   "
                                f"{'portrait' if rig.roll else 'landscape'}\n"
                                f"room: {room}\nphoto folder (P): {photo_folder()}\n"
                                f"photos: {nph}   takes: {len(sess.d['takes'])}   lighting: {lighting}\n"
                                + ("WARNING: " + ", ".join(warn) if warn else ""))
            self.msg.text = state["msg"] if now - state["msg_t"] < 6 else ""
            self.draw()

    hud = Hud()

# -------------------------------------------------------------------------------------------------- main loop
script = []
if args.selftest:                      # (seconds, keys held, keys pressed once at start)
    script = [(0.5, set(), [K.R]), (3.0, {K.W}, []), (2.0, {K.LEFT}, []), (1.0, {K.UP}, []), (0.5, set(), [K.R]),
              (0.3, set(), [K.P]), (1.0, {K.RIGHT}, [K.P]), (0.3, set(), [K.KEY_2, K.P, K.KEY_0, K.T, K.P])]
say(f"session {args.session}  start ({rig.x:.2f}, {rig.y:.2f})")
last = time.time()
seg, seg_t = 0, time.time()
t_start = time.time()
while app.is_running():
    now = time.time()
    if args.autoclose and now - t_start > args.autoclose:
        break
    dt = min(now - last, 0.1)
    last = now
    if args.selftest:
        if seg >= len(script):
            break
        dur, held, once = script[seg]
        if now - seg_t == 0 or not getattr(app, "_seg_started", False) or app._seg_started != seg:
            app._seg_started = seg
            events.extend(once)
        pressed.clear(); pressed.update(held)
        if now - seg_t > dur:
            seg += 1
            seg_t = now
    while events:
        handle(events.pop(0))
    warn = rig.step(dt, command())
    ic.set_pose(cam_op, rig.matrix())
    if sess.take is not None:
        sess.add_sample(now - sess.t0, rig.pose())
    if timeline.is_playing():          # the house's objects carry rigid-body physics: never let them move
        timeline.stop()
        say("timeline stopped (physics would move the furniture)")
    if hud:
        hud.update(warn)
    app.update()

if sess.take is not None:
    sess.stop_take()
sess.save()
print(f"[teleop] saved {args.session / 'session.json'}: {len(sess.d['takes'])} takes, {len(sess.d['photos'])} photos",
      flush=True)
app.close()
