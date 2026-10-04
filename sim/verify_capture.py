#!/usr/bin/env python
"""Verification pack for a simulated capture, so a person can check it in two minutes (main .venv).

<session>/verify/
  path_map.png          floor plan (ground truth) with the walk path (colour = time), doorway pauses, and every photo
                        as an arrow (position + view direction), coloured by its folder
  photos_<folder>.jpg   contact sheet per photo folder (file name, EXIF time, 35 mm focal, ISO)
  walk_preview.mp4      the phone video, 640 px wide, 4x speed
  summary.md            durations, distances, how much height / tilt / roll varied, photos per folder, LiDAR stats

Usage: python sim/verify_capture.py <session> [--capture <capture dir>]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def path_map(session: Path, cap: Path, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = json.loads((cap / "gt" / "sim_gt.json").read_text())
    sess = json.loads((session / "session.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 12))
    for r in gt["rooms"]:
        P = np.asarray(r["polygon"] + [r["polygon"][0]])
        ax.fill(P[:, 0], P[:, 1], color="0.95")
        ax.plot(P[:, 0], P[:, 1], "k-", lw=1.2)
        ax.text(*r["ceiling"]["center_xy"], r["id"], ha="center", fontsize=9, color="navy", weight="bold")
    for o in gt["openings"]:
        mn, mx = np.asarray(o["box_min"]), np.asarray(o["box_max"])
        ax.add_patch(plt.Rectangle(mn[:2], *(mx - mn)[:2], fill=False, color="tab:red" if o["cls"] == "door" else "tab:blue", lw=0.8))
    for s in gt["sheets"]:
        ax.plot(*s["xy"], "s", color="orange", ms=7, zorder=5)
    rdir = session / "render"
    for tk in sorted(rdir.iterdir()) if rdir.exists() else []:
        if (tk / "poses.npz").exists():
            z = np.load(tk / "poses.npz")
            T = z["T"]
            sc = ax.scatter(T[:, 0, 3], T[:, 1, 3], c=z["t"], cmap="viridis", s=2, zorder=3)
            ax.plot(T[0, 0, 3], T[0, 1, 3], "go", ms=10, zorder=6, label=f"{tk.name} start")
            ax.plot(T[-1, 0, 3], T[-1, 1, 3], "rx", ms=10, zorder=6, label=f"{tk.name} end")
            fig.colorbar(sc, ax=ax, shrink=0.4, label=f"{tk.name} time (s)")
    folders = sorted({p["folder"] for p in sess["photos"]})
    cmap = plt.get_cmap("tab10")
    pj = rdir / "photos" / "photos.json"
    shots = json.loads(pj.read_text())["photos"] if pj.exists() else [dict(folder=p["folder"], pose=p["pose"]) for p in sess["photos"]]
    for p in shots:
        x, y, z, yaw = p["pose"][:4]
        c = cmap(folders.index(p["folder"]) % 10)
        ax.arrow(x, y, 0.35 * np.cos(np.radians(yaw)), 0.35 * np.sin(np.radians(yaw)), width=0.015, color=c, zorder=7)
    for i, f in enumerate(folders):
        ax.plot([], [], "-", color=cmap(i % 10), lw=4, label=f"photos: {f}")
    ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(1.02, 1.0))
    ax.set_aspect("equal")
    ax.set_title(f"{session.name}: walk path (time-coloured), photo arrows, A4 sheets (orange)")
    fig.savefig(out / "path_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def contact_sheets(cap: Path, out: Path, roles: dict | None = None):
    from floorplan.photo import images as I
    root = cap / "photos"
    if not root.exists():
        return []
    made = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(d.glob("*.JPG")) + sorted(d.glob("*.jpg"))
        if not files:
            continue
        tw, th, cols = 400, 300, 4
        rows = (len(files) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * (th + 36)), "white")
        dr = ImageDraw.Draw(sheet)
        t0 = None
        for k, f in enumerate(files):
            ph = I.load_photo(f, d.name)
            t0 = ph.time_s if t0 is None else t0
            im = Image.fromarray(ph.rgb).resize((tw, th))
            x, y = (k % cols) * tw, (k // cols) * (th + 36)
            sheet.paste(im, (x, y))
            idx = int(re.sub(r"\D", "", f.stem) or 0)
            dr.text((x + 4, y + th + 2), f"{f.name}  t+{(ph.time_s or 0) - (t0 or 0):.1f}s  f35={ph.f35}", fill="black")
            if roles and roles.get(idx):
                dr.text((x + 4, y + th + 16), roles[idx], fill=(160, 0, 0))
        p = out / f"photos_{d.name}.jpg"
        sheet.save(p, quality=85)
        made.append(p.name)
    return made


def summary(session: Path, cap: Path, out: Path, sheets):
    sess = json.loads((session / "session.json").read_text())
    L = [f"# Verification pack: {session.name}", ""]
    rdir = session / "render"
    for tk in sorted(rdir.iterdir()) if rdir.exists() else []:
        if not (tk / "poses.npz").exists():
            continue
        z = np.load(tk / "poses.npz")
        P = z["pose_rows"]
        nom = z["nominal"]
        dist = float(np.sum(np.hypot(np.diff(P[:, 1]), np.diff(P[:, 2]))))
        L += [f"## Walk {tk.name}", "",
              f"- {P[-1, 0]:.0f} s, {len(P)} frames at {float(z['fps']):.0f} fps, {dist:.1f} m walked, lighting {z['lighting']}",
              f"- phone height {P[:, 3].min():.2f}-{P[:, 3].max():.2f} m (intended {nom[:, 3].min():.2f}-{nom[:, 3].max():.2f})",
              f"- tilt {P[:, 5].min():.0f} to {P[:, 5].max():.0f} deg; roll {P[:, 6].min():.1f} to {P[:, 6].max():.1f} deg "
              f"(human level {float(z['human_level']):.1f})",
              f"- fastest turn {np.max(np.abs(np.diff(np.unwrap(np.radians(P[:, 4]))))) * float(z['fps']) * 57.3:.0f} deg/s", ""]
    from collections import Counter
    c = Counter(p["folder"] for p in sess["photos"])
    L += ["## Photos", "", "| folder | photos |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(c.items())] + [""]
    for f in sorted((cap / "truth").glob("lidar_*_frame.json")) if (cap / "truth").exists() else []:
        d = json.loads(f.read_text())
        L += [f"## LiDAR {f.stem}", "", f"- frames {d['n']}, high-confidence share {d['conf2_mean']:.2f}, drift model "
              f"{d['drift']}, mirror pixels {d['mirror_px']}, glass pixels {d['glass_px']}", ""]
    L += ["## Files", "", "- `path_map.png`", "- `walk_preview.mp4`"] + [f"- `{s}`" for s in sheets]
    (out / "summary.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--capture", type=Path)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    cap = a.capture or next(iter(sorted(a.session.glob("capture_*"))), None)
    out = a.session / "verify"
    out.mkdir(exist_ok=True)
    path_map(a.session, cap, out)
    roles = {ph["index"]: ph.get("role", "") for ph in json.loads((a.session / "session.json").read_text()).get("photos", [])}
    sheets = contact_sheets(cap, out, roles)
    vids = sorted((cap / "video").glob("*")) if (cap / "video").exists() else []
    if vids:
        subprocess.call(["ffmpeg", "-loglevel", "error", "-y", "-i", str(vids[0]), "-vf", "scale=640:-2,setpts=0.25*PTS",
                         "-an", "-c:v", "libx264", "-crf", "28", str(out / "walk_preview.mp4")])
    summary(a.session, cap, out, sheets)
    print(f"[verify] -> {out}")


if __name__ == "__main__":
    main()
