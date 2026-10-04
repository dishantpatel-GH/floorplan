#!/usr/bin/env python
"""Editable PDF of my hand-drawn house sketch (own-capture ground truth, 4 Oct).

The sketch (MyHouse_Dataset/IMG_20261004_223533.jpg) is redrawn as a clean schematic in the same layout. Every
dimension becomes a fillable box pre-filled in cm, so the user can correct numbers in any PDF viewer; grey text under
each box says what the sketch says. Units: values written with "cm" are cm, all others are inches (x 2.54).
`scripts/read_plan_pdf.py` reads the boxes back into a CSV.

Run with the small PDF env (reportlab is not in the main .venv):
  envs/pdf/bin/python floorplan-capture/scripts/sketch_plan_pdf.py --sketch MyHouse_Dataset/IMG_20261004_223533.jpg \
      --out MyHouse_Dataset/house_plan_editable.pdf --csv MyHouse_Dataset/house_measurements.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

IN = 2.54
# id, room, what, as written, unit, inch terms (summed) or cm value, field centre in sketch pixels (1500 x 2000 view)
M = [
    ("H1", "Hall", "top wall, open part on the left (dashed)", "104 cm", "cm", 104, (355, 212)),
    ("H2", "Hall", "top wall, solid part", "118+17", "in", (118, 17), (760, 212)),
    ("H3", "Hall", "column at top-right corner, depth", "5", "in", (5,), (862, 300)),
    ("H4", "Hall", "column at top-right corner, width", "9", "in", (9,), (1007, 330)),
    ("H5", "Hall", "right wall (window wall)", "113.5", "in", (113.5,), (1240, 470)),
    ("H6", "Hall", "window on right wall, width", "69 (x 46)", "in", (69,), (1212, 690)),
    ("H7", "Hall", "window on right wall, height", "(69 x) 46", "in", (46,), (1320, 690)),
    ("H8", "Hall", "left wall", "112", "in", (112,), (150, 530)),
    ("H9", "Hall", "bottom wall, left part (read as '9t' = 94?)", "9t (94?)", "in", (94,), (390, 752)),
    ("H10", "Hall", "bottom wall, right part (next to kitchen)", "70.5 cm", "cm", 70.5, (1005, 742)),
    ("H11", "Hall", "step down to the kitchen top wall (dashed)", "28.8 cm", "cm", 28.8, (850, 812)),
    ("H12", "Hall", "dashed drop at the end of the bottom-left wall", "43 cm", "cm", 43, (610, 842)),
    ("P1", "Passage", "opening from the hall side (dashed)", "81 cm", "cm", 81, (465, 862)),
    ("P2", "Passage", "left wall: door", "Door 70 cm", "cm", 70, (318, 952)),
    ("P3", "Passage", "bottom wall (with a door)", "81+32 cm", "cm", 113, (500, 1040)),
    ("P4", "Passage", "right side, down to the room's top wall", "46", "in", (46,), (692, 1030)),
    ("R1", "Room", "top wall, left part", "46", "in", (46,), (445, 1145)),
    ("R2", "Room", "door on top wall (dashed)", "31", "in", (31,), (762, 1084)),
    ("R3", "Room", "wall thickness, kitchen-room wall", "15 cm", "cm", 15, (835, 1145)),
    ("R4", "Room", "top wall, right part 1 (below kitchen)", "59 cm", "cm", 59, (962, 1172)),
    ("R5", "Room", "step in top wall", "12.6 cm", "cm", 12.6, (1262, 1112)),
    ("R6", "Room", "top wall, right part 2", "44 cm", "cm", 44, (1108, 1180)),
    ("R7", "Room", "right wall", "115+22", "in", (115, 22), (1250, 1245)),
    ("R8", "Room", "door on right wall", "29", "in", (29,), (1250, 1345)),
    ("R9", "Room", "bottom wall (window wall)", "118", "in", (118,), (720, 1690)),
    ("R10", "Room", "window on bottom wall, width", "58 (x 47)", "in", (58,), (632, 1592)),
    ("R11", "Room", "window on bottom wall, height", "(58 x) 47", "in", (47,), (740, 1592)),
    ("R12", "Room", "left wall", "110+37", "in", (110, 37), (160, 1380)),
    ("K1", "Kitchen", "top wall", "63", "in", (63,), (1035, 872)),
    ("K2", "Kitchen", "right wall (window wall)", "87", "in", (87,), (1245, 960)),
    ("K3", "Kitchen", "window on right wall, width", "115.5 cm (x 90)", "cm", 115.5, (975, 975)),
    ("K4", "Kitchen", "window on right wall, height", "(115.5 x) 90 cm", "cm", 90, (1083, 975)),
    ("K5", "Kitchen", "bottom wall (kitchen side)", "87", "in", (87,), (1030, 1052)),
]


def cm(v) -> float:
    return round(sum(v) * IN, 1) if isinstance(v, tuple) else float(v)


# walls in sketch pixels (cleaned to straight lines); kind: wall | open (dashed, no wall) | window | door
SEG = [
    # hall
    ((220, 260), (490, 260), "open"), ((490, 260), (1080, 260), "wall"),
    ((1080, 260), (1090, 780), "wall"), ((1085, 420), (1088, 650), "window"),
    ((220, 260), (220, 800), "wall"), ((220, 800), (550, 800), "wall"),
    ((550, 800), (550, 890), "open"), ((550, 890), (378, 890), "open"),
    ((920, 780), (1090, 780), "wall"), ((920, 780), (920, 845), "open"),
    # hall column (5 x 9 in) at the top-right corner
    ((935, 260), (935, 295), "wall"), ((935, 295), (1080, 295), "wall"),
    # passage
    ((378, 890), (378, 1010), "wall"), ((378, 930), (378, 1000), "door"),
    ((378, 1010), (630, 1010), "wall"), ((460, 1010), (560, 1010), "door"),
    ((630, 1010), (630, 1110), "wall"),
    # room
    ((265, 1110), (630, 1110), "wall"), ((630, 1110), (880, 1110), "open"),
    ((880, 1080), (880, 1120), "wall"), ((880, 1120), (1045, 1120), "wall"),
    ((1045, 1120), (1045, 1135), "wall"), ((1045, 1135), (1170, 1135), "wall"),
    ((1170, 1135), (1170, 1640), "wall"), ((1170, 1290), (1170, 1400), "door"),
    ((270, 1640), (1170, 1640), "wall"), ((600, 1640), (920, 1640), "window"),
    ((265, 1110), (270, 1640), "wall"),
    # kitchen
    ((920, 845), (1170, 845), "wall"), ((1170, 845), (1170, 1080), "wall"),
    ((1170, 915), (1170, 1015), "window"), ((880, 1080), (1170, 1080), "wall"),
]
PAIR = {"H6": ("H7", "H6 x H7 (width x height): 69 x 46 in"), "K3": ("K4", "K3 x K4 (width x height): 115.5 x 90 cm"),
        "R10": ("R11", "R10 x R11 (width x height): 58 x 47 in")}
PAIR_SECOND = {v[0] for v in PAIR.values()}
LABELS = [("HALL", (640, 480)), ("KITCHEN", (1040, 925)), ("PASSAGE", (505, 945)), ("ROOM", (720, 1360))]
NOTES = [(1150, 175, "corner: NOT 90 deg"), (1205, 405, "dim. line")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()

    W, H = A4
    s = 0.40
    ox, oy = 44 - 150 * s, H - 70 + 170 * s       # sketch px -> page pt

    def P(x, y):
        return ox + x * s, oy - y * s

    c = canvas.Canvas(str(a.out), pagesize=A4)
    c.setTitle("House plan from the sketch (editable, cm)")
    c.setFont("Helvetica-Bold", 13)
    c.drawString(20, H - 30, "House plan from your sketch: every number is an editable box, in cm")
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.grey)
    c.drawString(20, H - 43, "Grey text under a box = what the sketch says. Numbers without 'cm' were taken as inches and "
                             "converted (x 2.54). '?' = unclear reading: please check.")
    c.setFillColor(colors.black)

    # rooms and walls
    for (x0, y0), (x1, y1), k in SEG:
        (X0, Y0), (X1, Y1) = P(x0, y0), P(x1, y1)
        if k == "wall":
            c.setStrokeColor(colors.black); c.setLineWidth(2.4); c.setDash()
        elif k == "open":
            c.setStrokeColor(colors.grey); c.setLineWidth(1.0); c.setDash(4, 3)
        elif k == "window":
            c.setStrokeColor(colors.HexColor("#1f77b4")); c.setLineWidth(4.0); c.setDash()
        else:  # door
            c.setStrokeColor(colors.HexColor("#d62728")); c.setLineWidth(4.0); c.setDash()
        c.line(X0, Y0, X1, Y1)
    c.setDash()
    c.setStrokeColor(colors.grey); c.setLineWidth(0.6)               # hall right-wall dimension line (as sketched)
    (X0, Y0), (X1, Y1) = P(1180, 262), P(1180, 778)
    c.line(X0, Y0, X1, Y1); c.line(X0 - 4, Y0, X0 + 4, Y0); c.line(X1 - 4, Y1, X1 + 4, Y1)
    (X0, Y0), (X1, Y1) = P(1205, 1112), P(1050, 1128)                 # leader to the step (R5)
    c.line(X0, Y0, X1, Y1)
    for t, (x, y) in LABELS:
        X, Y = P(x, y); c.setFont("Helvetica-Bold", 11); c.setFillColor(colors.HexColor("#444444"))
        c.drawCentredString(X, Y, t)
    c.setFont("Helvetica-Oblique", 7); c.setFillColor(colors.HexColor("#d62728"))
    X, Y = P(1150, 228); c.drawCentredString(X, Y, "top-right corner: NOT 90 deg")

    # editable boxes
    fw, fh = 36, 11
    for mid, room, what, written, unit, v, (x, y) in M:
        X, Y = P(x, y)
        c.acroForm.textfield(name=mid, tooltip=f"{mid} {room}: {what} (sketch: {written}{'' if unit == 'cm' else ' in'})",
                             x=X - fw / 2, y=Y - fh / 2, width=fw, height=fh, value=f"{cm(v):.1f}",
                             fontName="Helvetica", fontSize=8, borderWidth=0.6,
                             borderColor=colors.HexColor("#1f4e9c"), fillColor=colors.HexColor("#fff8d6"),
                             textColor=colors.black, forceBorder=True)
        c.setFont("Helvetica", 5.6); c.setFillColor(colors.grey)
        if mid in PAIR_SECOND:
            continue
        if mid in PAIR:                               # one label under a width x height pair, and an "x" between
            mate = next(m for m in M if m[0] == PAIR[mid][0]); X2, _ = P(*mate[6])
            c.setFont("Helvetica", 8); c.setFillColor(colors.black); c.drawCentredString((X + X2) / 2, Y - 3, "x")
            c.setFont("Helvetica", 5.6); c.setFillColor(colors.grey)
            c.drawCentredString((X + X2) / 2, Y - fh / 2 - 6, PAIR[mid][1])
            continue
        c.drawCentredString(X, Y - fh / 2 - 6, f"{mid}: {written}{'' if unit == 'cm' else ' in'}")

    # legend and notes
    y0 = 118
    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 8); c.drawString(20, y0, "Legend")
    items = [("wall", colors.black, 2.4, None), ("open / no wall (dashed)", colors.grey, 1.0, (4, 3)),
             ("window", colors.HexColor("#1f77b4"), 4, None), ("door", colors.HexColor("#d62728"), 4, None)]
    x = 60
    for t, col, lw, dash in items:
        c.setStrokeColor(col); c.setLineWidth(lw)
        c.setDash(*dash) if dash else c.setDash()
        c.line(x, y0 + 3, x + 22, y0 + 3); c.setDash()
        c.setFont("Helvetica", 7.5); c.setFillColor(colors.black); c.drawString(x + 26, y0, t); x += 26 + 7.5 * len(t) * 0.5 + 30
    room_top = cm((46,)) + cm((31,)) + 59 + 44
    lines = [
        "Notes from you: in the ROOM the lengths leave out the beams (columns), and the small recesses on both sides of",
        "the window wall. The scoring will adjust for that. Hall: the top-right corner is not 90 deg.",
        f"Check: Room top wall R1+R2+R4+R6 = {room_top:.1f} cm against bottom wall R9 = {cm((118,)):.1f} cm "
        f"(difference {abs(room_top - cm((118,))):.1f} cm).",
        f"Room left wall R12 = {cm((110, 37)):.1f} cm, right wall R7 = {cm((115, 22)):.1f} cm: "
        f"{cm((110, 37)) - cm((115, 22)):.1f} cm apart (the top wall steps, R3/R5?). Please confirm.",
        "Drawing is a clean copy of your sketch's layout, not to scale. Page 2: your sketch. Page 3: every value in a table.",
    ]
    c.setFont("Helvetica", 7.3)
    for i, t in enumerate(lines):
        c.drawString(20, y0 - 13 - 10 * i, t)
    c.setFont("Helvetica-Bold", 7.5); c.drawString(20, 40, "Your corrections / comments:")
    c.acroForm.textfield(name="notes", tooltip="free text", x=140, y=14, width=W - 160, height=36, value="",
                         fontName="Helvetica", fontSize=8, borderWidth=0.6, borderColor=colors.HexColor("#1f4e9c"),
                         fillColor=colors.HexColor("#fff8d6"), fieldFlags="multiline", forceBorder=True)
    c.showPage()

    # page 2: the sketch itself
    im = ImageOps.exif_transpose(Image.open(a.sketch)).convert("RGB")
    im.thumbnail((1400, 1400 * im.size[1] // im.size[0]))
    tmp = a.out.with_suffix(".sketch.jpg"); im.save(tmp, quality=80)
    c.setFont("Helvetica-Bold", 12); c.drawString(20, H - 30, "Your sketch (reference)")
    iw, ih = im.size; sc = min((W - 40) / iw, (H - 70) / ih)
    c.drawImage(ImageReader(str(tmp)), 20, H - 45 - ih * sc, iw * sc, ih * sc)
    c.showPage()
    tmp.unlink(missing_ok=True)

    # page 3: the table
    c.setFont("Helvetica-Bold", 12); c.drawString(20, H - 30, "Every measurement (the editable boxes are on page 1)")
    cols = [("ID", 20), ("Room", 52), ("What", 100), ("Sketch says", 350), ("Unit", 445), ("cm", 480)]
    y = H - 52
    c.setFont("Helvetica-Bold", 8)
    for t, x in cols: c.drawString(x, y, t)
    c.setFont("Helvetica", 8)
    rows = []
    for mid, room, what, written, unit, v, _ in M:
        y -= 13
        vals = [mid, room, what, written, "cm" if unit == "cm" else "inch", f"{cm(v):.1f}"]
        for (t, x), val in zip(cols, vals): c.drawString(x, y, val)
        rows.append(dict(id=mid, room=room, item=what, sketch_says=written, unit=unit, value_cm=f"{cm(v):.1f}"))
    c.showPage()
    c.save()
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(M)} editable values)" + (f" and {a.csv}" if a.csv else ""))


if __name__ == "__main__":
    main()
