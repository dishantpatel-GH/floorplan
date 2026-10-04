#!/usr/bin/env python
"""Own-home ground truth from the tape measurements on the hand sketch (MyHouse_Dataset, 4 Oct).

Input: house_measurements.csv (written by scripts/sketch_plan_pdf.py; scripts/read_plan_pdf.py writes the user's
corrected copy with a value_cm_checked column, which wins when present).
Output, in the capture's gt/ folder:
  ground_truth.csv  walls W1..Wn per room, clockwise from above starting at the wall with the room's main door,
                    plus door/window widths and ceiling heights (the scorer's format);
  gt_polygons.json  room outlines in one frame, built from the sketch's layout and the tape lengths, so walls are
                    paired by geometry (scripts/eval_own_capture.py --gt-json), not by order.
The sketch does not pin every room's position exactly (it is not to scale, and the Room's outline does not close by
about 13 cm), so the outlines are for pairing only; every scored length comes from the tape.

Usage: python scripts/house_gt.py MyHouse_Dataset/house_measurements.csv outputs/own_house/capture/gt [--ceiling 2.6289]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict[str, float]:
    out = {}
    for r in csv.DictReader(open(path)):
        v = r.get("value_cm_checked") or r["value_cm"]
        out[r["id"]] = float(v) / 100.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("measurements", type=Path)
    ap.add_argument("gt_dir", type=Path)
    ap.add_argument("--ceiling", type=float, default=2.6289, help="floor to ceiling, m (same in every room)")
    a = ap.parse_args()
    m = load(a.measurements)
    rows = []

    def add(room, kind, item, value, note=""):
        rows.append([room, kind, item, f"{value:.4f}", note])

    # Room (bedroom): W1 = top wall with the door R2, then clockwise
    room_w1 = m["R1"] + m["R2"] + m["R4"]
    add("room", "wall", "W1", room_w1, "top wall: R1 46 in + door R2 31 in + R4 59 cm (door D1 on it)")
    add("room", "wall", "W2", m["R5"], "step in the top wall (R5)")
    add("room", "wall", "W3", m["R6"], "top wall, right part (R6)")
    add("room", "wall", "W4", m["R7"], "right wall 115+22 in (door D2 on it)")
    add("room", "wall", "W5", m["R9"], "window wall 118 in; tape leaves out the small recesses at both ends (user note)")
    add("room", "wall", "W6", m["R12"], "left wall 110+37 in")
    add("room", "door_width", "D1", m["R2"], "door on W1 (31 in)")
    add("room", "door_width", "D2", m["R8"], "door on W4 (29 in)")
    add("room", "window_width", "Win1", m["R10"], "window on W5 (58 x 47 in)")
    add("room", "ceiling_height", "C1", a.ceiling, "tape, same in every room (user)")
    # Hall: W1 = top wall with the 104 cm open entrance part, then clockwise
    hall_w1 = m["H1"] + m["H2"]
    add("hall", "wall", "W1", hall_w1, "top wall: 104 cm open part + 118+17 in (column 5 x 9 in at the right end)")
    add("hall", "wall", "W2", m["H5"], "right wall 113.5 in (window); top-right corner not 90 deg (user)")
    add("hall", "wall", "W3", m["H10"], "bottom wall, right part 70.5 cm (next to the kitchen)")
    add("hall", "wall", "W4", m["H9"], "bottom wall, left part: written '9t', read as 94 in (check)")
    add("hall", "wall", "W5", m["H8"], "left wall 112 in")
    add("hall", "door_width", "D1", m["H1"], "open entrance part of W1 (dashed on the sketch)")
    add("hall", "window_width", "Win1", m["H6"], "window on W2 (69 x 46 in)")
    add("hall", "ceiling_height", "C1", a.ceiling, "tape, same in every room (user)")
    # Kitchen: W1 = top wall (next to the hall); the left side is open to the passage
    add("kitchen", "wall", "W1", m["K1"], "top wall 63 in")
    add("kitchen", "wall", "W2", m["K2"], "right wall 87 in (window 115.5 x 90 cm)")
    add("kitchen", "wall", "W3", m["K5"], "bottom wall 87 in (kitchen side)")
    add("kitchen", "window_width", "Win1", m["K3"], "window on W2")
    add("kitchen", "ceiling_height", "C1", a.ceiling, "tape, same in every room (user)")
    # Passage: only the bathroom door is a clean measurement
    add("passage", "door_width", "D1", m["P2"], "bathroom door on the passage's left wall (70 cm)")

    # outlines in one frame: x east, y north (up on the sketch), origin at the hall's bottom-left corner
    hw, hh = hall_w1, m["H8"]
    hall = dict(id="hall", polygon=[[0, hh], [hw, hh], [hw, 0], [hw - m["H10"], 0], [m["H9"], 0], [0, 0]],
                walls_cw=[dict(name="W1", edge=0, openings=["door_entrance"]), dict(name="W2", edge=1, openings=["window"]),
                          dict(name="W3", edge=2, openings=[]), dict(name="W4", edge=4, openings=[]),
                          dict(name="W5", edge=5, openings=[])])
    kx, ky = hw - m["H10"], -m["H11"]                       # kitchen top-left: below the end of the hall's W3
    kr, kb = kx + m["K1"], ky - m["K2"]
    kitchen = dict(id="kitchen", polygon=[[kx, ky], [kr, ky], [kr, kb], [kr - m["K5"], kb]],
                   walls_cw=[dict(name="W1", edge=0, openings=[]), dict(name="W2", edge=1, openings=["window"]),
                             dict(name="W3", edge=2, openings=[])])
    rt = kb - m["R3"]                                        # room top line: the 15 cm wall below the kitchen
    rr = kr                                                  # room's right wall lines up with the kitchen's (sketch)
    rl = rr - m["R6"] - room_w1
    rb = rt - m["R12"]                                       # close on the left wall (the right wall is 13 cm short)
    room = dict(id="room", polygon=[[rl, rt], [rl + room_w1, rt], [rl + room_w1, rt - m["R5"]], [rr, rt - m["R5"]],
                                    [rr, rb], [rl, rb]],
                walls_cw=[dict(name="W1", edge=0, openings=["door_R2"]), dict(name="W2", edge=1, openings=[]),
                          dict(name="W3", edge=2, openings=[]), dict(name="W4", edge=3, openings=["door_R8"]),
                          dict(name="W5", edge=4, openings=["window"]), dict(name="W6", edge=5, openings=[])])

    a.gt_dir.mkdir(parents=True, exist_ok=True)
    with open(a.gt_dir / "ground_truth.csv", "w", newline="") as f:
        f.write("room_id,item_type,item_id,value_m,notes\n")
        f.write("# own home (OnePlus Nord capture, 4 Oct 2026): tape measurements from the hand sketch; values without "
                "'cm' on the sketch are inches (x 2.54). Room: lengths leave out beams and the recesses at both ends of "
                "the window wall (user note).\n")
        csv.writer(f).writerows(rows)
    (a.gt_dir / "gt_polygons.json").write_text(json.dumps(dict(
        note="outlines for wall pairing only (sketch layout + tape lengths, not to scale); scored lengths come "
             "from ground_truth.csv", rooms=[hall, kitchen, room]), indent=1))
    print(f"{len(rows)} GT rows, 3 outlines -> {a.gt_dir}")


if __name__ == "__main__":
    main()
