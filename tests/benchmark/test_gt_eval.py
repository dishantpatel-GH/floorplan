"""Synthetic check of floorplan.benchmark.gt_eval: known plan + GT with deliberate errors, missed and phantom openings."""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from floorplan.benchmark.gt_eval import evaluate, read_gt

def M(v, s=0.005): return dict(value=v, lo=v - 1.96 * s, hi=v + 1.96 * s, unit="m", method="", status="measured")

def make_plan():
    # Room A: 4.0 x 3.0 rectangle; polygon CCW in (u,v) (== clockwise seen from above, issue I-001)
    A = [(0, 0), (4.0, 0), (4.0, 3.0), (0, 3.0)]
    B = [(4.1, 0), (7.1, 0), (7.1, 2.5), (4.1, 2.5)]
    walls, rooms = [], []
    for rid, poly in (("R1", A), ("R2", B)):
        ids = []
        for k in range(4):
            p0, p1 = poly[k], poly[(k + 1) % 4]
            L = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2) ** 0.5
            wid = f"{rid}W{k}"; ids.append(wid)
            walls.append(dict(id=wid, room_id=rid, p0=p0, p1=p1, length=M(L + (0.012 if (rid, k) == ("R1", 2) else 0.0)),
                              normal=(0, 0), opening_ids=["O1"] if (rid, k) == ("R1", 1) else []))
        rooms.append(dict(id=rid, label="", polygon=poly, wall_ids=ids, ceiling_height=M(2.70)))
    openings = [dict(id="O1", kind="door", wall_ids=["R1W1"], room_ids=["R1", "R2"], center=(4.05, 1.0), width=M(0.815)),
                dict(id="O2", kind="door", wall_ids=["R2W2"], room_ids=["R2"], center=(5.5, 2.5), width=M(0.70))]
    return dict(rooms=rooms, walls=walls, openings=openings)

GT = """room_id,item_type,item_id,description,value_m,measured_with,notes
bed,wall,W1,door wall,3.000,tape,
bed,wall,W2,,4.000,tape,
bed,wall,W3,,3.000,tape,
bed,wall,W4,,4.000,tape,
bed,ceiling_height,C1,,2.705,tape,
bed,door_width,D1,,0.810,tape,
bed,door_width,D2,,0.900,tape,a door we will miss
study,wall,W1,,2.500,tape,
study,wall,W2,,3.000,tape,
study,wall,W3,,2.500,tape,
study,wall,W4,,3.000,tape,
"""

def test():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "gt.csv"; p.write_text(GT)
        res = evaluate(make_plan(), read_gt(p), "video")
    walls = [r for r in res["rows"] if r["kind"] == "wall"]
    assert len(walls) == 8, walls
    bed = {r["item"]: r for r in walls if r["room"] == "bed"}
    assert bed["W1"]["pred_id"] == "R1W1", bed["W1"]          # the door wall must be W1
    errs = sorted(round(abs(r["err"]), 4) for r in walls)
    assert errs[-1] == 0.012 and errs[0] == 0.0, errs           # the one injected error is recovered
    doors = [r for r in res["rows"] if r["kind"] == "door_width"]
    assert any("MISSED" in r["note"] for r in doors) and any(r["pred_id"] == "O1" and abs(r["err"] - 0.005) < 1e-9 for r in doors)
    g = res["gates"]
    assert g["wall_gate"]["passed"] and g["ceiling_gate"]["passed"] and not g["opening_gate"]["passed"]
    assert g["opening_gate"]["phantom"] == 1, g  # O2 only; O1 (shared door) matched in bed
    print("gt_eval synthetic test passed:", json.dumps(g, indent=1))

if __name__ == "__main__":
    test()
