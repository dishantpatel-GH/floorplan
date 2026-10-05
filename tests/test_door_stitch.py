"""Door-anchored stitching (D-081): the geometry of the door snap, the ray crossings and the joint adjustment."""
import numpy as np

from floorplan.photo import door_stitch as DS
from floorplan.photo.params import PhotoParams


def _frame(box_off, status="measured"):
    off = dict(zip(("+x", "-x", "+z", "-z"), box_off))
    return dict(off=off, box=(-off["-x"], off["+x"], -off["-z"], off["+z"]), photos={}, small=False,
                status={s: status for s in off}, first=None)


def test_rot2_matches_ry():
    for th in (0.3, -1.2, np.pi / 2):
        v = np.array([0.7, -0.4])
        R3 = DS._ry(th) @ np.array([v[0], 0.0, v[1]])
        assert np.allclose(DS._rot2(th) @ v, R3[[0, 2]])


def test_snap_puts_doors_together_a_wall_apart():
    A, B = _frame((2.0, 2.0, 1.5, 1.5)), _frame((1.0, 1.0, 1.2, 1.2))
    dA = dict(side="+x", c=0.4, face=None)                  # A's east wall, 0.4 m along it
    dB = dict(side="+z", c=-0.2, face=None)                 # B's north wall
    psi, t = DS._snap(dA, A, dB, B, 0.2)
    pB = DS._rot2(psi) @ DS.door_point(dB, B) + t           # B's door, in A's frame
    assert np.allclose(pB, DS.door_point(dA, A) + 0.2 * np.array([1.0, 0.0]))
    nB = DS._rot2(psi) @ DS._DIR["+z"]
    assert np.allclose(nB, [-1.0, 0.0], atol=1e-9)         # faces each other


def test_crossings_first_exit():
    fr = _frame((2.0, 2.0, 2.0, 2.0))
    X = np.array([[3.0, 0.5], [3.5, 0.6], [2.8, 0.4]])      # beyond the +x wall, seen from the centre
    c = DS._crossings(np.zeros(2), X, fr)
    assert c["side"] == "+x" and abs(c["c"] - 0.33) < 0.05


def test_adjust_places_room_and_avoids_overlap():
    p = PhotoParams()
    frames = {"A": _frame((2.0, 2.0, 1.5, 1.5)), "B": _frame((1.0, 1.0, 1.0, 1.0))}
    # B east of A: B's origin at x = 2 + 0.2 + 1 = 3.2
    cons = [dict(A="A", B="B", psi=0.0, t=np.array([3.2, 0.3]), n=np.array([1.0, 0.0]), sn=0.05, st=0.1,
                 kind="pnp+door", weight=100)]
    place, dropped = DS.adjust(frames, cons, [], "A", p)
    assert not dropped
    assert np.allclose(place["B"][1], [3.2, 0.3], atol=0.02)
    # a weak constraint that would make them overlap by 0.5 m is pulled back to the overlap tolerance
    cons[0]["t"] = np.array([2.5, 0.0])
    cons[0]["sn"] = cons[0]["st"] = 1.0
    place, _ = DS.adjust(frames, cons, [], "A", p)
    assert place["B"][1][0] > 3.0 - p.overlap_tol_m - 0.05


def test_photo_param_default_and_known():
    p = PhotoParams.from_dict({"door_stitch": True})
    assert p.door_stitch is True
    assert isinstance(PhotoParams().door_stitch, bool)


def test_adjust_drops_a_turn_the_pose_graph_contradicts():
    p = PhotoParams()
    frames = {"A": _frame((2.0, 2.0, 1.5, 1.5)), "B": _frame((1.0, 1.0, 1.0, 1.0))}
    wrong = dict(A="A", B="B", psi=np.pi / 2, t=np.array([3.2, 0.0]), n=None, sn=0.15, st=0.15, kind="pnp+door",
                 weight=200)
    right = dict(A="A", B="B", psi=0.0, t=np.array([3.3, 0.1]), n=None, sn=0.15, st=0.15, kind="pnp", weight=50)
    place, dropped = DS.adjust(frames, [wrong, right], [], "A", p, fixed_rot={"A": 0.0, "B": 0.0})
    assert abs(place["B"][0]) < 1e-9 and np.allclose(place["B"][1], [3.3, 0.1], atol=0.05)
    assert dropped and dropped[0]["reason"] == "rotation contradicts the pose graph"


def test_adjust_keeps_clear_of_fixed_rooms():
    p = PhotoParams()
    frames = {"A": _frame((2.0, 2.0, 1.5, 1.5)), "B": _frame((1.0, 1.0, 1.0, 1.0)), "C": _frame((1.0, 1.0, 1.0, 1.0))}
    # B is pulled onto C's spot (weakly); C is not stitched but placed by the pose graph: an obstacle
    cons = [dict(A="A", B="B", psi=0.0, t=np.array([3.2, 0.0]), n=None, sn=1.0, st=1.0, kind="pnp", weight=50)]
    place, _ = DS.adjust(frames, cons, [], "A", p, obstacles={"C": (0.0, np.array([3.2, 0.5]))})
    bB = DS._box_moved(frames["B"]["box"], 0.0, place["B"][1])
    bC = DS._box_moved(frames["C"]["box"], 0.0, np.array([3.2, 0.5]))
    pen = min(min(bB[1], bC[1]) - max(bB[0], bC[0]), min(bB[3], bC[3]) - max(bB[2], bC[2]))
    assert pen < p.overlap_tol_m + 0.05


def test_undo_stitch_puts_rooms_and_cameras_back():
    T0, T1 = np.eye(4), np.eye(4)
    T1[:3, 3] = [0.4, 0.0, -0.2]                            # cam b moved with its room by the stitch
    old_b = np.eye(4).tolist()
    scene = dict(cam_names=np.array(["a", "b"]), T_wc=np.stack([T0, T1]), traj=np.zeros((2, 3), np.float32))
    info = dict(room_layouts={"A": dict(anchor=dict(centre_uv=[0.0, 0.0])),
                              "B": dict(anchor=dict(centre_uv=[3.2, 0.3]), sides_plan={"+x": 1.0})},
                door_stitch=dict(reference_room="A",
                                 rooms={"A": dict(stitched=True), "B": dict(stitched=True), "C": dict(stitched=False)},
                                 undo=dict(layouts={"A": dict(anchor=dict(centre_uv=[0.0, 0.0]), sides_plan=None),
                                                    "B": dict(anchor=dict(centre_uv=[2.9, 0.0]), sides_plan=None)},
                                           T_wc={"a": np.eye(4).tolist(), "b": old_b})))
    assert DS.moved_rooms(info) == ["B"]
    assert DS.moved_rooms({}) == []
    sc, inf = DS.undo_stitch(scene, info)
    assert inf["room_layouts"]["B"]["anchor"]["centre_uv"] == [2.9, 0.0]
    assert "sides_plan" not in inf["room_layouts"]["B"]
    assert np.allclose(sc["T_wc"][1], np.eye(4)) and np.allclose(sc["traj"][1], 0.0)
    # the stitched scene itself is left as it was (undo_stitch works on copies)
    assert info["room_layouts"]["B"]["anchor"]["centre_uv"] == [3.2, 0.3]
    assert np.allclose(scene["T_wc"][1][:3, 3], [0.4, 0.0, -0.2])
