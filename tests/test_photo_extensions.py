"""D-082: alcoves found from their own walls (photo/polygon.py _open_mouths), on a synthetic 4 x 4 m room."""
import numpy as np

from floorplan.photo.params import PhotoParams
from floorplan.photo.polygon import fit_polygon


def _wall(a, b, normal, step=0.02):
    n = int(np.hypot(*(np.subtract(b, a))) / step) + 1
    P = np.linspace(a, b, n)
    return P, np.tile(normal, (n, 1))


def _room(alcove_depth: float, head: bool):
    """Square room +-2 m around the spin centre; the +x wall has a 1.0 m gap at z -0.5..0.5 leading into an alcove
    alcove_depth deep (its side walls and far wall seen). head: wall above door-head height across the gap (a door)."""
    parts = [_wall((2.0, -2.0), (2.0, -0.5), (-1, 0)), _wall((2.0, 0.5), (2.0, 2.0), (-1, 0)),    # +x, with the gap
             _wall((-2.0, -2.0), (-2.0, 2.0), (1, 0)), _wall((-2.0, 2.0), (2.0, 2.0), (0, -1)),
             _wall((-2.0, -2.0), (2.0, -2.0), (0, 1)),
             _wall((2.0, -0.5), (2.0 + alcove_depth, -0.5), (0, 1)),                                # alcove sides
             _wall((2.0, 0.5), (2.0 + alcove_depth, 0.5), (0, -1)),
             _wall((2.0 + alcove_depth, -0.5), (2.0 + alcove_depth, 0.5), (-1, 0))]                 # far wall
    W = np.concatenate([q[0] for q in parts])
    N = np.concatenate([q[1] for q in parts]).astype(float)
    hi = [_wall((2.0 + alcove_depth, -0.5), (2.0 + alcove_depth, 0.5), (-1, 0)),
          _wall((2.0, 0.5), (2.0 + alcove_depth, 0.5), (0, -1))]
    if head:
        hi.append(_wall((2.0, -0.5), (2.0, 0.5), (-1, 0)))
    H = np.concatenate([q[0] for q in hi])
    HN = np.concatenate([q[1] for q in hi]).astype(float)
    nb = 720
    vis_pts = np.concatenate([W] + ([H] if head else []))
    az = np.arctan2(vis_pts[:, 1], vis_pts[:, 0])
    r = np.hypot(vis_pts[:, 0], vis_pts[:, 1])
    b = ((az + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    R = np.full(nb, np.nan)
    for k in range(nb):
        if (b == k).any():
            R[k] = r[b == k].max()
    data = dict(ok=True, W=W, N=N, pid=np.zeros(len(W), int), cams=np.zeros((1, 2)), vis=R[None], names=["spin"],
                nb=nb, H=H, HN=HN, hid=np.zeros(len(H), int))
    sides = {s: dict(offset=2.0, status="measured", sigma=0.05, tan_extent=(-2.0, 2.0)) for s in ("+x", "-x", "+z", "-z")}
    return dict(ok=True, sides=sides, manhattan_yaw=0.0, cam_height_m=1.35), data


def test_open_alcove_becomes_a_step():
    lay, data = _room(1.0, head=False)
    out = fit_polygon(lay, data, PhotoParams())
    al = [c for c in out["changes"] if c["kind"] == "alcove"]
    assert out["n_vertices"] == 8 and len(al) == 1
    assert al[0]["rule"] == "D-082 open mouth" and al[0]["side"] == "+x"
    assert abs(al[0]["depth_m"] - 1.0) < 0.1 and abs(al[0]["span_m"] - 1.0) < 0.15
    assert abs(out["area_m2"] - 17.0) < 0.3


def test_door_with_a_head_stays_a_rectangle():
    lay, data = _room(1.0, head=True)
    out = fit_polygon(lay, data, PhotoParams())
    assert out["n_vertices"] == 4
    assert any(c["kind"] == "alcove_rejected" and c.get("rule") == "D-082 open mouth" for c in out["changes"])


def test_door_reveal_is_not_an_alcove_wall():
    lay, data = _room(0.2, head=False)            # 0.2 m "alcove" = the reveal of a door in a 0.2 m wall
    out = fit_polygon(lay, data, PhotoParams())
    assert out["n_vertices"] == 4


def test_switch_off_keeps_the_rectangle():
    lay, data = _room(1.0, head=False)
    p = PhotoParams()
    p.poly_ext_walls = False
    assert fit_polygon(lay, data, p)["n_vertices"] == 4
