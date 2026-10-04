"""damage v3 (damage_precision): stain-on-the-surface relief check and post-tier counts."""
import numpy as np

from floorplan.damage import RELIEF_MAX_M, _region_relief, tier_counts
from floorplan.damage.project import Ortho, Surface


def _wall_ortho(res=0.005, S=1.0, T=1.0):
    srf = Surface(id="W", kind="wall", room_id="R", origin=np.zeros(3), ax_s=np.array([1.0, 0, 0]),
                  ax_t=np.array([0, 1.0, 0]), normal=np.array([0, 0, 1.0]), size=(S, T))
    H, W = int(T / res), int(S / res)
    return Ortho(surface=srf, res=res, median=np.zeros((H, W, 3), np.float32), count=np.full((H, W), 5),
                 stack=np.zeros((1, H, W, 3), np.float16), view_ids=[0], mask=np.ones((H, W), bool))


def _points(rng, n, s0, s1, t0, t1, d, noise=0.003):
    s, t = rng.uniform(s0, s1, n), rng.uniform(t0, t1, n)
    return np.c_[s, t, d + rng.normal(0, noise, n)]


def _region(o, s0, s1, t0, t1):
    H, W = o.mask.shape
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    s, t = o.surface.pixel_st(cols, rows, o.res)
    return (s >= s0) & (s <= s1) & (t >= t0) & (t <= t1)


def test_flat_stain_is_kept_and_object_is_rejected():
    rng = np.random.default_rng(0)
    o = _wall_ortho()
    wall = _points(rng, 40000, 0, 1, 0, 1, 0.0)
    reg = _region(o, 0.45, 0.55, 0.45, 0.55)
    flat = _region_relief(o, reg, wall)
    assert flat["relief_m"] is not None and abs(flat["relief_m"]) < 0.003
    assert flat["relief_m"] < RELIEF_MAX_M["lidar"]
    # a 15 x 15 cm jar standing 4 cm proud covers the candidate (and its ring, as on floor_only R6-W10)
    jar = _points(rng, 3000, 0.40, 0.60, 0.40, 0.60, 0.04)
    pts = np.r_[wall[~((wall[:, 0] > 0.40) & (wall[:, 0] < 0.60) & (wall[:, 1] > 0.40) & (wall[:, 1] < 0.60))], jar]
    obj = _region_relief(o, reg, pts)
    assert obj["relief_m"] > RELIEF_MAX_M["lidar"]
    assert obj["relief_m"] < RELIEF_MAX_M["video"]          # learned-depth tier keeps it (looser, noisier depth)


def test_abstains_without_points():
    rng = np.random.default_rng(1)
    o = _wall_ortho()
    sparse = _points(rng, 200, 0, 1, 0, 1, 0.0)             # ~2 points per 10x10 cm: photo-tier density
    r = _region_relief(o, _region(o, 0.45, 0.55, 0.45, 0.55), sparse)
    assert r["relief_m"] is None and r["note"].startswith("abstain")


def test_wall_level_is_local():
    """A wall bowed 1 cm from the fitted plane must not look like relief."""
    rng = np.random.default_rng(2)
    o = _wall_ortho()
    r = _region_relief(o, _region(o, 0.45, 0.55, 0.45, 0.55), _points(rng, 40000, 0, 1, 0, 1, 0.010))
    assert abs(r["relief_m"]) < 0.003


def test_tier_counts_after_filtering():
    res = dict(damage=[dict(id="D1", cls="crack", surface_id="W", confidence=0.58),
                       dict(id="D2", cls="water_stain", surface_id="W", confidence=0.70)],
               concealed_flags=[], scope_items=[
                   dict(surface_id="W", region_ids=["D1"], text="crack fill", quantity=dict(value=1, lo=1, hi=1),
                        item_id="X", why="", confidence=0.58),
                   dict(surface_id="W", region_ids=["D2"], text="stain", quantity=dict(value=1, lo=1, hi=1),
                        item_id="Y", why="", confidence=0.70)])
    c = tier_counts(res)
    assert c == dict(confirmed=1, confirmed_by_class={"water_stain": 1}, review=1, concealed_flags=0, scope_items=1)


def test_points_behind_the_plane_reject_two_sided():
    """Glass / mirror / opening: the returns inside the candidate lie behind the wall plane (with_ceiling R7-W2)."""
    rng = np.random.default_rng(3)
    o = _wall_ortho()
    wall = _points(rng, 40000, 0, 1, 0, 1, 0.0)
    keep = ~((wall[:, 0] > 0.40) & (wall[:, 0] < 0.60) & (wall[:, 1] > 0.40) & (wall[:, 1] < 0.60))
    glass = _points(rng, 500, 0.40, 0.60, 0.40, 0.60, -0.04)
    r = _region_relief(o, _region(o, 0.45, 0.55, 0.45, 0.55), np.r_[wall[keep], glass])
    assert r["relief_m"] < -RELIEF_MAX_M["lidar"]


def test_learned_depth_relief_only_demotes_to_review():
    """Video/photo: an off-surface stain stays visible but is never quoted (status review, no scope, no flag)."""
    res = dict(damage=[dict(id="D1", cls="water_stain", surface_id="W", confidence=0.90,
                            relief_review="in front of the surface by 5.0 cm (learned depth)")],
               concealed_flags=[dict(region_id="D1", flag="concealed_x", surface_id="W", rule_id="R", confidence=0.9,
                                     p_condition=1.0, evidence={}, severity="medium", actions=[], rationale="",
                                     reference="")],
               scope_items=[dict(surface_id="W", region_ids=["D1"], text="stain", quantity=dict(value=1, lo=1, hi=1),
                                 item_id="Y", why="", confidence=0.9)])
    c = tier_counts(res)
    assert c == dict(confirmed=0, confirmed_by_class={}, review=1, concealed_flags=0, scope_items=0)
