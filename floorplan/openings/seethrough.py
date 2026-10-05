"""Open doorways from the depth seen through them (photo tier).

Why: the own bedroom's door to the kitchen side (D1, 0.787 m) stands open, its leaf swung into the room against the
wardrobe. No plan had it: the segmenter measures door-leaf pixels on the wall line, and an open leaf stands at right
angles to the wall, off it; the prior rules need a plan-step gap, a doorway photo or a cut segmenter door. But the dim
photo IMG_20261004_224001 sees the whole gap: through it, the next room's floor and far wall, 1-5 m beyond the wall's
line; on both sides of it, the wall (the hinge-side jamb and the latch-side frame). An opening is where the camera
sees through a wall.

Per view (levelled on its floor like the segmenter's views), per plan wall the camera faces (camera 0.3 m or more on
the room side), every 2nd pixel's ray is crossed with the wall's vertical plane: position t along the wall and height
above the floor at the crossing. Each pixel's 3-D point says what the ray found:
  - through: the point lies 0.5 m or more beyond the plane, or the pixel has no depth (the photo depth stops at 6 m);
  - wall: the point lies within 0.15 m of the plane, 0.15 m or more above the floor (wall, jamb, frame, head);
  - in front: nearer than the plane (furniture, an open leaf): the plane is hidden there.
  1. A 2 cm bin of t is open when, of the rays crossing it 0.3-1.8 m above the floor that are not hidden, at least 60%
     look through and at most 10% end on the wall. A window has wall under it, so its bins are not open.
  2. A span = a run of open bins (gaps up to 6 cm) at least 0.25 m long. Rejected:
     - nothing seen through it lies on the floor beyond the wall (a window, a hatch);
     - wall under it: of the rays crossing it 0.15-0.7 m above the floor, a quarter or more end on the wall (a sill);
     - everything seen through it lies within 1.0 m behind the line (an alcove, a niche, a step; D-082);
     - the view sees the plane 2.2-2.5 m up over it and finds no wall there (no head: an opening to the ceiling).
  3. Jambs: the wall points beside each end (t within 0.3 m outside, 0.1 m inside). Their inner edge (95th / 5th
     percentile of their positions) is the jamb's room-side edge: measured on the wall line, not where the rays first
     get through, so the reveal an oblique view sees does not narrow the opening. An end with no wall seen beside it
     is not seen when the image border cuts the span there or furniture in front hides the jamb (the width is then a
     lower bound); with neither, the see-through just stops: a reflection in a glossy wall or floor, dropped.
Across views: spans on parallel walls within 0.45 m that overlap are one opening (semantic.py's rule). Width and centre
= median of the views that saw both jambs: a photo's pose error moves both of its ends together, so jambs taken from
two photos would put their pose difference into the width. No such view: a lower bound, the widest part one photo
saw. Kept: both jambs seen and 0.55-1.20 m wide; or a lower bound of 0.4-0.95 m (wider is not a door the prior
describes), in 2 views or one with 5% of the photo seen through it.
Into the plan: a door or passage with a width (not from a prior) within 0.5 m on the same wall or the partition's other
face wins, as in priors.py; so does a window there. A door there without a width (a doorway-pair door) takes this one
when both jambs were seen. The rest are added: kind door, "source": "see_through", evidence "seen through". A lower
bound has width status "inferred": the door prior rule d gives it the prior width from the jamb that was seen.
Measured: docs/modules/openings.md, "Open doorways seen through".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from floorplan.model import Adjacency, Measurement, Opening, Plan
from floorplan.openings.semantic import (SemParams, SemView, WallLine, _clusters, _cross, _rays, _robust_sigma,
                                         level_view)


@dataclass
class GapParams:
    step_px: int = 2                    # every 2nd pixel of the depth map (518 x 388 for a photo)
    camera_side_m: float = 0.3          # the camera this far on the wall's room side
    min_wall_m: float = 0.55            # shorter walls (jogs) hold no doorway
    min_incidence_deg: float = 15.0
    beyond_m: float = 0.5               # a point this far beyond the wall plane was seen through it
    face_tol_m: float = 0.15            # a point this close to the plane is on the wall (jamb, frame, head) ...
    face_min_h_m: float = 0.15          # ... when it is this high above the floor (lower: the floor at the threshold)
    band_lo_m: float = 0.3              # rays crossing the plane this high ...
    door_h_m: float = 1.8               # ... up to door height decide whether a bin is open
    bin_m: float = 0.02
    bin_min_px: int = 3
    open_share: float = 0.6             # a bin is open when this share of its unhidden rays look through ...
    wall_share: float = 0.1             # ... and at most this share end on the wall
    hidden_max: float = 0.5             # ... and at most this share of its rays are hidden by something in front
    gap_m: float = 0.06
    min_span_m: float = 0.25
    floor_beyond_m: float = 0.3         # floor points this far beyond the plane: the see-through reaches the floor
    floor_h_m: float = 0.15
    floor_min_px: int = 20
    low_band_m: tuple = (0.15, 0.7)     # rays crossing this low inside the span ...
    low_wall_share: float = 0.25        # ... ending on the wall this often: wall under it (a sill)
    alcove_m: float = 1.0               # everything seen through within this of the line: alcove or step (D-082)
    alcove_share: float = 0.8           # ... in this share of the span's bins
    head_band_m: tuple = (2.2, 2.5)     # the plane this high over the span, when the view sees it, is wall (the head)
    head_min_px: int = 20
    head_wall_share: float = 0.3
    jamb_out_m: float = 0.3             # wall points this far outside an end ...
    jamb_in_m: float = 0.1              # ... or this far inside it are its jamb
    jamb_min_px: int = 15
    border_px: int = 3
    width_m: tuple = (0.55, 1.20)       # both jambs seen
    lower_bound_min_m: float = 0.4      # one jamb seen: what was seen must be this wide ...
    lower_bound_max_m: float = 0.95     # ... and at most this (more than a standard door: not one the prior fits)
    big_frac: float = 0.05              # ... and seen in 2 views, or one view with this share of the photo seen through
    min_views: int = 2
    merge_m: float = 0.5                # a measured door or passage this close along the wall wins (priors.py)
    end_sigma_floor_m: float = 0.03

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def view_spans(v: SemView, walls: list[WallLine], floor_y: float, floor_ids: np.ndarray,
               gp: GapParams) -> tuple[list[dict], list[dict]]:
    """Doorway spans one view sees through the plan's walls, and the spans it left out (with why)."""
    if floor_ids is not None and len(floor_ids):
        v, floor_y, _ = level_view(v, floor_ids, floor_y, SemParams())
    h, w = v.depth.shape
    gv, gu = np.mgrid[0:h:gp.step_px, 0:w:gp.step_px]
    gv, gu = gv.ravel(), gu.ravel()
    d = v.depth[gv, gu].astype(float)
    r = _rays(v, gv, gu)
    C = v.T[:3, 3]
    has_d = d > 0
    P = C[None] + d[:, None] * r
    hp = P[:, 1] - floor_y
    sin_min = np.sin(np.radians(gp.min_incidence_deg))
    edge = (gu <= gp.border_px) | (gu >= w - 1 - gp.border_px)
    found, dropped = [], []
    for wl in walls:
        if wl.L < gp.min_wall_m:
            continue
        lam, t, s_p, inc, s_c = _cross(C, r, d, wl)
        if s_c < gp.camera_side_m:
            continue
        hx = C[1] + lam * r[:, 1] - floor_y
        ok = (lam > 0) & (inc >= sin_min) & (t >= 0.0) & (t <= wl.L)
        thru = ok & ((has_d & (s_p <= -gp.beyond_m)) | ~has_d)
        q = P[:, [0, 2]] - wl.p0
        t_pt = q @ wl.e
        on = has_d & (np.abs(s_p) <= gp.face_tol_m) & (hp >= gp.face_min_h_m)
        front = ok & has_d & (s_p > gp.face_tol_m)
        band = ok & (hx >= gp.band_lo_m) & (hx <= gp.door_h_m)
        if (thru & band).sum() < 20:
            continue
        nb = int(np.ceil(wl.L / gp.bin_m)) + 1
        b = np.clip(np.floor(t / gp.bin_m).astype(int), 0, nb - 1)
        cnt = lambda m: np.bincount(b[m], minlength=nb)  # noqa: E731
        n_all, n_thru, n_front, n_wall = cnt(band), cnt(band & thru), cnt(band & front), cnt(band & on)
        known = n_all - n_front
        open_ = (n_thru >= gp.bin_min_px) & (n_thru >= gp.open_share * np.maximum(known, 1)) & \
                (n_wall <= gp.wall_share * np.maximum(known, 1)) & (n_front <= gp.hidden_max * np.maximum(n_all, 1))
        gap = int(round(gp.gap_m / gp.bin_m))
        runs, start, last = [], None, None
        for i in np.flatnonzero(open_):
            if start is None:
                start = last = i
            elif i - last > gap + 1:
                runs.append((start, last))
                start = last = i
            else:
                last = i
        if start is not None:
            runs.append((start, last))
        for a, z in runs:
            t_lo, t_hi = a * gp.bin_m, (z + 1) * gp.bin_m
            if t_hi - t_lo < gp.min_span_m:
                continue
            rec = dict(view=v.name, kind="door", wall=wl.idx, wall_id=wl.wall.id, room_id=wl.wall.room_id,
                       span=[round(t_lo, 3), round(t_hi, 3)], cam_dist=round(float(s_c), 2))
            ins = ok & (t >= t_lo) & (t < t_hi)
            ins_in = ok & (t >= t_lo + 0.05) & (t < t_hi - 0.05)
            th = ins & thru
            rec["through_frac"] = round(float(th.sum()) * gp.step_px ** 2 / (h * w), 4)
            # the see-through reaches the floor: the next room's floor is seen through it
            n_floor = int((ins & has_d & (hp <= gp.floor_h_m) & (s_p <= -gp.floor_beyond_m)).sum())
            low = ins_in & (hx >= gp.low_band_m[0]) & (hx <= gp.low_band_m[1])
            n_low_wall = int((low & on).sum())
            far = np.where(has_d, -s_p, 99.0)              # no depth: beyond the photo depth range
            bins = np.unique(b[th])
            q90 = np.array([np.quantile(far[th & (b == k)], 0.9) for k in bins])
            shallow = float((q90 <= gp.alcove_m).mean()) if len(q90) else 1.0
            head = ins_in & (hx >= gp.head_band_m[0]) & (hx <= gp.head_band_m[1])
            n_head, n_head_wall = int(head.sum()), int((head & on).sum())
            rec.update(floor_px=n_floor, low_wall_share=round(n_low_wall / max(int(low.sum()), 1), 2),
                       shallow_share=round(shallow, 2), head_px=n_head, head_wall_px=n_head_wall,
                       top_seen=round(float(np.max(hx[ins])), 2))
            why = None
            if n_floor < gp.floor_min_px:
                why = "nothing seen through it lies on the floor beyond the wall (window or hatch)"
            elif n_low_wall >= gp.low_wall_share * max(int(low.sum()), 1) and low.sum() >= 10:
                why = f"wall under it ({rec['low_wall_share']:.0%} of the low rays): a window"
            elif shallow >= gp.alcove_share:
                why = f"everything seen through it within {gp.alcove_m} m of the line: alcove or step"
            elif n_head >= gp.head_min_px and n_head_wall < gp.head_wall_share * n_head:
                why = "the view sees the plane over it and finds no wall there: no head"
            if why:
                dropped.append(dict(rec, reason=why))
                continue
            # jambs: the wall points beside each end; their inner edge is the jamb's room-side edge
            jam = on & (hp >= 0.2) & (hp <= gp.door_h_m)
            lo_pts = t_pt[jam & (t_pt >= t_lo - gp.jamb_out_m) & (t_pt <= t_lo + gp.jamb_in_m)]
            hi_pts = t_pt[jam & (t_pt >= t_hi - gp.jamb_in_m) & (t_pt <= t_hi + gp.jamb_out_m)]
            seen_lo, seen_hi = len(lo_pts) >= gp.jamb_min_px, len(hi_pts) >= gp.jamb_min_px
            e_lo = float(np.quantile(lo_pts, 0.95)) if seen_lo else t_lo
            e_hi = float(np.quantile(hi_pts, 0.05)) if seen_hi else t_hi
            bord_lo = bool((th & edge & (t < t_lo + 0.1)).any())
            bord_hi = bool((th & edge & (t > t_hi - 0.1)).any())
            # an end without a jamb is cut by the image border, or hidden by something in front of the wall beside it
            hid_lo = int((front & (t >= t_lo - gp.jamb_out_m) & (t < t_lo)).sum()) >= gp.jamb_min_px
            hid_hi = int((front & (t > t_hi) & (t <= t_hi + gp.jamb_out_m)).sum()) >= gp.jamb_min_px
            rec.update(t0=e_lo, t1=e_hi, cut0=not seen_lo, cut1=not seen_hi, jamb_px=[len(lo_pts), len(hi_pts)],
                       border=[bord_lo, bord_hi], hidden=[hid_lo, hid_hi], width=round(e_hi - e_lo, 3),
                       used_px=int(th.sum()))
            if e_hi - e_lo < gp.min_span_m:
                dropped.append(dict(rec, reason=f"jamb to jamb only {e_hi - e_lo:.2f} m"))
                continue
            if (not seen_lo and not (bord_lo or hid_lo)) or (not seen_hi and not (bord_hi or hid_hi)):
                # the see-through just stops, with no wall, frame border or furniture beside it: a reflection on a
                # glossy wall or floor (k22's black bathroom tiles), not a doorway
                dropped.append(dict(rec, reason="an end with neither wall, the image border nor furniture beside it"))
                continue
            found.append(rec)
    return found, dropped


def merge_spans(items: list[dict], walls: dict[int, WallLine], gp: GapParams) -> tuple[list[dict], list[dict]]:
    """One record per doorway over the views (semantic.py's clustering and end rule); kept and rejected."""
    kept, rejected = [], []
    p = SemParams()
    for g in _clusters(items, walls, p):
        mem = [items[i] for i in g]
        votes: dict[int, float] = {}
        for m in mem:
            votes[m["wall"]] = votes.get(m["wall"], 0.0) + m["used_px"]
        host = walls[max(votes, key=votes.get)]
        lo_obs, hi_obs, lo_all, hi_all, whole = [], [], [], [], []
        for m in mem:
            wl = walls[m["wall"]]
            a = float((wl.p0 + m["t0"] * wl.e - host.p0) @ host.e)
            z = float((wl.p0 + m["t1"] * wl.e - host.p0) @ host.e)
            ca, cz = m["cut0"], m["cut1"]
            if a > z:
                a, z, ca, cz = z, a, cz, ca
            lo_all.append(a)
            hi_all.append(z)
            if not ca:
                lo_obs.append(a)
            if not cz:
                hi_obs.append(z)
            if not ca and not cz:
                whole.append((a, z))
        views = sorted({m["view"] for m in mem})
        if whole:
            # width and centre from the views that saw both jambs: a photo's pose error moves both of its ends
            # together, so jambs from two photos would put their pose difference into the width (lit: +30 cm)
            wd = float(np.median([z - a for a, z in whole]))
            c = float(np.median([(a + z) / 2 for a, z in whole]))
            lo, hi = c - wd / 2, c + wd / 2
            sig = _robust_sigma([z - a for a, z in whole], gp.end_sigma_floor_m * np.sqrt(2)) / np.sqrt(2)
            sig_ends = [sig, sig]
        else:
            # no photo saw both jambs: a lower bound, the widest part one photo saw, with the jamb it saw
            best = max(range(len(lo_all)), key=lambda i: hi_all[i] - lo_all[i])
            lo, hi = lo_all[best], hi_all[best]
            m = mem[best]
            a_seen, z_seen = not m["cut0"], not m["cut1"]
            wl = walls[m["wall"]]
            if float((wl.p0 + m["t0"] * wl.e - host.p0) @ host.e) > float((wl.p0 + m["t1"] * wl.e - host.p0) @ host.e):
                a_seen, z_seen = z_seen, a_seen
            lo_obs, hi_obs = ([lo] if a_seen else []), ([hi] if z_seen else [])
            sig_ends = [gp.end_sigma_floor_m, gp.end_sigma_floor_m]
        both = bool(whole)
        rec = dict(kind="door", host=host.idx, host_id=host.wall.id, room_id=host.wall.room_id, t0=lo, t1=hi,
                   width=hi - lo, views=len(views), view_names=views[:12], ends_seen=[len(lo_obs), len(hi_obs)],
                   whole_views=len(whole), sigma_ends=sig_ends, px=int(sum(m["used_px"] for m in mem)),
                   max_frac=max(m["through_frac"] for m in mem), walls_voted={walls[k].wall.id: int(v)
                                                                             for k, v in votes.items()})
        why = None
        if both:
            if not gp.width_m[0] <= rec["width"] <= gp.width_m[1]:
                why = f"jamb to jamb {rec['width']:.2f} m outside {gp.width_m[0]}-{gp.width_m[1]} m"
        elif not (lo_obs or hi_obs):
            why = "no jamb seen in any view"
        elif rec["width"] < gp.lower_bound_min_m:
            why = f"one jamb seen, only {rec['width']:.2f} m seen through"
        elif rec["width"] > gp.lower_bound_max_m:
            why = (f"one jamb seen and {rec['width']:.2f} m seen through: wider than a standard door, so the door "
                   f"prior cannot describe it")
        elif len(views) < gp.min_views and rec["max_frac"] < gp.big_frac:
            why = f"one jamb seen, in {len(views)} view(s), {rec['max_frac']:.1%} of the photo seen through"
        if not why and ((hi + lo) / 2 < 0 or (hi + lo) / 2 > host.L):
            why = "centre beyond the ends of its wall"
        (rejected if why else kept).append(dict(rec, reason=why) if why else rec)
    return kept, rejected


def _measured(o: Opening) -> bool:
    """As in priors.py: an opening with a width that no prior gave it wins."""
    return o.width is not None and o.width.value is not None and o.source != "prior"


def _near_door(plan: Plan, lines, c, ln, half: float, pp):
    """A door or passage on this wall line or the partition's other face, centre within max(0.5 m, half of either
    width) of c along the wall; measured ones first (priors._near_opening), then any (a doorway-pair door has no
    width: its centre is the two cameras' mean)."""
    from floorplan.openings.priors import _near_opening
    o = _near_opening(plan, lines, c, ln, pp, measured_only=True)
    if o is not None:
        return o
    for o in plan.openings:
        if o.kind not in ("door", "passage"):
            continue
        d = np.asarray(o.center, float) - np.asarray(c, float)
        if abs(float(d @ ln.n)) > pp.partition_m:
            continue
        hosts = [lines[i] for i in o.wall_ids if i in lines]
        if hosts and not any(abs(float(h.e @ ln.e)) >= pp.parallel_cos for h in hosts):
            continue
        w = o.width.value / 2 if o.width is not None and o.width.value is not None else 0.0
        if abs(float(d @ ln.e)) <= max(pp.merge_m, w, half):
            return o
    return None


def add_to_plan(plan: Plan, kept: list[dict], walls: dict[int, WallLine], gp: GapParams) -> list[dict]:
    """The doorways into the plan: a measured door or passage (or a window) on the same wall within 0.5 m wins."""
    from floorplan.openings.priors import PriorParams, _lines
    pp = PriorParams()
    lines = _lines(plan)
    rooms = {r.id for r in plan.rooms}
    log, k = [], 1
    ids = {o.id for o in plan.openings}
    for rec in sorted(kept, key=lambda r: -r["views"]):
        host = walls[rec["host"]]
        ln = lines.get(host.wall.id)
        lo, hi = rec["t0"], rec["t1"]
        c = host.p0 + host.e * (lo + hi) / 2
        both = rec["ends_seen"][0] > 0 and rec["ends_seen"][1] > 0
        win = None
        for o in plan.openings:
            if o.kind != "window" or o.width is None or o.width.value is None:
                continue
            dd = np.asarray(o.center, float) - c
            if abs(float(dd @ host.n)) <= pp.partition_m and \
                    abs(float(dd @ host.e)) < o.width.value / 2 + (hi - lo) / 2:
                win = o
                break
        if win is not None:
            log.append(dict(rec, action="window there wins", opening_id=win.id))
            continue
        meas = _near_door(plan, lines, c, ln, (hi - lo) / 2, pp) if ln is not None else None
        if meas is not None and _measured(meas):
            meas.evidence = (meas.evidence or "") + f"; also seen through ({rec['views']} view(s), {rec['width']:.2f} m)"
            log.append(dict(rec, action="measured opening wins", opening_id=meas.id))
            continue
        sig = float(np.hypot(*rec["sigma_ends"]))
        how = (f"seen through: {rec['views']} photo(s) see 0.5 m or more beyond the wall here, from the floor up; "
               f"width = jamb to jamb on the wall line (wall seen beside each end in {rec['ends_seen'][0]} / "
               f"{rec['ends_seen'][1]} views)")
        if not both:
            sig = max(sig, 0.45 / 1.96)
            how += "; a jamb was never seen: the width is a lower bound"
        width = Measurement.from_sigma(rec["width"], sig, method=how, status="measured" if both else "inferred")
        if meas is not None:                         # a door there without a width: it takes this one
            if not both:
                log.append(dict(rec, action="door there (width not measured) kept; this one is a lower bound too",
                                opening_id=meas.id))
                continue
            meas.width = width
            meas.center = (float(c[0]), float(c[1]))
            meas.evidence = (meas.evidence or "") + f"; width and centre seen through ({rec['width']:.2f} m)"
            log.append(dict(rec, action="width onto the door there", opening_id=meas.id))
            continue
        wall_ids, room_ids = [host.wall.id], [host.wall.room_id]
        for wl in walls.values():                    # the partition's other face (the next room's wall)
            if wl.wall.id in wall_ids or abs(float(wl.e @ host.e)) < np.cos(np.radians(10)):
                continue
            if abs(float((wl.p0 - host.p0) @ host.n)) > pp.partition_m:
                continue
            a, z = sorted(float((x - host.p0) @ host.e) for x in (wl.p0, wl.p0 + wl.L * wl.e))
            if min(z, hi) - max(a, lo) > 0.5 * (hi - lo):
                wall_ids.append(wl.wall.id)
                if wl.wall.room_id not in room_ids and wl.wall.room_id in rooms:
                    room_ids.append(wl.wall.room_id)
        while f"T{k}" in ids:
            k += 1
        oid = f"T{k}"
        ids.add(oid)
        conf = min(0.8, 0.5 + 0.1 * rec["views"]) if both else 0.45
        o = Opening(id=oid, kind="door", wall_ids=wall_ids, room_ids=[r for r in room_ids if r in rooms],
                    center=(float(c[0]), float(c[1])), width=width, confidence=round(conf, 2), source="see_through",
                    evidence=(f"seen through: an open doorway in {rec['views']} photo(s) "
                              f"({', '.join(n.split('/')[-1] for n in rec['view_names'][:3])}); host {host.wall.id}; "
                              f"jambs seen in {rec['ends_seen'][0]} / {rec['ends_seen'][1]} views"))
        plan.openings.append(o)
        for wid in wall_ids:
            wobj = next((x for x in plan.walls if x.id == wid), None)
            if wobj is not None and oid not in wobj.opening_ids:
                wobj.opening_ids.append(oid)
        if len(o.room_ids) == 2:
            pair = set(o.room_ids)
            if not any({a.room_a, a.room_b} == pair for a in plan.adjacency):
                plan.adjacency.append(Adjacency(o.room_ids[0], o.room_ids[1], oid))
        log.append(dict(rec, action="added", opening_id=oid))
    return log


def see_through_doors(views: list[SemView], labels: list[str], plan: Plan, floor_y: float,
                      walls: dict[int, WallLine], gp: GapParams | None = None) -> dict:
    """The whole rule on one plan: views -> spans -> doorways -> plan. Returns what it did and what it left out."""
    from floorplan.openings.semantic import FLOOR_LABELS
    gp = gp or GapParams()
    idx = {n: i for i, n in enumerate(labels)}
    floor_ids = np.array([idx[n] for n in FLOOR_LABELS if n in idx])
    items, dropped = [], []
    for v in views:
        f, dr = view_spans(v, list(walls.values()), floor_y, floor_ids, gp)
        items += f
        dropped += dr
    kept, rejected = merge_spans(items, walls, gp) if items else ([], [])
    done = add_to_plan(plan, kept, walls, gp)
    return dict(view_spans=items, spans_left_out=dropped, kept=kept, rejected=rejected, done=done,
                params=gp.to_dict())
