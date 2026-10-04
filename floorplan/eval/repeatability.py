"""Repeatability: two captures of the same rooms at the same tier, compared element by element.

Gate text (case study, Part 2):
  * "Two captures of the same room at the same tier agree within 1 cm or 0.5% per wall." We read "or" as the more
    lenient of the two: a wall passes if |delta| <= max(1 cm, 0.5% of its length). For walls shorter than 2 m the
    1 cm term applies, for longer walls the 0.5% term. The length used for the percentage is the mean of the two.
  * "Ceiling height ... where a room is captured more than once, spread across captures <= 1 cm.
    Repeatable-but-biased and unrepeatable both fail, and your report says which one you have."
    Spread = max - min over captures. Bias needs ground truth, so without it we can only say
    "repeatable (bias not verifiable)" or "unrepeatable".
Repeatability needs no ground truth, so it is the one accuracy-type gate we can fully evaluate on the sample data.

A wall that exists in one capture's room but has no partner in the other is also a repeatability failure ("same
room in, same plan out"); it is counted, not silently dropped.
"""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass

import numpy as np

from floorplan.eval.match import MatchResult, as_plan_dict

WALL_ABS_TOL_M = 0.01
WALL_REL_TOL = 0.005
CEILING_SPREAD_TOL_M = 0.01
CEILING_GT_TOL_M = 0.015


def wall_tolerance(length_m: float) -> float:
    """Gate tolerance for one wall: within 1 cm OR within 0.5% -> whichever is larger."""
    return max(WALL_ABS_TOL_M, WALL_REL_TOL * length_m)


def _m(meas: dict | None) -> tuple[float | None, float | None]:
    """(value, 1-sigma) of a Measurement dict; sigma from the 95% interval half-width / 1.96."""
    if not meas or meas.get("value") is None:
        return None, None
    lo, hi = meas.get("lo"), meas.get("hi")
    sig = (hi - lo) / 2 / 1.96 if lo is not None and hi is not None else None
    return float(meas["value"]), sig


def _z(a: float, sa: float | None, b: float, sb: float | None) -> float | None:
    """Difference in units of its combined sigma. |z| <= 1.96 for 95% of repeat pairs if intervals are honest."""
    if sa is None or sb is None or (sa ** 2 + sb ** 2) <= 0:
        return None
    return (a - b) / np.sqrt(sa ** 2 + sb ** 2)


@dataclass
class WallRow:
    wall_a: str
    wall_b: str
    room_a: str
    length_a_m: float | None
    length_b_m: float | None
    delta_m: float | None
    tol_m: float | None
    status: str                  # pass | fail | not_verifiable (a length was not observed) | unmatched
    z: float | None


@dataclass
class RoomRow:
    room_a: str
    room_b: str
    area_a_m2: float | None
    area_b_m2: float | None
    area_delta_m2: float | None
    area_delta_rel: float | None
    ceiling_a_m: float | None
    ceiling_b_m: float | None
    ceiling_spread_m: float | None
    ceiling_status: str


@dataclass
class OpeningRow:
    opening_a: str
    opening_b: str
    kind_a: str
    kind_b: str
    width_a_m: float | None
    width_b_m: float | None
    delta_m: float | None


def wall_rows(match: MatchResult, A: dict, B: dict) -> list[WallRow]:
    wa = {w["id"]: w for w in A["walls"]}
    wb = {w["id"]: w for w in B["walls"]}
    rows = []
    for p in match.walls:
        (la, sa), (lb, sb) = _m(wa[p.a].get("length")), _m(wb[p.b].get("length"))
        if la is None or lb is None:
            rows.append(WallRow(p.a, p.b, p.room_a, la, lb, None, None, "not_verifiable", None))
            continue
        d, tol = lb - la, wall_tolerance((la + lb) / 2)
        rows.append(WallRow(p.a, p.b, p.room_a, la, lb, d, tol, "pass" if abs(d) <= tol else "fail",
                            _z(lb, sb, la, sa)))
    matched_rooms_a = {r.a for r in match.rooms}
    matched_rooms_b = {r.b for r in match.rooms}
    for wid in match.unmatched.get("walls_a", []):
        if wa[wid].get("room_id") in matched_rooms_a:
            rows.append(WallRow(wid, "", wa[wid]["room_id"], _m(wa[wid].get("length"))[0], None, None, None,
                                "unmatched", None))
    for wid in match.unmatched.get("walls_b", []):
        if wb[wid].get("room_id") in matched_rooms_b:
            rows.append(WallRow("", wid, "", None, _m(wb[wid].get("length"))[0], None, None, "unmatched", None))
    return rows


def classify_ceiling(values_m: list[float], gt_m: float | None = None) -> str:
    """Name the ceiling-height outcome the way the gate asks for.

    pass                   spread <= 1 cm and |mean - GT| <= 1.5 cm
    repeatable_but_biased  spread <= 1 cm but |mean - GT| > 1.5 cm   (fails the gate)
    unrepeatable           spread > 1 cm                             (fails the gate)
    repeatable_bias_unknown spread <= 1 cm, no GT                     (repeatability part passes; bias unverified)
    not_verifiable         fewer than two captures observed the ceiling
    """
    v = [x for x in values_m if x is not None]
    if len(v) < 2:
        return "not_verifiable"
    spread = max(v) - min(v)
    if spread > CEILING_SPREAD_TOL_M:
        return "unrepeatable"
    if gt_m is None:
        return "repeatable_bias_unknown"
    return "pass" if abs(float(np.mean(v)) - gt_m) <= CEILING_GT_TOL_M else "repeatable_but_biased"


def room_rows(match: MatchResult, A: dict, B: dict) -> list[RoomRow]:
    ra = {r["id"]: r for r in A["rooms"]}
    rb = {r["id"]: r for r in B["rooms"]}
    rows = []
    for p in match.rooms:
        aa, ab = _m(ra[p.a].get("floor_area"))[0], _m(rb[p.b].get("floor_area"))[0]
        ca, cb = _m(ra[p.a].get("ceiling_height"))[0], _m(rb[p.b].get("ceiling_height"))[0]
        da = None if aa is None or ab is None else ab - aa
        spread = None if ca is None or cb is None else abs(cb - ca)
        rows.append(RoomRow(p.a, p.b, aa, ab, da, None if da is None or not aa else da / aa, ca, cb, spread,
                            classify_ceiling([ca, cb])))
    return rows


def opening_rows(match: MatchResult, A: dict, B: dict) -> list[OpeningRow]:
    oa = {o["id"]: o for o in A["openings"]}
    ob = {o["id"]: o for o in B["openings"]}
    rows = []
    for p in match.openings:
        wa, wb = _m(oa[p.a].get("width"))[0], _m(ob[p.b].get("width"))[0]
        rows.append(OpeningRow(p.a, p.b, p.kind_a, p.kind_b, wa, wb,
                               None if wa is None or wb is None else wb - wa))
    return rows


@dataclass
class RepeatabilityReport:
    capture_a: str
    capture_b: str
    tier: str
    walls: list[WallRow]
    rooms: list[RoomRow]
    openings: list[OpeningRow]
    unmatched: dict

    def summary(self) -> dict:
        st = [w.status for w in self.walls]
        judged = [s for s in st if s in ("pass", "fail", "unmatched")]
        deltas = np.array([abs(w.delta_m) for w in self.walls if w.delta_m is not None])
        z = np.array([w.z for w in self.walls if w.z is not None])
        spreads = [r.ceiling_spread_m for r in self.rooms if r.ceiling_spread_m is not None]
        return dict(
            capture_a=self.capture_a, capture_b=self.capture_b, tier=self.tier,
            rooms_matched=len(self.rooms), walls_matched=sum(s in ("pass", "fail", "not_verifiable") for s in st),
            walls_pass=st.count("pass"), walls_fail=st.count("fail"), walls_unmatched=st.count("unmatched"),
            walls_not_verifiable=st.count("not_verifiable"),
            wall_pass_rate=(st.count("pass") / len(judged)) if judged else None,
            wall_delta_median_m=float(np.median(deltas)) if len(deltas) else None,
            wall_delta_p90_m=float(np.percentile(deltas, 90)) if len(deltas) else None,
            wall_gate=("not_verifiable" if not judged else "pass" if st.count("pass") == len(judged) else "fail"),
            wall_interval_coverage=float(np.mean(np.abs(z) <= 1.96)) if len(z) else None,
            ceiling_spread_max_m=max(spreads) if spreads else None,
            ceiling_status=_worst([r.ceiling_status for r in self.rooms]),
            openings_matched=len(self.openings),
            openings_unmatched=len(self.unmatched.get("openings_a", [])) + len(self.unmatched.get("openings_b", [])),
        )

    def to_markdown(self) -> str:
        s = self.summary()
        out = [f"### Repeatability: {self.capture_a} vs {self.capture_b} ({self.tier})", "",
               f"Wall gate: **{s['wall_gate']}** ({s['walls_pass']} pass / {s['walls_fail']} fail / "
               f"{s['walls_unmatched']} unmatched / {s['walls_not_verifiable']} not verifiable). "
               f"Ceiling: **{s['ceiling_status']}**.", "",
               "| wall A | wall B | room | L_A (m) | L_B (m) | delta (cm) | tol (cm) | z | status |",
               "|---|---|---|---|---|---|---|---|---|"]
        for w in self.walls:
            out.append(f"| {w.wall_a} | {w.wall_b} | {w.room_a} | {_f(w.length_a_m, 3)} | {_f(w.length_b_m, 3)} | "
                       f"{_f(w.delta_m, 2, 100)} | {_f(w.tol_m, 2, 100)} | {_f(w.z, 2)} | {w.status} |")
        out += ["", "| room A | room B | area A (m2) | area B (m2) | d area (%) | ceil A (m) | ceil B (m) | "
                "spread (cm) | ceiling status |", "|---|---|---|---|---|---|---|---|---|"]
        for r in self.rooms:
            out.append(f"| {r.room_a} | {r.room_b} | {_f(r.area_a_m2, 2)} | {_f(r.area_b_m2, 2)} | "
                       f"{_f(r.area_delta_rel, 2, 100)} | {_f(r.ceiling_a_m, 3)} | {_f(r.ceiling_b_m, 3)} | "
                       f"{_f(r.ceiling_spread_m, 2, 100)} | {r.ceiling_status} |")
        if self.openings:
            out += ["", "| opening A | opening B | kind A/B | width A (m) | width B (m) | delta (cm) |",
                    "|---|---|---|---|---|---|"]
            for o in self.openings:
                out.append(f"| {o.opening_a} | {o.opening_b} | {o.kind_a}/{o.kind_b} | {_f(o.width_a_m, 3)} | "
                           f"{_f(o.width_b_m, 3)} | {_f(o.delta_m, 2, 100)} |")
        return "\n".join(out) + "\n"

    def walls_csv(self) -> str:
        return _csv([asdict(w) for w in self.walls], extra=dict(capture_a=self.capture_a, capture_b=self.capture_b,
                                                                 tier=self.tier))


_SEVERITY = ["unrepeatable", "repeatable_but_biased", "repeatable_bias_unknown", "pass", "not_verifiable"]


def _worst(statuses: list[str]) -> str:
    """The most severe ceiling status across rooms (one failing room fails the row)."""
    present = [s for s in _SEVERITY if s in statuses]
    if not present:
        return "not_verifiable"
    judged = [s for s in present if s != "not_verifiable"]
    return judged[0] if judged else "not_verifiable"


def _f(x, nd: int, scale: float = 1.0) -> str:
    return "-" if x is None else f"{x * scale:.{nd}f}"


def _csv(rows: list[dict], extra: dict | None = None) -> str:
    if not rows:
        return ""
    rows = [{**(extra or {}), **r} for r in rows]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def repeatability(plan_a, plan_b, match: MatchResult, capture_a: str = "A", capture_b: str = "B",
                  tier: str = "") -> RepeatabilityReport:
    A, B = as_plan_dict(plan_a), as_plan_dict(plan_b)
    return RepeatabilityReport(capture_a, capture_b, tier or A.get("tier", ""), wall_rows(match, A, B),
                               room_rows(match, A, B), opening_rows(match, A, B), match.unmatched)


def ceiling_groups(room_values: dict[tuple[str, str], float | None], links: list[tuple[tuple, tuple]]) -> list[dict]:
    """Group (capture, room) nodes linked by room matches (union-find) and classify each group's ceiling spread.

    Needed because "spread across captures" can involve more than two captures of the same room."""
    parent = {k: k for k in room_values}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: dict = {}
    for k in room_values:
        groups.setdefault(find(k), []).append(k)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        vals = [room_values[m] for m in members]
        seen = [v for v in vals if v is not None]
        out.append(dict(members=[f"{c}:{r}" for c, r in sorted(members)], values_m=vals,
                        spread_m=(max(seen) - min(seen)) if len(seen) >= 2 else None,
                        status=classify_ceiling(vals)))
    return out
