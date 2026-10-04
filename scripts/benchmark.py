"""Benchmark: every plan found under outputs/ -> repeatability, reference-based tier gates, calibration, drift.

Usage:
  python scripts/benchmark.py                      # discover outputs/<producer>/<capture>/plan.json
  python scripts/benchmark.py --gt-dir GT/         # GT/<capture>/plan.json (tape/laser, same schema) if it exists
  python scripts/benchmark.py --include-synthetic  # also score plans whose meta says synthetic (normally skipped)

Writes outputs/benchmark/report.md, gates.csv, repeatability_walls.csv and benchmark.json.

Design rules (eval.md explains each):
  * Never crash on a missing piece: no plans, no registration, one tier missing, a plan with no ceiling. Every
    missing piece becomes a "not_verifiable" row that says why.
  * No capture-specific constants. Capture pairs come from registration (any two captures that overlap are a
    repeat pair for the rooms they share); tiers come from each plan's "tier" field.
  * Repeatability compares plans of the SAME producer and tier only (different producers are different methods).
  * Video/photo plans are compared with the LiDAR plan(s) of the same capture: basis "lidar_reference", which is
    labelled everywhere as reference-based, not ground truth.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplan.eval.gates import (GATES, GateResult, ceiling_spread_gate, drift_gate,  # noqa: E402
                                  interval_calibration, not_verifiable_without_gt, reference_gates,
                                  repeatability_gate)
from floorplan.eval.match import aligned_plan_b, as_plan_dict, match_plans  # noqa: E402
from floorplan.eval.register import RegisterParams, plan_evidence, register  # noqa: E402
from floorplan.eval.repeatability import ceiling_groups, repeatability  # noqa: E402

SKIP_PRODUCERS = {"eval", "benchmark", "explore"}
TIER_DIRS = {"lidar", "video", "photo"}   # layouts like outputs/runs/<capture>/<tier>/plan.json
MIN_DISTINCT = 1.2          # registration accepted only if the winning yaw quadrant is clearly better ...
MIN_INLIER = 0.2            # ... and enough wall points agree after ICP


@dataclass
class PlanEntry:
    producer: str
    capture: str             # directory name holding plan.json
    scene: str | None        # prepared scene this capture corresponds to (for registration)
    tier: str
    path: Path
    plan: dict


# ----------------------------------------------------------------------------------------------- discovery

def discover(outputs: Path, include_synthetic: bool) -> tuple[list[PlanEntry], list[str]]:
    scenes = sorted(p.parent.name for p in outputs.glob("*/scene.npz"))
    entries, skipped = [], []
    for path in sorted(outputs.rglob("plan.json")):
        rel = path.relative_to(outputs).parts
        if len(rel) < 3 or rel[0] in SKIP_PRODUCERS:
            continue
        try:
            plan = as_plan_dict(path)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append(f"{path}: unreadable ({e})")
            continue
        if plan.get("meta", {}).get("synthetic") and not include_synthetic:
            skipped.append(f"{path}: synthetic test plan")
            continue
        capture, scene = _capture_of(rel[1:-1], scenes)
        entries.append(PlanEntry(rel[0], capture, scene, str(plan.get("tier", "unknown")), path, plan))
    return entries, skipped


def _capture_of(parts: tuple[str, ...], scenes: list[str]) -> tuple[str, str | None]:
    """The path component naming the capture, and its prepared scene. Accepts <capture>/plan.json and
    <capture>/<tier>/plan.json; a component matching a known scene wins."""
    for part in parts:
        scene = next((s for s in scenes if part == s or part.startswith(s)), None)
        if scene is not None:
            return part, scene
    named = [p for p in parts if p not in TIER_DIRS]
    return (named[-1] if named else parts[-1]), None


# ----------------------------------------------------------------------------------------------- registration

class Registry:
    """Scene-to-scene transforms from scripts/register_scenes.py, with plan-to-plan registration as a fallback."""

    def __init__(self, reg_dir: Path):
        self.T: dict[tuple[str, str], np.ndarray] = {}
        self.info: dict[tuple[str, str], dict] = {}
        for f in sorted(reg_dir.glob("*__*.json")) if reg_dir.exists() else []:
            d = json.loads(f.read_text())
            if "T2" in d:
                self.T[(d["a"], d["b"])] = np.array(d["T2"])
                self.info[(d["a"], d["b"])] = d

    def scene_T(self, a: str, b: str) -> tuple[np.ndarray | None, str]:
        """T mapping scene b's plan coordinates into scene a's, and a note on where it came from."""
        if a == b:
            return np.eye(3), "same capture"
        for key, inv in (((a, b), False), ((b, a), True)):
            if key in self.T:
                d = self.info[key]
                if d["distinctiveness"] < MIN_DISTINCT or d["inlier_frac_b"] < MIN_INLIER:
                    return None, f"registration {key} unreliable"
                return (np.linalg.inv(self.T[key]) if inv else self.T[key]), "scene registration"
        return None, "no scene registration (run scripts/register_scenes.py)"


def plan_T(A: dict, B: dict) -> tuple[np.ndarray | None, str]:
    """Register two plans directly from their walls (for plans in different frames, e.g. video vs LiDAR)."""
    ea, eb = plan_evidence(A), plan_evidence(B)
    if len(ea.wall_pts) < 50 or len(eb.wall_pts) < 50:
        return None, "too few walls to register plans"
    reg = register(ea, eb, RegisterParams())
    if reg.distinctiveness < MIN_DISTINCT:
        return None, f"plan registration ambiguous (distinctiveness {reg.distinctiveness:.2f})"
    return np.array(reg.T2), f"plan registration (rmse {reg.residual_rmse_m * 100:.1f} cm)"


def best_match(a: PlanEntry, b: PlanEntry, registry: Registry):
    """Match two plans of different captures. Candidate transforms: the scene registration (independent of the
    plans) and a plan-to-plan registration (needed when a plan was built in another frame, e.g. on the
    drift-corrected scene). The candidate that matches more walls wins; the report says which one was used."""
    cands = []
    if a.scene and b.scene:
        cands.append(registry.scene_T(a.scene, b.scene))
    cands.append(plan_T(a.plan, b.plan))
    tried = [(match_plans(a.plan, b.plan, T), how, T) for T, how in cands if T is not None]
    if not tried:
        return None, "; ".join(how for _, how in cands), None
    return max(tried, key=lambda x: (len(x[0].walls), len(x[0].rooms)))


# ----------------------------------------------------------------------------------------------- stages

def plot_repeat(a: PlanEntry, b: PlanEntry, m, T: np.ndarray, rep, path: Path) -> None:
    """A's walls in grey; B's walls (as matched) coloured by repeatability status: where do the plans differ?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B = aligned_plan_b(b.plan, T, m)
    status = {w.wall_b: w.status for w in rep.walls if w.wall_b}
    colour = {"pass": "#2a9d8f", "fail": "#e76f51", "unmatched": "#8d5fd3", "not_verifiable": "#999999"}
    fig, ax = plt.subplots(figsize=(10, 10))
    for w in a.plan["walls"]:
        ax.plot(*np.array([w["p0"], w["p1"]]).T, color="0.6", lw=4, alpha=0.5, solid_capstyle="butt")
    matched_rooms = {r.b for r in m.rooms}
    for w in B["walls"]:
        if w.get("room_id") in matched_rooms:
            ax.plot(*np.array([w["p0"], w["p1"]]).T, color=colour[status.get(w["id"], "unmatched")], lw=1.5)
    for k, c in colour.items():
        ax.plot([], [], color=c, label=f"B wall: {k}")
    ax.plot([], [], color="0.6", lw=4, alpha=0.5, label=f"A walls ({a.capture})")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    s = rep.summary()
    ax.set_title(f"{a.producer}: {b.capture} on {a.capture}\n{s['walls_pass']} pass / {s['walls_fail']} fail / "
                 f"{s['walls_unmatched']} unmatched walls in {s['rooms_matched']} matched rooms", fontsize=10)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_repeatability(entries: list[PlanEntry], registry: Registry, log,
                      fig_dir: Path | None = None) -> tuple[list, list, list[GateResult]]:
    reports, notes, gates = [], [], []
    groups: dict[tuple[str, str], list[PlanEntry]] = {}
    for e in entries:
        groups.setdefault((e.producer, e.tier), []).append(e)
    for (producer, tier), es in sorted(groups.items()):
        ceil_vals, links = {}, []
        for e in es:
            for r in e.plan.get("rooms", []):
                ceil_vals[(e.capture, r["id"])] = (r.get("ceiling_height") or {}).get("value")
        for a, b in itertools.combinations(es, 2):
            if (a.scene or a.capture) == (b.scene or b.capture):
                notes.append(f"{producer}/{tier}: {a.capture} vs {b.capture} skipped (same capture)")
                continue
            m, how, T = best_match(a, b, registry)
            if m is None:
                notes.append(f"{producer}/{tier}: {a.capture} vs {b.capture} skipped: {how}")
                continue
            if not m.rooms:
                notes.append(f"{producer}/{tier}: {a.capture} vs {b.capture}: no shared rooms")
                continue
            rep = repeatability(a.plan, b.plan, m, a.capture, b.capture, tier)
            reports.append((producer, rep))
            if fig_dir is not None:
                fig_dir.mkdir(parents=True, exist_ok=True)
                plot_repeat(a, b, m, T, rep, fig_dir / f"repeat_{producer}_{tier}_{a.capture}__{b.capture}.png")
            links += [((a.capture, p.a), (b.capture, p.b)) for p in m.rooms]
            g = repeatability_gate(rep.summary(), producer)
            g.detail += f" [transform: {how}]"
            gates.append(g)
            log(f"[repeat] {producer}/{tier} {a.capture} vs {b.capture}: {rep.summary()['wall_gate']} ({how})")
        gates.append(ceiling_spread_gate(ceiling_groups(ceil_vals, links), tier, producer))
    return reports, notes, gates


def run_reference(entries: list[PlanEntry], gt_dir: Path | None, log) -> tuple[list[GateResult], list[str]]:
    """Video/photo vs LiDAR of the same capture (reference-based), any tier vs GT when it exists."""
    out, notes = [], []
    lidar = [e for e in entries if e.tier == "lidar"]
    for e in entries:
        refs: list[tuple[str, dict, str]] = []
        if gt_dir is not None and (gt_dir / e.capture / "plan.json").exists():
            refs.append(("ground_truth", as_plan_dict(gt_dir / e.capture / "plan.json"), "GT"))
        if e.tier != "lidar":
            refs += [("lidar_reference", r.plan, r.producer) for r in lidar if r.scene == e.scene]
        if not refs:
            if e.tier == "lidar":
                for g in not_verifiable_without_gt(e.tier, e.capture):
                    g.producer = e.producer
                    out.append(g)
            else:
                notes.append(f"{e.producer}/{e.capture} ({e.tier}): no LiDAR reference plan for this capture")
            continue
        for basis, ref, ref_name in refs:
            T, how = plan_T(ref, e.plan)
            if T is None:
                notes.append(f"{e.producer}/{e.capture} vs {ref_name}: {how}")
                continue
            m = match_plans(ref, e.plan, T)
            for g in reference_gates(ref, e.plan, m, basis, e.tier, e.capture):
                g.producer = e.producer
                g.detail = f"[ref {ref_name}; {how}] {g.detail}"
                out.append(g)
            log(f"[reference] {e.producer}/{e.capture} ({e.tier}) vs {ref_name} ({basis}): {len(m.rooms)} rooms")
    return out, notes


def drift_evidence(outputs: Path, scene: str | None) -> dict | None:
    """The drift module's on/off ablation for this capture, if it exists (outputs/drift/<scene>/ablation.json)."""
    f = outputs / "drift" / str(scene) / "ablation.json"
    if scene is None or not f.exists():
        return None
    v = json.loads(f.read_text()).get("variants", {})
    on = next((k for k in v if k.upper().startswith("ON")), None)
    off = next((k for k in v if k.upper().startswith("OFF")), None)
    if on is None or off is None:
        return None
    fp = lambda k: v[k]["scene"]["footprint"]["footprint_filled_m2"]   # noqa: E731
    return dict(method=on, ablation=dict(footprint_on_m2=fp(on), footprint_off_m2=fp(off), source=str(f)))


def run_drift(entries: list[PlanEntry], outputs: Path, cycles: list[dict]) -> list[GateResult]:
    out = []
    for e in entries:
        meta = e.plan.get("meta", {}) or {}
        g = drift_gate(meta, e.tier, e.capture, e.producer, cycles)
        ev = drift_evidence(outputs, e.scene)
        if g.status == "not_verifiable" and ev is not None:
            g.detail = (f"plan does not declare its drift handling (meta.drift); the drift module's ablation exists "
                        f"({ev['ablation']['source']}: footprint on {ev['ablation']['footprint_on_m2']:.2f} m2 vs off "
                        f"{ev['ablation']['footprint_off_m2']:.2f} m2) but we cannot tell whether this plan used it")
        out.append(g)
    return out


def repeat_pair_calibration(reports: list) -> list[dict]:
    """GT-free check of intervals: two independent captures of a wall should differ by <= 1.96 combined sigma for
    ~95% of walls. Tests the random part of the error only (a shared bias is invisible)."""
    out = []
    by = {}
    for producer, rep in reports:
        by.setdefault((producer, rep.tier), []).extend(w for w in rep.walls if w.z is not None)
    for (producer, tier), ws in sorted(by.items()):
        z = np.array([w.z for w in ws])
        sig = np.ones_like(z)
        c = interval_calibration(np.zeros_like(z), z, z - 1.96 * sig, z + 1.96 * sig) if len(z) else {"n": 0}
        keep = ("n", "coverage_used", "wilson_lo", "wilson_hi", "rms_z", "status")   # widths are in z units here
        out.append(dict(producer=producer, tier=tier, **{k: v for k, v in c.items() if k in keep}))
    return out


# ----------------------------------------------------------------------------------------------- report

def write_outputs(out: Path, entries, skipped, notes, reports, gates, calib, registry: Registry, cycles) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [g.row() for g in gates]
    with open(out / "gates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(GateResult("", "", "", "", "").row()))
        w.writeheader()
        w.writerows(rows)
    wall_csv = "".join(rep.walls_csv() if i == 0 else rep.walls_csv().split("\n", 1)[1]
                       for i, (_, rep) in enumerate(reports))
    (out / "repeatability_walls.csv").write_text(wall_csv)
    (out / "benchmark.json").write_text(json.dumps(dict(
        plans=[dict(producer=e.producer, capture=e.capture, scene=e.scene, tier=e.tier, path=str(e.path))
               for e in entries],
        skipped=skipped, notes=notes, repeatability=[dict(producer=p, **r.summary()) for p, r in reports],
        gates=[{**g.row(), "extra": g.extra} for g in gates], repeat_pair_calibration=calib,
        registration=[{k: v for k, v in d.items() if k not in ("yaw_curve", "local_tiles")}
                      for d in registry.info.values()], cycles=cycles), indent=2, default=str))
    (out / "report.md").write_text(render_markdown(entries, skipped, notes, reports, gates, calib, registry, cycles))


def render_markdown(entries, skipped, notes, reports, gates, calib, registry: Registry, cycles) -> str:
    L = ["# Benchmark report", "",
         "Generated by `scripts/benchmark.py`. No ground truth exists for the sample data: every row states its "
         "basis. `lidar_reference` means *compared with our own LiDAR-tier result*, which is NOT ground truth.", "",
         "## Inputs", ""]
    L += [f"- `{e.path.relative_to(ROOT) if e.path.is_relative_to(ROOT) else e.path}`: producer `{e.producer}`, "
          f"tier `{e.tier}`, scene `{e.scene}`" for e in entries] or ["- no plan.json found under outputs/"]
    L += [f"- skipped: {s}" for s in skipped]
    L += ["", "## Capture registration (scene to scene, independent of any plan)", "",
          "| B -> A | yaw (deg) | RMSE (cm) | inliers | floor IoU | distinct |", "|---|---|---|---|---|---|"]
    for (a, b), d in sorted(registry.info.items()):
        L.append(f"| {b} -> {a} | {d['yaw_deg']:.2f} | {d['residual_rmse_m'] * 100:.2f} | {d['inlier_frac_b']:.2f} | "
                 f"{d['overlap_iou']:.2f} | {d['distinctiveness']:.2f} |")
    for c in cycles:
        L.append(f"\nLoop check {' <- '.join(c['loop'])}: residual yaw {c['yaw_deg']:.2f} deg, max displacement "
                 f"{c['max_displacement_m'] * 100:.1f} cm (0 for rigid, drift-free captures).")
    L += ["", "## Gates", "", "| producer | tier | capture | gate | status | basis | value | threshold | n | detail |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g in sorted(gates, key=lambda g: (g.producer, g.tier, g.capture, g.gate)):
        v = "-" if g.value is None else f"{g.value:.3f}"
        L.append(f"| {g.producer} | {g.tier} | {g.capture} | {g.gate} | **{g.status}** | {g.basis} | {v} | "
                 f"{g.threshold} | {g.n} | {g.detail} |")
    fails = [g for g in gates if g.status == "fail"]
    L += ["", "## Failing gates (candidates for the fix loop)", ""]
    L += [f"- {g.producer}/{g.tier}/{g.capture}: **{g.gate}** = {g.value} ({g.threshold}); {g.detail}"
          for g in fails] or ["- none"]
    L += ["", "## Repeat-pair interval calibration (GT-free; tests the random error only)", "",
          "| producer | tier | n walls | coverage of 95% | Wilson CI | RMS z | status |",
          "|---|---|---|---|---|---|---|"]
    for c in calib:
        if c.get("n"):
            L.append(f"| {c['producer']} | {c['tier']} | {c['n']} | {c['coverage_used']:.2f} | "
                     f"[{c['wilson_lo']:.2f}, {c['wilson_hi']:.2f}] | {c['rms_z']:.2f} | {c['status']} |")
    L += ["", "## Repeatability details", ""]
    for producer, rep in reports:
        L += [f"Producer `{producer}`", "", rep.to_markdown()]
    if not reports:
        L.append("No repeatability pair could be evaluated (needs two plans of the same producer and tier for "
                 "two registered captures).")
    L += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    L += ["", "## Gate definitions", "", "| id | row | requirement | needs GT |", "|---|---|---|---|"]
    L += [f"| {g.id} | {g.row} | {g.requirement} | {g.needs_gt} |" for g in GATES]
    return "\n".join(L) + "\n"


def load_cycles(reg_dir: Path) -> list[dict]:
    f = reg_dir / "cycles.json"
    return json.loads(f.read_text()) if f.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--registration", type=Path, default=None, help="default: <outputs>/eval/registration")
    ap.add_argument("--gt-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: <outputs>/benchmark")
    ap.add_argument("--include-synthetic", action="store_true")
    args = ap.parse_args()
    reg_dir = args.registration or args.outputs / "eval" / "registration"
    out = args.out or args.outputs / "benchmark"
    entries, skipped = discover(args.outputs, args.include_synthetic)
    print(f"[benchmark] {len(entries)} plans, {len(skipped)} skipped")
    registry, cycles = Registry(reg_dir), load_cycles(reg_dir)
    reports, notes, gates = run_repeatability(entries, registry, print, out)
    ref_gates, ref_notes = run_reference(entries, args.gt_dir, print)
    gates += ref_gates + run_drift(entries, args.outputs, cycles)
    calib = repeat_pair_calibration(reports)
    write_outputs(out, entries, skipped, notes + ref_notes, reports, gates, calib, registry, cycles)
    n = {s: sum(g.status == s for g in gates) for s in ("pass", "fail", "not_verifiable")}
    print(f"[benchmark] gates: {n}; report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
