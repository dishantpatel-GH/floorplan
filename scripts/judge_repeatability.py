"""Judge criterion 1: repeatability of each plan extractor, scored with the shared harness (floorplan/eval).

    python scripts/judge_repeatability.py        (after scripts/judge_run.py has produced outputs/judge/runs/)

Three experiments, all with the SAME evaluation code for both extractors:

1. Repeat pair. floor_only and with_ceiling are two captures of the same apartment. B (floor_only) is put into
   A's (with_ceiling) frame with the scene-to-scene registration written by scripts/register_scenes.py. That
   transform comes from the reconstructed geometry, never from either plan, so neither extractor can flatter
   itself. floorplan/eval/match.py then matches rooms (IoU), re-aligns each matched room locally (residual
   drift) and matches walls; floorplan/eval/repeatability.py applies the gate |delta| <= max(1 cm, 0.5% L).
   Done twice: on the ARKit-pose scenes ("raw") and on the drift module's corrected scenes ("drift_on").
2. Invariance. The same scene moved rigidly (a 1.3 x 0.7 cm sub-cell shift; a 90 deg turn) or re-sampled (a random
   90% of the points, "thin90") must give the same plan. Any difference is the extractor's own instability: a
   floor that repeatability can never beat.
3. Determinism. The judge's re-run against the authors' own committed plan of the same scene.

Writes outputs/judge/repeatability.json, repeatability_walls.csv and figures repeat_<variant>.png,
invariance.png.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from floorplan.eval.match import aligned_plan_b, as_plan_dict  # noqa: E402
from judge_common import (OUT, PAIR, PRODUCERS, ROOT, compare, jsonable, load_run,  # noqa: E402
                          plan_T_original_to_moved)

REG = {"raw": ROOT / "outputs/eval/registration" / f"{PAIR[0]}__{PAIR[1]}.json",
       "drift_on": ROOT / "outputs/eval/registration_drift_on" / f"{PAIR[0]}@scene_on__{PAIR[1]}@scene_on.json"}
PERTURB = {"shift": ((0.013, 0.007), 0), "rot90": ((0.0, 0.0), 1), "thin90": ((0.0, 0.0), 0)}
OWN_OUTPUT = {"alpha": ROOT / "outputs/plan_alpha", "beta": ROOT / "outputs/plan_beta"}
COLOURS = {"pass": "#2a9d8f", "fail": "#e76f51", "unmatched": "#8d5fd3", "not_verifiable": "#999999"}


def _nan(x: float | None) -> float:
    return np.nan if x is None else x


def registration(variant: str) -> tuple[np.ndarray, dict]:
    d = json.loads(REG[variant].read_text())
    keep = ("yaw_deg", "residual_rmse_m", "inlier_frac_b", "overlap_iou", "distinctiveness", "tiles_summary")
    return np.array(d["T2"]), {k: d[k] for k in keep if k in d}


def plot_overlay(ax, A: dict, B: dict, res: dict, T: np.ndarray, title: str) -> None:
    """A's walls thick grey; B's walls (after global + per-room alignment) coloured by gate status."""
    Bal = aligned_plan_b(B, T, res["match"])
    status = {w.wall_b: w.status for w in res["report"].walls if w.wall_b}
    for w in A["walls"]:
        ax.plot(*np.array([w["p0"], w["p1"]]).T, color="0.55", lw=4, alpha=0.45, solid_capstyle="butt")
    matched_rooms = {r.b for r in res["match"].rooms}
    for w in Bal["walls"]:
        c = COLOURS[status.get(w["id"], "unmatched")] if w["room_id"] in matched_rooms else "#cccccc"
        ax.plot(*np.array([w["p0"], w["p1"]]).T, color=c, lw=1.4)
    s = res["summary"]
    ax.set_title(f"{title}\n{s['walls_pass']} pass / {s['walls_fail']} fail / {s['walls_unmatched']} unmatched; "
                 f"median |d| {100 * _nan(s['wall_delta_median_m']):.1f} cm", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def repeat_pair() -> tuple[dict, list[dict]]:
    out, rows = {}, []
    for variant in ("raw", "drift_on"):
        T, reg = registration(variant)
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        for ax, producer in zip(axes, PRODUCERS):
            A, B = load_run(variant, producer, PAIR[0]), load_run(variant, producer, PAIR[1])
            if A is None or B is None:
                out[f"{variant}/{producer}"] = dict(error="missing plan")
                continue
            res = compare(A, B, T, PAIR[0], PAIR[1])
            out[f"{variant}/{producer}"] = dict(registration=reg, **res["summary"])
            rows += [dict(variant=variant, producer=producer, **asdict(w)) for w in res["report"].walls]
            plot_overlay(ax, A, B, res, T, f"{producer} ({variant}): floor_only walls on with_ceiling (grey)")
        for k, c in COLOURS.items():
            axes[1].plot([], [], color=c, label=k)
        axes[1].legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"repeat_{variant}.png", dpi=90)
        plt.close(fig)
    return out, rows


def invariance() -> dict:
    out = {}
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    k = 0
    for producer in PRODUCERS:
        for capture in PAIR:
            base = load_run("raw", producer, capture)
            for tag, (shift, quarter) in PERTURB.items():
                moved = load_run(f"raw_{tag}", producer, capture)
                if base is None or moved is None:
                    out[f"{producer}/{capture}/{tag}"] = dict(error="missing plan")
                    continue
                T_ba = np.linalg.inv(plan_T_original_to_moved(quarter, shift))
                res = compare(base, moved, T_ba, capture, f"{capture}+{tag}")
                out[f"{producer}/{capture}/{tag}"] = res["summary"]
                if capture == PAIR[0]:
                    plot_overlay(axes.flat[k], base, moved, res, T_ba, f"{producer} {tag}: with_ceiling vs itself")
                    k += 1
    fig.tight_layout()
    fig.savefig(OUT / "invariance.png", dpi=80)
    plt.close(fig)
    return out


def determinism() -> dict:
    """Judge's re-run vs the authors' committed plan of the same raw scene (identity transform)."""
    out = {}
    for producer in PRODUCERS:
        for capture in PAIR:
            own = OWN_OUTPUT[producer] / capture / "plan.json"
            mine = load_run("raw", producer, capture)
            if not own.exists() or mine is None:
                continue
            res = compare(as_plan_dict(own), mine, np.eye(3))
            s = res["summary"]
            out[f"{producer}/{capture}"] = dict(rooms=(s["rooms_a"], s["rooms_b"]), walls_pass=s["walls_pass"],
                                                walls_fail=s["walls_fail"], walls_unmatched=s["walls_unmatched"],
                                                max_delta_m=max((abs(w.delta_m) for w in res["report"].walls
                                                                 if w.delta_m is not None), default=None))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pair, rows = repeat_pair()
    result = dict(repeat_pair=pair, invariance=invariance(), determinism=determinism())
    (OUT / "repeatability.json").write_text(json.dumps(jsonable(result), indent=1, default=float))
    if rows:
        with open(OUT / "repeatability_walls.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    keys = ("rooms_a", "rooms_b", "rooms_matched", "walls_matched", "walls_pass", "walls_fail", "walls_unmatched",
            "wall_pass_rate", "wall_match_rate", "wall_delta_median_m", "walls_within_2cm", "walls_within_5cm",
            "long_walls_matched", "long_walls_pass", "length_weighted_pass", "wall_interval_coverage",
            "openings_matched")
    for section in ("repeat_pair", "invariance"):
        for name, s in result[section].items():
            print(section, name, {k: (round(s[k], 4) if isinstance(s.get(k), float) else s.get(k)) for k in keys})
    for name, s in result["determinism"].items():
        print("determinism", name, s)


if __name__ == "__main__":
    main()
