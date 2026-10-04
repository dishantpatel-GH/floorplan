"""One-parameter experiments on the repeat pair: does the defect the judge suspects actually move the gate?

    python scripts/judge_experiments.py drift_on_jog15 drift_on_band60 drift_on_notilt

Each argument is a run folder written by `judge_run.py --variant drift_on --set KEY=VALUE --tag <tag>` for both
captures of the pair. The pair is scored exactly like the default runs (scripts/judge_repeatability.py), so the
numbers are directly comparable with the 'drift_on' row. Writes outputs/judge/experiments.json.
These runs only probe the extractors through their public parameters; no extractor code is changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_common import OUT, PAIR, PRODUCERS, compare, jsonable, load_run  # noqa: E402
from judge_repeatability import registration  # noqa: E402

KEYS = ("rooms_a", "rooms_b", "rooms_matched", "walls_pass", "walls_fail", "walls_unmatched", "wall_pass_rate",
        "wall_delta_median_m", "walls_within_2cm", "same_topology", "different_topology", "wall_interval_coverage")


def main() -> None:
    T, _ = registration("drift_on")
    out = {}
    for folder in ["drift_on", *sys.argv[1:]]:
        for producer in PRODUCERS:
            A, B = load_run(folder, producer, PAIR[0]), load_run(folder, producer, PAIR[1])
            if A is None or B is None:
                continue
            s = compare(A, B, T, PAIR[0], PAIR[1])["summary"]
            out[f"{folder}/{producer}"] = {k: s[k] for k in KEYS}
            print(folder, producer, out[f"{folder}/{producer}"])
    (OUT / "experiments.json").write_text(json.dumps(jsonable(out), indent=1, default=float))


if __name__ == "__main__":
    main()
