"""plan_beta remove_jogs: a snap that gives back the same outline ends the loop (D-088 follow-up).

The stitched k38 scenes gave an invalid room outline with a zero-width slit and a doubled corner; the snap of the
zero-length step between two same-direction walls moves nothing, simplify keeps the doubled corner, and the old loop
ran forever. Run: env -u PYTHONPATH .venv/bin/python -m pytest tests/test_remove_jogs.py -q
"""
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapely.geometry import Polygon  # noqa: E402

from floorplan.plan.beta.params import BetaParams  # noqa: E402
from floorplan.plan.beta.walls import remove_jogs  # noqa: E402


@pytest.fixture
def time_limit():
    def _boom(signum, frame):
        raise TimeoutError("remove_jogs did not return")
    old = signal.signal(signal.SIGALRM, _boom)
    signal.alarm(10)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)


def test_snap_that_returns_the_same_outline_terminates(time_limit):
    # slit along v = 2 (out to u = 1 and back to u = 1.5) with a doubled corner at (2, 2)
    slit = Polygon([(0, 0), (4, 0), (4, 2), (2, 2), (2, 2), (1, 2), (1.5, 2), (1.5, 4), (0, 4)])
    assert not slit.is_valid
    out = remove_jogs(slit, BetaParams())
    assert sorted(out.exterior.coords[:-1]) == sorted(slit.exterior.coords[:-1])


def test_short_step_is_still_removed(time_limit):
    p = BetaParams()
    step = p.jog_max_m / 2
    jog = Polygon([(0, 0), (4, 0), (4, 1.5), (4 + step, 1.5), (4 + step, 3), (0, 3)])
    out = remove_jogs(jog, p)
    assert len(out.exterior.coords) - 1 == 4
    assert out.is_valid
