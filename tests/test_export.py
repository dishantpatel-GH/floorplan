"""Tests for floorplan.export: schema conformance, round trip, validation catches real mistakes, renderers run.

Why these tests: the export layer is the contract with everything downstream. The tests pin the promises we make
in the docs: (1) the fixture is valid, (2) JSON -> Plan -> JSON is lossless up to 0.1 mm, (3) the validator
rejects the mistakes a pipeline actually makes (number with no interval, value outside its interval, dangling id,
an invented value on a not-observed measurement), (4) output is deterministic, (5) renderers survive awkward
plans (rotated, many tiny rooms, missing values) and the DXF is CAD-native (metres, DIMENSION entities).
Run: PYTHONPATH=. .venv/bin/python -m pytest tests/test_export.py -q
"""
import copy
import json
import sys
from pathlib import Path

import ezdxf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from make_synthetic_plan import stress_plan, synthetic_plan  # noqa: E402

from floorplan.export.dxf import export_dxf  # noqa: E402
from floorplan.export.json_export import (dumps, measurement_to_json, plan_from_json, plan_to_json,  # noqa: E402
                                         validate_plan_json)
from floorplan.export.render import render_plan  # noqa: E402
from floorplan.model import Measurement  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_plan.json"


@pytest.fixture()
def doc() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_is_valid_and_up_to_date(doc):
    assert validate_plan_json(doc) == []
    assert dumps(plan_to_json(synthetic_plan())) == FIXTURE.read_text(), "regenerate the fixture"


def test_round_trip_is_lossless(doc):
    assert plan_to_json(plan_from_json(doc)) == doc


def test_every_measurement_has_interval_or_is_not_observed(doc):
    wall = doc["walls"][0]
    assert wall["length"]["ci95"][0] <= wall["length"]["value"] <= wall["length"]["ci95"][1]
    wc = next(r for r in doc["rooms"] if r["id"] == "wc")
    assert wc["ceiling_height"] == {**wc["ceiling_height"], "value": None, "ci95": None, "status": "not_observed"}


@pytest.mark.parametrize("break_it, expected", [
    (lambda d: d["walls"][0]["length"].update(ci95=None), "ci95"),                   # number, no interval
    (lambda d: d["walls"][0]["length"].update(value=99.0), "outside its interval"),
    (lambda d: d["openings"][0]["wall_ids"].append("ghost"), "unknown id 'ghost'"),
    (lambda d: d["rooms"][-1]["ceiling_height"].update(value=2.5), "schema"),        # invented number on n/o
    (lambda d: d["openings"][0].update(kind="hatch"), "schema"),
    (lambda d: d.pop("tier"), "schema"),
    (lambda d: d["damage"].append({"id": "x", "class": "mould", "surface_id": "s", "concealed": True}), "rule"),
])
def test_validator_catches_mistakes(doc, break_it, expected):
    bad = copy.deepcopy(doc)
    break_it(bad)
    errors = validate_plan_json(bad)
    assert errors and any(expected in e for e in errors), errors


def test_missing_value_is_published_as_not_observed():
    out = measurement_to_json(Measurement(None, 1.0, 2.0, status="measured"))
    assert out["status"] == "not_observed" and out["ci95"] is None


def test_render_is_deterministic_and_robust(tmp_path):
    for name, plan in (("fixture", synthetic_plan()), ("stress", stress_plan(seed=1, n=5, angle_deg=17))):
        s1 = render_plan(plan, tmp_path / f"{name}_a")
        render_plan(plan, tmp_path / f"{name}_b")
        assert (tmp_path / f"{name}_a.svg").read_bytes() == (tmp_path / f"{name}_b.svg").read_bytes()
        assert (tmp_path / f"{name}_a.png").stat().st_size > 10_000
        assert s1.dims_drawn + s1.dims_skipped_short == len(plan.walls)
        assert s1.openings_unplaced == 0


def test_render_survives_empty_and_degenerate_plans(tmp_path):
    plan = synthetic_plan()
    plan.rooms[0].polygon = [(0, 0), (1, 1)]            # degenerate polygon
    plan.openings[0].width = Measurement.missing()
    stats = render_plan(plan, tmp_path / "degenerate")
    assert any("invalid polygon" in w for w in stats.warnings)
    empty = synthetic_plan()
    empty.rooms, empty.walls, empty.openings, empty.adjacency, empty.damage = [], [], [], [], []
    render_plan(empty, tmp_path / "empty")


def test_dxf_is_cad_native(tmp_path):
    summary = export_dxf(synthetic_plan(), tmp_path / "plan.dxf")
    doc = ezdxf.readfile(tmp_path / "plan.dxf")
    assert doc.header["$INSUNITS"] == 6 and summary["audit_errors"] == 0
    dims = doc.modelspace().query("DIMENSION")
    assert len(dims) == summary["dimensions"] > 0
    assert {"WALLS", "ROOMS", "DIMENSIONS", "OPENINGS", "TEXT"} <= {layer.dxf.name for layer in doc.layers}
    assert all(d.dxf.layer == "DIMENSIONS" and "±" in d.dxf.text for d in dims)
