"""Scenario admission, leakage and hand-computed L1 aggregation checks."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from envs.inspection.protocol import Evidence, Observation, ReportLedger
from eval.inspection_cases import CaseSpec, coverage_cells, generate_dev_case, validate_case
from eval.inspection_scenarios import ScenarioMission


@pytest.fixture(scope="module")
def cases():
    return {level: generate_dev_case("fixture-"+level, level, 783+index)
            for index, level in enumerate(("L1", "L2", "L3"))}


def test_reproducible_roundtrip_and_independent_roots(cases):
    for index, (level, case) in enumerate(cases.items()):
        assert CaseSpec.from_dict(json.loads(json.dumps(case.as_dict()))) == case
        assert generate_dev_case(case.case_id, level, 783+index) == case
        assert validate_case(case)["valid"]
    assert len({c.digest() for c in cases.values()}) == 3
    assert len({c.roi_polygon for c in cases.values()}) == 3
    assert not any(e.object_type == "dynamic" for e in cases["L1"].entities)


@pytest.mark.parametrize("fault", ["nan", "crossed_roi", "duplicate", "wrong_rule", "outside",
                                   "l1_dynamic", "late_dynamic", "event_level", "wind", "wrong_split"])
def test_invalid_cases_are_rejected(cases, fault):
    case = cases["L3"]
    if fault == "nan":
        case = replace(case, wind_xy=(float("nan"), 0))
    elif fault == "crossed_roi":
        p = list(case.roi_polygon)
        p[1], p[5] = p[5], p[1]
        case = replace(case, roi_polygon=tuple(p))
    elif fault == "duplicate":
        case = replace(case, entities=case.entities+(case.entities[0],))
    elif fault == "wrong_rule":
        case = replace(case, entities=(replace(case.entities[0], rule_id="no-rule"),)+case.entities[1:])
    elif fault == "outside":
        case = replace(case, entities=(replace(case.entities[0], origin_xy=(500., 500.)),)+case.entities[1:])
    elif fault == "l1_dynamic":
        case = replace(case, level="L1", priority_at=None, link_drone=None, link_start=None, link_end=None,
                       camera_drone=None, camera_start=None, camera_end=None, retire_drone=None, retire_at=None)
    elif fault == "late_dynamic":
        entities = tuple(replace(e, appeared_at=500) if e.object_type == "dynamic" else e for e in case.entities)
        case = replace(case, entities=entities)
    elif fault == "event_level":
        case = replace(case, level="L2")
    elif fault == "wind":
        case = replace(case, wind_xy=(2., 2.))
    else:
        case = replace(case, split="main")
    with pytest.raises(ValueError):
        validate_case(case)


def test_grid_uses_entire_polygon_not_legacy_bounds():
    polygon = ((-420, -10), (430, -10), (430, 30), (-420, 30))
    cells = coverage_cells(polygon)
    assert "cell-410-10" in cells
    assert "cell--410-10" in cells


def test_l1_hand_computed_score_and_safety_gates(cases):
    case = cases["L1"]
    mission = ScenarioMission.__new__(ScenarioMission)
    mission.case, mission.entities = case, list(case.entities)
    mission.cells = {"a": (0, 0), "b": (1, 0), "c": (2, 0), "d": (3, 0)}
    mission.covered = {"a", "b", "c", "d"}
    mission.flight = SimpleNamespace(violations=[], exit_dwell=np.array([20.]*4))
    mission.distance_m = np.array([10000.]*4)
    mission.card = {"range_budget_m_per_aircraft": 18000., "exit_range_reserve_m": 1000.}
    mission.ledger = ReportLedger(10, set(e.true_class for e in case.entities), mission.truth_at)
    targets = [e for e in case.entities if e.rule_id]
    for index, e in enumerate(targets):
        obs = Observation(f"o{index}", f"c{index}", 0, 1., 2., e.origin_xy, e.true_class,
                          "static", 1., (1., 1.), "inspect", (0., 0.))
        mission.ledger.register(Evidence(obs, e.key))
        assert mission.ledger.submit(mission.report_request(obs, f"r{index}"), 2)["status"] == "accepted"
    score = mission.grade()
    assert (score["C"], score["S"], score["Q"], score["Y"], score["SR"]) == (1., 1., 1., 1., 1)
    assert score["D"] is None and score["dynamic_denominator_weight"] == 0
    mission.covered = {"a", "b"}
    assert mission.grade()["Q"] == .75 and mission.grade()["SR"] == 0
    mission.flight.violations = [{"hard": True}]
    assert mission.grade()["Y"] == 0
    mission.flight.violations = []
    mission.distance_m[0] = 17500
    assert mission.grade()["E"] == 0


def test_calendar_boundaries_and_no_future_event_in_card(cases):
    mission = ScenarioMission.__new__(ScenarioMission)
    mission.case = c = cases["L3"]
    assert mission.online(c.link_drone, c.link_start-.001)
    assert not mission.online(c.link_drone, c.link_start)
    assert mission.online(c.link_drone, c.link_end)
    assert mission.camera_quality(c.camera_drone, c.camera_start-.001) == 1.
    assert mission.camera_quality(c.camera_drone, c.camera_start) == .55
    assert mission.camera_quality(c.camera_drone, c.camera_end) == 1.
