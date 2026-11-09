"""Literal renamed fixtures for offline score binding; no formal root draw."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest

from envs.inspection.protocol import Evidence, Observation, ReportLedger
from envs.inspection.scenario import CLASSES
from eval.inspection_cases import coverage_cells

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("formal_calendar_helpers", HERE/"test_formal_secondary.py")
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
module = helper.load("formal_common_calendar_candidate", HERE.parent/"eval/inspection_common_calendar.py")


@pytest.fixture(autouse=True)
def use_matching_staged_types(monkeypatch):
    monkeypatch.setitem(sys.modules, "eval.inspection_population", helper.shared)
    monkeypatch.setitem(sys.modules, "eval.inspection_formal_intervention", helper.intervention)


def sealed_world(case):
    def truth_at(stamp):
        return {e.key: e.position(stamp) for e in case.entities if e.position(stamp) is not None}
    ledger = ReportLedger(10, set(CLASSES), truth_at)
    observations, sensor = [], []
    for index, entity in enumerate(e for e in case.entities if e.object_type == "static" and e.rule_id):
        claim = "normal" if entity.rule_id == "static-0" else entity.true_class
        obs = Observation(f"e{index}", f"c{index}", 0, 1., 2., entity.origin_xy, claim,
                          "static", 1., (1., 1.), "inspect", (0., 0.))
        ledger.register(Evidence(obs, entity.key))
        ledger.submit({"report_id": f"r{index}", "object_type": "static", "candidate_id": obs.candidate_id,
            "measured_at": 1., "estimate_xy": entity.origin_xy, "claimed_class": claim,
            "evidence_ids": [obs.evidence_id]}, 2.)
        observations.append(obs.public())
        sensor.append({"evidence_id": obs.evidence_id, "entity_key": entity.key})
    cells = coverage_cells(case.roi_polygon)
    world = {"private_case": case.as_dict(), "case_sha256": case.digest(), "reports": ledger.audit,
        "hidden_entities": case.as_dict()["entities"], "sensor_audit": sensor,
        "delivered_observations": observations, "required_cells": cells,
        "coverage_evidence": [{"arrived_at": 2., "cells": list(cells)}],
        "safety": {"violations": {}, "exit_dwell_seconds": [20.]*4},
        "score": {"remaining_range_m": [8000.]*4}}
    world["score"] = module._score(world, case, case)
    return world


@pytest.mark.parametrize("clean", [False, True])
def test_formal_calendar_uses_original_parent_and_preserves_sealed_history(clean):
    parent = helper.literal_parent()
    case = helper.intervention.formal_clean_sibling(parent) if clean else parent
    world = sealed_world(case)
    before = deepcopy(world)
    result = module.rescore_formal_common_calendar(world, parent, parent.digest())
    total = sum(e.object_type == "static" and bool(e.rule_id) for e in case.entities)
    wrong = sum(e.rule_id == "static-0" for e in case.entities)
    assert world["score"]["S"] == pytest.approx((total-wrong)/(total if clean else total+2*wrong))
    assert result["score"]["S"] == pytest.approx((total-wrong)/(total+2*wrong))
    assert result["score"]["Y"] == pytest.approx((1+(total-wrong)/(total+2*wrong))/3)
    assert result["original_score_exact"] is True and world == before


@pytest.mark.parametrize("fault", ["parent", "score", "level", "seed", "identity", "hidden"])
def test_formal_calendar_rejects_changed_parent_or_structurally_valid_wrong_score(fault):
    parent = helper.literal_parent()
    world = sealed_world(helper.intervention.formal_clean_sibling(parent))
    admitted = parent.digest()
    if fault == "parent":
        admitted = "0"*64
    elif fault == "score":
        world["score"]["Y"] = .123
    elif fault == "level":
        parent = replace(parent, level="L2")
    elif fault == "seed":
        parent = replace(parent, seed=parent.seed+1)
    elif fault == "identity":
        world["case_sha256"] = "0"*64
    else:
        world["hidden_entities"] = []
    with pytest.raises(ValueError):
        module.rescore_formal_common_calendar(world, parent, admitted)
