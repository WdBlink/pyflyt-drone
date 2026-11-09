"""Hand-computed offline aggregation fixtures, never physical task samples."""

from dataclasses import replace

import pytest

from envs.inspection.protocol import Evidence, Observation, ReportLedger
from envs.inspection.scenario import CLASSES
from eval.inspection_cases import coverage_cells, generate_dev_case
from eval.inspection_common_calendar import _score, rescore_common_calendar


def fixture():
    event = generate_dev_case("calendar-fixture", "L3", 321)
    clean = replace(event, level="L2", priority_at=None, link_drone=None, link_start=None, link_end=None,
                    camera_drone=None, camera_start=None, camera_end=None, retire_drone=None, retire_at=None)
    def truth_at(stamp):
        return {e.key: e.position(stamp) for e in clean.entities if e.position(stamp) is not None}
    ledger, observations, sensor = ReportLedger(10, set(CLASSES), truth_at), [], []
    for index, entity in enumerate(e for e in clean.entities if e.object_type == "static" and e.rule_id):
        claim = "normal" if entity.rule_id == "static-0" else entity.true_class
        obs = Observation(f"e{index}", f"c{index}", 0, 1., 2., entity.origin_xy, claim,
                          "static", 1., (1., 1.), "inspect", (0., 0.))
        ledger.register(Evidence(obs, entity.key))
        ledger.submit({"report_id": f"r{index}", "object_type": "static", "candidate_id": obs.candidate_id,
                       "measured_at": 1., "estimate_xy": entity.origin_xy, "claimed_class": claim,
                       "evidence_ids": [obs.evidence_id]}, 2.)
        observations.append(obs.public())
        sensor.append({"evidence_id": obs.evidence_id, "entity_key": entity.key})
    cells = coverage_cells(clean.roi_polygon)
    world = {"private_case": clean.as_dict(), "case_sha256": clean.digest(), "reports": ledger.audit,
             "sensor_audit": sensor, "delivered_observations": observations, "required_cells": cells,
             "coverage_evidence": [{"arrived_at": 2., "cells": list(cells)}],
             "safety": {"violations": {}, "exit_dwell_seconds": [20.]*4},
             "score": {"remaining_range_m": [8000.]*4}}
    world["score"] = _score(world, clean, clean)
    return event, clean, world


def test_common_calendar_changes_weights_without_changing_reports_or_actor_history():
    event, clean, world = fixture()
    result = rescore_common_calendar(world, event)
    total = sum(e.object_type == "static" and bool(e.rule_id) for e in clean.entities)
    wrong = sum(e.rule_id == "static-0" for e in clean.entities)
    assert world["score"]["S"] == pytest.approx((total-wrong)/total)
    assert result["score"]["S"] == pytest.approx((total-wrong)/(total+2*wrong))
    assert result["score"]["Y"] == pytest.approx((1+(total-wrong)/(total+2*wrong))/3)
    assert world["private_case"] == clean.as_dict()
    assert result["original_score_exact"] is True


@pytest.mark.parametrize("fault", ["calendar", "schema", "geometry"])
def test_invalid_calendar_or_different_root_cannot_be_rescored(fault):
    event, _, world = fixture()
    event = replace(event, **{"calendar": {"priority_at": 0}, "schema": {"schema": "bad"},
                             "geometry": {"wind_xy": (0., 0.)}}[fault])
    with pytest.raises(ValueError):
        rescore_common_calendar(world, event)
