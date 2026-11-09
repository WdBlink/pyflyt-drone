"""Reconstruct sealed reports and rescore event siblings on one calendar.

Offline evaluator-only analysis. Callers must separately verify run/source/
receipt bindings; this module never supplies information to an actor.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from envs.inspection.protocol import Evidence, Observation, ReportLedger
from envs.inspection.scenario import CLASSES
from eval.inspection_population import load_development_case, validate_development_case
from eval.inspection_scenarios import ScenarioMission


def reconstruct_reports(world, case):
    hidden = {r["evidence_id"]: r["entity_key"] for r in world["sensor_audit"]}
    rows = world["delivered_observations"]
    if len({r["evidence_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate delivered evidence identity")
    observations = sorted((Observation(**r) for r in rows), key=lambda o: (o.arrived_at, o.evidence_id))
    def truth_at(stamp):
        return {e.key: point for e in case.entities if (point := e.position(stamp)) is not None}
    ledger = ReportLedger(10, set(CLASSES), truth_at)
    index, previous = 0, -math.inf
    for entry in world["reports"]:
        stamp = entry["t"]
        if type(stamp) not in (int, float) or not previous <= stamp <= 600:
            raise ValueError("report journal is not a causal monotonic history")
        previous = stamp
        while index < len(observations) and observations[index].arrived_at <= stamp:
            obs = observations[index]
            ledger.register(Evidence(obs, hidden[obs.evidence_id]))
            index += 1
        if ledger.submit(entry["request"], stamp) != entry["response"]:
            raise ValueError("sealed report reconstruction differs from original response")
    return ledger


def _score(world, geometry, demand_calendar):
    mission = ScenarioMission.__new__(ScenarioMission)
    mission.case, mission.entities = demand_calendar, list(geometry.entities)
    mission.ledger = reconstruct_reports(world, geometry)
    mission.cells = world["required_cells"]
    mission.covered = {cell for packet in world["coverage_evidence"] if packet["arrived_at"] <= 600
                       for cell in packet["cells"]}
    if not mission.covered <= set(mission.cells):
        raise ValueError("coverage contains cells outside the frozen demand")
    mission.card = {"range_budget_m_per_aircraft": 18000., "exit_range_reserve_m": 1000.}
    remaining = np.asarray(world["score"]["remaining_range_m"], dtype=float)
    if remaining.shape != (4,) or not np.isfinite(remaining).all():
        raise ValueError("four actual remaining-range measurements are required")
    mission.distance_m = 18000. - remaining
    mission.flight = SimpleNamespace(violations=world["safety"]["violations"],
        exit_dwell=np.asarray(world["safety"]["exit_dwell_seconds"], dtype=float))
    return mission.grade()


def rescore_common_calendar(world, demand_calendar):
    validate_development_case(demand_calendar)
    geometry = load_development_case(world["private_case"])
    if geometry.digest() != world["case_sha256"]:
        raise ValueError("stored scenario identity differs")
    fields = ("case_id", "lineage_group", "split", "seed", "roi_polygon", "wind_xy", "entities")
    if any(getattr(geometry, k) != getattr(demand_calendar, k) for k in fields):
        raise ValueError("event siblings must have identical underlying root geometry, motion and noise seed")
    if geometry.level == "L1" or demand_calendar.level == "L1":
        if geometry.level != demand_calendar.level:
            raise ValueError("a different demand level is not an event sibling")
    original = _score(world, geometry, geometry)
    if original != world["score"]:
        raise ValueError("offline original-score reconstruction differs; do not rescore")
    rescored = _score(world, geometry, demand_calendar)
    return {"role": "offline_common_demand_calendar_not_a_new_rollout_or_primary_score",
            "geometry_case_sha256": geometry.digest(), "demand_case_sha256": demand_calendar.digest(),
            "original_score_exact": True, "score": rescored,
            "range_scope": "uses original 240Hz-hook remaining-range result; no invented full-state odometry replay"}


def rescore_formal_common_calendar(world,demand_calendar,admitted_parent_sha256):
    """Offline content/score audit after upstream formal receipt admission.

    The caller obtains parent identity from its fresh dataset/dispatch proof.
    This function neither admits that proof nor supplies data to an actor.
    """
    from eval.inspection_formal_intervention import CLEAN_SCHEMA, SECONDARY_ROOTS, check_clean_world_identity
    from eval.inspection_population import decode_formal_population_case, validate_formal_population_case
    validate_formal_population_case(demand_calendar,'main',demand_calendar.case_id)
    if (demand_calendar.case_id not in SECONDARY_ROOTS or demand_calendar.level!='L3'
            or demand_calendar.digest()!=admitted_parent_sha256):
        raise ValueError('the admitted original L3 parent demand calendar is required')
    if world['private_case'].get('schema')==CLEAN_SCHEMA:
        geometry=check_clean_world_identity(world,admitted_parent_sha256)
    else:
        geometry=decode_formal_population_case(world['private_case'],'main',demand_calendar.case_id)
        if geometry.digest()!=world['case_sha256'] or geometry.digest()!=admitted_parent_sha256:
            raise ValueError('formal event world differs from its admitted original parent')
    fields=('case_id','lineage_group','split','level','seed','domain','roi_polygon','wind_xy','entities')
    if any(getattr(geometry,key)!=getattr(demand_calendar,key) for key in fields):
        raise ValueError('event and clean worlds must retain the same original geometry/motion/noise/level')
    original=_score(world,geometry,geometry)
    if original!=world['score']:raise ValueError('original formal score recount differs; do not reprice')
    rescored=_score(world,geometry,demand_calendar)
    return {'role':'offline_formal_common_demand_calendar_not_new_rollout_or_primary_score',
        'parent_case_sha256':admitted_parent_sha256,'geometry_case_sha256':geometry.digest(),
        'demand_case_sha256':demand_calendar.digest(),'original_score_exact':True,'score':rescored,
        'range_scope':'original240Hz-hook remaining range retained; no invented full-state odometry'}
