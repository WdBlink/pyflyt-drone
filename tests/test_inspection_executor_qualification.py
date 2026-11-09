"""Qualification admission guards; fixture records are not experiment evidence."""

import copy
import base64
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import zlib

import numpy as np
import pytest

from envs.inspection.control import inspection_gains
from eval import qualify_inspection_executor as qualification
from eval.inspection_tracking import COLUMNS
from envs.inspection.planning import OwnState


CONTRACT=Path(__file__).resolve().parents[1]/"paper/research/20261104-inspection/planning/gate-contract.v18.json"


def test_stricter_contract_cannot_be_ignored_by_current_runtime():
    contract=json.loads(CONTRACT.read_text())
    qualification.validate_contract(contract)
    bad=copy.deepcopy(contract)
    bad["executor_qualification"]["xy_tracking_tube_m"]=1
    with pytest.raises(ValueError):qualification.validate_contract(bad)
    bad=copy.deepcopy(contract)
    bad["G1"]["control_gains"]["pitch_p"]=1.5
    with pytest.raises(ValueError):qualification.validate_contract(bad)


def test_zero_tracking_or_truncated_physics_is_not_a_pass():
    contract=json.loads(CONTRACT.read_text())
    record={"family":"observe","simulated_seconds":360.,"actual_control_profile":asdict(inspection_gains()),
            "safety":{"physics_steps":86400,"physics_audits":86400,"violations":{},"exit_dwell_seconds":[25.]*4},
            "execution_tracking_240hz":{"physics_samples":86400,"max_horizontal_error_m":0.,
                "horizontal_tube_breach_steps":0,"altitude_envelope_breach_steps":0,"seam_capture_failures":0},
            "execution_infrastructure_errors":[],"remaining_range_m":[5000.]*4,"operational_error":None,
            "jobs":[{"kind":"observe","status":"succeeded"}],"single_target_delivered_D":.9,
            "execution_audit":[]}
    state=asdict(OwnState(0.,(0.,0.,96.),(0.,0.,0.),(0.,0.,0.),(17.,0.,0.),17.,(0.,0.)))
    record["execution_audit"]=[{"type":"plan_request","now":0.,"own_states":[state]*4,"remaining_range_m":[18000.]*4},
        {"type":"program_committed","at":0.,"own_state":state,"capture_metrics":{}}]
    for i in range(4):
        record["execution_audit"].append({"type":"physical_program_seam","at":0.,"own_state":state,
            "capture_valid":True,"label":"tail_start","drone":i})
    rows=np.zeros((86400,17));rows[:,0]=np.arange(1,86401)/240
    rows[:,2::4]=96;rows[:,3::4]=94;rows[:,4::4]=98
    raw=rows.astype('<f8').tobytes()
    record["execution_tracking_trace"]={"encoding":"base64-zlib-little-endian-float64","columns":COLUMNS,"xy_tube_m":10.,
        "rows":86400,"raw_sha256":hashlib.sha256(raw).hexdigest(),"data":base64.b64encode(zlib.compress(raw)).decode()}
    assert qualification.case_passed(record,contract)
    bad=copy.deepcopy(record);bad["execution_tracking_240hz"]["physics_samples"]=0
    assert not qualification.case_passed(bad,contract)
    bad=copy.deepcopy(record);bad["safety"]["physics_steps"]=bad["safety"]["physics_audits"]=86398
    assert not qualification.case_passed(bad,contract)
    bad=copy.deepcopy(record);bad["safety"]["exit_dwell_seconds"]=[]
    assert not qualification.case_passed(bad,contract)
    bad=copy.deepcopy(record);bad["execution_tracking_trace"]["raw_sha256"]="0"*64
    assert not qualification.case_passed(bad,contract)
    bad=copy.deepcopy(record);bad["execution_audit"][0]["own_states"]=[]
    assert not qualification.case_passed(bad,contract)
    bad=copy.deepcopy(record);bad["execution_audit"].pop()
    assert not qualification.case_passed(bad,contract)


def test_fixture_setup_failure_closes_the_actual_world(monkeypatch):
    original=qualification.mission_module.InspectionMission
    closed=[]
    def create(*args,**kwargs):
        mission=original(*args,**kwargs)
        close=mission.close
        def tracked_close():
            closed.append(True)
            close()
        def fail_reset(*args,**kwargs):
            raise RuntimeError("injected fixture setup failure")
        monkeypatch.setattr(mission,"close",tracked_close)
        monkeypatch.setattr(mission.flight.world,"resetBasePositionAndOrientation",fail_reset)
        return mission
    monkeypatch.setattr(qualification.mission_module,"InspectionMission",create)
    result=qualification.run_case("observe",800,json.loads(CONTRACT.read_text()))
    assert not result["passed"]
    assert "injected fixture setup failure" in result["operational_error"]
    assert closed==[True]
    assert result["safety"]["physics_steps"]==0
