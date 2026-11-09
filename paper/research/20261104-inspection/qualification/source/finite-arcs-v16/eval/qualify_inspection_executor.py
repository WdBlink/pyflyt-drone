"""Prospective control-mode qualification, not an experiment on model latency.

This harness executes the SAME job/planner/guidance code with a controlled
next-control-tick planner transport. Pure planning computation happens between
physics steps. Only G1 control/causal task feasibility is claimed; G3 uses the
unmodified asynchronous1x worker instead.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import queue
import sys
import time
import traceback
import zlib

import numpy as np
import PyFlyt
import pybullet

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
_sources={str(p):p.read_text() for p in [*sorted((ROOT/"envs/inspection").glob("*.py")),ROOT/"eval/qualify_inspection_g1.py",
    ROOT/"eval/inspection_tracking.py",Path(__file__)]}
import envs.inspection.mission as mission_module
from envs.inspection.execution import FlightExecutor
from envs.inspection.control import inspection_gains
from envs.inspection.flight import FlightLimits
from envs.inspection.planning import AIRSPEED,OwnState,PlanProposal,plan
from envs.inspection.scenario import Entity
from envs.inspection.scoring import grade_dynamic_bins
from eval.qualify_inspection_g1 import write_json_atomic
from eval.inspection_tracking import recount_tracking


class ControlledPlannerExecutor(FlightExecutor):
    def __init__(self,homes,altitudes,wind,**kwargs):
        super().__init__(homes,altitudes,wind,start_planner=False)
        for q in (self.requests,self.results):
            q.cancel_join_thread()
            q.close()
        self.requests,self.results=queue.Queue(maxsize=1),queue.Queue(maxsize=1)
        self.process=type("ControlledTransport",(),{"exitcode":None})()
        self.controlled_response=None

    def _request(self,job,now):
        super()._request(job,now)
        request=self.requests.get_nowait()
        started=time.monotonic()
        try:
            proposal=plan(request)
        except (ValueError,ArithmeticError) as error:
            proposal=PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                                  None,None,None,None,str(error))
        self.controlled_response=(now+1/120,pickle.dumps((proposal,time.monotonic()-started),protocol=5))
        self.pending_wall_deadline=None

    def _poll(self,now):
        if self.controlled_response is not None and now>=self.controlled_response[0]-1e-8:
            self.results.put_nowait(self.controlled_response[1])
            self.controlled_response=None
        super()._poll(now)

    def close(self):
        self.process=None


def validate_contract(contract):
    settings=contract["executor_qualification"]
    if contract["G1"]["control_gains"]!=asdict(inspection_gains()):
        raise ValueError("contract control profile differs from actual shared implementation")
    if contract["G1"]["nominal_airspeed_m_s"]!=AIRSPEED:
        raise ValueError("contract shared nominal airspeed differs from implementation")
    expected={"xy_tracking_tube_m":6,"steady_altitude_error_m":2,"exit_dwell_seconds_all_aircraft":20,
              "capture_lead_seconds":.5,"observe_seconds":180,"exit_trigger_seconds":240,
              "aircraft_pairing_block":2,"inspect_camera_trial_seconds":0,"work_seconds":8,
              "curvature_ramp_seconds":2,"renewal_interval_seconds":1,"maximum_active_navigation_track_age_seconds":8,
              "continuous_nominal_separation_m":27,"continuous_nominal_fence_abs_xy_m":544,
              "optical_ranking_preview_seconds":32,"nominal_bank_command_limit_degrees":30,
              "follower_requested_airspeed_m_s":[16,17.5],"observation_radius_options_m":[70,80]}
    if any(settings[k]!=v for k,v in expected.items()):
        raise ValueError("this implementation does not support the declared executor envelope")
    if contract["reference_spec"]["initial_altitude_agl_m_per_aircraft"]!=[96.]*4:
        raise ValueError("contract initial geometry differs from the actual shared task")
    limits=FlightLimits()
    actual={"altitude_m":[limits.min_altitude,limits.max_altitude],"airspeed_m_s":[limits.min_airspeed,limits.max_airspeed],
            "bank_abs_degrees":limits.max_bank_degrees,"minimum_separation_m":limits.min_separation,
            "wind_max_m_s":limits.max_wind,"physics_hz":240,"control_hz":120}
    if any(contract["G1"][k]!=v for k,v in actual.items()):
        raise ValueError("contract physical envelope differs from actual shared implementation")
    ids=settings["root_ids"]
    if len(ids)<20 or len(set(ids))!=len(ids) or any(type(r) is not int or r<0 for r in ids):
        raise ValueError("at least20 distinct nonnegative control roots required")


def coverage_valid(result,contract):
    duration=contract["executor_qualification"]["families"][result["family"]]
    expected=round(duration*contract["G1"]["physics_hz"])
    try:
        trace = recount_tracking(result)
        trace_good = all(result["execution_tracking_240hz"][k]==v for k,v in trace.items())
        requests = [r for r in result["execution_audit"] if r["type"]=="plan_request"]
        snapshots_good = bool(requests) and all(len(r["own_states"])==len(r["remaining_range_m"])==4
            and all(OwnState(**s).at==r["now"] for s in r["own_states"])
            and all(math.isfinite(v) and v>=0 for v in r["remaining_range_m"]) for r in requests)
        commits = [r for r in result["execution_audit"] if r["type"]=="program_committed"]
        seams = [r for r in result["execution_audit"] if r["type"]=="physical_program_seam"]
        state_logs = commits+seams
        seams_good = (bool(commits) and {r["drone"] for r in seams if r["label"]=="tail_start"}==set(range(4))
            and all(OwnState(**r["own_state"]).at==r["at"] for r in state_logs)
            and all(isinstance(r["capture_metrics"],dict) for r in commits)
            and result["execution_tracking_240hz"]["seam_capture_failures"]==sum(not r["capture_valid"] for r in seams))
    except (KeyError,ValueError,TypeError,zlib.error):
        return False
    return (trace_good and snapshots_good and seams_good and expected>0 and abs(result["simulated_seconds"]-duration)<1e-8
            and len(result["safety"]["exit_dwell_seconds"])==len(result["remaining_range_m"])==4
            and result["safety"]["physics_steps"]==result["safety"]["physics_audits"]
            ==result["execution_tracking_240hz"]["physics_samples"]==expected
            and result["actual_control_profile"]==contract["G1"]["control_gains"])


def case_passed(result,contract):
    if result["operational_error"]:
        return False
    settings=contract["executor_qualification"]
    surveys=[j for j in result["jobs"] if j["kind"]=="survey"]
    task=(any(j["kind"]=="observe" for j in result["jobs"]) and result["single_target_delivered_D"]>=settings["minimum_delivered_D"]
          if result["family"]=="observe" else len(surveys)==4 and all(j["status"]=="succeeded" for j in surveys))
    tracking=result["execution_tracking_240hz"]
    return (not result["operational_error"] and coverage_valid(result,contract) and task
            and not result["safety"]["violations"] and not result["execution_infrastructure_errors"]
            and tracking["horizontal_tube_breach_steps"]==tracking["altitude_envelope_breach_steps"]==0
            and tracking["seam_capture_failures"]==0
            and all(t>=settings["exit_dwell_seconds_all_aircraft"]-1e-6 for t in result["safety"]["exit_dwell_seconds"])
            and all(r>=contract["reference_spec"]["exit_range_reserve_m"] for r in result["remaining_range_m"]))


def run_case(family,root,contract):
    validate_contract(contract)
    settings=contract["executor_qualification"]
    if type(root) is not int or root not in settings["root_ids"]:
        raise ValueError("a registered integer control root required")
    duration=settings["families"][family]
    wind=(2*np.array([math.cos(root),math.sin(root)]) if root%2 else np.zeros(2))
    original=mission_module.FlightExecutor
    mission_module.FlightExecutor=ControlledPlannerExecutor
    try:
        mission=mission_module.InspectionMission(root,reference=True,events=False,wind_xy=wind)
    finally:
        mission_module.FlightExecutor=original
    selected=(root//settings["aircraft_pairing_block"])%4
    entity=None
    last_public=-1.
    candidate=None
    inspected=False
    observed=False
    target_report=None
    actor_history=[]
    exit_sent=set()
    bins=[]
    error=None
    started=time.monotonic()
    try:
        if (mission.flight.world.physics_hz!=contract["G1"]["physics_hz"]
                or abs(mission.flight.world.step_period-1/contract["G1"]["control_hz"])>1e-12
                or mission.executor.capture_lead_seconds!=settings["capture_lead_seconds"]
                or mission.executor.inspect_camera_trial_seconds!=settings["inspect_camera_trial_seconds"]
                or mission.flight.altitudes.tolist()!=contract["reference_spec"]["initial_altitude_agl_m_per_aircraft"]):
            raise ValueError("constructed world/transport differs from declared qualification")
        if family=="observe":
            body=next(iter(mission.entity_bodies.values()))
            for other in mission.entity_bodies.values():
                mission.flight.world.resetBasePositionAndOrientation(other,[0,0,-10],[0,0,0,1])
            entity=Entity("control-fixture","dynamic","inspection_vehicle",(-250.,float(mission.flight.centers[selected])),
                          0.,1.5,"dynamic-inspection")
            mission.entities=[entity]
            mission.entity_bodies={entity.key:body}
        if family=="transfer":
            shift=root%3+1
            for i in range(4):
                mission.tool("assign_survey",{"drone_id":i,"region_id":f"stripe-{(i+shift)%4}","duration":100.,
                    "expected_version":0,"rule_id":"static-0","task_card_version":1})
        while mission.t<duration-1e-8:
            # Coordinator below reads only the serialized public state. Fixture
            # setup and the independent grader above/below retain the truth.
            if mission.flight.world.physics_steps%120==0:
                state=mission.public_state(since=last_public)
                last_public=mission.t
                actor_history.extend(state["observations"])
                versions={r["drone_id"]:r["job_version"] for r in state["aircraft"]}
                if family=="observe":
                    if candidate is None:
                        seed=next((o for o in actor_history if o["drone_id"]==selected and o["object_type"]=="dynamic"),None)
                        if seed is not None:
                            candidate=seed["candidate_id"]
                    if candidate is not None and not inspected:
                        ack=mission.tool("inspect_candidate",{"drone_id":selected,"candidate_id":candidate,
                            "expected_version":versions[selected],"rule_id":"dynamic-inspection","task_card_version":1})
                        inspected=ack["status"]=="accepted"
                    confirmed=any(j["kind"]=="inspect" and j["status"]=="succeeded" and j["params"].get("candidate_id")==candidate
                                  for j in state["jobs"])
                    qualified=[o for o in actor_history if o["candidate_id"]==candidate and o["sensor_mode"]=="inspect" and o["quality"]>=.8]
                    if confirmed and qualified and not observed:
                        point=qualified[-1]
                        report={k:point[k] for k in ("object_type","candidate_id","measured_at","estimate_xy")}
                        report.update(report_id="control-report",claimed_class=point["estimated_class"],evidence_ids=[point["evidence_id"]])
                        ack=mission.tool("submit_or_revise_report",report)
                        if ack["status"]=="accepted":
                            target_report="control-report"
                            ack=mission.tool("observe_track",{"drone_id":selected,"candidate_id":candidate,"duration":settings["observe_seconds"],
                                "report_id":target_report,"expected_version":versions[selected],"rule_id":"dynamic-inspection","task_card_version":1})
                            observed=ack["status"]=="accepted"
                if mission.t>=settings["exit_trigger_seconds"]:
                    for i in range(4):
                        if i not in exit_sent:
                            ack=mission.tool("enter_safe_exit",{"drone_id":i,"expected_version":versions[i]})
                            if ack["status"]=="accepted":exit_sent.add(i)
            mission.step()
        business=mission.business_audit()
        dynamic=None
        if family=="observe":
            versions=mission.ledger.reports[target_report].versions if target_report else []
            dynamic,bins=grade_dynamic_bins(versions,entity.position,entity.true_class,0.,duration)
    except Exception:
        error=traceback.format_exc()
        business=mission.business_audit()
        dynamic=None
    finally:
        simulated=mission.t
        try:mission.close()
        except Exception:
            error=(error or "")+traceback.format_exc()
    result={"role":"G1_controlled_planner_transport_not_G3_or_method_sample","family":family,"root":root,
            "requested_seconds":duration,"simulated_seconds":simulated,"wind_xy_m_s":wind.tolist(),
            "planner_transport":"pure computation between physical steps; next-control-tick response, not1x",
            "wall_seconds":time.monotonic()-started,"passed":False,"single_target_delivered_D":dynamic,
            "single_target_dynamic_bins":[asdict(b) for b in bins],
            "actual_control_profile":asdict(inspection_gains()),"expected_physics_steps":round(duration*contract["G1"]["physics_hz"]),
            "remaining_range_m":(mission.card["range_budget_m_per_aircraft"]-mission.distance_m).tolist(),
            "operational_error":error,**business}
    if not error and not coverage_valid(result,contract):
        result["operational_error"]="incomplete physical/tracking audit coverage or runtime profile mismatch"
    result["passed"]=case_passed(result,contract)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--roots",type=int)
    parser.add_argument("--families",nargs="+")
    args=parser.parse_args()
    if args.out.exists():parser.error("choose a new output")
    args.out.mkdir(parents=True)
    contract_bytes=args.contract.read_bytes()
    contract=json.loads(contract_bytes)
    settings=contract["executor_qualification"]
    validate_contract(contract)
    if args.roots is not None and not 1<=args.roots<=len(settings["root_ids"]):parser.error("root subset must be positive and available")
    roots=settings["root_ids"][:args.roots] if args.roots is not None else settings["root_ids"]
    families=args.families or list(settings["families"])
    if not roots or len(set(families))!=len(families) or any(f not in {"observe","transfer"} for f in families):
        parser.error("nonempty unique valid grid required")
    if any(Path(p).read_text()!=s for p,s in _sources.items()):raise RuntimeError("source changed since import")
    package=Path(PyFlyt.__file__).resolve().parent
    dependencies=[p for p in package.rglob("*") if p.suffix.lower() in (".py",".yaml",".urdf",".obj",".stl",".dae")]+[Path(pybullet.__file__)]
    fingerprints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    grid=[{"family":f,"root":r,"status":"unstarted"} for f in families for r in roots]
    manifest={"contract":contract,"contract_sha256":hashlib.sha256(contract_bytes).hexdigest(),"sources":_sources,
              "dependency_hashes":fingerprints,"versions":{p:importlib.metadata.version(p) for p in ("PyFlyt","pybullet","numpy","numba")},
              "full_grid":args.roots is None and args.families is None,"grid":grid,"qualified":False}
    def save():write_json_atomic(args.out/"manifest.json",manifest)
    save()
    for slot in grid:
        slot["status"]="running";save()
        try:
            result=run_case(slot["family"],slot["root"],contract)
        except Exception:
            result={"passed":False,"operational_error":traceback.format_exc()}
        path=args.out/f"{slot['family']}-{slot['root']:02d}.json"
        write_json_atomic(path,result)
        slot.update(status="operational_failure" if result["operational_error"] else "completed",passed=result["passed"],
                    file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest());save()
        print(slot["family"],slot["root"],slot["status"],slot["passed"],flush=True)
    manifest["sources_unchanged"]=all(Path(p).read_text()==s for p,s in _sources.items())
    manifest["dependencies_unchanged"]=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in fingerprints.items())
    manifest["contract_unchanged"]=args.contract.read_bytes()==contract_bytes
    manifest["family_rates"]={f:sum(s["passed"] for s in grid if s["family"]==f)/len(roots) for f in families}
    manifest["qualified"]=(manifest["full_grid"] and manifest["sources_unchanged"] and manifest["dependencies_unchanged"]
        and manifest["contract_unchanged"] and all(s["status"]=="completed" for s in grid)
        and all(v>=settings["success_fraction_per_family"] for v in manifest["family_rates"].values()))
    save()
    return 0 if (manifest["qualified"] if manifest["full_grid"] else all(s["passed"] for s in grid)) else 1


if __name__=="__main__":sys.exit(main())
