"""Reconstruct recorded pure programs and locate physical tracking failures.

No new physics, policy sample or timing measurement. Reads the immutable run,
calls its frozen pure planner, and evaluates its saved10Hz physical trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.inspection_artifacts import verify_source_bundle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run",type=Path,required=True)
parser.add_argument("--source-root",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error("choose a new output")
run = json.loads((args.run/"run.json").read_text())
verify_source_bundle(run,args.source_root)
sys.path.insert(0,str(args.source_root.resolve()))
from envs.inspection.execution import program_hash
from envs.inspection.planning import ObservedTrack,PlanRequest,PlanProposal,initial_program,plan
import envs.inspection.planning as planning_module


def analyze():
    world_path = args.run/"world.json"
    if hashlib.sha256(world_path.read_bytes()).hexdigest() != run["world_sha256"]:
        raise ValueError("world hash mismatch")
    world = json.loads(world_path.read_text())
    events = world["execution_audit"]
    first = next(row for row in events if row["type"] == "plan_request")
    programs = tuple(initial_program(h,first["wind"]) for h in first["homes"])
    proposals,states = {},[(0.,programs,[None]*4)]
    last_commits = [None]*4
    checked_responses = 0
    for row in events:
        if row["type"] == "plan_request":
            values = {key:row[key] for key in PlanRequest.__dataclass_fields__ if key != "programs"}
            values["programs"] = programs
            values["tokens"] = tuple(tuple(t) for t in values["tokens"])
            if values["track"] is not None:
                values["track"] = ObservedTrack(**values["track"])
            if "own_states" in values:
                values["own_states"] = tuple(planning_module.OwnState(**s) for s in values["own_states"])
            request = PlanRequest(**values)
            try:
                proposal = plan(request)
            except (ValueError,ArithmeticError) as error:
                proposal = PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                                        None,None,None,None,str(error))
            proposals[row["serial"]] = proposal
        elif row["type"] == "plan_response":
            proposal = proposals[row["serial"]]
            digest = program_hash(proposal.program) if proposal.program else None
            if digest != row["program_sha256"] or proposal.reason != row["reason"]:
                raise ValueError("reconstructed planner response differs")
            checked_responses += 1
        elif row["type"] == "program_committed":
            proposal = proposals[row["serial"]]
            changed = list(programs)
            changed[row["drone"]] = proposal.program
            programs = tuple(changed)
            last_commits[row["drone"]] = row
            states.append((row["at"],programs,last_commits.copy()))
    jobs = {job["job_id"]:job for job in world["jobs"]}
    state_index = 0
    peaks = [None]*4
    by_kind = {}
    for trace in world["trace"]:
        t = trace["t"]
        while state_index+1 < len(states) and states[state_index+1][0] <= t+1e-8:
            state_index += 1
        _,programs,commits = states[state_index]
        for i,(pos,program) in enumerate(zip(trace["positions"],programs)):
            nominal = program.at(t)
            delta = np.asarray(pos[:2])-nominal[:2]
            error = float(np.linalg.norm(delta))
            tangent = np.array([math.cos(nominal[2]),math.sin(nominal[2])])
            commit = commits[i]
            kind = jobs[commit["job_id"]]["kind"] if commit else "initial"
            since = t-commit["at"] if commit else t
            row = {"t":t,"drone":i,"horizontal_error_m":error,"along_track_error_m":float(delta@tangent),
                   "cross_track_error_m":float(delta@np.array([-tangent[1],tangent[0]])),"actual_position":pos,
                   "nominal_xy":nominal[:2].tolist(),"nominal_bank_degrees":math.degrees(nominal[4]),
                   "committed_job_kind":kind,"seconds_after_commit":since,"last_commit":commit}
            if peaks[i] is None or error > peaks[i]["horizontal_error_m"]:
                peaks[i] = row
            group = kind+("_first20s" if since < 20 else "_after20s")
            if group not in by_kind or error > by_kind[group]["horizontal_error_m"]:
                by_kind[group] = row
    return {"role":"recorded_program_and10Hz_trace_diagnostic_not_new_sample_or240Hz_bound",
            "input_world_sha256":run["world_sha256"],"source_root":str(args.source_root.resolve()),
            "analyzer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reconstructed_responses":checked_responses,"peaks_by_aircraft":peaks,"peaks_by_job_phase":by_kind,
            "executed_tracking_240hz":world["execution_tracking_240hz"]}


try:
    result = analyze()
    result["status"] = "completed"
except Exception:
    result = {"status":"operational_failure","error":traceback.format_exc()}
args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("peaks_by_job_phase",)},indent=2))
sys.exit(0 if result["status"] == "completed" else 1)
