"""Replay a B0 attempt's executed tool journal through the same physics/scorer.

This replays execution, not the policy or model. It refuses changed local source
and nested handoff journals, whose top-level RPC identity is not recorded yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import inspect
import pickle
import queue
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.inspection_artifacts import verify_source_bundle

_preparser = argparse.ArgumentParser(add_help=False)
_preparser.add_argument("--source-root", type=Path)
_preparser.add_argument("--run", type=Path)
_early, _ = _preparser.parse_known_args()
if _early.source_root is not None:
    if _early.run is None:
        raise ValueError("archived replay requires --run")
    verify_source_bundle(json.loads((_early.run/"run.json").read_text()), _early.source_root)
    sys.path.insert(0, str(_early.source_root.resolve()))
from envs.inspection.mission import InspectionMission


def install_planner_replay(saved):
    """Recompute pure plans, release their responses at recorded logical ticks.

    This audit is deliberately synchronous and is NOT a latency experiment.
    Original planner wall times are preserved, never replaced by replay speed.
    """
    import envs.inspection.mission as mission_module
    from envs.inspection.execution import FlightExecutor, program_hash
    from envs.inspection.planning import PlanProposal, plan
    requests = [row for row in saved["execution_audit"] if row["type"] == "plan_request"]
    responses = [row for row in saved["execution_audit"] if row["type"] == "plan_response"]

    class RecordedProcess:
        exitcode = None

    class ReplayExecutor(FlightExecutor):
        def __init__(self,homes,altitudes,wind,**kwargs):
            super().__init__(homes,altitudes,wind,start_planner=False)
            for channel in (self.requests,self.results):
                channel.cancel_join_thread()
                channel.close()
            self.requests,self.results = queue.Queue(maxsize=1),queue.Queue(maxsize=1)
            self.process = RecordedProcess()
            self.request_index,self.response_index = 0,0
            self.recorded_proposals = {}

        def _request(self,job,now):
            if self.request_index >= len(requests):
                raise ValueError("replay emitted an extra internal plan request")
            expected = requests[self.request_index]
            if abs(expected["now"]-now) > 1e-8:
                raise ValueError("internal plan request is not on its recorded tick")
            super()._request(job,now)
            actual = self.requests.get_nowait()
            normalized = json.loads(json.dumps(self.audit[-1],allow_nan=False))
            if normalized != expected:
                raise ValueError("internal planner input/provenance differs from recorded history")
            try:
                proposal = plan(actual)
            except (ValueError,ArithmeticError) as error:
                proposal = PlanProposal(actual.serial,actual.revision,actual.commit_at,actual.drone,actual.tokens,
                                        None,None,None,None,str(error))
            self.recorded_proposals[actual.serial] = proposal
            self.pending_wall_deadline = None
            self.request_index += 1

        def _poll(self,now):
            if self.response_index >= len(responses):
                return
            row = responses[self.response_index]
            if row["received_at"] > now+1e-8:
                return
            if abs(row["received_at"]-now) > 1e-8:
                raise ValueError("missed the recorded internal planner response tick")
            proposal = self.recorded_proposals[row["serial"]]
            digest = program_hash(proposal.program) if proposal.program else None
            if (digest != row["program_sha256"] or proposal.reason != row["reason"]
                    or json.loads(json.dumps(proposal.geometry)) != row["geometry"]):
                raise ValueError("recomputed pure planner proposal differs from recorded result")
            self.results.put_nowait(pickle.dumps((proposal,row["planner_wall_seconds"]),protocol=5))
            super()._poll(now)
            if json.loads(json.dumps(self.audit[-1],allow_nan=False)) != row:
                raise ValueError("planner response validation differs from recorded result")
            self.response_index += 1

        def close(self):
            if self.request_index != len(requests) or self.response_index != len(responses):
                raise ValueError("unconsumed internal plan journal")

    original = mission_module.FlightExecutor
    mission_module.FlightExecutor = ReplayExecutor
    return mission_module,original


def replay(directory, source_root=None):
    run = json.loads((directory/"run.json").read_text())
    saved = json.loads((directory/"world.json").read_text())
    if run["status"] != "completed" or saved["status"] != "completed" or not run["sources_unchanged"]:
        raise ValueError("only a completed immutable source-bound attempt can be replayed")
    if hashlib.sha256((directory/"world.json").read_bytes()).hexdigest() != run["world_sha256"]:
        raise ValueError("world result hash mismatch")
    locations = verify_source_bundle(run, source_root)
    mission_source = next(locations[name] for name in locations if name.endswith("envs/inspection/mission.py"))
    if Path(inspect.getfile(InspectionMission)).resolve() != mission_source.resolve():
        raise ValueError("start a fresh replay process with the requested --source-root")
    calls = saved["tool_calls"]
    if saved.get("tool_journal_version") != "top-level-v2" and any(c["name"] == "handoff_observation" for c in calls):
        raise ValueError("this journal lacks top-level identities for nested handoff calls")
    planner_patch = install_planner_replay(saved) if "execution_audit" in saved else None
    try:
        mission = InspectionMission(run["seed"], reference=run.get("layout", "reference") == "reference", events=run["events"])
    finally:
        if planner_patch is not None:
            planner_patch[0].FlightExecutor = planner_patch[1]
    at, discrepancies = 0, []
    started = time.monotonic()
    try:
        while mission.t < saved["simulated_seconds"]-1e-8:
            while at < len(calls) and calls[at]["t"] <= mission.t+1e-10:
                call = calls[at]
                if abs(call["t"]-mission.t) > 1e-10:
                    raise ValueError("tool event is not aligned to its original execution tick")
                response = mission.tool(call["name"], call["args"])
                if response != call["response"]:
                    discrepancies.append({"call": at, "expected": call["response"], "actual": response})
                at += 1
            mission.step()
        audit = mission.audit()
        normalized = json.loads(json.dumps(audit, allow_nan=False))
        # Full structural equality includes evidence, all bins and absent slots,
        # not just a rounded summary score or a successful exit indicator.
        keys = ["score", "safety", "trace", "coverage_evidence", "required_cells", "sensor_audit",
                "delivered_observations", "jobs", "job_transitions", "tool_calls", "reports", "hidden_entities", "public_events"]
        keys.extend(key for key in ("execution_audit","development_envelope_id","execution_infrastructure_errors","execution_tracking_240hz","execution_tracking_trace",
                                    "sensor_opportunities","background_trials","candidate_gate_multi_detection_frames","tool_journal_version")
                    if key in saved)
        matches = {key: normalized[key] == saved[key] for key in keys}
        return {"role": "deterministic_execution_journal_replay_not_new_policy_or_model_sample",
                "input_run_sha256": hashlib.sha256((directory/"run.json").read_bytes()).hexdigest(),
                "input_world_sha256": run["world_sha256"], "simulated_seconds": mission.t,
                "replayer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_root": str(source_root) if source_root is not None else None,
                "wall_seconds": time.monotonic()-started, "tool_calls_replayed": at,
                "tool_calls_expected": len(calls), "response_discrepancies": discrepancies,
                "exact_matches": matches,
                "passed": all(matches.values()) and not discrepancies and at == len(calls)}
    finally:
        mission.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a new output path")
    try:
        result = replay(args.run, args.source_root)
    except Exception:
        result = {"passed": False, "status": "operational_failure", "error": traceback.format_exc()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
