"""Execute B0 through a public queue boundary while the physics worker runs."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.inspection.reference import ReferencePolicy


def worker(seed, events, realtime, duration, commands, updates, out, reference_layout=True, shared_progress=None):
    from envs.inspection.mission import InspectionMission
    from envs.inspection.protocol import audit_value
    mission = None
    result = {"status": "operational_failure"}
    try:
        warmup = InspectionMission(seed, reference=reference_layout, events=False)
        for _ in range(120):
            warmup.step()
        warmup.close()
        mission = InspectionMission(seed, reference=reference_layout, events=events)
        epoch = time.monotonic()
        lags, published_at = [], -1.0
        updates.put({"kind": "state", "state": mission.public_state()})
        with (out/"progress.jsonl").open("w") as stream:
            while mission.t < duration-1e-8:
                if realtime:
                    remaining = epoch+mission.t+1/120-time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                for _ in range(16):
                    try:
                        command = commands.get_nowait()
                    except queue.Empty:
                        break
                    if not isinstance(command, dict) or set(command) != {"name", "args"} or not isinstance(command["name"], str) or not isinstance(command["args"], dict):
                        mission.tool_audit.append({"t": mission.t, "invalid_rpc": audit_value(command)})
                        updates.put({"kind": "tool_result", "name": "invalid_rpc", "args": {},
                                     "response": {"status": "rejected", "reason": "malformed actor RPC envelope"}})
                        continue
                    response = mission.tool(command["name"], command["args"])
                    updates.put({"kind": "tool_result", **command, "response": response})
                mission.step()
                if shared_progress is not None:
                    with shared_progress.get_lock():
                        shared_progress[:] = [mission.t, *mission.flight.world.state(0)[3].tolist()]
                lag = time.monotonic()-epoch-mission.t
                lags.append(lag)
                if mission.flight.world.physics_steps % 120 == 0:
                    updates.put({"kind": "state", "state": mission.public_state(since=published_at)})
                    published_at = mission.t
                if mission.flight.world.physics_steps % 240 == 0:
                    stream.write(json.dumps({"t": mission.t, "wall_elapsed": time.monotonic()-epoch,
                                             "safety": mission.flight.summary(),
                                             "reports": len(mission.ledger.reports), "coverage_cells": len(mission.covered)})+"\n")
                    stream.flush()
        result = {"status": "completed", "role": "non_llm_dev_reference_not_formal_comparison",
                  "seed": seed, "events": events, "clock_mode": "wallclock_1x" if realtime else "accelerated_diagnostic",
                  "simulated_seconds": mission.t, "wall_seconds": time.monotonic()-epoch,
                  "max_clock_lag_seconds": max(lags) if realtime else None,
                  "clock_lag_p95_seconds": sorted(lags)[int(.95*(len(lags)-1))] if realtime else None,
                  **mission.audit()}
    except Exception:
        result["operational_error"] = traceback.format_exc()
        if mission is not None:
            result["simulated_seconds"] = mission.t
            result["partial_business_audit"] = mission.business_audit()
    finally:
        if mission is not None:
            try:
                mission.close()
            except Exception:
                result["status"] = "operational_failure"
                result["cleanup_error"] = traceback.format_exc()
        (out/"world.json").write_text(json.dumps(audit_value(result), indent=2, allow_nan=False)+"\n")
        updates.put({"kind": "done", "status": result["status"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--layout", choices=("reference", "generated"), default="reference")
    parser.add_argument("--coordinator-block-seconds", type=float, default=0)
    parser.add_argument("--coordinator-block-at", type=float, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out.exists() or not 0 < args.duration <= 600
            or not 0 <= args.coordinator_block_seconds <= 30
            or (args.coordinator_block_seconds and not 0 <= args.coordinator_block_at < args.duration-args.coordinator_block_seconds)):
        parser.error("choose a new output directory and duration in (0,600]")
    args.out.mkdir(parents=True)
    sources = [*Path("envs/inspection").glob("*.py"), Path(__file__)]
    frozen = {str(p): p.read_text() for p in sources}
    manifest = {"status": "running", "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "role": "dev_reference", "sources": frozen,
                "source_hashes": {p: hashlib.sha256(s.encode()).hexdigest() for p, s in frozen.items()},
                "seed": args.seed, "events": not args.no_events, "realtime": args.realtime,
                "layout": args.layout,
                "requested_seconds": args.duration}
    (args.out/"run.json").write_text(json.dumps(manifest, indent=2)+"\n")
    context = mp.get_context("spawn")
    commands, updates = context.Queue(), context.Queue()
    shared_progress = context.Array("d", [0.0]*4)
    process = context.Process(target=worker, args=(args.seed, not args.no_events, args.realtime,
                                                 args.duration, commands, updates, args.out, args.layout == "reference", shared_progress))
    policy = ReferencePolicy()
    process.start()
    timings, observed_states, error = [], 0, None
    blocked = False
    try:
        while True:
            try:
                update = updates.get(timeout=1)
            except queue.Empty:
                if not process.is_alive():
                    raise RuntimeError(f"worker exited without final message: {process.exitcode}")
                continue
            if update["kind"] == "done":
                break
            if update["kind"] == "tool_result":
                policy.result(update["name"], update["args"], update["response"])
            elif update["kind"] == "state":
                if args.coordinator_block_seconds and not blocked and update["state"]["sim_time"] >= args.coordinator_block_at:
                    with shared_progress.get_lock():
                        before = list(shared_progress[:])
                    block_start = time.monotonic()
                    time.sleep(args.coordinator_block_seconds)
                    actual_wait = time.monotonic()-block_start
                    with shared_progress.get_lock():
                        after = list(shared_progress[:])
                    manifest["coordinator_block_probe"] = {
                        "kind": "sleep_surrogate_not_model_latency", "requested_seconds": args.coordinator_block_seconds,
                        "actual_wait_seconds": actual_wait, "before": before, "after": after,
                        "sim_advance_seconds": after[0]-before[0],
                        "displacement_m": sum((a-b)**2 for a,b in zip(after[1:], before[1:]))**.5}
                    blocked = True
                started = time.perf_counter()
                proposed = policy.decide(update["state"])
                timings.append(time.perf_counter()-started)
                observed_states += 1
                for name, params in proposed:
                    commands.put({"name": name, "args": params})
                if observed_states % 60 == 0:
                    print(f"t={update['state']['sim_time']:.1f} observed={observed_states} calls={policy.calls}", flush=True)
        process.join(timeout=10)
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError(f"worker shutdown failure: {process.exitcode}")
    except Exception:
        error = traceback.format_exc()
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        manifest.update(status="operational_failure" if error else "completed", operational_error=error,
                        actor_mutating_calls=policy.calls, public_snapshots=observed_states,
                        policy_compute_seconds=timings,
                        sources_unchanged=all(Path(p).read_text() == source for p, source in frozen.items()))
        world_path = args.out/"world.json"
        if world_path.exists():
            manifest["world_sha256"] = hashlib.sha256(world_path.read_bytes()).hexdigest()
            world_result = json.loads(world_path.read_text())
            manifest["status"] = world_result["status"] if not error else "operational_failure"
            print(json.dumps({"status": world_result["status"], "simulated_seconds": world_result.get("simulated_seconds"),
                              "score": {k: world_result.get("score", {}).get(k) for k in ("C", "S", "D", "Q", "H", "E", "Y", "static_recall")},
                              "max_clock_lag_seconds": world_result.get("max_clock_lag_seconds")}))
        (args.out/"run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return 0 if manifest["status"] == "completed" and manifest["sources_unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
