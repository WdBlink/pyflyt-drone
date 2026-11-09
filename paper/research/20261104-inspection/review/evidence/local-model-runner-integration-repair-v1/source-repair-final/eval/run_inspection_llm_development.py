"""Reserved local-model development flight over the qualified worker queue.

Requires the sole owner's accepted plan and an unused development reservation.
This runner does not approve G4, dispatch main/holdout, retry or settle usage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.inspection_llm_actor import PublicLLMActor
from eval.inspection_model_client import LocalModelClient, ModelOperationUnknown
from eval.inspection_process_cleanup import stop_worker_tree
from eval.run_inspection_reference import check_private_copy, worker


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--model", choices=("M1", "M2", "M3"), required=True)
    parser.add_argument("--architecture", choices=("B3", "B4"), required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--sampler-seed", type=int, required=True)
    parser.add_argument("--ablation", choices=("periodic", "no-cross-aircraft-memory", "no-progressive-confirmation-guidance"))
    parser.add_argument("--llm-python", type=Path, default=ROOT/".venv-inspection-llm/bin/python")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out.exists() or not 0 <= args.sampler_seed < 2**32 or args.architecture == "B4" and args.model == "M3"
            or args.ablation is not None and args.architecture != "B4"):
        parser.error("choose a new development output and a declared condition/uint32 sampling seed")
    registry_path = args.run/"planning/local-model-registry.v1.json"
    registry = json.loads(registry_path.read_text())
    profile = registry["models"][args.model]
    limits = profile["generation_profile"]
    receipt_path = args.run/profile["download_receipt"]
    if digest(receipt_path) != profile["download_receipt_sha256"]:
        raise ValueError("model receipt differs from the registry")
    args.out.mkdir(parents=True)
    private = args.out/"private-case.json"
    private.write_bytes(args.case.read_bytes())
    case_sha = digest(private)
    sources = [*sorted((ROOT/"envs/inspection").glob("*.py")), ROOT/"envs/__init__.py",
        *[ROOT/"eval"/name for name in ("inspection_cases.py", "inspection_scenarios.py", "run_inspection_reference.py",
            "inspection_llm_actor.py", "inspection_llm_protocol.py", "inspection_model_client.py", "inspection_model_service.py",
            "inspection_process_cleanup.py")], Path(__file__)]
    frozen = {str(p): p.read_text() for p in sources}
    package = Path(importlib.util.find_spec("PyFlyt").origin).parent
    dependencies = [p for p in package.rglob("*") if p.suffix.lower() in
        (".py", ".yaml", ".urdf", ".obj", ".stl", ".dae")]
    dependencies.append(Path(importlib.util.find_spec("pybullet").origin))
    fingerprints = {str(p): digest(p) for p in dependencies}
    manifest = {"role": "actual local-model development runner, not main data or G4 qualification",
        "status": "setup", "started_at": datetime.now(timezone.utc).isoformat(),
        "policy": args.architecture+"-"+args.model, "requested_seconds": 600, "realtime": True,
        "reservation": args.reservation, "model_registry_sha256": digest(registry_path),
        "model_receipt_sha256": digest(receipt_path), "sources": frozen,
        "source_hashes": {p: hashlib.sha256(s.encode()).hexdigest() for p, s in frozen.items()},
        "dependency_hashes": fingerprints, "private_case_sha256": case_sha,
        "sampler_base_seed": args.sampler_seed, "model_generation_profile": limits,
        "ablation": args.ablation,
        "runtime_thread_environment": {name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "package_versions": {name: importlib.metadata.version(name) for name in ("PyFlyt", "pybullet", "numpy", "numba")},
        "worker_role_metadata": "shared reference worker has a legacy non_llm label; this runner's actual model/command receipts identify the policy",
        "usage_settlement": "not performed by this runner; owner must reconcile the complete service operation"}
    write(args.out/"run.json", manifest)
    context = mp.get_context("spawn")
    commands, updates = context.Queue(), context.Queue()
    progress = context.Array("d", [0.]*4)
    process = context.Process(target=worker, args=(0, False, True, 600, commands, updates, args.out,
        False, progress, private, case_sha))
    actor = PublicLLMActor(args.architecture, max_calls=limits["max_calls"],
        max_request_seconds=limits["max_request_seconds"], ablation=args.ablation)
    client, error, mutation_count, observed, last_start_progress = None, None, 0, 0, None
    service_command = [str(args.llm_python.resolve()), "-u", str(ROOT/"eval/inspection_model_service.py"),
        "--run", str(args.run.resolve()), "--reservation", args.reservation,
        "--model-receipt", str(receipt_path.resolve()), "--max-calls", str(limits["max_calls"]),
        "--max-total-tokens", str(limits["max_total_tokens"]), "--max-input-tokens", str(limits["max_input_tokens"]),
        "--max-output-tokens", str(limits["max_output_tokens"]), "--max-request-seconds", str(limits["max_request_seconds"])]
    if profile["tokenizer_config"].get("fix_mistral_regex"):
        service_command.append("--fix-mistral-regex")
    operation = args.run/"resource-operations"/args.reservation
    def physical_progress():
        with progress.get_lock():
            return list(progress[:])
    try:
        # Pin/load setup happens before physical t=0; the first real forward and
        # every subsequent wait occur after dispatch while the worker advances.
        client = LocalModelClient(service_command, args.out/"model-transport")
        if client.ready.get("reservation") != args.reservation:
            raise ModelOperationUnknown("ready reservation differs")
        process.start()
        manifest["status"] = "running"
        write(args.out/"run.json", manifest)
        with (args.out/"actor.jsonl").open("x") as journal:
            def log(value):
                journal.write(json.dumps(value, allow_nan=False)+"\n")
                journal.flush()
            def consume_response(response, *, final=False):
                nonlocal mutation_count
                current = physical_progress()
                log({"kind": "actual_model_response", "response": response,
                    "worker_before_request": last_start_progress, "worker_after_response": current,
                    "final_after_flight": final})
                if response.get("kind") == "response" and response.get("status") == "runtime_failure":
                    raise RuntimeError("known model runtime failure; retain usage and failed physical attempt")
                end_to_end = response["parent_end_to_end_wall_seconds"]
                decoded, audit = actor.finish_request({**response,
                    "wall_seconds": max(response.get("wall_seconds", 0), end_to_end),
                    "deadline_exceeded": response.get("deadline_exceeded", True) or end_to_end > limits["max_request_seconds"]})
                dispatched = []
                if not final and current[0] < 600:
                    for name, params in decoded:
                        if mutation_count >= 2048:
                            break
                        command = {"name": name, "args": params}
                        commands.put(command)
                        dispatched.append(command)
                        mutation_count += 1
                log({"kind": "decoded_reply", "audit": audit, "commands_enqueued_not_accepted": dispatched,
                     "worker_progress_at_enqueue": current})
            while True:
                response = client.poll()
                if response is not None:
                    consume_response(response)
                try:
                    update = updates.get(timeout=.02)
                except queue.Empty:
                    if not process.is_alive():
                        raise RuntimeError("physical worker exited without its final envelope")
                    continue
                if update["kind"] == "done":
                    if update["status"] != "completed":
                        raise RuntimeError("physical worker reported an operational failure")
                    break
                if update["kind"] == "tool_result":
                    actor.result(update["name"], update["args"], update["response"])
                elif update["kind"] == "state":
                    actor.ingest(update["state"])
                    observed += 1
                    if actor.due() and physical_progress()[0] < 590:
                        request = actor.start_request()
                        seed = (args.sampler_seed+request["request_ordinal"]-1) % 2**32
                        last_start_progress = physical_progress()
                        log({"kind": "public_model_request", **request, "sampler_seed": seed,
                             "worker_before_request": last_start_progress})
                        client.submit(request["messages"], seed)
            process.join(timeout=10)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError("physical worker did not close cleanly")
            final_response = client.finish()
            if final_response is not None:
                consume_response(final_response, final=True)
    except BaseException:
        error = traceback.format_exc()
    finally:
        if process.pid is not None and process.is_alive():
            try:
                manifest["worker_cleanup"] = stop_worker_tree(process)
            except BaseException:
                manifest["worker_cleanup_unknown"] = traceback.format_exc()
                error = error or manifest["worker_cleanup_unknown"]
        if client is not None and client.process.poll() is None:
            try:
                final_response = client.finish()
                if final_response is not None:
                    write(args.out/"ignored-final-model-response.json", final_response)
            except BaseException:
                manifest["service_shutdown_unknown"] = traceback.format_exc()
                if client.process.poll() is None:
                    client.stop_unknown()
        completion = operation/"service-completion.json"
        try:
            usage = json.loads(completion.read_text())["usage"]
            if (set(usage) != {"calls", "tokens"} or any(type(v) is not int or v < 0 for v in usage.values())
                    or usage["calls"] > limits["max_calls"] or usage["tokens"] > limits["max_total_tokens"]):
                raise ValueError("service completion usage is outside the admitted profile")
            manifest["model_operation"] = {"completion": str(completion.resolve()), "sha256": digest(completion),
                "usage": usage, "generation_calls_are_actual": usage["calls"] > 0}
        except BaseException:
            manifest["model_operation_unknown"] = traceback.format_exc()
            error = error or manifest["model_operation_unknown"]
        unchanged, verification_error = check_private_copy(private, case_sha)
        try:
            sources_unchanged = all(Path(p).read_text() == s for p, s in frozen.items())
            dependencies_unchanged = all(digest(p) == h for p, h in fingerprints.items())
        except (OSError, UnicodeError):
            sources_unchanged = dependencies_unchanged = False
        if not (unchanged and sources_unchanged and dependencies_unchanged):
            error = error or "source/dependency/private case validation failed; no scored admission"
        try:
            world = json.loads((args.out/"world.json").read_text())
            if (not isinstance(world, dict) or world["status"] != "completed" or world["simulated_seconds"] != 600
                    or world["physics_steps_executed"] != 144000):
                raise ValueError("physical completion differs")
        except (OSError, ValueError, KeyError):
            error = error or "missing/malformed/incomplete physical world; no scored admission"
        manifest.update(status="operational_failure" if error else "completed", operational_error=error,
            actor_mutating_calls=mutation_count, public_snapshots=observed,
            private_case_unchanged=unchanged, private_case_verification_error=verification_error,
            sources_unchanged=sources_unchanged, dependencies_unchanged=dependencies_unchanged)
        if (args.out/"world.json").is_file():
            manifest["world_sha256"] = digest(args.out/"world.json")
        write(args.out/"run.json", manifest)
    print(json.dumps({"status": manifest["status"], "out": str(args.out), "actual_calls": manifest.get("model_operation", {}).get("usage"),
        "error": error}, allow_nan=False))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
