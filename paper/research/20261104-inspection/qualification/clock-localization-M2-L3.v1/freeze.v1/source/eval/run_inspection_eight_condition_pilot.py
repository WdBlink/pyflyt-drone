"""Fixed six-root, eight-condition development clock/accounting pilot.

The existing recording SDK owns all108 inner slots and48 root means. It is
not the formal experiment, a second paper controller or a classic reselection.
Every LLM operation uses the sole owner's native reservation and settlement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.inspection_episode_receipts import canonical_digest, digest, execute_local_episode, validate_flight
from eval.inspection_recording import CONDITION_REPEATS, episode_schedule, record_matrix
from eval.inspection_model_service import _digest as frozen_owner_plan_digest
from eval.inspection_process_cleanup import SubprocessTree
from eval.run_inspection_llm_episode import execute_development_llm_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--flight-receipts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--schedule-seed", type=int, required=True)
    parser.add_argument("--sampler-seed", type=int, required=True)
    parser.add_argument("--reservation-prefix", required=True)
    parser.add_argument("--prepare-only", action="store_true", help="write source/order/seed intent only; never reserve or dispatch")
    args = parser.parse_args()
    if args.out.exists() or not 0 <= args.sampler_seed < 2**32:
        parser.error("choose a new complete SDK envelope and uint32 sampler master seed")
    # Thread counts are fixed before importing the physics runtime in children.
    threads = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    if any(os.environ.get(k) != v for k, v in threads.items()):
        parser.error("launch with all four declared BLAS/thread variables equal to1")
    if digest(args.selection) != args.selection_sha256:
        parser.error("fixed original classical selection bytes differ")
    selection = json.loads(args.selection.read_text())
    if selection["completed_slots"] != 18 or selection["root_count"] != 6 or selection["classic_best"] not in ("B1", "B2"):
        parser.error("the completed original18-slot classic selection is required")
    manifest = json.loads((args.suite/"manifest.json").read_text())
    rows = [r for r in manifest["records"] if r.get("first_pilot") is True]
    cases = sorted(r["case_id"] for r in rows)
    if (manifest.get("split") != "dev" or manifest.get("main_and_holdout_instantiated") is not False
            or cases != [f"dev-L{level}-{i:02d}" for level in (1, 2, 3) for i in (0, 1)]
            or len({r["lineage_group"] for r in rows}) != 6):
        parser.error("only the original six preselected development roots are admitted")
    paths = {r["case_id"]: (args.suite/r["file"]).resolve() for r in rows}
    if any(p.parent != args.suite.resolve() for p in paths.values()):
        parser.error("development case escaped its suite")
    registry_path = args.run/"planning/local-model-registry.v1.json"
    registry = json.loads(registry_path.read_text())
    owner = json.loads((args.run/".paper/control.json").read_text())
    plans = [receipt["packet"] for receipt in owner.get("receipts", []) if receipt.get("stage") == "plan"]
    if (not owner.get("active_round") or len(plans) != 1 or owner.get("frozen_plan") != frozen_owner_plan_digest(plans[0])
            or owner.get("task_stop") is not None or digest(registry_path) != plans[0]["local_model_registry_sha256"]):
        parser.error("an accepted native plan and active work round are required")
    design_path = args.run/plans[0]["experiment_design"]
    if digest(design_path) != plans[0]["experiment_design_sha256"]:
        parser.error("accepted scientific design changed")
    scientific_design = json.loads(design_path.read_text())
    evaluator = (args.run/scientific_design["evaluator"]["path"]).resolve()
    if digest(evaluator) != scientific_design["evaluator"]["sha256"]:
        parser.error("accepted evaluator entry changed")
    closure = evaluator.parent/"evaluator-source.v2"
    closure_manifest = evaluator.parent/"evaluator-source-manifest.v2.json"
    if digest(closure_manifest) != "ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80":
        parser.error("manifest differs from the accepted evaluator entry's fixed anchor")
    closure_spec = json.loads(closure_manifest.read_text())
    closure_files = {str(p.relative_to(closure)) for p in closure.rglob("*") if p.is_file()}
    if closure_files != set(closure_spec["sources"]) or any(p.is_symlink() for p in closure.rglob("*")):
        parser.error("frozen evaluator closure file set changed")
    if any(digest(closure/name) != h for name, h in closure_spec["sources"].items()):
        parser.error("frozen evaluator closure bytes changed")
    args.out.mkdir(parents=True)
    args.flight_receipts.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.sdk_root.resolve()))
    from lib.experiments import ExperimentRecorder
    from lib.production import json_bytes, write_once
    modules = ("inspection_cases.py", "inspection_population.py", "inspection_scenarios.py", "inspection_classic.py",
        "run_inspection_reference.py", "run_inspection_dev.py", "run_inspection_llm_development.py",
        "inspection_llm_actor.py", "inspection_llm_protocol.py", "inspection_model_service.py", "inspection_model_client.py",
        "inspection_model_receipts.py", "inspection_episode_receipts.py", "inspection_llm_episode_receipts.py",
        "run_inspection_llm_episode.py", "inspection_recording.py", "inspection_process_cleanup.py",
        "inspection_darwin_inventory.py")
    sources = [*sorted((ROOT/"envs/inspection").glob("*.py")), ROOT/"envs/__init__.py",
        *[ROOT/"eval"/name for name in modules], Path(__file__).resolve(), *sorted((args.sdk_root/"lib").glob("*.py")),
        evaluator, closure_manifest, *[closure/name for name in sorted(closure_files)]]
    migrations = [row for row in owner.get("history", [])
        if row.get("event") == "round_documents_runtime_migrated"]
    if migrations:
        ref = migrations[-1]["transport"]
        transport = (args.run/ref["path"]).resolve()
        if not transport.is_relative_to(args.run.resolve()) or digest(transport) != ref["sha256"]:
            parser.error("bound native-owner transport changed")
        owner_sources = json.loads(transport.read_text()).get("sources")
        if not isinstance(owner_sources, dict) or not owner_sources:
            parser.error("complete native-owner source inventory required")
        for name, expected in owner_sources.items():
            path = (args.run/name).resolve()
            if not path.is_relative_to(args.run.resolve()) or digest(path) != expected:
                parser.error("native-owner/compatibility source differs")
            sources.append(path)
        sources.append(transport)
    frozen = {str(p.resolve()): digest(p) for p in sources}
    package = Path(importlib.util.find_spec("PyFlyt").origin).parent
    deps = [p for p in package.rglob("*") if p.suffix.lower() in (".py", ".yaml", ".urdf", ".obj", ".stl", ".dae")]
    deps.append(Path(importlib.util.find_spec("pybullet").origin))
    dependency_hashes = {str(p): digest(p) for p in deps}
    input_hashes = {str(p.resolve()): digest(p) for p in [args.selection, registry_path, design_path, args.suite/"manifest.json", *paths.values()]}
    metadata = [{"case_id": r["case_id"], "domain": "ID", "level": int(r["case_id"].split("-")[1][1:])} for r in rows]
    schedule = episode_schedule(metadata, args.schedule_seed)
    rng = np.random.Generator(np.random.PCG64(args.sampler_seed))
    seeds = {s["order_index"]: int(rng.integers(0, 2**32)) for s in schedule["slots"] if CONDITION_REPEATS[s["condition"]] == 3}
    slots = {(s["case_id"], s["repeat_ordinal"], s["condition"]): s for s in schedule["slots"]}
    for name, value in (("source-hashes.json", frozen), ("source-files.json", {str(p.resolve()): p.read_text() for p in sources}),
            ("dependency-hashes.json", dependency_hashes), ("input-hashes.json", input_hashes),
            ("sampler-seeds.json", {"role": "prospective dev sampler seeds, not provider reproducibility guarantee",
                "master": args.sampler_seed, "rng": "NumPy PCG64", "by_order_index": seeds})):
        write_once(args.out, name, json_bytes(value))
    if args.prepare_only:
        write_once(args.out, "prepared-episode-order.json", json_bytes(schedule))
        write_once(args.out, "preparation.json", json_bytes({"role": "prospective development intent only; no SDK callbacks or dispatch",
            "episode_denominator": len(schedule["slots"]), "conditions": CONDITION_REPEATS,
            "original_classic_selection_sha256": args.selection_sha256, "classic_best": selection["classic_best"],
            "actual_model_generation_calls": 0, "actual_physical_flights": 0}))
        return
    def check():
        if any(digest(p) != h for table in (frozen, dependency_hashes, input_hashes) for p, h in table.items()):
            raise ValueError("frozen pilot source/dependency/input changed; retain complete denominator")
        if {str(p.relative_to(closure)) for p in closure.rglob("*") if p.is_file()} != closure_files or any(p.is_symlink() for p in closure.rglob("*")):
            raise ValueError("frozen evaluator closure file set changed")
    base_sources = [*sorted((ROOT/"envs/inspection").glob("*.py")), ROOT/"envs/__init__.py",
        ROOT/"eval/inspection_cases.py", ROOT/"eval/inspection_scenarios.py", ROOT/"eval/inspection_population.py",
        ROOT/"eval/run_inspection_reference.py"]
    llm_sources = [*base_sources, *[ROOT/"eval"/n for n in ("inspection_llm_actor.py", "inspection_llm_protocol.py",
        "inspection_model_client.py", "inspection_model_service.py", "inspection_process_cleanup.py", "run_inspection_llm_development.py")]]
    def measure(case, repeat, condition):
        check()
        slot = slots[case, repeat, condition]
        request = args.out/"checkpoint.request"
        if request.exists():
            # An internal owner checkpoint waits BEFORE any reservation/spawn.
            # Never suspend a physical worker or a live model operation.
            with (args.out/"checkpoint-pauses.jsonl").open("a") as stream:
                stream.write(json.dumps({"role": "between episodes; no request or flight started for this slot",
                    "order_index": slot["order_index"], "case_id": case, "condition": condition,
                    "at": datetime.now(timezone.utc).isoformat()})+"\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps({"status": "checkpoint_ready_before_dispatch", "order_index": slot["order_index"]}), flush=True)
            while request.exists():
                time.sleep(.2)
            check()
        row = next(r for r in rows if r["case_id"] == case)
        if CONDITION_REPEATS[condition] == 1:
            command = [sys.executable, str(ROOT/"eval/run_inspection_dev.py"), "--suite", str(args.suite.resolve()),
                "--case-id", case, "--policy", condition, "--receipts", str((args.flight_receipts/"classic").resolve())]
            invocation = args.out/f"invocations/{slot['order_index']:03d}"
            invocation.parent.mkdir(parents=True, exist_ok=True)
            tree, process, cleanup, code = SubprocessTree(), None, None, None
            try:
                with Path(str(invocation)+".stdout").open("xb") as stdout, Path(str(invocation)+".stderr").open("xb") as stderr:
                    try:
                        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
                        tree.attach(process)
                        code = tree.wait(1700)
                    finally:
                        if process is not None:
                            cleanup = tree.close()
            finally:
                write_once(args.out, f"invocations/{slot['order_index']:03d}.json", json_bytes({"argv": command,
                    "returncode": code, "cleanup": cleanup, "stdout_path": str(invocation)+".stdout",
                    "stderr_path": str(invocation)+".stderr"}))
            if code != 0 or cleanup is None or cleanup["status"] != "closed":
                raise ValueError("classical operational failure or unknown blocks the pilot")
            response = json.loads(Path(str(invocation)+".stdout").read_text().splitlines()[-1])
            terminal_path = Path(response["episode_receipt"])
            if not terminal_path.resolve().is_relative_to((args.flight_receipts/"classic").resolve()):
                raise ValueError("classical receipt escaped its fixed store")
            plan = json.loads((terminal_path.parent.parent/"plan.json").read_text())
            expected = {str(p): frozen[str(p)] for p in base_sources+([ROOT/"eval/inspection_classic.py"] if condition != "B0" else [])}
            if (plan["source_hashes"] != expected or plan["dependency_hashes"] != dependency_hashes
                    or plan["case_file_sha256"] != row["file_sha256"] or plan["case_sha256"] != row["case_sha256"]
                    or plan["case_id"] != case or plan["condition"] != condition
                    or plan["supervisor_sha256"] != frozen[str(ROOT/"eval/run_inspection_dev.py")]
                    or plan["receipt_adapter_sha256"] != frozen[str(ROOT/"eval/inspection_episode_receipts.py")]):
                raise ValueError("classical input/source receipt differs from pilot binding")
            def forbidden(_): raise ValueError("post-return verification cannot dispatch")
            y, cached = execute_local_episode(terminal_path.parent.parent, plan, forbidden)
            if y != response["Y"] or cached["episode_receipt_sha256"] != response["episode_receipt_sha256"]:
                raise ValueError("classical completed receipt differs")
            evidence = response
        else:
            architecture, model = condition.split("-")
            profile = registry["models"][model]
            limits = profile["generation_profile"]
            plan = {"role": "fixed all-eight development pilot, not main or G4 acceptance", "condition": condition,
                "case_id": case, "lineage_group": row["lineage_group"], "repeat_ordinal": repeat, "max_attempts": 2,
                "case_path": str(paths[case]), "case_file_sha256": row["file_sha256"], "case_sha256": row["case_sha256"],
                "suite_manifest_sha256": input_hashes[str((args.suite/"manifest.json").resolve())],
                "source_hashes": {str(p): frozen[str(p)] for p in llm_sources}, "dependency_hashes": dependency_hashes,
                "thread_environment": threads, "max_clock_lag_seconds": .1, "sampler_seed": seeds[slot["order_index"]],
                "model_registry_sha256": input_hashes[str(registry_path.resolve())],
                "model_receipt_sha256": profile["download_receipt_sha256"], "generation_profile": limits,
                "owner_frozen_plan": owner["frozen_plan"], "reservation_usage": {"calls": limits["max_calls"], "tokens": limits["max_total_tokens"]},
                "injected_delay_seconds": 0,
                "service_sha256": frozen[str(ROOT/"eval/inspection_model_service.py")], "evaluator": str(evaluator),
                "evaluator_sha256": frozen[str(evaluator)], "physics_python": sys.executable}
            def command(out, reservation):
                return [sys.executable, str(ROOT/"eval/run_inspection_llm_development.py"), "--run", str(args.run.resolve()),
                    "--reservation", reservation, "--model", model, "--architecture", architecture,
                    "--case", str(paths[case]), "--sampler-seed", str(plan["sampler_seed"]), "--out", str(out.resolve())]
            y, evidence = execute_development_llm_episode(args.flight_receipts/f"{case}-{condition}-{repeat}", plan, command,
                run=args.run, reservation_prefix=args.reservation_prefix+f"-{slot['order_index']:03d}")
        check()
        print(json.dumps({**slot, "Y": y, "cache_reused": evidence["cache_reused"]}), flush=True)
        return y, {**evidence, "role": "source-bound dev flight; no classic reselection or formal inference"}
    context = {"experiment_id": "fixed-six-root-eight-condition-development-v1", "claim_id": "G4-clock-accounting-development",
        "hypothesis": "Execute the full prospective condition/repeat matrix under continuous physics and receipt accounting.",
        "independent_variable": "eight declared conditions", "dependent_variable": "actual scored Y and metered request outcomes",
        "control_variables": ["same frozen six roots", "same shared worker/sensor/grader", "600s1x", "two-attempt infrastructure cap"],
        "data_roles": [{"id": "fixed-original-six-development-roots", "role": "development", "used_for_selection": True,
            "selection_scope": "original separate18-slot B1/B2 selection; this pilot does not reselect"}],
        "planned_at": datetime.now(timezone.utc).isoformat()}
    result = record_matrix(args.out, cases, measure, context=context, classic_best=selection["classic_best"],
        execution_order=schedule, recorder_type=ExperimentRecorder, write_json=lambda root, name, value: write_once(root, name, json_bytes(value)))
    check()
    write_once(args.out, "pilot-completion.json", json_bytes({**result, "role": "all-eight development callbacks complete, not G4 GO",
        "original_classic_selection_sha256": args.selection_sha256, "classic_best_unchanged": selection["classic_best"]}))


if __name__ == "__main__":
    main()
