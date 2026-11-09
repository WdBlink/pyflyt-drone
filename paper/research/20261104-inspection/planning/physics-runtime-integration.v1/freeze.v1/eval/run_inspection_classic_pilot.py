"""Six preselected development roots, three classical policies, supplied SDK.

Each new recording envelope may reconstruct verified cached flights. It never
resumes an incomplete SDK envelope in place or treats a cache read as a flight.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.inspection_episode_receipts import canonical_digest, digest, execute_local_episode, validate_flight


def development_schedule(manifest):
    rows = sorted((r for r in manifest["records"] if r["first_pilot"]), key=lambda r: r["case_id"])
    if (manifest.get("split") != "dev" or len(rows) != 6 or len({r["case_id"] for r in rows}) != 6
            or len({r["lineage_group"] for r in rows}) != 6
            or {r["case_id"] for r in rows} != {f"dev-L{l}-{i:02d}" for l in (1, 2, 3) for i in (0, 1)}):
        raise ValueError("the original six preselected independent development roots are required")
    rng = np.random.Generator(np.random.PCG64(2026110601))
    roots = [f"dev-L{level}-{ordinal:02d}" for ordinal in rng.permutation(2) for level in rng.permutation([1, 2, 3])]
    return [{"order_index": index, "case_id": case, "condition": condition, "repeat_ordinal": 0}
            for index, (case, condition) in enumerate((c, str(p)) for c in roots for p in rng.permutation(["B0", "B1", "B2"]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--flight-receipts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("preserve prior SDK envelopes; choose a new recording directory")
    args.out.mkdir(parents=True)
    manifest = json.loads((args.suite/"manifest.json").read_text())
    schedule = development_schedule(manifest)
    source_paths = [*sorted((ROOT/"envs/inspection").glob("*.py")), ROOT/"envs/__init__.py",
        *[ROOT/"eval"/name for name in ("run_inspection_reference.py", "run_inspection_dev.py", "inspection_cases.py",
          "inspection_scenarios.py", "inspection_population.py", "inspection_classic.py", "inspection_episode_receipts.py",
          "inspection_worker_runtime.py")], Path(__file__),
        *sorted((args.sdk_root/"lib").glob("*.py"))]
    frozen = {str(p.resolve()): digest(p) for p in source_paths}
    suite_hash = digest(args.suite/"manifest.json")
    cases_by_id = {r["case_id"]: r for r in manifest["records"]}
    package = Path(importlib.util.find_spec("PyFlyt").origin).parent
    dependencies = [p for p in package.rglob("*") if p.suffix.lower() in
                    (".py", ".yaml", ".urdf", ".obj", ".stl", ".dae")]
    dependencies.append(Path(importlib.util.find_spec("pybullet").origin))
    dependency_hashes = {str(p): digest(p) for p in dependencies}
    worker_sources = [*sorted((ROOT/"envs/inspection").glob("*.py")), ROOT/"envs/__init__.py",
                      ROOT/"eval/run_inspection_reference.py", ROOT/"eval/inspection_cases.py",
                      ROOT/"eval/inspection_scenarios.py", ROOT/"eval/inspection_population.py",
                      ROOT/"eval/inspection_worker_runtime.py"]
    def expected_worker_sources(condition):
        paths = worker_sources + ([ROOT/"eval/inspection_classic.py"] if condition != "B0" else [])
        return {str(path.resolve()): frozen[str(path.resolve())] for path in paths}
    def check_frozen():
        if (digest(args.suite/"manifest.json") != suite_hash
                or any(digest(name) != value for name, value in frozen.items())
                or any(digest(name) != value for name, value in dependency_hashes.items())):
            raise ValueError("frozen development input/source/dependency changed; preserve partial denominator")
    sys.path.insert(0, str(args.sdk_root.resolve()))
    from lib.experiments import ExperimentRecorder
    from lib.production import json_bytes, write_once
    write_once(args.out, "source-hashes.json", json_bytes(frozen))
    write_once(args.out, "source-files.json", json_bytes({name: Path(name).read_text() for name in frozen}))
    write_once(args.out, "dependency-hashes.json", json_bytes(dependency_hashes))
    write_once(args.out, "schedule.json", json_bytes({"role": "development_only_not_eight_condition_main",
        "seed": 2026110601, "rng": "NumPy PCG64", "slots": schedule, "suite_sha256": suite_hash}))
    cases = sorted({row["case_id"] for row in schedule})
    design = {"experiment_id": "six-root-classical-development-v1", "claim_id": "dev-classic-selection-only",
        "hypothesis": "Compare the full delivery chain of three classical candidates on preselected development roots.",
        "independent_variable": "B0/B1/B2 policy", "dependent_variable": "actual development Y",
        "control_variables": ["same frozen case", "same qualified physical/sensor/scoring components", "600s1x", "same finite infrastructure retry"],
        "data_roles": [{"id": "six-preselected-dev-roots", "role": "development", "used_for_selection": True}],
        "planned_at": datetime.now(timezone.utc).isoformat(), "cases": cases, "seeds": [0],
        "conditions": ["B0", "B1", "B2"], "statistical_unit": "case", "failure_policy": "penalty", "failure_value": 0.,
        "metric": {"name": "Y", "unit": "quality", "direction": "higher"},
        "statistics": {"estimator": "mean", "interval": "none", "paired": True,
            "contrasts": [{"id": "dev-online-candidates", "minuend": "B1", "subtrahend": "B2"}]},
        "selection": "Only after all18 completed: equal-level macro Y of B1/B2; tie<=1e-12 chooses B1. B0 remains in the main matrix.",
        "role": "development_diagnostic_not_G4_or_formal_eight_condition_acceptance"}
    write_once(args.out, "design.json", json_bytes(design))
    measurements = {}
    with ExperimentRecorder(args.out, design="design.json", results="results.json", raw_directory="raw") as recorder:
        for slot in schedule:
            case, condition = slot["case_id"], slot["condition"]
            with recorder.attempt(case, 0, condition) as complete:
                check_frozen()
                command = [sys.executable, str(ROOT/"eval/run_inspection_dev.py"), "--suite", str(args.suite.resolve()),
                    "--case-id", case, "--policy", condition, "--receipts", str(args.flight_receipts.resolve())]
                started = time.monotonic()
                process = subprocess.run(command, capture_output=True, text=True)
                write_once(args.out, f"invocations/{slot['order_index']:02d}.json", json_bytes({**slot,
                    "role": "observed_local_supervisor_return_not_proof_of_a_new_flight", "argv": command,
                    "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr,
                    "wall_seconds": time.monotonic()-started}))
                if process.returncode != 0:
                    raise RuntimeError("development infrastructure/unknown attempt blocks the matrix; inspect retained receipts")
                response = json.loads(process.stdout.splitlines()[-1])
                terminal_path = Path(response["episode_receipt"])
                if not terminal_path.resolve().is_relative_to(args.flight_receipts.resolve()):
                    raise ValueError("episode receipt escaped the planned local store")
                terminal = json.loads(terminal_path.read_text())
                plan = json.loads((terminal_path.parent.parent/"plan.json").read_text())
                check_frozen()
                if (terminal.get("status") != "completed" or plan["suite_manifest_sha256"] != suite_hash
                        or plan["case_file_sha256"] != cases_by_id[case]["file_sha256"]
                        or plan["case_sha256"] != cases_by_id[case]["case_sha256"]
                        or plan["dependency_hashes"] != dependency_hashes
                        or plan["source_hashes"] != expected_worker_sources(condition)
                        or plan["supervisor_sha256"] != frozen[str(ROOT/"eval/run_inspection_dev.py")]
                        or plan["receipt_adapter_sha256"] != frozen[str(ROOT/"eval/inspection_episode_receipts.py")]):
                    raise ValueError("receipt input/dependency differs from the original frozen pilot")
                def forbid_dispatch(_):
                    raise ValueError("post-return verification may never dispatch a new flight")
                # Shared verifier checks intent/start/stdout plus all flight hashes
                # under the same episode lock; this is an audit, not a new run.
                cache_value, cache_evidence = execute_local_episode(terminal_path.parent.parent, plan, forbid_dispatch)
                value, evidence = validate_flight(terminal_path.parent/"flight", plan, terminal["returncode"])
                if (value is None or value != response["Y"] or digest(terminal_path) != response["episode_receipt_sha256"]
                        or canonical_digest(plan) != response["plan_sha256"] or plan["case_id"] != case or plan["condition"] != condition
                        or evidence != terminal["evidence"] or cache_value != value
                        or cache_evidence["episode_receipt_sha256"] != response["episode_receipt_sha256"]):
                    raise ValueError("physical completion/source receipt binding differs")
                check_frozen()
                complete(value, {**response, **slot, "suite_sha256": suite_hash, "role": "actual_source_bound_dev_flight_or_verified_cache"})
                measurements[case, condition] = value
                print(json.dumps({**slot, "Y": value, "cache_reused": response["cache_reused"]}), flush=True)
        check_frozen()
        recorder.finish(outcome="inconclusive")
    check_frozen()
    means = {c: sum(sum(measurements[f"dev-L{level}-{i:02d}", c] for i in (0, 1))/2 for level in (1, 2, 3))/3
             for c in ("B0", "B1", "B2")}
    selected = "B1" if means["B1"] >= means["B2"]-1e-12 else "B2"
    write_once(args.out, "selection.json", json_bytes({"role": "development_selection_not_formal_result_or_G4_GO",
        "planned_episode_denominator": 18, "completed_slots": len(measurements), "root_count": 6,
        "level_macro_means": means, "classic_best": selected, "results_sha256": digest(args.out/"results.json")}))


if __name__ == "__main__":
    main()
