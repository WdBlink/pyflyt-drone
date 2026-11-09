"""Run one frozen classical development episode through durable receipts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.inspection_episode_receipts import digest, execute_local_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--policy", choices=("B0", "B1", "B2"), required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.suite/"manifest.json").read_text())
    if manifest.get("split") != "dev" or manifest.get("main_and_holdout_instantiated") is not False:
        parser.error("only the frozen development namespace is allowed")
    matches = [row for row in manifest["records"] if row["case_id"] == args.case_id]
    if len(matches) != 1:
        parser.error("case is missing or duplicated in the frozen suite")
    row = matches[0]
    case_path = (args.suite/row["file"]).resolve()
    if case_path.parent != args.suite.resolve() or digest(case_path) != row["file_sha256"]:
        parser.error("case path or frozen bytes changed")
    sources = [*sorted((ROOT/"envs/inspection").glob("*.py")), ROOT/"envs/__init__.py",
               ROOT/"eval/run_inspection_reference.py", ROOT/"eval/inspection_cases.py",
               ROOT/"eval/inspection_scenarios.py", ROOT/"eval/inspection_population.py",
               ROOT/"eval/inspection_worker_runtime.py"]
    if args.policy != "B0":
        sources.append(ROOT/"eval/inspection_classic.py")
    package = Path(importlib.util.find_spec("PyFlyt").origin).parent
    dependencies = [p for p in package.rglob("*") if p.suffix.lower() in
                    (".py", ".yaml", ".urdf", ".obj", ".stl", ".dae")]
    dependencies.append(Path(importlib.util.find_spec("pybullet").origin))
    thread_names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    for name in thread_names:
        os.environ[name] = "1"
    plan = {"role": "development_not_formal_or_G4_acceptance", "case_id": args.case_id,
            "lineage_group": row["lineage_group"], "condition": args.policy, "repeat_ordinal": 0,
            "case_file_sha256": row["file_sha256"], "case_sha256": row["case_sha256"],
            "suite_manifest_sha256": digest(args.suite/"manifest.json"), "max_attempts": 2,
            "max_clock_lag_seconds": .1, "source_hashes": {str(p): digest(p) for p in sources},
            "dependency_hashes": {str(p): digest(p) for p in dependencies},
            "thread_environment": {name: "1" for name in thread_names},
            "supervisor_sha256": digest(Path(__file__)),
            "receipt_adapter_sha256": digest(ROOT/"eval/inspection_episode_receipts.py")}
    def command(out):
        return [sys.executable, str(ROOT/"eval/run_inspection_reference.py"), "--case", str(case_path),
                "--policy", args.policy, "--realtime", "--duration", "600", "--out", str(out.resolve())]
    value, evidence = execute_local_episode(args.receipts/f"{args.case_id}-{args.policy}-0", plan, command)
    print(json.dumps({"Y": value, **evidence}, allow_nan=False))


if __name__ == "__main__":
    main()
