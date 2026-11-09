"""Exercise the reviewed 20-second wait path on one existing development root."""
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
sys.path.insert(0, str(PROJECT))

from eval.inspection_episode_receipts import digest, durable_once
from eval.run_inspection_llm_episode import execute_development_llm_episode

ORIGINAL = PROJECT / "logs/inspection-eight-condition-dev-flights.v4/dev-L3-00-B4-M1-0/plan.json"
OUT = PROJECT / "logs/inspection-final-development-delay.v1"
PREFIX = "g4-final-dev-B4-M1-delay20-v1"


def main():
    proof = RUN / "qualification/pilot-v4-complete.v1/binding.json"
    assert digest(proof) == "d51ef4f0552cd447c5c7450d8e389c345f7f12c947a1d037cbb8d22a99f6d622"
    audit = json.loads(proof.read_bytes())
    assert digest(ORIGINAL) == audit["files"][str(ORIGINAL.relative_to(PROJECT))]
    old = json.loads(ORIGINAL.read_bytes())
    assert old["case_id"] == "dev-L3-00" and old["condition"] == "B4-M1"
    assert old["repeat_ordinal"] == 0 and old["injected_delay_seconds"] == 0
    assert not OUT.exists() and not list((RUN/"resource-requests").glob(PREFIX+"*"))
    install_path = RUN / "qualification/formal-installation.v1/installation.json"
    install = json.loads(install_path.read_bytes())
    for path, row in install["members"].items():
        if not path.startswith("tests/"):
            assert digest(PROJECT/path) == row["after"]
    assert all(os.environ.get(k) == "1" for k in old["thread_environment"])
    assert digest(old["case_path"]) == old["case_file_sha256"]
    assert digest(RUN/"planning/local-model-registry.v1.json") == old["model_registry_sha256"]
    assert digest(old["evaluator"]) == old["evaluator_sha256"]
    assert all(digest(path) == pin for path, pin in old["dependency_hashes"].items())
    plan = {**old, "role": "final installed development delay integration; not formal evidence",
            "source_hashes": {name: digest(name) for name in old["source_hashes"]},
            "service_sha256": digest(PROJECT/"eval/inspection_model_service.py"),
            "injected_delay_seconds": 20, "ablation": None}
    out = RUN / "qualification/final-development-delay.v1"
    out.mkdir()
    durable_once(out/"intent.json", {"role":"one installed-development delay arm, no new root",
        "reviewed_installation_sha256":digest(install_path), "original_plan_sha256":digest(ORIGINAL),
        "reservation_prefix":PREFIX, "plan":plan})

    def command(flight, reservation):
        return [sys.executable, str(PROJECT/"eval/run_inspection_llm_development.py"),
                "--run", str(RUN), "--reservation", reservation,
                "--model", "M1", "--architecture", "B4", "--case", plan["case_path"],
                "--sampler-seed", str(plan["sampler_seed"]), "--injected-delay-seconds", "20",
                "--out", str(flight.resolve())]

    y, evidence = execute_development_llm_episode(OUT, plan, command, run=RUN, reservation_prefix=PREFIX)
    result = {"role":"actual installed-development delay measurement; not G4 acceptance",
              "Y":y, **evidence}
    durable_once(out/"result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
