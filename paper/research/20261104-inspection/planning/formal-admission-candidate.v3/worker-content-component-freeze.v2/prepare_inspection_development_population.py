"""Create five fresh development-only strata, preserving pre-draw seed intent.

This entry creates no main/holdout instances and never retries an exhausted
namespace. Prior suites are collision inputs, not templates to perturb.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.inspection_episode_receipts import durable_once
from eval.inspection_population import PopulationGenerationExhausted, generate_development_population_case, load_development_case

STRATA = (("ID", "L1"), ("ID", "L2"), ("ID", "L3"), ("OOD", "L2"), ("OOD", "L3"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_digest(case):
    value = asdict(case)
    for key in ("schema", "case_id", "lineage_group", "split", "seed", "domain"):
        value.pop(key, None)
    entities = [{k: v for k, v in e.items() if k != "key"} for e in value["entities"]]
    value["entities"] = sorted(entities, key=lambda e: json.dumps(e, sort_keys=True))
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prior_identities(suites):
    seeds, semantics, configurations, bindings = set(), set(), set(), []
    for suite in suites:
        suite = Path(suite).resolve()
        manifest = json.loads((suite/"manifest.json").read_text())
        if manifest.get("split") != "dev":
            raise ValueError("this preparation only reads declared development suites")
        bindings.append({"manifest": str(suite/"manifest.json"), "sha256": sha(suite/"manifest.json")})
        for row in manifest["records"]:
            path = (suite/row["file"]).resolve()
            if path.parent != suite or sha(path) != row["file_sha256"]:
                raise ValueError("prior case path or bytes differ")
            case = load_development_case(json.loads(path.read_text()))
            if case.digest() != row["case_sha256"]:
                raise ValueError("prior semantic digest differs")
            seeds.add(case.seed)
            if "root_generation_seed" in row:
                seeds.add(row["root_generation_seed"])
            semantics.add(case.digest())
            configurations.add(configuration_digest(case))
    return seeds, semantics, configurations, bindings


def prepare(out, prior_suites):
    if np.__version__ != "1.26.4":
        raise ValueError("the frozen population requires NumPy1.26.4")
    out = Path(out)
    if out.exists() or not prior_suites:
        raise ValueError("fresh output and explicit prior development collision inputs required")
    seeds, semantics, configurations, bindings = prior_identities(prior_suites)
    master = secrets.randbits(128)
    children = np.random.SeedSequence(master).spawn(len(STRATA))
    roots = [int.from_bytes(child.generate_state(4).tobytes(), "little") for child in children]
    out.mkdir(parents=True)
    sources = {str(p): sha(p) for p in (Path(__file__).resolve(), ROOT/"eval/inspection_population.py",
        ROOT/"eval/inspection_cases.py", ROOT/"envs/inspection/scenario.py")}
    durable_once(out/"namespace-intent.json", {"role": "fresh development seed intent before any draw, not G4",
        "master_seed": master, "root_generation_seeds": roots, "strata": STRATA,
        "rng": "NumPy1.26.4 PCG64/SeedSequence", "byte_order": "little",
        "source_hashes": sources, "prior_development_manifests": bindings,
        "main_and_holdout_instantiated": False})
    records = []
    try:
        if len(set(roots)) != len(roots) or master in seeds or set(roots) & (seeds | {master}):
            raise ValueError("namespace/root seed collision; no automatic replacement")
        seeds.update([master, *roots])
        for (domain, level), seed in zip(STRATA, roots):
            case_id = f"devext-{domain}-{level}-00"
            try:
                case, audit = generate_development_population_case(case_id, domain, level, seed)
            except PopulationGenerationExhausted as error:
                durable_once(out/(case_id+"-exhaustion.json"), {"root_generation_seed": seed, "attempts": error.audit})
                raise
            durable_once(out/(case_id+"-generation.json"), audit)
            configuration = configuration_digest(case)
            if case.seed in seeds or case.digest() in semantics or configuration in configurations:
                raise ValueError("root/sensor/configuration collision; no automatic replacement")
            seeds.add(case.seed); semantics.add(case.digest()); configurations.add(configuration)
            filename = case_id+".json"
            durable_once(out/filename, case.as_dict())
            records.append({"case_id": case_id, "lineage_group": case.lineage_group, "domain": domain,
                "level": level, "file": filename, "file_sha256": sha(out/filename), "case_sha256": case.digest(),
                "configuration_sha256": configuration, "root_generation_seed": seed, "sensor_seed": case.seed,
                "generation_audit": case_id+"-generation.json", "generation_audit_sha256": sha(out/(case_id+"-generation.json"))})
        if any(sha(p) != h for p, h in sources.items()) or any(sha(b["manifest"]) != b["sha256"] for b in bindings):
            raise ValueError("source/prior manifest changed during generation")
        durable_once(out/"manifest.json", {"role": "five fresh development population strata; not main/holdout or pilot replacement",
            "split": "dev", "main_and_holdout_instantiated": False, "first_pilot": False,
            "namespace_intent_sha256": sha(out/"namespace-intent.json"), "records": records})
    except BaseException as error:
        durable_once(out/"failure.json", {"status": "namespace_blocked_no_automatic_reseed",
            "error": type(error).__name__+": "+str(error), "completed_cases": len(records)})
        raise
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-suite", type=Path, action="append", required=True)
    args = parser.parse_args()
    records = prepare(args.out, args.prior_suite)
    print(json.dumps({"status": "development_geometry_only", "roots": len(records),
        "manifest_sha256": sha(args.out/"manifest.json")}, allow_nan=False))


if __name__ == "__main__":
    main()
