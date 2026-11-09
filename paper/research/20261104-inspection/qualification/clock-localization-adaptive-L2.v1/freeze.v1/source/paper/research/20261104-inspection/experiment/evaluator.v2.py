"""Frozen offline score recount. A matching score is not episode admission.

Run in a fresh Python process. The adjacent import closure is immutable and
checked before importing. This entry never instantiates a mission or aircraft.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--expected-world-sha256", required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    closure = here/"evaluator-source.v2"
    manifest_path = here/"evaluator-source-manifest.v2.json"
    if sha(manifest_path) != "ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80":
        raise ValueError("frozen evaluator manifest differs")
    manifest = json.loads(manifest_path.read_text())
    actual = {str(p.relative_to(closure)) for p in closure.rglob("*") if p.is_file()}
    if actual != set(manifest["sources"]) or any(p.is_symlink() for p in closure.rglob("*")):
        raise ValueError("unlisted, missing or symlinked evaluator file")
    for name, expected in manifest["sources"].items():
        if sha(closure/name) != expected:
            raise ValueError("frozen evaluator import differs: "+name)
    if sha(args.world) != args.expected_world_sha256:
        raise ValueError("world differs from the caller's admitted receipt")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(closure))
    import numpy as np
    from eval.inspection_cases import CaseEntity, coverage_cells
    from eval.inspection_common_calendar import _score
    if np.__version__ != manifest["numpy_version"]:
        raise ValueError("frozen numerical evaluator dependency differs")
    world = json.loads(args.world.read_text())
    private = world["private_case"]
    identity = hashlib.sha256(json.dumps(private, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if identity != world["case_sha256"]:
        raise ValueError("worker semantic case digest differs")
    expected_cells = {k: list(v) for k, v in coverage_cells(private["roi_polygon"]).items()}
    if expected_cells != world["required_cells"]:
        raise ValueError("required cells differ from the frozen ROI/lattice rule")
    if private["entities"] != world["hidden_entities"]:
        raise ValueError("private geometry and original world differ")
    case = SimpleNamespace(level=private["level"], priority_at=private["priority_at"],
        entities=tuple(CaseEntity(**row) for row in private["entities"]))
    score = _score(world, case, case)
    if score != world["score"]:
        raise ValueError("original full score recount differs")
    print(json.dumps({"role": "offline recount, not new rollout or provenance admission",
        "world_sha256": args.expected_world_sha256, "evaluator_sha256": sha(__file__),
        "source_manifest_sha256": sha(here/"evaluator-source-manifest.v2.json"),
        "score_equal": True, "score": score}, allow_nan=False))


if __name__ == "__main__":
    main()
