"""Pure geometry/software counterexamples; no physics, model or formal roots."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from dataclasses import replace
from fractions import Fraction

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent


def module(name, filename, expected):
    path = HERE / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


joint = module("review_joint_population", "inspection_population.joint-999.py",
    "999b5a97ae49d55f9e58136c05f631be4a720429249dcbdde00300e351c467fc")
historical = module("review_historical_population", "inspection_population.local-rejection.reconstructed.py",
    "516ef909bac05145a397d86135da31eb78b9cae5f006f63fa646c4bc25239add")

args = ("devext-independent-validator-fixture", "ID", "L1", 2026110604)
case, audit = joint.generate_development_population_case(*args)
old_case, old_audit = historical.generate_development_population_case(*args)

mutations = [
    ("empty_hidden_key", replace(case,
        entities=(replace(case.entities[0], key=""), *case.entities[1:]))),
    ("boolean_hidden_key", replace(case,
        entities=(replace(case.entities[0], key=True), *case.entities[1:]))),
    ("boolean_wind", replace(case, wind_xy=(True, 0.0))),
]
target = next(e for e in case.entities if e.true_class == joint.CLASSES[0])
mutations.append(("ID_static_target_x300_outside_frozen_support", replace(case,
    entities=tuple(replace(e, origin_xy=(300.0, -200.0)) if e is target else e
                   for e in case.entities))))
actual = []
for name, changed in mutations:
    try:
        joint.validate_population_case(changed)
        actual.append({"mutation": name, "accepted": True})
    except Exception as error:
        actual.append({"mutation": name, "accepted": False,
            "error": type(error).__name__ + ": " + str(error)})

# The exact finite-K version of the source's sequential placement algorithm.
# Draw two labelled positions independently from {0,1,3}; require distance>=2.
# Joint conditioning gives each of four allowed ordered pairs probability1/4.
# Sequential local rejection changes the feasible-volume normalizer with X.
K = 1000
A1 = 1 - Fraction(2, 3) ** K
A2 = 1 - Fraction(1, 3) ** K
denominator = 2 * A1 + A2
p_one_neighbour = A1 / denominator
p_two_neighbours = A2 / (2 * denominator)
assert 2 * p_one_neighbour + 2 * p_two_neighbours == 1
assert p_one_neighbour != Fraction(1, 4)
assert p_two_neighbours != Fraction(1, 4)

result = {
    "role": "independent pure geometry fixtures, not scientific roots or flights",
    "validator_source": "joint-proposal repaired SHA999b; validator not repaired",
    "validator_mutations": actual,
    "historical_source_is_reverse_place_reconstruction": True,
    "same_input_diagnostic": {
        "historical_whole_candidates": len(old_audit["attempts"]),
        "joint_whole_candidates": len(audit["attempts"]),
        "historical_max_local_candidate_index": max(p["candidate_index"]
            for row in old_audit["attempts"] for p in row["placements"]),
        "joint_max_local_candidate_index": max(p["candidate_index"]
            for row in audit["attempts"] for p in row["placements"]),
        "same_case": old_case.as_dict() == case.as_dict(),
    },
    "finite_discrete_proof": {
        "positions": [0, 1, 3], "minimum_separation": 2, "local_cap": K,
        "joint_each_allowed_ordered_pair": "1/4",
        "A1": "1-(2/3)^1000", "A2": "1-(1/3)^1000",
        "pairs_03_and_13": {"formula": "A1/(2*A1+A2)",
            "probability": float(p_one_neighbour), "exactly_differs_from_one_quarter": True},
        "pairs_30_and_31": {"formula": "A2/(2*(2*A1+A2))",
            "probability": float(p_two_neighbours), "exactly_differs_from_one_quarter": True},
        "scope": "exact algorithm counterexample, not fitted frequencies of the continuous benchmark",
    },
    "pyflyt_or_pybullet_imported": any(k.startswith(("PyFlyt", "pybullet")) for k in sys.modules),
}
(HERE / "counterexamples.reproduce.result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
