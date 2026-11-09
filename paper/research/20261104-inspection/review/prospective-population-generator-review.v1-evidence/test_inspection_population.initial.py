"""Prospective geometry generation only; no physical or formal instances."""

from dataclasses import replace

import pytest

from eval.inspection_population import (
    PopulationGenerationExhausted, generate_development_population_case, validate_population_case,
)


@pytest.mark.parametrize("domain,level", [("ID", "L1"), ("ID", "L3"), ("OOD", "L2"), ("OOD", "L3")])
def test_population_diagnostics_are_reproducible_and_keep_separate_sensor_channel(domain, level):
    # Same seed across diagnostic parametrizations is deliberate and is not a
    # claim of independent scientific roots or a newly frozen real dev suite.
    args = (f"devext-fixture-{domain}-{level}", domain, level, 2026110604)
    a, audit = generate_development_population_case(*args)
    b, second = generate_development_population_case(*args)
    assert a == b and audit == second
    assert a.split == "dev" and a.domain == domain and a.seed == audit["sensor_seed"]
    assert a.seed != audit["root_generation_seed"]
    assert audit["attempts"][-1]["status"] == "accepted"
    assert all(row["status"] == "rejected" for row in audit["attempts"][:-1])
    assert validate_population_case(a)["case_sha256"] == a.digest()
    assert all(e.bend == (0 if domain == "ID" else 1) for e in a.entities if e.object_type == "dynamic")


@pytest.mark.parametrize("args", [("main-ID-L1-00", "ID", "L1", 1),
    ("holdout-ID-L2-00", "ID", "L2", 1), ("devext-fixture", "OOD", "L1", 1),
    ("devext-fixture", "ID", "L1", True)])
def test_formal_namespaces_and_invalid_strata_are_not_generated(args):
    with pytest.raises(ValueError):
        generate_development_population_case(*args)


def test_45m_identity_guard_rejects_dynamic_static_near_miss_and_bad_calendar():
    case, _ = generate_development_population_case("devext-guard-fixture", "ID", "L3", 2026110604)
    moving = next(e for e in case.entities if e.object_type == "dynamic")
    point = moving.position(moving.appeared_at)
    normal = next(e for e in case.entities if e.true_class == "normal")
    changed = replace(normal, origin_xy=(point[0], point[1]+30))
    with pytest.raises(ValueError):
        validate_population_case(replace(case, entities=tuple(changed if e is normal else e for e in case.entities)))
    with pytest.raises(ValueError, match="event"):
        validate_population_case(replace(case, link_end=case.link_start+100))


def test_finite_exhaustion_retains_every_candidate_without_a_new_root(monkeypatch):
    import eval.inspection_population as module
    def reject(case):
        raise ValueError("deliberate geometry fixture rejection, no scores")
    monkeypatch.setattr(module, "validate_population_case", reject)
    with pytest.raises(PopulationGenerationExhausted) as caught:
        generate_development_population_case("devext-exhaustion-fixture", "ID", "L1", 5)
    assert [row["candidate_index"] for row in caught.value.audit] == list(range(1000))
    assert all(row["status"] == "rejected" for row in caught.value.audit)
