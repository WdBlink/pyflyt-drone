"""Prospective geometry generation only; no physical or formal instances."""

from dataclasses import replace

import pytest

from eval.inspection_population import (
    PopulationCase, PopulationGenerationExhausted, generate_development_population_case,
    load_development_case, validate_population_case,
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


def test_population_input_roundtrip_keeps_ood_motion_and_rejects_namespace_or_extra_fields():
    case, _ = generate_development_population_case("devext-reader-fixture", "OOD", "L2", 2026110604)
    import json
    value = json.loads(json.dumps(case.as_dict()))
    restored = load_development_case(value)
    assert isinstance(restored, PopulationCase) and restored == case and restored.digest() == case.digest()
    for change in ({"split": "main"}, {"case_id": "holdout-OOD-L2-00"}, {"undeclared": 1}):
        with pytest.raises(ValueError):
            load_development_case({**value, **change})


def test_collision_digest_ignores_entity_handles_namespace_and_sensor_identity():
    from eval.prepare_inspection_development_population import configuration_digest
    case, _ = generate_development_population_case("devext-reader-fixture", "ID", "L1", 2026110604)
    changed = replace(case, case_id="devext-other", lineage_group="devext-other", seed=case.seed+1,
        entities=tuple(replace(e, key="new-"+e.key) for e in reversed(case.entities)))
    assert case.digest() != changed.digest() and configuration_digest(case) == configuration_digest(changed)


@pytest.mark.parametrize("fault", ["empty_key", "bool_key", "bool_wind", "outside_ID_support"])
def test_reviewed_population_schema_counterexamples(fault):
    case, _ = generate_development_population_case("devext-schema-fixture", "ID", "L1", 2026110604)
    if fault == "bool_wind":
        case = replace(case, wind_xy=(True, 0.))
    else:
        entity = next(e for e in case.entities if e.true_class == "road_obstruction")
        change = ({"key": ""} if fault == "empty_key" else {"key": True} if fault == "bool_key"
            else {"origin_xy": (300., -200.)})
        case = replace(case, entities=tuple(replace(e, **change) if e is entity else e for e in case.entities))
    with pytest.raises(ValueError):
        validate_population_case(case)


def test_clean_intervention_roundtrip_preserves_public_level_and_entire_root():
    import json
    from eval.inspection_population import EVENT_FIELDS, EventCleanSibling, event_clean_sibling, validate_development_case
    parent, _ = generate_development_population_case("devext-clean-fixture", "OOD", "L3", 2026110604)
    clean = event_clean_sibling(parent)
    loaded = load_development_case(json.loads(json.dumps(clean.as_dict())))
    assert isinstance(loaded, EventCleanSibling) and loaded == clean
    assert loaded.level == parent.level == "L3"
    assert loaded.parent_sha256 == parent.digest()
    assert all(getattr(loaded, k) is None for k in EVENT_FIELDS)
    for key in PopulationCase.__dataclass_fields__:
        if key not in (*EVENT_FIELDS, "schema"):
            assert getattr(loaded, key) == getattr(parent, key)
    assert loaded.entities[0].position(0) == parent.entities[0].position(0)
    assert validate_development_case(loaded)["parent_sha256"] == parent.digest()
    assert load_development_case(json.loads(json.dumps(parent.as_dict()))) == parent


@pytest.mark.parametrize("fault", ["level", "seed", "wind", "calendar", "parent_digest", "parent", "extra"])
def test_clean_intervention_cannot_change_any_other_factor(fault):
    import copy
    from eval.inspection_population import event_clean_sibling
    parent, _ = generate_development_population_case("devext-clean-fixture", "ID", "L3", 2026110604)
    value = copy.deepcopy(event_clean_sibling(parent).as_dict())
    if fault == "level":
        value["level"] = "L2"
    elif fault == "seed":
        value["seed"] += 1
    elif fault == "wind":
        value["wind_xy"] = (0., 0.)
    elif fault == "calendar":
        value["retire_at"] = parent.retire_at
    elif fault == "parent_digest":
        value["parent_sha256"] = "0"*64
    elif fault == "parent":
        value["parent_case"]["level"] = "L2"
    else:
        value["extra"] = True
    with pytest.raises(ValueError):
        load_development_case(value)


def test_clean_intervention_does_not_admit_an_ordinary_l3_without_events():
    from eval.inspection_population import EVENT_FIELDS, event_clean_sibling
    parent, _ = generate_development_population_case("devext-clean-fixture", "ID", "L3", 2026110604)
    with pytest.raises(ValueError):
        load_development_case({**parent.as_dict(), **dict.fromkeys(EVENT_FIELDS)})
    low, _ = generate_development_population_case("devext-clean-low-fixture", "ID", "L2", 2026110604)
    with pytest.raises(ValueError):
        event_clean_sibling(low)
