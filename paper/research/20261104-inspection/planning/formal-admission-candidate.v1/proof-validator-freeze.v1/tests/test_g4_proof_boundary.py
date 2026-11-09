"""Synthetic proof-shape tests only; no real owner, model, physics or data draw."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


source = Path(__file__).resolve().parents[1]/"source/eval/inspection_formal_admission.py"
spec = importlib.util.spec_from_file_location("staged_formal_admission", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def synthetic_native_binding():
    artifacts = [module.Artifact("fixture-subject.json", "1"*64),
                 module.Artifact("fixture-review.json", "2"*64)]
    request = {"round": {"id": "synthetic-round"},
               "evidence": {a.path: a.sha256 for a in artifacts}}
    issued = module.canonical_sha(request)
    request["request_sha256"] = issued
    row = {"request": request, "validation": "structure_and_binding_only",
           "decision": {"request_sha256": issued, "status": "active", "progress": "verified_progress",
                        "finding": {"id": "synthetic-g4-finding"}}}
    return {"rounds": [row]}, artifacts, issued


def test_applied_subject_binding_shape_is_not_just_a_qualified_flag():
    owner, artifacts, issued = synthetic_native_binding()
    assert module.require_native_binding(owner, "synthetic-g4-finding", artifacts) == issued
    with pytest.raises(ValueError):
        module.require_native_binding({"status": "qualified"}, "synthetic-g4-finding", artifacts)


@pytest.mark.parametrize("fault", ["request_bytes", "decision_pointer", "evidence", "progress",
                                  "status", "validation", "duplicate", "missing"])
def test_native_round_counterexamples_rejected(fault):
    owner, artifacts, _ = synthetic_native_binding()
    row = owner["rounds"][0]
    if fault == "request_bytes":
        row["request"]["unexpected"] = True
    elif fault == "decision_pointer":
        row["decision"]["request_sha256"] = "0"*64
    elif fault == "evidence":
        artifacts[0] = module.Artifact(artifacts[0].path, "3"*64)
    elif fault == "progress":
        row["decision"]["progress"] = "stale"
    elif fault == "status":
        row["decision"]["status"] = "stopped"
    elif fault == "validation":
        row["validation"] = "claimed"
    elif fault == "duplicate":
        owner["rounds"].append(copy.deepcopy(row))
    else:
        owner["rounds"] = []
    with pytest.raises(ValueError):
        module.require_native_binding(owner, "synthetic-g4-finding", artifacts)


@pytest.mark.parametrize("value", [None, {"path": "/outside", "sha256": "1"*64},
                                  {"path": "../outside", "sha256": "1"*64},
                                  {"path": "safe", "sha256": "short"},
                                  {"path": "safe", "sha256": "1"*64, "qualified": True}])
def test_artifact_identity_rejects_ambiguous_or_escaping_inputs(value):
    with pytest.raises(ValueError):
        module.Artifact.parse(value)


def test_changed_or_symlinked_artifact_is_not_a_proof(tmp_path):
    original = tmp_path/"original.json"
    original.write_text('{"fixture":true}')
    artifact = module.Artifact(original.name, module.file_sha(original))
    assert artifact.checked_path(tmp_path) == original
    link = tmp_path/"link.json"
    link.symlink_to(original)
    with pytest.raises(ValueError):
        module.Artifact(link.name, artifact.sha256).checked_path(tmp_path)
    original.write_text('{"fixture":false}')
    with pytest.raises(ValueError):
        artifact.checked_path(tmp_path)


@pytest.mark.parametrize("text", ['{"status":"qualified","status":"qualified"}', '{"value":NaN}'])
def test_untrusted_json_does_not_hide_duplicate_or_nonfinite_fields(tmp_path, text):
    path = tmp_path/"fixture.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        module.read_json(path)


def test_unapplied_status_file_cannot_open_any_formal_data(tmp_path):
    # This temporary synthetic root contains no real .paper or dataset.
    run = tmp_path/"synthetic-research-run"
    run.mkdir()
    path = run/"claimed-qualification.json"
    path.write_text(json.dumps({"status": "qualified"}))
    with pytest.raises(ValueError, match="typed G4"):
        module.read_g4_proof(run, tmp_path, {"path": path.name, "sha256": module.file_sha(path)})
    assert not (run/"datasets").exists() and not (run/".paper").exists()


def synthetic_g4_fixture(project, fault=None, active=True):
    """Invent proof-shaped software fixtures, never actual scientific evidence."""
    run = project/"synthetic-run"
    run.mkdir()

    def artifact(name, value):
        path = run/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {"path": name, "sha256": module.file_sha(path)}

    population = artifact("population.json", {"role": "synthetic fixture, no generator"})
    design = artifact("design.json", {"split_contract": {"population": population}})
    registry = artifact("registry.json", {"role": "synthetic fixture, no actual weights"})
    plan = {"stage": "plan", "experiment_design": design["path"],
            "experiment_design_sha256": design["sha256"], "local_model_registry": registry["path"],
            "local_model_registry_sha256": registry["sha256"]}
    identity = {"created_at": "2000-01-01T00:00:00Z", "brief_sha256": "b"*64,
                "frozen_plan": module.canonical_sha(plan)}
    sources = {}
    for name in module.REQUIRED_SOURCES:
        path = project/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic test bytes, never executed as a pipeline.\n")
        sources[name] = module.file_sha(path)
    dependency = project/"synthetic-dependency.bin"
    dependency.write_bytes(b"software-fixture-only")
    if fault == "omitted_source":
        sources.pop("eval/inspection_formal_admission.py")
    closure = artifact("sources.json", {"schema": "inspection-final-source-closure-v1",
                       "sources": sources, "dependencies": {str(dependency): module.file_sha(dependency)}})
    evidence = artifact("evidence.json", {"role": "synthetic proof shape, not a measured run"})
    hardware = artifact("hardware.json", {"role": "synthetic profile, not a physical host"})
    subject_value = {"schema": "inspection-g4-subject-v1", "owner": identity,
                     "design": design, "population": population, "registry": registry,
                     "source_manifest": closure, "qualification_evidence": [evidence],
                     "hardware_profile": hardware}
    if fault == "changed_science":
        subject_value["registry"] = artifact("other-registry.json", {"role": "different synthetic model"})
    elif fault == "foreign_owner":
        subject_value["owner"] = {**identity, "created_at": "2001-01-01T00:00:00Z"}
    elif fault == "no_evidence":
        subject_value["qualification_evidence"] = []
    subject = artifact("subject.json", subject_value)
    review = artifact("review.json", {"schema": "inspection-g4-independent-review-v1",
                      "decision": "accept_declared_simulation_scope",
                      "subject_sha256": "0"*64 if fault == "foreign_review" else subject["sha256"],
                      "unresolved_blocking_findings": ["synthetic-blocker"] if fault == "open_blocker" else [],
                      "reviewer": "synthetic-software-fixture-not-a-real-reviewer"})
    acceptance = artifact("acceptance.json", {"schema": "inspection-g4-acceptance-v1",
                          "status": "qualified", "subject": subject, "review": review,
                          "native_finding_id": "synthetic-finding"})
    request = {"round": {"id": "synthetic-round"},
               "evidence": {a["path"]: a["sha256"] for a in (acceptance, subject, review)}}
    issued = module.canonical_sha(request)
    request["request_sha256"] = issued
    owner = {**identity, "receipts": [{"stage": "plan", "packet": plan}], "task_stop": None,
             "active_round": {"id": "synthetic-active", "stage": "experiment"} if active else None,
             "rounds": [{"request": request, "validation": "structure_and_binding_only",
                         "decision": {"request_sha256": issued, "status": "active",
                                      "progress": "verified_progress", "finding": {"id": "synthetic-finding"}}}]}
    path = run/".paper/control.json"
    path.parent.mkdir()
    path.write_text(json.dumps(owner))
    return run, acceptance


def test_complete_synthetic_proof_shape_loads_without_opening_data(tmp_path):
    run, acceptance = synthetic_g4_fixture(tmp_path)
    proof = module.read_g4_proof(run, tmp_path, acceptance)
    assert isinstance(proof, module.G4Proof) and proof.acceptance.sha256 == acceptance["sha256"]
    assert not (run/"datasets").exists()


@pytest.mark.parametrize("fault", ["omitted_source", "changed_science", "foreign_owner", "no_evidence",
                                  "foreign_review", "open_blocker"])
def test_applied_qualified_status_still_cannot_override_scientific_or_source_subject(tmp_path, fault):
    run, acceptance = synthetic_g4_fixture(tmp_path, fault=fault)
    with pytest.raises(ValueError):
        module.read_g4_proof(run, tmp_path, acceptance)
    assert not (run/"datasets").exists()


def test_offline_proof_read_is_not_live_dispatch_authority(tmp_path):
    run, acceptance = synthetic_g4_fixture(tmp_path, active=False)
    with pytest.raises(ValueError, match="live dispatch"):
        module.read_g4_proof(run, tmp_path, acceptance)
    assert module.read_g4_proof(run, tmp_path, acceptance, for_dispatch=False)
    assert not (run/"datasets").exists()
