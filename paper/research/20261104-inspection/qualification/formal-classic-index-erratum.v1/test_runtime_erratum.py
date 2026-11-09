"""Isolated software counterexamples; never creates a real formal effect."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_formal_dispatch import helper, module as dispatch

execution = helper.module
admission = helper.helper.helper.admission


def fixture(project, monkeypatch):
    f = helper.helper.DatasetFixture(project)
    selection = f.put("qualification/synthetic-selection.json", {
        "completed_slots": 18, "root_count": 6, "classic_best": "B1"})
    subject = f.read(f.subject)
    assert "classic_selection" not in subject
    subject["qualification_evidence"].append(selection)
    f.subject = f.put(f.subject["path"], subject)
    f.review = f.put(f.review["path"], {**f.read(f.review), "subject_sha256": f.subject["sha256"]})
    base = f.read(f.acceptance)
    f.acceptance = f.put(f.acceptance["path"], {**base, "subject": f.subject, "review": f.review})
    f.owner["rounds"] = []
    f.bind(base["native_finding_id"], [f.acceptance, f.subject, f.review])
    for module in (admission, execution):
        monkeypatch.setattr(module, "CLASSIC_INDEX_G4", admission.Artifact.parse(f.acceptance))
        monkeypatch.setattr(module, "CLASSIC_INDEX_SELECTION", admission.Artifact.parse(selection))
    intent, dataset, review = f.dataset()
    changes = {}
    after_bytes = {}
    folder = admission._CLASSIC_INDEX_DIR
    for name in admission._CLASSIC_INDEX_FILES:
        original = (project/name).read_bytes()
        after_bytes[name] = original + b"\n# synthetic replacement\n"
        changes[name] = {}
        for field, suffix, data in (("before", "preimage", original),
                                    ("after", "source", after_bytes[name])):
            path = f.run/(folder + suffix + "/" + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            changes[name][field] = {"path": str(path.relative_to(f.run)), "sha256": admission.file_sha(path)}
    subject = {"schema": admission.CLASSIC_INDEX_FINDING, "g4_acceptance": f.acceptance,
               "source_manifest": f.source, "classic_selection": selection,
               "main_namespace": intent, "main_dataset": dataset, "changes": changes}
    return f, subject, review, after_bytes


def bind(f, subject, fault=None):
    folder = admission._CLASSIC_INDEX_DIR
    manifest = f.put(folder+"subject.json", subject)
    review = f.put(folder+"independent-review.json", {
        "schema": "inspection-classic-index-erratum-review-v1",
        "decision": "accept_exact_runtime_erratum", "subject_sha256": manifest["sha256"],
        "reviewer": "synthetic fixture only", "unresolved_blocking_findings": []})
    refs = [manifest, review, *[subject[name] for name in
            ("g4_acceptance", "source_manifest", "classic_selection", "main_namespace", "main_dataset")]]
    refs.extend(ref for row in subject["changes"].values() for ref in row.values())
    if fault == "unbound_postimage":
        refs.pop()
    f.bind(admission.CLASSIC_INDEX_FINDING, refs)
    return manifest


def test_same_effective_source_pins_in_g4_and_dispatch(tmp_path, monkeypatch):
    f, subject, review, after = fixture(tmp_path, monkeypatch)
    erratum = bind(f, subject)
    for name, data in after.items():
        (tmp_path/name).write_bytes(data)
    proof = admission.read_g4_proof(f.run, f.project, f.acceptance, verification=f.verification)
    assert asdict(proof.runtime_erratum) == erratum
    assert asdict(proof.source_manifest) == f.source
    namespace = SimpleNamespace(g4_acceptance=proof.acceptance, source_manifest=proof.source_manifest)
    rec = SimpleNamespace(dataset=SimpleNamespace(namespace=namespace), owner_snapshot=f.owner)
    packet = dispatch.DispatchAdmission(None, SimpleNamespace(execution=rec), {})
    graph = packet.frozen_sources(f.run, tmp_path)
    assert all(graph[str(tmp_path/name)].encode("utf-8") == data for name, data in after.items())
    assert len(graph) == len(f.read(f.source)["sources"])
    chosen, best, registry = execution._selection(f.run, rec.dataset)
    assert best == "B1" and asdict(chosen) == subject["classic_selection"]
    unmodified_name = next(name for name in f.read(f.source)["sources"] if name not in after)
    (tmp_path/unmodified_name).write_text("unapproved current source change")
    with pytest.raises(ValueError):
        packet.frozen_sources(f.run, tmp_path)


@pytest.mark.parametrize("fault", ["no_finding", "mixed_install", "stale_review",
    "revoked", "unbound_postimage", "wrong_preimage", "extra_source", "too_late"])
def test_repair_cannot_bypass_exact_parent_review_or_code_pins(tmp_path, monkeypatch, fault):
    f, subject, review, after = fixture(tmp_path, monkeypatch)
    name = next(iter(after))
    if fault == "wrong_preimage":
        subject["changes"][name]["before"]["sha256"] = "0" * 64
    if fault == "extra_source":
        subject["changes"]["envs/inspection/mission.py"] = subject["changes"][name]
    if fault == "too_late":
        f.bind("inspection-main-primary-execution-v1", [subject["main_dataset"]])
    if fault != "no_finding":
        bind(f, subject, fault=fault)
    if fault == "revoked":
        f.bind("synthetic-revocation", [f.source], revises=admission.CLASSIC_INDEX_FINDING)
    if fault == "stale_review":
        p = f.run/admission._CLASSIC_INDEX_DIR/"independent-review.json"
        p.write_bytes(p.read_bytes() + b" ")
    for candidate, data in after.items():
        if fault != "mixed_install" or candidate != name:
            (tmp_path/candidate).write_bytes(data)
    with pytest.raises(ValueError):
        admission.read_g4_proof(f.run, f.project, f.acceptance, verification=f.verification)
