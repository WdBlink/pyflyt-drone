"""Actual secondary SDK, literal scores and Fake physical/native-shaped inputs."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / "eval"
spec = importlib.util.spec_from_file_location("secondary_analysis_runner_helpers", HERE / "test_formal_secondary_runner.py")
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
load = helper.helper.load
load("eval.inspection_statistics", STAGE / "inspection_statistics.py")
load("eval.inspection_secondary_statistics", STAGE / "inspection_secondary_statistics.py")
load("eval.inspection_intervention_statistics", STAGE / "inspection_intervention_statistics.py")
load("eval.inspection_common_calendar", STAGE / "inspection_common_calendar.py")
load("eval.inspection_event_endpoints", STAGE / "inspection_event_endpoints.py")
load("eval.inspection_response_diagnostics", STAGE / "inspection_response_diagnostics.py")
primary_analysis = load("eval.analyze_inspection_formal", STAGE / "analyze_inspection_formal.py")
analysis = load("secondary_analysis_candidate", STAGE / "analyze_inspection_formal_secondary.py")


@pytest.fixture(autouse=True)
def matching_types(monkeypatch):
    monkeypatch.setitem(sys.modules, "eval.inspection_population", helper.helper.shared)
    monkeypatch.setitem(sys.modules, "eval.inspection_formal_intervention", helper.helper.intervention)
    monkeypatch.setitem(sys.modules, "eval.inspection_formal_secondary", helper.helper.module)
    monkeypatch.setitem(sys.modules, "eval.inspection_formal_dispatch", helper.dispatch)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def prepared_recording(tmp_path, monkeypatch, track):
    f, refs, execution, secondary, proof = helper.prepared(tmp_path, monkeypatch, track)
    request = {"schema": "inspection-secondary-run-v1", "run": str(f.run),
        "runtime": str(f.project / "runtime"), "sdk": str(helper.SDK), "g4": f.acceptance,
        "namespace": refs[0], "dataset": refs[1], "dataset_review": refs[2], "execution": execution,
        "secondary_execution": secondary, "opening": None}
    primary_slots = {(s.case_id, s.condition, s.repeat_ordinal): s for s in proof.primary.slots}
    secondary_slots = {(s.case_id, s.condition, s.repeat_ordinal, s.arm): s for s in proof.slots}

    def measure(directory, plan):
        y = plan["repeat_ordinal"] / 4
        score = {key: y for key in primary_analysis.SCORE_FIELDS}
        score.update(H=1, E=1, SR=0, dynamic_entities=[{"bins": [{"quality": y}]}])
        if primary_slots[plan["case_id"], plan["condition"], plan["repeat_ordinal"]].level == 1:
            score["D"] = None
        return y, {"role": "Fake physical validation only; no experiment result", "score": score}

    def plan_for(slot):
        return {key: getattr(slot, key) for key in ("case_id", "condition", "repeat_ordinal", "sampler_seed")} | (
            {"track": slot.track, "arm": slot.arm} if hasattr(slot, "track") else {})

    def create_receipt(slot, directory):
        plan = plan_for(slot); write(directory / "plan.json", plan)
        _, evidence = measure(directory, plan)
        write(directory / "attempt-00/flight/world.json", {"role": "synthetic file, no physics", "score": evidence["score"]})
        write(directory / "attempt-00/terminal.json", {"status": "completed"})
        if slot.sampler_seed is not None:
            kind = getattr(slot, "track", "main")
            reservation = f"formal-{kind}-{request['g4']['sha256'][:12]}-{slot.order_index:04d}-a0"
            write(f.project / "logs/inspection-model-resource-operations.v1" / reservation / "service-completion.json",
                {"role": "literal fixture, no model operation"})
            write(f.run / "qualification/local-model-episodes" / reservation / "usage-receipt.json",
                {"role": "literal fixture, no native usage", "usage": {"calls": 0, "tokens": 0}})
            for suffix in (".reserve.json", ".settle.json"):
                write(f.run / "resource-requests" / (reservation + suffix), {"role": "literal fixture, no native operation"})

    admitted = []
    needed = {slot.parent_order_index for slot in proof.slots}
    for slot in proof.primary.slots:
        directory = f.project / "logs/inspection-formal-episodes.v1/main" / f"{slot.case_id}-{slot.condition}-{slot.repeat_ordinal}"
        _, evidence = measure(directory, plan_for(slot))
        score = evidence["score"]
        admitted.append({"case_id": slot.case_id, "domain": slot.domain, "level": slot.level,
            "condition": slot.condition, "repeat_ordinal": slot.repeat_ordinal, "status": "completed", "SR": score["SR"],
            "score": primary_analysis.scalar_score(score, slot.level, score["Y"]),
            "full_score_sha256": analysis.canonical_sha(score), "source_episode": str(directory.relative_to(f.project))})
        if slot.order_index in needed:
            create_receipt(slot, directory)
    scores = f.put("analysis/main.v1/admitted-episodes.json", admitted)
    evidence = f.put("analysis/main.v1/synthetic-other-evidence.json", {"role": "software fixture, not a measured analysis"})
    source_subject = f.read(f.subject)
    subject = f.put("analysis/main.v1/subject.json", {"schema": "inspection-main-primary-analysis-subject-v1", "phase": "main",
        "root_denominator": 72, "actual_episode_denominator": 1296, "root_mean_denominator": 576, "confirmatory_family_size": 6,
        "design": source_subject["design"], "source_manifest": f.source, "main_dataset": refs[1],
        "admitted_episode_scores": scores, **dict.fromkeys(("recording_results", "primary_statistics", "all_model_settlements",
        "component_summary", "sr_statistics", "recording_index"), evidence)})
    review = f.put("analysis/main.v1/synthetic-review.json", {"schema": "inspection-main-primary-analysis-review-v1",
        "decision": "accept_main_primary_analysis_lock", "subject_sha256": subject["sha256"],
        "unresolved_blocking_findings": [], "reviewer": "literal software fixture, not actual scientific review"})
    finding = analysis.read_primary_analysis_lock.__globals__["MAIN_LOCK_FINDING"]
    lock = f.put("analysis/main-primary-lock.v1.json", {"schema": "inspection-main-primary-analysis-lock-v1",
        "status": "locked", "subject": subject, "review": review, "native_finding_id": finding})
    f.bind(finding, [lock, subject, review])
    proof = replace(proof, primary=replace(proof.primary, owner_snapshot=f.owner), owner_snapshot=f.owner)
    monkeypatch.setattr(analysis, "admitted_request", lambda *args: (f.verification, proof.primary))
    monkeypatch.setattr(helper.driver, "admitted_request", lambda *args: (f.verification, proof.primary))
    monkeypatch.setattr(helper.driver, "read_secondary_execution", lambda *args, **kwargs: proof)
    monkeypatch.setattr(primary_analysis, "operation_directory", lambda reservation:
        f.project / "logs/inspection-model-resource-operations.v1" / reservation)
    pending = {}
    def packet(run, project, g4, namespace, dataset, dataset_review, execution, secondary,
               case, condition, repeat, track, arm, **kwargs):
        pending["slot"] = secondary_slots[case, condition, repeat, arm]
        return {"path": "literal-software-packet", "sha256": "a" * 64}
    monkeypatch.setattr(helper.driver, "write_secondary_dispatch", packet)
    monkeypatch.setattr(helper.driver, "admit_dispatch", lambda *args: SimpleNamespace(episode=SimpleNamespace(slot=pending["slot"])))
    monkeypatch.setattr(helper.driver, "episode_plan", lambda admitted, *args: plan_for(admitted.episode.slot))
    def execute(directory, plan, command, **kwargs):
        create_receipt(pending["slot"], directory)
    monkeypatch.setattr(helper.driver, "execute_local_episode", execute)
    monkeypatch.setattr(helper.driver, "execute_development_llm_episode", execute)
    monkeypatch.setattr(helper.driver, "measure_completed_episode", measure)
    monkeypatch.setattr(analysis, "measure_completed_episode", measure)
    def admit(plan, project):
        slot = (secondary_slots[plan["case_id"], plan["condition"], plan["repeat_ordinal"], plan["arm"]]
                if plan.get("track") else primary_slots[plan["case_id"], plan["condition"], plan["repeat_ordinal"]])
        binding = {"phase": "main", "execution": secondary if plan.get("track") else execution,
            **{key: getattr(slot, key) for key in ("case_id", "condition", "repeat_ordinal", "sampler_seed", "order_index")}}
        if plan.get("track"):
            binding.update(track=track, arm=slot.arm, parent_order_index=slot.parent_order_index)
        return SimpleNamespace(binding=lambda: binding)
    monkeypatch.setattr(analysis, "admit_episode_plan", admit)
    monkeypatch.setattr(analysis, "decode_formal_population_case", lambda value, phase, case: SimpleNamespace(case_id=case))
    monkeypatch.setattr(analysis, "rescore_formal_common_calendar", lambda world, parent, sha:
        {"score": world["score"], "role": "Fake common-calendar join; original helper separately tested"})
    monkeypatch.setattr(analysis, "analyze_formal_event_trace", lambda *args:
        {"rows": [], "role": "Fake event connector; adoption/recovery independently checked in endpoint tests"})
    monkeypatch.setattr(analysis, "analyze_interventions", lambda track, rows: {
        "role": "software connector stub; numerical engine tested separately", "track": track, "input_rows": len(rows)})
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.setenv(key, "1")
    recording = f.project / "logs/secondary-analysis-sdk"
    helper.driver.run_secondary(request, recording, project_root=f.project)
    return f, request, proof, recording, measure


@pytest.mark.parametrize("track,additional,parents,means", [("clean", 216, 216, 96), ("ablation", 108, 36, 36), ("latency", 180, 36, 60)])
def test_actual_secondary_sdk_and_locked_primary_subset_are_recounted(tmp_path, monkeypatch, track, additional, parents, means):
    f, request, proof, recording, measure = prepared_recording(tmp_path, monkeypatch, track)
    subject = analysis.analyze_secondary(request, recording, project_root=f.project)
    assert subject["additional_actual_episode_denominator"] == additional
    assert subject["referenced_primary_episode_denominator"] == parents
    rows = f.read(subject["admitted_episode_scores"])
    assert len(rows) == additional + parents and all(row["level"] == 3 for row in rows)
    assert all(("common_score" in row) == (track == "clean") for row in rows)
    statistics = f.read(subject["statistics"])
    assert statistics["input_rows"] == len(rows)
    assert f.read(subject["response_resource_summary"])["money_cost"] is None
    from lib.science import _experiment
    _, result, raw = _experiment(f.run, subject["recording_design"]["path"], subject["recording_results"]["path"])
    assert len(result["runs"]) == len(raw) == means
    index = f.read(subject["recording_index"])
    copied = f.run / index["SDK_relative_reference_root"]
    for ref in index["files"]:
        path = f.run / ref["path"]
        assert path.read_bytes() == (recording / path.relative_to(copied)).read_bytes()
    with zipfile.ZipFile(f.run / index["closed_receipt_inputs"]["archive"]["path"]) as archive:
        assert len(archive.namelist()) == len(rows)
        assert archive.testzip() is None
    with zipfile.ZipFile(f.run / index["diagnostic_records"]["archive"]["path"]) as archive:
        assert len(archive.namelist()) == len(rows)
        assert archive.testzip() is None
        if track == "clean":
            for row, member in zip(rows, index["diagnostic_records"]["members"]):
                detail = json.loads(archive.read(member["member"]))
                assert analysis.canonical_sha(detail["common_calendar"]["score"]) == row["common_full_score_sha256"]
    assert not (f.run / "datasets/holdout.v1").exists()
    with pytest.raises(ValueError, match="preserve prior analysis"):
        analysis.analyze_secondary(request, recording, project_root=f.project)


@pytest.mark.parametrize("fault", ["missing_sdk_slot", "wrong_order_arm", "unknown", "main_score_tamper", "missing_main_lock", "main_world_during_statistics"])
def test_bad_secondary_or_locked_comparator_blocks_publication(tmp_path, monkeypatch, fault):
    f, request, proof, recording, measure = prepared_recording(tmp_path, monkeypatch, "ablation")
    if fault == "missing_sdk_slot":
        path = recording / "episodes/results.json"; value = json.loads(path.read_text()); value["runs"].pop(); write(path, value)
    elif fault == "wrong_order_arm":
        path = recording / "episode-order.json"; value = json.loads(path.read_text()); value["slots"][0]["arm"] = "other"; write(path, value)
    elif fault == "unknown":
        path = recording / "episodes/results.json"; value = json.loads(path.read_text()); value["execution_status"] = "running"; write(path, value)
    elif fault == "main_score_tamper":
        path = f.run / "analysis/main.v1/admitted-episodes.json"; path.write_bytes(path.read_bytes() + b" ")
    elif fault == "missing_main_lock":
        (f.run / "analysis/main-primary-lock.v1.json").unlink()
    else:
        slot = proof.primary.slots[proof.slots[0].parent_order_index]
        path = f.project / "logs/inspection-formal-episodes.v1/main" / f"{slot.case_id}-{slot.condition}-{slot.repeat_ordinal}" / "attempt-00/flight/world.json"
        def mutate(track, rows):
            write(path, {"changed_after_validation": True})
            return {"role": "literal mutation probe, no research result"}
        monkeypatch.setattr(analysis, "analyze_interventions", mutate)
    with pytest.raises((ValueError, FileNotFoundError)):
        analysis.analyze_secondary(request, recording, project_root=f.project)
    assert not (f.run / "analysis/secondary-ablation.v1").exists()
    assert not (f.run / "experiment/secondary-ablation-recording.v1").exists()
    assert not (f.run / "experiment/results.secondary-ablation.v1.json").exists()


def test_secondary_analysis_cli_emits_json(tmp_path, monkeypatch, capsys):
    request = tmp_path / "request.json"; write(request, {"software": "fixture"})
    value = {"role": "software fixture", "flags": [True, False, None]}
    monkeypatch.setattr(analysis, "analyze_secondary", lambda *args: value)
    monkeypatch.setattr(sys, "argv", ["analysis", "--request", str(request), "--recording", str(tmp_path / "unused")])
    analysis.main()
    assert json.loads(capsys.readouterr().out) == value
