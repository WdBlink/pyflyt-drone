"""Actual SDK with Fake physical values; no real formal draw/model/flight."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / "source/eval"
spec = importlib.util.spec_from_file_location("formal_analysis_runner_helpers", HERE / "test_formal_runner.py")
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)
load = helpers.load
load("eval.run_inspection_formal", STAGE / "run_inspection_formal.py")
load("eval.inspection_statistics", STAGE / "inspection_statistics.py")
load("eval.inspection_secondary_statistics", STAGE / "inspection_secondary_statistics.py")
analysis = load("formal_primary_analysis_candidate", STAGE / "analyze_inspection_formal.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def prepared_recording(tmp_path, monkeypatch):
    f, request, proof, seen = helpers.fake_driver(tmp_path, monkeypatch, "main")
    slots = {(s.case_id, s.condition, s.repeat_ordinal): s for s in proof.slots}

    def execute(directory, plan, command, **kwargs):
        directory.mkdir(parents=True)
        write(directory / "plan.json", plan)
        terminal = {"status": "completed", "settlement_file_hashes": {}}
        slot = slots[plan["case_id"], plan["condition"], plan["repeat_ordinal"]]
        write(directory / "attempt-00/flight/world.json", {"role": "synthetic closure file, no physics", "value": plan["repeat_ordinal"] / 4})
        if plan["sampler_seed"] is not None:
            reservation = f"formal-main-{request['g4']['sha256'][:12]}-{slot.order_index:04d}-a0"
            write(directory / "attempt-00/intent.json", {"reservation": reservation})
            resource = f.run / "qualification/local-model-episodes" / reservation / "usage-receipt.json"
            write(resource, {"role": "literal software fixture; no model usage or native operation"})
            terminal["settlement_file_hashes"] = {str(resource): hashlib.sha256(resource.read_bytes()).hexdigest()}
            operation = f.project / "logs/inspection-model-resource-operations.v1" / reservation
            write(operation / "service-completion.json", {"role": "software fixture, no service or model"})
            for suffix in (".reserve.json", ".settle.json"):
                write(f.run / "resource-requests" / (reservation + suffix), {"role": "software fixture, no native operation"})
        write(directory / "attempt-00/terminal.json", terminal)
        seen.append(slot)

    def measure(directory, expected):
        y = expected["repeat_ordinal"] / 4
        score = {key: y for key in ("C", "S", "D", "Q", "Y")}
        score.update(H=1, E=1, SR=0)
        score["dynamic_entities"] = [{"bins": [{"index": 1, "quality": y}]}]
        return y, {"role": "Fake physical validator for software integration only", "score": score}

    def admitted(plan, project):
        slot = slots[plan["case_id"], plan["condition"], plan["repeat_ordinal"]]
        return SimpleNamespace(binding=lambda: {"phase": "main", "execution": request["execution"],
            **{key: getattr(slot, key) for key in ("case_id", "condition", "repeat_ordinal", "sampler_seed", "order_index")}})

    monkeypatch.setattr(helpers.driver, "execute_local_episode", execute)
    monkeypatch.setattr(helpers.driver, "execute_development_llm_episode", execute)
    monkeypatch.setattr(helpers.driver, "measure_completed_episode", measure)
    monkeypatch.setattr(analysis, "admitted_request", lambda *args: (f.verification, proof))
    monkeypatch.setattr(analysis, "admit_episode_plan", admitted)
    monkeypatch.setattr(analysis, "measure_completed_episode", measure)
    monkeypatch.setattr(analysis, "operation_directory", lambda reservation:
        f.project / "logs/inspection-model-resource-operations.v1" / reservation)
    recording = f.project / "logs/analysis-software-sdk"
    helpers.driver.run_primary(request, recording, project_root=f.project)
    assert len(seen) == 1296
    return f, request, proof, recording, measure


def test_actual_sdk_full_main_projection_and_fixed_statistics_with_fake_receipts(tmp_path, monkeypatch):
    f, request, proof, recording, measure = prepared_recording(tmp_path, monkeypatch)
    subject = analysis.analyze_primary(request, recording, project_root=f.project)
    assert (subject["root_denominator"], subject["actual_episode_denominator"], subject["root_mean_denominator"]) == (72, 1296, 576)
    assert subject["confirmatory_family_size"] == 6
    admitted = json.loads((f.run / subject["admitted_episode_scores"]["path"]).read_text())
    assert len(admitted) == 1296 and all(set(row["score"]) == set(analysis.SCORE_FIELDS) for row in admitted)
    for name in ("component_summary", "sr_statistics"):
        assert (f.run / subject[name]["path"]).is_file()
    first = admitted[0]
    _, full = measure(f.project / first["source_episode"], first)
    assert first["full_score_sha256"] == analysis.canonical_sha(full["score"])
    assert not (f.run / "analysis/main-primary-lock.v1.json").exists()
    primary = json.loads((f.run / subject["primary_statistics"]["path"]).read_text())
    assert primary["spec"]["draws"] == 99999
    assert sum(len(row["contrasts"]) for row in primary["domains"].values()) == 6
    assert primary["domains"]["ID"]["contrasts"]["classic-M1"]["effect"] == .25
    assert primary["domains"]["OOD"]["contrasts"]["architecture-M2"]["effect"] == 0
    components = json.loads((f.run / "analysis/main.v1/component-summary.json").read_text())
    assert "D" not in components["stratum_means"]["ID.L1"]["B0"]
    assert components["stratum_means"]["ID.L2"]["B3-M1"]["SR"] == 0
    index = json.loads((f.run / subject["recording_index"]["path"]).read_text())
    for ref in index["files"]:
        original = recording / Path(ref["path"]).relative_to("experiment/main-recording.v1")
        assert original.read_bytes() == (f.run / ref["path"]).read_bytes()
    with zipfile.ZipFile(f.run / index["closed_receipt_inputs"]["archive"]["path"]) as archive:
        assert len(archive.namelist()) == 1296 and archive.testzip() is None
        for member in index["closed_receipt_inputs"]["members"]:
            raw = archive.read(member["member"])
            assert hashlib.sha256(raw).hexdigest() == member["sha256"]
            assert any(name.endswith("/flight/world.json") for name in json.loads(raw)["snapshot"]["files"])
    from lib.science import _experiment
    _, projected, raw = _experiment(f.run, subject["design"]["path"], subject["recording_results"]["path"], subject["design"]["sha256"])
    assert len(raw) == len(projected["runs"]) == 576
    settlements = json.loads((f.run / subject["all_model_settlements"]["path"]).read_text())
    assert len(settlements["attempts"]) == 1080
    with pytest.raises(ValueError, match="preserve prior analysis"):
        analysis.analyze_primary(request, recording, project_root=f.project)


@pytest.mark.parametrize("fault", ["inner_raw", "missing_slot", "physical_value", "outer_mean"])
def test_bad_sdk_or_recount_blocks_analysis_before_output(tmp_path, monkeypatch, fault):
    f, request, proof, recording, measure = prepared_recording(tmp_path, monkeypatch)
    if fault == "inner_raw":
        result = json.loads((recording / "classic/results.json").read_text())
        path = recording / result["runs"][0]["raw"]
        raw = json.loads(path.read_text()); raw["value"] = .125; write(path, raw)
    elif fault == "missing_slot":
        path = recording / "llm/results.json"
        result = json.loads(path.read_text()); result["runs"].pop(); write(path, result)
    elif fault == "physical_value":
        def wrong(*args):
            y, evidence = measure(*args)
            return y + .125, evidence
        monkeypatch.setattr(analysis, "measure_completed_episode", wrong)
    else:
        path = recording / "root-means/results.json"
        result = json.loads(path.read_text()); row = result["runs"][0]; row["value"] += .125
        raw_path = recording / row["raw"]
        raw = json.loads(raw_path.read_text()); raw["value"] = row["value"]; write(raw_path, raw); write(path, result)
        index_path = recording / "recording-completion-index.json"
        index = json.loads(index_path.read_text()); index["results_sha256"]["root-means"] = hashlib.sha256(path.read_bytes()).hexdigest(); write(index_path, index)
    with pytest.raises((ValueError, RuntimeError)):
        analysis.analyze_primary(request, recording, project_root=f.project)
    assert not (f.run / "analysis/main.v1").exists()
    assert not (f.run / "experiment/main-recording.v1").exists()
    assert not (f.run / "experiment/results.main.v1.json").exists()


@pytest.mark.parametrize("fault", ["plan_after_validation", "terminal_after_validation", "world_during_statistics"])
def test_validation_window_changes_reject_original_closed_snapshot(tmp_path, monkeypatch, fault):
    f, request, proof, recording, measure = prepared_recording(tmp_path, monkeypatch)
    if fault != "world_during_statistics":
        changed = []
        def mutate(directory, expected):
            result = measure(directory, expected)
            if not changed:
                path = directory / ("plan.json" if fault.startswith("plan") else "attempt-00/terminal.json")
                value = json.loads(path.read_text())
                value["case_sha256" if fault.startswith("plan") else "status"] = "0" * 64 if fault.startswith("plan") else "unknown_model_usage"
                write(path, value); changed.append(path)
            return result
        monkeypatch.setattr(analysis, "measure_completed_episode", mutate)
    else:
        slot = proof.slots[0]
        path = f.project / "logs/inspection-formal-episodes.v1/main" / f"{slot.case_id}-{slot.condition}-{slot.repeat_ordinal}" / "attempt-00/flight/world.json"
        def mutate_statistics(*args):
            path.write_text('{"changed_after_collection":true}\n')
            return {"role": "fixed software statistics stub; not research results"}
        monkeypatch.setattr(analysis, "analyze_roots", mutate_statistics)
        monkeypatch.setattr(analysis, "sr_cluster_intervals", lambda *args: {"role": "fixed software stub"})
    with pytest.raises(ValueError, match="closed receipt input changed"):
        analysis.analyze_primary(request, recording, project_root=f.project)
    assert not (f.run / "analysis/main.v1").exists()
    assert not (f.run / "experiment/main-recording.v1").exists()
    assert not (f.run / "experiment/results.main.v1.json").exists()


def test_holdout_directions_are_twelve_root_descriptive_with_no_extra_test():
    metadata = [{"case_id": f"software-L{level}-{i}", "domain": "ID", "level": level,
                 "lineage_group": f"software-L{level}-{i}"} for level in (2, 3) for i in range(6)]
    means = {row["case_id"]: {**dict.fromkeys(analysis.CONDITION_REPEATS, .25),
        "B4-M1": .4, "B3-M1": .2, "B1": .3} for row in metadata}
    table = analysis.RootTable.parse(metadata, means, require_main_shape=False)
    contrasts = {}
    for name, left, right, effects in [("architecture-M1", "B4-M1", "B3-M1", (-.9, .1, .1)),
            ("architecture-M2", "B4-M2", "B3-M2", (.5, -.1, -.1)),
            ("classic-M1", "B4-M1", "B1", (-.9, .2, .2))]:
        contrasts[name] = {"minuend": left, "subtrahend": right, "effect": sum(effects) / 3,
            "root_differences": [{"case_id": f"software-main-L{level}-{i}", "level": level,
                "difference": effects[level - 1]} for level in (1, 2, 3) for i in range(16)]}
    main = {"domains": {"ID": {"contrasts": contrasts}}}
    result = analysis.holdout_directions(table, "B1", main)
    assert result["p_values"] == "not_computed"
    assert result["multiplicity_family"] == "unchanged_main_global6"
    assert result["contrasts"]["architecture-M1"]["effect"] == pytest.approx(.2)
    assert result["contrasts"]["architecture-M1"]["locked_main_ID_all_levels_effect"] < 0
    assert result["contrasts"]["architecture-M1"]["locked_main_ID_L2_L3_effect"] == pytest.approx(.1)
    assert result["contrasts"]["architecture-M1"]["same_nonzero_direction_vs_main_ID_L2_L3"] is True
    assert result["contrasts"]["architecture-M2"]["same_nonzero_direction_vs_main_ID_L2_L3"] is False


def test_ref_rejects_symlink_before_reading_target(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    outside = tmp_path / "outside"; outside.write_text("unread target")
    (root / "link").symlink_to(outside)
    def forbidden(*args):
        raise AssertionError("must validate path before hashing its bytes")
    monkeypatch.setattr(analysis, "file_sha", forbidden)
    with pytest.raises(ValueError, match="symlinks"):
        analysis._ref(root / "link", root)


@pytest.mark.parametrize("changed", ["operation_new_file", "settlement_bytes", "empty_call_directory", "nonregular_entry"])
def test_closed_snapshot_includes_model_operation_and_settlement(tmp_path, monkeypatch, changed):
    project = tmp_path / "project"; project.mkdir()
    root = project / "run"; root.mkdir()
    directory = project / "logs/receipt"
    write(directory / "plan.json", {"role": "software fixture"})
    write(directory / "attempt-00/terminal.json", {"role": "software fixture"})
    slot = SimpleNamespace(sampler_seed=1, order_index=0)
    request = {"run": str(root), "opening": None, "g4": {"sha256": "a" * 64}}
    reservation = "formal-main-aaaaaaaaaaaa-0000-a0"
    operation = project / "logs/operation" / reservation
    resource = root / "qualification/local-model-episodes" / reservation
    write(operation / "service-completion.json", {"role": "software fixture"})
    write(resource / "usage-receipt.json", {"role": "software fixture"})
    for suffix in (".reserve.json", ".settle.json"):
        write(root / "resource-requests" / (reservation + suffix), {"role": "software fixture"})
    monkeypatch.setattr(analysis, "operation_directory", lambda _: operation)
    before = analysis._receipt_snapshot(directory, slot, request, project)
    if changed == "operation_new_file":
        write(operation / "call-0001/tokens.jsonl", {"new": True})
    elif changed == "settlement_bytes":
        write(resource / "usage-receipt.json", {"changed": True})
    elif changed == "empty_call_directory":
        (operation / "call-0002").mkdir()
    else:
        os.mkfifo(operation / "unexpected-fifo")
    with pytest.raises(ValueError, match="closed receipt input"):
        analysis._check_snapshots([(slot, directory, before)], request, project)


def test_cli_emits_standard_json_not_python_repr(tmp_path, monkeypatch, capsys):
    request = tmp_path / "request.json"; write(request, {"software": "fixture"})
    value = {"role": "software fixture", "flags": [True, False, None]}
    monkeypatch.setattr(analysis, "analyze_primary", lambda *args: value)
    monkeypatch.setattr(sys, "argv", ["analysis", "--request", str(request), "--recording", str(tmp_path / "unused")])
    analysis.main()
    assert json.loads(capsys.readouterr().out) == value


@pytest.mark.parametrize("track", ["clean", "ablation", "latency", "unadmitted"])
def test_secondary_snapshot_uses_only_the_typed_track_prefix(tmp_path, monkeypatch, track):
    project = tmp_path / "project"; project.mkdir()
    root = project / "run"; root.mkdir()
    directory = project / "logs/receipt"
    write(directory / "attempt-00/terminal.json", {"role": "software fixture"})
    slot = SimpleNamespace(track=track, sampler_seed=1, order_index=2)
    request = {"run": str(root), "opening": None, "g4": {"sha256": "a" * 64}}
    reservation = f"formal-{track}-aaaaaaaaaaaa-0002-a0"
    operation = project / "logs/operation" / reservation
    resource = root / "qualification/local-model-episodes" / reservation
    write(operation / "service-completion.json", {"role": "software fixture"})
    write(resource / "usage-receipt.json", {"role": "software fixture"})
    for suffix in (".reserve.json", ".settle.json"):
        write(root / "resource-requests" / (reservation + suffix), {"role": "software fixture"})
    seen = []
    def operation_path(value):
        seen.append(value)
        return operation
    monkeypatch.setattr(analysis, "operation_directory", operation_path)
    if track == "unadmitted":
        with pytest.raises(ValueError, match="fixed secondary tracks"):
            analysis._receipt_snapshot(directory, slot, request, project)
        assert seen == []
    else:
        snapshot = analysis._receipt_snapshot(directory, slot, request, project)
        assert seen == [reservation]
        assert str(resource.relative_to(project)) in snapshot["directories"]
        assert all("formal-main-" not in name for name in snapshot["files"])
