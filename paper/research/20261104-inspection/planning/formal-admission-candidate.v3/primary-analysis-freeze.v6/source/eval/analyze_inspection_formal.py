"""Revalidate completed primary receipts before fixed main/holdout analysis.

No sampling, model/flight effect, native finding, analysis lock or holdout
opening is produced. A main subject still needs actual independent review.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.inspection_formal_admission import Artifact, bounded_bytes, canonical_sha, file_sha, parse_json_bytes, read_json
from eval.inspection_formal_dispatch import admit_episode_plan
from eval.inspection_measurement_loader import measure_completed_episode
from eval.inspection_model_service import operation_directory
from eval.inspection_recording import CONDITION_REPEATS, designs
from eval.inspection_secondary_statistics import sr_cluster_intervals, sr_root_table
from eval.inspection_statistics import RootTable, analyze_roots
from eval.run_inspection_formal import admitted_request, read_request

SCORE_FIELDS = ("C", "S", "D", "Q", "H", "E", "Y", "SR")


def scalar_score(score, level, y):
    checked = [key for key in SCORE_FIELDS if key != "D" or level != 1]
    if (type(level) is not int or level not in (1, 2, 3)
            or level == 1 and score["D"] is not None
            or any(type(score[key]) not in (int, float) or not math.isfinite(score[key])
                   or not 0 <= score[key] <= 1 for key in checked)
            or any(type(score[key]) is not int or score[key] not in (0, 1) for key in ("H", "E", "SR"))
            or score["Y"] != y):
        raise ValueError("finite bounded revalidated score components with L1 D not applicable required")
    return {key: score[key] for key in SCORE_FIELDS}


def _ref(path, root):
    path, root = Path(path).absolute(), Path(root).resolve()
    name = str(path.relative_to(root))
    path = Artifact(name, "0" * 64).path_in(root)
    return {"path": name, "sha256": file_sha(path)}


def _receipt_snapshot(directory, slot, request, project):
    """Bind the closed input set before validation; never trust a later rebind."""
    root = Path(request["run"]).resolve()
    bases, files = [directory], set()
    attempts = sorted(directory.glob("attempt-*"))
    if (not 1 <= len(attempts) <= 2
            or [path.name for path in attempts] != [f"attempt-{i:02d}" for i in range(len(attempts))]):
        raise ValueError("complete finite receipt sequence required before analysis")
    if slot.sampler_seed is not None:
        kind = getattr(slot, "track", None)
        if kind is not None and kind not in ("clean", "ablation", "latency"):
            raise ValueError("only admitted primary phases or fixed secondary tracks have receipts")
        if kind is None:
            kind = "main" if request["opening"] is None else "holdout"
        for ordinal in range(len(attempts)):
            reservation = f"formal-{kind}-{request['g4']['sha256'][:12]}-{slot.order_index:04d}-a{ordinal}"
            bases.extend([operation_directory(reservation), root / "qualification/local-model-episodes" / reservation])
            files.update(root / "resource-requests" / (reservation + suffix) for suffix in (".reserve.json", ".settle.json"))
    directories = set(bases)
    for base in bases:
        if base.is_symlink() or not base.is_dir() or not base.resolve().is_relative_to(project):
            raise ValueError("closed receipt directories must remain inside the project")
        for path in base.rglob("*"):
            if path.is_symlink():
                raise ValueError("closed receipt inputs cannot traverse symlinks")
            if path.is_dir():
                directories.add(path)
            elif path.is_file():
                files.add(path)
            else:
                raise ValueError("closed receipt inputs must be regular files or directories")
    return {"directories": sorted(str(path.relative_to(project)) for path in directories),
            "files": {ref["path"]: ref["sha256"] for ref in (_ref(path, project) for path in sorted(files))}}


def _check_snapshots(snapshots, request, project):
    for slot, directory, before in snapshots:
        if _receipt_snapshot(directory, slot, request, project) != before:
            raise ValueError("closed receipt input changed after validation")


def collect_recording(request, recording, *, project_root=ROOT):
    """Read-only recount; returns captured SDK bytes and validated scores."""
    request = read_request(request)
    project, root = Path(project_root).resolve(), Path(request["run"]).resolve()
    recording = Path(recording).absolute()
    if not recording.resolve().is_relative_to(project / "logs"):
        raise ValueError("completed SDK recording must be inside project logs")
    _, proof = admitted_request(request, project)
    phase = proof.dataset.namespace.phase
    acceptance = proof.dataset.namespace.g4_acceptance.json(root)
    subject = Artifact.parse(acceptance["subject"]).json(root)
    design_ref = Artifact.parse(subject["design"])
    outer = deepcopy(design_ref.json(root))
    cases = [row.case_id for row in proof.dataset.records]
    metadata = [{"case_id": row.case_id, "domain": row.domain, "level": row.level,
                 "lineage_group": row.case_id} for row in proof.dataset.records]
    if phase == "holdout":
        outer.update(cases=cases, experiment_id="continuous-inspection-holdout-root-means-v1",
            hypothesis="Once-only descriptive direction and size after locked main analysis; no seventh confirmatory test.")
        outer["recording_semantics"].update(root_denominator=12, actual_episode_denominator=216,
            root_metadata=[{k: row[k] for k in ("case_id", "domain", "level")} for row in metadata])
    elif phase != "main" or outer["cases"] != cases:
        raise ValueError("only the admitted main or once-opened holdout can be analyzed")
    captured, receipt_snapshots = {}, []

    def capture(name, *, source=False):
        ref = _ref(recording / name, project)
        raw = bounded_bytes(Artifact.parse(ref).path_in(project), 16 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != ref["sha256"] or name in captured and captured[name] != raw:
            raise ValueError("SDK recording changed during collection")
        captured[name] = raw
        return raw if source else parse_json_bytes(raw)

    if canonical_sha(capture("run-request.json")) != canonical_sha(request):
        raise ValueError("SDK run request differs from the admitted phase")
    order = capture("episode-order.json")
    if canonical_sha(order) != canonical_sha(proof.episode_order.json(root)):
        raise ValueError("SDK order differs from the frozen primary execution plan")
    schedule_sha = hashlib.sha256(captured["episode-order.json"]).hexdigest()
    context = {key: outer[key] for key in ("claim_id", "hypothesis", "independent_variable",
        "dependent_variable", "control_variables", "data_roles", "planned_at")}
    context["experiment_id"] = outer["experiment_id"] + "-inner"
    specs = designs(cases, context, proof.classic_best)
    specs["root-means"] = outer
    outer_path = design_ref.path if phase == "main" else "root-means/design.json"
    design_paths = {"classic": "classic/design.json", "llm": "llm/design.json", "root-means": outer_path}
    sys.path.insert(0, str(Path(request["sdk"]).resolve()))
    from lib.science import _experiment
    results, raw_rows = {}, {}
    for group, spec in specs.items():
        if group != "root-means":
            spec["execution_schedule"] = {"path": "episode-order.json", "sha256": schedule_sha}
        if canonical_sha(capture(design_paths[group])) != canonical_sha(spec):
            raise ValueError("SDK design differs from the accepted phase and complete repetition grid")
        if group == "root-means" and phase == "main" and captured[outer_path] != design_ref.checked_path(root).read_bytes():
            raise ValueError("main must retain exact accepted outer design bytes")
        document = capture(group + "/results.json")
        if document.get("execution_status") != "completed" or document.get("outcome") != "inconclusive":
            raise ValueError("incomplete or operational SDK recording cannot become analysis")
        for row in document["runs"]:
            name = f"{group}/raw/{row['case_id']}-{row['condition']}-{row['seed']}.json"
            if row["raw"] != name or row["status"] != "completed":
                raise ValueError("complete canonical SDK raw identities required")
            raw_rows[group, row["case_id"], row["seed"], row["condition"]] = capture(name)
        results[group] = document
    evaluator = capture(outer["evaluator"]["path"], source=True)
    if hashlib.sha256(evaluator).hexdigest() != outer["evaluator"]["sha256"]:
        raise ValueError("SDK evaluator bytes differ from the frozen evaluator")
    closure = proof.dataset.namespace.source_manifest.json(root)
    prefix = str((root / "experiment").relative_to(project)) + "/"
    for name, sha in closure["sources"].items():
        if name.startswith(prefix + "evaluator-source"):
            raw = capture(str((project / name).relative_to(root)), source=True)
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("SDK evaluator-source bytes differ from the frozen closure")
    complete = capture("recording-completion-index.json")
    expected_result_hashes = {group: hashlib.sha256(captured[group + "/results.json"]).hexdigest() for group in specs}
    if (complete.get("episode_denominator") != len(proof.slots)
            or type(complete.get("episode_denominator")) is not int
            or complete.get("schedule_sha256") != schedule_sha
            or canonical_sha(complete.get("results_sha256")) != canonical_sha(expected_result_hashes)):
        raise ValueError("SDK completion index is not bound to the full recording")
    done = capture("primary-callbacks-complete.json")
    expected_done = {"independent_roots": len(cases), "actual_episode_denominator": len(proof.slots),
        "classic_episode_denominator": len(cases) * 3, "llm_episode_denominator": len(cases) * 15,
        "outer_aggregate_rows": len(cases) * 8, "conditions": dict(CONDITION_REPEATS),
        "primary_inference_ready": False, "phase": phase,
        "role": "full source-bound callbacks, not primary analysis or scientific acceptance",
        "execution_manifest": request["execution"]}
    if canonical_sha(done) != canonical_sha(expected_done):
        raise ValueError("complete source-bound primary callback index required")
    # Validate the exact captured bytes, rather than reopening mutable SDK files.
    with tempfile.TemporaryDirectory(prefix="inspection-sdk-validation-") as directory:
        snapshot = Path(directory)
        for name, raw in captured.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        for group in specs:
            _experiment(snapshot, design_paths[group], group + "/results.json",
                        design_ref.sha256 if phase == "main" and group == "root-means" else None)
    rows, values, settlements = [], {}, []
    for slot in proof.slots:
        group = "classic" if slot.sampler_seed is None else "llm"
        identity = slot.case_id, slot.repeat_ordinal, slot.condition
        raw = raw_rows[(group, *identity)]
        receipt = capture(f"order-receipts/{slot.order_index:06d}.json")
        expected_receipt = {"role": "measurement_callback_returned_not_independent_flight_receipt",
            **order["slots"][slot.order_index], "schedule_sha256": schedule_sha,
            "raw": f"{group}/raw/{slot.case_id}-{slot.condition}-{slot.repeat_ordinal}.json"}
        if canonical_sha(receipt) != canonical_sha(expected_receipt):
            raise ValueError("SDK order receipt differs from the frozen next slot")
        directory = project / "logs/inspection-formal-episodes.v1" / phase / f"{slot.case_id}-{slot.condition}-{slot.repeat_ordinal}"
        before = _receipt_snapshot(directory, slot, request, project)
        plan_name = str((directory / "plan.json").relative_to(project))
        plan = Artifact(plan_name, before["files"][plan_name]).json(project)
        admission = admit_episode_plan(plan, project)
        binding = admission.binding()
        if (binding.get("track") is not None or binding["phase"] != phase
                or canonical_sha(binding["execution"]) != canonical_sha(request["execution"])
                or any(binding[key] != getattr(slot, key) for key in
                       ("case_id", "condition", "repeat_ordinal", "sampler_seed", "order_index"))):
            raise ValueError("physical receipt is not this exact primary slot")
        y, evidence = measure_completed_episode(directory, {**plan, "phase": phase})
        if _receipt_snapshot(directory, slot, request, project) != before:
            raise ValueError("closed receipt input changed after validation")
        receipt_snapshots.append((slot, directory, before))
        recorded_evidence = {**evidence, "adapter_repeat_ordinal": slot.repeat_ordinal,
            "adapter_condition": slot.condition, "order_index": slot.order_index,
            "schedule_sha256": schedule_sha}
        if raw["value"] != y or canonical_sha(raw["evidence"]) != canonical_sha(recorded_evidence):
            raise ValueError("SDK value/evidence differs from independently revalidated physical receipt")
        score = evidence["score"]
        scalars = scalar_score(score, slot.level, y)
        rows.append({"case_id": slot.case_id, "domain": slot.domain, "level": slot.level,
            "condition": slot.condition, "repeat_ordinal": slot.repeat_ordinal, "status": "completed",
            "SR": score["SR"], "score": scalars,
            "full_score_sha256": canonical_sha(score), "source_episode": str(directory.relative_to(project))})
        values[identity] = y
        if slot.sampler_seed is not None:
            for path in sorted(directory.glob("attempt-*/terminal.json")):
                terminal = read_json(path)
                refs = []
                for name, sha in terminal["settlement_file_hashes"].items():
                    ref = _ref(Path(name), root)
                    if ref["sha256"] != sha:
                        raise ValueError("settlement changed after read-only native-backed receipt admission")
                    refs.append(ref)
                settlements.append({"case_id": slot.case_id, "condition": slot.condition,
                    "repeat_ordinal": slot.repeat_ordinal, "attempt": path.parent.name,
                    "status": terminal["status"], "evidence": refs})
    means = {case: {condition: mean(values[case, repeat, condition] for repeat in range(count))
                   for condition, count in CONDITION_REPEATS.items()} for case in cases}
    for case in cases:
        for condition, count in CONDITION_REPEATS.items():
            raw = raw_rows["root-means", case, 0, condition]
            inner = "classic" if count == 1 else "llm"
            bindings = [{"repeat_ordinal": repeat,
                "path": f"{inner}/raw/{case}-{condition}-{repeat}.json",
                "sha256": hashlib.sha256(captured[f"{inner}/raw/{case}-{condition}-{repeat}.json"]).hexdigest()}
                for repeat in range(count)]
            expected = {"aggregation": "arithmetic mean within independent root", "actual_episode_count": count,
                "inner_attempts": bindings, "inner_results": inner + "/results.json",
                "inner_results_sha256": expected_result_hashes[inner], "schedule_sha256": schedule_sha,
                "not_an_additional_episode": True}
            if raw["value"] != means[case][condition] or canonical_sha(raw["evidence"]) != canonical_sha(expected):
                raise ValueError("outer root mean is not the complete original repeat aggregation")
    admitted_request(request, project)
    for name, raw in captured.items():
        Artifact(str((recording / name).relative_to(project)), hashlib.sha256(raw).hexdigest()).checked_path(project)
    _check_snapshots(receipt_snapshots, request, project)
    return proof, metadata, rows, means, settlements, captured, results, receipt_snapshots


def component_summary(metadata, rows):
    """Repeat means first; D has no L1 cell and is never filled with a fake zero."""
    roots = {}
    for root in metadata:
        roots[root["case_id"]] = {}
        for condition in CONDITION_REPEATS:
            selected = [row["score"] for row in rows if row["case_id"] == root["case_id"] and row["condition"] == condition]
            roots[root["case_id"]][condition] = {key: mean(row[key] for row in selected)
                for key in SCORE_FIELDS if key != "D" or root["level"] != 1}
    levels = {}
    for domain, level in sorted({(row["domain"], row["level"]) for row in metadata}):
        members = [row["case_id"] for row in metadata if (row["domain"], row["level"]) == (domain, level)]
        levels[f"{domain}.L{level}"] = {condition: {key: mean(roots[case][condition][key] for case in members)
            for key in roots[members[0]][condition]} for condition in CONDITION_REPEATS}
    return {"role": "descriptive_repeat_then_root_component_means", "root_means": roots,
            "stratum_means": levels, "L1_D": "not_applicable"}


def holdout_directions(table, classic_best, main_statistics):
    pairs = (("architecture-M1", "B4-M1", "B3-M1"), ("architecture-M2", "B4-M2", "B3-M2"),
             ("classic-M1", "B4-M1", classic_best))
    columns = {name: i for i, name in enumerate(CONDITION_REPEATS)}
    contrasts = {}
    for name, left, right in pairs:
        differences = {level: [float(table.values[i, columns[left]] - table.values[i, columns[right]])
            for i, group in enumerate(table.groups) if group == ("ID", level)] for level in (2, 3)}
        if set(table.groups) != {("ID", 2), ("ID", 3)} or any(len(values) != 6 for values in differences.values()):
            raise ValueError("once-only twelve-root ID L2/L3 descriptive holdout required")
        effect = mean(mean(values) for values in differences.values())
        locked = main_statistics["domains"]["ID"]["contrasts"][name]
        original = locked["root_differences"]
        if (locked["minuend"] != left or locked["subtrahend"] != right or len(original) != 48
                or len({row["case_id"] for row in original}) != 48
                or any(type(row["level"]) is not int or row["level"] not in (1, 2, 3)
                       or type(row["difference"]) not in (int, float) or not math.isfinite(row["difference"])
                       or not -1 <= row["difference"] <= 1 for row in original)
                or any(sum(row["level"] == level for row in original) != 16 for level in (1, 2, 3))):
            raise ValueError("complete locked main ID root differences required for matched-level description")
        matched = mean(mean(row["difference"] for row in original if row["level"] == level) for level in (2, 3))
        contrasts[name] = {"minuend": left, "subtrahend": right, "effect": effect,
            "locked_main_ID_all_levels_effect": locked["effect"], "locked_main_ID_L2_L3_effect": matched,
            "same_nonzero_direction_vs_main_ID_L2_L3": effect * matched > 0,
            "root_differences_by_level": differences}
    return {"role": "once_only_descriptive_direction_and_size_no_seventh_test", "contrasts": contrasts,
            "p_values": "not_computed", "multiplicity_family": "unchanged_main_global6",
            "comparison": "equal L2/L3 means in both holdout and locked main; full-ID primary effect retained separately"}


def analyze_primary(request, recording, *, project_root=ROOT):
    request = read_request(request)
    project, root = Path(project_root).resolve(), Path(request["run"]).resolve()
    _, initial = admitted_request(request, project)
    phase = initial.dataset.namespace.phase
    output, copied = root / "analysis" / (phase + ".v1"), root / "experiment" / (phase + "-recording.v1")
    projected = root / "experiment" / ("results." + phase + ".v1.json")
    sys.path.insert(0, str(Path(request["sdk"]).resolve()))
    from lib.runtime import safe
    for path in (output, copied, projected):
        safe(root, str(path.relative_to(root)), output=True)
    if output.exists() or copied.exists() or projected.exists():
        raise ValueError("preserve prior analysis; canonical phase output must be new")
    lock_path = project / "logs/inspection-formal-episodes.v1/.matrix.lock"
    _ref(lock_path, project)
    with lock_path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof, metadata, rows, means, settlements, captured, results, receipt_snapshots = collect_recording(
            request, recording, project_root=project)
        table = RootTable.parse(metadata, means, require_main_shape=phase == "main")
        if phase == "main":
            primary = analyze_roots(table, proof.classic_best)
            secondary = sr_cluster_intervals(sr_root_table(metadata, rows))
        else:
            opening = proof.dataset.namespace.holdout_opening.json(root)
            main_lock = Artifact.parse(opening["main_analysis_lock"]).json(root)
            subject = Artifact.parse(main_lock["subject"]).json(root)
            main_statistics = Artifact.parse(subject["primary_statistics"]).json(root)
            primary = holdout_directions(table, proof.classic_best, main_statistics)
            secondary = {"role": "holdout_SR_descriptive_component_means_only_no_new_test"}
        components = component_summary(metadata, rows)
        admitted_request(request, project)
        _check_snapshots(receipt_snapshots, request, project)
        copied.mkdir(parents=True)
        for name, raw in sorted(captured.items()):
            path = safe(root, str((copied / name).relative_to(root)), output=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        from lib.production import json_bytes, write_once
        relative = lambda path: str(path.relative_to(root))
        acceptance = proof.dataset.namespace.g4_acceptance.json(root)
        source_subject = Artifact.parse(acceptance["subject"]).json(root)
        projection = deepcopy(results["root-means"])
        projection["design"] = (source_subject["design"]["path"] if phase == "main"
                                else relative(copied / "root-means/design.json"))
        for row in projection["runs"]:
            row["raw"] = relative(copied / row["raw"])
        write_once(root, relative(projected), json_bytes(projection))
        output.mkdir(parents=True)
        closure_members = []
        with zipfile.ZipFile(output / "receipt-closures.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for slot, directory, snapshot in receipt_snapshots:
                name = f"{slot.order_index:04d}.json"
                raw = json_bytes({"case_id": slot.case_id, "condition": slot.condition,
                    "repeat_ordinal": slot.repeat_ordinal, "snapshot": snapshot})
                archive.writestr(name, raw)
                closure_members.append({"member": name, "sha256": hashlib.sha256(raw).hexdigest()})
        documents = {"primary-statistics.json": primary, "sr-statistics.json": secondary,
            "component-summary.json": components, "admitted-episodes.json": rows,
            "all-model-settlements.json": {"role": "all_actual_main_or_holdout_model_attempt_settlements", "attempts": settlements},
            "recording-index.json": {"role": "SDK_original_bytes_with_raw_path_only_outer_projection",
                "files": [_ref(copied / name, root) for name in sorted(captured)],
                "projection": _ref(projected, root), "closed_receipt_inputs": {
                    "archive": _ref(output / "receipt-closures.zip", root), "members": closure_members}}}
        for name, value in documents.items():
            write_once(root, relative(output / name), json_bytes(value))
        subject = {"schema": "inspection-main-primary-analysis-subject-v1" if phase == "main" else "inspection-holdout-descriptive-analysis-subject-v1",
            "phase": phase, "root_denominator": len(metadata), "actual_episode_denominator": len(rows),
            "root_mean_denominator": len(metadata) * 8, "confirmatory_family_size": 6 if phase == "main" else 0,
            "design": source_subject["design"], "source_manifest": {"path": proof.dataset.namespace.source_manifest.path,
                "sha256": proof.dataset.namespace.source_manifest.sha256},
            "main_dataset" if phase == "main" else "holdout_dataset": request["dataset"],
            "recording_results": _ref(projected, root), "primary_statistics": _ref(output / "primary-statistics.json", root),
            "all_model_settlements": _ref(output / "all-model-settlements.json", root),
            "admitted_episode_scores": _ref(output / "admitted-episodes.json", root),
            "component_summary": _ref(output / "component-summary.json", root),
            "sr_statistics": _ref(output / "sr-statistics.json", root),
            "admitted_score_scope": "Eight scalar endpoints; full original score retained unchanged in SDK raw and source world",
            "recording_index": _ref(output / "recording-index.json", root),
            "role": "actual_analysis_subject_needs_independent_review_not_a_lock_or_G4_decision"}
        write_once(root, relative(output / "subject.json"), json_bytes(subject))
        return subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_primary(read_json(args.request), args.recording)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
