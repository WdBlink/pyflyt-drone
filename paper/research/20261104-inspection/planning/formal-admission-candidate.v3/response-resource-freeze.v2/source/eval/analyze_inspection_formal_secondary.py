"""Recount fixed secondary flights against the independently locked main analysis.

No flight/model/native effect, main lock, holdout opening or G4 decision.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analyze_inspection_formal import _check_snapshots, _receipt_snapshot, _ref, scalar_score
from eval.inspection_common_calendar import rescore_formal_common_calendar
from eval.inspection_event_endpoints import analyze_formal_event_trace, summarize_formal_events
from eval.inspection_formal_admission import Artifact, bounded_bytes, canonical_sha, parse_json_bytes, read_g4_proof, read_json
from eval.inspection_formal_dispatch import admit_episode_plan
from eval.inspection_formal_secondary import ARMS, read_secondary_execution
from eval.inspection_intervention_statistics import analyze_interventions
from eval.inspection_measurement_loader import measure_completed_episode
from eval.inspection_namespace_admission import read_primary_analysis_lock
from eval.inspection_population import decode_formal_population_case
from eval.inspection_recording import CONDITION_REPEATS, designs
from eval.inspection_response_diagnostics import episode_response_diagnostics, summarize_response_resources
from eval.run_inspection_formal import admitted_request
from eval.run_inspection_formal_secondary import read_secondary_request


def _proofs(request, project):
    request, base = read_secondary_request(request)
    root = Path(request["run"]).resolve()
    verification, primary = admitted_request(base, project)
    proof = read_secondary_execution(root, project, request["g4"], request["namespace"], request["dataset"],
        request["dataset_review"], request["execution"], request["secondary_execution"], verification=verification)
    if proof.primary != primary:
        raise ValueError("secondary analysis must use this exact freshly admitted main parent execution")
    g4 = read_g4_proof(root, project, request["g4"], verification=verification)
    lock = _ref(root / "analysis/main-primary-lock.v1.json", root)
    read_primary_analysis_lock(root, lock, g4, primary.owner_snapshot, verification)
    subject_ref = Artifact.parse(Artifact.parse(lock).json(root)["subject"])
    subject = subject_ref.json(root)
    if canonical_sha(subject["main_dataset"]) != canonical_sha(base["dataset"]):
        raise ValueError("locked main comparator differs from the admitted intervention dataset")
    for key in ("admitted_episode_scores", "component_summary", "sr_statistics", "recording_index"):
        Artifact.parse(subject[key]).checked_path(root)
    return proof, base, lock, subject_ref, subject


def _recount(slot, directory, request, project):
    before = _receipt_snapshot(directory, slot, request, project)
    name = str((directory / "plan.json").relative_to(project))
    plan = Artifact(name, before["files"][name]).json(project)
    binding = admit_episode_plan(plan, project).binding()
    track = getattr(slot, "track", None)
    execution = request["secondary_execution"] if track else request["execution"]
    keys = ("case_id", "condition", "repeat_ordinal", "sampler_seed", "order_index")
    if track:
        keys += ("track", "arm", "parent_order_index")
    if (binding["phase"] != "main" or canonical_sha(binding["execution"]) != canonical_sha(execution)
            or binding.get("track") != track or any(binding[key] != getattr(slot, key) for key in keys)):
        raise ValueError("recounted receipt is not this exact original primary or secondary slot")
    y, evidence = measure_completed_episode(directory, {**plan, "phase": "main"})
    if _receipt_snapshot(directory, slot, request, project) != before:
        raise ValueError("closed receipt input changed after validation")
    score = evidence["score"]
    row = {"case_id": slot.case_id, "domain": slot.domain, "level": slot.level,
        "condition": slot.condition, "repeat_ordinal": slot.repeat_ordinal, "status": "completed",
        "SR": score["SR"], "score": scalar_score(score, slot.level, y), "full_score_sha256": canonical_sha(score),
        "source_episode": str(directory.relative_to(project))}
    return row, evidence, (slot, directory, before)


def _world(snapshot, project):
    slot, directory, before = snapshot
    path = sorted(directory.glob("attempt-*"))[-1] / "flight/world.json"
    name = str(path.relative_to(project))
    ref = Artifact(name, before["files"][name])
    raw = bounded_bytes(ref.path_in(project), 512 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != ref.sha256:
        raise ValueError("closed world bytes changed before offline diagnostics")
    return parse_json_bytes(raw)


def collect_secondary(request, recording, *, project_root=ROOT):
    request, base = read_secondary_request(request)
    project, root = Path(project_root).resolve(), Path(request["run"]).resolve()
    proof, _, lock, subject_ref, subject = _proofs(request, project)
    recording = Path(recording).absolute()
    if not recording.resolve().is_relative_to(project / "logs"):
        raise ValueError("completed secondary SDK envelope must be inside project logs")
    track = proof.track
    cases = sorted({slot.case_id for slot in proof.slots})
    captured, raw_rows, results, snapshots = {}, {}, {}, []

    def capture(name):
        ref = _ref(recording / name, project)
        raw = bounded_bytes(Artifact.parse(ref).path_in(project), 16 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != ref["sha256"] or name in captured and captured[name] != raw:
            raise ValueError("secondary SDK bytes changed during collection")
        captured[name] = raw
        return parse_json_bytes(raw)

    if canonical_sha(capture("run-request.json")) != canonical_sha(request):
        raise ValueError("secondary SDK request differs from the admitted track")
    order = capture("episode-order.json")
    if track == "clean":
        expected_order = {"slots": [{key: value for key, value in slot.__dict__.items()
            if key not in ("parent_order_index", "track", "arm", "sampler_seed")} for slot in proof.slots],
            "cases": [{"case_id": row.case_id, "domain": row.domain, "level": row.level}
                for row in proof.dataset.records if row.case_id in cases]}
    else:
        expected_order = {"role": "frozen secondary execution order, not measurements",
            "execution_manifest": request["secondary_execution"], "slots": [slot.__dict__ for slot in proof.slots]}
    if canonical_sha(order) != canonical_sha(expected_order):
        raise ValueError("secondary SDK order/arm/seed differs from the primary-derived full grid")
    schedule_sha = hashlib.sha256(captured["episode-order.json"]).hexdigest()
    design = Artifact.parse(subject["design"]).json(root)
    context = {key: design[key] for key in ("claim_id", "independent_variable", "dependent_variable",
        "control_variables", "data_roles", "planned_at")}
    context.update(experiment_id="continuous-inspection-secondary-" + track + "-v1",
        hypothesis="Prespecified descriptive within-parent intervention analysis; no additional confirmatory tests.")
    if track == "clean":
        specs = designs(cases, context, proof.classic_best)
        for spec in specs.values():
            spec["execution_schedule"] = {"path": "episode-order.json", "sha256": schedule_sha}
    else:
        common = {**context, "cases": cases, "conditions": list(ARMS[track]), "statistical_unit": "case",
            "failure_policy": "penalty", "failure_value": 0., "metric": {"name": "Y", "unit": "quality", "direction": "higher"},
            "statistics": {"estimator": "mean", "interval": "none", "paired": True, "contrasts": [
                {"id": "descriptive-arm-" + str(i), "minuend": arm, "subtrahend": ARMS[track][0]}
                for i, arm in enumerate(ARMS[track][1:], 1)]}}
        specs = {group: {**common, "seeds": seeds, "experiment_id": context["experiment_id"] + "-" + group,
            "adapter_semantics": {"primary_inference": "none; descriptive secondary interventions only",
                "seed_field": "repeat ordinal" if group == "episodes" else "one aggregate pass, not a flight"}}
            for group, seeds in (("episodes", [0, 1, 2]), ("root-means", [0]))}
    for group, spec in specs.items():
        if canonical_sha(capture(group + "/design.json")) != canonical_sha(spec):
            raise ValueError("secondary SDK design differs from the fixed descriptive track")
        document = capture(group + "/results.json")
        if document.get("execution_status") != "completed" or document.get("outcome") != "inconclusive":
            raise ValueError("incomplete or operational secondary recording blocks statistics")
        for row in document["runs"]:
            name = f"{group}/raw/{row['case_id']}-{row['condition']}-{row['seed']}.json"
            if row["raw"] != name or row["status"] != "completed":
                raise ValueError("complete canonical secondary raw identities required")
            raw_rows[group, row["case_id"], row["seed"], row["condition"]] = capture(name)
        results[group] = document
    result_hashes = {group: hashlib.sha256(captured[group + "/results.json"]).hexdigest() for group in specs}
    if track == "clean":
        complete = capture("recording-completion-index.json")
        if canonical_sha(complete) != canonical_sha({"role": "recording_callbacks_complete_not_physical_or_scientific_acceptance",
                "episode_denominator": 216,
                "schedule_sha256": schedule_sha, "results_sha256": result_hashes}):
            raise ValueError("complete clean recording index required")
        expected_done = {"independent_roots": 12, "actual_episode_denominator": 216,
            "classic_episode_denominator": 36, "llm_episode_denominator": 180,
            "outer_aggregate_rows": 96, "conditions": dict(CONDITION_REPEATS), "primary_inference_ready": False}
    else:
        expected_done = {"independent_parent_roots": 12, "additional_actual_episode_denominator": len(proof.slots),
            "root_mean_rows": 12 * len(ARMS[track]), "role": "secondary callbacks only; no confirmatory test or new roots"}
    if canonical_sha(capture("secondary-callbacks-complete.json")) != canonical_sha({**expected_done,
            "track": track, "execution_manifest": request["secondary_execution"], "new_independent_roots": 0}):
        raise ValueError("source-bound complete secondary callback index required")
    sys.path.insert(0, str(Path(request["sdk"]).resolve()))
    from lib.science import _experiment
    with tempfile.TemporaryDirectory(prefix="inspection-secondary-sdk-validation-") as directory:
        temporary = Path(directory)
        for name, raw in captured.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        for group in specs:
            _experiment(temporary, group + "/design.json", group + "/results.json")
    rows, values, resources = [], {}, []
    for slot in proof.slots:
        group = ("classic" if slot.sampler_seed is None else "llm") if track == "clean" else "episodes"
        condition = slot.condition if track == "clean" else slot.arm
        raw = raw_rows[group, slot.case_id, slot.repeat_ordinal, condition]
        if track == "clean":
            receipt = capture(f"order-receipts/{slot.order_index:06d}.json")
            expected = {"role": "measurement_callback_returned_not_independent_flight_receipt",
                **order["slots"][slot.order_index], "schedule_sha256": schedule_sha,
                "raw": f"{group}/raw/{slot.case_id}-{condition}-{slot.repeat_ordinal}.json"}
        else:
            receipt = capture(f"starts/{slot.order_index:06d}.json")
            expected = {"role": "secondary attempt start, not completion", **slot.__dict__}
        if canonical_sha(receipt) != canonical_sha(expected):
            raise ValueError("secondary SDK slot marker differs from the fixed order")
        directory = project / "logs/inspection-formal-episodes.v1" / ("secondary-" + track) / (
            f"{slot.case_id}-{slot.condition}-{slot.arm}-{slot.repeat_ordinal}")
        row, evidence, snapshot = _recount(slot, directory, request, project)
        recorded_evidence = ({**evidence, "adapter_repeat_ordinal": slot.repeat_ordinal,
            "adapter_condition": slot.condition, "order_index": slot.order_index,
            "schedule_sha256": schedule_sha} if track == "clean" else evidence)
        if raw["value"] != row["score"]["Y"] or canonical_sha(raw["evidence"]) != canonical_sha(recorded_evidence):
            raise ValueError("secondary SDK value/evidence differs from real admitted flight receipt")
        row["method"] = "clean:" + slot.condition if track == "clean" else slot.arm
        resource = episode_response_diagnostics(snapshot, request, evidence, project)
        resource["method"] = row["method"]
        resources.append(resource)
        rows.append(row); snapshots.append(snapshot)
        values[slot.case_id, slot.repeat_ordinal, condition] = row["score"]["Y"]
    conditions = CONDITION_REPEATS if track == "clean" else dict.fromkeys(ARMS[track], 3)
    for case in cases:
        for condition, count in conditions.items():
            group = ("classic" if count == 1 else "llm") if track == "clean" else "episodes"
            raw = raw_rows["root-means", case, 0, condition]
            expected = {"aggregation": "arithmetic mean within independent root" if track == "clean"
                else "arithmetic mean within original independent parent root",
                "actual_episode_count": count, "not_an_additional_episode": True,
                "inner_attempts": [{"repeat_ordinal": repeat,
                    "path": f"{group}/raw/{case}-{condition}-{repeat}.json",
                    "sha256": hashlib.sha256(captured[f"{group}/raw/{case}-{condition}-{repeat}.json"]).hexdigest()}
                    for repeat in range(count)], "inner_results": group + "/results.json",
                "inner_results_sha256": result_hashes[group]}
            if track == "clean":
                expected["schedule_sha256"] = schedule_sha
            if (raw["value"] != mean(values[case, repeat, condition] for repeat in range(count))
                    or canonical_sha(raw["evidence"]) != canonical_sha(expected)):
                raise ValueError("secondary root mean is not the complete original repeat aggregation")
    admitted_scores = Artifact.parse(subject["admitted_episode_scores"]).json(root)
    if type(admitted_scores) is not list or len(admitted_scores) != 1296:
        raise ValueError("complete locked main admitted-score artifact required")
    indexed = {(row["case_id"], row["condition"], row["repeat_ordinal"]): row for row in admitted_scores}
    if len(indexed) != 1296:
        raise ValueError("locked main score identities must be unique")
    parents = sorted({slot.parent_order_index for slot in proof.slots})
    for index in parents:
        slot = proof.primary.slots[index]
        directory = project / "logs/inspection-formal-episodes.v1/main" / f"{slot.case_id}-{slot.condition}-{slot.repeat_ordinal}"
        row, evidence, snapshot = _recount(slot, directory, base, project)
        if canonical_sha(row) != canonical_sha(indexed[slot.case_id, slot.condition, slot.repeat_ordinal]):
            raise ValueError("revalidated primary comparator differs from the independently locked main score")
        row["method"] = "event:" + slot.condition if track == "clean" else "primary"
        resource = episode_response_diagnostics(snapshot, base, evidence, project)
        resource["method"] = row["method"]
        resources.append(resource)
        rows.append(row); snapshots.append(snapshot)
    records = {record.case_id: record for record in proof.dataset.records}
    diagnostics = []
    for row, snapshot, resource in zip(rows, snapshots, resources):
        record = records[row["case_id"]]
        parent = decode_formal_population_case(record.case.json(root), "main", record.case_id)
        world = _world(snapshot, project)
        detail = {key: row[key] for key in ("case_id", "domain", "method", "repeat_ordinal", "source_episode")}
        detail["response_resources"] = resource
        if track == "clean":
            rescored = rescore_formal_common_calendar(world, parent, record.case_sha256)
            row["common_score"] = scalar_score(rescored["score"], 3, rescored["score"]["Y"])
            row["common_full_score_sha256"] = canonical_sha(rescored["score"])
            detail["common_calendar"] = rescored
        detail["events"] = (None if row["method"].startswith("clean:") else
            analyze_formal_event_trace(world, parent, record.case_sha256))
        if detail["events"] is None:
            detail["event_scope"] = "clean actor received no event notification; common demand-calendar scores retained"
        diagnostics.append(detail)
    fresh, _, fresh_lock, fresh_subject, _ = _proofs(request, project)
    if fresh != proof or fresh_lock != lock or fresh_subject != subject_ref:
        raise ValueError("original main lock or secondary execution changed during collection")
    for name, raw in captured.items():
        Artifact(str((recording / name).relative_to(project)), hashlib.sha256(raw).hexdigest()).checked_path(project)
    _check_snapshots(snapshots, request, project)
    return proof, lock, subject_ref, rows, captured, results, snapshots, diagnostics


def analyze_secondary(request, recording, *, project_root=ROOT):
    request, _ = read_secondary_request(request)
    project, root = Path(project_root).resolve(), Path(request["run"]).resolve()
    initial, _, _, _, _ = _proofs(request, project)
    track = initial.track
    output = root / "analysis" / ("secondary-" + track + ".v1")
    copied = root / "experiment" / ("secondary-" + track + "-recording.v1")
    projected = root / "experiment" / ("results.secondary-" + track + ".v1.json")
    sys.path.insert(0, str(Path(request["sdk"]).resolve()))
    from lib.production import json_bytes, write_once
    from lib.runtime import safe
    for path in (output, copied, projected):
        safe(root, str(path.relative_to(root)), output=True)
        if path.exists():
            raise ValueError("preserve prior analysis; canonical secondary output must be new")
    lock_path = project / "logs/inspection-formal-episodes.v1/.matrix.lock"
    _ref(lock_path, project)
    with lock_path.open("rb") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof, main_lock, main_subject, rows, captured, results, snapshots, diagnostics = collect_secondary(
            request, recording, project_root=project)
        statistics = analyze_interventions(track, rows)
        events = summarize_formal_events(diagnostics)
        resource_summary = summarize_response_resources([detail["response_resources"] for detail in diagnostics])
        fresh, _, fresh_lock, fresh_subject, _ = _proofs(request, project)
        if fresh != proof or fresh_lock != main_lock or fresh_subject != main_subject:
            raise ValueError("original main lock or secondary execution changed before publication")
        main_subject.checked_path(root); Artifact.parse(main_lock).checked_path(root)
        _check_snapshots(snapshots, request, project)
        for name, raw in sorted(captured.items()):
            path = safe(root, str((copied / name).relative_to(root)), output=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        relative = lambda path: str(path.relative_to(root))
        projection = deepcopy(results["root-means"])
        projection["design"] = relative(copied / "root-means/design.json")
        for row in projection["runs"]:
            row["raw"] = relative(copied / row["raw"])
        write_once(root, relative(projected), json_bytes(projection))
        output.mkdir(parents=True)
        members = []
        with zipfile.ZipFile(output / "receipt-closures.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, (slot, directory, snapshot) in enumerate(snapshots):
                name = f"{index:04d}.json"
                raw = json_bytes({"slot": slot.__dict__, "snapshot": snapshot})
                archive.writestr(name, raw)
                members.append({"member": name, "sha256": hashlib.sha256(raw).hexdigest()})
        detail_members = []
        with zipfile.ZipFile(output / "diagnostic-records.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, detail in enumerate(diagnostics):
                name = f"{index:04d}.json"
                raw = json_bytes(detail)
                archive.writestr(name, raw)
                detail_members.append({"member": name, "sha256": hashlib.sha256(raw).hexdigest(),
                    **{key: detail[key] for key in ("case_id", "method", "repeat_ordinal")}})
        documents = {"statistics.json": statistics, "admitted-episodes.json": rows,
            "event-summary.json": events,
            "response-resource-summary.json": resource_summary,
            "recording-index.json": {"role": "SDK_original_bytes_with_raw_path_only_outer_projection",
                "SDK_relative_reference_root": relative(copied),
                "files": [_ref(copied / name, root) for name in sorted(captured)],
                "projection": _ref(projected, root), "closed_receipt_inputs": {
                    "archive": _ref(output / "receipt-closures.zip", root), "members": members},
                "diagnostic_records": {"archive": _ref(output / "diagnostic-records.zip", root), "members": detail_members}}}
        for name, value in documents.items():
            write_once(root, relative(output / name), json_bytes(value))
        subject = {"schema": "inspection-secondary-descriptive-analysis-subject-v1", "track": track,
            "independent_parent_roots": 12, "new_independent_roots": 0,
            "additional_actual_episode_denominator": len(proof.slots),
            "referenced_primary_episode_denominator": len(snapshots) - len(proof.slots),
            "g4_acceptance": request["g4"], "dataset": request["dataset"],
            "source_manifest": {"path": proof.dataset.namespace.source_manifest.path,
                "sha256": proof.dataset.namespace.source_manifest.sha256},
            "execution": request["secondary_execution"], "primary_execution": request["execution"],
            "main_analysis_lock": main_lock, "locked_main_subject": {"path": main_subject.path, "sha256": main_subject.sha256},
            "recording_results": _ref(projected, root), "recording_design": _ref(copied / "root-means/design.json", root),
            "statistics": _ref(output / "statistics.json", root),
            "event_summary": _ref(output / "event-summary.json", root),
            "response_resource_summary": _ref(output / "response-resource-summary.json", root),
            "admitted_episode_scores": _ref(output / "admitted-episodes.json", root),
            "recording_index": _ref(output / "recording-index.json", root),
            "role": "descriptive_analysis_subject_needs_independent_review_not_G4_or_new_tests"}
        write_once(root, relative(output / "subject.json"), json_bytes(subject))
        return subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_secondary(read_json(args.request), args.recording), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
