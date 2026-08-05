#!/usr/bin/env python3
"""Validate the frozen paper protocol and project-local OPC loop guards.

This module intentionally depends only on the Python standard library so the
guard can run before simulator or training dependencies are available.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "paper" / "protocol"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UNIT_RE = re.compile(r"^-\s+(R\d+\.\d+):\s+([a-z][a-z0-9-]*)\s+—", re.MULTILINE)
FINDING_RE = re.compile(
    r"`(?P<id>(?:RL|RS|SK)-\d{3})`\s+"
    r"(?P<severity>critical|warning|suggestion)/",
    re.IGNORECASE,
)


class ProtocolError(ValueError):
    """A fail-closed protocol violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    require(path.stat().st_size > 0, f"empty JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"malformed JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_ids(items: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    values = list(items)
    ids = [item.get("id") for item in values]
    require(all(isinstance(item_id, str) and item_id for item_id in ids), f"{label} has an empty id")
    duplicates = sorted(item_id for item_id, count in Counter(ids).items() if count != 1)
    require(not duplicates, f"{label} ids are not unique: {duplicates}")
    return {item["id"]: item for item in values}


def validate_method_matrix(matrix: dict[str, Any]) -> None:
    require(matrix.get("schemaVersion") == "ppg.method-matrix.v1", "wrong method-matrix schemaVersion")
    mechanisms = unique_ids(matrix.get("mechanisms", []), "mechanism")
    expected_mechanisms = {
        "recurrence",
        "critic_privilege",
        "belief_supervision",
        "constrained_objective",
        "boundary_curriculum",
    }
    require(set(mechanisms) == expected_mechanisms, "method matrix must define exactly five required mechanisms")
    for mechanism in mechanisms.values():
        require(mechanism.get("switch"), f"mechanism {mechanism['id']} lacks a semantic switch")
        require(mechanism.get("offSemantics"), f"mechanism {mechanism['id']} lacks off semantics")
        require(mechanism.get("claimContrast"), f"mechanism {mechanism['id']} lacks its claim contrast")

    methods = unique_ids(matrix.get("methods", []), "method")
    require(set(methods) == {"B0", "B1", "B2", "B3", "B4", "P"}, "required method identities are B0-B4 and P")
    require(methods["B0"].get("role") == "diagnostic-only", "B0 must remain diagnostic-only")
    require(methods["B1"].get("role") == "diagnostic-only", "B1 must remain diagnostic-only")
    eligible = {method_id for method_id, method in methods.items() if method.get("eligibleForInternalBaselineSelection")}
    require(eligible == {"B2", "B3", "B4"}, "only B2-B4 may enter baseline selection")
    for method_id, method in methods.items():
        flags = method.get("mechanisms")
        require(isinstance(flags, dict) and set(flags) == expected_mechanisms, f"{method_id} mechanism flags are incomplete")
        require(all(isinstance(value, bool) for value in flags.values()), f"{method_id} mechanism flags must be booleans")
    require(all(methods["P"]["mechanisms"].values()), "P must enable all five mechanisms")
    require(methods["B4"]["mechanisms"]["critic_privilege"], "B4 must be the critic-privileged reference")
    require(methods["B4"]["mechanisms"]["recurrence"], "B4 must be recurrent")

    cells = unique_ids(matrix.get("mechanismCells", []), "mechanism cell")
    leave_out = {f"P_MINUS_{name.upper()}" for name in expected_mechanisms}
    require(leave_out.issubset(cells), "every mechanism needs a leave-one-out cell")
    factorials = unique_ids(matrix.get("requiredFactorials", []), "factorial")
    require(
        set(factorials)
        == {"critic_privilege_x_belief_supervision", "constrained_objective_x_boundary_curriculum"},
        "both targeted interactions are required",
    )
    for factorial in factorials.values():
        require(len(factorial.get("factors", [])) == 2, f"{factorial['id']} must name two factors")
        require(len(factorial.get("cells", [])) == 4, f"{factorial['id']} must name four cells")
        require(set(factorial["cells"]).issubset(cells), f"{factorial['id']} references an unknown cell")

    shared = matrix.get("algorithm", {}).get("shared", {})
    for key in (
        "trainingEnvSteps",
        "rolloutSteps",
        "batchSize",
        "trainingSeeds",
        "validationCheckpointSelector",
        "actorNormalization",
        "criticNormalization",
        "recurrent",
        "deploymentExport",
    ):
        require(key in shared, f"shared method identity missing {key}")
    require(shared["trainingSeeds"] == [11, 23, 47, 83, 131], "training seeds are not frozen")
    require(shared["criticNormalization"].get("sharedWithActor") is False, "actor and critic normalizers must be separate")
    comparison = matrix.get("comparisonPolicy", {})
    require(comparison.get("primary") == "P_vs_frozen_selected_internal_baseline", "wrong primary comparison")
    require(comparison.get("baselineCandidates") == ["B2", "B3", "B4"], "wrong baseline candidate set")


def validate_evaluation_protocol(protocol: dict[str, Any]) -> None:
    require(protocol.get("schemaVersion") == "ppg.evaluation-protocol.v1", "wrong evaluation protocol schemaVersion")
    sensor = protocol.get("deploymentSensorSuite", {})
    require(sensor.get("modalities") == ["rgb", "depth", "proprioception"], "deployment sensor suite must be synchronized RGB-D plus proprioception")
    forbidden_depth = set(sensor.get("depth", {}).get("forbiddenSources", []))
    require(
        {"renderer-z-buffer-unnoised", "segmentation-object-id", "simulator-body-id"}.issubset(forbidden_depth),
        "depth privilege sources are not fully forbidden",
    )
    observations = protocol.get("observationContracts", {})
    actor_fields = set(observations.get("actor", {}).get("fields", []))
    evaluator_fields = set(observations.get("evaluator", {}).get("fields", []))
    require(actor_fields and evaluator_fields, "actor and evaluator field sets must be nonempty")
    require(actor_fields.isdisjoint(evaluator_fields), "actor and evaluator fields must be disjoint")
    forbidden_actor = {
        "adapter_outcome",
        "association_outcome",
        "detector_exception_type",
        "target_truth_body_m",
        "realized_wind_enu_mps",
        "renderer_depth_unnoised",
        "simulator_object_ids",
    }
    require(actor_fields.isdisjoint(forbidden_actor), "actor includes evaluator-only truth or failure metadata")
    missing = observations.get("actor", {}).get("missing", {})
    require(missing.get("allFailureCausesShareRepresentation") is True, "missing detections must use one canonical actor representation")
    require(observations.get("actionPathInvariant"), "critic-only action-path invariance is unspecified")

    association = protocol.get("targetAssociation", {})
    require(association.get("tieBehavior") == "missing", "ambiguous association must return missing")
    require("target_truth" in association.get("forbiddenInputs", []), "target association must forbid truth")
    require(association.get("maximumMissingSteps", 0) > 0, "association expiry must be positive")
    degradation = protocol.get("degradedTruth", {})
    require(degradation.get("rng", {}).get("methodIndependent") is True, "degradation RNG must be method-independent")
    require("method_id" in degradation.get("forbiddenDependencies", []), "degradation must forbid method-id dependence")
    require(degradation.get("freezeReceipt", {}).get("immutableBeforeComparativeEvaluation") is True, "degradation receipt must freeze before comparison")
    curriculum = protocol.get("curriculum", {})
    require(set(curriculum.get("allowedFeedback", [])), "curriculum training feedback must be declared")
    require("validation_manifest" in curriculum.get("forbiddenFeedback", []), "validation cannot update curriculum")
    require("confirmatory_manifest" in curriculum.get("forbiddenFeedback", []), "confirmation cannot update curriculum")
    require(curriculum.get("minimumStratumProbability", 0) > 0, "curriculum needs positive minimum support")

    matrix = protocol.get("evaluationMatrix", {})
    require(len(matrix.get("cellIds", [])) == 9, "evaluation matrix must freeze nine cell IDs")
    require(len(set(matrix.get("cellIds", []))) == 9, "evaluation cell IDs must be unique")
    require(set(matrix.get("hardSafetyCellIds", [])).issubset(matrix["cellIds"]), "unknown hard-safety cell")
    episode_count = matrix.get("episodeCount", {})
    allowed_counts = episode_count.get("allowedPerCell", [])
    require(allowed_counts and all(isinstance(value, int) and value > 0 for value in allowed_counts), "episode count choices must be positive integers")
    require(len(set(allowed_counts)) == len(allowed_counts), "episode count choices must be unique")
    require(matrix.get("requiredManifestKey") == ["method_id", "training_seed", "cell_id", "scenario_id", "episode_seed", "adapter_id"], "paired manifest key is not frozen")
    require("NONPASS" in matrix.get("completeness", ""), "incomplete records must force NONPASS")

    statistics = protocol.get("statistics", {})
    bootstrap = statistics.get("bootstrap", {})
    require(bootstrap.get("type") == "hierarchical-paired-percentile", "bootstrap must be hierarchical, paired, and percentile")
    require(bootstrap.get("replicates") == 10000, "bootstrap must use 10,000 replicates")
    require(bootstrap.get("rngSeed") == 20260805, "bootstrap RNG seed is not frozen")
    require(statistics.get("minimumTaskEffect") == 0.05, "primary effect threshold must be five percentage points")
    required_safety = {"collision_rate", "out_of_bounds_rate", "timeout_rate", "minimum_clearance_m_difference", "reacquisition_time_s_difference", "control_smoothness_difference"}
    require(set(statistics.get("safetyNonInferiorityMargins", {})) == required_safety, "all safety margins must be frozen")
    require(set(statistics.get("hardEvents", {})) == {"collision", "out_of_bounds", "clearance_violation"}, "hard-event definitions are incomplete")

    claims = protocol.get("claimPolicy", {})
    require(claims.get("maximumAllowedPerformanceClaim") == "best among the frozen evaluated internal methods", "claim vocabulary is too broad")
    forbidden_phrases = {phrase.lower() for phrase in claims.get("forbiddenPhrases", [])}
    require("sota" in forbidden_phrases and "safety superiority" in forbidden_phrases and "deployment-ready" in forbidden_phrases, "forbidden claim vocabulary is incomplete")
    selection = protocol.get("baselineSelection", {})
    require(selection.get("eligibleMethodIds") == ["B2", "B3", "B4"], "baseline selection candidate set is wrong")
    require(selection.get("freezeBeforeConfirmatoryInspection") is True, "baseline selection must freeze before confirmation")
    require("receipt_sha256" in selection.get("receiptRequiredFields", []), "baseline selection receipt must be content-addressed")
    lifecycle = protocol.get("evidenceLifecycle", {})
    require(lifecycle.get("hashAlgorithm") == "sha256", "evidence hashing must use SHA-256")
    require(lifecycle.get("uriTemplate") == "cas://sha256/{digest}", "evidence URI must be content-addressed")
    require(lifecycle.get("missingBehavior", "").endswith("NONPASS"), "missing evidence must force NONPASS")
    require(lifecycle.get("corruptBehavior", "").endswith("NONPASS"), "corrupt evidence must force NONPASS")
    timing = protocol.get("pilotTimingGate", {})
    require(timing.get("budgetSeconds") == 86400, "pilot gate must enforce the 24-hour budget")
    require(timing.get("reserveFractionMinimum", 0) >= 0.15, "pilot timing reserve must be at least 15%")


def validate_loop_guard(guard: dict[str, Any]) -> None:
    require(guard.get("schemaVersion") == "ppg.loop-guard.v1", "wrong loop-guard schemaVersion")
    known_types = set(guard.get("knownUnitTypes", []))
    require(known_types, "known unit types are empty")
    units = unique_ids(guard.get("units", []), "loop unit")
    require(all(unit.get("type") in known_types for unit in units.values()), "loop guard contains an unknown unit type")
    require(len(units) == 25, "loop guard must cover all 25 declared units")
    for unit in units.values():
        review_id = unit.get("requiresReview")
        if review_id is not None:
            require(review_id in units, f"{unit['id']} references unknown review {review_id}")
            require(units[review_id]["type"] == "review", f"{review_id} is not a review unit")
    fixes = {entry.get("unit"): entry for entry in guard.get("fixClosure", [])}
    require(set(fixes) == {"R1.3", "R2.4", "R4.3", "R5.5"}, "all fix units need closure rules")
    evidence_units = {entry.get("unit") for entry in guard.get("evidenceRequirements", [])}
    require({"R5.1", "R5.3", "R6.3", "R6.4"}.issubset(evidence_units), "experiment/final evidence guard coverage is incomplete")


def validate_contracts(protocol_dir: Path = PROTOCOL_DIR) -> None:
    validate_method_matrix(load_json(protocol_dir / "method-matrix.json"))
    validate_evaluation_protocol(load_json(protocol_dir / "evaluation-protocol.json"))
    validate_loop_guard(load_json(protocol_dir / "loop-guard.json"))


def parse_plan_units(plan_path: Path) -> dict[str, str]:
    require(plan_path.is_file() and plan_path.stat().st_size > 0, f"missing/empty session plan: {plan_path}")
    units = UNIT_RE.findall(plan_path.read_text(encoding="utf-8"))
    ids = [unit_id for unit_id, _ in units]
    require(units, f"no units parsed from {plan_path}")
    require(len(ids) == len(set(ids)), "duplicate unit IDs in session plan")
    return dict(units)


def validate_session_state(session: Path, guarded_units: dict[str, dict[str, Any]]) -> None:
    state_path = session / "loop-state.json"
    if not state_path.exists():
        return
    state = load_json(state_path)
    for key in ("unit", "next_unit"):
        value = state.get(key)
        if value is not None:
            require(value in guarded_units, f"loop-state {key} contains unknown unit {value}")
    unit_ids = state.get("unit_ids", [])
    require(isinstance(unit_ids, list), "loop-state unit_ids must be a list")
    unknown = sorted(set(unit_ids) - set(guarded_units))
    require(not unknown, f"loop-state contains unknown unit IDs: {unknown}")


def validate_nonempty(path: Path) -> None:
    require(path.is_file(), f"required artifact is missing: {path}")
    require(path.stat().st_size > 0, f"required artifact is empty: {path}")
    if path.suffix == ".json":
        load_json(path)


def parse_iso8601(value: Any, label: str) -> dt.datetime:
    require(isinstance(value, str) and value, f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolError(f"{label} is not ISO-8601: {value}") from exc
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def validate_pilot_timing(path: Path) -> None:
    receipt = load_json(path)
    require(receipt.get("schemaVersion") == "ppg.pilot-timing.v1", "wrong pilot timing schema")
    numeric = [
        "training_run_seconds",
        "checkpoint_eval_seconds",
        "fastsam_episode_seconds",
        "declared_training_runs",
        "declared_evaluation_episodes",
        "analysis_seconds",
        "reserve_seconds",
        "remaining_budget_seconds",
    ]
    require(all(isinstance(receipt.get(key), (int, float)) and receipt[key] >= 0 for key in numeric), "pilot timing fields must be nonnegative numbers")
    total = (
        receipt["declared_training_runs"] * receipt["training_run_seconds"]
        + receipt["declared_evaluation_episodes"] * receipt["fastsam_episode_seconds"]
        + receipt["checkpoint_eval_seconds"]
        + receipt["analysis_seconds"]
        + receipt["reserve_seconds"]
    )
    require(receipt["reserve_seconds"] >= 0.15 * 86400, "pilot timing reserve is below 15% of the fixed budget")
    require(total <= receipt["remaining_budget_seconds"] <= 86400, "full workload plus reserve does not fit the remaining 24-hour budget")


def validate_confirmatory_manifest(path: Path) -> None:
    manifest = load_json(path)
    require(manifest.get("schemaVersion") == "ppg.confirmatory-manifest.v1", "wrong confirmatory manifest schema")
    require(manifest.get("frozen") is True, "confirmatory manifest is not frozen")
    require(isinstance(manifest.get("episodesPerCell"), int) and manifest["episodesPerCell"] in {16, 32, 64}, "confirmatory episode count is not an allowed sealed value")
    records = manifest.get("expectedRecords")
    require(isinstance(records, list) and records, "confirmatory manifest has no expected records")
    required_key = {"method_id", "training_seed", "cell_id", "scenario_id", "episode_seed", "adapter_id"}
    keys = []
    for record in records:
        require(isinstance(record, dict) and required_key.issubset(record), "confirmatory manifest record is incomplete")
        keys.append(tuple(record[name] for name in sorted(required_key)))
    require(len(keys) == len(set(keys)), "confirmatory manifest contains duplicate expected keys")


def validate_research_gate(path: Path) -> None:
    gate = load_json(path)
    require(gate.get("schemaVersion") == "ppg.research-gate.v1", "wrong research gate schema")
    require(gate.get("status") in {"PASS", "NONPASS"}, "research gate status must be PASS or NONPASS")
    require(SHA256_RE.fullmatch(str(gate.get("manifestSha256", ""))) is not None, "research gate manifest hash is invalid")
    evidence_ids = gate.get("evidenceIds")
    require(isinstance(evidence_ids, list) and evidence_ids and len(evidence_ids) == len(set(evidence_ids)), "research gate evidence IDs must be nonempty and unique")
    states = gate.get("requiredRecordStateSummary", {})
    required_states = {"completed", "failed", "blocked", "unrun", "corrupt"}
    require(set(states) == required_states, "research gate record-state summary is incomplete")
    require(all(isinstance(value, int) and value >= 0 for value in states.values()), "research gate state counts must be nonnegative integers")
    if gate["status"] == "PASS":
        require(all(states[state] == 0 for state in required_states - {"completed"}), "PASS gate contains non-completed required records")


def git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def validate_final_evidence(path: Path, repo_root: Path) -> None:
    receipt = load_json(path)
    require(receipt.get("schemaVersion") == "ppg.final-evidence.v1", "wrong final evidence schema")
    generated_at = parse_iso8601(receipt.get("generatedAt"), "generatedAt")
    now = dt.datetime.now(dt.timezone.utc)
    require(generated_at <= now + dt.timedelta(minutes=5), "final evidence timestamp is in the future")
    require(now - generated_at <= dt.timedelta(hours=24), "final evidence is stale")
    source_revision = receipt.get("sourceRevision")
    require(isinstance(source_revision, str) and len(source_revision) == 40, "final evidence sourceRevision is invalid")
    head = git_head(repo_root)
    if head is not None:
        require(source_revision == head, "final evidence sourceRevision is not current HEAD")
    evidence_ids = receipt.get("evidenceIds")
    require(isinstance(evidence_ids, list) and evidence_ids and len(evidence_ids) == len(set(evidence_ids)), "final evidence IDs must be nonempty and unique")
    require(receipt.get("terminalState") in {"paper", "nonpass"}, "invalid terminal state")
    require(receipt.get("openFindings") == [], "final evidence has open findings")
    verdicts = receipt.get("reviewVerdicts", {})
    for review_id in ("R1.4", "R3.2", "R5.2", "R5.4", "R6.2"):
        require(verdicts.get(review_id) == "PASS", f"final evidence lacks PASS for {review_id}")
    upstream = receipt.get("upstreamArtifacts")
    require(isinstance(upstream, list) and upstream, "final evidence has no upstream artifacts")
    newest_mtime = 0.0
    for item in upstream:
        require(isinstance(item, dict) and set(item) >= {"evidenceId", "path", "sha256"}, "invalid upstream artifact entry")
        artifact = (repo_root / item["path"]).resolve()
        require(artifact.is_relative_to(repo_root.resolve()), f"upstream artifact escapes repository: {item['path']}")
        validate_nonempty(artifact)
        require(SHA256_RE.fullmatch(str(item["sha256"])) is not None, f"invalid SHA-256 for {item['path']}")
        require(sha256_file(artifact) == item["sha256"], f"hash mismatch for {item['path']}")
        newest_mtime = max(newest_mtime, artifact.stat().st_mtime)
    require(generated_at.timestamp() >= newest_mtime, "final evidence predates an upstream artifact")


def resolve_required_artifact(spec: dict[str, Any], session: Path, repo_root: Path) -> Path:
    scope = spec.get("scope")
    require(scope in {"repo", "session"}, f"unknown artifact scope: {scope}")
    base = repo_root if scope == "repo" else session
    candidate = (base / str(spec.get("path", ""))).resolve()
    require(candidate.is_relative_to(base.resolve()), f"required artifact escapes {scope} root")
    return candidate


def validate_evidence_kind(kind: str, path: Path, repo_root: Path) -> None:
    validate_nonempty(path)
    if kind == "nonempty":
        return
    validators = {
        "pilot-timing": validate_pilot_timing,
        "confirmatory-manifest": validate_confirmatory_manifest,
        "research-gate": validate_research_gate,
    }
    if kind == "final-evidence":
        validate_final_evidence(path, repo_root)
        return
    require(kind in validators, f"unknown evidence kind: {kind}")
    validators[kind](path)


def validate_protected_review(session: Path, review_id: str, guard: dict[str, Any]) -> None:
    template = guard.get("protectedReview", {}).get("handshakePathTemplate", "")
    handshake_path = session / template.format(review=review_id)
    handshake = load_json(handshake_path)
    require(handshake.get("nodeId") == review_id, f"review handshake node mismatch for {review_id}")
    require(handshake.get("nodeType") == "review", f"{review_id} handshake is not a review")
    require(handshake.get("status") == "completed", f"{review_id} review is incomplete")
    require(handshake.get("verdict") == guard["protectedReview"]["acceptedVerdict"], f"protected review {review_id} is not PASS")
    findings = handshake.get("findings", {})
    require(findings.get("critical") == 0, f"protected review {review_id} is not critical-clean")
    require(findings.get("warning") == 0, f"protected review {review_id} is not warning-clean")


def extract_findings(ledger_path: Path) -> dict[str, str]:
    validate_nonempty(ledger_path)
    findings = [(match.group("id"), match.group("severity").lower()) for match in FINDING_RE.finditer(ledger_path.read_text(encoding="utf-8"))]
    require(findings, f"no findings parsed from {ledger_path}")
    counts = Counter(finding_id for finding_id, _ in findings)
    require(all(count == 1 for count in counts.values()), f"ledger finding IDs are not unique: {counts}")
    return dict(findings)


def parse_disposition(path: Path) -> list[tuple[str, str, str]]:
    validate_nonempty(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*((?:RL|RS|SK)-\d{3})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if match:
            rows.append((match.group(1), match.group(2).strip().lower(), match.group(3).strip()))
    return rows


def resolve_evidence_path(value: str, session: Path, repo_root: Path) -> Path:
    require(value and value != "—", "disposition evidence path is empty")
    base_value = value.split("#", 1)[0]
    raw = Path(base_value)
    if raw.is_absolute():
        return raw.resolve()
    if base_value.startswith("nodes/") or base_value == "acceptance-criteria.md":
        return (session / raw).resolve()
    return (repo_root / raw).resolve()


def validate_fix_closure(session: Path, repo_root: Path, rule: dict[str, Any]) -> None:
    ledger = extract_findings(session / rule["ledger"])
    disposition_path = session / rule["disposition"]
    rows = parse_disposition(disposition_path)
    counts = Counter(finding_id for finding_id, _, _ in rows)
    require(set(counts) == set(ledger), "disposition IDs do not exactly match the upstream ledger")
    require(all(count == 1 for count in counts.values()), "each finding must have exactly one disposition row")
    for finding_id, status, evidence in rows:
        expected = "retained" if ledger[finding_id] == "suggestion" else "addressed"
        require(status == expected, f"{finding_id} must be {expected}, got {status}")
        evidence_path = resolve_evidence_path(evidence, session, repo_root)
        validate_nonempty(evidence_path)


def validate_loop_complete(
    session: Path,
    unit_id: str,
    artifacts: Iterable[Path],
    *,
    repo_root: Path = REPO_ROOT,
    protocol_dir: Path = PROTOCOL_DIR,
) -> None:
    validate_contracts(protocol_dir)
    guard = load_json(protocol_dir / "loop-guard.json")
    units = unique_ids(guard["units"], "loop unit")
    require(unit_id in units, f"unknown unit ID: {unit_id}")
    plan_units = parse_plan_units(session / "plan.md")
    unknown_plan_ids = sorted(set(plan_units) - set(units))
    require(not unknown_plan_ids, f"session plan contains unknown unit IDs: {unknown_plan_ids}")
    unknown_plan_types = sorted(set(plan_units.values()) - set(guard["knownUnitTypes"]))
    require(not unknown_plan_types, f"session plan contains unknown unit types: {unknown_plan_types}")
    require(unit_id in plan_units, f"unit {unit_id} is absent from the session plan")
    require(plan_units[unit_id] == units[unit_id]["type"], f"unit type mismatch for {unit_id}")
    validate_session_state(session, units)

    declared = {Path(path).resolve() for path in artifacts}
    require(declared, "at least one artifact must be declared")
    for artifact in declared:
        validate_nonempty(artifact)

    required_review = units[unit_id].get("requiresReview")
    if required_review:
        validate_protected_review(session, required_review, guard)

    fix_rules = {entry["unit"]: entry for entry in guard.get("fixClosure", [])}
    if unit_id in fix_rules:
        disposition = (session / fix_rules[unit_id]["disposition"]).resolve()
        require(disposition in declared, f"fix unit must declare disposition artifact: {disposition}")
        validate_fix_closure(session, repo_root, fix_rules[unit_id])

    evidence_rules = {entry["unit"]: entry for entry in guard.get("evidenceRequirements", [])}
    for artifact_spec in evidence_rules.get(unit_id, {}).get("artifacts", []):
        required_path = resolve_required_artifact(artifact_spec, session, repo_root)
        require(required_path in declared, f"required evidence artifact was not declared: {required_path}")
        validate_evidence_kind(artifact_spec["kind"], required_path, repo_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("contracts", help="validate all three protocol contracts")
    loop_parser = subparsers.add_parser("loop-complete", help="validate a unit before heartbeat completion")
    loop_parser.add_argument("--session", type=Path, required=True)
    loop_parser.add_argument("--unit", required=True)
    loop_parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "contracts":
            validate_contracts()
            print("OK: method matrix, evaluation protocol, and loop guard contracts are valid")
        else:
            validate_loop_complete(args.session.resolve(), args.unit, args.artifacts)
            print(f"OK: loop completion guard passed for {args.unit}")
    except ProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
