from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts import validate_paper_protocol as validator


SOURCE_PROTOCOL = Path(__file__).resolve().parents[1] / "paper" / "protocol"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def protocol_copy(tmp_path: Path) -> Path:
    target = tmp_path / "protocol"
    shutil.copytree(SOURCE_PROTOCOL, target)
    return target


def make_session(tmp_path: Path, unit_id: str, unit_type: str) -> Path:
    session = tmp_path / "session"
    session.mkdir(parents=True)
    (session / "plan.md").write_text(
        f"# Plan\n\n## Units\n\n- {unit_id}: {unit_type} — fixture unit\n",
        encoding="utf-8",
    )
    return session


def make_artifact(path: Path, text: str = "evidence\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_pass_review(session: Path, review_id: str) -> Path:
    path = session / "nodes" / review_id / "handshake.json"
    write_json(
        path,
        {
            "nodeId": review_id,
            "nodeType": "review",
            "status": "completed",
            "verdict": "PASS",
            "findings": {"critical": 0, "warning": 0, "suggestion": 0},
        },
    )
    return path


def run_loop(
    session: Path,
    unit_id: str,
    artifacts: list[Path],
    repo: Path,
    protocol: Path,
) -> None:
    validator.validate_loop_complete(
        session,
        unit_id,
        artifacts,
        repo_root=repo,
        protocol_dir=protocol,
    )


def test_contracts_positive_repository_protocols() -> None:
    validator.validate_contracts(SOURCE_PROTOCOL)


def test_contracts_negative_actor_evaluator_overlap(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    path = protocol / "evaluation-protocol.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["observationContracts"]["actor"]["fields"].append("adapter_outcome")
    write_json(path, data)
    with pytest.raises(validator.ProtocolError, match="disjoint"):
        validator.validate_contracts(protocol)


def test_contracts_boundary_reject_zero_episode_choice(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    path = protocol / "evaluation-protocol.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["evaluationMatrix"]["episodeCount"]["allowedPerCell"] = [0, 16]
    write_json(path, data)
    with pytest.raises(validator.ProtocolError, match="positive integers"):
        validator.validate_contracts(protocol)


def test_contracts_boundary_reject_malformed_json(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    (protocol / "method-matrix.json").write_text("{", encoding="utf-8")
    with pytest.raises(validator.ProtocolError, match="malformed JSON"):
        validator.validate_contracts(protocol)


def test_artifact_guard_positive_nonempty_file(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R1.1", "spec")
    artifact = make_artifact(tmp_path / "repo" / "artifact.md")
    run_loop(session, "R1.1", [artifact], tmp_path / "repo", protocol)


def test_artifact_guard_negative_missing_file(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R1.1", "spec")
    with pytest.raises(validator.ProtocolError, match="missing"):
        run_loop(session, "R1.1", [tmp_path / "missing.md"], tmp_path / "repo", protocol)


def test_artifact_guard_boundary_empty_file(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R1.1", "spec")
    artifact = make_artifact(tmp_path / "repo" / "empty.md", "")
    with pytest.raises(validator.ProtocolError, match="empty"):
        run_loop(session, "R1.1", [artifact], tmp_path / "repo", protocol)


def test_review_guard_positive_pass_and_clean(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R2.1", "test-design")
    make_pass_review(session, "R1.4")
    artifact = make_artifact(tmp_path / "repo" / "test-plan.md")
    run_loop(session, "R2.1", [artifact], tmp_path / "repo", protocol)


def test_review_guard_negative_nonpass(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R2.1", "test-design")
    handshake = make_pass_review(session, "R1.4")
    data = json.loads(handshake.read_text(encoding="utf-8"))
    data["verdict"] = "FAIL"
    write_json(handshake, data)
    artifact = make_artifact(tmp_path / "repo" / "test-plan.md")
    with pytest.raises(validator.ProtocolError, match="not PASS"):
        run_loop(session, "R2.1", [artifact], tmp_path / "repo", protocol)


def test_review_guard_boundary_pass_with_warning_is_not_clean(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R2.1", "test-design")
    handshake = make_pass_review(session, "R1.4")
    data = json.loads(handshake.read_text(encoding="utf-8"))
    data["findings"]["warning"] = 1
    write_json(handshake, data)
    artifact = make_artifact(tmp_path / "repo" / "test-plan.md")
    with pytest.raises(validator.ProtocolError, match="warning-clean"):
        run_loop(session, "R2.1", [artifact], tmp_path / "repo", protocol)


def make_fix_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R1.3", "fix")
    repo = tmp_path / "repo"
    evidence = make_artifact(repo / "evidence.md")
    ledger = session / "nodes" / "R1.2" / "finding-ledger.md"
    make_artifact(
        ledger,
        "- `RL-001` critical/open — leak\n"
        "- `RS-001` suggestion/retain — positive control\n"
        "- `SK-001` warning/open — guard\n",
    )
    disposition = session / "nodes" / "R1.3" / "run_1" / "disposition.md"
    make_artifact(
        disposition,
        "| ID | Status | Evidence |\n"
        "|---|---|---|\n"
        "| RL-001 | addressed | evidence.md |\n"
        "| RS-001 | retained | evidence.md |\n"
        "| SK-001 | addressed | evidence.md |\n",
    )
    return protocol, session, repo, disposition


def test_fix_guard_positive_exact_disposition(tmp_path: Path) -> None:
    protocol, session, repo, disposition = make_fix_fixture(tmp_path)
    run_loop(session, "R1.3", [disposition], repo, protocol)


def test_fix_guard_negative_missing_disposition_row(tmp_path: Path) -> None:
    protocol, session, repo, disposition = make_fix_fixture(tmp_path)
    text = disposition.read_text(encoding="utf-8").replace("| SK-001 | addressed | evidence.md |\n", "")
    disposition.write_text(text, encoding="utf-8")
    with pytest.raises(validator.ProtocolError, match="exactly match"):
        run_loop(session, "R1.3", [disposition], repo, protocol)


def test_fix_guard_boundary_duplicate_and_suggestion_status(tmp_path: Path) -> None:
    protocol, session, repo, disposition = make_fix_fixture(tmp_path)
    text = disposition.read_text(encoding="utf-8")
    disposition.write_text(
        text.replace("| RS-001 | retained", "| RS-001 | addressed")
        + "| RL-001 | addressed | evidence.md |\n",
        encoding="utf-8",
    )
    with pytest.raises(validator.ProtocolError, match="exactly one"):
        run_loop(session, "R1.3", [disposition], repo, protocol)


def make_pilot_receipt(path: Path) -> Path:
    write_json(
        path,
        {
            "schemaVersion": "ppg.pilot-timing.v1",
            "training_run_seconds": 100,
            "checkpoint_eval_seconds": 50,
            "fastsam_episode_seconds": 1,
            "declared_training_runs": 10,
            "declared_evaluation_episodes": 100,
            "analysis_seconds": 50,
            "reserve_seconds": 13000,
            "remaining_budget_seconds": 20000,
        },
    )
    return path


def make_pilot_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[Path]]:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R5.1", "experiment-pilot")
    make_pass_review(session, "R4.2")
    repo = tmp_path / "repo"
    report = make_artifact(repo / "paper" / "protocol" / "pilot-report.md")
    receipt = make_pilot_receipt(repo / "paper" / "protocol" / "pilot-timing-receipt.json")
    return protocol, session, repo, [report, receipt]


def test_experiment_guard_positive_timing_receipt(tmp_path: Path) -> None:
    protocol, session, repo, artifacts = make_pilot_fixture(tmp_path)
    run_loop(session, "R5.1", artifacts, repo, protocol)


def test_experiment_guard_negative_undeclared_required_evidence(tmp_path: Path) -> None:
    protocol, session, repo, artifacts = make_pilot_fixture(tmp_path)
    with pytest.raises(validator.ProtocolError, match="not declared"):
        run_loop(session, "R5.1", artifacts[:1], repo, protocol)


def test_experiment_guard_boundary_budget_exactly_exceeded(tmp_path: Path) -> None:
    protocol, session, repo, artifacts = make_pilot_fixture(tmp_path)
    receipt = artifacts[1]
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["remaining_budget_seconds"] = 14000
    write_json(receipt, data)
    with pytest.raises(validator.ProtocolError, match="does not fit"):
        run_loop(session, "R5.1", artifacts, repo, protocol)


def make_confirmatory_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[Path]]:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R5.3", "experiment-confirmatory")
    make_pass_review(session, "R5.2")
    repo = tmp_path / "repo"
    manifest = repo / "paper" / "protocol" / "confirmatory-manifest.json"
    write_json(
        manifest,
        {
            "schemaVersion": "ppg.confirmatory-manifest.v1",
            "frozen": True,
            "episodesPerCell": 16,
            "expectedRecords": [
                {
                    "method_id": "P",
                    "training_seed": 11,
                    "cell_id": "C_IDEAL_CLEAN",
                    "scenario_id": "S001",
                    "episode_seed": 1001,
                    "adapter_id": "ideal_rgbd",
                }
            ],
        },
    )
    gate = repo / "paper" / "results" / "research-gate.json"
    write_json(
        gate,
        {
            "schemaVersion": "ppg.research-gate.v1",
            "status": "PASS",
            "manifestSha256": "0" * 64,
            "evidenceIds": ["EV-001"],
            "requiredRecordStateSummary": {
                "completed": 1,
                "failed": 0,
                "blocked": 0,
                "unrun": 0,
                "corrupt": 0,
            },
        },
    )
    return protocol, session, repo, [manifest, gate]


def test_confirmatory_guard_positive_complete_gate(tmp_path: Path) -> None:
    protocol, session, repo, artifacts = make_confirmatory_fixture(tmp_path)
    run_loop(session, "R5.3", artifacts, repo, protocol)


def test_confirmatory_guard_negative_pass_with_failed_record(tmp_path: Path) -> None:
    protocol, session, repo, artifacts = make_confirmatory_fixture(tmp_path)
    gate = artifacts[1]
    data = json.loads(gate.read_text(encoding="utf-8"))
    data["requiredRecordStateSummary"]["failed"] = 1
    write_json(gate, data)
    with pytest.raises(validator.ProtocolError, match="non-completed"):
        run_loop(session, "R5.3", artifacts, repo, protocol)


def test_confirmatory_guard_boundary_unsealed_episode_count(tmp_path: Path) -> None:
    protocol, session, repo, artifacts = make_confirmatory_fixture(tmp_path)
    manifest = artifacts[0]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["episodesPerCell"] = 15
    write_json(manifest, data)
    with pytest.raises(validator.ProtocolError, match="allowed sealed value"):
        run_loop(session, "R5.3", artifacts, repo, protocol)


def make_final_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[Path], Path]:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R6.4", "acceptance")
    make_pass_review(session, "R6.2")
    repo = tmp_path / "repo"
    upstream = repo / "paper" / "results" / "research-gate.json"
    write_json(upstream, {"schemaVersion": "ppg.research-gate.v1", "status": "NONPASS"})
    final_md = make_artifact(repo / "paper" / "FINAL.md")
    final_json = repo / "paper" / "evidence" / "final-verification.json"
    write_json(
        final_json,
        {
            "schemaVersion": "ppg.final-evidence.v1",
            "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sourceRevision": "0" * 40,
            "evidenceIds": ["EV-FINAL-001"],
            "terminalState": "nonpass",
            "openFindings": [],
            "reviewVerdicts": {review: "PASS" for review in ("R1.4", "R3.2", "R5.2", "R5.4", "R6.2")},
            "upstreamArtifacts": [
                {
                    "evidenceId": "EV-GATE-001",
                    "path": "paper/results/research-gate.json",
                    "sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    return protocol, session, repo, [final_md, final_json], final_json


def test_final_guard_positive_hashes_and_evidence_ids(tmp_path: Path) -> None:
    protocol, session, repo, artifacts, _ = make_final_fixture(tmp_path)
    run_loop(session, "R6.4", artifacts, repo, protocol)


def test_final_guard_negative_hash_mismatch(tmp_path: Path) -> None:
    protocol, session, repo, artifacts, final_json = make_final_fixture(tmp_path)
    data = json.loads(final_json.read_text(encoding="utf-8"))
    data["upstreamArtifacts"][0]["sha256"] = "f" * 64
    write_json(final_json, data)
    with pytest.raises(validator.ProtocolError, match="hash mismatch"):
        run_loop(session, "R6.4", artifacts, repo, protocol)


def test_final_guard_boundary_stale_receipt(tmp_path: Path) -> None:
    protocol, session, repo, artifacts, final_json = make_final_fixture(tmp_path)
    data = json.loads(final_json.read_text(encoding="utf-8"))
    data["generatedAt"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=25)).isoformat()
    write_json(final_json, data)
    with pytest.raises(validator.ProtocolError, match="stale"):
        run_loop(session, "R6.4", artifacts, repo, protocol)


def test_unit_guard_negative_unknown_unit_id(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R9.9", "spec")
    artifact = make_artifact(tmp_path / "repo" / "artifact.md")
    with pytest.raises(validator.ProtocolError, match="unknown unit ID"):
        run_loop(session, "R9.9", [artifact], tmp_path / "repo", protocol)


def test_unit_guard_boundary_unknown_unit_type(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R1.1", "mystery")
    artifact = make_artifact(tmp_path / "repo" / "artifact.md")
    with pytest.raises(validator.ProtocolError, match="unknown unit types"):
        run_loop(session, "R1.1", [artifact], tmp_path / "repo", protocol)


def test_unit_guard_boundary_malformed_loop_state(tmp_path: Path) -> None:
    protocol = protocol_copy(tmp_path)
    session = make_session(tmp_path, "R1.1", "spec")
    (session / "loop-state.json").write_text("{", encoding="utf-8")
    artifact = make_artifact(tmp_path / "repo" / "artifact.md")
    with pytest.raises(validator.ProtocolError, match="malformed JSON"):
        run_loop(session, "R1.1", [artifact], tmp_path / "repo", protocol)
