"""Synthetic owner-state and blob guards; never load or invoke an MLX model."""

import hashlib
import json

import pytest

from eval.inspection_model_service import _digest, admitted_limits, verify_model_files


def owner_fixture(tmp_path):
    receipt = tmp_path/"download.json"
    receipt.write_text('{"role":"synthetic fixture, no model"}')
    receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    profile = {"download_receipt_sha256": receipt_hash, "file_count": 1}
    registry = tmp_path/"registry.json"
    registry.write_text(json.dumps({"models": {"fixture": profile}}))
    registry_hash = hashlib.sha256(registry.read_bytes()).hexdigest()
    plan = {"stage": "plan", "local_model_registry": registry.name, "local_model_registry_sha256": registry_hash}
    purpose = {"kind": "inspection_local_generation", "phase": "development",
               "model_registry_sha256": registry_hash, "model_receipt_sha256": receipt_hash}
    state = {"frozen_plan": _digest(plan), "receipts": [{"stage": "plan", "packet": plan}],
             "active_round": {"id": "round-fixture", "stage": "plan"}, "task_stop": None,
             "usage_ledger": {"fixture": {"status": "reserved", "reserved": {"calls": 2, "tokens": 1000},
                                           "purpose": json.dumps(purpose)}}}
    (tmp_path/".paper").mkdir()
    return state, receipt, profile


@pytest.mark.parametrize("fault", ["no_plan", "not_reserved", "no_round", "calls", "tokens", "stopped",
                                   "fake_plan", "empty_round", "nan_calls", "wrong_model"])
def test_generation_cannot_bypass_owner_reservation(tmp_path, fault):
    state, receipt, _ = owner_fixture(tmp_path)
    if fault == "no_plan":
        state["frozen_plan"] = None
    elif fault == "not_reserved":
        state["usage_ledger"]["fixture"]["status"] = "known"
    elif fault == "no_round":
        state["active_round"] = None
    elif fault == "calls":
        state["usage_ledger"]["fixture"]["reserved"]["calls"] = 0
    elif fault == "tokens":
        state["usage_ledger"]["fixture"]["reserved"]["tokens"] = 10
    elif fault == "stopped":
        state["task_stop"] = {"reason": "fixture"}
    elif fault == "fake_plan":
        state["frozen_plan"] = {"fixture": True}
    elif fault == "empty_round":
        state["active_round"] = {}
    elif fault == "nan_calls":
        state["usage_ledger"]["fixture"]["reserved"]["calls"] = float("nan")
    else:
        receipt.write_text('{"role":"different model"}')
    (tmp_path/".paper/control.json").write_text(json.dumps(state))
    with pytest.raises(ValueError):
        admitted_limits(tmp_path, "fixture", 1, 100, receipt)


def test_valid_synthetic_owner_binding_does_not_invoke_model(tmp_path):
    state, receipt, profile = owner_fixture(tmp_path)
    (tmp_path/".paper/control.json").write_text(json.dumps(state))
    assert admitted_limits(tmp_path, "fixture", 1, 100, receipt) == profile


def test_same_size_model_mutation_is_detected(tmp_path):
    path = tmp_path/"fixture.bin"
    path.write_bytes(b"abcd")
    files = [{"path": path.name, "bytes": 4, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    verify_model_files(tmp_path, files)
    path.write_bytes(b"abce")
    with pytest.raises(ValueError, match="SHA changed"):
        verify_model_files(tmp_path, files)


def test_empty_or_omitted_model_manifest_is_not_a_pin(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        verify_model_files(tmp_path, [])
    path = tmp_path/"fixture.bin"
    path.write_bytes(b"a")
    entry = {"path": path.name, "bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()}
    (tmp_path/"unlisted.bin").write_bytes(b"b")
    with pytest.raises(ValueError, match="unlisted"):
        verify_model_files(tmp_path, [entry])
