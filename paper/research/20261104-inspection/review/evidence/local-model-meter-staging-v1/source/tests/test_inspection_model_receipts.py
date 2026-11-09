"""Synthetic receipt corruption and accounting, no model dispatch."""

import json

import pytest

from eval.inspection_model_receipts import UnresolvedModelOperation, validate_model_operation


def fixture(tmp_path):
    profile = {"max_input_tokens": 8192, "max_output_tokens": 1024, "temperature": .2, "top_p": .95,
        "max_request_seconds": 60, "max_calls": 32, "max_total_tokens": 180000}
    expected = {"reservation": "synthetic", "owner_frozen_plan": "a"*64, "model_receipt_sha256": "b"*64,
        "reserved": {"calls": 32, "tokens": 180000}, "service_sha256": "c"*64, "generation_profile": profile}
    call = tmp_path/"call-0001"
    call.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value))
    write(tmp_path/"service-intent.json", {**{k: expected[k] for k in
        ("reservation", "owner_frozen_plan", "model_receipt_sha256", "reserved")}, "source_sha256": expected["service_sha256"]})
    write(call/"request.json", {"messages": [{"role": "user", "content": "synthetic"}], "seed": 1,
        "input_tokens_submitted": 3, "max_output_tokens": 1024, "temperature": .2, "top_p": .95, "enable_thinking": False})
    (call/"tokens.jsonl").write_text(json.dumps({"token": 7, "text": "{}", "runtime_prompt_tokens": 3, "runtime_generation_tokens": 1})+"\n")
    write(call/"response.json", {"kind": "response", "status": "completed", "text": "{}", "wall_seconds": 1.,
        "deadline_exceeded": False, "usage": {"calls": 1, "tokens": 4}, "input_tokens_submitted": 3,
        "generated_tokens_observed": 1, "emitted_tokens_journaled": 1, "receipt": str((call/"response.json").resolve())})
    write(tmp_path/"service-completion.json", {"usage": {"calls": 1, "tokens": 4}})
    return expected


def test_meter_counts_requests_and_runtime_tokens_not_imagined_flight(tmp_path):
    result = validate_model_operation(tmp_path, fixture(tmp_path))
    assert result["usage"] == {"calls": 1, "tokens": 4}
    assert result["generated_tokens_runtime_observed"] == result["tokens_durably_journaled"] == 1
    assert "not episode/G4" in result["role"]


@pytest.mark.parametrize("fault", ["missing_response", "truncated_tokens", "different_text", "different_input",
    "wrong_source", "wrong_model", "wrong_owner", "wrong_total", "late_completed", "token_bool", "call_gap"])
def test_incomplete_or_mismatched_accounting_never_becomes_zero(tmp_path, fault):
    expected = fixture(tmp_path)
    call = tmp_path/"call-0001"
    if fault == "missing_response":
        (call/"response.json").unlink()
    elif fault == "truncated_tokens":
        (call/"tokens.jsonl").write_text('{"token":')
    elif fault == "call_gap":
        call.rename(tmp_path/"call-0002")
    elif fault in ("wrong_source", "wrong_model", "wrong_owner"):
        field = {"wrong_source": "service_sha256", "wrong_model": "model_receipt_sha256", "wrong_owner": "owner_frozen_plan"}[fault]
        expected[field] = "f"*64
    else:
        path = call/("tokens.jsonl" if fault == "token_bool" else "response.json")
        value = json.loads(path.read_text())
        if fault == "different_text": value["text"] = "changed"
        elif fault == "different_input": value["input_tokens_submitted"] = 4
        elif fault == "wrong_total": value["usage"]["tokens"] = 2
        elif fault == "late_completed": value["wall_seconds"], value["deadline_exceeded"] = 61, True
        elif fault == "token_bool": value["runtime_generation_tokens"] = True
        path.write_text(json.dumps(value)+"\n")
    with pytest.raises(ValueError):
        validate_model_operation(tmp_path, expected)


def test_known_runtime_failure_reports_unjournaled_count_separately(tmp_path):
    expected = fixture(tmp_path)
    path = tmp_path/"call-0001/response.json"
    result = json.loads(path.read_text())
    result.update(status="runtime_failure", generated_tokens_observed=2, usage={"calls": 1, "tokens": 5})
    path.write_text(json.dumps(result))
    (tmp_path/"service-completion.json").write_text(json.dumps({"usage": {"calls": 1, "tokens": 5}}))
    recounted = validate_model_operation(tmp_path, expected)
    assert recounted["generated_tokens_runtime_observed"] == 2 and recounted["tokens_durably_journaled"] == 1
    assert recounted["calls"][0]["reported_unjournaled_observations"] == 1
