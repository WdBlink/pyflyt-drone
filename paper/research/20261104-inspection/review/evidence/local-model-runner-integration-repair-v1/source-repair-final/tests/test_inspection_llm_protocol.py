"""Wire-format checks only; no model request or physics execution."""

import json

import pytest

from eval.inspection_llm_protocol import ModelReplyError, PUBLIC_FIELDS, parse_reply, serialize_public_frame


def test_valid_json_batch_and_empty_wait():
    text = '{"commands":[{"name":"enter_safe_exit","args":{"drone_id":0,"expected_version":1}}],"note":"exit"}'
    _, commands = parse_reply(text)
    assert commands == [("enter_safe_exit", {"drone_id": 0, "expected_version": 1})]
    assert parse_reply('{"commands":[]}')[1] == []


@pytest.mark.parametrize("text", [
    '{"commands":[],"commands":[]}', '{"commands":[],"value":NaN}',
    '```json\n{"commands":[]}\n```', '{"commands":[]} trailing',
    '{"commands":[{"name":"read_truth","args":{}}]}',
    '{"commands":[{"name":"enter_safe_exit","args":{"drone_id":0}}]}',
    '{"commands":[],"private_case":{}}', '{"commands":[],"plan":[1]}',
    '{"commands":[{"name":"enter_safe_exit","args":{"drone_id":1e400,"expected_version":1}}]}',
    '{"commands":'+('['*1500)+(']'*1500)+'}',
])
def test_malformed_reply_is_policy_failure(text):
    with pytest.raises(ModelReplyError):
        parse_reply(text)


def test_hidden_fields_are_rejected_recursively():
    state = {field: [] for field in PUBLIC_FIELDS}
    state["sim_time"] = 1.
    state["task_card"] = {"level": "L1"}
    assert json.loads(serialize_public_frame(state))["sim_time"] == 1.
    state["observations"] = [{"candidate_id": "c1", "entity_key": "hidden"}]
    with pytest.raises(ValueError, match="private"):
        serialize_public_frame(state)
    state["observations"] = []
    state["private_case"] = {}
    with pytest.raises(ValueError, match="declared public"):
        serialize_public_frame(state)


@pytest.mark.parametrize("field", ["score", "truth", "grader_output"])
def test_raw_evaluator_output_cannot_hide_in_an_observation(field):
    state = {name: [] for name in PUBLIC_FIELDS}
    state["sim_time"] = 1.
    state["observations"] = [{field: {"value": 1}}]
    with pytest.raises(ValueError, match="private"):
        serialize_public_frame(state)
