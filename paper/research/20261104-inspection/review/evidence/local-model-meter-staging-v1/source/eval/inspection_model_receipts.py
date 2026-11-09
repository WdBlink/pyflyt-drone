"""Read-only accounting of a completed local operation, never model dispatch.

The caller binds expected values before dispatch to the owner reservation and
the frozen model/source profile. Token counts are the declared local request
meter, not API billing, GPU FLOPs or independent measurement of hidden work.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class UnresolvedModelOperation(ValueError):
    pass


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate receipt field")
            result[key] = value
        return result
    try:
        result = json.loads(Path(path).read_text(), object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(result, dict):
            raise ValueError("receipt must be an object")
        return result
    except (OSError, ValueError) as error:
        raise UnresolvedModelOperation("missing/corrupt model receipt: "+str(path)) from error


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_model_operation(directory, expected):
    directory = Path(directory).resolve()
    intent = _read(directory/"service-intent.json")
    for field in ("reservation", "owner_frozen_plan", "model_receipt_sha256", "reserved"):
        if intent.get(field) != expected[field]:
            raise ValueError("model intent differs from the admitted operation: "+field)
    if intent.get("source_sha256") != expected["service_sha256"]:
        raise ValueError("actual service source differs")
    profile = expected["generation_profile"]
    completion = _read(directory/"service-completion.json")
    calls = sorted(directory.glob("call-*"))
    if [p.name for p in calls] != [f"call-{i+1:04d}" for i in range(len(calls))] or not all(p.is_dir() for p in calls):
        raise UnresolvedModelOperation("call sequence is incomplete or ambiguous")
    used, observed, journaled, bindings = 0, 0, 0, []
    for call in calls:
        if any(p.is_symlink() for p in call.rglob("*")) or call.is_symlink():
            raise ValueError("model call receipts cannot be linked")
        request, response = _read(call/"request.json"), _read(call/"response.json")
        input_count = request["input_tokens_submitted"]
        if (type(input_count) is not int or not 0 < input_count <= profile["max_input_tokens"]
                or request["max_output_tokens"] != profile["max_output_tokens"]
                or request["temperature"] != profile["temperature"] or request["top_p"] != profile["top_p"]
                or request["enable_thinking"] is not False):
            raise ValueError("request differs from the admitted generation profile")
        try:
            rows = [json.loads(line) for line in (call/"tokens.jsonl").read_text().splitlines()]
        except (OSError, ValueError) as error:
            raise UnresolvedModelOperation("unreconciled token journal") from error
        generated = response["generated_tokens_observed"]
        if (type(generated) is not int or not 0 <= len(rows) <= generated <= profile["max_output_tokens"]
                or type(response["emitted_tokens_journaled"]) is not int or response["emitted_tokens_journaled"] != len(rows)
                or type(response["input_tokens_submitted"]) is not int or response["input_tokens_submitted"] != input_count
                or not isinstance(response["usage"], dict) or any(type(v) is not int for v in response["usage"].values())
                or response["usage"] != {"calls": 1, "tokens": input_count+generated}
                or response.get("kind") != "response" or response["status"] not in ("completed", "policy_timeout", "runtime_failure")
                or Path(response["receipt"]).resolve() != (call/"response.json").resolve()):
            raise ValueError("runtime response/count bindings differ")
        if any(type(row.get("token")) is not int or row["token"] < 0 or not isinstance(row.get("text"), str)
                or type(row.get("runtime_prompt_tokens")) is not int or row["runtime_prompt_tokens"] != input_count
                or type(row.get("runtime_generation_tokens")) is not int or row["runtime_generation_tokens"] != i+1
                for i, row in enumerate(rows)):
            raise ValueError("token journal runtime counters differ")
        if response["text"] != "".join(row["text"] for row in rows):
            raise ValueError("response text differs from durable token segments")
        wall = response["wall_seconds"]
        if (type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0
                or type(response["deadline_exceeded"]) is not bool
                or response["deadline_exceeded"] != (wall > profile["max_request_seconds"])
                or response["status"] == "completed" and response["deadline_exceeded"]
                or response["status"] == "policy_timeout" and not response["deadline_exceeded"]
                or response["status"] != "runtime_failure" and generated != len(rows)):
            raise ValueError("runtime status/deadline/journal count differs")
        used += input_count+generated
        observed += generated
        journaled += len(rows)
        bindings.append({"ordinal": len(bindings)+1, "status": response["status"],
            "request_sha256": _sha(call/"request.json"), "response_sha256": _sha(call/"response.json"),
            "tokens_sha256": _sha(call/"tokens.jsonl"), "reported_unjournaled_observations": generated-len(rows)})
    usage = {"calls": len(calls), "tokens": used}
    if (completion.get("usage") != usage or usage["calls"] > profile["max_calls"]
            or usage["tokens"] > profile["max_total_tokens"]
            or any(usage[k] > expected["reserved"][k] for k in usage)):
        raise ValueError("completed local usage differs from retained calls or reservation")
    return {"role": "completed local request-meter recount, not episode/G4/hidden-compute certification",
        "usage": usage, "generated_tokens_runtime_observed": observed, "tokens_durably_journaled": journaled,
        "intent_sha256": _sha(directory/"service-intent.json"),
        "completion_sha256": _sha(directory/"service-completion.json"), "calls": bindings,
        "capital_energy_cost_unmetered": True}
