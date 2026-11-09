"""Local MLX request service; requires a live controller reservation first."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def operation_directory(reservation):
    if (not isinstance(reservation, str) or not reservation
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in reservation)):
        raise ValueError("invalid reservation identity")
    # Live journals stay outside research documents so native checkpoints can
    # run between flights without hashing an actively growing token journal.
    return Path(__file__).resolve().parents[1]/"logs/inspection-model-resource-operations.v1"/reservation


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def admitted_limits(run, reservation, calls, tokens, model_receipt=None, *, formal_dispatch=None):
    """Read the sole owner's state. Never create or settle its ledger entries."""
    formal = None
    if formal_dispatch is not None:
        from eval.inspection_formal_dispatch import admit_dispatch
        formal = admit_dispatch(run, Path(__file__).resolve().parents[1], formal_dispatch)
    state = json.loads((Path(run)/".paper/control.json").read_text())
    entry = state.get("usage_ledger", {}).get(reservation, {})
    plans = [r.get("packet") for r in state.get("receipts", []) if r.get("stage") == "plan"]
    active = state.get("active_round")
    held = entry.get("reserved", {})
    if (len(plans) != 1 or not isinstance(plans[0], dict) or type(state.get("frozen_plan")) is not str
            or state["frozen_plan"] != _digest(plans[0])
            or not isinstance(active, dict) or not isinstance(active.get("id"), str) or not active["id"]
            or not isinstance(active.get("stage"), str) or not active["stage"] or state.get("task_stop") is not None
            or entry.get("status") != "reserved"
            or type(calls) is not int or calls <= 0 or type(tokens) is not int or tokens <= 0
            or type(held.get("calls")) is not int or held["calls"] < calls
            or type(held.get("tokens")) is not int or held["tokens"] < tokens):
        raise ValueError("generation requires an accepted complete plan and a sufficient live owner reservation")
    if model_receipt is None:
        raise ValueError("model receipt must be bound to the frozen plan and reservation")
    root = Path(run).resolve()
    registry_path = (root/plans[0]["local_model_registry"]).resolve()
    if not registry_path.is_relative_to(root):
        raise ValueError("model registry is outside the frozen run")
    registry_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    if registry_hash != plans[0]["local_model_registry_sha256"]:
        raise ValueError("model registry changed after plan acceptance")
    receipt_hash = hashlib.sha256(Path(model_receipt).read_bytes()).hexdigest()
    registry = json.loads(registry_path.read_text())
    profiles = [v for v in registry["models"].values() if v["download_receipt_sha256"] == receipt_hash]
    purpose = json.loads(entry["purpose"])
    if (len(profiles) != 1 or purpose.get("kind") != "inspection_local_generation"
            or purpose.get("phase") != ("development" if formal is None else formal.binding()["phase"]) or purpose.get("model_registry_sha256") != registry_hash
            or purpose.get("model_receipt_sha256") != receipt_hash):
        raise ValueError("reservation is not bound to this frozen phase/model")
    if formal is not None:
        slot=formal.episode.slot
        if (purpose.get("formal_dispatch") != formal_dispatch
                or purpose.get("formal_dispatch_binding") != formal.binding()
                or any(purpose.get(key) != getattr(slot,key) for key in ("case_id","condition","repeat_ordinal"))
                or slot.sampler_seed is None
                or registry["models"].get(slot.condition.split("-")[1]) != profiles[0]):
            raise ValueError("formal generation reservation differs from its admitted primary slot")
    return profiles[0]


def write_once(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def journal_token(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
    stream.flush()
    os.fsync(stream.fileno())


def verify_model_files(path, files):
    path = Path(path).resolve()
    if not isinstance(files, list) or not files or len({f["path"] for f in files}) != len(files):
        raise ValueError("complete nonempty model file list required")
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()
              and ".cache" not in p.relative_to(path).parts and p.name != ".DS_Store"}
    if actual != {f["path"] for f in files}:
        raise ValueError("unlisted or omitted model files")
    for entry in files:
        target = (path/entry["path"]).resolve()
        if (not target.is_relative_to(path) or not target.is_file()
                or target.stat().st_size != entry["bytes"]):
            raise ValueError("pinned local model file missing, relocated or resized")
        value = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                value.update(block)
        if value.hexdigest() != entry["sha256"]:
            raise ValueError("pinned local model blob SHA changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--formal-dispatch")
    parser.add_argument("--formal-dispatch-sha256")
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--max-request-seconds", type=float, default=60.)
    parser.add_argument("--fix-mistral-regex", action="store_true")
    args = parser.parse_args()
    if (not args.reservation or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.reservation)
            or not 1 <= args.max_input_tokens <= 32768 or not 1 <= args.max_output_tokens <= 4096
            or not 0 < args.max_request_seconds <= 120):
        parser.error("invalid bounded local service configuration")
    if (args.formal_dispatch is None) != (args.formal_dispatch_sha256 is None):
        parser.error("formal generation requires its complete dispatch artifact")
    formal_ref = (None if args.formal_dispatch is None else
                  {"path":args.formal_dispatch,"sha256":args.formal_dispatch_sha256})
    def admit():
        return admitted_limits(args.run, args.reservation, args.max_calls, args.max_total_tokens,
                               args.model_receipt, formal_dispatch=formal_ref)
    profile = admit()
    limits = profile["generation_profile"]
    if (args.max_input_tokens > limits["max_input_tokens"] or args.max_output_tokens > limits["max_output_tokens"]
            or args.max_request_seconds > limits["max_request_seconds"]
            or args.max_calls > limits["max_calls"] or args.max_total_tokens > limits["max_total_tokens"]
            or limits["enable_thinking"] is not False
            or args.fix_mistral_regex != profile["tokenizer_config"].get("fix_mistral_regex", False)):
        raise ValueError("service configuration differs from the frozen model profile")
    receipt = json.loads(args.model_receipt.read_text())
    if len(receipt["files"]) != profile["file_count"]:
        raise ValueError("download receipt file set differs from frozen registry")
    path = Path(receipt["destination"])
    verify_model_files(path, receipt["files"])
    admit()
    operation = operation_directory(args.reservation)
    operation.mkdir(parents=True, exist_ok=False)  # one service admission per reservation
    owner = json.loads((args.run/".paper/control.json").read_text())
    request_budget = {"max_calls": args.max_calls, "max_total_tokens": args.max_total_tokens,
        "max_input_tokens": args.max_input_tokens, "max_output_tokens": args.max_output_tokens,
        "max_request_seconds": args.max_request_seconds, "temperature": limits["temperature"],
        "top_p": limits["top_p"], "enable_thinking": False}
    formal_intent = {}
    if formal_ref is not None:
        from eval.inspection_formal_dispatch import admit_dispatch
        formal_intent["formal_dispatch_binding"] = admit_dispatch(args.run,
            Path(__file__).resolve().parents[1], formal_ref).binding()
    write_once(operation/"service-intent.json", {**formal_intent, "reservation": args.reservation,
        "owner_frozen_plan": owner["frozen_plan"], "reserved": owner["usage_ledger"][args.reservation]["reserved"],
        "pid": os.getpid(), "process_group": os.getpgrp(),
        "request_budget": request_budget,
        "model_receipt_sha256": hashlib.sha256(args.model_receipt.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "role": "local request intent, not model generation or scientific acceptance"})
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    config = {"local_files_only": True, **({"fix_mistral_regex": True} if args.fix_mistral_regex else {})}
    admit()
    model, tokenizer = load(str(path), tokenizer_config=config, trust_remote_code=False)
    print(json.dumps({"kind": "ready", "reservation": args.reservation}), flush=True)
    used_calls, used_tokens = 0, 0
    for line in sys.stdin:
        request = json.loads(line)
        messages, seed = request["messages"], request["seed"]
        if (not isinstance(messages, list) or not 1 <= len(messages) <= 64 or type(seed) is not int or not 0 <= seed < 2**32
                or any(not isinstance(m, dict) or set(m) != {"role", "content"}
                       or m["role"] not in ("system", "user", "assistant") or not isinstance(m["content"], str)
                       or len(m["content"]) > 100000 for m in messages)):
            raise ValueError("malformed bounded model request")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer.encode(prompt)
        if (len(encoded) > args.max_input_tokens or used_calls >= args.max_calls
                or used_tokens+len(encoded)+args.max_output_tokens > args.max_total_tokens):
            print(json.dumps({"kind": "not_started", "reason": "frozen local request budget/context exhausted"}), flush=True)
            continue
        admit()
        used_calls += 1
        call = operation/f"call-{used_calls:04d}"
        call.mkdir()
        write_once(call/"request.json", {"messages": messages, "seed": seed, "input_tokens_submitted": len(encoded),
            "max_output_tokens": args.max_output_tokens, "temperature": limits["temperature"], "top_p": limits["top_p"],
            "enable_thinking": False, "meter": "admitted local requests, submitted input plus runtime-observed generated tokens; journaled tokens separate; not FLOPs",
            "deadline_scope": "soft result deadline; backend may overrun until yielding/returning; late result is not a command"})
        mx.random.seed(seed)
        started, text, generated, journaled, status, error = time.monotonic(), "", 0, 0, "completed", None
        try:
            with (call/"tokens.jsonl").open("x") as journal:
                for chunk in stream_generate(model, tokenizer, prompt=encoded, max_tokens=args.max_output_tokens,
                                             sampler=make_sampler(temp=limits["temperature"], top_p=limits["top_p"])):
                    generated += 1
                    journal_token(journal, {"token": chunk.token, "text": chunk.text,
                        "runtime_prompt_tokens": chunk.prompt_tokens, "runtime_generation_tokens": chunk.generation_tokens})
                    journaled += 1
                    text += chunk.text
                    if generated >= args.max_output_tokens or time.monotonic()-started > args.max_request_seconds:
                        break
        except Exception as caught:
            status, error = "runtime_failure", repr(caught)
        wall_seconds = time.monotonic()-started
        deadline_exceeded = wall_seconds > args.max_request_seconds
        if deadline_exceeded and status == "completed":
            status = "policy_timeout"
        used_tokens += len(encoded)+generated
        result = {"kind": "response", "status": status, "text": text, "error": error,
            "wall_seconds": wall_seconds, "deadline_exceeded": deadline_exceeded,
            "deadline_scope": "soft result deadline; no guarantee of hard GPU cancellation",
            "usage": {"calls": 1, "tokens": len(encoded)+generated},
            "input_tokens_submitted": len(encoded), "generated_tokens_observed": generated, "emitted_tokens_journaled": journaled,
            "receipt": str((call/"response.json").resolve())}
        write_once(call/"response.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    write_once(operation/"service-completion.json", {"role": "local metered service completion, not an episode score",
        "usage": {"calls": used_calls, "tokens": used_tokens}, "capital_energy_cost_unmetered": True})


if __name__ == "__main__":
    main()
