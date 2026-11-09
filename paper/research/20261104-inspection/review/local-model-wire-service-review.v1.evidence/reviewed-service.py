"""Local MLX request service; requires a live controller reservation first."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def admitted_limits(run, reservation, calls, tokens):
    """Read the sole owner's state. Never create or settle its ledger entries."""
    state = json.loads((Path(run)/".paper/control.json").read_text())
    entry = state.get("usage_ledger", {}).get(reservation, {})
    if (not state.get("frozen_plan") or not any(r["stage"] == "plan" for r in state["receipts"])
            or state.get("active_round") is None or state.get("task_stop") is not None
            or entry.get("status") != "reserved"
            or type(calls) is not int or calls <= 0 or type(tokens) is not int or tokens <= 0
            or entry.get("reserved", {}).get("calls", 0) < calls
            or entry.get("reserved", {}).get("tokens", 0) < tokens):
        raise ValueError("generation requires an accepted complete plan and a sufficient live owner reservation")
    return entry


def write_once(path, value):
    import os
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_model_files(path, files):
    path = Path(path).resolve()
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
    admitted_limits(args.run, args.reservation, args.max_calls, args.max_total_tokens)
    receipt = json.loads(args.model_receipt.read_text())
    path = Path(receipt["destination"])
    verify_model_files(path, receipt["files"])
    operation = args.run/"resource-operations"/args.reservation
    operation.mkdir(parents=True, exist_ok=False)  # one service admission per reservation
    write_once(operation/"service-intent.json", {"reservation": args.reservation,
        "model_receipt_sha256": hashlib.sha256(args.model_receipt.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "role": "local request intent, not model generation or scientific acceptance"})
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    config = {"local_files_only": True, **({"fix_mistral_regex": True} if args.fix_mistral_regex else {})}
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
        admitted_limits(args.run, args.reservation, args.max_calls, args.max_total_tokens)
        used_calls += 1
        call = operation/f"call-{used_calls:04d}"
        call.mkdir()
        write_once(call/"request.json", {"messages": messages, "seed": seed, "input_tokens_submitted": len(encoded),
            "max_output_tokens": args.max_output_tokens, "temperature": .2, "top_p": .95,
            "enable_thinking": False, "meter": "admitted local requests, submitted prompt tokens and journaled emitted tokens; not FLOPs"})
        mx.random.seed(seed)
        started, text, count, status, error = time.monotonic(), "", 0, "completed", None
        try:
            for chunk in stream_generate(model, tokenizer, prompt=encoded, max_tokens=args.max_output_tokens,
                                         sampler=make_sampler(temp=.2, top_p=.95)):
                count += 1
                text += chunk.text
                write_once(call/f"token-{count:04d}.json", {"token": chunk.token, "text": chunk.text,
                    "runtime_prompt_tokens": chunk.prompt_tokens, "runtime_generation_tokens": chunk.generation_tokens})
                if time.monotonic()-started > args.max_request_seconds:
                    status = "policy_timeout"
                    break
        except Exception as caught:
            status, error = "runtime_failure", repr(caught)
        used_tokens += len(encoded)+count
        result = {"kind": "response", "status": status, "text": text, "error": error,
            "wall_seconds": time.monotonic()-started, "usage": {"calls": 1, "tokens": len(encoded)+count},
            "input_tokens_submitted": len(encoded), "emitted_tokens_journaled": count,
            "receipt": str((call/"response.json").resolve())}
        write_once(call/"response.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    write_once(operation/"service-completion.json", {"role": "local metered service completion, not an episode score",
        "usage": {"calls": used_calls, "tokens": used_tokens}, "capital_energy_cost_unmetered": True})


if __name__ == "__main__":
    main()
