"""One actually reserved pinned-weight generation smoke, without a flight.

Does not reserve, settle, retry or approve G4. The owner must admit the exact
operation first, then settle its immutable receipt (or reconcile unknown usage).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.inspection_llm_protocol import parse_reply
from eval.inspection_model_client import LocalModelClient
from eval.inspection_model_receipts import validate_model_operation
from eval.inspection_model_service import operation_directory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=("M1", "M2", "M3"), required=True)
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not 0 <= args.seed < 2**32:
        parser.error("choose a new output and uint32 sampling seed")
    args.out.mkdir(parents=True)
    registry_path = args.run/"planning/local-model-registry.v1.json"
    profile = json.loads(registry_path.read_text())["models"][args.model]
    budget = {**profile["generation_profile"], "max_calls": 1, "max_total_tokens": 16384, "max_output_tokens": 64}
    receipt = args.run/profile["download_receipt"]
    owner = json.loads((args.run/".paper/control.json").read_text())
    expected = {"reservation": args.reservation, "owner_frozen_plan": owner["frozen_plan"],
        "model_receipt_sha256": sha(receipt), "service_sha256": sha(ROOT/"eval/inspection_model_service.py"),
        "reserved": owner["usage_ledger"][args.reservation]["reserved"], "generation_profile": budget}
    messages = [{"role": "system", "content": "Return exactly one strict JSON object. No Markdown or other text."},
        {"role": "user", "content": 'Reply with {"commands":[],"note":"ready"}. This is a text generation smoke; do not call tools.'}]
    command = [str(ROOT/".venv-inspection-llm/bin/python"), "-u", str(ROOT/"eval/inspection_model_service.py"),
        "--run", str(args.run.resolve()), "--reservation", args.reservation, "--model-receipt", str(receipt.resolve()),
        "--max-calls", "1", "--max-total-tokens", "16384", "--max-output-tokens", "64",
        "--max-input-tokens", str(budget["max_input_tokens"]), "--max-request-seconds", str(budget["max_request_seconds"])]
    if profile["tokenizer_config"].get("fix_mistral_regex"):
        command.append("--fix-mistral-regex")
    source_paths = [Path(__file__), *[ROOT/"eval"/name for name in
        ("inspection_model_service.py", "inspection_model_client.py", "inspection_model_receipts.py", "inspection_llm_protocol.py")]]
    sources = {str(p): sha(p) for p in source_paths}
    result = {"role": "one local pinned-weight text-generation smoke, not an episode or G4 qualification",
        "started_at": datetime.now(timezone.utc).isoformat(), "model_alias": args.model,
        "expected": expected, "messages": messages, "seed": args.seed, "sources": sources,
        "registry_sha256": sha(registry_path), "status": "intent_not_generation"}
    def save():
        (args.out/"probe.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    save()
    client = None
    try:
        client = LocalModelClient(command, args.out/"transport")
        if client.ready.get("reservation") != args.reservation:
            raise ValueError("ready reservation differs")
        client.submit(messages, args.seed)
        response = None
        while response is None:
            response = client.poll()
            time.sleep(.01)
        result["response"] = response
        client.finish()
        result["resource_recount"] = validate_model_operation(operation_directory(args.reservation), expected)
        result["sources_unchanged"] = all(sha(p) == h for p, h in sources.items())
        if not result["sources_unchanged"]:
            raise ValueError("probe source changed during execution")
        try:
            result["parsed_reply"], _ = parse_reply(response.get("text"))
        except ValueError as error:
            result["policy_format_error"] = str(error)
        result["status"] = ("receipted_generation_smoke_not_capability_qualification"
            if response.get("status") == "completed" and result["resource_recount"]["generated_tokens_runtime_observed"] > 0
            else "receipted_request_failure_not_generation_qualification")
    except BaseException:
        result["status"], result["error"] = "operational_failure_or_unknown", traceback.format_exc()
    finally:
        if client is not None and client.process.poll() is None:
            try:
                final = client.finish()
                if final is not None:
                    result["ignored_final_response"] = final
            except BaseException:
                result["shutdown_unknown"] = traceback.format_exc()
        save()
    print(json.dumps({"status": result["status"], "probe": str((args.out/"probe.json").resolve()),
        "meter": result.get("resource_recount", {}).get("usage"),
        "generated_tokens": result.get("resource_recount", {}).get("generated_tokens_runtime_observed")}, allow_nan=False))
    return 0 if result["status"] == "receipted_generation_smoke_not_capability_qualification" else 1


if __name__ == "__main__":
    raise SystemExit(main())
