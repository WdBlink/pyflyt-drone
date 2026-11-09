"""Structural local-model preflight, deliberately without a generation call."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fix-mistral-regex", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output record")
    receipt = json.loads(args.download_receipt.read_text())
    path = Path(receipt["destination"])
    for entry in receipt["files"]:
        target = path/entry["path"]
        if not target.is_file() or target.stat().st_size != entry["bytes"]:
            raise ValueError("downloaded model file size changed")
    import mlx.core as mx
    from mlx_lm import load
    started = time.monotonic()
    tokenizer_config = {"fix_mistral_regex": True} if args.fix_mistral_regex else {}
    model, tokenizer, config = load(str(path), return_config=True, trust_remote_code=False,
                                    tokenizer_config=tokenizer_config)
    loaded_at = time.monotonic()
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": 'Reply with JSON: {"ready":true}.'}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.encode(prompt)
    result = {"role": "structural_load_and_tokenizer_only_not_generation_or_policy_qualification",
        "repo_id": receipt["repo_id"], "revision": receipt["revision"],
        "download_receipt_sha256": hashlib.sha256(args.download_receipt.read_bytes()).hexdigest(),
        "load_seconds": loaded_at-started, "model_class": type(model).__module__+"."+type(model).__name__,
        "model_type": config["model_type"], "rendered_prompt": prompt, "rendered_prompt_tokens": len(tokens),
        "tokenizer_config": tokenizer_config,
        "tokenizer_backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "generated_tokens": 0, "generation_calls": 0, "paid_inference_calls": 0,
        "peak_memory_bytes": mx.get_peak_memory(), "device": mx.device_info(),
        "python": sys.version, "versions": {p: importlib.metadata.version(p) for p in
            ("mlx", "mlx-lm", "transformers", "tokenizers", "safetensors", "huggingface-hub", "numpy")},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["No forward/generation call, policy efficacy or inference latency measured.",
                        "Download receipt already verifies blob SHA; this load checks file presence/size, not a second full SHA pass."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("repo_id", "load_seconds", "model_type", "peak_memory_bytes", "rendered_prompt_tokens")}))


if __name__ == "__main__":
    main()
