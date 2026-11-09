"""Fetch one pinned public MLX candidate and verify every downloaded blob."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def file_sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("choose a new acquisition receipt")
    metadata_bytes = args.metadata.read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata["gated"] or not metadata["id"].startswith("mlx-community/") or len(metadata["sha"]) != 40:
        parser.error("an ungated pinned MLX metadata record is required")
    from huggingface_hub import snapshot_download
    started = time.monotonic()
    snapshot_download(repo_id=metadata["id"], revision=metadata["sha"], local_dir=args.destination,
                      token=False, max_workers=2)
    files = []
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        path = args.destination/name
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"missing or size-mismatched blob: {name}")
        sha = file_sha256(path)
        lfs = entry.get("lfs")
        if lfs and sha != lfs["sha256"]:
            raise ValueError(f"LFS SHA mismatch: {name}")
        if not lfs:
            raw = path.read_bytes()
            git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
            if git_blob != entry["blobId"]:
                raise ValueError(f"Git blob mismatch: {name}")
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha,
                      "upstream_identity": lfs["sha256"] if lfs else entry["blobId"]})
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"role": "download_verification_not_model_quality_or_G4_acceptance", "repo_id": metadata["id"],
               "revision": metadata["sha"], "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
               "destination": str(args.destination.resolve()), "files": files,
               "elapsed_seconds": time.monotonic()-started, "paid_inference_calls": 0,
               "source_sha256": file_sha256(Path(__file__))}
    with args.receipt.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: receipt[k] for k in ("repo_id", "revision", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
