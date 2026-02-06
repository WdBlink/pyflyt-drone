#!/usr/bin/env python3
"""
Download FastSAM weights into this repo (default path used by ArdupilotGazeboObjLockEnv).

Outputs:
  - ./FastSAM-s.pt

Notes:
  - This script only downloads the weight file. You still need Python deps for inference
    (e.g. ultralytics + torch).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request


DEFAULT_URL = "https://github.com/opengeos/datasets/releases/download/models/FastSAM-s.pt"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", type=str, default=DEFAULT_URL)
    ap.add_argument("--out", type=str, default="FastSAM-s.pt")
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tmp_path = out_path + ".partial"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Already exists: {out_path} ({os.path.getsize(out_path)} bytes)")
        print(f"sha256: {_sha256(out_path)}")
        return 0

    print(f"Downloading: {args.url}")
    print(f"To: {out_path}")
    try:
        with urllib.request.urlopen(args.url) as r, open(tmp_path, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"Download failed: {e!r}", file=sys.stderr)
        return 1

    os.replace(tmp_path, out_path)
    print(f"Downloaded: {out_path} ({os.path.getsize(out_path)} bytes)")
    print(f"sha256: {_sha256(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

