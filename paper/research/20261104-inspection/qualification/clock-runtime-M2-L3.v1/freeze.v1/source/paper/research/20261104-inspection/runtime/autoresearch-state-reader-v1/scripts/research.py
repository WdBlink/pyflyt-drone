#!/usr/bin/env python3
"""Reviewed stable transport to this run's sole migrated owner; no old imports."""
import os
from pathlib import Path
import sys

target = Path(__file__).resolve().parents[2]/"autoresearch-document-snapshots-v2/scripts/research.py"
os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])
