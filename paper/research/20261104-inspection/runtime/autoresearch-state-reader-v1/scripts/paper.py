#!/usr/bin/env python3
"""Compatibility file entry; the public workflow is research.py."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('research.py')), run_name='__main__')
