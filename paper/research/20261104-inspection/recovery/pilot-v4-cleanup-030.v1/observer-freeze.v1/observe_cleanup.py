"""Passive metadata for the original v4 close decision; never admit an episode."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

RUN = Path(__file__).resolve().parents[3]
PROJECT = RUN.parents[2]
OUTPUT = PROJECT / "logs/inspection-eight-condition-pilot.v4.continuation.v2/cleanup-observations"
sys.path.insert(0, str(PROJECT))
from eval import inspection_process_cleanup as cleanup

_active = None


class CloseObserver:
    """Capture existing global scans, preserving the original close return/error."""

    def __init__(self, context):
        if (type(context) is not dict
                or set(context) != {"case_id", "condition", "repeat_ordinal", "plan_sha256", "reservation_prefix"}
                or context["case_id"] not in {f"dev-L{level}-{n:02d}" for level in (1, 2, 3) for n in (0, 1)}
                or context["condition"] not in {"B3-M1", "B3-M2", "B3-M3", "B4-M1", "B4-M2"}
                or type(context["repeat_ordinal"]) is not int or context["repeat_ordinal"] not in (0, 1, 2)
                or type(context["plan_sha256"]) is not str or re.fullmatch("[0-9a-f]{64}", context["plan_sha256"]) is None
                or type(context["reservation_prefix"]) is not str or re.fullmatch("dev-all8-pilot-v4-[0-9]{3}", context["reservation_prefix"]) is None):
            raise ValueError("only the fixed v4 development context is observed")
        self.context = dict(context)
        self.errors, self.records = [], []
        self.thread = threading.get_ident()
        self.inside = False
        self.source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def _incomplete(self, reason):
        self.errors.append(str(reason))
        try:
            print(json.dumps({"status": "cleanup_observation_incomplete", "reason": str(reason)}, allow_nan=False), file=sys.stderr, flush=True)
        except BaseException:
            pass

    def __enter__(self):
        global _active
        if _active is not None or threading.get_ident() != self.thread:
            raise RuntimeError("cleanup observer must be serial and entered by its owner thread")
        path = OUTPUT.absolute()
        if not path.is_relative_to(PROJECT) or ".." in path.parts:
            raise ValueError("declared project-log sidecar required")
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise ValueError("sidecar path cannot use symbolic aliases")
            if parent == PROJECT:
                break
        self.original_close = cleanup.SubprocessTree.close
        self.original_inventory = cleanup._inventory
        observer = self

        def close(tree):
            return observer._close(tree)

        self.wrapper = close
        _active = self
        cleanup.SubprocessTree.close = self.wrapper
        return self

    def __exit__(self, exc_type, exc, traceback):
        global _active
        cleanup.SubprocessTree.close = self.original_close
        _active = None
        return False

    def _rows(self, inventory, owned):
        return {str(pid): dict(row) for pid, row in inventory.items()
                if pid in owned or row.get("uid") == os.getuid()
                and "python" in row.get("command", "").lower()}

    def _close(self, tree):
        if (threading.get_ident() != self.thread or self.inside
                or cleanup._inventory is not self.original_inventory):
            self._incomplete("nonserial, foreign-thread or changed inventory hook")
            return self.original_close(tree)
        self.inside = True
        scans, result, exception = [], None, None
        started = datetime.now(timezone.utc).isoformat()
        start_ns = time.monotonic_ns()

        def inventory():
            if threading.get_ident() != self.thread:
                self._incomplete("another thread used the temporary inventory hook")
                return self.original_inventory()
            before_utc, before_ns = datetime.now(timezone.utc).isoformat(), time.monotonic_ns()
            scan = {"ordinal": len(scans), "started_at": before_utc, "monotonic_start_ns": before_ns}
            scans.append(scan)
            try:
                rows = self.original_inventory()
            except BaseException as error:
                scan["inventory_exception_type"] = type(error).__name__
                raise
            after_ns = time.monotonic_ns()
            scan["monotonic_end_ns"] = after_ns
            try:
                scan["rows"] = self._rows(rows, {int(pid) for pid in tree.owned})
            except BaseException as error:
                scan["observation_error"] = type(error).__name__
                self._incomplete("snapshot copy failed: " + type(error).__name__)
            return rows

        cleanup._inventory = inventory
        try:
            result = self.original_close(tree)
            return result
        except BaseException as error:
            exception = {"type": type(error).__name__}
            raise
        finally:
            # Restore the original global before sidecar I/O or any metadata failure.
            cleanup._inventory = self.original_inventory
            self.inside = False
            close_end_ns = time.monotonic_ns()
            try:
                owned = {str(pid): dict(row) for pid, row in tree.owned.items()}
                value = {"schema": "inspection-passive-close-observation-v1", "role": "metadata only; original decision retained",
                         "context": self.context, "adapter_source_sha256": self.source_sha,
                         "close_started_at": started, "close_monotonic_start_ns": start_ns,
                         "close_monotonic_end_ns": close_end_ns,
                         "baseline_python_rows": self._rows(tree.baseline, set()),
                         "owned_births": owned, "global_scans": scans,
                         "original_result": result, "original_exception": exception,
                         "observation_errors": list(self.errors),
                         "limits": "only existing close global scans; no full hot/native history or reconstructed ancestry; timeout cleanup may include a live worker"}
                self._write(tree.process.pid, value)
                self.records[-1]["post_close_metadata_wall_ns"] = time.monotonic_ns() - close_end_ns
            except BaseException as error:
                self._incomplete("sidecar failed: " + type(error).__name__)

    def _write(self, root_pid, value):
        if type(root_pid) is not int or root_pid <= 0:
            raise ValueError("positive original root PID required")
        for parent in (OUTPUT, *OUTPUT.parents):
            if parent.is_symlink():
                raise ValueError("sidecar path changed to symbolic alias")
            if parent == PROJECT:
                break
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / f"root-{root_pid}-{time.monotonic_ns()}.json"
        before_ns = time.monotonic_ns()
        data = json.dumps({**value, "sidecar_write_started_at": datetime.now(timezone.utc).isoformat()}, allow_nan=False)
        if len(data.encode()) > 16 * 1024 * 1024:
            raise ValueError("sidecar exceeds the declared JSON bound")
        with path.open("x") as stream:
            stream.write(data)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append({"path": str(path.relative_to(PROJECT)),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "write_wall_ns": time.monotonic_ns() - before_ns})

    def check(self):
        """Check inside this context, after normal metering/settlement."""
        if (self.errors or self.inside or threading.get_ident() != self.thread or _active is not self
                or cleanup.SubprocessTree.close is not self.wrapper
                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self.source_sha):
            raise RuntimeError("cleanup observation incomplete; original close result remains unchanged")
        return list(self.records)
