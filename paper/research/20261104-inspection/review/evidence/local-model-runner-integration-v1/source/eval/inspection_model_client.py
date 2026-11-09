"""Single-request asynchronous pipe client for the reserved local service.

The caller owns episode/source validation and resource settlement. A missing
completion is an operational unknown, never a free retry or inferred zero use.
"""

from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import threading
import time


class ModelOperationUnknown(RuntimeError):
    pass


class LocalModelClient:
    def __init__(self, argv, log_directory, *, ready_timeout=120.):
        self.directory = Path(log_directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stderr = (self.directory/"stderr.log").open("xb")
        self.stdout = (self.directory/"stdout.log").open("x")
        self.messages = queue.Queue()
        self.inflight = None
        self.submitted = 0
        try:
            self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, text=True, bufsize=1, start_new_session=True)
        except BaseException:
            self.stdout.close()
            self.stderr.close()
            raise
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            ready = self.messages.get(timeout=ready_timeout)
            if ready.get("kind") != "ready":
                raise ModelOperationUnknown("service did not provide a ready envelope")
            self.ready = ready
        except BaseException:
            self.stop_unknown()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                self.stdout.write(line)
                self.stdout.flush()
                try:
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError("response must be an object")
                except (ValueError, TypeError):
                    self.messages.put({"kind": "protocol_failure", "reason": "non-JSON service stdout"})
                else:
                    self.messages.put(message)
        finally:
            self.messages.put({"kind": "eof"})

    def submit(self, messages, seed):
        if self.inflight is not None or self.process.poll() is not None:
            raise ModelOperationUnknown("service stopped or prior request is unresolved")
        started = time.monotonic()
        # Mark submitted before pipe I/O: an interrupted write may have reached
        # the child and must not be treated as proof of no generation.
        self.submitted += 1
        self.inflight = {"submitted_ordinal": self.submitted, "started": started}
        self.process.stdin.write(json.dumps({"messages": messages, "seed": seed}, allow_nan=False)+"\n")
        self.process.stdin.flush()

    def poll(self):
        if self.inflight is None:
            return None
        try:
            message = self.messages.get_nowait()
        except queue.Empty:
            if self.process.poll() is not None:
                raise ModelOperationUnknown("service exited without a reconciled response")
            return None
        if message.get("kind") not in ("response", "not_started"):
            raise ModelOperationUnknown("unexpected/unknown service response: "+str(message.get("kind")))
        elapsed = time.monotonic()-self.inflight["started"]
        message = {**message, "parent_end_to_end_wall_seconds": elapsed,
            "submitted_ordinal": self.inflight["submitted_ordinal"]}
        self.inflight = None
        return message

    def finish(self, *, timeout=120.):
        """Close stdin, retain actual final response, require process completion."""
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=timeout)
            self.reader.join(timeout=5)
            if code != 0 or self.reader.is_alive():
                raise ModelOperationUnknown("service did not complete cleanly")
            response = self.poll() if self.inflight is not None else None
            if self.inflight is not None:
                raise ModelOperationUnknown("unreconciled final request")
            return response
        finally:
            if self.process.poll() is None:
                self.stop_unknown()
            else:
                self.stdout.close()
                self.stderr.close()

    def stop_unknown(self):
        from eval.inspection_episode_receipts import _cleanup
        if self.process.poll() is None:
            try:
                _cleanup(self.process)
            except PermissionError:
                if self.process.poll() is None:
                    raise
        else:
            self.process.wait()
        self.reader.join(timeout=5)
        self.stdout.close()
        self.stderr.close()
