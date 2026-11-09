"""Bounded cleanup of our direct worker and its recorded descendants.

Uses the host process inventory without command arguments or a new dependency.
Identity checks avoid sending signals to an unrelated/reused PID. This is
process recovery, not a hardware/power-loss or hostile-host guarantee.
"""

import os
import signal
import subprocess
import sys
import time

_inventory_counts={"hot_native":0,"hot_legacy":0,"fallback_reasons":{}}


def inventory_diagnostics():
    return {**_inventory_counts,"fallback_reasons":dict(_inventory_counts["fallback_reasons"])}


def _observed_inventory(owned):
    if sys.platform=="darwin":
        try:
            from eval.inspection_darwin_inventory import descendant_inventory,NativeInventoryUnavailable
        except ImportError:
            pass
        else:
            try:rows=descendant_inventory(owned)
            except (NativeInventoryUnavailable,OSError,subprocess.SubprocessError,UnicodeError,ValueError) as error:
                name=type(error).__name__;reasons=_inventory_counts["fallback_reasons"]
                reasons[name]=reasons.get(name,0)+1
            else:
                _inventory_counts["hot_native"]+=1
                return rows
    _inventory_counts["hot_legacy"]+=1
    return _inventory()


def _inventory():
    result = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid=,uid=,stat=,lstart=,comm="],
        check=True, capture_output=True, text=True, timeout=3)
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split(None, 10)
        if len(fields) == 11:
            rows[int(fields[0])] = {"parent": int(fields[1]), "group": int(fields[2]), "uid": int(fields[3]),
                "status": fields[4], "born": " ".join(fields[5:10]), "command": fields[10]}
    return rows


def stop_worker_tree(process):
    if process.pid is None:
        return {"status": "not_started", "descendants": []}
    initial = _inventory()
    root = initial.get(process.pid)
    if root is not None and root["parent"] != os.getpid():
        raise RuntimeError("cleanup target is not this coordinator's direct child")
    owned, frontier = {}, [process.pid]
    while frontier:
        parent = frontier.pop()
        children = [pid for pid, row in initial.items() if row["parent"] == parent]
        for pid in children:
            if pid not in owned:
                owned[pid] = initial[pid]
                frontier.append(pid)

    def live_owned():
        current = _inventory()
        return [pid for pid, old in owned.items() if pid in current
                and current[pid]["born"] == old["born"] and not current[pid]["status"].startswith("Z")]

    for pid in reversed(live_owned()):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    for pid in reversed(live_owned()):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    deadline = time.monotonic()+2
    remaining = live_owned()
    while remaining and time.monotonic() < deadline:
        time.sleep(.02)
        remaining = live_owned()
    if process.is_alive() or remaining:
        raise RuntimeError("worker cleanup remains unknown: "+str(remaining))
    return {"status": "stopped", "descendants": sorted(owned), "remaining_live": []}


class SubprocessTree:
    """Track born identities while a coordinator is alive, then clean them.

An unregistered new same-user Python orphan is uncertainty, not permission to
kill it. This can conservatively block an unrelated daemon launch; it is not a
kernel subreaper or a guarantee against arbitrary fork/setsid/host-loss races.
    """

    def __init__(self):
        self.baseline = _inventory()
        self.owned = {}
        self.process = None

    def attach(self, process):
        self.process = process
        current = _inventory()
        row = current.get(process.pid)
        if row is None or row["parent"] != os.getpid():
            raise RuntimeError("child birth could not be bound before exit")
        self.owned[process.pid] = row
        self.observe(current)

    def observe(self, current=None):
        if current is None:
            # close() can recover a direct child after attach() failed. Keep
            # that original global read until the current root is bound.
            current = (_observed_inventory(self.owned) if self.process is not None
                       and self.process.pid in self.owned else _inventory())
        changed = True
        while changed:
            changed = False
            parents = {pid for pid, old in self.owned.items() if pid in current and current[pid]["born"] == old["born"]}
            for pid, row in current.items():
                if row["parent"] in parents and pid not in self.owned:
                    self.owned[pid] = row
                    changed = True
        return current

    def wait(self, timeout):
        deadline = time.monotonic()+timeout
        while True:
            self.observe()
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.process.args, timeout)
            try:
                code = self.process.wait(timeout=min(.1, remaining))
            except subprocess.TimeoutExpired:
                continue
            self.observe()
            return code

    def close(self):
        current = self.observe()
        root = self.process.pid
        if root not in self.owned:
            row = current.get(root)
            if row is not None and row["parent"] == os.getpid() and self.process.poll() is None:
                self.owned[root] = row
                self.observe(current)
        def owned_live(rows):
            return [pid for pid, old in self.owned.items() if pid in rows
                and rows[pid]["born"] == old["born"] and not rows[pid]["status"].startswith("Z")]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in reversed(owned_live(current)):
                try: os.kill(pid, sig)
                except ProcessLookupError: pass
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
            deadline = time.monotonic()+1
            current = _inventory()
            while owned_live(current) and time.monotonic() < deadline:
                time.sleep(.02)
                current = _inventory()
        unknown = [pid for pid, row in current.items() if pid not in self.owned
            and (pid not in self.baseline or self.baseline[pid]["born"] != row["born"])
            and row.get("uid") == os.getuid() and row["parent"] == 1
            and "python" in row.get("command", "").lower() and not row["status"].startswith("Z")]
        evidence = {"status": "unknown" if root not in self.owned or owned_live(current) or unknown else "closed",
            "root_pid": root, "owned_births": self.owned, "remaining_owned": owned_live(current),
            "unclaimed_new_python_orphans": unknown,
            "scope": "periodic born-identity inventory, not arbitrary descendant/host-loss guarantee"}
        return evidence
