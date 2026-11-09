"""Bounded cleanup of our direct worker and its recorded descendants.

Uses the host process inventory without command arguments or a new dependency.
Identity checks avoid sending signals to an unrelated/reused PID. This is
process recovery, not a hardware/power-loss or hostile-host guarantee.
"""

import os
import signal
import subprocess
import time


def _inventory():
    result = subprocess.run(["ps", "-axo", "pid=,ppid=,stat=,lstart="],
        check=True, capture_output=True, text=True, timeout=3)
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4:
            rows[int(fields[0])] = {"parent": int(fields[1]), "status": fields[2], "born": fields[3]}
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
