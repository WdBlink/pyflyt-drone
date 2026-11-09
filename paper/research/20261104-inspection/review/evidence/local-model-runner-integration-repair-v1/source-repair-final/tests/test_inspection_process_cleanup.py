"""Actual owned worker/daemon process cleanup; no simulation or models."""

import multiprocessing as mp
import time

from eval.inspection_process_cleanup import _inventory, stop_worker_tree


def _hold():
    time.sleep(30)


def _parent(ready):
    context = mp.get_context("spawn")
    child = context.Process(target=_hold, daemon=True)
    child.start()
    ready.put(child.pid)
    time.sleep(30)


def test_stopping_worker_reaps_its_daemon_descendant():
    context = mp.get_context("spawn")
    ready = context.Queue()
    parent = context.Process(target=_parent, args=(ready,))
    parent.start()
    try:
        child_pid = ready.get(timeout=5)
        result = stop_worker_tree(parent)
        assert child_pid in result["descendants"] and result["remaining_live"] == []
        assert not parent.is_alive()
        child = _inventory().get(child_pid)
        assert child is None or child["status"].startswith("Z")
    finally:
        if parent.is_alive():
            stop_worker_tree(parent)
        ready.close()
