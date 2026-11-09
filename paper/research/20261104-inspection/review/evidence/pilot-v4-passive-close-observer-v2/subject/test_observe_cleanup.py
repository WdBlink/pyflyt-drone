"""Isolated existing-scan and real sidecar I/O checks; no flight or model."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch, tmp_path):
    path = Path(__file__).with_name("observe_cleanup.py")
    spec = importlib.util.spec_from_file_location("_passive_cleanup_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PROJECT", tmp_path)
    monkeypatch.setattr(mod, "OUTPUT", tmp_path / "logs/pilot/cleanup-observations")
    return mod


def context():
    return {"case_id": "dev-L3-01", "condition": "B4-M1", "repeat_ordinal": 1,
            "plan_sha256": "a" * 64, "reservation_prefix": "dev-all8-pilot-v4-030"}


def row(parent, command="/opt/Python"):
    return {"parent": parent, "group": 10, "uid": os.getuid(), "status": "R",
            "born": "Sat Nov 7 08:23:13 2026", "command": command}


def tree(pid=10):
    return SimpleNamespace(process=SimpleNamespace(pid=pid), owned={pid: row(123)}, baseline={8: row(1)})


@pytest.mark.parametrize("scan_count", [0, 1, 2, 5])
def test_return_identity_scan_count_and_snapshot(module, monkeypatch, scan_count):
    inventory = {10: row(123), 99: row(1), 88: row(1, "/bin/zsh")}
    before = copy.deepcopy(inventory)
    result = {"status": "unknown", "root_pid": 10, "unclaimed_new_python_orphans": [99]}
    calls = []

    def original_inventory():
        calls.append(1)
        return inventory

    def original_close(value):
        for _ in range(scan_count):
            assert module.cleanup._inventory() is inventory
        return result

    monkeypatch.setattr(module.cleanup, "_inventory", original_inventory)
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", original_close)
    with module.CloseObserver(context()) as observer:
        assert module.cleanup.SubprocessTree.close(tree()) is result
        assert module.cleanup._inventory is original_inventory
        records = observer.check()
    assert len(calls) == scan_count and inventory == before
    assert module.cleanup.SubprocessTree.close is original_close
    assert len(records) == 1 and records[0]["post_close_metadata_wall_ns"] >= records[0]["write_wall_ns"] >= 0
    saved = json.loads((module.PROJECT / records[0]["path"]).read_text())
    assert saved["original_result"] == result and saved["original_exception"] is None
    assert len(saved["global_scans"]) == scan_count
    for index, scan in enumerate(saved["global_scans"]):
        assert scan["ordinal"] == index and scan["rows"] == {"10": before[10], "99": before[99]}
        assert scan["monotonic_end_ns"] >= scan["monotonic_start_ns"]
    assert saved["baseline_python_rows"] == {"8": row(1)}
    inventory[99]["born"] = "later mutated input"
    assert json.loads((module.PROJECT / records[0]["path"]).read_text()) == saved


@pytest.mark.parametrize("inventory_error", [False, True])
def test_original_exception_identity_and_finally_restore(module, monkeypatch, inventory_error):
    error = OSError("original failure")

    def original_inventory():
        if inventory_error:
            raise error
        return {99: row(1)}

    def original_close(value):
        module.cleanup._inventory()
        raise error

    monkeypatch.setattr(module.cleanup, "_inventory", original_inventory)
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", original_close)
    with module.CloseObserver(context()) as observer:
        with pytest.raises(OSError) as caught:
            module.cleanup.SubprocessTree.close(tree())
        assert caught.value is error and module.cleanup._inventory is original_inventory
        record = observer.check()[0]
    saved = json.loads((module.PROJECT / record["path"]).read_text())
    assert saved["original_exception"] == {"type": "OSError"} and saved["original_result"] is None
    if inventory_error:
        assert saved["global_scans"][0]["inventory_exception_type"] == "OSError"
    assert module.cleanup.SubprocessTree.close is original_close


def test_sidecar_failure_cannot_change_return_or_hide_incomplete(module, monkeypatch):
    result = {"status": "closed"}
    original_inventory = lambda: {}
    original_close = lambda value: result
    monkeypatch.setattr(module.cleanup, "_inventory", original_inventory)
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", original_close)
    with module.CloseObserver(context()) as observer:
        def fail_write(*args):
            assert module.cleanup._inventory is original_inventory
            raise OSError("disk full")
        monkeypatch.setattr(observer, "_write", fail_write)
        assert module.cleanup.SubprocessTree.close(tree()) is result
        with pytest.raises(RuntimeError, match="incomplete"):
            observer.check()
    assert module.cleanup.SubprocessTree.close is original_close


def test_metadata_and_stderr_errors_do_not_mask_close(module, monkeypatch):
    result = {"status": "closed"}
    monkeypatch.setattr(module.cleanup, "_inventory", lambda: {99: row(1)})
    def original_close(value):
        module.cleanup._inventory()
        return result
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", original_close)
    with module.CloseObserver(context()) as observer:
        monkeypatch.setattr(observer, "_rows", lambda *args: (_ for _ in ()).throw(ValueError("bad metadata")))
        with monkeypatch.context() as local:
            local.setattr(module.sys, "stderr", SimpleNamespace(write=lambda *args: (_ for _ in ()).throw(BrokenPipeError())))
            assert module.cleanup.SubprocessTree.close(tree()) is result
        with pytest.raises(RuntimeError, match="incomplete"):
            observer.check()


def test_foreign_thread_still_cleans_but_cannot_pass_observation(module, monkeypatch):
    result, calls, returned = {"status": "closed"}, [], []
    def original_close(value):
        calls.append(value)
        return result
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", original_close)
    with module.CloseObserver(context()) as observer:
        value = tree()
        thread = threading.Thread(target=lambda: returned.append(module.cleanup.SubprocessTree.close(value)))
        thread.start()
        thread.join()
        assert calls == [value] and returned[0] is result
        with pytest.raises(RuntimeError, match="incomplete"):
            observer.check()


def test_nested_context_rejected_without_displacing_original_owner(module):
    with module.CloseObserver(context()) as first:
        with pytest.raises(RuntimeError, match="serial"):
            with module.CloseObserver(context()):
                pass
        assert module.cleanup.SubprocessTree.close is first.wrapper and first.check() == []


def test_reentry_keeps_each_original_call_and_marks_incomplete(module, monkeypatch):
    outer, inner, calls = tree(10), tree(20), []
    results = {10: {"status": "closed"}, 20: {"status": "unknown"}}
    def original_close(value):
        calls.append(value.process.pid)
        if value is outer:
            assert module.cleanup.SubprocessTree.close(inner) is results[20]
        return results[value.process.pid]
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", original_close)
    with module.CloseObserver(context()) as observer:
        assert module.cleanup.SubprocessTree.close(outer) is results[10]
        assert calls == [10, 20]
        with pytest.raises(RuntimeError, match="incomplete"):
            observer.check()


@pytest.mark.parametrize("replacement", [{"repeat_ordinal": True}, {"case_id": "main-L3-01"},
                                           {"reservation_prefix": "formal-main-000"}, {"extra": "field"}])
def test_foreign_or_malformed_context_refused(module, replacement):
    with pytest.raises(ValueError):
        module.CloseObserver({**context(), **replacement})


def test_symlink_before_enter_refused(module, tmp_path):
    module.OUTPUT.parent.mkdir(parents=True)
    target = tmp_path / "outside"
    target.mkdir()
    module.OUTPUT.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        with module.CloseObserver(context()):
            pass
    assert list(target.iterdir()) == []


def test_symlink_after_enter_preserves_result_but_blocks_observation(module, monkeypatch, tmp_path):
    result = {"status": "closed"}
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", lambda value: result)
    with module.CloseObserver(context()) as observer:
        module.OUTPUT.parent.mkdir(parents=True)
        target = tmp_path / "outside"
        target.mkdir()
        module.OUTPUT.symlink_to(target, target_is_directory=True)
        assert module.cleanup.SubprocessTree.close(tree()) is result
        with pytest.raises(RuntimeError, match="incomplete"):
            observer.check()
    assert list(target.iterdir()) == []


def test_real_original_close_unknown_is_unchanged(module, monkeypatch):
    # Synthetic process table and wait object; no process or signal is sent.
    original = module.cleanup.SubprocessTree.close
    table = {99: row(1)}
    calls = []
    def inventory():
        calls.append(1)
        return table
    monkeypatch.setattr(module.cleanup, "_inventory", inventory)
    monkeypatch.setattr(module.cleanup, "_observed_inventory", lambda owned: module.cleanup._inventory())
    monkeypatch.setattr(module.cleanup.os, "kill", lambda *args: pytest.fail("no live owned process may be signaled"))
    def value():
        result = module.cleanup.SubprocessTree.__new__(module.cleanup.SubprocessTree)
        result.baseline, result.owned = {}, {10: row(123)}
        result.process = SimpleNamespace(pid=10, wait=lambda timeout: 0, poll=lambda: 0)
        return result
    expected = original(value())
    count = len(calls)
    calls.clear()
    with module.CloseObserver(context()) as observer:
        actual = module.cleanup.SubprocessTree.close(value())
        assert actual == expected and actual["status"] == "unknown" and actual["unclaimed_new_python_orphans"] == [99]
        record = observer.check()[0]
    assert len(calls) == count
    saved = json.loads((module.PROJECT / record["path"]).read_text())
    assert saved["original_result"] == json.loads(json.dumps(expected))
    assert saved["global_scans"][-1]["rows"]["99"] == table[99]


@pytest.mark.parametrize("mutation", ["delete", "status", "external_symlink", "hardlink"])
def test_fixed_sidecar_references_checked_before_return(module, monkeypatch, tmp_path, mutation):
    result = {"status": "unknown", "root_pid": 10, "unclaimed_new_python_orphans": [99]}
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", lambda value: result)
    with module.CloseObserver(context()) as observer:
        assert module.cleanup.SubprocessTree.close(tree()) is result
        reference = observer.check()[0]
        path = module.PROJECT / reference["path"]
        if mutation == "delete":
            path.unlink()
        elif mutation == "status":
            data = json.loads(path.read_text())
            data["original_result"]["status"] = "closed"
            path.write_text(json.dumps(data))
        else:
            outside = tmp_path.parent / (tmp_path.name + "-outside.json")
            if mutation == "hardlink":
                os.link(path, outside)
            else:
                outside.write_bytes(path.read_bytes())
                path.unlink()
                path.symlink_to(outside)
        with pytest.raises(RuntimeError, match="incomplete"):
            observer.check()


def test_returned_refs_cannot_rebind_the_private_write_digest(module, monkeypatch):
    result = {"status": "unknown"}
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", lambda value: result)
    with module.CloseObserver(context()) as observer:
        assert module.cleanup.SubprocessTree.close(tree()) is result
        first = observer.check()
        first[0]["sha256"] = "0" * 64
        assert observer.check()[0]["sha256"] != "0" * 64


def test_parent_alias_inserted_in_hash_window_rejected(module, monkeypatch, tmp_path):
    result = {"status": "unknown"}
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", lambda value: result)
    with module.CloseObserver(context()) as observer:
        assert module.cleanup.SubprocessTree.close(tree()) is result
        reference = observer.check()[0]
        payload = (module.PROJECT / reference["path"]).read_bytes()
        target = tmp_path.parent / (tmp_path.name + "-moved")
        original_sha = module.hashlib.sha256
        moved = []
        def changing_sha(data):
            value = original_sha(data)
            if data == payload and not moved:
                module.OUTPUT.rename(target)
                module.OUTPUT.symlink_to(target, target_is_directory=True)
                moved.append(True)
            return value
        monkeypatch.setattr(module.hashlib, "sha256", changing_sha)
        with pytest.raises(RuntimeError, match="incomplete"):
            observer.check()
        assert moved == [True]
        module.OUTPUT.unlink()
        target.rename(module.OUTPUT)


def test_missing_second_record_is_not_replaced_by_previous_record(module, monkeypatch):
    result = {"status": "closed"}
    monkeypatch.setattr(module.cleanup.SubprocessTree, "close", lambda value: result)
    with module.CloseObserver(context()) as observer:
        assert module.cleanup.SubprocessTree.close(tree()) is result
        assert len(observer.check()) == 1
        monkeypatch.setattr(observer, "_write", lambda *args: None)
        assert module.cleanup.SubprocessTree.close(tree()) is result
        with pytest.raises(RuntimeError, match="incomplete"):
            observer.check()
