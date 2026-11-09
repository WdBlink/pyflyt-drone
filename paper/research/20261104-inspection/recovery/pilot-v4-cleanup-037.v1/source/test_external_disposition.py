"""Fail-closed checks using the frozen original sidecar and exact OS events."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import admit_external as admission
from continue_pilot import UnresolvedEpisode, run_observed
from inspect_original import BUNDLE, EPISODE, RESERVATION, SIDE, context, previous, read
from launchd_provenance import match_sidecar


@pytest.fixture
def original():
    sidecar = read(SIDE)
    tree = deepcopy(sidecar["original_result"])
    events = json.loads((BUNDLE / "launchd.v1/slot037.json").read_bytes())
    plan = read(EPISODE / "plan.json")
    return sidecar, tree, context(plan, RESERVATION[:-3]), previous.OBSERVER_SHA, events, 501


def test_real_scan_and_log_rows(original):
    proof = match_sidecar(*original)
    assert proof["pid"] == 65758 and len(proof["original_scan_rows"]) == 2
    assert proof["python_spawn"]["bootUUID"] == "AAE87917-4736-4BAA-99B9-F7C816816E6A"


@pytest.mark.parametrize("change", ("baseline", "owned", "live", "unregistered", "mixed", "observer",
    "error", "scan-end", "wrong-pid", "wrong-uid", "wrong-parent", "wrong-group", "wrong-born",
    "wrong-path", "different-job", "wrong-boot", "wrong-producer", "duplicate", "no-removal"))
def test_rejects_incomplete_or_wrong_process_identity(original, change):
    sidecar, tree, _, _, events, _ = original
    scan = sidecar["global_scans"][-1]
    row = scan["rows"]["65758"]
    if change == "baseline":
        sidecar["baseline_python_rows"]["65758"] = row
    elif change == "owned":
        tree["owned_births"]["60625"]["group"] = 65758
        sidecar["owned_births"] = deepcopy(tree["owned_births"])
    elif change == "live":
        tree["remaining_owned"] = [60625]
    elif change == "unregistered":
        tree["root_pid"] = 99999
    elif change == "mixed":
        tree["unclaimed_new_python_orphans"].append(65759)
    elif change == "observer":
        sidecar["adapter_source_sha256"] = "0" * 64
    elif change == "error":
        sidecar["observation_errors"] = ["probe"]
    elif change == "scan-end":
        scan["monotonic_end_ns"] += 300_000_000
        sidecar["close_monotonic_end_ns"] += 300_000_000
    elif change == "wrong-pid":
        scan["rows"]["657580"] = scan["rows"].pop("65758")
    elif change == "wrong-uid":
        row["uid"] = 502
    elif change == "wrong-parent":
        row["parent"] = 9
    elif change == "wrong-group":
        row["group"] = 60623
    elif change == "wrong-born":
        row["born"] = "Sat Nov 7 10:53:05 2026"
    elif change == "wrong-path":
        row["command"] = "/usr/bin/python"
    elif change == "different-job":
        events[0]["subsystem"] = "gui/501/local.unrelated [65758]"
    elif change == "wrong-boot":
        events[0]["bootUUID"] = "other-boot"
    elif change == "wrong-producer":
        events[0]["senderImagePath"] = "/usr/bin/python"
    elif change == "duplicate":
        events.append(deepcopy(events[0]))
    else:
        events[:] = [e for e in events if e["eventMessage"] != "removing child: pid/65758"]
    sidecar["original_result"] = deepcopy(tree)
    with pytest.raises(ValueError):
        match_sidecar(*original)


def test_cold_collector_seals_exact_query_and_result(original, tmp_path, monkeypatch):
    sidecar, _, _, _, events, _ = original
    before = datetime.fromisoformat(sidecar["close_started_at"])
    class Clock:
        @staticmethod
        def now(tz):
            return (before + timedelta(seconds=5)).astimezone(tz)
    monkeypatch.setattr(admission, "datetime", Clock)
    monkeypatch.setattr(admission, "PROJECT", tmp_path)
    calls = []
    def run(argv, *, capture_output, timeout):
        calls.append(argv)
        subset = events if len(calls) == 2 else [e for e in events if "removing child" not in e["eventMessage"]]
        return SimpleNamespace(returncode=0, stderr=b"", stdout=json.dumps(subset).encode())
    def checked(path, expected=None):
        sha = previous.digest(path)
        assert expected is None or sha == expected
        return {"path": str(path.relative_to(tmp_path)), "sha256": sha}
    monkeypatch.setattr(admission.subprocess, "run", run)
    monkeypatch.setattr(admission, "_boot", lambda: events[0]["bootUUID"])
    monkeypatch.setattr(admission, "checked", checked)
    monkeypatch.setattr(admission.time, "sleep", lambda _: None)
    match, ref = admission._events(65758, sidecar, tmp_path / "slot", lambda rows: match_sidecar(*original[:4], rows, 501))
    assert len(calls) == 2 and match["pid"] == 65758
    assert all(argv[0:2] == ["/usr/bin/log", "show"] for argv in calls)
    assert all('removing child: pid/65758"' in argv[-1] and "subsystem" not in argv[-1] for argv in calls)
    assert (tmp_path / "slot/query-0.json").is_file() and ref["path"] == "slot/query-1.json"
    meta = json.loads((tmp_path / ref["path"]).read_bytes())
    assert meta["boot"] == events[0]["bootUUID"] and meta["events"] == checked(tmp_path / "slot/launchd-query-1.json")


@pytest.mark.parametrize("mode", ("original-closed", "unknown", "old-unknown", "unrelated-error", "no-sidecar"))
def test_dispatch_boundary(mode, tmp_path, monkeypatch):
    import continue_pilot as continuation
    plan = read(EPISODE / "plan.json")
    kwargs = {"reservation_prefix": "dev-all8-pilot-v4-038"}
    if mode == "old-unknown":
        (tmp_path / "attempt-00").mkdir()
    class Observer:
        def __init__(self, info):
            assert info == context(plan, kwargs["reservation_prefix"])
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass
        def check(self):
            return [] if mode == "no-sidecar" else [{"path": "sealed"}]
    def execute(*args, **kwargs):
        events.append("execute")
        if mode == "original-closed":
            return .4, {"cache_reused": False, "actual_attempts": 1}
        raise UnresolvedEpisode("unrelated" if mode == "unrelated-error" else
                                "unknown or source/input/score/causal fault blocks this LLM slot")
    events = []
    def reconcile(directory, actual_plan, prefix, ref, policy):
        assert directory == tmp_path and actual_plan == plan and ref == {"path": "sealed"} and policy == {"pin": 1}
        events.append("reconcile")
        return .5, {"cache_reused": False, "actual_attempts": 1, "cleanup_disposition": {"proof": "sealed"}}
    monkeypatch.setattr(continuation, "reconcile_new", reconcile)
    monkeypatch.setattr(continuation, "check_disposition", lambda *args: events.append("checked"))
    args = (execute, SimpleNamespace(CloseObserver=Observer), tmp_path, plan, None, kwargs, {"pin": 1},
            lambda: events.append("metadata"))
    if mode in {"old-unknown", "unrelated-error", "no-sidecar"}:
        with pytest.raises((UnresolvedEpisode, ValueError)):
            run_observed(*args)
        assert events == ["execute"]
    else:
        y, evidence = run_observed(*args)
        assert evidence["passive_close_observations"] == [{"path": "sealed"}]
        assert (y, events) == ((.5, ["execute", "reconcile", "metadata", "checked"])
                               if mode == "unknown" else (.4, ["execute", "metadata"]))
