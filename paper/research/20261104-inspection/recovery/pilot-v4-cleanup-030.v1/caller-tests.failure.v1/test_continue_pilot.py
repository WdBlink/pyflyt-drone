"""Software-only recovery checks; fake native rows never enter the real owner."""
import copy
import importlib.util
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest


@pytest.fixture
def recovery():
    path = Path(__file__).with_name("continue_pilot.py")
    spec = importlib.util.spec_from_file_location("v4_cleanup_recovery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mutation", ["plan", "status", "retry", "owned", "orphan", "second_attempt"])
def test_exact_target(recovery, monkeypatch, tmp_path, mutation):
    m = recovery
    plan, terminal, tree = [m.read(p) for p in
                            (m.EPISODE / "plan.json", m.ATTEMPT / "terminal.json", m.ATTEMPT / "process-tree.json")]
    m.target(plan, terminal, tree)
    if mutation == "plan":
        plan["condition"] = "B3-M1"
    elif mutation == "status":
        terminal["status"] = "completed"
    elif mutation == "retry":
        terminal["retry_allowed"] = True
    elif mutation == "owned":
        tree["remaining_owned"] = [21710]
    elif mutation == "orphan":
        tree["unclaimed_new_python_orphans"] = [1]
    else:
        monkeypatch.setattr(m, "EPISODE", tmp_path)
        (tmp_path / "attempt-01").mkdir()
    with pytest.raises(ValueError):
        m.target(plan, terminal, tree)


def test_quiet_observation_does_not_signal(recovery, monkeypatch):
    m = recovery
    tree = m.read(m.ATTEMPT / "process-tree.json")
    monkeypatch.setattr(m, "_inventory", lambda: {})
    assert m.quiet_observation(tree)["original_unclaimed_pid_present"] is False
    row = {"parent": 1, "uid": m.os.getuid(), "command": "Python", "status": "S", "born": "fixture"}
    monkeypatch.setattr(m, "_inventory", lambda: {99001: row})
    with pytest.raises(ValueError, match="cleanup remains"):
        m.quiet_observation(tree)
    monkeypatch.setattr(m, "_inventory", lambda: {86715: {**row, "parent": 10}})
    with pytest.raises(ValueError, match="cleanup remains"):
        m.quiet_observation(tree)


@pytest.fixture
def approved_fixture(recovery, monkeypatch):
    m = recovery
    with tempfile.TemporaryDirectory(prefix="software-fixture-", dir=m.BUNDLE) as name:
        base = Path(name)
        proposal = m.read(m.BUNDLE / "proposal.v2.json")
        old_closed = m.older().closed_inputs()
        prior_proof = copy.deepcopy(proposal["prior_cleanup_evidence"])
        monkeypatch.setattr(m, "older", lambda: SimpleNamespace(
            closed_inputs=lambda: copy.deepcopy(old_closed),
            verify=lambda _: (.368353660488492, copy.deepcopy(prior_proof))))
        proposal_path, review_path, acceptance_path = [base / x for x in ("proposal.json", "review.json", "acceptance.json")]
        def install():
            for path in (proposal_path, review_path, acceptance_path):
                if path.exists():
                    path.unlink()
            m.durable_once(proposal_path, proposal)
            review = {"schema": "inspection-dev-late-cleanup-review-v1", "decision": "accept_late_cleanup_only",
                      "proposal_sha256": m.digest(proposal_path), "reviewer": "software-fixture-independent"}
            m.durable_once(review_path, review)
            refs = [{"path": str(p.relative_to(m.RUN)), "sha256": m.digest(p)} for p in (proposal_path, review_path)]
            m.durable_once(acceptance_path, {"schema": "inspection-dev-late-cleanup-acceptance-v1",
                                          "proposal": refs[0], "review": refs[1], "finding_id": m.FINDING})
            source = {"path": str(Path(m.__file__).relative_to(m.RUN)), "sha256": m.digest(Path(m.__file__))}
            observed = [{"path": str((m.PROJECT / row["path"]).relative_to(m.RUN)), "sha256": row["sha256"]}
                        for row in m.observer_refs()]
            request = {"evidence": {r["path"]: r["sha256"] for r in [*refs, source, *observed]}}
            request["request_sha256"] = m.admission_module().canonical_sha(request)
            decision = {"request_sha256": request["request_sha256"], "status": "active",
                        "progress": "verified_progress", "finding": {"id": m.FINDING}}
            resource = m.RUN / "qualification/local-model-episodes" / m.RESERVATION / "usage-receipt.json"
            owner = {"rounds": [{"request": request, "decision": decision, "validation": "structure_and_binding_only"}],
                     "usage_ledger": {m.RESERVATION: {"status": "known", "receipt_sha256": m.digest(resource),
                                                   "reported_usage": {"calls": 24, "tokens": 173756}}}}
            monkeypatch.setattr(m, "owner", lambda: owner)
            return owner
        current = install()
        yield m, proposal, acceptance_path, current, install


def test_native_approved_cache_is_immutable_and_offline(approved_fixture, monkeypatch):
    m, _, acceptance, _, _ = approved_fixture
    monkeypatch.setattr(m, "quiet_observation", lambda _: pytest.fail("closed cache must not inspect future processes"))
    monkeypatch.setattr(m, "inspect_original", lambda: pytest.fail("closed cache must not require old live sources"))
    y, evidence = m.verify(acceptance)
    assert y == .3436623024638006
    assert evidence["actual_attempts"] == 1 and evidence["cache_reused"] is True
    assert evidence["cleanup_disposition"]["original_status_retained"] == "unknown_process_cleanup"
    assert m.read(m.ATTEMPT / "terminal.json")["Y"] is None


@pytest.mark.parametrize("mutation", ["unapplied", "revised", "target", "observation", "usage", "file_set", "prior_proof"])
def test_invalid_dispositions_block(approved_fixture, monkeypatch, mutation):
    m, proposal, acceptance, current, install = approved_fixture
    if mutation == "unapplied":
        current["rounds"] = []
    elif mutation == "revised":
        current["rounds"].append({"decision": {"finding": {"revises": m.FINDING}}})
    elif mutation in ("target", "observation"):
        if mutation == "target":
            proposal["target"]["order_index"] = 25
        else:
            proposal["observations"][0]["original_unclaimed_pid_present"] = True
        install()
    elif mutation == "prior_proof":
        proposal["prior_cleanup_evidence"]["reservation"] = "unaccepted-other-reservation"
        install()
    elif mutation == "usage":
        current["usage_ledger"][m.RESERVATION]["reported_usage"]["tokens"] += 1
    else:
        original = m._bindings
        def changed(*args):
            payload, model = original(*args)
            return {**payload, "unbound-extra": "f" * 64}, model
        monkeypatch.setattr(m, "_bindings", changed)
    with pytest.raises(ValueError):
        m.verify(acceptance)


def test_no_native_acceptance_no_dispatch(recovery, tmp_path):
    with pytest.raises(ValueError):
        recovery.verify(tmp_path / "missing.json")


@pytest.mark.parametrize("mutation", ["file", "acceptance", "directory"])
def test_complete_final_window(approved_fixture, monkeypatch, mutation):
    m, proposal, acceptance, _, install = approved_fixture
    extra = acceptance.parent / "bound-archive.bin"
    extra.write_bytes(b"original fixture archive")
    proposal["files"].append(m.checked(extra))
    current = install()
    changed = False
    original_snapshot = m.closed_inputs
    def snapshot():
        result = original_snapshot()
        if changed and mutation == "directory":
            result["directories"].append("fixture-added-empty-call")
        return result
    monkeypatch.setattr(m, "closed_inputs", snapshot)
    def owner_window():
        nonlocal changed
        changed = True
        if mutation == "file":
            extra.write_bytes(b"mutated fixture archive")
        elif mutation == "acceptance":
            value = m.read(acceptance)
            value["finding_id"] = "unvalidated-new-finding"
            acceptance.write_text(json.dumps(value))
        return current
    monkeypatch.setattr(m, "owner", owner_window)
    with pytest.raises(ValueError):
        m.verify(acceptance)


def test_closed_entry_types(recovery, monkeypatch, tmp_path):
    m = recovery
    monkeypatch.setattr(m, "older", lambda: SimpleNamespace(closed_inputs=lambda: {"directories": [], "files": {}}))
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    monkeypatch.setattr(m, "EPISODE", tmp_path / "episode")
    monkeypatch.setattr(m, "RUN", tmp_path / "run")
    monkeypatch.setattr(m, "operation_directory", lambda _: tmp_path / "operation")
    resource = m.RUN / "qualification/local-model-episodes" / m.RESERVATION
    for base in (m.EPISODE, tmp_path / "operation", resource):
        base.mkdir(parents=True)
        (base / "one.json").write_text("{}")
    before = m.closed_inputs()
    (tmp_path / "operation/call-empty").mkdir()
    assert m.closed_inputs() != before
    m.os.mkfifo(tmp_path / "operation/fifo")
    with pytest.raises(ValueError, match="regular file or directory"):
        m.closed_inputs()


@pytest.mark.parametrize("reviewer", [[], "", "/root"])
def test_observer_review_requires_independent_string_identity(recovery, monkeypatch, reviewer):
    m = recovery
    value = m.read(m.OBSERVER_REVIEW)
    value["reviewer"] = reviewer
    with tempfile.TemporaryDirectory(prefix="observer-review-fixture-", dir=m.BUNDLE) as name:
        path = Path(name) / "review.json"
        path.write_text(json.dumps(value))
        monkeypatch.setattr(m, "OBSERVER_REVIEW", path)
        with pytest.raises(ValueError, match="independent passive"):
            m.observer_refs()


@pytest.mark.parametrize("scenario", ["new", "cached", "missing", "sidecar_error"])
def test_two_exact_caches_delegate_and_observer_integration(recovery, monkeypatch, scenario):
    """Real observer/fileIO; fake providedSDK/worker, no owner/effect dispatch."""
    m = recovery
    old, instrument = m.older(), m.observer_module()
    previous_plan = m.read(old.EPISODE / "plan.json")
    current_plan = m.read(m.EPISODE / "plan.json")
    observed_refs = m.observer_refs()
    prior_y, prior_evidence = .368353660488492, {"fixture": "accepted024", "cache_reused": True, "actual_attempts": 1}
    current_y, current_evidence = .3436623024638006, {"fixture": "accepted030", "cache_reused": True, "actual_attempts": 1}
    monkeypatch.setattr(m, "older", lambda: SimpleNamespace(EPISODE=old.EPISODE, PLAN_CANON=old.PLAN_CANON,
        verify=lambda _: (prior_y, copy.deepcopy(prior_evidence))))
    monkeypatch.setattr(m, "verify", lambda _: (current_y, copy.deepcopy(current_evidence)))
    monkeypatch.setattr(m, "live_frozen_sources", lambda: [])
    monkeypatch.setattr(m, "observer_refs", lambda: copy.deepcopy(observed_refs))
    monkeypatch.setattr(m, "observer_module", lambda: instrument)
    calls, returned = [], []
    original_result = {"status": "closed", "root_pid": 10, "remaining_owned": [], "unclaimed_new_python_orphans": []}
    def original_close(value):
        instrument.cleanup._inventory()
        returned.append(original_result)
        return original_result
    monkeypatch.setattr(instrument.cleanup, "_inventory", lambda: {})
    monkeypatch.setattr(instrument.cleanup.SubprocessTree, "close", original_close)
    if scenario == "sidecar_error":
        monkeypatch.setattr(instrument.CloseObserver, "_write", lambda *args: (_ for _ in ()).throw(OSError("fixture disk failure")))
    with tempfile.TemporaryDirectory(prefix="v4-observed-wrapper-fixture-", dir=m.PROJECT / "logs") as name:
        project = Path(name)
        output = project / "logs/inspection-eight-condition-pilot.v4.continuation.v2"
        monkeypatch.setattr(m, "PROJECT", project)
        monkeypatch.setattr(instrument, "PROJECT", project)
        monkeypatch.setattr(instrument, "OUTPUT", output / "cleanup-observations")
        monkeypatch.setattr(m, "checked", lambda path, expected=None: {"path": str(path), "sha256": m.digest(path) if Path(path).is_file() else "f" * 64})
        class Loader:
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                def execute(directory, plan, command_factory, **kwargs):
                    calls.append((directory, plan, kwargs))
                    if scenario in ("new", "sidecar_error"):
                        value = SimpleNamespace(process=SimpleNamespace(pid=10), owned={}, baseline={})
                        assert instrument.cleanup.SubprocessTree.close(value) is original_result
                    return .1, {"cache_reused": scenario == "cached", "actual_attempts": 1}
                module.execute_development_llm_episode = execute
                def main():
                    output.mkdir(parents=True)
                    for leaf in ("episode-order.json", "sampler-seeds.json"):
                        (output / leaf).write_bytes((m.PILOT / leaf).read_bytes())
                    result = module.execute_development_llm_episode(m.EPISODE.parent / "other-slot", current_plan, None,
                        run=m.RUN, reservation_prefix="dev-all8-pilot-v4-031")
                    assert result[0] == .1
                    if scenario == "new":
                        assert len(result[1]["passive_close_observations"]) == 1
                        ref = result[1]["passive_close_observations"][0]
                        assert m.digest(project / ref["path"]) == ref["sha256"]
                    else:
                        assert "passive_close_observations" not in result[1]
                    first = module.execute_development_llm_episode(old.EPISODE, previous_plan,
                        lambda *_: pytest.fail("024 cache dispatched"), run=m.RUN, reservation_prefix="dev-all8-pilot-v4-024")
                    second = module.execute_development_llm_episode(m.EPISODE, current_plan,
                        lambda *_: pytest.fail("030 cache dispatched"), run=m.RUN, reservation_prefix="dev-all8-pilot-v4-030")
                    assert first == (prior_y, prior_evidence) and second == (current_y, current_evidence)
                module.main = main
        monkeypatch.setattr(m.importlib.util, "spec_from_file_location", lambda *a, **k: ModuleSpec(a[0], Loader()))
        argv = ["--run", str(m.RUN), "--out", str(output), "--flight-receipts", str(m.EPISODE.parent),
                "--reservation-prefix", "dev-all8-pilot-v4", "--schedule-seed", "2026110641", "--sampler-seed", "2026110642"]
        before_argv = list(m.sys.argv)
        if scenario in ("missing", "sidecar_error"):
            with pytest.raises((ValueError, RuntimeError)):
                m.run_continuation(m.BUNDLE / "fixture-acceptance", argv)
            assert not (output / "checkpoint.request").exists()
        else:
            m.run_continuation(m.BUNDLE / "fixture-acceptance", argv)
            assert (output / "checkpoint.request").is_file()
            assert m.read(output / "recovery-adapter.json")["physical_denominator"] == 108
        assert len(calls) == 1 and m.sys.argv == before_argv
        assert instrument.cleanup.SubprocessTree.close is original_close
        if returned:
            assert returned[0] is original_result


