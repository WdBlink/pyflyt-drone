"""Software-only recovery checks; fake native rows never enter the real owner."""
import copy
import importlib.util
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import tempfile

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
        plan["condition"] = "B4-M1"
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
    monkeypatch.setattr(m, "_inventory", lambda: {26337: {**row, "parent": 10}})
    with pytest.raises(ValueError, match="cleanup remains"):
        m.quiet_observation(tree)


@pytest.fixture
def approved_fixture(recovery, monkeypatch):
    m = recovery
    with tempfile.TemporaryDirectory(prefix="software-fixture-", dir=m.BUNDLE) as name:
        base = Path(name)
        proposal = m.read(m.BUNDLE / "proposal.v2.json")
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
            request = {"evidence": {r["path"]: r["sha256"] for r in [*refs, source]}}
            request["request_sha256"] = m.admission_module().canonical_sha(request)
            decision = {"request_sha256": request["request_sha256"], "status": "active",
                        "progress": "verified_progress", "finding": {"id": m.FINDING}}
            resource = m.RUN / "qualification/local-model-episodes" / m.RESERVATION / "usage-receipt.json"
            owner = {"rounds": [{"request": request, "decision": decision, "validation": "structure_and_binding_only"}],
                     "usage_ledger": {m.RESERVATION: {"status": "known", "receipt_sha256": m.digest(resource),
                                                   "reported_usage": {"calls": 25, "tokens": 166000}}}}
            monkeypatch.setattr(m, "owner", lambda: owner)
            return owner
        current = install()
        yield m, proposal, acceptance_path, current, install


def test_native_approved_cache_is_immutable_and_offline(approved_fixture, monkeypatch):
    m, _, acceptance, _, _ = approved_fixture
    monkeypatch.setattr(m, "quiet_observation", lambda _: pytest.fail("closed cache must not inspect future processes"))
    monkeypatch.setattr(m, "inspect_original", lambda: pytest.fail("closed cache must not require old live sources"))
    y, evidence = m.verify(acceptance)
    assert y == .368353660488492
    assert evidence["actual_attempts"] == 1 and evidence["cache_reused"] is True
    assert evidence["cleanup_disposition"]["original_status_retained"] == "unknown_process_cleanup"
    assert m.read(m.ATTEMPT / "terminal.json")["Y"] is None


@pytest.mark.parametrize("mutation", ["unapplied", "revised", "target", "observation", "usage", "file_set"])
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


def test_wrapper_delegates_other_slots_and_pauses_after_reconciled_cache(recovery, monkeypatch):
    m = recovery
    calls = []
    monkeypatch.setattr(m, "verify", lambda _: (.368353660488492, {"fixture": "reconciled"}))
    monkeypatch.setattr(m, "live_frozen_sources", lambda: [])
    with tempfile.TemporaryDirectory(prefix="v4-recovery-wrapper-fixture-", dir=m.PROJECT / "logs") as name:
        output = Path(name) / "successor"
        class Loader:
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                module.execute_development_llm_episode = lambda *a, **k: (calls.append((a, k)) or (.1, {}))
                def main():
                    output.mkdir()
                    for leaf in ("episode-order.json", "sampler-seeds.json"):
                        (output / leaf).write_bytes((m.PILOT / leaf).read_bytes())
                    module.execute_development_llm_episode(m.EPISODE.parent / "other-slot", {}, None, run=m.RUN)
                    result = module.execute_development_llm_episode(m.EPISODE, m.read(m.EPISODE / "plan.json"),
                        lambda *_: pytest.fail("reconciled cache dispatched"), run=m.RUN,
                        reservation_prefix="dev-all8-pilot-v4-024")
                    assert result[0] == .368353660488492
                module.main = main
        monkeypatch.setattr(m.importlib.util, "spec_from_file_location", lambda *a, **k: ModuleSpec(a[0], Loader()))
        m.run_continuation(m.BUNDLE / "fixture-acceptance", ["--run", str(m.RUN), "--out", str(output),
            "--flight-receipts", str(m.EPISODE.parent), "--reservation-prefix", "dev-all8-pilot-v4",
            "--schedule-seed", "2026110641", "--sampler-seed", "2026110642"])
        assert len(calls) == 1 and calls[0][0][0].name == "other-slot"
        assert (output / "checkpoint.request").is_file()
        assert m.read(output / "recovery-adapter.json")["physical_denominator"] == 108
