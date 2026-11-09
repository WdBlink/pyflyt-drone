"""Real subprocess/recovery faults; synthetic flight JSON is never a flight."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys

import pytest

from eval.inspection_episode_receipts import (
    UnresolvedEpisode, canonical_digest, durable_once, execute_local_episode,
)


SCRIPT = r'''
import hashlib,json,sys
from pathlib import Path
out,plan_path,mode,counter_path=map(Path,sys.argv[1:])
plan=json.loads(plan_path.read_text())
count=int(counter_path.read_text())+1 if counter_path.exists() else 1
counter_path.write_text(str(count))
if str(mode)=='fail' or str(mode)=='first-fail' and count==1:
    sys.exit(7)
out.mkdir()
case=b'private fixture, not a real case'
(out/'private-case.json').write_bytes(case)
world={'status':'completed','clock_mode':'wallclock_1x','simulated_seconds':600,
 'physics_steps_executed':144000,'physics_hz':240,'control_hz':120,
 'max_clock_lag_seconds':0.02,'private_case_file_sha256':plan['case_file_sha256'],
 'score':{'C':0.5,'S':0.5,'Q':0.5,'Y':0.,'H':1,'E':0,'SR':0}}
(out/'world.json').write_text(json.dumps(world))
run={'status':'completed','source_hashes':plan['source_hashes'],
 'dependency_hashes':plan['dependency_hashes'],'sources_unchanged':True,'dependencies_unchanged':True,
 'private_case_sha256':hashlib.sha256(case).hexdigest(),'policy':plan['condition'],
 'world_sha256':hashlib.sha256((out/'world.json').read_bytes()).hexdigest(),
 'requested_seconds':600,'realtime':True,'private_case_unchanged':True,
 'runtime_thread_environment':plan['thread_environment']}
if str(mode).startswith('bad-source'):run['sources_unchanged']=False
(out/'run.json').write_text(json.dumps(run))
if str(mode)=='bad-source-missing-case':(out/'private-case.json').unlink()
if str(mode)=='missing-run-bad-copy':
    (out/'run.json').unlink()
    (out/'private-case.json').write_text('known wrong input')
if str(mode)=='missing-run-bad-world':
    (out/'run.json').unlink()
    world['private_case_file_sha256']='known wrong input'
    (out/'world.json').write_text(json.dumps(world))
'''


def inputs(tmp_path):
    import hashlib
    plan = {"condition": "B0", "case_id": "fixture-not-real", "repeat_ordinal": 0,
            "max_attempts": 2, "max_clock_lag_seconds": .1,
            "source_hashes": {"fixture": "a"*64}, "dependency_hashes": {"dependency": "b"*64},
            "case_file_sha256": hashlib.sha256(b"private fixture, not a real case").hexdigest()}
    plan["thread_environment"] = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    script, plan_path = tmp_path/"fixture.py", tmp_path/"fixture-plan.json"
    script.write_text(SCRIPT)
    plan_path.write_text(json.dumps(plan))
    count = tmp_path/"count"

    def command(mode):
        return lambda out: [sys.executable, str(script), str(out), str(plan_path), mode, str(count)]

    return plan, command, count


def test_zero_policy_score_is_completed_and_cached_without_rerun(tmp_path):
    plan, command, counter = inputs(tmp_path)
    root = tmp_path/"episode"
    value, receipt = execute_local_episode(root, plan, command("ok"))
    assert value == 0 and not receipt["cache_reused"]
    value, receipt = execute_local_episode(root, plan, command("fail"))
    assert value == 0 and receipt["cache_reused"] and counter.read_text() == "1"


def test_one_known_failure_then_retry_and_no_more(tmp_path):
    plan, command, counter = inputs(tmp_path)
    _, receipt = execute_local_episode(tmp_path/"episode", plan, command("first-fail"))
    assert counter.read_text() == "2" and receipt["actual_attempts"] == 2
    assert (tmp_path/"episode/attempt-00/terminal.json").exists()


def test_retry_exhaustion_preserves_two_attempts_and_denominator(tmp_path):
    plan, command, counter = inputs(tmp_path)
    with pytest.raises(UnresolvedEpisode, match="exhausted"):
        execute_local_episode(tmp_path/"episode", plan, command("fail"))
    with pytest.raises(UnresolvedEpisode, match="exhausted"):
        execute_local_episode(tmp_path/"episode", plan, command("ok"))
    assert counter.read_text() == "2"


def test_unknown_intent_never_automatically_retries(tmp_path):
    plan, command, counter = inputs(tmp_path)
    root = tmp_path/"episode"
    (root/"attempt-00").mkdir(parents=True)
    durable_once(root/"plan.json", plan)
    durable_once(root/"attempt-00/intent.json", {"plan_sha256": canonical_digest(plan)})
    with pytest.raises(UnresolvedEpisode, match="unknown attempt"):
        execute_local_episode(root, plan, command("ok"))
    assert not counter.exists()


def test_timeout_keeps_unknown_and_recovery_blocks(tmp_path):
    plan, command, counter = inputs(tmp_path)
    root = tmp_path/"episode"
    with pytest.raises(subprocess.TimeoutExpired):
        execute_local_episode(root, plan, lambda out: [sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=.05)
    with pytest.raises(UnresolvedEpisode, match="unknown attempt"):
        execute_local_episode(root, plan, command("ok"))
    assert not (root/"attempt-00/terminal.json").exists() and not counter.exists()


def test_cache_modification_or_new_plan_blocks(tmp_path):
    plan, command, _ = inputs(tmp_path)
    root = tmp_path/"episode"
    execute_local_episode(root, plan, command("ok"))
    with pytest.raises(UnresolvedEpisode, match="identity changed"):
        execute_local_episode(root, {**plan, "case_id": "another"}, command("ok"))
    p = root/"attempt-00/flight/world.json"
    world = json.loads(p.read_text())
    world["score"]["Q"] = .6
    p.write_text(json.dumps(world))
    with pytest.raises(UnresolvedEpisode, match="cached flight|terminal"):
        execute_local_episode(root, plan, command("ok"))


def test_source_fault_is_not_a_free_retry(tmp_path):
    plan, command, counter = inputs(tmp_path)
    with pytest.raises(UnresolvedEpisode, match="source/input/score"):
        execute_local_episode(tmp_path/"episode", plan, command("bad-source"))
    assert counter.read_text() == "1"


def test_concurrent_owner_blocks_without_launch(tmp_path):
    plan, command, counter = inputs(tmp_path)
    root = tmp_path/"episode"
    root.mkdir()
    with (root/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(UnresolvedEpisode, match="another supervisor"):
            execute_local_episode(root, plan, command("ok"))
    assert not counter.exists()


@pytest.mark.parametrize("fault", ["missing_start", "changed_intent", "changed_stdout", "changed_terminal_value"])
def test_entire_receipt_chain_is_bound(tmp_path, fault):
    plan, command, counter = inputs(tmp_path)
    root = tmp_path/"episode"
    execute_local_episode(root, plan, command("ok"))
    attempt = root/"attempt-00"
    if fault == "missing_start":
        (attempt/"started.json").unlink()
    elif fault == "changed_stdout":
        (attempt/"stdout.log").write_text("different output")
    else:
        target = attempt/("intent.json" if fault == "changed_intent" else "terminal.json")
        value = json.loads(target.read_text())
        value["argv" if fault == "changed_intent" else "Y"] = ["another process"] if fault == "changed_intent" else 1
        target.write_text(json.dumps(value))
    with pytest.raises(UnresolvedEpisode):
        execute_local_episode(root, plan, command("ok"))
    assert counter.read_text() == "1"


def test_start_receipt_io_failure_reaps_process_and_leaves_unknown(tmp_path, monkeypatch):
    import eval.inspection_episode_receipts as module
    plan, command, counter = inputs(tmp_path)
    original_write, original_popen = module.durable_once, module.subprocess.Popen
    processes = []
    def injected_write(path, value):
        if Path(path).name == "started.json":
            raise OSError(28, "injected ENOSPC")
        original_write(path, value)
    def capture_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(module, "durable_once", injected_write)
    monkeypatch.setattr(module.subprocess, "Popen", capture_process)
    root = tmp_path/"episode"
    with pytest.raises(OSError, match="ENOSPC"):
        execute_local_episode(root, plan, lambda out: [sys.executable, "-c", "import time; time.sleep(5)"])
    assert len(processes) == 1 and processes[0].poll() is not None
    with pytest.raises(UnresolvedEpisode, match="unknown attempt"):
        execute_local_episode(root, plan, command("ok"))
    assert not counter.exists()


def test_missing_case_cannot_hide_a_nonretryable_source_fault(tmp_path):
    plan, command, counter = inputs(tmp_path)
    with pytest.raises(UnresolvedEpisode, match="source/input/score"):
        execute_local_episode(tmp_path/"episode", plan, command("bad-source-missing-case"))
    assert counter.read_text() == "1"


@pytest.mark.parametrize("mode", ["missing-run-bad-copy", "missing-run-bad-world"])
def test_missing_run_cannot_hide_any_visible_input_binding_fault(tmp_path, mode):
    plan, command, counter = inputs(tmp_path)
    with pytest.raises(UnresolvedEpisode, match="source/input/score"):
        execute_local_episode(tmp_path/"episode", plan, command(mode))
    assert counter.read_text() == "1"
