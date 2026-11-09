"""Independent SYNTHETIC-only CPU/subprocess probes. No MLX/physics/owner state."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
spec = importlib.util.spec_from_file_location("frozen_unit_fixture", HERE / "source/tests/test_inspection_llm_episode_receipts.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
module = fixtures.module
results = []


def setup(name, transform=None):
    directory = HERE / "probes" / name
    directory.mkdir(parents=True, exist_ok=False)
    monkey = pytest.MonkeyPatch()
    plan, episode, counter, effects, execute = fixtures.fixture(directory, monkey)
    if transform is not None:
        script = directory / "synthetic-child.py"
        script.write_text(transform(script.read_text()))
        plan["source_hashes"][str(script)] = module.digest(script)
        (directory / "plan-input.json").write_text(json.dumps(plan))
    return directory, monkey, plan, episode, counter, effects, execute


def save(name, **values):
    row = {"probe": name, "role": "synthetic CPU fixture, not a real model or flight", **values}
    results.append(row)
    print(json.dumps(row), flush=True)


# 1. A clock failure masks a simultaneously wrong actual generation seed.
d, monkey, plan, episode, counter, effects, execute = setup(
    "clock-plus-wrong-seed",
    lambda s: s.replace("(mode=='wrong-seed')", "(mode=='wrong-seed' or mode=='first-clock' and count==1)"),
)
try:
    y, receipt = execute("first-clock")
    first = json.loads((episode / "attempt-00/terminal.json").read_text())
    call = json.loads((d / "operations/fixture-M1-a0/call-0001/request.json").read_text())
    actor = [json.loads(line) for line in (episode / "attempt-00/flight/actor.jsonl").read_text().splitlines()]
    save("clock-plus-wrong-seed", returned_y=y, actual_dispatches=int(counter.read_text()),
         owner_actions=[action for action, _ in effects], first_status=first["status"],
         first_retry=first["retry_allowed"], first_admission_error=first["admission_error"],
         first_physical_reasons=first["physical_validation"]["failure_reasons"],
         actual_request_seed=call["seed"], public_seed=actor[0]["sampler_seed"],
         completed_attempts=receipt["actual_attempts"])
finally:
    monkey.undo()


# 2. A post-spawn started.json write failure returns with the child still alive.
d, monkey, plan, episode, counter, effects, execute = setup(
    "started-write-failure", lambda s: "import time\ntime.sleep(30)\n" + s)
children = []
real_popen, real_once = module.subprocess.Popen, module.durable_once
def capture(*args, **kwargs):
    p = real_popen(*args, **kwargs)
    children.append(p)
    return p
def fail_started(path, value):
    if Path(path).name == "started.json":
        raise OSError("independent synthetic started fsync/write fault")
    return real_once(path, value)
monkey.setattr(module.subprocess, "Popen", capture)
monkey.setattr(module, "durable_once", fail_started)
try:
    try:
        execute()
    except OSError as error:
        save("started-write-failure", exception=str(error), child_alive_after_raise=children[0].poll() is None,
             pid=children[0].pid, terminal_exists=(episode / "attempt-00/terminal.json").exists(),
             owner_actions=[action for action, _ in effects])
    else:
        raise AssertionError("fault was not injected")
finally:
    monkey.undo()
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


# 3. Normal return does not clean an independently grouped lingering descendant.
prefix = """import subprocess,sys\nfrom pathlib import Path\nleak=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nPath(sys.argv[-1]+'.orphan').write_text(str(leak.pid))\n"""
d, monkey, plan, episode, counter, effects, execute = setup("normal-orphan", lambda s: prefix + s)
orphan = None
try:
    y, receipt = execute()
    orphan = int(Path(str(counter) + ".orphan").read_text())
    os.kill(orphan, 0)
    save("normal-orphan", returned_y=y, cached=receipt["cache_reused"], live_orphan_pid=orphan,
         terminal_status=json.loads((episode / "attempt-00/terminal.json").read_text())["status"],
         owner_actions=[action for action, _ in effects])
finally:
    monkey.undo()
    if orphan is not None:
        try:
            os.killpg(orphan, signal.SIGKILL)
        except ProcessLookupError:
            pass


# 4. The settled external archive is not part of cache payload validation.
d, monkey, plan, episode, counter, effects, execute = setup("missing-settled-archive")
try:
    execute()
    archive = d / "owner/qualification/local-model-episodes/fixture-M1-a0/complete-attempt.zip"
    archive.unlink()
    y, receipt = execute("always-clock")
    save("missing-settled-archive", returned_y=y, cache_reused=receipt["cache_reused"],
         archive_exists=archive.exists(), actual_dispatches=int(counter.read_text()),
         owner_actions=[action for action, _ in effects])
finally:
    monkey.undo()


# 5. Admission ignores the logged enqueue witness, accepting its clock past 600.
def enqueue_after_deadline(s):
    s = s.replace("text='{\"commands\":[]}'", "text='{\"commands\":[{\"name\":\"enter_safe_exit\",\"args\":{\"drone_id\":0,\"expected_version\":0}}]}'")
    s = s.replace("'reply':{'commands':[]}", "'reply':json.loads(text)")
    s = s.replace("'commands_enqueued_not_accepted':[],'worker_progress_at_enqueue':after", "'commands_enqueued_not_accepted':json.loads(text)['commands'],'worker_progress_at_enqueue':[601.,20,0,80]")
    return s.replace("'actor_mutating_calls':0", "'actor_mutating_calls':1")
d, monkey, plan, episode, counter, effects, execute = setup("enqueue-after-deadline", enqueue_after_deadline)
try:
    y, receipt = execute()
    rows = [json.loads(line) for line in (episode / "attempt-00/flight/actor.jsonl").read_text().splitlines()]
    save("enqueue-after-deadline", returned_y=y, terminal_status=json.loads((episode / "attempt-00/terminal.json").read_text())["status"],
         response_clock=rows[1]["worker_after_response"][0], enqueue_clock=rows[2]["worker_progress_at_enqueue"][0],
         accepted_enqueued=rows[2]["commands_enqueued_not_accepted"])
finally:
    monkey.undo()


# 6. Silent deletion of actor-call links / original model call is conservative.
for name, target in (("missing-actor-call", "actor"), ("missing-model-response", "response")):
    d, monkey, plan, episode, counter, effects, execute = setup(name)
    try:
        execute()
        flight = episode / "attempt-00/flight"
        operation = d / "operations/fixture-M1-a0"
        if target == "actor":
            (flight / "actor.jsonl").write_text("")
        else:
            (operation / "call-0001/response.json").unlink()
        # Call the read-only validator directly, excluding cache's earlier hash gate.
        try:
            module.validate_llm_episode(flight, operation, plan, "fixture-M1-a0", 0)
        except (ValueError, OSError, KeyError, TypeError) as error:
            save(name, rejected=True, error=str(error), actual_dispatches=int(counter.read_text()))
        else:
            save(name, rejected=False)
    finally:
        monkey.undo()


(HERE / "probe-results.json").write_text(json.dumps(results, indent=2) + "\n")
