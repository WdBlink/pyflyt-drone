"""Additional SYNTHETIC CPU boundary evidence; no MLX/physics/live owner writes."""
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
spec = importlib.util.spec_from_file_location("frozen_fixture_details", HERE / "source/tests/test_inspection_llm_episode_receipts.py")
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
module = f.module
rows = []

def setup(name, change=None):
    d = HERE / "probes" / name
    d.mkdir(parents=True, exist_ok=False)
    mp = pytest.MonkeyPatch()
    plan, episode, counter, effects, execute = f.fixture(d, mp)
    if change:
        script = d / "synthetic-child.py"
        script.write_text(change(script.read_text()))
        plan["source_hashes"][str(script)] = module.digest(script)
        (d / "plan-input.json").write_text(json.dumps(plan))
    return d, mp, plan, episode, counter, effects, execute

def save(name, **kw):
    row = dict(probe=name, role="synthetic-only boundary evidence", **kw)
    rows.append(row)
    print(json.dumps(row), flush=True)

def ps(pid):
    r = subprocess.run(["ps", "-p", str(pid), "-o", "pid=,ppid=,stat=,lstart="], text=True, capture_output=True)
    return r.stdout.strip()

prefix = """import subprocess,sys,time,os\nfrom pathlib import Path\n_timing=Path(sys.argv[-1]+'.timing')\n_leak=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nPath(sys.argv[-1]+'.orphan').write_text(str(_leak.pid))\n_timing.write_text(__import__('json').dumps({'event':'forked','wall_ns':time.time_ns(),'root_pid':os.getpid(),'orphan_pid':_leak.pid})+'\\n')\ntime.sleep(.10)\n"""
suffix = "\nwith _timing.open('a') as _s: _s.write(json.dumps({'event':'root_before_exit','wall_ns':time.time_ns(),'root_pid':os.getpid()})+'\\n')\n"
d, mp, plan, episode, counter, effects, execute = setup("normal-orphan-timing", lambda s: prefix + s + suffix)
orphan = None
try:
    y, receipt = execute()
    returned = time.time_ns()
    orphan = int(Path(str(counter)+".orphan").read_text())
    save("normal-orphan-timing", returned_y=y, returned_wall_ns=returned, orphan_pid=orphan,
         orphan_ps_after_return=ps(orphan), events=[json.loads(x) for x in Path(str(counter)+".timing").read_text().splitlines()],
         terminal_sha256=module.digest(episode / "attempt-00/terminal.json"))
finally:
    mp.undo()
    if orphan is not None:
        try: os.killpg(orphan, signal.SIGKILL)
        except ProcessLookupError: pass
        time.sleep(.05)
        save("normal-orphan-cleanup", pid=orphan, ps_after_kill=ps(orphan))

d, mp, plan, episode, counter, effects, execute = setup("terminal-write-after-known-settle")
real_once = module.durable_once
def fail_terminal(path, value):
    if Path(path).name == "terminal.json": raise OSError("synthetic terminal write fault after known settle")
    return real_once(path, value)
mp.setattr(module, "durable_once", fail_terminal)
try:
    try: execute()
    except OSError: pass
    else: raise AssertionError("missing fault")
    try: execute()
    except module.UnresolvedEpisode as error: second_error = str(error)
    else: raise AssertionError("unexpected replay")
    save("terminal-write-after-known-settle", terminal_exists=(episode/"attempt-00/terminal.json").exists(),
         owner_actions=[a for a,_ in effects], settle_status=effects[-1][1]["status"],
         settled_usage=effects[-1][1]["usage"], dispatches=int(counter.read_text()), retry_error=second_error)
finally: mp.undo()

d, mp, plan, episode, counter, effects, execute = setup("inconsistent-embedded-source", lambda s: s.replace(
    "put(flight/'run.json',run)", "run['sources']={name:'mismatched embedded source' for name in p['source_hashes']}\nput(flight/'run.json',run)"))
try:
    y, receipt = execute()
    run = json.loads((episode/"attempt-00/flight/run.json").read_text())
    save("inconsistent-embedded-source", returned_y=y, terminal_status=json.loads((episode/"attempt-00/terminal.json").read_text())["status"],
         embedded_source_hashes_match=all(__import__('hashlib').sha256(value.encode()).hexdigest()==run['source_hashes'][name] for name,value in run['sources'].items()))
finally: mp.undo()

# The default owner adapter's real subprocess argv/transport, against a fake CLI.
d = HERE/"probes/owner-action-transport"
cli = d/"owner/runtime/autoresearch-state-reader-v1/scripts/research.py"
cli.parent.mkdir(parents=True, exist_ok=False)
cli.write_text("import json,sys\nfrom pathlib import Path\np=json.loads(Path(sys.argv[sys.argv.index('--packet')+1]).read_text())\nprint(json.dumps({'status':'refused' if p.get('fail') else 'synthetic-only','argv':sys.argv[1:]}))\nraise SystemExit(2 if p.get('fail') else 0)\n")
packet = d/"packet.json"
packet.write_text('{"usage":{"calls":1,"tokens":4}}')
value = module.owner_action(d/"owner", "budget-reserve", packet, "fixture-transport", d/"reserve.log")
save("owner-action-transport", argv=value['argv'], log_exists=(d/"reserve.log").is_file(), live_owner_touched=False)
packet.write_text('{"fail":true}')
try: module.owner_action(d/"owner", "budget-settle", packet, "fixture-refusal", d/"refused.log", "fixture-transport")
except module.UnresolvedEpisode as error: save("owner-action-refusal", rejected=True, error=str(error), transport_kept=(d/"refused.log").is_file())
else: raise AssertionError("refusal ignored")

# Native SDK budget functions over an in-memory synthetic state: no .paper I/O.
root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(root/"paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1"))
from lib import budget
from datetime import datetime, timezone
state = {'brief':{'budget':{}},'created_at':datetime.now(timezone.utc).isoformat(),
    'usage_ledger':{'synthetic-operation':{'status':'unknown','reserved':{'calls':32,'tokens':180000},'reported_usage':{}}}}
summary = budget.summary(state)
try: budget.admit(state, {'usage':{'calls':1,'tokens':4},'purpose':'synthetic CPU budget semantics'})
except ValueError as error: blocked = str(error)
else: raise AssertionError("unknown usage allowed another operation")
save("native-budget-unknown-semantics", known=summary['known_usage'], held=summary['reserved'],
     unknown=summary['unknown_usage'], new_operation_refused=blocked,
     budget_source_sha256=module.digest(Path(budget.__file__)), state_was_in_memory_only=True)

(HERE/"boundary-results.json").write_text(json.dumps(rows,indent=2)+"\n")
