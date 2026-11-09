"""Independent CPU-only replay of R1-R5; all world/model/owner data synthetic."""
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
spec = importlib.util.spec_from_file_location("repair_fixture", HERE/"source/tests/test_inspection_llm_episode_receipts.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
module = fixtures.module
rows = []

def setup(name, transform=None):
    d = HERE/"probes"/name
    d.mkdir(parents=True, exist_ok=False)
    mp = pytest.MonkeyPatch()
    plan, episode, counter, effects, execute = fixtures.fixture(d, mp)
    if transform is not None:
        script = d/"synthetic-child.py"
        script.write_text(transform(script.read_text()))
        plan["source_hashes"][str(script)] = module.digest(script)
        (d/"plan-input.json").write_text(json.dumps(plan))
    return d, mp, plan, episode, counter, effects, execute

def record(name, **value):
    item = dict(probe=name, role="synthetic CPU-only replay, not real flight/model/native-owner evidence", **value)
    rows.append(item)
    print(json.dumps(item), flush=True)

def terminal(episode):
    return json.loads((episode/"attempt-00/terminal.json").read_text())

def run_blocked(execute, mode):
    try: execute(mode)
    except module.UnresolvedEpisode as error: return str(error)
    raise AssertionError("expected admission to block")

def ps(pid):
    r = subprocess.run(["ps", "-p", str(pid), "-o", "pid=,ppid=,stat=,lstart="], capture_output=True, text=True)
    return r.stdout.strip()

def alive(pid):
    row = ps(pid)
    return bool(row) and not row.split()[2].startswith("Z")

def combined(s):
    old = "(mode in ('wrong-seed','clock-and-seed'))"
    assert old in s
    return s.replace(old, "(mode in ('wrong-seed','clock-and-seed') or mode=='first-clock' and count==1)")

d, mp, plan, episode, counter, effects, execute = setup("R1-clock-plus-seed", combined)
try:
    error = run_blocked(execute, "first-clock")
    t = terminal(episode)
    call = json.loads((d/"operations/fixture-M1-a0/call-0001/request.json").read_text())
    public = json.loads((episode/"attempt-00/flight/actor.jsonl").read_text().splitlines()[0])
    assert t["retry_allowed"] is False and counter.read_text()=="1" and len(effects)==2
    assert call["seed"]==43 and public["sampler_seed"]==42
    second = run_blocked(execute, "ok")
    record("R1-clock-plus-seed", blocked=error, second_call_blocked=second, dispatches=int(counter.read_text()),
           owner_actions=[a for a,_ in effects], terminal_status=t["status"], retry=t["retry_allowed"],
           admission_error=t["admission_error"], physical=t["physical_validation"],
           actual_seed=call["seed"], public_seed=public["sampler_seed"])
finally: mp.undo()

d, mp, plan, episode, counter, effects, execute = setup("R2-start-write", lambda s:"import time\ntime.sleep(30)\n"+s)
real_popen, real_once = module.subprocess.Popen, module.durable_once
children=[]
def capture(*args, **kw):
    p=real_popen(*args,**kw)
    if kw.get("start_new_session") is True: children.append(p)
    return p
def fault(path,value):
    if Path(path).name=="started.json": raise OSError("independent replay started write fault")
    return real_once(path,value)
mp.setattr(module.subprocess,"Popen",capture)
mp.setattr(module,"durable_once",fault)
try:
    try: execute()
    except OSError as error: error_text=str(error)
    else: raise AssertionError("fault did not happen")
    assert len(children)==1 and children[0].poll() is not None
    tree=json.loads((episode/"attempt-00/process-tree.json").read_text())
    second=run_blocked(execute,"ok")
    record("R2-start-write", error=error_text, child_pid=children[0].pid, child_alive_after_raise=children[0].poll() is None,
           child_returncode=children[0].returncode, process_tree=tree, terminal_exists=(episode/"attempt-00/terminal.json").exists(),
           second_call_blocked=second, owner_actions=[a for a,_ in effects])
finally:
    mp.undo()
    for child in children:
        if child.poll() is None: os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=5)

def add_orphan(s,delay):
    return ("import subprocess,sys,time\nfrom pathlib import Path\n"
      "_leak=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
      "Path(sys.argv[-1]+'.orphan').write_text(str(_leak.pid))\n"
      f"time.sleep({delay})\n"+s)

for name,delay in (("R2-normal-fast-orphan",0),("R2-normal-tracked-orphan",.25)):
    d, mp, plan, episode, counter, effects, execute = setup(name,lambda s:add_orphan(s,delay))
    orphan=None
    try:
        try:
            value,receipt=execute()
            error=None
        except (module.UnresolvedEpisode,RuntimeError,OSError,subprocess.SubprocessError) as e:
            value,receipt,error=None,None,str(e)
        orphan=int(Path(str(counter)+'.orphan').read_text())
        live=alive(orphan)
        tree=json.loads((episode/'attempt-00/process-tree.json').read_text())
        t=terminal(episode) if (episode/'attempt-00/terminal.json').exists() else {}
        assert not (value is not None and live), (value,ps(orphan))
        if value is None:
            assert not t.get('retry_allowed',False)
            run_blocked(execute,'ok')
        record(name,returned_y=value,error=error,orphan_pid=orphan,orphan_live_after_return=live,
               orphan_ps=ps(orphan),process_tree=tree,terminal_status=t.get('status'),retry=t.get('retry_allowed'),
               dispatches=int(counter.read_text()),owner_actions=[a for a,_ in effects])
    finally:
        mp.undo()
        if orphan is None and Path(str(counter)+'.orphan').exists():
            orphan=int(Path(str(counter)+'.orphan').read_text())
        if orphan is not None and alive(orphan):
            os.killpg(orphan,signal.SIGKILL)
            time.sleep(.05)
        if orphan is not None: record(name+'-review-cleanup',pid=orphan,ps_after_review_cleanup=ps(orphan))

d, mp, plan, episode, counter, effects, execute=setup('R3-missing-archive')
try:
    execute()
    t=terminal(episode)
    archive=next(Path(p) for p in t['settlement_file_hashes'] if p.endswith('.zip'))
    archive.unlink()
    error=run_blocked(execute,'ok')
    record('R3-missing-archive',blocked=error,dispatches=int(counter.read_text()),owner_actions=[a for a,_ in effects])
finally: mp.undo()

# A bounded residual check: the settlement manifest has no independently
# required keyset. Terminal is not included in its own payload hash map.
d, mp, plan, episode, counter, effects, execute=setup('R3-omitted-archive-binding')
try:
    execute()
    t=terminal(episode)
    archive=next(Path(p) for p in t['settlement_file_hashes'] if p.endswith('.zip'))
    archive.unlink()
    del t['settlement_file_hashes'][str(archive)]
    (episode/'attempt-00/terminal.json').write_text(json.dumps(t))
    try:
        y,r=execute('always-clock')
    except (module.UnresolvedEpisode,ValueError,OSError,KeyError,TypeError) as error:
        record('R3-omitted-archive-binding',blocked=True,error=str(error),dispatches=int(counter.read_text()))
    else:
        record('R3-omitted-archive-binding',blocked=False,returned_y=y,cache_reused=r['cache_reused'],
               archive_exists=archive.exists(),remaining_settlement_keys=list(t['settlement_file_hashes']),
               dispatches=int(counter.read_text()),owner_actions=[a for a,_ in effects])
finally: mp.undo()

def after_deadline(s):
    s=s.replace("text='{\"commands\":[]}'","text='{\"commands\":[{\"name\":\"enter_safe_exit\",\"args\":{\"drone_id\":0,\"expected_version\":0}}]}'")
    s=s.replace("'reply':{'commands':[]}","'reply':json.loads(text)")
    s=s.replace("'commands_enqueued_not_accepted':[],'worker_progress_at_enqueue':after","'commands_enqueued_not_accepted':json.loads(text)['commands'],'worker_progress_at_enqueue':[601.,20,0,80]")
    return s.replace("'actor_mutating_calls':0","'actor_mutating_calls':1")

for name,mode,transform in (('R4-enqueue601','ok',after_deadline),('R5-source-bytes','embedded-source',None)):
    d,mp,plan,episode,counter,effects,execute=setup(name,transform)
    try:
        error=run_blocked(execute,mode)
        t=terminal(episode)
        assert t['retry_allowed'] is False and counter.read_text()=='1' and len(effects)==2
        second=run_blocked(execute,'ok')
        record(name,blocked=error,second_call_blocked=second,terminal_status=t['status'],retry=t['retry_allowed'],
               admission_error=t['admission_error'],dispatches=int(counter.read_text()),owner_actions=[a for a,_ in effects])
    finally:mp.undo()

(HERE/'replay-results.json').write_text(json.dumps(rows,indent=2)+'\n')
