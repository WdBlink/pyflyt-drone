"""Light independent receipt repair probes; no flight or API execution."""
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(REPO))
from eval import inspection_episode_receipts as module


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def main():
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=False)
    paths=['eval/inspection_episode_receipts.py','tests/test_inspection_episode_receipts.py',
           'eval/run_inspection_dev.py','eval/run_inspection_reference.py']
    frozen={p:digest(REPO/p) for p in paths}
    write(out/'source-snapshot.json',{p:(REPO/p).read_text() for p in paths})
    fixture=runpy.run_path(str(REPO/'tests/test_inspection_episode_receipts.py'))
    results={}
    def setup(name):
        base=out/name;base.mkdir();plan,command,counter=fixture['inputs'](base)
        return base,plan,command,counter
    for change in ('missing-start','intent','stdout','terminal-y','not-started'):
        base,plan,command,counter=setup('cache-'+change);episode=base/'episode'
        module.execute_local_episode(episode,plan,command('ok'));attempt=episode/'attempt-00'
        if change=='missing-start':(attempt/'started.json').unlink()
        elif change=='stdout':(attempt/'stdout.log').write_text('changed\n')
        else:
            p=attempt/('intent.json' if change=='intent' else 'terminal.json');obj=json.loads(p.read_text())
            if change=='intent':obj['argv']=['another-command']
            if change=='terminal-y':obj['Y']=1.
            if change=='not-started':obj['not_started']=True
            p.write_text(json.dumps(obj))
        try:module.execute_local_episode(episode,plan,command('fail'))
        except module.UnresolvedEpisode as e:
            results['cache-'+change]=dict(blocked=True,message=str(e),worker_calls=int(counter.read_text()))
        else:raise AssertionError('corrupt cache admitted: '+change)

    original_once,original_popen=module.durable_once,module.subprocess.Popen
    base,plan,command,counter=setup('start-enospc');spawned=[]
    def capture(*args,**kwargs):
        p=original_popen(*args,**kwargs);spawned.append(p);return p
    def fail_start(path,value):
        if Path(path).name=='started.json':raise OSError(errno.ENOSPC,'injected start ENOSPC')
        return original_once(path,value)
    module.durable_once=fail_start;module.subprocess.Popen=capture
    try:
        try:module.execute_local_episode(base/'episode',plan,lambda out:[sys.executable,'-c','import time; time.sleep(30)'])
        except OSError as e:
            results['start-enospc']=dict(message=str(e),child_reaped_before_return=spawned[0].poll() is not None)
    finally:
        module.durable_once=original_once;module.subprocess.Popen=original_popen
        for p in spawned:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
            p.wait(timeout=2)
    try:module.execute_local_episode(base/'episode',plan,command('ok'))
    except module.UnresolvedEpisode as e:results['start-enospc']['unknown_blocks']='unknown attempt' in str(e)
    assert results['start-enospc']['child_reaped_before_return']

    base,plan,command,counter=setup('orphan-group')
    child_pid=base/'descendant.pid';survived=base/'survived'
    extra=("\nimport subprocess\n"
           "subprocess.Popen([sys.executable,'-c',\"import os,pathlib,time; pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(.4); pathlib.Path(%r).write_text('survived'); time.sleep(30)\"])\n"
           "import time\n"
           "end=time.monotonic()+2\n"
           "while not Path(%r).exists() and time.monotonic()<end:time.sleep(.01)\n") % (str(child_pid),str(survived),str(child_pid))
    (base/'fixture.py').write_text(fixture['SCRIPT']+extra)
    value,evidence=module.execute_local_episode(base/'episode',plan,command('ok'))
    pid=int(child_pid.read_text());time.sleep(.45)
    state=subprocess.run(['ps','-o','stat=','-p',str(pid)],capture_output=True,text=True).stdout.strip()
    alive=bool(state) and not state.startswith('Z')
    if alive:
        os.kill(pid,signal.SIGKILL)
    assert not alive and not survived.exists()
    results['orphan-group']=dict(Y=value,descendant_state=state or 'absent',
                                 descendant_running_after_cleanup=alive,survival_marker_exists=survived.exists())

    for mode in ('bad-source-missing-case','missing-run-wrong-case'):
        base,plan,command,counter=setup(mode)
        if mode=='missing-run-wrong-case':
            (base/'fixture.py').write_text(fixture['SCRIPT']+"\n(out/'run.json').unlink()\n(out/'private-case.json').write_bytes(b'wrong case bytes')\n")
        try:module.execute_local_episode(base/'episode',plan,command(mode if mode=='bad-source-missing-case' else 'ok'))
        except module.UnresolvedEpisode as e:
            terminals=[json.loads(p.read_text()) for p in sorted((base/'episode').glob('attempt-*/terminal.json'))]
            results[mode]=dict(message=str(e),worker_calls=int(counter.read_text()),
                reasons=[t['evidence']['failure_reasons'] for t in terminals],retry_allowed=[t['retry_allowed'] for t in terminals])
    assert results['bad-source-missing-case']['worker_calls']==1

    base,plan,command,counter=setup('directory-sync')
    episode=base/'new-parent'/'episode';events=[]
    original_sync,original_popen=module._sync_directory,module.subprocess.Popen
    def sync(path):
        events.append(dict(kind='sync',path=str(Path(path).relative_to(base)),attempt_exists=(episode/'attempt-00').exists()))
        return original_sync(path)
    def at_spawn(*args,**kwargs):events.append(dict(kind='spawn'));return original_popen(*args,**kwargs)
    module._sync_directory=sync;module.subprocess.Popen=at_spawn
    try:module.execute_local_episode(episode,plan,command('ok'))
    finally:module._sync_directory=original_sync;module.subprocess.Popen=original_popen
    before_spawn=events[:next(i for i,x in enumerate(events) if x['kind']=='spawn')]
    assert any(x.get('path')=='.' for x in before_spawn)
    assert any(x.get('path')=='new-parent' for x in before_spawn)
    assert any(x.get('path')=='new-parent/episode' and x['attempt_exists'] for x in before_spawn)
    results['directory-sync']=dict(new_root_and_attempt_parents_synced_before_spawn=True,events=events)

    for fault in ('case-read-sha','case-end','thread-env'):
        base,plan,command,counter=setup(fault)
        if fault=='case-read-sha':
            script=fixture['SCRIPT'].replace("'private_case_file_sha256':plan['case_file_sha256']", "'private_case_file_sha256':'0'*64")
        elif fault=='case-end':script=fixture['SCRIPT'].replace("'private_case_unchanged':True","'private_case_unchanged':False")
        else:script=fixture['SCRIPT'].replace("'runtime_thread_environment':plan['thread_environment']","'runtime_thread_environment':{}")
        (base/'fixture.py').write_text(script)
        try:module.execute_local_episode(base/'episode',plan,command('ok'))
        except module.UnresolvedEpisode as e:
            terminal=json.loads((base/'episode/attempt-00/terminal.json').read_text())
            results[fault]=dict(blocked=True,worker_calls=int(counter.read_text()),retry_allowed=terminal['retry_allowed'],reasons=terminal['evidence']['failure_reasons'])
        else:raise AssertionError('source/input condition fault admitted')

    # A normal process exit remains known, while interrupted terminal writing
    # must preserve unknown and may not be replayed.
    base,plan,command,counter=setup('terminal-enospc')
    def fail_terminal(path,value):
        if Path(path).name=='terminal.json':raise OSError(errno.ENOSPC,'injected terminal ENOSPC')
        return original_once(path,value)
    module.durable_once=fail_terminal
    try:
        try:module.execute_local_episode(base/'episode',plan,command('ok'))
        except OSError:pass
    finally:module.durable_once=original_once
    try:module.execute_local_episode(base/'episode',plan,command('ok'))
    except module.UnresolvedEpisode as e:
        results['terminal-enospc']=dict(unknown_blocks='unknown attempt' in str(e),worker_calls=int(counter.read_text()))

    assert frozen=={p:digest(REPO/p) for p in paths},'reviewed source changed during probes'
    report=dict(role='independent_synthetic_IO_subprocess_repair_review_not_flight_evidence',
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=frozen,sources_unchanged=True,
        script_sha256=digest(Path(__file__)),results=results)
    write(out/'report.json',report);print(json.dumps(results,indent=2))


if __name__=='__main__':main()
