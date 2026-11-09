"""Bounded 37a40 cache guard replay: synthetic CPU only, no model/physics/owner."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source-final'))
spec=importlib.util.spec_from_file_location('cache_guard_fixture',HERE/'source-final/tests/test_inspection_llm_episode_receipts.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f);module=f.module
rows=[]
def setup(name,transform=None):
 d=HERE/'probes-final'/name;d.mkdir(parents=True,exist_ok=False);mp=pytest.MonkeyPatch()
 plan,episode,counter,effects,execute=f.fixture(d,mp)
 if transform:
  p=d/'synthetic-child.py';p.write_text(transform(p.read_text()));plan['source_hashes'][str(p)]=module.digest(p);(d/'plan-input.json').write_text(json.dumps(plan))
 return d,mp,plan,episode,counter,effects,execute
def record(name,**data):
 row={'probe':name,'role':'synthetic-only narrow cache guard',**data};rows.append(row);print(json.dumps(row),flush=True)
def term(ep):return json.loads((ep/'attempt-00/terminal.json').read_text())
def denied(execute):
 try:execute('always-clock')
 except (module.UnresolvedEpisode,ValueError,OSError,KeyError,TypeError) as error:return str(error)
 raise AssertionError('cache fault accepted')
def ps(pid):return subprocess.run(['ps','-p',str(pid),'-o','pid=,ppid=,stat=,lstart='],capture_output=True,text=True).stdout.strip()

for name,kind in (('normal-cache','normal'),('K1-archive-binding-omitted','archive'),('required-intent-omitted','intent'),('required-owner-log-omitted','owner')):
 d,mp,plan,ep,counter,effects,execute=setup(name)
 try:
  y,first=execute();assert y==0 and not first['cache_reused'];t=term(ep)
  if kind=='normal':
   value,cached=execute('always-clock');assert value==0 and cached['cache_reused']
   record(name,returned_y=value,cache_reused=True,dispatches=int(counter.read_text()),effects=[a for a,_ in effects])
  else:
   if kind=='archive':
    p=next(Path(p) for p in t['settlement_file_hashes'] if p.endswith('.zip'));p.unlink();del t['settlement_file_hashes'][str(p)]
   else:
    key='intent.json' if kind=='intent' else 'reserve-owner.json';(ep/'attempt-00'/key).unlink();del t['payload_hashes'][key]
   (ep/'attempt-00/terminal.json').write_text(json.dumps(t));error=denied(execute)
   record(name,blocked=True,error=error,dispatches=int(counter.read_text()),effects=[a for a,_ in effects])
  assert counter.read_text()=='1' and len(effects)==2
 finally:mp.undo()

prefix="import subprocess,sys\nfrom pathlib import Path\n_leak=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nPath(sys.argv[-1]+'.orphan').write_text(str(_leak.pid))\n"
d,mp,plan,ep,counter,effects,execute=setup('K2-unknown-tree-terminal-rewrite',lambda s:prefix+s)
orphan=None
try:
 try:execute()
 except module.UnresolvedEpisode as error:first_error=str(error)
 else:raise AssertionError('expected same real fast orphan to be unknown')
 orphan=int(Path(str(counter)+'.orphan').read_text());t=term(ep);tree=json.loads((ep/'attempt-00/process-tree.json').read_text())
 assert t['status']=='unknown_process_cleanup' and t['Y'] is None and tree['status']=='unknown'
 before=ps(orphan);t['status']='completed';t['Y']=0.;(ep/'attempt-00/terminal.json').write_text(json.dumps(t))
 error=denied(execute)
 record('K2-unknown-tree-terminal-rewrite',initial_error=first_error,tree_status=tree['status'],orphan_pid=orphan,orphan_ps_during_probe=before,
        blocked=True,error=error,dispatches=int(counter.read_text()),effects=[a for a,_ in effects])
 assert counter.read_text()=='1' and len(effects)==2
finally:
 mp.undo()
 if orphan is None and Path(str(counter)+'.orphan').exists():orphan=int(Path(str(counter)+'.orphan').read_text())
 if orphan is not None and ps(orphan):
  try:os.killpg(orphan,signal.SIGKILL)
  except ProcessLookupError:pass
  time.sleep(.05)
 record('review-cleanup',pid=orphan,ps_after_cleanup=ps(orphan) if orphan else None)

(HERE/'probe-results-final.json').write_text(json.dumps(rows,indent=2)+'\n')
