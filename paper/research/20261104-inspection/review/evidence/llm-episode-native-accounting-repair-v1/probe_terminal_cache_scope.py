"""Synthetic-only cache semantic cross-check; no MLX/physics/native owner."""
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
sys.path.insert(0,str(HERE/'source'))
spec=importlib.util.spec_from_file_location('terminal_scope_fixture',HERE/'source/tests/test_inspection_llm_episode_receipts.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f);m=f.module
d=HERE/'probes/R2-terminal-status-rewrite';d.mkdir(parents=True,exist_ok=False)
mp=pytest.MonkeyPatch();plan,episode,counter,effects,execute=f.fixture(d,mp)
child=d/'synthetic-child.py'
prefix="import subprocess,sys\nfrom pathlib import Path\n_leak=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nPath(sys.argv[-1]+'.orphan').write_text(str(_leak.pid))\n"
child.write_text(prefix+child.read_text());plan['source_hashes'][str(child)]=m.digest(child);(d/'plan-input.json').write_text(json.dumps(plan))
result={'role':'synthetic terminal mutation counterexample; not spontaneous transition or real LLM flight'}
orphan=None
def ps(pid):
 return subprocess.run(['ps','-p',str(pid),'-o','pid=,ppid=,stat=,lstart='],text=True,capture_output=True).stdout.strip()
try:
 try:
  value,_=execute()
  result['initial_returned_y']=value
 except m.UnresolvedEpisode as error:
  result['initial_error']=str(error)
 orphan=int(Path(str(counter)+'.orphan').read_text())
 terminal=episode/'attempt-00/terminal.json';t=json.loads(terminal.read_text());tree=json.loads((episode/'attempt-00/process-tree.json').read_text())
 result.update(initial_terminal_status=t['status'],original_y=t['Y'],tree_status=tree['status'],orphan_pid=orphan,orphan_ps_before_cache=ps(orphan))
 if t['status']=='unknown_process_cleanup':
  # No payload hash, evidence, source, operation, archive or settle file changes.
  t['status']='completed';t['Y']=0.0;terminal.write_text(json.dumps(t))
  try:
   value,receipt=execute('always-clock')
  except (m.UnresolvedEpisode,ValueError,OSError,KeyError,TypeError) as error:
   result.update(cache_blocked=True,error=str(error))
  else:
   result.update(cache_blocked=False,returned_y=value,cache_reused=receipt['cache_reused'],orphan_ps_after_cache=ps(orphan))
 else:
  result['rewrite_not_attempted']='original real monitor captured and closed this child'
 result.update(dispatches=int(counter.read_text()),owner_actions=[a for a,_ in effects])
finally:
 mp.undo()
 if orphan is None and Path(str(counter)+'.orphan').exists():orphan=int(Path(str(counter)+'.orphan').read_text())
 if orphan is not None and ps(orphan):
  try:os.killpg(orphan,signal.SIGKILL)
  except ProcessLookupError:pass
  time.sleep(.05)
 result['orphan_ps_after_review_cleanup']=ps(orphan) if orphan else None
(HERE/'terminal-cache-scope-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
