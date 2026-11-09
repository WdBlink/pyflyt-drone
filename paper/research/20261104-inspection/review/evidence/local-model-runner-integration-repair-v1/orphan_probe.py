"""Original daemon-descendant fault, now invoking production cleanup only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
sys.path.insert(0,'/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
from eval.inspection_process_cleanup import stop_worker_tree, _inventory

parent_code='''
import multiprocessing as mp,time
def wait():time.sleep(20)
p=mp.get_context('fork').Process(target=wait,daemon=True)
p.start();print(p.pid,flush=True);time.sleep(20)
'''
p=subprocess.Popen([sys.executable,'-u','-c',parent_code],stdout=subprocess.PIPE,text=True,start_new_session=True)
class Adapter:
    pid=p.pid
    def is_alive(self):return p.poll() is None
    def terminate(self):p.terminate()
    def kill(self):p.kill()
    def join(self,timeout=None):
        try:p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:pass
try:
    child=int(p.stdout.readline());assert os.getpgid(p.pid)==p.pid
    result=stop_worker_tree(Adapter())
    remaining=_inventory().get(child)
    result.update(role='isolated OS-only owned-parent adapter; no physics/model',
        expected_daemon=child,daemon_recorded=child in result['descendants'],
        daemon_stopped=remaining is None or remaining['status'].startswith('Z'),parent_stopped=p.poll() is not None)
finally:
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    if p.poll() is None:p.wait(timeout=2)
Path('/tmp/inspection-llm-integration-repair-review-v1/orphan-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
