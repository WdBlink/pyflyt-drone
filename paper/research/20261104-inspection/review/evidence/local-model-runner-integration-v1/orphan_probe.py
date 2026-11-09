"""OS-only cleanup check, an isolated process group, no physics/model."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

parent_code = '''
import multiprocessing as mp,os,time
def wait(): time.sleep(20)
p=mp.get_context('fork').Process(target=wait,daemon=True)
p.start()
print(p.pid,flush=True)
time.sleep(20)
'''
p=subprocess.Popen([sys.executable,'-u','-c',parent_code],stdout=subprocess.PIPE,text=True,start_new_session=True)
try:
    child=int(p.stdout.readline())
    pgid=os.getpgid(p.pid)
    assert pgid==p.pid
    p.terminate();p.wait(timeout=2)
    try: os.kill(child,0);alive=True
    except ProcessLookupError: alive=False
    result={'role':'isolated CPU-only analogue of forced mp worker termination',
        'parent_stopped':p.poll() is not None,'daemon_grandchild_survives':alive,
        'isolated_process_group':pgid}
finally:
    try: os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    if p.poll() is None:p.wait(timeout=2)
Path('/tmp/inspection-llm-integration-review-v1/orphan-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
