"""Bounded pure-CPU child fault; no model imports, no physics."""
import json
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,'/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
from eval.inspection_model_client import LocalModelClient

ROOT=Path('/tmp/inspection-llm-integration-review-v1')
FAKE="import json,time;print(json.dumps({'kind':'ready','reservation':'synthetic'}),flush=True);time.sleep(20)"
c=LocalModelClient([sys.executable,'-u','-c',FAKE],ROOT/'pipe-block-client',ready_timeout=2)
result={'role':'pure fake ready-but-not-reading service; no model/physics','completed':False}
def send():
    start=time.monotonic()
    try: c.submit([{'role':'user','content':'x'*70000}],1)
    except BaseException as e: result['exception']=type(e).__name__
    finally: result['completed']=True;result['submit_wall_seconds']=time.monotonic()-start
t=threading.Thread(target=send,daemon=True);t.start();t.join(timeout=.15)
result['blocked_after_150ms']=t.is_alive();result['submitted_before_cleanup']=c.submitted
result['inflight_before_cleanup']=c.inflight is not None
c.stop_unknown();t.join(timeout=2)
result['child_stopped']=c.process.poll() is not None
(ROOT/'pipe-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
