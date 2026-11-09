"""Original unread-70k-input fault against repaired client, CPU only."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
from eval.inspection_model_client import LocalModelClient, ModelOperationUnknown

BASE=Path('/tmp/inspection-llm-integration-repair-review-v1')
FAKE="import json,time;print(json.dumps({'kind':'ready','reservation':'synthetic'}),flush=True);time.sleep(20)"
rows=[]
for mode in ('poll_watchdog','finish_blocked_writer'):
    c=LocalModelClient([sys.executable,'-u','-c',FAKE],BASE/mode,ready_timeout=2,request_watchdog_seconds=.08)
    start=time.monotonic();c.submit([{'role':'user','content':'x'*70000}],1)
    row={'mode':mode,'submit_wall_seconds':time.monotonic()-start,'exception':None}
    try:
        if mode=='poll_watchdog':
            while True:
                c.poll();time.sleep(.003)
        else:c.finish(timeout=.04)
    except ModelOperationUnknown as e:row['exception']=str(e)
    finally:
        row.update(total_wall_seconds=time.monotonic()-start,submitted=c.submitted,
            inflight_retained=c.inflight is not None,child_stopped=c.process.poll() is not None,
            writer_stopped=not c.writer.is_alive())
        if c.process.poll() is None:c.stop_unknown()
    rows.append(row)
(BASE/'pipe-probe.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
