from pathlib import Path
import os,subprocess,sys,signal,time,json
D=Path('/tmp/inspection-eight-driver-review'); child=D/'classic-fake-child.pid'
script=D/'classic-fake-supervisor.py'
script.write_text('import subprocess,sys,time\nfrom pathlib import Path\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nPath(sys.argv[1]).write_text(str(p.pid))\ntime.sleep(30)\n')
result={}
try:
    try:
        subprocess.run([sys.executable,str(script),str(child)],capture_output=True,text=True,timeout=.15)
    except subprocess.TimeoutExpired:
        result['outer_timeout']=True
    pid=int(child.read_text());result['worker_pid']=pid
    try:os.kill(pid,0);result['detached_worker_survived_outer_timeout']=True
    except ProcessLookupError:result['detached_worker_survived_outer_timeout']=False
finally:
    if child.exists():
        try:os.killpg(int(child.read_text()),signal.SIGKILL)
        except ProcessLookupError:pass
result['cleanup_sent_by_reviewer']=True
(D/'classic-timeout-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
