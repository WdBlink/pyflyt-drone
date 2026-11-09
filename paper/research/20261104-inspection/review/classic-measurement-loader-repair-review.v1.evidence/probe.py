"""Targeted loader repair checks. No source/log writes, flight or generation."""
from pathlib import Path
import hashlib,json,os,shutil
from eval.inspection_measurement_loader import measure_completed_episode
from eval.inspection_episode_receipts import _bindings,canonical_digest,digest,validate_flight
O=Path('/tmp/classic-measurement-review');T=Path('/tmp/classic-measurement-repair-review');B=O/'baseline';plan=json.loads((B/'plan.json').read_text());expected={**plan,'phase':'development'};out={}
def check(name,path,exp=expected):
 try:v,m=measure_completed_episode(path,exp);out[name]={'accepted':True,'Y':v,'retained_attempts':m['retained_attempts']}
 except Exception as e:out[name]={'accepted':False,'error_type':type(e).__name__,'error':str(e)}
for n in ['missing_stdout_rebound','embedded_source_rebound','valid_first_relabelled_failure','relabelled_claimed_clock_failure']:check(n,O/n)
check('unchanged_completed',B)
l1=Path('logs/inspection-classic-dev-flights.v1/dev-L1-00-B0-0');e=json.loads((l1/'plan.json').read_text());e['phase']='development';assert json.loads((l1/'attempt-01/terminal.json').read_text())['status']=='completed';check('actual_valid_retry',l1,e)
def write(path,value):
 tmp=path.with_name(path.name+'.review-tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def bytes_write(path,value):
 tmp=path.with_name(path.name+'.review-tmp');tmp.write_bytes(value);tmp.replace(path)
def make(name):
 p=T/name;shutil.copytree(B,p,copy_function=os.link);shutil.copytree(p/'attempt-00',p/'attempt-01',copy_function=os.link);a=p/'attempt-01';i=json.loads((a/'intent.json').read_text());i['attempt_ordinal']=1;write(a/'intent.json',i);t=json.loads((a/'terminal.json').read_text());t['record_hashes']=_bindings(a);t['stdout_sha256']=t['record_hashes']['stdout.log'];write(a/'terminal.json',t);return p
for kind in ['known_spawn','known_worker_before_world','known_worker_after_audit']:
 p=make(kind);a=p/'attempt-00';bytes_write(a/'stdout.log',b'' if kind=='known_spawn' else b'controlled synthetic known worker failure\n')
 if kind!='known_worker_after_audit':shutil.rmtree(a/'flight')
 else:
  wp=a/'flight/world.json';w=json.loads(wp.read_text());w['status']='operational_failure';w['operational_error']='controlled synthetic cleanup failure';write(wp,w);rp=a/'flight/run.json';r=json.loads(rp.read_text());r['status']='operational_failure';r['world_sha256']=digest(wp);write(rp,r)
 if kind=='known_spawn':
  (a/'started.json').unlink();t={'status':'known_operational_failure','plan_sha256':canonical_digest(plan),'returncode':None,'retry_allowed':True,'spawn_error':'controlled synthetic OSError','not_started':True,'stdout_sha256':digest(a/'stdout.log'),'record_hashes':_bindings(a)}
 else:
  value,evidence=validate_flight(a/'flight',plan,1);assert value is None and evidence['failure_reasons']==['worker_operational_failure'];t={'status':'known_operational_failure','plan_sha256':canonical_digest(plan),'returncode':1,'Y':None,'evidence':evidence,'retry_allowed':True,'stdout_sha256':digest(a/'stdout.log'),'record_hashes':_bindings(a)}
 write(a/'terminal.json',t);check(kind,p)
for n in ['missing_stdout_rebound','embedded_source_rebound','valid_first_relabelled_failure','relabelled_claimed_clock_failure']:assert not out[n]['accepted']
for n in ['unchanged_completed','actual_valid_retry','known_spawn','known_worker_before_world','known_worker_after_audit']:assert out[n]['accepted']
result={'role':'targeted_loader_repair_only_synthetic_failed_attempts_not_new_flights','results':out,'loader_sha':hashlib.sha256(Path('eval/inspection_measurement_loader.py').read_bytes()).hexdigest(),'context_sha':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['eval/inspection_episode_receipts.py','eval/inspection_common_calendar.py','eval/inspection_recording.py']}};(T/'probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
