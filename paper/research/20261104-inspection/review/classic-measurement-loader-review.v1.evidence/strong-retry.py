"""Prior retry evidence is not trusted merely because it claims clock failure."""
from pathlib import Path
import json,os,shutil,hashlib
from eval.inspection_measurement_loader import measure_completed_episode
from eval.inspection_episode_receipts import _bindings,validate_flight
T=Path('/tmp/classic-measurement-review');B=T/'baseline';p=T/'relabelled_claimed_clock_failure';shutil.copytree(B,p,copy_function=os.link)
def write(path,value):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
shutil.copytree(p/'attempt-00',p/'attempt-01',copy_function=os.link);plan=json.loads((p/'plan.json').read_text());expected={**plan,'phase':'development'}
a=p/'attempt-01';intent=json.loads((a/'intent.json').read_text());intent['attempt_ordinal']=1;write(a/'intent.json',intent);t=json.loads((a/'terminal.json').read_text());t['record_hashes']=_bindings(a);t['stdout_sha256']=t['record_hashes']['stdout.log'];value,evidence=validate_flight(a/'flight',plan,t['returncode']);t['Y'],t['evidence']=value,evidence;write(a/'terminal.json',t)
a=p/'attempt-00';t=json.loads((a/'terminal.json').read_text());t.update(status='known_operational_failure',retry_allowed=True,Y=None);t['evidence']['failure_reasons']=['clock_lag_exceeded'];t['evidence']['max_clock_lag_seconds']=.2;write(a/'terminal.json',t)
first,first_evidence=validate_flight(a/'flight',plan,0);value,meta=measure_completed_episode(p,expected);assert first is not None and not first_evidence['failure_reasons']
out={'accepted':True,'Y':value,'retained_attempts':meta['retained_attempts'],'claimed_first_Y':None,'claimed_first_reason':'clock_lag_exceeded','actual_first_validation_reasons':first_evidence['failure_reasons'],'actual_first_Y':first,'loader_sha':hashlib.sha256(Path('eval/inspection_measurement_loader.py').read_bytes()).hexdigest()};(T/'strong-retry.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
