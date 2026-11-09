"""Read-only real completed rows; all mutations are isolated /tmp copies."""
from pathlib import Path
import copy,hashlib,json,os,shutil
from eval.inspection_measurement_loader import measure_completed_episode
from eval.inspection_episode_receipts import _bindings,validate_flight
S=Path('logs/inspection-classic-dev-flights.v1/dev-L3-00-B0-0');T=Path('/tmp/classic-measurement-review');B=T/'baseline';terminal=json.loads((S/'attempt-00/terminal.json').read_text());assert terminal['status']=='completed';shutil.copytree(S,B)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected=json.loads((B/'plan.json').read_text());expected['phase']='development';results={}
def write(path,value):
 tmp=path.with_name(path.name+'.review-tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def clone(name):
 p=T/name;shutil.copytree(B,p,copy_function=os.link);return p
def bind(p,ordinal=0,revalidate=True):
 a=p/f'attempt-{ordinal:02d}';t=json.loads((a/'terminal.json').read_text());t['record_hashes']=_bindings(a);t['stdout_sha256']=t['record_hashes']['stdout.log']
 if revalidate and t['status']=='completed':
  value,evidence=validate_flight(a/'flight',json.loads((p/'plan.json').read_text()),t['returncode']);t['Y']=value;t['evidence']=evidence
 write(a/'terminal.json',t)
def run(name,path,exp=None):
 try:v,m=measure_completed_episode(path,expected if exp is None else exp);results[name]={'accepted':True,'Y':v,'retained_attempts':m['retained_attempts'],'source_terminal_sha':sha(path/'attempt-00/terminal.json')}
 except Exception as e:results[name]={'accepted':False,'error_type':type(e).__name__,'error':str(e)}
run('original_completed',B)
# Loss/corruption without rewriting the receipt is always rejected.
for name,file in [('missing_terminal','terminal.json'),('missing_intent','intent.json'),('missing_start','started.json'),('missing_stdout_unrebound','stdout.log')]:
 p=clone(name);(p/'attempt-00'/file).unlink();run(name,p)
p=clone('unknown_status');t=json.loads((p/'attempt-00/terminal.json').read_text());t['status']='unknown';write(p/'attempt-00/terminal.json',t);run('unknown_status',p)
p=clone('operational_only');t=json.loads((p/'attempt-00/terminal.json').read_text());t.update(status='known_operational_failure',retry_allowed=True,Y=None);write(p/'attempt-00/terminal.json',t);run('operational_only',p)
p=clone('appended_unknown');(p/'attempt-01').mkdir();run('appended_unknown',p)
p=clone('noncontiguous_attempt');(p/'attempt-00').rename(p/'attempt-01');run('noncontiguous_attempt',p)
# Distinguish hashes that are checked from mandatory/semantic information.
p=clone('missing_stdout_rebound');(p/'attempt-00/stdout.log').unlink();bind(p);run('missing_stdout_rebound',p)
p=clone('embedded_source_rebound');rp=p/'attempt-00/flight/run.json';r=json.loads(rp.read_text());key=next(iter(r['sources']));r['sources'][key]='modified source payload, hash declaration unchanged';write(rp,r);bind(p);run('embedded_source_rebound',p);results['embedded_source_rebound']['embedded_hash_now_wrong']=hashlib.sha256(r['sources'][key].encode()).hexdigest()!=r['source_hashes'][key]
p=clone('source_claim_changed');rp=p/'attempt-00/flight/run.json';r=json.loads(rp.read_text());r['source_hashes'][next(iter(r['source_hashes']))]='0'*64;write(rp,r);bind(p);run('source_claim_changed',p)
p=clone('case_file_changed');cp=p/'attempt-00/flight/private-case.json';tmp=cp.with_name('case.review-tmp');tmp.write_bytes(cp.read_bytes()+b' ');tmp.replace(cp);bind(p);run('case_file_changed',p)
p=clone('score_changed_rebound');wp=p/'attempt-00/flight/world.json';w=json.loads(wp.read_text());sc=w['score'];sc['S']-=.1;sc['Q']=(sc['C']+sc['S']+sc['D'])/3;sc['Y']=sc['H']*sc['E']*sc['Q'];write(wp,w);rp=p/'attempt-00/flight/run.json';r=json.loads(rp.read_text());r['world_sha256']=sha(wp);write(rp,r);bind(p);run('score_changed_rebound',p)
p=clone('valid_first_relabelled_failure');shutil.copytree(p/'attempt-00',p/'attempt-01',copy_function=os.link);ip=p/'attempt-01/intent.json';i=json.loads(ip.read_text());i['attempt_ordinal']=1;write(ip,i);bind(p,1);tp=p/'attempt-00/terminal.json';t=json.loads(tp.read_text());t.update(status='known_operational_failure',retry_allowed=True);write(tp,t);run('valid_first_relabelled_failure',p)
results['valid_first_relabelled_failure']['first_still_valid_by_flight_validator']=validate_flight(p/'attempt-00/flight',json.loads((p/'plan.json').read_text()),0)[0] is not None
# Caller-frozen role/model gates are separate from operational and score zero.
e=copy.deepcopy(expected);e['condition']='B3-M1';run('LLM_explicit_reject',B,e)
e=copy.deepcopy(expected);e['phase']='main';run('formal_without_g4_reject',B,e)
e=copy.deepcopy(expected);e['case_id']='wrong-frozen-case';run('frozen_expected_mismatch',B,e)
# The actual completed retry row remains accepted; no files in it are changed.
L1=Path('logs/inspection-classic-dev-flights.v1/dev-L1-00-B0-0');assert json.loads((L1/'attempt-01/terminal.json').read_text())['status']=='completed';e=json.loads((L1/'plan.json').read_text());e['phase']='development';run('actual_valid_retry',L1,e)
assert results['original_completed']['accepted'] and results['actual_valid_retry']['accepted']
for name in ['missing_terminal','missing_intent','missing_start','missing_stdout_unrebound','unknown_status','operational_only','appended_unknown','noncontiguous_attempt','source_claim_changed','case_file_changed','score_changed_rebound','LLM_explicit_reject','formal_without_g4_reject','frozen_expected_mismatch']:assert not results[name]['accepted']
assert results['missing_stdout_rebound']['accepted'] and results['embedded_source_rebound']['accepted'] and results['valid_first_relabelled_failure']['accepted']
out={'role':'isolated_loader_audit_not_new_flight_or_measurement','source_episode':str(S),'source_plan_sha':sha(S/'plan.json'),'source_terminal_sha':sha(S/'attempt-00/terminal.json'),'results':results,'source_sha':{n:sha(n) for n in ['eval/inspection_measurement_loader.py','eval/inspection_episode_receipts.py','eval/inspection_common_calendar.py','eval/inspection_recording.py']}}
(T/'probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
