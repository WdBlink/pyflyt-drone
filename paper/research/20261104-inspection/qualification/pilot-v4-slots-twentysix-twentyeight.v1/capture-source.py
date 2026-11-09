"""Capture original v4 orders25–27; use separately frozen classic recount, no flight/SDK/effect execution."""
from collections import Counter
from datetime import datetime, timezone
import hashlib,json
from pathlib import Path
import shutil

project=Path.cwd();run=project/'paper/research/20261104-inspection';supervisor=project/'logs/inspection-eight-condition-pilot.v4.continuation.v1'
flights=project/'logs/inspection-eight-condition-dev-flights.v4';out=run/'qualification/pilot-v4-slots-twentysix-twentyeight.v1'
assert not out.exists()
barrier=json.loads((supervisor/'checkpoint-pauses.jsonl').read_text().splitlines()[-1])
assert barrier['order_index']==28 and (supervisor/'checkpoint.request').exists()
assert not (supervisor/'order-receipts/000028.json').exists()
slots=json.loads((supervisor/'episode-order.json').read_text())['slots'][25:28]
assert [(s['order_index'],s['case_id'],s['condition'],s['repeat_ordinal']) for s in slots]==[
 (25,'dev-L3-01','B4-M2',2),(26,'dev-L3-01','B1',0),(27,'dev-L3-01','B3-M1',0)]
files={};rows=[]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def bind(p):
 value={'sha256':sha(p),'bytes':p.stat().st_size};files[str(p.relative_to(project))]=value
 return value['sha256']

for slot in slots:
 classic=slot["condition"]=="B1"
 base=(flights/"classic" if classic else flights)/f"{slot['case_id']}-{slot['condition']}-{slot['repeat_ordinal']}";attempt=base/'attempt-00'
 terminal=json.loads((attempt/'terminal.json').read_text());assert terminal['status']=='completed' and terminal['returncode']==0
 assert not (base/'attempt-01').exists()
 p=attempt/'flight/world.json';assert p.stat().st_size<512*1024*1024;raw=p.read_bytes();world=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==(terminal['evidence'] if classic else terminal['physical_validation'])['world_sha256']
 assert world['status']=='completed' and world['simulated_seconds']==600 and world['physics_steps_executed']==144000
 assert world['clock_lag_over_0p1_seconds_ticks']==0 and world['max_clock_lag_seconds']<.1
 assert world['physics_runtime']['errors']==[]
 recount=json.loads(Path('/tmp/inspection-v4-order26-classic-frozen-recount.v1.json').read_text()) if classic else terminal['evidence']['frozen_score_recount']
 assert recount['score_equal'] is True and world['score']==recount['score']
 assert recount['world_sha256']==hashlib.sha256(raw).hexdigest()
 assert world['score']['Y']==terminal['Y'] and world['score']['D'] is not None
 for leaf in ['plan.json','attempt-00/terminal.json','attempt-00/flight/run.json','attempt-00/flight/world.json','attempt-00/flight/private-case.json']:
  bind(base/leaf)
 for path,digest in terminal.get('settlement_file_hashes',{}).items():assert bind(Path(path))==digest
 p=supervisor/f"order-receipts/{slot['order_index']:06d}.json";callback=json.loads(p.read_text())
 assert all(callback[k]==slot[k] for k in ['order_index','case_id','condition','repeat_ordinal'])
 bind(p);bind(supervisor/callback['raw'])
 accepted={r['response']['report_id'] for r in world['reports'] if r['response'].get('status')=='accepted'}
 progress=json.loads((attempt/'flight/progress.jsonl').read_text().splitlines()[-1])
 assert progress['t']==600 and progress['reports']==len(accepted)
 assert all(0<=r['t']<=600 for r in world['reports'])
 evidence=terminal['evidence']
 rows.append({**slot,'attempts':1,'terminal_status':terminal['status'],'clock':{k:world[k] for k in [
  'simulated_seconds','physics_steps_executed','max_clock_lag_seconds','clock_lag_p95_seconds','clock_lag_over_0p1_seconds_ticks']},
  'runtime_errors':world['physics_runtime']['errors'],'score':{k:world['score'][k] for k in ['C','S','D','Q','H','E','Y','SR']},
  'report_tool_audit_entries_count':len(world['reports']),
  'accepted_unique_report_ids_count':len({r['response']['report_id'] for r in world['reports'] if r['response'].get('status')=='accepted'}),'world_bytes':len(raw),'usage':None if classic else evidence['model_meter']['usage'],
  'decision_attempts':evidence.get('decision_attempts'),'commands_enqueued':evidence.get('commands_enqueued'),
  'causal_join_decode_status':dict(Counter(j['decode_status'] for j in evidence.get('causal_request_joins',[])))})
 del raw,world

out.mkdir();captures={}
for source,name in [(Path('/tmp/inspection-pilot-v4-continuation.v1.stdout'),'supervisor.at-twentyeight.stdout'),
 (supervisor/'checkpoint-pauses.jsonl','checkpoint-pauses.at-twentyeight.jsonl'),
 (Path('/tmp/inspection-v4-slots26-28-native-status.v1.json'),'native-status.at-twentyeight.json'),
 (Path('/tmp/inspection-v4-order26-classic-frozen-recount.v1.json'),'classic-order26-frozen-recount.json'),
 (Path(__file__),'capture-source.py')]:
 target=out/name;shutil.copyfile(source,target);captures[str(target.relative_to(run))]=sha(target)
status=json.loads((out/'native-status.at-twentyeight.json').read_text());budget=status['budget']
assert status['stale']==[] and budget['reserved']['calls']==0 and budget['reserved']['tokens']==0
assert budget['unknown_usage']==[] and budget['unresolved_usage']==[]
assert budget['known_usage']=={'calls':1099,'tokens':7125459,'cost':{}}
value={'role':'Actual original v4 callbacks26–28 after exact25-cache reconstruction; no formal/G4 comparison',
 'at':datetime.now(timezone.utc).isoformat(),'completed_callbacks':28,'planned_callbacks':108,'complete_physical_root_blocks':1,
 'provided_sdk_final_outer_recording_complete':False,'slots':rows,'native_budget':budget,'barrier':barrier,
 'bound_files':files,'immutable_captures':captures,
 'limits':['One complete L1 root and partial L3 root, not population estimates or actual formal data.',
 'Clock summaries/witness, not complete72000-phase vectors.','Cost/host/Codex/capital/energy remain unmetered.']}
(out/'binding.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
print(json.dumps({'binding':str((out/'binding.json').relative_to(run)),'sha256':sha(out/'binding.json'),
 'new_slots':rows,'native_known_usage':budget['known_usage']},allow_nan=False))
