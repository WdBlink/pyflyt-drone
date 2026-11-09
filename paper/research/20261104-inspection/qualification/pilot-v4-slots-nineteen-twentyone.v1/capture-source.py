"""Capture three closed L3 development slots; no evaluator/SDK/effect execution."""
from collections import Counter
from datetime import datetime, timezone
import hashlib,json
from pathlib import Path
import shutil

project=Path.cwd();run=project/'paper/research/20261104-inspection';supervisor=project/'logs/inspection-eight-condition-pilot.v4'
flights=project/'logs/inspection-eight-condition-dev-flights.v4';out=run/'qualification/pilot-v4-slots-nineteen-twentyone.v1'
assert not out.exists()
barrier=json.loads((supervisor/'checkpoint-pauses.jsonl').read_text().splitlines()[-1])
assert barrier['order_index']==21 and (supervisor/'checkpoint.request').exists()
assert not (supervisor/'order-receipts/000021.json').exists()
slots=json.loads((supervisor/'episode-order.json').read_text())['slots'][18:21]
assert [(s['order_index'],s['case_id'],s['condition'],s['repeat_ordinal']) for s in slots]==[
 (18,'dev-L3-01','B4-M1',0),(19,'dev-L3-01','B4-M2',0),(20,'dev-L3-01','B3-M3',2)]
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
 base=flights/f"{slot['case_id']}-{slot['condition']}-{slot['repeat_ordinal']}";attempt=base/'attempt-00'
 terminal=json.loads((attempt/'terminal.json').read_text());assert terminal['status']=='completed' and terminal['returncode']==0
 assert not (base/'attempt-01').exists()
 p=attempt/'flight/world.json';assert p.stat().st_size<512*1024*1024;raw=p.read_bytes();world=json.loads(raw)
 assert hashlib.sha256(raw).hexdigest()==terminal['physical_validation']['world_sha256']
 assert world['status']=='completed' and world['simulated_seconds']==600 and world['physics_steps_executed']==144000
 assert world['clock_lag_over_0p1_seconds_ticks']==0 and world['max_clock_lag_seconds']<.1
 assert world['physics_runtime']['errors']==[] and world['score']==terminal['evidence']['frozen_score_recount']['score']
 assert world['score']['Y']==terminal['Y'] and world['score']['D'] is not None
 for leaf in ['plan.json','attempt-00/terminal.json','attempt-00/flight/run.json','attempt-00/flight/world.json','attempt-00/flight/private-case.json']:
  bind(base/leaf)
 for path,digest in terminal['settlement_file_hashes'].items():assert bind(Path(path))==digest
 p=supervisor/f"order-receipts/{slot['order_index']:06d}.json";callback=json.loads(p.read_text())
 assert all(callback[k]==slot[k] for k in ['order_index','case_id','condition','repeat_ordinal'])
 bind(p);bind(supervisor/callback['raw'])
 evidence=terminal['evidence']
 rows.append({**slot,'attempts':1,'terminal_status':terminal['status'],'clock':{k:world[k] for k in [
  'simulated_seconds','physics_steps_executed','max_clock_lag_seconds','clock_lag_p95_seconds','clock_lag_over_0p1_seconds_ticks']},
  'runtime_errors':world['physics_runtime']['errors'],'score':{k:world['score'][k] for k in ['C','S','D','Q','H','E','Y','SR']},
  'reports_count':len(world['reports']),'world_bytes':len(raw),'usage':evidence['model_meter']['usage'],
  'decision_attempts':evidence['decision_attempts'],'commands_enqueued':evidence['commands_enqueued'],
  'causal_join_decode_status':dict(Counter(j['decode_status'] for j in evidence['causal_request_joins']))})
 del raw,world

out.mkdir();captures={}
for source,name in [(supervisor.parent/'inspection-eight-condition-pilot.v4.supervisor.stdout','supervisor.at-twentyone.stdout'),
 (supervisor/'checkpoint-pauses.jsonl','checkpoint-pauses.at-twentyone.jsonl'),
 (Path('/tmp/inspection-pilot-v4-at-twentyone-native-status.v1.json'),'native-status.at-twentyone.json'),
 (Path(__file__),'capture-source.py')]:
 target=out/name;shutil.copyfile(source,target);captures[str(target.relative_to(run))]=sha(target)
status=json.loads((out/'native-status.at-twentyone.json').read_text());budget=status['budget']
assert status['stale']==[] and budget['reserved']['calls']==0 and budget['reserved']['tokens']==0
assert budget['unknown_usage']==[] and budget['unresolved_usage']==[]
value={'role':'Actual v4 callbacks19–21, first3 of second development root; no formal/G4 comparison',
 'at':datetime.now(timezone.utc).isoformat(),'completed_callbacks':21,'planned_callbacks':108,'complete_physical_root_blocks':1,
 'provided_sdk_final_outer_recording_complete':False,'slots':rows,'native_budget':budget,'barrier':barrier,
 'bound_files':files,'immutable_captures':captures,
 'limits':['One complete L1 root and partial L3 root, not population estimates or actual formal data.',
 'Clock summaries/witness, not complete72000-phase vectors.','Cost/host/Codex/capital/energy remain unmetered.']}
(out/'binding.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
print(json.dumps({'binding':str((out/'binding.json').relative_to(run)),'sha256':sha(out/'binding.json'),
 'new_slots':rows,'native_known_usage':budget['known_usage']},allow_nan=False))
