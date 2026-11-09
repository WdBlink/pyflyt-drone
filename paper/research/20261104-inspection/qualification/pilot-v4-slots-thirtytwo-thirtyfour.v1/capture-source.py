"""Capture three closed L3 development slots; no evaluator/SDK/effect execution."""
from collections import Counter
from datetime import datetime, timezone
import hashlib,json
from pathlib import Path
import shutil

project=Path.cwd();run=project/'paper/research/20261104-inspection';supervisor=project/'logs/inspection-eight-condition-pilot.v4.continuation.v2'
flights=project/'logs/inspection-eight-condition-dev-flights.v4';out=run/'qualification/pilot-v4-slots-thirtytwo-thirtyfour.v1'
assert not out.exists()
barrier=json.loads((supervisor/'checkpoint-pauses.jsonl').read_text().splitlines()[-1])
assert barrier['order_index']==34 and (supervisor/'checkpoint.request').exists()
assert not (supervisor/'order-receipts/000034.json').exists()
slots=json.loads((supervisor/'episode-order.json').read_text())['slots'][31:34]
assert [(s['order_index'],s['case_id'],s['condition'],s['repeat_ordinal']) for s in slots]==[
 (31,'dev-L3-01','B3-M1',1),(32,'dev-L3-01','B3-M2',1),(33,'dev-L3-01','B3-M2',2)]
files={};rows=[];observation_paths=[]

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
 sdk_raw=json.loads((supervisor/callback['raw']).read_text())
 assert sdk_raw['status']=='completed' and sdk_raw['value']==terminal['Y'] and sdk_raw['evidence']['cache_reused'] is False
 observations=sdk_raw['evidence']['passive_close_observations'];assert len(observations)==1
 observation=observations[0];sidecar=project/observation['path']
 assert sidecar.parent==supervisor/'cleanup-observations' and not sidecar.is_symlink() and sidecar.stat().st_nlink==1
 assert bind(sidecar)==observation['sha256'];observation_paths.append(sidecar)
 observed=json.loads(sidecar.read_text());original_tree=json.loads((attempt/'process-tree.json').read_text());bind(attempt/'process-tree.json')
 assert observed['original_result']==original_tree and original_tree['status']=='closed'
 assert original_tree['remaining_owned']==[] and original_tree['unclaimed_new_python_orphans']==[]
 assert observed['original_exception'] is None and observed['observation_errors']==[]
 assert observed['adapter_source_sha256']=='83ffb02d0582c243e33421ba0f831d754010c1cb99c70b3022b5a743d96c646c'
 assert observed['context']=={'case_id':slot['case_id'],'condition':slot['condition'],'repeat_ordinal':slot['repeat_ordinal'],
  'plan_sha256':terminal['plan_sha256'],'reservation_prefix':f"dev-all8-pilot-v4-{slot['order_index']:03d}"}
 scans=observed['global_scans'];assert [q['ordinal'] for q in scans]==list(range(len(scans)))
 assert all(q['monotonic_start_ns']<=q['monotonic_end_ns'] for q in scans)
 assert all(a['monotonic_end_ns']<=b['monotonic_start_ns'] for a,b in zip(scans,scans[1:]))
 assert 0<=observation['write_wall_ns']<=observation['post_close_metadata_wall_ns']
 passive_summary={'original_status':original_tree['status'],'captured_global_scans':len(scans),
  'write_wall_ns':observation['write_wall_ns'],'post_close_metadata_wall_ns':observation['post_close_metadata_wall_ns'],
  'observed_close_wall_ns':observed['close_monotonic_end_ns']-observed['close_monotonic_start_ns'],
  'ref':observation,'scope':'existing close global scans only; no complete hot history or net instrument tax'}

 accepted={r['response']['report_id'] for r in world['reports'] if r['response'].get('status')=='accepted'}
 progress=json.loads((attempt/'flight/progress.jsonl').read_text().splitlines()[-1])
 assert progress['t']==600 and progress['reports']==len(accepted)
 assert all(0<=r['t']<=600 for r in world['reports'])
 evidence=terminal['evidence']
 rows.append({**slot,'attempts':1,'terminal_status':terminal['status'],'clock':{k:world[k] for k in [
  'simulated_seconds','physics_steps_executed','max_clock_lag_seconds','clock_lag_p95_seconds','clock_lag_over_0p1_seconds_ticks']},
  'runtime_errors':world['physics_runtime']['errors'],'score':{k:world['score'][k] for k in ['C','S','D','Q','H','E','Y','SR']},
  'report_tool_audit_entries_count':len(world['reports']),
  'accepted_unique_report_ids_count':len({r['response']['report_id'] for r in world['reports'] if r['response'].get('status')=='accepted'}),'world_bytes':len(raw),'usage':evidence['model_meter']['usage'],
  'passive_close':passive_summary,'decision_attempts':evidence['decision_attempts'],'commands_enqueued':evidence['commands_enqueued'],
  'causal_join_decode_status':dict(Counter(j['decode_status'] for j in evidence['causal_request_joins']))})
 del raw,world

out.mkdir();captures={}
for source,name in [(Path('/tmp/inspection-pilot-v4-continuation.v2.stdout'),'supervisor.at-thirtyfour.stdout'),
 (supervisor/'checkpoint-pauses.jsonl','checkpoint-pauses.at-thirtyfour.jsonl'),
 (Path('/tmp/inspection-v4-slots32-34-native-status.v1.json'),'native-status.at-thirtyfour.json'),
 (Path(__file__),'capture-source.py')]:
 target=out/name;shutil.copyfile(source,target);captures[str(target.relative_to(run))]=sha(target)
for source in observation_paths:
 target=out/'passive-close'/source.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
 captures[str(target.relative_to(run))]=sha(target)
status=json.loads((out/'native-status.at-thirtyfour.json').read_text());budget=status['budget']
assert status['stale']==[] and budget['reserved']['calls']==0 and budget['reserved']['tokens']==0
assert budget['unknown_usage']==[] and budget['unresolved_usage']==[]
assert budget['known_usage']=={'calls':1240,'tokens':7976190,'cost':{}}
value={'role':'Actual original v4 callbacks32–34 and first3 passive close sidecars; no formal/G4 comparison',
 'at':datetime.now(timezone.utc).isoformat(),'completed_callbacks':34,'planned_callbacks':108,'complete_physical_root_blocks':1,
 'provided_sdk_final_outer_recording_complete':False,'slots':rows,'native_budget':budget,'barrier':barrier,
 'bound_files':files,'immutable_captures':captures,
 'limits':['One complete L1 root and partial L3 root, not population estimates or actual formal data.',
 'Clock summaries/witness, not complete72000-phase vectors.','Cost/host/Codex/capital/energy remain unmetered.']}
(out/'binding.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
print(json.dumps({'binding':str((out/'binding.json').relative_to(run)),'sha256':sha(out/'binding.json'),
 'new_slots':rows,'native_known_usage':budget['known_usage']},allow_nan=False))
