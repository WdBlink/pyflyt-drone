"""Closed development evidence capture only; no scoring, SDK writes or effects."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

project=Path.cwd()
run=project/'paper/research/20261104-inspection'
supervisor=project/'logs/inspection-eight-condition-pilot.v4'
flights=project/'logs/inspection-eight-condition-dev-flights.v4'
out=run/'qualification/pilot-v4-first-root.v1'
assert not out.exists()
pauses=(supervisor/'checkpoint-pauses.jsonl').read_text().splitlines()
barrier=json.loads(pauses[-1])
assert barrier['order_index']==18 and (supervisor/'checkpoint.request').is_file()
assert not (supervisor/'order-receipts/000018.json').exists()
schedule=json.loads((supervisor/'episode-order.json').read_text())
slots=schedule['slots'][:18]
assert len(slots)==18 and {s['case_id'] for s in slots}=={'dev-L1-01'}
expected={'B0':1,'B1':1,'B2':1,'B3-M1':3,'B3-M2':3,'B3-M3':3,'B4-M1':3,'B4-M2':3}
assert Counter(s['condition'] for s in slots)==expected
out.mkdir()
files={}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def bind(path):
 key=str(path.relative_to(project))
 files[key]={'sha256':sha(path),'bytes':path.stat().st_size}
 return files[key]['sha256']

records=[]
for s in slots:
 i=s['order_index']; condition=s['condition']; name=f"{s['case_id']}-{condition}-{s['repeat_ordinal']}"
 base=flights/('classic' if condition in {'B0','B1','B2'} else '')/name
 attempt=base/'attempt-00'; terminal=json.loads((attempt/'terminal.json').read_text())
 assert terminal['status']=='completed' and terminal['returncode']==0
 assert not (base/'attempt-01').exists()
 world_path=attempt/'flight/world.json'; raw_world=world_path.read_bytes(); world=json.loads(raw_world)
 assert world['status']=='completed' and world['simulated_seconds']==600 and world['physics_steps_executed']==144000
 assert world['clock_lag_over_0p1_seconds_ticks']==0 and world['max_clock_lag_seconds']<=.1
 assert world['physics_runtime']['errors']==[]
 assert world['score']['Y']==terminal['Y']
 evidence=terminal['evidence']; physical=terminal.get('physical_validation',evidence)
 for leaf in ['plan.json','attempt-00/terminal.json','attempt-00/flight/run.json','attempt-00/flight/world.json','attempt-00/flight/private-case.json']:
  bind(base/leaf)
 if 'world_sha256' in physical:assert files[str(world_path.relative_to(project))]['sha256']==physical['world_sha256']
 receipt=supervisor/f'order-receipts/{i:06d}.json'; r=json.loads(receipt.read_text())
 assert all(r[k]==s[k] for k in ['order_index','case_id','condition','repeat_ordinal'])
 bind(receipt); raw_path=supervisor/r['raw']; bind(raw_path)
 score={k:world['score'][k] for k in ['C','S','D','Q','H','E','Y','SR']}
 assert score['D'] is None
 row={**s,'name':str(base.relative_to(flights)),'attempts':1,'terminal_status':terminal['status'],
      'clock':{k:world[k] for k in ['simulated_seconds','physics_steps_executed','max_clock_lag_seconds','clock_lag_p95_seconds','clock_lag_over_0p1_seconds_ticks']},
      'runtime_errors':world['physics_runtime']['errors'],'score':score,'reports_count':len(world['reports'])}
 if 'model_meter' in evidence:
  row['usage']=evidence['model_meter']['usage']
  row['decision_attempts']=evidence['decision_attempts'];row['commands_enqueued']=evidence['commands_enqueued']
  row['causal_join_dispositions']=dict(Counter(x['decode_status'] for x in evidence['causal_request_joins']))
  for key,value in terminal['settlement_file_hashes'].items():assert bind(Path(key))==value
 records.append(row)
 del world,raw_world

for leaf in ['episode-order.json','sampler-seeds.json','source-hashes.json','dependency-hashes.json','input-hashes.json']:
 bind(supervisor/leaf)
captures={}
for source,name in [(supervisor.parent/'inspection-eight-condition-pilot.v4.supervisor.stdout','supervisor.at-eighteen.stdout'),
                    (supervisor/'checkpoint-pauses.jsonl','checkpoint-pauses.at-eighteen.jsonl'),
                    (Path('/tmp/inspection-pilot-v4-at-eighteen-native-status.v1.json'),'native-status.at-eighteen.json')]:
 target=out/name;shutil.copyfile(source,target);captures[str(target.relative_to(run))]=sha(target)
status=json.loads((out/'native-status.at-eighteen.json').read_text())
assert status['stale']==[] and status['budget']['reserved']['calls']==0 and status['budget']['reserved']['tokens']==0
assert status['budget']['unknown_usage']==[] and status['budget']['unresolved_usage']==[]
means=[]
for condition,count in expected.items():
 rows=[r for r in records if r['condition']==condition]
 assert len(rows)==count
 means.append({'case_id':'dev-L1-01','condition':condition,'original_repeats':count,
               'components':{k:None if k=='D' else sum(r['score'][k] for r in rows)/count for k in ['C','S','D','Q','H','E','Y','SR']},
               'scope':'one complete development root, descriptive only; not a formal SDK outer recording or population comparison'})
value={'role':'Complete first development root, actual closed callbacks; not G4 or formal data',
       'at':datetime.now(timezone.utc).isoformat(),'completed_callbacks':18,'planned_callbacks':108,'complete_physical_root_blocks':1,
       'provided_sdk_final_outer_recording_complete':False,'barrier':barrier,'slots':records,'root_component_means':means,
       'native_budget':status['budget'],'bound_files':files,'immutable_captures':captures,
       'limits':['Ordinary clock summary/witness, not full72000-phase vectors.','Cost, capital, energy and host/Codex usage remain unmetered.','No new root, model request, seed, flight, SDK result or formal lock created.']}
(out/'binding.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
shutil.copyfile(Path(__file__),out/'capture-source.py')
print(json.dumps({'binding':str((out/'binding.json').relative_to(run)),'sha256':sha(out/'binding.json'),
 'callbacks':18,'physical_root_blocks':1,'max_clock_lag':max(r['clock']['max_clock_lag_seconds'] for r in records),
 'new_slots':records[-3:],'root_means':means},allow_nan=False))
