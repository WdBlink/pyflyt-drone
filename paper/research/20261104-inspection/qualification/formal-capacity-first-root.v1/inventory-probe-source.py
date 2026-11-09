"""Partial offline capacity measurement at an acknowledged pilot barrier."""
from collections import Counter
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time

started=time.monotonic();project=Path.cwd();run=project/'paper/research/20261104-inspection'
package=run/'runtime/autoresearch-document-snapshots-v2'
sys.path.insert(0,str(package));from lib.runtime import Controller,digest,file_hash
from lib.budget import summary
out=Path('/tmp/inspection-capacity-first-root.v1');assert not out.exists();out.mkdir()
ctl=Controller(run,package)
with ctl.lock():
 state=ctl.load();baseline=file_hash(ctl.path)
 assert baseline==json.loads(Path('/tmp/inspection-native-capacity-at-first-root.v1.json').read_text())['base_file_sha256']
 golden=out/'golden';golden.mkdir()
 for leaf in ['control.json','checkpoint.json']:
  source=run/'.paper'/leaf
  shutil.copyfile(source,golden/leaf)
  assert file_hash(source)==file_hash(golden/leaf)
assert summary(state)['unknown_usage']==[] and summary(state)['reserved']['calls']==0

def sizes(root):
 files=[]
 for p in root.rglob('*'):
  if p.is_file():
   assert not p.is_symlink()
   files.append((str(p),p.stat().st_size))
 return {'files':len(files),'logical_bytes':sum(v for _,v in files),
  'max_file':max(files,key=lambda r:r[1]) if files else None,
  'longest_path_bytes':max((len(k.encode()) for k,_ in files),default=0)}

capture=json.loads((run/'qualification/pilot-v4-first-root.v1/binding.json').read_text())
assert capture['barrier']['order_index']==18
inventory=[];groups=Counter();archive_bytes=[];flight_bytes=[];world_bytes=[]
for slot in capture['slots']:
 path=project/'logs/inspection-eight-condition-dev-flights.v4'/slot['name'];m=sizes(path)
 inventory.append({'kind':'episode_originals','slot':slot['order_index'],'root':str(path.relative_to(project)),**m});groups['episode_originals']+=m['logical_bytes']
 flight_bytes.append(m['logical_bytes']);world_bytes.append((path/'attempt-00/flight/world.json').stat().st_size)
 if 'usage' in slot:
  key=f"dev-all8-pilot-v4-{slot['order_index']:03d}-a0"
  for kind,path in [('model_originals',project/'logs/inspection-model-resource-operations.v1'/key),('settlement_archive_and_receipt',run/'qualification/local-model-episodes'/key)]:
   m=sizes(path);inventory.append({'kind':kind,'slot':slot['order_index'],'root':str(path.relative_to(project)),**m});groups[kind]+=m['logical_bytes']
  archive_bytes.append((run/'qualification/local-model-episodes'/key/'complete-attempt.zip').stat().st_size)
for kind,path in [('owner_store',run/'.paper'),('all_native_cli_trace_baseline',run.parent/(run.name+'-cli'))]:
 m=sizes(path);inventory.append({'kind':kind,'root':str(path.relative_to(project)),**m});groups[kind]+=m['logical_bytes']

# Shape projections use copies only. They are not native records or claimed usage.
ledger=copy.deepcopy(state['usage_ledger']);template=max(ledger.values(),key=lambda v:len(json.dumps(v)))
assert 'cost' not in template['reported_usage']
max_token_budget=max(m['generation_profile']['max_total_tokens'] for m in json.loads((run/'planning/local-model-registry.v1.json').read_text())['models'].values())
max_calls=max(m['generation_profile']['max_calls'] for m in json.loads((run/'planning/local-model-registry.v1.json').read_text())['models'].values())
for i in range(3456):
 key=f'formal-ablation-'+('f'*12)+f'-{i:04d}-a1'
 assert len(key)<96 and key not in ledger
 row=copy.deepcopy(template);row['reported_usage']={'calls':max_calls,'tokens':max_token_budget};row['status']='known'
 ledger[key]=row
shape={**state,'usage_ledger':ledger};budget=summary(shape)
assert len(budget['unreported_fields'])==len(summary(state)['unreported_fields'])+3456
assert budget['known_usage']['cost']=={} and budget['unknown_usage']==[]
peak_result={'status':'known','reservation':'formal-ablation-'+('f'*12)+'-9999-a1','budget':budget}
compact=len(json.dumps(peak_result,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())+1
wire=len(json.dumps(peak_result,indent=2,ensure_ascii=False,allow_nan=False).encode())+1
trace=len(json.dumps({'command':'budget-settle','exit_code':0,'response':peak_result},ensure_ascii=False,allow_nan=False).encode())+1
store=shutil.disk_usage(project);fs=project.stat().st_dev
observed_max={'episode_originals_bytes':max(flight_bytes),'world_bytes':max(world_bytes),
 'model_originals_bytes':max(r['logical_bytes'] for r in inventory if r['kind']=='model_originals'),
 'archive_bytes':max(archive_bytes)}
# These are explicit sample-based projections, not worst-case file-size bounds.
per_attempt=observed_max['episode_originals_bytes']+observed_max['model_originals_bytes']+observed_max['archive_bytes']
result={'role':'First-root partial offline inventory and representation projection, not capacity/G4 acceptance',
 'at':datetime.now(timezone.utc).isoformat(),'barrier':capture['barrier'],'base_state_sha256':digest(state),'base_file_sha256':baseline,
 'golden_pair':{leaf:{'path':str(golden/leaf),'sha256':file_hash(golden/leaf),'bytes':(golden/leaf).stat().st_size} for leaf in ['control.json','checkpoint.json']},
 'source_sha256':{str(p.relative_to(project)):file_hash(p) for p in [Path(__file__) if Path(__file__).is_relative_to(project) else package/'scripts/research.py',package/'lib/runtime.py',package/'lib/budget.py']},
 'inventory':inventory,'nonoverlapping_inventory_logical_bytes':dict(groups),'observed_L1_maxima':observed_max,
 'future_workload':{'formal_episodes':2016,'max_total_attempts':4032,'LLM_attempts':3456,'reserve_settle_ops':6912,
 'status_plus_other_wire_assumption':2000,'assumption_not_a_budget_limit':True},
 'peak_summary_shape':{'future_entries':3456,'compact_result_bytes':compact,'pretty_wire_bytes':wire,'CLI_trace_bytes':trace,
 'single_object_limit':16*1024*1024,'below_single_object_limit':compact<16*1024*1024,
 'conservative_transport_stdout_and_trace_bytes':(wire+trace)*(6912+2000),
 'conservative_unique_result_object_bytes':compact*(6912+2000),
 'scope':'Peak-size times all operations is an upper estimate within this shape; distinct raw/ZIP growth is separate. Unknown cost remains omitted, not zero.'},
 'sample_based_future_bytes':{'2016_attempts_at_L1_observed_max':2016*per_attempt,'4032_attempts_at_L1_observed_max':4032*per_attempt,
 'not_worst_case':True,'limitations':'Only one L1 root; L2/L3 events, failures, final metadata and future round growth not bounded here.'},
 'filesystem':{'device':fs,'total_bytes':store.total,'used_bytes':store.used,'free_bytes':store.free,'measurement':'logical stat bytes and current disk_usage; APFS physical allocation/external changes not modeled'},
 'elapsed_seconds':time.monotonic()-started,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'limits':['Partial source-bound shape/inventory only. No actual reserve/settle, init, migration, formal entropy/model/physics or second ledger.',
 'Old 13000ops/5000attempts/2000round projection is separately repeated, not a guarantee for larger future round requests.',
 'No new whole-graph import/save/fsync/recovery/export/120s end-to-end stress test; complete actual owner status observed separately.',
 'Final 108-pilot, source freeze, L3/failure/archive/round-document/analysis-output and disk headroom refresh remain required.']}
(out/'measurement.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
shutil.copyfile(Path(__file__),out/'probe-source.py')
print(json.dumps({k:result[k] for k in ['role','base_file_sha256','observed_L1_maxima','peak_summary_shape','sample_based_future_bytes','filesystem','elapsed_seconds','process_peak_rss_bytes']},allow_nan=False))
