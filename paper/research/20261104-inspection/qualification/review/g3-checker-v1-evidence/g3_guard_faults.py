from pathlib import Path
from copy import deepcopy
import hashlib,json,tempfile
from eval.qualify_inspection_reference import qualify
ROOT=Path('.').resolve();R=ROOT/'paper/research/20261104-inspection';D=R/'qualification/reference/executor-realtime.v2';A=R/'qualification/source/executor-v19-reference-v2';C=R/'planning/gate-contract.v19.json'
run=json.loads((D/'run.json').read_text());world=json.loads((D/'world.json').read_text())
results=[]
with tempfile.TemporaryDirectory(prefix='g3-guard-faults-') as t:
 p=Path(t)
 for name in ['dependency_subset','omit_unemitted_opportunities','broken_observe_chain']:
  r=deepcopy(run);w=deepcopy(world)
  if name=='dependency_subset':r['dependency_hashes']={next(iter(r['dependency_hashes'])):next(iter(r['dependency_hashes'].values()))}
  elif name=='omit_unemitted_opportunities':w['sensor_opportunities']=[q for q in w['sensor_opportunities'] if q['emitted_evidence_id'] is not None];w['background_trials']=[q for q in w['background_trials'] if q['emitted_evidence_id'] is not None]
  else:
   for j in w['jobs']:
    if j['kind']=='observe':j['params']['candidate_id']='never-inspected-or-reported-candidate';j['params']['report_id']='never-submitted-report'
  (p/'world.json').write_text(json.dumps(w));r['world_sha256']=hashlib.sha256((p/'world.json').read_bytes()).hexdigest();(p/'run.json').write_text(json.dumps(r))
  q=qualify(p,C,A);results.append({'fault':name,'passed':q['passed'],'checks':q['checks'],'errors':q['errors'],'opportunities':q['sensor_opportunity_recount']})
Path('/tmp/g3_guard_faults_results.json').write_text(json.dumps(results,indent=2)+'\n')
for q in results:print(q['fault'],q['passed'],q['errors'])
