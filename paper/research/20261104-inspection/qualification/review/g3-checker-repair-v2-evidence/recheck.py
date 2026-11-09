from pathlib import Path
from copy import deepcopy
import hashlib,json,tempfile
import eval.qualify_inspection_reference as admission
ROOT=Path.cwd();R=ROOT/'paper/research/20261104-inspection';D=R/'qualification/reference/executor-realtime.v2';A=R/'qualification/source/executor-v19-reference-v2';C=R/'planning/gate-contract.v19.json';P=R/'qualification/reference/executor-realtime.v2.replay.v2.json'
run=json.loads((D/'run.json').read_text());world=json.loads((D/'world.json').read_text());receipt=json.loads(P.read_text());out={}
assert admission.verify_dependencies(run);sub=deepcopy(run);sub['dependency_hashes']=dict(list(sub['dependency_hashes'].items())[:1]);assert not admission.verify_dependencies(sub)
bad=deepcopy(run);bad['dependency_hashes'][next(iter(bad['dependency_hashes']))]='0'*64;assert not admission.verify_dependencies(bad)
out['dependency_helper']={'actual_set':len(admission.expected_dependency_paths()),'full_pass':True,'subset_rejected':True,'wrong_bytes_rejected':True}
assert admission.verify_replay(receipt,D/'run.json',D/'world.json',world)
replay_faults={}
for name in ['no_receipt','run_hash','world_hash','replayer_hash','missing_group','group_false','rpc_count','response_diff']:
 r=deepcopy(receipt)
 if name=='no_receipt':r={}
 elif name in ['run_hash','world_hash','replayer_hash']:r[{'run_hash':'input_run_sha256','world_hash':'input_world_sha256','replayer_hash':'replayer_sha256'}[name]]='0'*64
 elif name=='missing_group':r['exact_matches'].pop('execution_tracking_trace')
 elif name=='group_false':r['exact_matches']['sensor_opportunities']=False
 elif name=='rpc_count':r['tool_calls_replayed']=115
 else:r['response_discrepancies']=[{'bad':'response'}]
 assert not admission.verify_replay(r,D/'run.json',D/'world.json',world),name
 replay_faults[name]='rejected'
out['replay_helper']=replay_faults
witness=admission.causal_chain_witness(world);assert witness and witness['candidate_id']=='candidate-0-00005';out['actual_witness']=witness
bad=deepcopy(world)
for j in bad['jobs']:
 if j['kind']=='observe':j['params'].update(candidate_id='never-inspected-or-reported-candidate',report_id='never-submitted-report')
assert admission.causal_chain_witness(bad) is None;out['original_disconnected_chain_helper']='rejected'
with tempfile.TemporaryDirectory(prefix='g3-repair-full-check-') as temp:
 p=Path(temp)
 for name in ['dependency_subset','omit_unemitted_opportunities','broken_observe_chain']:
  r=deepcopy(run);w=deepcopy(world)
  if name=='dependency_subset':r['dependency_hashes']=dict(list(r['dependency_hashes'].items())[:1])
  elif name=='omit_unemitted_opportunities':w['sensor_opportunities']=[q for q in w['sensor_opportunities'] if q['emitted_evidence_id'] is not None];w['background_trials']=[q for q in w['background_trials'] if q['emitted_evidence_id'] is not None]
  else:
   for j in w['jobs']:
    if j['kind']=='observe':j['params'].update(candidate_id='never-inspected-or-reported-candidate',report_id='never-submitted-report')
  (p/'world.json').write_text(json.dumps(w));r['world_sha256']=hashlib.sha256((p/'world.json').read_bytes()).hexdigest();(p/'run.json').write_text(json.dumps(r))
  q=admission.qualify(p,C,A,P);assert not q['passed'];out[name]={'passed':q['passed'],'checks':q['checks'],'witness':q['causal_chain_witness']}
 q=admission.qualify(D,C,A,P);assert q['passed'];out['actual_qualify']={'passed':True,'checks':q['checks'],'witness':q['causal_chain_witness']}
 q=admission.qualify(D,C,A);assert not q['passed'] and not q['checks']['complete_bound_execution_replay'];out['no_replay_qualify']='rejected'
Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2)+'\n');print('all original3 faults rejected; every dependency/replay helper negative passed; actual witness/result accepted; no receipt rejected')
