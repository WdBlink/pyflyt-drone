from pathlib import Path
import importlib.util,sys,json,hashlib
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-030.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'source/continue_pilot.py')=='4c78d5b130f27f15f34483f7f88bf5a7c348cef2399d272faae3c63ba3502a8f'
assert sha(Q/'proposal.v3.json')=='cffa9a073ce7819b087fdfe84c33d13ca29a643f18107bcf855bbf1fbc2cc7ae'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent030_delta',Q/'source/continue_pilot.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
a=m.read(Q/'continuation-freeze.v2/proposal.json');b=m.read(Q/'proposal.v3.json')
assert set(a)==set(b) and [k for k in a if a[k]!=b[k]]==['files','observations']
old={r['path']:r['sha256'] for r in a['files']};new={r['path']:r['sha256'] for r in b['files']};changes=[p for p in old if old[p]!=new[p]]
assert set(old)==set(new) and changes==[str((Q/'source/continue_pilot.py').relative_to(REPO))]
for ref in b['files']:m.checked(REPO/ref['path'],ref['sha256'])
closed=m.closed_inputs();assert closed==b['closed_inputs']
old_y,old_evidence=m.older().verify(m.OLDER_ACCEPTANCE);assert old_evidence==b['prior_cleanup_evidence']
state=m.owner();entry=state['usage_ledger'][m.RESERVATION];purpose=json.loads(entry['purpose'])
resource=R/'qualification/local-model-episodes'/m.RESERVATION/'usage-receipt.json'
assert entry['status']=='known' and entry['reported_usage']=={'calls':24,'tokens':173756} and entry['receipt_sha256']==sha(resource)
assert purpose['episode_plan_sha256']==m.PLAN_CANON and purpose['attempt_ordinal']==0 and purpose['repeat_ordinal']==1 and purpose['case_id']=='dev-L3-01' and purpose['condition']=='B4-M1'
assert state['task_stop'] is None and state['frozen_plan']==m.read(m.EPISODE/'plan.json')['owner_frozen_plan']
assert m.observer_refs()[1]['sha256']==m.OBSERVER_REVIEW_SHA
result={'proposal_sha256':sha(Q/'proposal.v3.json'),'proposal_diff_only_helper_sha_and_observation_timestamp':True,'all_file_refs_checked':len(b['files']),'combined_directories':len(closed['directories']),'combined_files':len(closed['files']),'combined_closure_matches_and_is_same_as_reviewed_v2':True,'Y030_unchanged':b['Y'],'evidence030_sha256_unchanged':b['evidence_sha256'],'024_verify_current_owner_passed':True,'Y024':old_y,'030_current_native_known_usage':entry['reported_usage'],'030_current_native_purpose_matches':True,'observer_review_fixed_sha':m.OBSERVER_REVIEW_SHA,'no_actual030_acceptance':not (Q/'acceptance.v1.json').exists(),'scope':'reuse unchanged full validator data from prior review; fresh refs/closure/old024verify/native known checks only; no continuation/write'}
(OUT/'proposal-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
