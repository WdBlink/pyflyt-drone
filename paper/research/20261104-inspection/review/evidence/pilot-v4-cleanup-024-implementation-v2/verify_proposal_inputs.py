from pathlib import Path
import importlib.util,sys,json,hashlib
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-024.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'source/continue_pilot.py')=='bb196f6c6939d9c050eb56f1b435eee68264347093b1cf85f0556cae12101c54'
assert sha(Q/'proposal.v3.json')=='c59a4e15439da55708acb84127ed02b9c7c2a0a7d962eb837a4f3c67cd742aac'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_cleanup_proposal',Q/'source/continue_pilot.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
proposal=m.read(Q/'proposal.v3.json')
for ref in proposal['files']:m.checked(REPO/ref['path'],ref['sha256'])
actual=m.closed_inputs();assert actual==proposal['closed_inputs']
assert proposal['new_resource_operations']==0 and proposal['disposition']=='late_cleanup_observed_complete_not_original_closed'
assert proposal['observations'] and all(not row['remaining_original_live'] and not row['current_same_user_python_orphans'] and row['original_unclaimed_pid_present'] is False for row in proposal['observations'])
result={'proposal_sha256':sha(Q/'proposal.v3.json'),'all_proposal_file_refs_match':True,'proposal_file_ref_count':len(proposal['files']),'complete_closed_inputs_match':True,'closed_directories':len(actual['directories']),'closed_files':len(actual['files']),'new_resource_operations':0,'observations_describe_only_later_quiet':True,'native_written':False}
(OUT/'proposal-inputs.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
