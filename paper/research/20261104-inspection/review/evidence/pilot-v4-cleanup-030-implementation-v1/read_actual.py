from pathlib import Path
import importlib.util,sys,json,hashlib,time
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-030.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'source/continue_pilot.py')=='722eedd58a0ed0f2f47e7db916047ef6391e24677f527fe06d9d0a8be305c098'
assert sha(Q/'proposal.v2.json')=='a4307b9ade56c82ddd60dc3af6d116be8b4525f8146545f43b17ce862ee09ced'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent030_actual',Q/'source/continue_pilot.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
start=time.monotonic();plan,tree,y,evidence,files=m.inspect_original();proposal=m.read(Q/'proposal.v2.json');assert y==proposal['Y'] and m.canonical_digest(evidence)==proposal['evidence_sha256']
old_y,old_evidence=m.older().verify(m.OLDER_ACCEPTANCE);assert old_evidence==proposal['prior_cleanup_evidence']
closed=m.closed_inputs();assert closed==proposal['closed_inputs']
for ref in proposal['files']:m.checked(REPO/ref['path'],ref['sha256'])
result={'role':'readonly actual030 inspection,024verify,combined closure; not continuation/new030acceptance','source_sha256':sha(Q/'source/continue_pilot.py'),'proposal_sha256':sha(Q/'proposal.v2.json'),'Y030':y,'Y024':old_y,'evidence030_sha256':m.canonical_digest(evidence),'usage030':evidence['model_meter']['usage'],'joins030':len(evidence['causal_request_joins']),'combined_directories':len(closed['directories']),'combined_files':len(closed['files']),'all_proposal_refs_checked':len(proposal['files']),'unknown030_retained':m.read(m.ATTEMPT/'terminal.json')['status'],'no_actual030_acceptance':not (Q/'acceptance.v1.json').exists(),'elapsed':time.monotonic()-start}
(OUT/'actual-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
