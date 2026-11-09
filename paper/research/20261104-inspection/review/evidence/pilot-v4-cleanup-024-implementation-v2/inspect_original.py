from pathlib import Path
import importlib.util,sys,json,hashlib,time
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-024.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'source/continue_pilot.py')=='bb196f6c6939d9c050eb56f1b435eee68264347093b1cf85f0556cae12101c54'
assert sha(Q/'proposal.v3.json')=='c59a4e15439da55708acb84127ed02b9c7c2a0a7d962eb837a4f3c67cd742aac'
assert sha(Q/'source/test_continue_pilot.py')=='988afab06e2381277f8fee8c218cb52281e35e007d51cff19a20b2f733595b10'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_cleanup_recovery',Q/'source/continue_pilot.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
start=time.monotonic();plan,tree,y,evidence,files=m.inspect_original();proposal=json.loads((Q/'proposal.v3.json').read_bytes())
assert y==proposal['Y'] and m.canonical_digest(evidence)==proposal['evidence_sha256']
result={'role':'actual original read-only validator recount, not continuation or recovery acceptance','source_sha256':sha(Q/'source/continue_pilot.py'),'proposal_sha256':sha(Q/'proposal.v3.json'),'tests_sha256':sha(Q/'source/test_continue_pilot.py'),'Y':y,'evidence_sha256':m.canonical_digest(evidence),'model_usage':evidence['model_meter']['usage'],'causal_join_count':len(evidence['causal_request_joins']),'files_returned':len(files),'elapsed_wall_seconds':time.monotonic()-start,'tree_status':tree['status'],'original_unknown_retained':m.read(m.ATTEMPT/'terminal.json')['status'],'no_continuation_called':True}
(OUT/'actual-inspect-original.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
