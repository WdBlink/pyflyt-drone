from pathlib import Path
import importlib.util,sys,json,hashlib,time
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-024.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'source/continue_pilot.py')=='72cb11b3cde95756fcf661a103e6042a3fcdd5b941c963a88a9438c24f5ce023'
assert sha(Q/'proposal.v2.json')=='f81f4500466d5793636626afbb335564ecf5460fcc82305a9168f89af11ef52b'
assert sha(Q/'source/test_continue_pilot.py')=='1073a1baaec2631f04cfd91a51949ac094fa8d2ad6cc5848bf4745609d9e8da3'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_cleanup_recovery',Q/'source/continue_pilot.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
start=time.monotonic();plan,tree,y,evidence,files=m.inspect_original();proposal=json.loads((Q/'proposal.v2.json').read_bytes())
assert y==proposal['Y'] and m.canonical_digest(evidence)==proposal['evidence_sha256']
result={'role':'actual original read-only validator recount, not continuation or recovery acceptance','source_sha256':sha(Q/'source/continue_pilot.py'),'proposal_sha256':sha(Q/'proposal.v2.json'),'tests_sha256':sha(Q/'source/test_continue_pilot.py'),'Y':y,'evidence_sha256':m.canonical_digest(evidence),'model_usage':evidence['model_meter']['usage'],'causal_join_count':len(evidence['causal_request_joins']),'files_returned':len(files),'elapsed_wall_seconds':time.monotonic()-start,'tree_status':tree['status'],'original_unknown_retained':m.read(m.ATTEMPT/'terminal.json')['status'],'no_continuation_called':True}
(OUT/'actual-inspect-original.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
