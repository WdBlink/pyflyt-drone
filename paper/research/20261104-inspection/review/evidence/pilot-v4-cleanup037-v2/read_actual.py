from pathlib import Path
import sys,types,json,hashlib,copy,tempfile
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-037.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'proposal.v2.json')=='4f94dbf6ed55fe748368b9d58dd74e7de2cc89c2bbcdc0cb36fa0a104143f238'
assert sha(Q/'source/disposition037.py')=='ebfc8a8f2efecf06362f01f078fef4c144aa5dcaecb925351c41d2bcb65cd2b3'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Q/'source'))
for name in ['launchd_provenance','inspect_original','disposition037']:
 path=Q/'source/disposition037.py' if name=='disposition037' else Q/'source'/f'{name}.py';module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module;exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
m=sys.modules['disposition037'];i=sys.modules['inspect_original'];l=sys.modules['launchd_provenance']
proposal=m.read(m.PROPOSAL);facts=m.read(m.FACTS);prior=m.read(Q/'proposal.v1.json');assert [k for k in prior if prior[k]!=proposal[k]]==['source']
assert m.checked(m.FACTS,m.FACTS_SHA)==proposal['facts']
for ref in [*facts['files'],*facts['captures']]:m.checked(REPO/ref['path'],ref['sha256'])
snapshot=m.receipt_snapshot(m.EPISODE,m.RESERVATION);assert snapshot==proposal['original_inputs']
owner=m.previous.owner();plan,terminal,tree,y,proof,refs=m.exact037(owner)
assert y==proposal['Y'] and proof==proposal['source_attribution']==facts['original037_external'] and refs==proposal['refs']
assert m.previous.canonical_digest(terminal['evidence'])==proposal['evidence_sha256']
assert m.receipt_snapshot(m.EPISODE,m.RESERVATION)==snapshot
try:m.verify()
except (ValueError,OSError) as error:reject=str(error)
else:raise AssertionError('actual acceptance unexpectedly passed')
result={'scope':'actual stored flight frozen validator recount, known native readonly owner, attribution and closed inputs; no capture/prepare/new flight','Y':y,'evidence_sha256':proposal['evidence_sha256'],'usage':terminal['evidence']['model_meter']['usage'],'causal_joins':len(terminal['evidence']['causal_request_joins']),'directories':len(snapshot['directories']),'files':len(snapshot['files']),'facts_refs':len(facts['files'])+len(facts['captures']),'original_unknown':terminal['status'],'actual_verify_reject':reject,'actual_acceptance_absent':not m.ACCEPTANCE.exists(),'proof_original_rows':len(proof['original_scan_rows'])}
(OUT/'actual-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
