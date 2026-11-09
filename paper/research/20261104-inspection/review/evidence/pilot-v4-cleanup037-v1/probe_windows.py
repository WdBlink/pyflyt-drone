from pathlib import Path
import sys,types,json,hashlib,copy,tempfile
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-037.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'proposal.v1.json')=='4d63026f0ba4e1e0d3cbaa22942c78aa60c4325b0b5bceff366eeeed1c7f1c2b'
assert sha(Q/'rejected.v1/disposition037.py')=='b672966121e7789f7541a37c75f3cbfad3863e70969cf45b73c45606b6c48e07'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Q/'source'))
for name in ['launchd_provenance','inspect_original','disposition037']:
 path=Q/'rejected.v1/disposition037.py' if name=='disposition037' else Q/'source'/f'{name}.py';module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module;exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
m=sys.modules['disposition037'];i=sys.modules['inspect_original'];l=sys.modules['launchd_provenance']
source=Path(m.__file__).read_bytes();proposal=m.read(m.PROPOSAL);facts=m.read(m.FACTS);plan_raw=(m.EPISODE/'plan.json').read_bytes();terminal_raw=(m.EPISODE/'attempt-00/terminal.json').read_bytes();tree_raw=(m.EPISODE/'attempt-00/process-tree.json').read_bytes();side_raw=m.SIDE.read_bytes();admission=m.previous.admission_module()
result={'scope':'real verify and native-binding checker, isolated real IO, synthetic native accepted finding, exact037 recount stub using actual plan/terminal/tree/proof; no owner write or dispatch'}
with tempfile.TemporaryDirectory(prefix='exact037-') as tmp:
 project=Path(tmp).resolve();run=project/'run';bundle=run/'recovery/037';ep=project/'logs/episode';at=ep/'attempt-00';op=project/'logs/operation';res=run/'qualification/local-model-episodes'/m.RESERVATION
 for d in (bundle/'source',at,op,res):d.mkdir(parents=True,exist_ok=True)
 for path,raw in [(ep/'plan.json',plan_raw),(at/'terminal.json',terminal_raw),(at/'process-tree.json',tree_raw),(bundle/'sidecar.json',side_raw),(bundle/'source/disposition037.py',source)]:path.write_bytes(raw)
 m.PROJECT=project;m.RUN=run;m.BUNDLE=bundle;m.EPISODE=ep;m.SIDE=bundle/'sidecar.json';m.__file__=str(bundle/'source/disposition037.py');m.FACTS=bundle/'facts.json';m.PROPOSAL=bundle/'proposal.json';m.ACCEPTANCE=bundle/'acceptance.json';m.previous.PROJECT=project;m.previous.operation_directory=lambda _:op;m.previous.admission_module=lambda:admission
 def write(p,v):p.write_text(json.dumps(v,sort_keys=True)+'\n')
 witness=bundle/'archived-witness.json';witness.write_bytes(b'{"original":true}\n');witness_bytes=witness.read_bytes();facts['files']=[m.checked(witness)];facts['captures']=[];write(m.FACTS,facts);facts_bytes=m.FACTS.read_bytes();m.FACTS_SHA=sha(m.FACTS)
 refs=[m.checked(ep/'plan.json'),m.checked(at/'terminal.json'),m.checked(at/'process-tree.json'),m.checked(m.SIDE)]
 proposal.update(source=m.checked(Path(m.__file__)),facts=m.checked(m.FACTS),observer=m.checked(m.SIDE),refs=refs,original_inputs=m.receipt_snapshot(ep,m.RESERVATION));write(m.PROPOSAL,proposal)
 review=bundle/'review.json';write(review,{'schema':'inspection-dev-external-cleanup-review-v1','decision':'accept_exact037_only','open_findings':[],'reviewer':'isolated-test-reviewer','proposal_sha256':sha(m.PROPOSAL),'disposition_source_sha256':sha(Path(m.__file__))})
 art=lambda p:{'path':str(p.relative_to(run)),'sha256':sha(p)}
 write(m.ACCEPTANCE,{'schema':'inspection-dev-external-cleanup-acceptance-v1','finding_id':m.FINDING,'proposal':art(m.PROPOSAL),'review':art(review)})
 native_refs=[art(p) for p in (m.PROPOSAL,review,m.FACTS,Path(m.__file__))];request={'evidence':{r['path']:r['sha256'] for r in native_refs}};request['request_sha256']=admission.canonical_sha(request)
 row={'request':request,'decision':{'request_sha256':request['request_sha256'],'status':'active','progress':'verified_progress','finding':{'id':m.FINDING}},'validation':'structure_and_binding_only'};owner={'rounds':[row]}
 m.exact037=lambda _: (json.loads(plan_raw),json.loads(terminal_raw),json.loads(tree_raw),proposal['Y'],proposal['source_attribution'],refs)
 m.previous.owner=lambda:owner
 y,ev=m.verify();assert y==proposal['Y'];result['normal_control_passes']=True
 def changed_facts():
  write(m.FACTS,{**facts,'files':[],'captures':[]});witness.write_bytes(b'{"original":false}\n');return owner
 m.previous.owner=changed_facts
 y,ev=m.verify();assert y==proposal['Y'] and sha(m.FACTS)!=m.FACTS_SHA
 result['facts_and_witness_changed_in_owner_window_incorrectly_accepted']={'Y':y,'expected_facts_sha':m.FACTS_SHA,'actual_facts_sha':sha(m.FACTS),'witness_sha_changed':sha(witness)!=facts['files'][0]['sha256'],'returned_facts_sha':ev['cleanup_disposition']['external_attribution']['sha256']}
 m.FACTS.write_bytes(facts_bytes);witness.write_bytes(witness_bytes)
 def changed_source():Path(m.__file__).write_bytes(source+b'\n# late source drift\n');return owner
 m.previous.owner=changed_source
 y,ev=m.verify();assert y==proposal['Y'] and sha(Path(m.__file__))!=proposal['source']['sha256']
 result['helper_source_changed_in_owner_window_incorrectly_accepted']={'Y':y,'expected_source_sha':proposal['source']['sha256'],'actual_source_sha':sha(Path(m.__file__))};Path(m.__file__).write_bytes(source)
 m.previous.owner=lambda:{'rounds':[]}
 try:m.verify()
 except ValueError as err:result['missing_native_finding_rejected']=str(err)
 else:raise AssertionError('missing native finding accepted')
 m.previous.owner=lambda:owner
 (op/'unexpected-empty-call').mkdir()
 try:m.verify()
 except ValueError as err:result['extra_empty_dir_rejected']=str(err)
 else:raise AssertionError('extra empty dir accepted')
 (op/'unexpected-empty-call').rmdir()
 y,ev=m.verify();assert y==proposal['Y'];result['restored_normal_control_passes']=True
(OUT/'window-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
