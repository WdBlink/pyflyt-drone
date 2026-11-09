from pathlib import Path
import sys,importlib.util,json,hashlib,tempfile,copy,zipfile,os
from types import SimpleNamespace
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-030.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'source/continue_pilot.py')=='722eedd58a0ed0f2f47e7db916047ef6391e24677f527fe06d9d0a8be305c098'
assert sha(Q/'proposal.v2.json')=='a4307b9ade56c82ddd60dc3af6d116be8b4525f8146545f43b17ce862ee09ced'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_recovery_windows',Q/'source/continue_pilot.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
original_plan=(m.EPISODE/'plan.json').read_bytes();original_terminal=(m.ATTEMPT/'terminal.json').read_bytes();original_tree=(m.ATTEMPT/'process-tree.json').read_bytes();original_receipt=(m.RUN/'qualification/local-model-episodes'/m.RESERVATION/'usage-receipt.json').read_bytes();source=(Q/'source/continue_pilot.py').read_bytes();admission=m.ADMISSION.read_bytes();proposal=copy.deepcopy(m.read(Q/'proposal.v2.json'));actual_bindings=m._bindings;observer_bytes=m.OBSERVER.read_bytes();observer_review_bytes=m.OBSERVER_REVIEW.read_bytes();prior=copy.deepcopy(proposal['prior_cleanup_evidence'])
result={'source_sha256':sha(Q/'source/continue_pilot.py'),'proposal_sha256':sha(Q/'proposal.v2.json'),'scope':'isolated /tmp fixture; real require_native_binding checker; original plan/terminal/tree/receipt byte copies; synthetic reviewed proposal and owner; payload binding returns trusted original sets, separately checks real _bindings directory behavior; no actual owner/data changes'}
with tempfile.TemporaryDirectory(prefix='recovery-window-') as tmp:
 p=Path(tmp).resolve();run=p/'run';bundle=run/'recovery/b';ep=p/'logs/episode';attempt=ep/'attempt-00';resource=run/'qualification/local-model-episodes'/m.RESERVATION;op=p/'logs/operation'
 for d in (bundle/'source',attempt,resource,op,run/'readers'):d.mkdir(parents=True,exist_ok=True)
 def put(path,raw):path.write_bytes(raw)
 put(ep/'plan.json',original_plan);put(attempt/'terminal.json',original_terminal);put(attempt/'process-tree.json',original_tree);put(resource/'usage-receipt.json',original_receipt);put(bundle/'source/continue_pilot.py',source);put(run/'readers/admission.py',admission)
 archive=resource/'complete-attempt.zip'
 with zipfile.ZipFile(archive,'w') as z:z.writestr('fixture','valid synthetic archive')
 archive_before=archive.read_bytes()
 m.PROJECT=p;m.RUN=run;m.BUNDLE=bundle;m.EPISODE=ep;m.ATTEMPT=attempt;m.ADMISSION=run/'readers/admission.py';m.__file__=str(bundle/'source/continue_pilot.py');m.operation_directory=lambda _:op
 m.OBSERVER=run/'readers/observer.py';m.OBSERVER.write_bytes(observer_bytes);m.OBSERVER_REVIEW=run/'readers/observer-review.json';m.OBSERVER_REVIEW.write_bytes(observer_review_bytes)
 older_stub=SimpleNamespace(closed_inputs=lambda:{'directories':[],'files':{}},verify=lambda _: (.368353660488492,copy.deepcopy(prior)))
 m.older=lambda:older_stub
 terminal=json.loads(original_terminal);m._bindings=lambda *_:(terminal['payload_hashes'],terminal['model_payload_hashes'])
 m.quiet_observation=lambda _:(_ for _ in ()).throw(AssertionError('history verify must not poll future processes'))
 pp=bundle/'proposal.json';rp=bundle/'review.json';ap=bundle/'acceptance.json'
 proposal['files']=[m.checked(x) for x in (ep/'plan.json',attempt/'terminal.json',attempt/'process-tree.json',Path(m.__file__),m.ADMISSION,m.OBSERVER,m.OBSERVER_REVIEW,resource/'usage-receipt.json',archive)]
 proposal['closed_inputs']=m.closed_inputs()
 def write(path,v):path.write_text(json.dumps(v,sort_keys=True)+'\n')
 write(pp,proposal);write(rp,{'schema':'inspection-dev-late-cleanup-review-v1','decision':'accept_late_cleanup_only','proposal_sha256':sha(pp),'reviewer':'isolated-synthetic-independent'})
 refs=[{'path':str(x.relative_to(run)),'sha256':sha(x)} for x in (pp,rp)]
 value={'schema':'inspection-dev-late-cleanup-acceptance-v1','proposal':refs[0],'review':refs[1],'finding_id':m.FINDING};write(ap,value);acceptance_before=ap.read_bytes()
 native=m.admission_module();source_ref={'path':str(Path(m.__file__).relative_to(run)),'sha256':sha(Path(m.__file__))}
 observed=[{'path':str((p/row['path']).relative_to(run)),'sha256':row['sha256']} for row in m.observer_refs()]
 request={'evidence':{x['path']:x['sha256'] for x in [*refs,source_ref,*observed]}};request['request_sha256']=native.canonical_sha(request)
 current={'rounds':[{'request':request,'decision':{'request_sha256':request['request_sha256'],'status':'active','progress':'verified_progress','finding':{'id':m.FINDING}},'validation':'structure_and_binding_only'}],'usage_ledger':{m.RESERVATION:{'status':'known','receipt_sha256':sha(resource/'usage-receipt.json'),'reported_usage':{'calls':24,'tokens':173756}}}}
 m.owner=lambda:current
 y,e=m.verify(ap);assert y==proposal['Y'];result['normal_approved_fixture_passes']=True
 def expect_rejected(label,action):
  def owner_window():
   action()
   return current
  m.owner=owner_window
  try:m.verify(ap)
  except ValueError as exc:
   result[label]={'rejected':True,'reason':str(exc)}
  else:raise AssertionError(label+' incorrectly accepted')
 expect_rejected('archive_changed_during_native_read',lambda:archive.write_bytes(archive_before+b'late byte drift'))
 archive.write_bytes(archive_before)
 def corrupt_acceptance():
  v=copy.deepcopy(value);v['finding_id']='unaccepted-other-finding';write(ap,v)
 expect_rejected('acceptance_changed_during_native_read',corrupt_acceptance)
 ap.write_bytes(acceptance_before)
 before=actual_bindings(attempt,op)
 for kind in ('empty_directory','FIFO'):
  extra=op/('call-9999' if kind=='empty_directory' else 'unexpected-fifo')
  def add_entry(extra=extra,kind=kind):
   if kind=='empty_directory':extra.mkdir()
   else:os.mkfifo(extra)
  expect_rejected(kind+'_added_during_native_read',add_entry)
  assert actual_bindings(attempt,op)==before
  result[kind+'_added_during_native_read']['real_original_bindings_unchanged']=True
  if kind=='empty_directory':extra.rmdir()
  else:extra.unlink()
 m.owner=lambda:current
 y,e=m.verify(ap);assert y==proposal['Y'];result['restored_normal_passes']=True
 assert e['cleanup_disposition']['acceptance']['sha256']==hashlib.sha256(acceptance_before).hexdigest()
 result['normal_acceptance_ref_held']=True
 # Direct exact-target guard: same genuine plan/terminal/tree bytes pass; one changed plan field fails canonical identity.
 m.target(json.loads(original_plan),json.loads(original_terminal),json.loads(original_tree))
 wrong=json.loads(original_plan);wrong['repeat_ordinal']=2
 try:m.target(wrong,json.loads(original_terminal),json.loads(original_tree))
 except ValueError as exc:result['wrong_repeat_identity_rejected']={'rejected':True,'reason':str(exc)}
 else:raise AssertionError('wrong plan repeat accepted')
 saved_rounds=current['rounds'];current['rounds']=[]
 try:m.verify(ap)
 except ValueError as exc:result['unapplied_native_finding_rejected']={'rejected':True,'reason':str(exc)}
 else:raise AssertionError('unapplied native finding accepted')
 current['rounds']=saved_rounds
 m.older=lambda:SimpleNamespace(closed_inputs=older_stub.closed_inputs,verify=lambda _: (.368353660488492,{**prior,'reservation':'unapproved-prior'}))
 try:m.verify(ap)
 except ValueError as exc:result['prior024_proof_change_rejected']={'rejected':True,'reason':str(exc)}
 else:raise AssertionError('prior proof drift accepted')
 m.older=lambda:older_stub
 current['rounds'][0]['decision']['finding']['id']='g4-dev-v4-slot024-late-cleanup-disposition-v1'
 try:m.verify(ap)
 except ValueError as exc:result['024_finding_cannot_authorize030']={'rejected':True,'reason':str(exc)}
 else:raise AssertionError('024 finding authorized030')
 current['rounds'][0]['decision']['finding']['id']=m.FINDING
 y,e=m.verify(ap);assert y==proposal['Y'];result['normal_after_negative_controls_passes']=True
(OUT/'window-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
