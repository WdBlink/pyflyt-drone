from pathlib import Path
import sys,importlib.util,json,hashlib,tempfile,copy,zipfile,os
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-024.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'source/continue_pilot.py')=='72cb11b3cde95756fcf661a103e6042a3fcdd5b941c963a88a9438c24f5ce023'
assert sha(Q/'proposal.v2.json')=='f81f4500466d5793636626afbb335564ecf5460fcc82305a9168f89af11ef52b'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_recovery_windows',Q/'source/continue_pilot.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
original_plan=(m.EPISODE/'plan.json').read_bytes();original_terminal=(m.ATTEMPT/'terminal.json').read_bytes();original_tree=(m.ATTEMPT/'process-tree.json').read_bytes();original_receipt=(m.RUN/'qualification/local-model-episodes'/m.RESERVATION/'usage-receipt.json').read_bytes();source=(Q/'source/continue_pilot.py').read_bytes();admission=m.ADMISSION.read_bytes();proposal=copy.deepcopy(m.read(Q/'proposal.v2.json'));actual_bindings=m._bindings
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
 terminal=json.loads(original_terminal);m._bindings=lambda *_:(terminal['payload_hashes'],terminal['model_payload_hashes'])
 m.quiet_observation=lambda _:(_ for _ in ()).throw(AssertionError('history verify must not poll future processes'))
 pp=bundle/'proposal.json';rp=bundle/'review.json';ap=bundle/'acceptance.json'
 proposal['files']=[m.checked(x) for x in (ep/'plan.json',attempt/'terminal.json',attempt/'process-tree.json',Path(m.__file__),m.ADMISSION,resource/'usage-receipt.json',archive)]
 def write(path,v):path.write_text(json.dumps(v,sort_keys=True)+'\n')
 write(pp,proposal);write(rp,{'schema':'inspection-dev-late-cleanup-review-v1','decision':'accept_late_cleanup_only','proposal_sha256':sha(pp),'reviewer':'isolated-synthetic-independent'})
 refs=[{'path':str(x.relative_to(run)),'sha256':sha(x)} for x in (pp,rp)]
 value={'schema':'inspection-dev-late-cleanup-acceptance-v1','proposal':refs[0],'review':refs[1],'finding_id':m.FINDING};write(ap,value);acceptance_before=ap.read_bytes()
 native=m.admission_module();source_ref={'path':str(Path(m.__file__).relative_to(run)),'sha256':sha(Path(m.__file__))}
 request={'evidence':{x['path']:x['sha256'] for x in [*refs,source_ref]}};request['request_sha256']=native.canonical_sha(request)
 current={'rounds':[{'request':request,'decision':{'request_sha256':request['request_sha256'],'status':'active','progress':'verified_progress','finding':{'id':m.FINDING}},'validation':'structure_and_binding_only'}],'usage_ledger':{m.RESERVATION:{'status':'known','receipt_sha256':sha(resource/'usage-receipt.json'),'reported_usage':{'calls':25,'tokens':166000}}}}
 m.owner=lambda:current
 y,e=m.verify(ap);assert y==proposal['Y'];result['normal_approved_fixture_passes']=True
 def corrupt_archive():archive.write_bytes(archive_before+b'late byte drift');return current
 m.owner=corrupt_archive;y,e=m.verify(ap);assert sha(archive)!=proposal['files'][-1]['sha256'];result['archive_changed_during_native_read_incorrectly_accepted']={'Y':y,'before':proposal['files'][-1]['sha256'],'after':sha(archive)};archive.write_bytes(archive_before)
 def corrupt_acceptance():
  v=copy.deepcopy(value);v['finding_id']='unaccepted-other-finding';write(ap,v);return current
 m.owner=corrupt_acceptance;y,e=m.verify(ap);assert e['cleanup_disposition']['acceptance']['sha256']==sha(ap) and sha(ap)!=hashlib.sha256(acceptance_before).hexdigest();result['acceptance_rebound_to_unvalidated_late_bytes']={'Y':y,'returned_acceptance_sha':e['cleanup_disposition']['acceptance']['sha256'],'initial_sha':hashlib.sha256(acceptance_before).hexdigest(),'mutated_finding':json.loads(ap.read_bytes())['finding_id']};ap.write_bytes(acceptance_before)
 before=actual_bindings(attempt,op)
 for kind in ('empty_directory','FIFO'):
  extra=op/('call-9999' if kind=='empty_directory' else 'unexpected-fifo')
  def add_entry(extra=extra,kind=kind):
   if kind=='empty_directory':extra.mkdir()
   else:os.mkfifo(extra)
   return current
  m.owner=add_entry;y,e=m.verify(ap);assert actual_bindings(attempt,op)==before;result[kind+'_added_during_native_read_incorrectly_accepted']={'Y':y,'real_original_bindings_unchanged':True}
  if kind=='empty_directory':extra.rmdir()
  else:extra.unlink()
 m.owner=lambda:current;y,e=m.verify(ap);assert y==proposal['Y'];result['restored_normal_passes']=True
(OUT/'window-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
