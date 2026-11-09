from pathlib import Path
from datetime import datetime,timedelta
from types import ModuleType,SimpleNamespace
import tempfile,json,hashlib,sys,copy

REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-037.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
policy_path=Q/'prospective-policy.v1.json';assert sha(policy_path)=='172c847eeed52bf6a0018c837e8aa24801eeaedde522cff7e72063a64833764d'
policy=json.loads(policy_path.read_bytes())
for path,h in policy['sources'].items():assert sha(REPO/path)==h
for path,h in policy['evidence'].items():assert sha(R/path)==h
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Q/'source'))
for name in ['launchd_provenance','inspect_original','disposition037','admit_external','continue_pilot']:
 path=Q/'source'/f'{name}.py';mod=ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod;exec(compile(path.read_bytes(),str(path),'exec'),mod.__dict__)
a=sys.modules['admit_external'];c=sys.modules['continue_pilot'];d=sys.modules['disposition037'];i=sys.modules['inspect_original']
side_original=i.read(i.SIDE);tree=copy.deepcopy(side_original['original_result']);plan=i.read(i.EPISODE/'plan.json');terminal=i.read(i.EPISODE/'attempt-00/terminal.json');events_bytes=(Q/'launchd.v1/slot037.json').read_bytes();boot=json.loads(events_bytes)[0]['bootUUID'];Y=.3314065510597302
orig_close=side_original['close_started_at'];anchor=datetime.fromisoformat(orig_close)+timedelta(seconds=5)
class Clock:
 @staticmethod
 def now(tz):return anchor.astimezone(tz)
result={'scope':'real reconcile_new, _events, match_sidecar, run_observed and file IO; archived exact 037 event bytes reused in synthetic 038 context; Fake original execution/validator/native policy/OS transport and observer.check hook; no actual SDK/model/physics/native/OS query'}
with tempfile.TemporaryDirectory(prefix='external-policy-') as tmp:
 base=Path(tmp).resolve()
 for mode in ['normal','proof_deleted_after_reconcile','preexisting_unknown']:
  project=base/mode;run=project/'run';flights=project/'logs/flights';output=project/'logs/new-sdk';observation=output/'cleanup-observations';observation.mkdir(parents=True);run.mkdir()
  prefix='dev-all8-pilot-v4-038';reservation=prefix+'-a0';directory=flights/f"{plan['case_id']}-{plan['condition']}-{plan['repeat_ordinal']}";attempt=directory/'attempt-00';op=project/'logs/model';resource=run/'qualification/local-model-episodes'/reservation
  for m in [a,c,d]:m.PROJECT=project;m.RUN=run
  a.FLIGHTS=flights;a.OUTPUT=output;a.OBSERVER=observation;a.previous.PROJECT=project;a.previous.operation_directory=lambda _:op
  def write(p,v):p.write_text(json.dumps(v,sort_keys=True)+'\n')
  held_policy=run/'policy.json';held_policy.write_bytes(policy_path.read_bytes());held_review=run/'review.json';write(held_review,{'fixture':'native policy precondition stub'})
  refs={'policy':a.checked(held_policy),'review':a.checked(held_review),'sources':[],'native_request_sha256':'1'*64};a.policy_refs=lambda _:refs;a.previous.owner=lambda:{'fixture':'known owner'}
  side=copy.deepcopy(side_original);side['context']=i.context(plan,prefix);side_path=observation/'sidecar.json';write(side_path,side);side_ref=a.checked(side_path)
  a.datetime=Clock;a._boot=lambda:boot;a.previous._inventory=lambda:{}
  steps=[];deleted=[]
  def log_run(argv,**kwargs):
   assert steps==['old_complete_known_stub','full_validator_stub'];assert argv[:2]==['/usr/bin/log','show'];steps.append('fake_OS_transport');return SimpleNamespace(returncode=0,stderr=b'',stdout=events_bytes)
  a.subprocess=SimpleNamespace(run=log_run)
  def inspect_attempt(path,ordinal,reservation_arg,owner):
   assert path==directory and ordinal==0 and reservation_arg==reservation and steps==['old_complete_known_stub']
   steps.append('full_validator_stub')
   return plan,terminal,tree,Y,[a.checked(p) for p in (directory/'plan.json',attempt/'terminal.json',attempt/'process-tree.json',resource/'usage-receipt.json')]
  a.inspect_attempt=inspect_attempt
  proof=output/'external-cleanup-dispositions/038/proof.json'
  class Observer:
   def __init__(self,info):assert info==side['context'];self.n=0
   def __enter__(self):return self
   def __exit__(self,*exc):return False
   def check(self):
    self.n+=1;held=a.checked(side_path,side_ref['sha256'])
    if self.n==2 and mode=='proof_deleted_after_reconcile':
     assert proof.is_file();deleted.append(sha(proof));proof.unlink()
    return [held]
  original=c.UnresolvedEpisode('unknown or source/input/score/causal fault blocks this LLM slot')
  def execute(path,actual_plan,command,**kwargs):
   assert path==directory and actual_plan==plan
   if mode!='preexisting_unknown':
    for p in [attempt,op,resource]:p.mkdir(parents=True)
    write(directory/'plan.json',plan);write(attempt/'terminal.json',terminal);write(attempt/'process-tree.json',tree);write(resource/'usage-receipt.json',{'fixture':True,'usage':{'calls':22,'tokens':160664}})
    steps.append('old_complete_known_stub')
   raise original
  if mode=='preexisting_unknown':attempt.mkdir(parents=True)
  try:y,ev=c.run_observed(execute,SimpleNamespace(CloseObserver=Observer),directory,plan,None,{'reservation_prefix':prefix},refs)
  except c.UnresolvedEpisode as error:
   assert mode=='preexisting_unknown' and error is original and steps==[] and not proof.exists();result[mode]={'original_exception_preserved':True,'no_lookup_or_proof':True}
  else:
   ref=ev['cleanup_disposition']['proof'];assert mode!='preexisting_unknown' and y==Y and ev['cleanup_disposition']['original_status_retained']=='unknown_process_cleanup' and ev['passive_close_observations']==[side_ref]
   if mode=='normal':assert sha(proof)==ref['sha256'];result[mode]={'passed':True,'proof_exists':True,'old_unknown_preserved':True,'steps':steps}
   else:
    assert deleted==[ref['sha256']] and not proof.exists();result[mode]={'incorrectly_returned_Y':y,'returned_ref':ref,'proof_exists_at_return':False,'sidecar_still_matches':sha(side_path)==side_ref['sha256'],'steps':steps}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
