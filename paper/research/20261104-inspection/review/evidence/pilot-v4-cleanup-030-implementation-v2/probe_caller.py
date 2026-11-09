from pathlib import Path
from types import SimpleNamespace
import ast,sys,importlib.util,tempfile,json,hashlib,copy
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-030.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'source/continue_pilot.py')=='4c78d5b130f27f15f34483f7f88bf5a7c348cef2399d272faae3c63ba3502a8f'
assert sha(Q/'proposal.v3.json')=='cffa9a073ce7819b087fdfe84c33d13ca29a643f18107bcf855bbf1fbc2cc7ae'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_cleanup030_v3',Q/'source/continue_pilot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('independent_observer030_v3',Q/'source/observe_cleanup.py');instrument=importlib.util.module_from_spec(spec);spec.loader.exec_module(instrument)
source=(Q/'source/continue_pilot.py').read_bytes();observer_source=(Q/'source/observe_cleanup.py').read_bytes();review_source=m.OBSERVER_REVIEW.read_bytes();oldplan=m.read(m.older().EPISODE/'plan.json');currentplan=m.read(m.EPISODE/'plan.json')
node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='run_continuation');nested=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='recovered')
orig_close=instrument.cleanup.SubprocessTree.close;close_return={'status':'closed','remaining_owned':[]};close_returns=[]
def fake_close(tree):close_returns.append(close_return);return close_return
instrument.cleanup.SubprocessTree.close=fake_close
result={'source_sha256':sha(Q/'source/continue_pilot.py'),'scope':'exact AST recovered incl final_metadata; real accepted observer/tmp IO; Fake delegates/cache verifiers; no outer continuation/SDK main/actual process/owner/resource/signal'}
try:
 with tempfile.TemporaryDirectory(prefix='caller030-v3-') as tmp:
  base=Path(tmp).resolve()
  for scenario in ['new_normal','cached_normal','review_drift','new_companion_drift','024_companion_drift','030_companion_drift','024_normal','030_normal','delegate_exception']:
   project=base/scenario;run=project/'run';output=project/'logs/new-sdk';output.mkdir(parents=True);bundle=run/'recovery/b';(bundle/'source').mkdir(parents=True);(run/'review').mkdir()
   m.PROJECT=project;m.RUN=run;m.__file__=str(bundle/'source/continue_pilot.py');Path(m.__file__).write_bytes(source);m.OBSERVER=bundle/'source/observe_cleanup.py';m.OBSERVER.write_bytes(observer_source);m.OBSERVER_REVIEW=run/'review/observer.json';m.OBSERVER_REVIEW.write_bytes(review_source);m.EPISODE=project/'logs/slot030'
   instrument.PROJECT=project;instrument.OUTPUT=output/'cleanup-observations'
   for name in ['episode-order.json','sampler-seeds.json']:(output/name).write_text('synthetic fixed input')
   context={'original_episode_order_sha256':sha(output/'episode-order.json'),'original_sampler_seeds_sha256':sha(output/'sampler-seeds.json'),'passive_observer':m.observer_refs(),'fixture':'not actual SDK or acceptance'}
   oldev={'fixture':'old accepted proof'};newev={'fixture':'030 proof'};companion=output/'recovery-adapter.json';error=OSError('original fake delegate error');calls=[]
   def drift_companion():companion.write_bytes(companion.read_bytes()+b' ')
   def old_verify(_):
    calls.append('oldverify')
    if scenario=='024_companion_drift':drift_companion()
    return .368353660488492,copy.deepcopy(oldev)
   def new_verify(_):
    calls.append('newverify')
    if scenario=='030_companion_drift':drift_companion()
    return .3436623024638006,copy.deepcopy(newev)
   previous=SimpleNamespace(EPISODE=project/'logs/slot024',PLAN_CANON=m.canonical_digest(oldplan),verify=old_verify)
   def execute(directory,plan,command_factory,**kwargs):
    calls.append('delegate')
    if scenario!='cached_normal':
     tree=SimpleNamespace(process=SimpleNamespace(pid=10),baseline={},owned={});assert instrument.cleanup.SubprocessTree.close(tree) is close_return
    if scenario=='review_drift':
     new=json.loads(review_source);new['reviewer']='unaccepted-new-reviewer';m.OBSERVER_REVIEW.write_text(json.dumps(new))
    if scenario=='new_companion_drift':drift_companion()
    if scenario=='delegate_exception':raise error
    return .1,{'cache_reused':scenario=='cached_normal','actual_attempts':1}
   env=dict(m.__dict__);env.update(source_sha=sha(Path(m.__file__)),output=output,context=context,previous=previous,execute=execute,instrument=instrument,prior_y=.368353660488492,prior_evidence=oldev,y=.3436623024638006,recovery_evidence=newev,acceptance=bundle/'synthetic-acceptance.json',verify=new_verify)
   exec(compile(ast.Module(body=[nested],type_ignores=[]),str(Q/'source/continue_pilot.py'),'exec'),env)
   if scenario.startswith('024_'):directory,plan,prefix=previous.EPISODE,oldplan,'dev-all8-pilot-v4-024'
   elif scenario.startswith('030_'):directory,plan,prefix=m.EPISODE,currentplan,'dev-all8-pilot-v4-030'
   else:directory,plan,prefix=project/'logs/synthetic-slot031',{'case_id':'dev-L3-01','condition':'B4-M1','repeat_ordinal':1},'dev-all8-pilot-v4-031'
   try:answer=env['recovered'](directory,plan,None,run=run,reservation_prefix=prefix)
   except BaseException as caught:
    if scenario=='delegate_exception':assert caught is error;result[scenario]={'same_original_exception_object':True}
    elif scenario.endswith('_drift'):result[scenario]={'rejected':True,'reason':str(caught)}
    else:raise
   else:
    assert not scenario.endswith('_drift') and scenario!='delegate_exception'
    if scenario in ['new_normal','cached_normal']:
     assert answer[0]==.1
     count=len(answer[1].get('passive_close_observations',[]));assert count==(0 if scenario=='cached_normal' else 1)
     if count:
      ref=answer[1]['passive_close_observations'][0];assert sha(project/ref['path'])==ref['sha256']
     result[scenario]={'passed':True,'sidecar_count':count}
    else:
     assert answer==(env['prior_y'],oldev) if scenario=='024_normal' else answer==(env['y'],newev)
     result[scenario]={'passed':True,'delegate_not_called':'delegate' not in calls,'marker_exists':(output/'checkpoint.request').exists()}
   assert instrument.cleanup.SubprocessTree.close is fake_close
   if 'delegate' in calls and scenario!='cached_normal':assert close_returns[-1] is close_return
finally:instrument.cleanup.SubprocessTree.close=orig_close
(OUT/'caller-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
