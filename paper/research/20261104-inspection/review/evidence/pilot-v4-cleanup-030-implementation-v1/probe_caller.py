from pathlib import Path
from types import SimpleNamespace
import ast,sys,importlib.util,tempfile,json,hashlib,shutil
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-030.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'source/continue_pilot.py')=='722eedd58a0ed0f2f47e7db916047ef6391e24677f527fe06d9d0a8be305c098'
assert sha(Q/'proposal.v2.json')=='a4307b9ade56c82ddd60dc3af6d116be8b4525f8146545f43b17ce862ee09ced'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_cleanup030',Q/'source/continue_pilot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('independent_observer030',Q/'source/observe_cleanup.py');instrument=importlib.util.module_from_spec(spec);spec.loader.exec_module(instrument)
source=(Q/'source/continue_pilot.py').read_bytes();observer_source=(Q/'source/observe_cleanup.py').read_bytes();review_source=m.OBSERVER_REVIEW.read_bytes()
node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='run_continuation');nested=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='recovered')
orig_close=instrument.cleanup.SubprocessTree.close;expected_result={'status':'closed','remaining_owned':[]};instrument.cleanup.SubprocessTree.close=lambda _:expected_result
result={'source_sha256':sha(Q/'source/continue_pilot.py'),'scope':'exact AST extracted recovered function; real accepted observer+real tmp IO; Fake delegate; no outer run_continuation, SDK main, owner, process, model, resource or signal'}
try:
 with tempfile.TemporaryDirectory(prefix='caller030-') as tmp:
  project=Path(tmp).resolve();run=project/'run';output=project/'logs/new-sdk';output.mkdir(parents=True);bundle=run/'recovery/b';(bundle/'source').mkdir(parents=True);(run/'review').mkdir()
  m.PROJECT=project;m.RUN=run;m.__file__=str(bundle/'source/continue_pilot.py');Path(m.__file__).write_bytes(source);m.OBSERVER=bundle/'source/observe_cleanup.py';m.OBSERVER.write_bytes(observer_source);m.OBSERVER_REVIEW=run/'review/observer.json';m.OBSERVER_REVIEW.write_bytes(review_source);m.EPISODE=project/'logs/slot030'
  instrument.PROJECT=project;instrument.OUTPUT=output/'cleanup-observations'
  for name in ['episode-order.json','sampler-seeds.json']:(output/name).write_text('synthetic fixed input')
  fixed_refs=m.observer_refs();context={'original_episode_order_sha256':sha(output/'episode-order.json'),'original_sampler_seeds_sha256':sha(output/'sampler-seeds.json'),'passive_observer':fixed_refs,'fixture':'not actual SDK or acceptance'}
  state={'mutate':False,'calls':0}
  def execute(directory,plan,command_factory,**kwargs):
   state['calls']+=1
   tree=SimpleNamespace(process=SimpleNamespace(pid=10),baseline={},owned={})
   assert instrument.cleanup.SubprocessTree.close(tree) is expected_result
   if state['mutate']:
    new=json.loads(review_source);new['reviewer']='unaccepted-new-reviewer';m.OBSERVER_REVIEW.write_text(json.dumps(new))
   return .1,{'cache_reused':False,'actual_attempts':1}
  env=dict(m.__dict__);env.update(source_sha=sha(Path(m.__file__)),output=output,context=context,previous=SimpleNamespace(EPISODE=project/'logs/slot024'),execute=execute,instrument=instrument)
  exec(compile(ast.Module(body=[nested],type_ignores=[]),str(Q/'source/continue_pilot.py'),'exec'),env)
  plan={'case_id':'dev-L3-01','condition':'B4-M1','repeat_ordinal':1}
  args=(project/'logs/synthetic-slot031',plan,None);kwargs={'run':run,'reservation_prefix':'dev-all8-pilot-v4-031'}
  y,e=env['recovered'](*args,**kwargs);assert y==.1 and len(e['passive_close_observations'])==1
  result['normal_fake_delegate_with_real_sidecar_passes']=True
  before=sha(m.OBSERVER_REVIEW);state['mutate']=True
  y,e=env['recovered'](*args,**kwargs)
  after=sha(m.OBSERVER_REVIEW);saved=json.loads((output/'recovery-adapter.json').read_bytes());held=saved['passive_observer'][1]['sha256']
  assert y==.1 and len(e['passive_close_observations'])==1 and before==held and before!=after
  result['observer_review_drift_during_delegate_incorrectly_accepted']={'accepted_callback_Y':y,'initial_review_sha256':before,'after_delegate_review_sha256':after,'companion_held_old_review_sha256':held,'new_review_reviewer':json.loads(m.OBSERVER_REVIEW.read_bytes())['reviewer'],'returned_sidecar_count':len(e['passive_close_observations'])}
  m.OBSERVER_REVIEW.write_bytes(review_source)
  y,e=env['recovered'](*args,**kwargs) if False else (.1,{})
  result['all_changes_tmp_only']=True
finally:instrument.cleanup.SubprocessTree.close=orig_close
(OUT/'caller-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
