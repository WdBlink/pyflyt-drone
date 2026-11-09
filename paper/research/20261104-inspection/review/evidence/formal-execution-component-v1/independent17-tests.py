"""Bounded /tmp literal/Fake primary execution audit; every actual RNG forbidden."""
from pathlib import Path
import importlib.util,json,sys,ast,secrets
from dataclasses import replace
import pytest
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO));R=REPO/'paper/research/20261104-inspection';T=R/'planning/formal-admission-candidate.v3/tests'
s=importlib.util.spec_from_file_location('independent_execution_fixture_v1',T/'test_formal_execution.py');x=importlib.util.module_from_spec(s);sys.modules[s.name]=x;s.loader.exec_module(x);m=x.module;p=x.producer
obs=[]
def no_rng(*a,**k):raise AssertionError('actual formal random construction forbidden')
@pytest.fixture(autouse=True)
def forbidden(monkeypatch):
 for name in ('PCG64','Generator','SeedSequence','default_rng'):monkeypatch.setattr(x.recording.np.random,name,no_rng)
 monkeypatch.setattr(secrets,'randbits',no_rng)
def prepare(f,phase):
 opening=f.typed_main_opening(f.dataset()[1]) if phase=='holdout' else None
 refs,execution=f.execution(phase,opening);return refs,execution,opening
@pytest.mark.parametrize('phase',['main','holdout'])
def test_normal_reader_snapshots_and_full_slot_vector(tmp_path,phase):
 f=x.ExecutionFixture(tmp_path);refs,execution,opening=prepare(f,phase);original=f.verification.owner_reader;calls=0
 def read(run):
  nonlocal calls
  calls+=1;return original(run)
 plan=f.execution_proof(refs,execution,opening,replace(f.verification,owner_reader=read))
 assert calls==(5 if phase=='main' else 9)
 expected=1296 if phase=='main' else 216;assert len(plan.slots)==expected
 assert len({(s.case_id,s.condition,s.repeat_ordinal) for s in plan.slots})==expected
 assert sum(s.sampler_seed is not None for s in plan.slots)==expected//18*15
 obs.append({'probe':'normal reader','phase':phase,'owner_reader_calls':calls,'slots':expected,'LLM_sampler_keys':expected//18*15,'actual_RNG':0})
@pytest.mark.parametrize('phase',['main','holdout'])
@pytest.mark.parametrize('target',['source','dataset_case','dataset_revocation','execution_revocation','dataset_execution_order'])
def test_latest_snapshot_failure_controls(tmp_path,phase,target):
 f=x.ExecutionFixture(tmp_path);refs,execution,opening=prepare(f,phase);original=f.verification.owner_reader;calls=0;last=5 if phase=='main' else 9
 def read(run):
  nonlocal calls
  calls+=1
  if calls==last:
   if target=='source':(f.project/'eval/inspection_formal_execution.py').write_text('# changed synthetic qualified source\n')
   elif target=='dataset_case':
    manifest=f.read(refs[1]);path=f.run/manifest['records'][0]['case']['path'];path.write_bytes(path.read_bytes()+b' ')
   elif target in ('dataset_revocation','execution_revocation'):
    finding=x.helper.module.DATASET_FINDINGS[phase] if target=='dataset_revocation' else m.EXECUTION_FINDINGS[phase]
    f.bind('synthetic-independent-revoke-'+target,[f.catalog],revises=finding)
   else:
    names=[x.helper.module.DATASET_FINDINGS[phase],m.EXECUTION_FINDINGS[phase]]
    indices=[next(i for i,v in enumerate(f.owner['rounds']) if v['decision']['finding']['id']==n) for n in names]
    a,b=indices;f.owner['rounds'][a],f.owner['rounds'][b]=f.owner['rounds'][b],f.owner['rounds'][a];f.save()
  return original(run)
 with pytest.raises(ValueError) as e:f.execution_proof(refs,execution,opening,replace(f.verification,owner_reader=read))
 assert calls==last
 obs.append({'probe':'latest native snapshot control','phase':phase,'target':target,'at_owner_call':calls,'reason':str(e.value)})
@pytest.mark.parametrize('revoked',[False,True])
def test_native_execution_history_blocks_rebuild_after_plan_moved(tmp_path,monkeypatch,revoked):
 f=x.ExecutionFixture(tmp_path);refs,execution,opening=prepare(f,'main');before=(f.run/execution['path']).read_bytes();(f.run/'execution/main.v1').rename(f.run/'retained-primary-plan')
 if revoked:f.bind('synthetic-independent-exec-revoke',[f.catalog],revises=m.EXECUTION_FINDINGS['main'])
 monkeypatch.setattr(p,'episode_schedule',no_rng)
 with pytest.raises(ValueError,match='already issued'):p.prepare_primary_execution(f.run,f.project,f.acceptance,*refs,verification=f.verification)
 assert not(f.run/'execution/main.v1').exists() and (f.run/'retained-primary-plan/manifest.json').read_bytes()==before
 obs.append({'probe':'producer native history','revoked':revoked,'canonical_not_recreated':True,'schedule_sampler_not_called':True})

def test_producer_partial_failure_is_retained_without_retry(tmp_path,monkeypatch):
 f=x.ExecutionFixture(tmp_path);refs=f.dataset();seen=[]
 monkeypatch.setattr(p,'episode_schedule',lambda md,seed:(seen.append(seed) or x.literal_order(md,seed)))
 class Fake:
  def __init__(self,seed):seen.append(seed)
  def integers(self,a,b):assert(a,b)==(0,2**32);return 100
 monkeypatch.setattr(x.recording.np.random,'PCG64',lambda seed:('FakePCG',seed));monkeypatch.setattr(x.recording.np.random,'Generator',Fake)
 old=p.durable_once
 def write(path,value):
  if path.name=='sampler-seeds.json':raise OSError('injected sampler file write failure')
  return old(path,value)
 monkeypatch.setattr(p,'durable_once',write)
 with pytest.raises(OSError):p.prepare_primary_execution(f.run,f.project,f.acceptance,*refs,verification=f.verification)
 folder=f.run/'execution/main.v1';assert(folder/'episode-order.json').exists();assert(folder/'preparation-failure.json').exists();assert not(folder/'manifest.json').exists()
 before=list(seen)
 with pytest.raises(ValueError):p.prepare_primary_execution(f.run,f.project,f.acceptance,*refs,verification=f.verification)
 assert seen==before
 obs.append({'probe':'producer partial IO failure','order_and_failure_retained':True,'no_manifest':True,'retry_no_new_Fake_calls':True})
@pytest.mark.parametrize('phase',['main','holdout'])
def test_sampler_expression_matches_existing_pilot_algorithm_with_Fake_only(tmp_path,monkeypatch,phase):
 f=x.ExecutionFixture(tmp_path);opening=f.typed_main_opening(f.dataset()[1]) if phase=='holdout' else None;refs=f.dataset(phase,opening);intent=f.read(refs[0]);seen=[]
 class Fake:
  def __init__(self,seed):self.seed=seed;self.n=0
  def integers(self,low,high):assert(low,high)==(0,2**32);self.n+=1;return 100000+self.n
 def order(md,seed):seen.append(('schedule',seed));return x.literal_order(md,seed)
 monkeypatch.setattr(p,'episode_schedule',order);monkeypatch.setattr(x.recording.np.random,'PCG64',lambda seed:seed);monkeypatch.setattr(x.recording.np.random,'Generator',Fake)
 ref=p.prepare_primary_execution(f.run,f.project,f.acceptance,*refs,verification=f.verification,holdout_opening_value=opening)
 value=f.read(ref);actual=f.read(value['sampler_seeds'])['by_order_index'];schedule=f.read(value['episode_order'])
 # Execute only the original pilot's seed-map assignment with a literal Fake RNG.
 tree=ast.parse((REPO/'eval/run_inspection_eight_condition_pilot.py').read_text())
 node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='seeds' for t in n.targets))
 env={'rng':Fake(intent['sampler_master_seed']),'schedule':schedule,'CONDITION_REPEATS':x.recording.CONDITION_REPEATS}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<original pilot sampler assignment>','exec'),env)
 assert actual=={str(k):v for k,v in env['seeds'].items()}
 assert len(actual)==(1080 if phase=='main' else 180);assert seen==[('schedule',intent['schedule_seed'])]
 obs.append({'probe':'sampler algorithm original-pilot expression','phase':phase,'same_keys_values_and_iteration_order':True,'sampler_calls':len(actual),'schedule_master':intent['schedule_seed'],'sampler_master':intent['sampler_master_seed'],'actual_RNG':0})
def teardown_module():Path('/tmp/formal-execution-independent-probes-v1.json').write_text(json.dumps(obs,indent=2)+'\n')
