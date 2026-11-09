"""Synthetic /tmp proof integration only; every actual formal random entry forbidden."""
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
R=REPO/'paper/research/20261104-inspection';T=R/'planning/formal-admission-candidate.v3/tests'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
helper=load('independent_population_namespace_fixture_v1',T/'test_namespace_admission.py')
module=load('independent_formal_population_wrapper_v1',T.parent/'source/eval/inspection_formal_population.py')
observations=[]
@pytest.mark.parametrize('kind',['main','holdout','unknown_root','dev_label','missing_seed','boolean_seed','development_collision','holdout_missing_opening'])
def test_actual_proof_wrapper_before_fake_generator(monkeypatch,tmp_path,kind):
 def no_random(*a,**k):raise AssertionError('actual formal RNG forbidden in review')
 monkeypatch.setattr(np.random,'SeedSequence',no_random);monkeypatch.setattr(np.random,'default_rng',no_random)
 f=helper.Fixture(tmp_path);phase='holdout' if kind.startswith('holdout') else 'main'
 opening=f.opening() if phase=='holdout' else None
 value,intent=f.intent(phase,opening);root=value['root_metadata'][0]['case_id'];expected_seed=value['root_generation_seeds'][root]
 if kind=='missing_seed':value['root_generation_seeds'].pop(root)
 elif kind=='boolean_seed':value['root_generation_seeds'][root]=True
 elif kind=='development_collision':value['root_generation_seeds'][root]=7
 intent=f.put(intent['path'],value);f.seal_intent(value,intent)
 seen=[]
 def fake(*args):seen.append(args);return 'synthetic-result-no-formal-geometry',{'role':'fake only'}
 monkeypatch.setitem(sys.modules,'eval.inspection_population',SimpleNamespace(_generate_population_case=fake))
 name='main-ID-L1-99' if kind=='unknown_root' else 'devext-ID-L1-00' if kind=='dev_label' else root
 if kind=='holdout_missing_opening':opening=None
 kwargs=dict(verification=f.verification,holdout_opening_value=opening)
 if kind in ('main','holdout'):
  result,audit=module.generate_formal_population_case(f.run,f.project,f.acceptance,intent,name,**kwargs)
  assert result=='synthetic-result-no-formal-geometry'
  assert seen==[(root,phase,value['root_metadata'][0]['domain'],'L'+str(value['root_metadata'][0]['level']),expected_seed)]
  assert audit['namespace_intent']==intent and audit['native_namespace_request_sha256']
  observations.append({'kind':kind,'fake_args':seen,'actual_formal_RNG':0})
 else:
  with pytest.raises(ValueError) as e:module.generate_formal_population_case(f.run,f.project,f.acceptance,intent,name,**kwargs)
  assert not seen;observations.append({'kind':kind,'error':str(e.value),'actual_formal_RNG':0})
def teardown_module():Path('/tmp/population-independent-namespace-probes-v1.json').write_text(json.dumps(observations,indent=2)+'\n')
