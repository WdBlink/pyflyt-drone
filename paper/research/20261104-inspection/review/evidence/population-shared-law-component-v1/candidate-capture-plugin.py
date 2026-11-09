"""Capture already-authorized five dev replays; forbid every other SeedSequence."""
from pathlib import Path
import json
import numpy as np
RUN=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection')
manifest=json.loads((RUN/'qualification/dev-population.v1/manifest.json').read_text())
allowed={x['root_generation_seed'] for x in manifest['records']}
original_seedsequence=np.random.SeedSequence
seed_calls=[];records=[]
def checked_seedsequence(seed,*a,**k):
 if type(seed)is not int or seed not in allowed:raise AssertionError('new or formal root draw forbidden by independent review')
 seed_calls.append(seed);return original_seedsequence(seed,*a,**k)
def pytest_sessionstart(session):np.random.SeedSequence=checked_seedsequence
def pytest_runtest_setup(item):
 if not item.name.startswith('test_existing_five_development_draws_and_all_rejections_are_exactly_preserved'):return
 old=item.module.previous;new=item.module.candidate
 row={'test':item.nodeid,'old':[],'new':[]};state=[]
 for label,module in [('old',old),('new',new)]:
  orig=module.validate_population_case
  def capture(case,_orig=orig,_label=label):
   d={'candidate_semantic_sha256':case.digest()}
   try:v=_orig(case)
   except ValueError as e:
    d.update(status='rejected',reason=str(e));row[_label].append(d);raise
   else:d.update(status='accepted',validation=v);row[_label].append(d);return v
  module.validate_population_case=capture;state.append((module,orig))
 item._independent_population_capture=(row,state)
def pytest_runtest_teardown(item,nextitem):
 if not hasattr(item,'_independent_population_capture'):return
 row,state=item._independent_population_capture
 for module,orig in state:module.validate_population_case=orig
 assert row['old']==row['new'],'candidate-level geometry or validation changed'
 assert row['old'] and row['old'][-1]['status']=='accepted'
 row['candidate_count']=len(row['old']);row['candidate_vectors_exactly_equal']=True;records.append(row)
def pytest_sessionfinish(session,exitstatus):
 np.random.SeedSequence=original_seedsequence
 result={'role':'only five already-persisted dev roots replayed; zero actual formal draw','pytest_exitstatus':exitstatus,'allowed_existing_root_seeds':sorted(allowed),'root_seed_calls':seed_calls,'old_and_new_candidate_count':sum(x['candidate_count'] for x in records),'per_root':records}
 Path('/tmp/population-component-independent-candidate-capture.v1.json').write_text(json.dumps(result,indent=2)+'\n')
