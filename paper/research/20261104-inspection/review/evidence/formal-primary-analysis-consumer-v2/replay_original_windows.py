"""Synthetic real-SDK fixture, no actual receipt/owner/model/world; targeted trust boundary."""
from pathlib import Path
import importlib.util,sys,json,hashlib,tempfile
import pytest
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';W=R/'planning/formal-admission-candidate.v3';F=W/'primary-analysis-freeze.v3';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((F/'binding.json').read_text());assert sha(F/'binding.json')=='e24c021d1ddf297712c756bab2a06d2fe9c0eb3a4d67bb2c8f76e5b6d276186e'
for n,v in b['members'].items():assert sha(F/n)==v['sha256'] and sha(W/n)==v['sha256']
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_analysis_fixture',W/'tests/test_formal_analysis.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
results=[]
for fault in ['validated_plan_changed_before_first_capture','terminal_changed_after_validated_return','world_changed_after_statistic_start']:
 patch=pytest.MonkeyPatch();project=Path(tempfile.mkdtemp(prefix='plan-binding-',dir=OUT)).resolve()
 try:
  f,request,proof,recording,measure=h.prepared_recording(project,patch)
  target=f.project/'logs/inspection-formal-episodes.v1/main'/f'{proof.slots[0].case_id}-{proof.slots[0].condition}-{proof.slots[0].repeat_ordinal}'
  plan=target/'plan.json';terminal=target/'attempt-00/terminal.json'
  world=target/'attempt-00/flight/world.json';world.parent.mkdir(parents=True,exist_ok=True);world.write_text('{"role":"original synthetic world"}\n')
  before={'plan':sha(plan),'terminal':sha(terminal),'world':sha(world)};changed=[]
  def revalidate(directory,expected):
   result=measure(directory,expected)
   if Path(directory)==target and not changed:
    if fault.startswith('validated_plan'):
     value=json.loads(plan.read_text());value['case_sha256']='0'*64;plan.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');changed.append('plan')
    elif fault.startswith('terminal'):
     value=json.loads(terminal.read_text());value['status']='unknown_model_usage';terminal.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');changed.append('terminal')
   return result
  patch.setattr(h.analysis,'measure_completed_episode',revalidate)
  def statistics(*args,**kwargs):
   if fault.startswith('world'):
    world.write_text('{"role":"changed after all successful collection"}\n');changed.append('world')
   return {'role':'fixed software substitute; no numeric inference/RNG needed for binding counterexample'}
  patch.setattr(h.analysis,'analyze_roots',statistics);patch.setattr(h.analysis,'sr_cluster_intervals',lambda *a,**k:{'role':'fixed software substitute'})
  try:h.analysis.analyze_primary(request,recording,project_root=f.project)
  except ValueError as error:reason=str(error);assert 'closed receipt input changed' in reason
  else:raise AssertionError('original counterexample was still accepted')
  after={'plan':sha(plan),'terminal':sha(terminal),'world':sha(world)}
  assert changed and all(before[k]!=after[k] for k in changed)
  destinations=[f.run/'analysis/main.v1',f.run/'experiment/main-recording.v1',f.run/'experiment/results.main.v1.json']
  assert not any(x.exists() for x in destinations)
  results.append({'fault':fault,'rejected':True,'reason':reason,'all_analysis_destinations_absent':True,'before_sha256':before,'after_sha256':after,'mutated':changed,'scope':'same original synthetic Fake validated-return counterexample, actual SDK grid; no model/physics/native state'})
 finally:patch.undo()
(OUT/'receipt-binding-window-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
