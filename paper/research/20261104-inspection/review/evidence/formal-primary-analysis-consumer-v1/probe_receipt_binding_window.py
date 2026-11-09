"""Synthetic real-SDK fixture, no actual receipt/owner/model/world; targeted trust boundary."""
from pathlib import Path
import importlib.util,sys,json,hashlib,tempfile
import pytest
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';W=R/'planning/formal-admission-candidate.v3';F=W/'primary-analysis-freeze.v2';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((F/'binding.json').read_text());assert sha(F/'binding.json')=='61f4f22c821201f422721fae4289f660c9adbc4fb6702a5b6fb0852a2a43f7ff'
for n,v in b['members'].items():assert sha(F/n)==v['sha256'] and sha(W/n)==v['sha256']
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_analysis_fixture',W/'tests/test_formal_analysis.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
results=[]
for fault in ['validated_plan_changed_before_first_capture','terminal_changed_after_validated_return','world_changed_after_statistic_start']:
 patch=pytest.MonkeyPatch();project=Path(tempfile.mkdtemp(prefix='plan-binding-',dir=OUT)).resolve()
 try:
  f,request,proof,recording,measure=h.prepared_recording(project,patch)
  target=f.project/'logs/inspection-formal-episodes.v1/main'/f'{proof.slots[0].case_id}-{proof.slots[0].condition}-{proof.slots[0].repeat_ordinal}'
  plan=target/'plan.json';terminal=target/'attempt-00/terminal.json'
  world=target/'attempt-00/flight/world.json';world.parent.mkdir(parents=True);world.write_text('{"role":"original synthetic world"}\n')
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
  subject=h.analysis.analyze_primary(request,recording,project_root=f.project)
  refs=json.loads((f.run/'analysis/main.v1/recording-index.json').read_text())['physical_plan_and_terminal_sha256']
  after={'plan':sha(plan),'terminal':sha(terminal),'world':sha(world)}
  assert changed and all(before[k]!=after[k] for k in changed)
  if fault.startswith('validated_plan'):assert refs[str(plan.relative_to(f.project))]==after['plan']
  elif fault.startswith('terminal'):assert refs[str(terminal.relative_to(f.project))]==after['terminal']
  else:assert str(world.relative_to(f.project)) not in refs
  results.append({'fault':fault,'analysis_returned_subject':True,'outputs_written':(f.run/'analysis/main.v1/subject.json').exists(),'before_sha256':before,'after_sha256':after,'mutated':changed,'fresh_sha_accepted_without_readmission':True,'world_not_in_final_receipt_ref_set':str(world.relative_to(f.project)) not in refs,'scope':'synthetic Fake validated-return boundary, real SDK full1296 grid; no actual formal data/owner/physics/model'})
 finally:patch.undo()
(OUT/'receipt-binding-window-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
