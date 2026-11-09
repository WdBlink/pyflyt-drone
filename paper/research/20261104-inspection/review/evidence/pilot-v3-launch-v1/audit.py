"""Read-only stdlib validation of existing v3 preparation; never imports runners/RNG."""
from pathlib import Path
import ast,collections,hashlib,json,zipfile
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
R=REPO/'paper/research/20261104-inspection';P=REPO/'logs/inspection-eight-condition-pilot-preparation.v3';Q=R/'qualification/pilot-source-preparation.v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
b=read(Q/'binding.json');assert sha(Q/'binding.json')=='aeb15ca38f736b1aa4fc31c05d6e71a61c8b690b89824c8ef25af8401f2e7413'
assert sha(R/b['archive']['path'])==b['archive']['sha256']
with zipfile.ZipFile(R/b['archive']['path']) as z:
 assert len(z.namelist())==len(set(z.namelist()))==7
 assert set(z.namelist())=={Path(n).name for n in b['original_files']}
 for n,h in b['original_files'].items():
  p=REPO/n;assert sha(p)==h and z.read(p.name)==p.read_bytes()
sources=read(P/'source-hashes.json');texts=read(P/'source-files.json');deps=read(P/'dependency-hashes.json');inputs=read(P/'input-hashes.json')
assert len(sources)==103 and len(deps)==61 and len(inputs)==10 and set(sources)==set(texts)
for group in (sources,deps,inputs):
 for p,h in group.items():assert sha(p)==h,p
for p,t in texts.items():assert hashlib.sha256(t.encode()).hexdigest()==sources[p] and Path(p).read_text()==t
owner=read(R/'.paper/control.json');plans=[x['packet'] for x in owner['receipts'] if x.get('stage')=='plan'];assert len(plans)==1
assert owner['frozen_plan']==hashlib.sha256(json.dumps(plans[0],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert owner['active_round'] and owner['task_stop'] is None
plan=plans[0];design=read(R/plan['experiment_design']);assert sha(R/plan['experiment_design'])==plan['experiment_design_sha256']
assert sha(R/plan['local_model_registry'])==plan['local_model_registry_sha256']
evaluator=(R/design['evaluator']['path']).resolve();assert sha(evaluator)==design['evaluator']['sha256']
closure=evaluator.parent/'evaluator-source.v2';cm=evaluator.parent/'evaluator-source-manifest.v2.json';cs=read(cm)['sources']
assert sha(cm)=='ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80'
assert {str(p.relative_to(closure)) for p in closure.rglob('*') if p.is_file()}==set(cs) and not any(p.is_symlink() for p in closure.rglob('*'))
assert len(cs)==17
for n,h in cs.items():assert sha(closure/n)==h
sdk=R/'runtime/autoresearch-state-reader-v1';tree=ast.parse((REPO/'eval/run_inspection_eight_condition_pilot.py').read_text());modules=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='modules' for t in n.targets))
migrations=[x for x in owner['history'] if x.get('event')=='round_documents_runtime_migrated'];tr=migrations[-1]['transport'];tp=R/tr['path'];assert sha(tp)==tr['sha256'];transport=read(tp)
expected=[*sorted((REPO/'envs/inspection').glob('*.py')),REPO/'envs/__init__.py',*[REPO/'eval'/x for x in modules],REPO/'eval/run_inspection_eight_condition_pilot.py',*sorted((sdk/'lib').glob('*.py')),evaluator,cm,*[closure/n for n in sorted(cs)],*[R/n for n in transport['sources']],tp]
assert set(sources)=={str(p.resolve()) for p in expected}
for n,h in transport['sources'].items():assert sha(R/n)==h
pkg=REPO/'.venv/lib/python3.11/site-packages/PyFlyt';assets={str(p) for p in pkg.rglob('*') if p.suffix.lower() in ('.py','.yaml','.urdf','.obj','.stl','.dae')};assert len(assets)==60 and set(deps)-assets=={str(next((pkg.parent).glob('pybullet*.so')))}
s=read(P/'prepared-episode-order.json');seed=read(P/'sampler-seeds.json');prep=read(P/'preparation.json');repeat={'B0':1,'B1':1,'B2':1,'B3-M1':3,'B3-M2':3,'B3-M3':3,'B4-M1':3,'B4-M2':3}
cases=[f'dev-L{l}-{i:02d}' for l in (1,2,3) for i in (0,1)];slots=s['slots']
assert s['seed']==2026110621 and seed['master']==2026110622 and len(slots)==108 and s['rng']==seed['rng']=='NumPy PCG64'
expected_slots={(c,k,a) for c in cases for a,n in repeat.items() for k in range(n)}
assert len({(x['case_id'],x['repeat_ordinal'],x['condition']) for x in slots})==108
assert {(x['case_id'],x['repeat_ordinal'],x['condition']) for x in slots}==expected_slots
assert [x['order_index'] for x in slots]==list(range(108))
for block in range(6):
 part=slots[18*block:18*(block+1)];assert len({x['case_id'] for x in part})==1 and {x['root_block_index'] for x in part}=={block}
 for x in part:assert x['domain']=='ID' and x['level']==int(x['case_id'][5])
llm={str(x['order_index']) for x in slots if repeat[x['condition']]==3};assert set(seed['by_order_index'])==llm and len(llm)==90
assert all(type(v) is int and 0<=v<2**32 for v in seed['by_order_index'].values())
assert prep['episode_denominator']==108 and prep['conditions']==repeat and prep['actual_model_generation_calls']==prep['actual_physical_flights']==0
sel=next(Path(x) for x in inputs if x.endswith('/selection.json'));assert sha(sel)=='75f57580f0f5371d91a741fd4f9573e52192d019e9730335123a7ae67897a32e'
selection=read(sel);assert selection['completed_slots']==18 and selection['root_count']==6 and selection['classic_best']=='B1'
suite=next(Path(x) for x in inputs if x.endswith('/dev-cases.v2/manifest.json'));manifest=read(suite);rows=[x for x in manifest['records'] if x.get('first_pilot') is True]
assert manifest['split']=='dev' and manifest['main_and_holdout_instantiated'] is False and sorted(x['case_id'] for x in rows)==cases and len({x['lineage_group'] for x in rows})==6
for row in rows:assert sha(suite.parent/row['file'])==row['file_sha256'] and str((suite.parent/row['file']).resolve()) in inputs
old=REPO/'logs/inspection-eight-condition-dev-flights.v2/dev-L1-00-B4-M1-2';old_terms=[]
for k in range(2):
 p=old/f'attempt-{k:02d}'/'terminal.json';v=read(p);assert v['status']=='known_operational_failure' and v['physical_validation']['failure_reasons']==['clock_lag_exceeded'];old_terms.append({'path':str(p.relative_to(REPO)),'sha256':sha(p),'status':v['status'],'max_lag':v['physical_validation']['max_clock_lag_seconds']})
assert not (old/'attempt-02').exists()
oldplan=read(old/'plan.json');old_delta={p:{'v2':h,'v3':sources.get(p)} for p,h in oldplan['source_hashes'].items() if sources.get(p)!=h}
clockq=R/'qualification/clock-localization-adaptive-L2-postinventory.v1/freeze.v1';clock=read(clockq/'binding.json');print('clockkeys',list(clock))
prefix=str(clockq.relative_to(R))+'/source/'
clock_sources={str((REPO/k[len(prefix):]).resolve()):h for k,h in clock['files'].items() if k.startswith(prefix)}
assert set(sources)<=set(clock_sources) and all(clock_sources[p]==h for p,h in sources.items())
report=R/'review/postinventory-clock-evidence-review.v1.md'
result={'preparation_binding_sha256':sha(Q/'binding.json'),'archive_sha256':sha(Q/'prepared-source-order-only.zip'),'source_count':len(sources),'dependency_count':len(deps),'input_count':len(inputs),'source_text_current_all_equal':True,'exact_expected_source_set':True,'exact_dependency_inventory':True,'evaluator_closure_members':17,'owner_transport_sources':len(transport['sources']),'sdk_lib_sources':len(list((sdk/'lib').glob('*.py'))),'driver_sha256':sha(REPO/'eval/run_inspection_eight_condition_pilot.py'),'schedule_sha256':sha(P/'prepared-episode-order.json'),'schedule_seed':s['seed'],'sampler_master':seed['master'],'slots':108,'classic_slots':18,'llm_slots':90,'root_means':48,'root_count':6,'conditions':dict(collections.Counter(x['condition'] for x in slots)),'root_block_order':[slots[i*18]['case_id'] for i in range(6)],'stored_sampler_keys':90,'sampler_draw_not_reexecuted':True,'classic_selection_sha256':sha(sel),'classic_best':'B1','native_current_active_round':owner['active_round']['id'],'native_implementation_sha256':owner['implementation_sha256'],'native_plan_sha256':owner['frozen_plan'],'native_usage_statuses':dict(collections.Counter(x['status'] for x in owner['usage_ledger'].values())),'old_v2_failed_attempts':old_terms,'old_v2_attempt02_absent':True,'old_v2_llm_source_deltas':old_delta,'postinventory_report_sha256':sha(report),'all_103_sources_match_postinventory_freeze':True,'current_formal_entry_installed':(REPO/'eval/run_inspection_formal.py').exists(),'production_source_hashes':sources,'dependency_hashes':deps,'input_hashes':inputs,'preparation_file_hashes':b['original_files']}
Path('/tmp/inspection-pilot-v3-launch-review/recount.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('production_source_hashes','dependency_hashes','input_hashes','preparation_file_hashes')},indent=2))
