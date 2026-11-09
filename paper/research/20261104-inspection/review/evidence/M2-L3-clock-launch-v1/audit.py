"""Source-only M2/L3 diagnostic readiness; no imports/effect/RNG."""
from pathlib import Path
import ast,difflib,hashlib,json
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';P=R/'planning/clock-localization-M2-L3.v1';Q=R/'qualification/clock-localization-M2-L3.v1/freeze.v1';OUT=Path('/tmp/inspection-M2-L3-clock-launch-review')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
b=read(Q/'binding.json');intent=read(P/'intent.json');assert sha(Q/'binding.json')=='b6aa5715bf4855db70f5eff2fcc5f1b954951f5722ff27479b1b4d539e87e068';assert sha(P/'intent.json')=='394d416179cd77c1188762e4875653f23127e837224af29a45f898c8e320a902'
assert len(b['files'])==111 and len(b['live_source_hashes'])==107 and len(b['dependencies'])==61 and len(b['input_hashes'])==11
for p,h in b['files'].items():assert sha(Q/p)==h,p
for group in ('live_source_hashes','dependencies','input_hashes'):
 for p,h in b[group].items():assert sha(p)==h,p
old=REPO/'eval/probe_inspection_adaptive_clock.py';oldtext=old.read_text();newtext=(P/'probe.py').read_text();replacements=[('One prospective, unscored B4-M1 development clock diagnostic.','One prospective, unscored B3-M2 development L3 clock diagnostic.'),("ROOT=Path(__file__).resolve().parents[1]","ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent/'envs/inspection/mission.py').is_file())"),("ROLE='instrumented_adaptive_B4_M1_new_development_case_not_pilot_retry'","ROLE='instrumented_adaptive_B3_M2_existing_development_L3_not_pilot_retry'"),("CASE='qualification/dev-cases.v2/dev-L2-00.json'","CASE='qualification/dev-cases.v2/dev-L3-00.json'"),('SAMPLER_SEED=2026110616','SAMPLER_SEED=1096691202'),("EXTRA_SOURCES=('eval/probe_inspection_adaptive_clock.py'","EXTRA_SOURCES=('paper/research/20261104-inspection/planning/clock-localization-M2-L3.v1/probe.py'"),("['models']['M1']","['models']['M2']"),("'--model','M1','--architecture','B4'","'--model','M2','--architecture','B3'")]
normalized=oldtext
for a,c in replacements:assert normalized.count(a)==1;normalized=normalized.replace(a,c)
assert normalized==newtext
(OUT/'harness-8-change.diff').write_text(''.join(difflib.unified_diff(oldtext.splitlines(True),newtext.splitlines(True),fromfile='production/prior-M1-L2-harness',tofile='planning/M2-L3-probe')))
t=ast.parse(newtext);literals={}
for n in t.body:
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
  try:literals[n.targets[0].id]=ast.literal_eval(n.value)
  except (ValueError,TypeError):pass
assert literals['ROLE']==intent['role'] and literals['CASE']==intent['case']['path'] and literals['SAMPLER_SEED']==intent['sampler_seed']==1096691202
assert next(p for p in (P/'probe.py').resolve().parents if (p/'envs/inspection/mission.py').is_file())==REPO
extra={str(REPO/p):sha(REPO/p) for p in literals['EXTRA_SOURCES']};assert extra==intent['extra_sources'] and len(extra)==6
pilot=read(REPO/'logs/inspection-eight-condition-pilot-preparation.v3/source-hashes.json');assert b['live_source_hashes']=={**pilot,**extra}
assert b['dependencies']==read(REPO/'logs/inspection-eight-condition-pilot-preparation.v3/dependency-hashes.json')
assert b['input_hashes']=={**read(REPO/'logs/inspection-eight-condition-pilot-preparation.v3/input-hashes.json'),str(REPO/'logs/inspection-eight-condition-pilot-preparation.v3/sampler-seeds.json'):sha(REPO/'logs/inspection-eight-condition-pilot-preparation.v3/sampler-seeds.json')}
seeds=read(REPO/'logs/inspection-eight-condition-pilot-preparation.v3/sampler-seeds.json');assert intent['sampler_seed']==seeds['by_order_index']['0'] and seeds['master']==2026110622
assert sha(R/intent['case']['path'])==intent['case']['sha256'];assert read(R/intent['case']['path'])['split']=='dev'
registry=read(R/'planning/local-model-registry.v1.json');m=registry['models']['M2'];assert sha(R/m['download_receipt'])==m['download_receipt_sha256'];assert m['generation_profile']==intent['request_budget']
post=R/'qualification/clock-localization-adaptive-L2-postinventory.v1/freeze.v1';prior=read(post/'binding.json')['files'];unchanged={}
for n in ['eval/inspection_diagnostic_worker.py','eval/inspection_clock_profile.py','eval/inspection_clock_profile_rusage.py','eval/run_inspection_llm_development.py','eval/run_inspection_reference.py','envs/inspection/control.py','eval/inspection_process_cleanup.py','eval/inspection_darwin_inventory.py']:
 key=str(post.relative_to(R))+'/source/'+n;assert prior[key]==sha(REPO/n);unchanged[n]=sha(REPO/n)
owner=read(R/'.paper/control.json');rid=intent['reservation'];assert rid not in owner['usage_ledger'] and not (REPO/intent['output']).exists() and not (REPO/'logs/inspection-model-resource-operations.v1'/rid).exists() and not (R/'resource-requests'/f'{rid}.reserve.json').exists()
assert intent['failed_pilot']['sha256']==sha(R/intent['failed_pilot']['path'])
result={'freeze_binding_sha256':sha(Q/'binding.json'),'intent_sha256':sha(P/'intent.json'),'probe_sha256':sha(P/'probe.py'),'prior_harness_sha256':sha(old),'frozen_files_verified':111,'live_sources_verified':107,'dependencies_verified':61,'inputs_verified':11,'eight_parameterization_changes_only':True,'working_probe_root_resolves_to_actual_repo':True,'exact_pilot103_union_extra6_is107':True,'exact_extra_sources':extra,'prior_reviewed_runtime_bytes_unchanged':unchanged,'model':'M2','architecture':'B3','case':intent['case'],'sampler_seed':intent['sampler_seed'],'sampler_source':'existing frozen v3 order0; not redrawn','budget':intent['request_budget'],'fresh_output':intent['output'],'new_reservation':rid,'reservation_and_output_absent_at_review':True,'caffeinate_preserved_and_disclosed':True,'no_automatic_retry_or_sdk_admission_in_probe':True,'native_implementation':owner['implementation_sha256'],'active_round':owner['active_round']['id'],'scope':'source-only permission for one unscored engineering localization after native reservation, not G4 or oldslot retry'}
(OUT/'recount.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
