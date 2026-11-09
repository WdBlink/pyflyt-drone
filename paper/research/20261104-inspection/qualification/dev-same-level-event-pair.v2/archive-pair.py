from pathlib import Path
import hashlib,json,zipfile,shutil,sys
sys.path.insert(0,str(Path.cwd()))
from eval.inspection_artifacts import archive_source_bundle
from eval.inspection_population import load_development_case
from eval.inspection_common_calendar import rescore_common_calendar
r=Path('paper/research/20261104-inspection');d=r/'qualification/dev-same-level-event-pair.v2'
roots={k:Path('logs/inspection-same-level-'+label+'-dev-flights.v2/devext-OOD-L3-00-B0-0') for k,label in [('event','event'),('clean','clean')]}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
worlds={};runs={};summary={}
for label,root in roots.items():
 assert len(list(root.glob('attempt-*')))==1
 a=root/'attempt-00';w=json.loads((a/'flight/world.json').read_text());run=json.loads((a/'flight/run.json').read_text());t=json.loads((a/'terminal.json').read_text());assert t['status']=='completed' and w['status']=='completed' and w['simulated_seconds']==600 and w['physics_steps_executed']==144000 and w['max_clock_lag_seconds']<=.1
 assert run['sources_unchanged'] and run['dependencies_unchanged'] and run['private_case_unchanged'];worlds[label]=w;runs[label]=run
 summary[label]={'root':str(root.resolve()),'actual_attempts':1,'terminal_sha256':sha(a/'terminal.json'),'run_sha256':sha(a/'flight/run.json'),'world_sha256':sha(a/'flight/world.json'),'simulated_seconds':w['simulated_seconds'],'physics_steps':w['physics_steps_executed'],'max_clock_lag_seconds':w['max_clock_lag_seconds'],'score':{k:w['score'][k] for k in ('C','S','D','Q','H','E','Y','SR')}}
assert runs['event']['source_hashes']==runs['clean']['source_hashes'] and runs['event']['dependency_hashes']==runs['clean']['dependency_hashes']
assert runs['event']['runtime_thread_environment']==runs['clean']['runtime_thread_environment']
assert worlds['clean']['public_events']==[]
parent=load_development_case(worlds['event']['private_case']);child=load_development_case(worlds['clean']['private_case']);assert parent.level==child.level=='L3' and child.parent_sha256==parent.digest()
common=rescore_common_calendar(worlds['clean'],parent);(d/'clean-on-original-demand-calendar.json').write_text(json.dumps(common,indent=2)+'\n')
archive_source_bundle(runs['event'],d/'flight-source')
z=d/'complete-reference-pair.zip'
with zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED) as a:
 for label,root in roots.items():
  for p in sorted(root.rglob('*')):
   if p.is_file():a.write(p,Path(label)/p.relative_to(root))
shutil.copy2('eval/replay_inspection_reference.py',d/'execution-replayer.py')
record={'role':'one source-identical same-level event/clean development reference pair; no formal or population-level causal inference','flights':summary,'source_file_count':len(runs['event']['sources']),'dependency_file_count':len(runs['event']['dependency_hashes']),'same_source_and_dependencies':True,'parent_case_sha256':parent.digest(),'clean_case_sha256':child.digest(),'initial_public_level_both':'L3','initial_card_proof':'binding.json and initial-public-task-card.json actual pre-dispatch constructor reads','explicit_intervention':'event-clean-sibling-v1 plus complete parent and parentSHA; only nine event fields removed','clean_no_actual_public_events':True,'legacy_world_events_metadata':'Both true because derived from L3; not activation or analysis input. Use explicit calendar/intervention/journal.','common_demand_clean_score':{k:common['score'][k] for k in ('C','S','D','Q','H','E','Y','SR')},'event_minus_common_demand_clean_Y':worlds['event']['score']['Y']-common['score']['Y'],'archive_sha256':sha(z),'replayer_sha256':sha(d/'execution-replayer.py'),'replay_status':'pending separate actual execution receipts','limits':'One root and one policy; retain original low event score and all prior-generation pairs. This is not full G4 or actual108slotpilot.'}
(d/'flight-pair-before-replay.json').write_text(json.dumps(record,indent=2)+'\n');shutil.copy2(__file__,d/'archive-pair.py');print(json.dumps({k:record[k] for k in ['source_file_count','dependency_file_count','event_minus_common_demand_clean_Y','archive_sha256']}))
