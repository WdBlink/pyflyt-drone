from pathlib import Path
import hashlib,json,shutil,sys
sys.path.insert(0,str(Path.cwd()))
from eval.inspection_population import load_development_case,event_clean_sibling
from eval.inspection_scenarios import ScenarioMission
r=Path('paper/research/20261104-inspection');dest=r/'qualification/dev-same-level-event-pair.v2';dest.mkdir()
parentpath=r/'qualification/dev-population.v1/devext-OOD-L3-00.json';parent=load_development_case(json.loads(parentpath.read_text()));clean=event_clean_sibling(parent)
cards=[]
for label,case in [('event',parent),('clean',clean)]:
 m=ScenarioMission(case,start_planner=False)
 try:cards.append(m.public_state()['task_card'])
 finally:m.close()
 d=dest/label;d.mkdir();p=d/(case.case_id+'.json');p.write_text(json.dumps(case.as_dict(),indent=2)+'\n')
 row={'case_id':case.case_id,'lineage_group':case.lineage_group,'file':p.name,'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'case_sha256':case.digest()}
 (d/'manifest.json').write_text(json.dumps({'split':'dev','main_and_holdout_instantiated':False,'role':'same-root event intervention qualification; not new independent roots','records':[row]},indent=2)+'\n')
assert cards[0]==cards[1] and cards[0]['level']=='L3'
source_names=list(json.loads((r/'qualification/public-contract-clean-candidate.v2/source-binding.json').read_text())['sources'])
binding={'role':'one prospective same-level explicit event-removal qualification pair; both run under same sources, no new root draw','parent_source_path':str(parentpath),'parent_file_sha256':hashlib.sha256(parentpath.read_bytes()).hexdigest(),'parent_case_sha256':parent.digest(),'clean_case_sha256':clean.digest(),'initial_public_task_cards_equal':True,'initial_public_level':'L3','source_hashes':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_names},'events_removed_only':True,'known_limits':'Two declared600s B0 flights required; these initial t0 constructor reads are not flight qualification. No model or main/holdout.'}
(dest/'binding.json').write_text(json.dumps(binding,indent=2)+'\n');(dest/'initial-public-task-card.json').write_text(json.dumps(cards[0],indent=2)+'\n');shutil.copy2(__file__,dest/'prepare.py')
print(json.dumps(binding,indent=2))
