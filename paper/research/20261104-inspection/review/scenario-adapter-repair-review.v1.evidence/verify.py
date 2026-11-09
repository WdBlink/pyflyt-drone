"""Read-only v2 admission/lineage checks, no physical world construction."""
from pathlib import Path
import json,hashlib,math
from eval.inspection_cases import CaseSpec,validate_case,generate_dev_case,CaseEntity
ROOT=Path.cwd();R=ROOT/'paper/research/20261104-inspection';Q=R/'qualification';old=json.loads((R/'review/scenario-adapter-review.v1.evidence/admission.json').read_text());out={}
for name in ['coincident_ten_dynamic','roi_aliasing','boolean_numeric_and_nonstring_identity','valid_disjoint_collinear_polygon']:
 try:c=CaseSpec.from_dict(old[name]['case']);v=validate_case(c);out[name]={'accepted':True,'validation':v}
 except Exception as e:out[name]={'accepted':False,'type':type(e).__name__,'reason':str(e)}
assert all(not out[n]['accepted'] for n in ['coincident_ten_dynamic','roi_aliasing','boolean_numeric_and_nonstring_identity']) and out['valid_disjoint_collinear_polygon']['accepted']
m1=json.loads((Q/'dev-cases.v1/manifest.json').read_text());m2=json.loads((Q/'dev-cases.v2/manifest.json').read_text());records=[];c1s={}
for r in m2['records']:
 a=json.loads((Q/'dev-cases.v1'/r['file']).read_text());b=json.loads((Q/'dev-cases.v2'/r['file']).read_text());c=CaseSpec.from_dict(b);v=validate_case(c);c1=CaseSpec(**{**a,'roi_polygon':tuple(tuple(p) for p in a['roi_polygon']),'wind_xy':tuple(a['wind_xy']),'entities':tuple(CaseEntity(**{**e,'origin_xy':tuple(e['origin_xy'])}) for e in a['entities'])});c1s[c1.case_id]=c1
 try:validate_case(c1);old_ok=True;reason=None
 except Exception as e:old_ok=False;reason=str(e)
 records.append({'case_id':c.case_id,'lineage_seed_level_same':all(a[k]==b[k] for k in ['case_id','lineage_group','seed','level']),'bytes_same':(Q/'dev-cases.v1'/r['file']).read_bytes()==(Q/'dev-cases.v2'/r['file']).read_bytes(),'file_sha_match':hashlib.sha256((Q/'dev-cases.v2'/r['file']).read_bytes()).hexdigest()==r['file_sha256'],'digest_match':c.digest()==r['case_sha256'],'validation_match':v==r['validation'],'old_valid_under_new_rule':old_ok,'old_invalid_reason':reason})
changed=[r['case_id'] for r in records if not r['bytes_same']];assert changed==['dev-L2-03']
a=c1s['dev-L2-03'];regen=generate_dev_case(a.case_id,a.level,a.seed);target=CaseSpec.from_dict(json.loads((Q/'dev-cases.v2/dev-L2-03.json').read_text()));assert regen==target
# Old invalid layout has an actual sub-1m pair, independently sampled at .1s.
best=(float('inf'),None,None,None)
for i,e in enumerate(a.entities):
 for f in a.entities[i+1:]:
  if e.object_type==f.object_type=='static':continue
  start=max(e.appeared_at,f.appeared_at)
  for k in range(math.ceil((600-start)*10)+1):
   t=min(600,start+k/10);d=math.dist(e.position(t),f.position(t))
   if d<best[0]:best=(d,e.key,f.key,t)
out['suite']={'records':records,'changed':changed,'same_seed_regeneration_exact':regen==target,'same_master_seed':m1['master_seed']==m2['master_seed'],'root_count':len(records),'all_24_bindings':all(r['file_sha_match'] and r['digest_match'] and r['validation_match'] and r['lineage_seed_level_same'] for r in records),'pilot_v1':[r['case_id'] for r in m1['records'] if r['first_pilot']],'pilot_v2':[r['case_id'] for r in m2['records'] if r['first_pilot']],'original_L2_03_min_pair_0p1s':best,'parent_manifest_match':m2['parent_manifest_sha256']==hashlib.sha256((Q/'dev-cases.v1/manifest.json').read_bytes()).hexdigest()}
assert out['suite']['all_24_bindings'] and out['suite']['pilot_v1']==out['suite']['pilot_v2']
files=['eval/inspection_cases.py','eval/inspection_scenarios.py','eval/run_inspection_reference.py','tests/test_inspection_cases.py','envs/inspection/mission.py','envs/inspection/scoring.py'];out['source_hashes']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files};out['input_manifests']={v:hashlib.sha256((Q/f'dev-cases.{v}/manifest.json').read_bytes()).hexdigest() for v in ['v1','v2']}
archive=Q/'source/executor-v19-reference-v2';out['env_core_vs_v19']={str(p):p.read_bytes()==(archive/p).read_bytes() for p in Path('envs/inspection').glob('*.py')};assert all(out['env_core_vs_v19'].values())
Path('/tmp/scenario-adapter-repair-review/verify.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['suite','env_core_vs_v19']},indent=2));print('suite',json.dumps({k:v for k,v in out['suite'].items() if k!='records'}));print('env-core',len(out['env_core_vs_v19']),'all_same',all(out['env_core_vs_v19'].values()))
