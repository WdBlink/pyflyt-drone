"""Independent bounded R1 repair probes; no draws, writers, flights or owner CLI."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
RUN=REPO/'paper/research/20261104-inspection'
TEST=RUN/'planning/formal-admission-candidate.v3/tests/test_development_catalog.py'
spec=importlib.util.spec_from_file_location('independent_catalog_repair_fixture_v2', TEST)
test=importlib.util.module_from_spec(spec);sys.modules[spec.name]=test;spec.loader.exec_module(test)
m=test.module
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def no_draw(*a,**k):raise AssertionError('independent review forbids RNG')
m.np.random.default_rng=no_draw;m.np.random.SeedSequence=no_draw
manifest_names=['qualification/dev-cases.v1/manifest.json','qualification/dev-cases.v2/manifest.json','qualification/dev-population.v1/manifest.json']
extra_names=['qualification/dev-clean-sibling.v1/devext-OOD-L3-00-clean.json',*[f'qualification/dev-same-level-event-pair.v2/{part}/devext-OOD-L3-00.json' for part in ('event','clean')]]
refs=lambda names:[{'path':n,'sha256':sha(RUN/n)} for n in names]
inputs={n:sha(RUN/n) for n in manifest_names+extra_names}
for n in manifest_names:
 for r in json.loads((RUN/n).read_text())['records']:
  p=str(Path(n).parent/r['file']);inputs[p]=sha(RUN/p)
before_datasets=(RUN/'datasets').exists()
result=m.build_development_catalog(RUN,refs(manifest_names),refs(extra_names))
assert len(result['case_records'])==56
statuses={};matches=0;raw_matches=0
for r in result['case_records']:
 v=json.loads((RUN/r['case']['path']).read_text());arch=m._archived_identity_case(v)
 assert arch.digest()==r['case_sha256']==canonical(v)
 raw_matches+=1
 status=r['current_mission_validation']['status'];statuses[status]=statuses.get(status,0)+1
 if status=='accepted_by_current_mission_validator':
  actual=m.load_development_case(v)
  assert actual.digest()==arch.digest()
  assert m.physical_configuration_digest(actual)==r['configuration_sha256']
  matches+=1
old=next(r for r in result['case_records'] if r['case']['path']=='qualification/dev-cases.v1/dev-L2-03.json')
assert old['case']['sha256']=='12806f5d90bde3a4b8ac99d89d91c00c5ca7dc95d6d85029dafdb83243b5bc49'
assert old['case_sha256']=='ed93254799d5b4634b9ed50c974f21f67a62b387f189f35a1d1f38c3e4d38947'
v=json.loads((RUN/old['case']['path']).read_text())
# Independent configuration reconstruction from raw JSON, excluding CaseSpec identity fields.
physical={k:copy.deepcopy(v[k]) for k in m.CaseSpec.__dataclass_fields__}
for k in ('schema','case_id','lineage_group','split','seed','domain'):physical.pop(k,None)
physical['entities']=sorted([{k:x for k,x in e.items() if k!='key'} for e in physical['entities']],key=lambda e:json.dumps(e,sort_keys=True))
assert canonical(physical)==old['configuration_sha256']
try:m.load_development_case(v)
except ValueError as e:
 assert 'ground proxies cannot be certified separated between samples' in str(e);old_error=str(e)
else:raise AssertionError('original mission refusal disappeared')
clean=json.loads((RUN/extra_names[-1]).read_text())
assert type(m._archived_identity_case(clean)) is m.EventCleanSibling
faults={}
def reject(name,value):
 try:m._archived_identity_case(value)
 except ValueError as e:faults[name]=str(e)
 else:raise AssertionError('archive parser accepted structural fault '+name)
for key,val in [('schema','unknown-schema'),('seed',True),('seed',-1),('priority_at',False),('link_drone',True)]:
 x=copy.deepcopy(v);x[key]=val;reject(f'{key}={val!r}',x)
x=copy.deepcopy(v);x['entities'][0]['speed']=True;reject('boolean_entity_speed',x)
x=copy.deepcopy(v);x['entities'][0]['origin_xy'][0]=float('inf');reject('infinite_entity_coordinate',x)
x=copy.deepcopy(v);x['extra_field']='x';reject('unknown_top_level_field',x)
x=copy.deepcopy(clean);x['parent_case']=copy.deepcopy(clean);reject('nested_clean_parent',x)
x=copy.deepcopy(clean);x['parent_case']['unexpected']='x';reject('parent_extra_field',x)
x=copy.deepcopy(clean);x['parent_case']['seed']=True;reject('boolean_parent_seed',x)
x=copy.deepcopy(clean);x['parent_sha256']=False;reject('boolean_parent_digest',x)
# A well-typed historically invalid relationship remains inventory identity only.
x=copy.deepcopy(clean);x['parent_sha256']='0'*64
m._archived_identity_case(x)
try:m.load_development_case(x)
except ValueError as e:relationship_error=str(e)
else:raise AssertionError('typed archive container was mistaken for physical admission')
assert all(r['identity_role']=='archived_input_not_mission_qualification' for r in result['case_records'])
assert before_datasets==(RUN/'datasets').exists()
assert all(sha(RUN/n)==h for n,h in inputs.items())
print(json.dumps({'role':'independent bounded pure read-only R1 repair probe; existing artifacts and isolated in-memory mutations only','python':sys.version,'numpy':m.np.__version__,'artifact_records':len(result['case_records']),'not_independent_roots':True,'statuses':statuses,'raw_canonical_semantic_matches':raw_matches,'accepted_identity_configuration_matches':matches,'old_invalid_record':old,'old_executable_refusal':old_error,'independent_raw_configuration_sha256':canonical(physical),'structural_rejections':faults,'typed_invalid_clean_parent_relation_still_refused_by_mission':relationship_error,'clean_parent_depth_bound':'only population schema allowed as parent; its exact fields exclude parent_case','input_bindings':inputs,'input_bytes_unchanged':True,'no_persisted_catalog':True,'RNG_functions_forbidden':True},indent=2,ensure_ascii=False))
