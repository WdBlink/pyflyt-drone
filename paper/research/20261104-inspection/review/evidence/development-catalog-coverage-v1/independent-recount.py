"""Independent read-only persisted-input coverage recount; no adapter writer or draws."""
from pathlib import Path
import hashlib,importlib.util,json,secrets,sys,zipfile
from collections import Counter
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');RUN=REPO/'paper/research/20261104-inspection'
ROOT=RUN/'qualification/development-identity-catalog.v1';OUT=ROOT/'results.v2'
sys.path.insert(0,str(REPO))
def read(path):
 def unique(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('duplicate JSON key')
   d[k]=v
  return d
 return json.loads(Path(path).read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
stage=RUN/'planning/formal-admission-candidate.v3/source/eval'
a=load('eval.inspection_formal_admission',stage/'inspection_formal_admission.py')
p=load('eval.inspection_development_catalog',stage/'inspection_development_catalog.py')
def nodraw(*args,**kw):raise AssertionError('coverage reviewer forbids RNG')
p.np.random.default_rng=nodraw;p.np.random.SeedSequence=nodraw;secrets.randbits=nodraw
before=(RUN/'datasets').exists();assert not before
catalog=read(OUT/'catalog.json');coverage=read(OUT/'coverage.json');binding=read(OUT/'binding.json')
assert sha(OUT/'binding.json')=='f90c3d2f06f0721cc9313cdc3fa0cdb74962140a7806960550d026858f187580'
assert all(sha(RUN/n)==v for n,v in binding['files'].items())
assert all(sha(REPO/n)==v for n,v in coverage['source_hashes'].items())
assert binding['producer_sha256']==sha(ROOT/'source/build_catalog.py')
manifest_paths=[]
for path in sorted((RUN/'qualification').rglob('manifest.json')):
 v=read(path)
 if isinstance(v,dict) and v.get('split')=='dev':manifest_paths.append(str(path.relative_to(RUN)))
assert set(manifest_paths)=={v['path'] for v in catalog['manifest_refs']}
rows={r['case']['path']:r for r in catalog['case_records']};known={}
base_seeds=set();manifest_counts={};input_bindings={};statuses=Counter()
def seed(v):assert type(v)is int and 0<=v<2**128;return v
for name in manifest_paths:
 f=RUN/name;v=read(f);input_bindings[name]=sha(f);manifest_counts[name]=len(v['records'])
 if 'master_seed' in v:base_seeds.add(seed(v['master_seed']))
 for record in v['records']:
  case_name=str(Path(name).parent/record['file']);cf=RUN/case_name
  assert not cf.is_symlink() and cf.stat().st_nlink==1
  digest=sha(cf);value=read(cf);input_bindings[case_name]=digest
  assert digest==record['file_sha256']==rows[case_name]['case']['sha256']
  assert canon(value)==record['case_sha256']==rows[case_name]['case_sha256']
  # Independently reproduce physical configuration from raw CaseSpec fields.
  config={k:value[k] for k in p.CaseSpec.__dataclass_fields__}
  for k in ('schema','case_id','lineage_group','split','seed','domain'):config.pop(k,None)
  config['entities']=sorted([{k:x for k,x in e.items() if k!='key'} for e in config['entities']],key=lambda e:json.dumps(e,sort_keys=True))
  assert canon(config)==rows[case_name]['configuration_sha256']
  base_seeds.add(seed(value['seed']))
  if 'root_generation_seed'in record:base_seeds.add(seed(record['root_generation_seed']))
  known.setdefault(digest,[]).append(case_name)
  try:p.load_development_case(value)
  except (ValueError,TypeError,KeyError,IndexError,OverflowError) as e:
   status={'status':'rejected_by_current_mission_validator','reason':str(e)}
  else:status={'status':'accepted_by_current_mission_validator','reason':None}
  assert rows[case_name]['current_mission_validation']==status;statuses[status['status']]+=1
 if 'namespace_intent_sha256'in v:
  ip=Path(name).parent/'namespace-intent.json';iv=read(RUN/ip);input_bindings[str(ip)]=sha(RUN/ip)
  assert sha(RUN/ip)==v['namespace_intent_sha256']
  base_seeds.add(seed(iv['master_seed']));base_seeds.update(seed(x) for x in iv['root_generation_seeds'])
assert len(rows)==sum(manifest_counts.values())==56
assert sorted({x['case_sha256'] for x in rows.values()})==catalog['case_sha256_values']
assert sorted({x['configuration_sha256'] for x in rows.values()})==catalog['configuration_sha256_values']
known={k:sorted(v) for k,v in known.items()}
# Independently walk both trees without the adapter's source/freeze exclusions.
copy_paths=sorted(pth for base in (REPO/'logs',RUN/'qualification') for pth in base.rglob('private-case.json'))
claimed={v['path']:v for v in coverage['private_copies']}
assert {str(x.relative_to(REPO)) for x in copy_paths}==set(claimed)
copy_counts=Counter();run_binding=Counter();private_bindings={}
for path in copy_paths:
 name=str(path.relative_to(REPO));v=claimed[name];digest=sha(path)
 assert not path.is_symlink() and digest==v['sha256'] and digest in known
 assert sorted(v['catalog_paths'])==known[digest]
 private_bindings[name]=digest;copy_counts['logs' if name.startswith('logs/') else 'qualification']+=1
 rp=path.parent/'run.json'
 if rp.exists():
  assert sha(rp)==v['run_manifest_sha256']
  rv=read(rp);assert rv.get('private_case_sha256')==digest;run_binding['exact_private_SHA_in_adjacent_run']+=1
 else:
  assert v['run_manifest_sha256'] is None
  dp=path.parent/'diagnostic.json';assert read(dp)['case_sha256']==digest
  private_bindings[str(dp.relative_to(REPO))]=sha(dp);run_binding['exact_SHA_in_diagnostic_no_adjacent_run']+=1
# Independently enumerate every qualification ZIP and hash each private member (CRC checked by read).
zip_paths=sorted((RUN/'qualification').rglob('*.zip'));zc={v['path']:v for v in coverage['qualification_archives']}
assert {str(x.relative_to(RUN)) for x in zip_paths}==set(zc)
zip_members=0;zip_bytes=0;zip_bindings={};duplicate_private_member_names=[]
for path in zip_paths:
 name=str(path.relative_to(RUN));row=zc[name];digest=sha(path)
 assert not path.is_symlink() and digest==row['sha256'];zip_bindings[name]=digest;zip_bytes+=path.stat().st_size
 actual=[]
 with zipfile.ZipFile(path) as z:
  seen=set()
  for info in z.infolist():
   if Path(info.filename).name!='private-case.json':continue
   assert '\\' not in info.filename
   if info.filename in seen:duplicate_private_member_names.append(name+'!'+info.filename)
   seen.add(info.filename);body=z.read(info);h=hashlib.sha256(body).hexdigest();assert h in known
   raw=json.loads(body);assert canon(raw) in catalog['case_sha256_values']
   actual.append({'member':info.filename,'sha256':h,'catalog_paths':known[h]})
 assert sorted((x['member'],x['sha256'],sorted(x['catalog_paths'])) for x in actual)==sorted((x['member'],x['sha256'],sorted(x['catalog_paths'])) for x in row['private_members'])
 zip_members+=len(actual)
# All persisted root-grid manifests are included, even without assuming their directory prefix.
procedural_paths=set((RUN/'qualification/reference').glob('*/run.json'))
for path in (RUN/'qualification').rglob('manifest.json'):
 v=read(path)
 if isinstance(v,dict) and isinstance(v.get('grid'),list):procedural_paths.add(path)
pc={x['path']:x for x in coverage['procedural_inputs']};assert {str(x.relative_to(RUN)) for x in procedural_paths}==set(pc)
additional=set();procedural_bindings={}
for path in sorted(procedural_paths):
 name=str(path.relative_to(RUN));v=read(path);row=pc[name];assert sha(path)==row['sha256'];s=set()
 if 'seed'in v:s.add(seed(v['seed']))
 for g in v.get('grid',[]):
  if g.get('status')!='unstarted':s.add(seed(g['root']))
 assert sorted(s)==row['parameter_seeds'];additional.update(s);procedural_bindings[name]=sha(path)
assert sorted(additional)==coverage['additional_procedural_seed_values']
assert sorted(base_seeds|additional)==catalog['seed_values']
rebuilt=p.build_development_catalog(RUN,catalog['manifest_refs'])
rebuilt['seed_values']=sorted(set(rebuilt['seed_values'])|additional)
rebuilt['procedural_seed_scope']=catalog['procedural_seed_scope']
assert rebuilt==catalog
assert (json.dumps(rebuilt,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()==(OUT/'catalog.json').read_bytes()
assert (ROOT/'results.v1/catalog.json').read_bytes()==(OUT/'catalog.json').read_bytes()
assert coverage['datasets_exists_before'] is False and (RUN/'datasets').exists()==before
assert all(sha(RUN/n)==h for n,h in input_bindings.items())
assert all(sha(REPO/n)==h for n,h in private_bindings.items())
print(json.dumps({'reviewer':'/root/statistics_design_review','scope':'current persisted explicitly declared dev manifests plus all persisted private-case.json under logs/qualification and all qualification ZIPs; procedural seeds separately excluded','python':sys.version,'numpy':p.np.__version__,'binding_sha256':sha(OUT/'binding.json'),'manifest_counts':manifest_counts,'artifact_records':len(rows),'semantic_identities':len(catalog['case_sha256_values']),'physical_configurations':len(catalog['configuration_sha256_values']),'current_validation_statuses':dict(statuses),'private_copy_counts':dict(copy_counts),'private_run_binding_counts':dict(run_binding),'qualification_ZIPs':len(zip_paths),'ZIP_total_bytes':zip_bytes,'ZIP_private_members':zip_members,'duplicate_private_member_names':duplicate_private_member_names,'procedural_input_count':len(pc),'base_seed_count':len(base_seeds),'additional_procedural_seed_count':len(additional),'seed_set_overlap':len(base_seeds&additional),'total_exclusion_seeds':len(catalog['seed_values']),'raw_canonical_semantic_and_configuration_recounts':True,'exact_catalog_rebuild_matches_bytes':True,'catalog_v1_v2_bytes_identical':True,'all_input_bytes_unchanged':True,'datasets_absent_before_and_after':True,'no_RNG_no_adapter_writer':True,'input_bindings':input_bindings,'private_bindings':private_bindings,'archive_bindings':zip_bindings,'procedural_bindings':procedural_bindings,'source_hashes':coverage['source_hashes'],'additional_current_import_source_bindings':{n:sha(REPO/n) for n in ('envs/inspection/scenario.py','eval/inspection_episode_receipts.py')},'not_proven':['unpersisted or outside-tree historical RNG use','flight/model/usage/G4 admission','new formal seed or dataset generation']},ensure_ascii=False,indent=2))
