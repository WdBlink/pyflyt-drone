from pathlib import Path
import importlib.util,sys,json,copy,tempfile,hashlib,math
ROOT=Path.cwd().resolve();R=ROOT/'paper/research/20261104-inspection';C=R/'planning/formal-admission-candidate.v1/proof-validator-freeze.v1';T=Path('/tmp/inspection-g4-proof-review');sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('proof_fixture_tests',C/'tests/test_g4_proof_boundary.py');f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f);m=f.module
binding=json.loads((C/'binding.json').read_text());assert all(m.file_sha(C/k)==v for k,v in binding['files'].items())
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,allow_nan=False))
def make(name):
 project=Path(tempfile.mkdtemp(prefix=name+'-',dir=T));run,a=f.synthetic_g4_fixture(project);return project,run,a

def result(project,run,a):
 try:
  proof=m.read_g4_proof(run,project,a);return {'accepted':True,'proof_native_request':proof.native_request_sha256,'source_manifest':proof.source_manifest.path}
 except Exception as e:return {'accepted':False,'exception':type(e).__name__,'reason':str(e)}

def rebind(run,a,mutate_subject=None):
 owner=read(run/'.paper/control.json');subject=read(run/'subject.json')
 if mutate_subject:mutate_subject(subject,owner)
 write(run/'subject.json',subject);sub={'path':'subject.json','sha256':m.file_sha(run/'subject.json')}
 review=read(run/'review.json');review['subject_sha256']=sub['sha256'];write(run/'review.json',review);rv={'path':'review.json','sha256':m.file_sha(run/'review.json')}
 accept=read(run/'acceptance.json');accept['subject']=sub;accept['review']=rv;write(run/'acceptance.json',accept);a={'path':'acceptance.json','sha256':m.file_sha(run/'acceptance.json')}
 q=owner['rounds'][0]['request'];q['evidence']={x['path']:x['sha256'] for x in (a,sub,rv)};q.pop('request_sha256',None);q['request_sha256']=m.canonical_sha(q);owner['rounds'][0]['decision']['request_sha256']=q['request_sha256'];write(run/'.paper/control.json',owner);return a

out={}
p,r,a=make('positive');out['positive']=result(p,r,a)
p,r,a=make('initializer-drift');init=p/'envs/inspection/__init__.py';init.write_text('# qualified-time initializer marker\n');before=result(p,r,a);init.write_text('# altered initializer marker after qualification\n');out['initializer_drift']={'before':before,'after':result(p,r,a),'initializer_is_omitted':str(init.relative_to(p)) not in read(r/'sources.json')['sources']}
p,r,a=make('arbitrary-dependency');out['arbitrary_one_dependency']={'result':result(p,r,a),'dependency_keys':list(read(r/'sources.json')['dependencies'])}
p,r,a=make('unapplied-control');owner=read(r/'.paper/control.json');prior=copy.deepcopy(owner);prior['rounds']=[];write(r/'.paper/checkpoint.json',{'state':prior,'sha256':m.canonical_sha(prior)});out['unapplied_control_round']={'result':result(p,r,a),'control_round_count':len(owner['rounds']),'checkpoint_round_count':0,'paired_digest_matches':m.canonical_sha(owner)==m.canonical_sha(prior)}
p,r,a=make('brief-drift');owner=read(r/'.paper/control.json');owner['brief']={'fixture':True};owner['brief_sha256']=m.canonical_sha(owner['brief']);write(r/'.paper/control.json',owner);a=rebind(r,a,lambda s,o:s.update(owner={k:o[k] for k in ('created_at','brief_sha256','frozen_plan')}));before=result(p,r,a);owner=read(r/'.paper/control.json');owner['brief']={'fixture':True,'changed_scientific_contract':'not reflected in brief_sha256'};write(r/'.paper/control.json',owner);write(r/'.paper/checkpoint.json',{'state':owner,'sha256':m.canonical_sha(owner)});out['brief_drift']={'before':before,'after':result(p,r,a),'brief_digest_matches':m.canonical_sha(owner['brief'])==owner['brief_sha256'],'checkpoint_pair_matches':True}
p,r,a=make('superseded');owner=read(r/'.paper/control.json');rev={'schema':'synthetic-revocation','status':'needs_work','subject_sha256':m.file_sha(r/'subject.json')};write(r/'revocation.json',rev);q={'round':{'id':'synthetic-round-2'},'evidence':{'revocation.json':m.file_sha(r/'revocation.json')}};q['request_sha256']=m.canonical_sha(q);owner['rounds'].append({'request':q,'validation':'structure_and_binding_only','decision':{'request_sha256':q['request_sha256'],'status':'active','progress':'regression','reason':'synthetic later withdrawal','finding':{'id':'synthetic-revocation','revises':'synthetic-finding','claim':'G4 qualification withdrawn for this subject','limitations':'G4needs_work','validation':'revocation.json'}}});write(r/'.paper/control.json',owner);write(r/'.paper/checkpoint.json',{'state':owner,'sha256':m.canonical_sha(owner)});out['superseded_finding']={'result':result(p,r,a),'later_finding_revises_qualification':True,'later_status':'G4needs_work'}
p,r,a=make('json-hash-read-race');source=p/'envs/inspection/mission.py';source.write_text('# changed source not in the qualified manifest\n');normal=result(p,r,a);closure_path=r/'sources.json';old_sha=m.file_sha(closure_path);replacement=read(closure_path);replacement['sources']['envs/inspection/mission.py']=m.file_sha(source);real_sha=m.file_sha;triggered=[]
def racing_sha(path):
 h=real_sha(path)
 if Path(path)==closure_path and not triggered:
  triggered.append(True);write(closure_path,replacement)
 return h
m.file_sha=racing_sha
try:raced=result(p,r,a)
finally:m.file_sha=real_sha
out['json_hash_read_race']={'without_race':normal,'with_race':raced,'manifest_current_sha_differs_from_bound':m.file_sha(closure_path)!=old_sha,'trigger_count':len(triggered)}
# Helper finite-number guarantee, isolated from the proof's fixed string fields.
q=T/'overflow.json';q.write_text('{"value":1e999}');overflow=m.read_json(q);out['json_overflow_helper']={'accepts_nonfinite_overflow':math.isinf(overflow['value'])}
# Real task: use an existing intention as an attempted proof; never add or mutate its state.
actual=r0=R;candidate={'path':'qualification/g4-next-integration-intent.v1.json','sha256':m.file_sha(R/'qualification/g4-next-integration-intent.v1.json')};before=m.file_sha(R/'.paper/control.json');actual_result=result(ROOT,R,candidate);out['real_task_intent_is_not_acceptance']={**actual_result,'writes_performed':False}
(T/'results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2))
