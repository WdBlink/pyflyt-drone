from pathlib import Path
import importlib.util,sys,json,tempfile,dataclasses,shutil,difflib
ROOT=Path.cwd().resolve();R=ROOT/'paper/research/20261104-inspection';W=R/'planning/formal-admission-candidate.v3';F=W/'proof-validator-freeze.v3';T=Path('/tmp/inspection-g4-proof-review-v3');sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('independent_v3_fixture',W/'tests/test_g4_proof_boundary.py');f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f);m=f.module
assert m.file_sha(F/'binding.json')=='99c73827c94649f0c43cf708c32365eaac21e6909d2a3df7baf8dfdc3f9bc4c9'
b=json.loads((F/'binding.json').read_text());assert all(m.file_sha(R/k)==v for k,v in b['files'].items());assert m.file_sha(W/'source/eval/inspection_formal_admission.py')==m.file_sha(F/'inspection_formal_admission.py');assert m.file_sha(W/'tests/test_g4_proof_boundary.py')==m.file_sha(F/'test_g4_proof_boundary.py')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,allow_nan=False))
def attempt(p,r,a,v):
 try:q=m.read_g4_proof(r,p,a,verification=v);return {'accepted':True,'native_request':q.native_request_sha256}
 except Exception as e:return {'accepted':False,'exception':type(e).__name__,'reason':str(e)}
def rebind(r,a,s):
 o=read(r/'.paper/control.json');write(r/'subject.json',s);sr={'path':'subject.json','sha256':m.file_sha(r/'subject.json')};review=read(r/'review.json');review['subject_sha256']=sr['sha256'];write(r/'review.json',review);rr={'path':'review.json','sha256':m.file_sha(r/'review.json')};ac=read(r/'acceptance.json');ac.update(subject=sr,review=rr);write(r/'acceptance.json',ac);a={'path':'acceptance.json','sha256':m.file_sha(r/'acceptance.json')};q=o['rounds'][0]['request'];q['evidence']={x['path']:x['sha256'] for x in (a,sr,rr)};q.pop('request_sha256',None);q['request_sha256']=m.canonical_sha(q);o['rounds'][0]['decision']['request_sha256']=q['request_sha256'];write(r/'.paper/control.json',o);write(r/'.paper/checkpoint.json',{'state':o,'sha256':m.canonical_sha(o)});return a
p=Path(tempfile.mkdtemp(prefix='entry-',dir=T));r,a,v=f.synthetic_g4_fixture(p);out={'positive':attempt(p,r,a,v)};assert out['positive']['accepted']
ev=r/'evaluator.json';expected=read(r/'design.json')['evaluator']['sha256'];ev.write_text('{"role":"new unaccepted evaluator"}');c=read(r/'sources.json');key=str(ev.relative_to(p));c['sources'][key]=m.file_sha(ev);write(r/'sources.json',c);s=read(r/'subject.json');s['source_manifest']['sha256']=m.file_sha(r/'sources.json');a=rebind(r,a,s);v=dataclasses.replace(v,source_names=frozenset([*v.source_names,key]));out['entry_drift_fully_rebound']={**attempt(p,r,a,v),'accepted_entry_sha256':expected,'actual_entry_sha256':m.file_sha(ev),'all_native_subject_refs_rebound':True};assert not out['entry_drift_fully_rebound']['accepted'] and 'artifact changed' in out['entry_drift_fully_rebound']['reason']
actual=m.context_from_environment(R,ROOT,f.RUNTIME,f.RUNTIME);out['actual_metadata_factory_positive']={'dependency_count':len(actual.dependency_names),'source_name_count':len(actual.source_names),'owner_reader_invoked':False};assert len(actual.dependency_names)==61
manifest=read(R/'experiment/evaluator-source-manifest.v2.json');members=[]
for name in manifest['sources']:
 p=Path(tempfile.mkdtemp(prefix='member-',dir=T));r=p/'synthetic-run';ex=r/'experiment';ex.mkdir(parents=True);shutil.copy2(R/'experiment/evaluator-source-manifest.v2.json',ex/'evaluator-source-manifest.v2.json');shutil.copytree(R/'experiment/evaluator-source.v2',ex/'evaluator-source.v2');q=ex/'evaluator-source.v2'/name;q.write_bytes(q.read_bytes()+b'\n# independent single-member drift fixture\n')
 try:m.context_from_environment(r,p,p/'owner',p/'sdk');result={'accepted':True}
 except Exception as e:result={'accepted':False,'exception':type(e).__name__,'reason':str(e)}
 assert not result['accepted'] and 'artifact changed' in result['reason'];members.append({'name':name,**result})
out['manifest_member_drifts']=members;out['source_sha256']=m.file_sha(F/'inspection_formal_admission.py');out['tests_sha256']=m.file_sha(F/'test_g4_proof_boundary.py');(T/'independent-results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
old=R/'planning/formal-admission-candidate.v2/proof-validator-freeze.v2/inspection_formal_admission.py';(T/'source.diff').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),(F/'inspection_formal_admission.py').read_text().splitlines(True),fromfile='v2',tofile='v3')))
print(json.dumps({'positive':out['positive']['accepted'],'entry_drift_rejected':not out['entry_drift_fully_rebound']['accepted'],'individually_changed_manifest_members_rejected':len(members),'actual_factory_dependencies':len(actual.dependency_names),'real_owner_load':False},indent=2))
