from pathlib import Path
import importlib.util,sys,tempfile,json
ROOT=Path.cwd().resolve();C=ROOT/'paper/research/20261104-inspection/planning/formal-admission-candidate.v1/proof-validator-freeze.v1';T=Path('/tmp/inspection-g4-proof-review');sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('race_fixture_tests',C/'tests/test_g4_proof_boundary.py');f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f);m=f.module
project=Path(tempfile.mkdtemp(prefix='json-read-race-final-',dir=T));run,a=f.synthetic_g4_fixture(project);closure_path=(run/'sources.json').resolve();source=project/'envs/inspection/mission.py';source.write_text('# source drift beyond original G4 closure\n');old_sha=m.file_sha(closure_path);replacement=json.loads(closure_path.read_text());replacement['sources']['envs/inspection/mission.py']=m.file_sha(source)
def attempt():
 try:p=m.read_g4_proof(run,project,a);return {'accepted':True,'proof_source_manifest_sha256':p.source_manifest.sha256}
 except Exception as e:return {'accepted':False,'exception':type(e).__name__,'reason':str(e)}
normal=attempt();real=m.file_sha;fired=[]
def switch_after_hash(path):
 result=real(path)
 if Path(path).resolve()==closure_path and not fired:
  fired.append(True);closure_path.write_text(json.dumps(replacement))
 return result
m.file_sha=switch_after_hash
try:changed=attempt()
finally:m.file_sha=real
out={'project':str(project),'without_race':normal,'with_check_parse_race':changed,'hook_fired':len(fired),'qualified_manifest_sha256':old_sha,'actual_manifest_sha256_after_read':m.file_sha(closure_path),'bytes_changed_after_hash_before_parse':old_sha!=m.file_sha(closure_path)}
assert normal['accepted'] is False and changed['accepted'] is True and len(fired)==1 and old_sha!=m.file_sha(closure_path)
(T/'json-binding-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
