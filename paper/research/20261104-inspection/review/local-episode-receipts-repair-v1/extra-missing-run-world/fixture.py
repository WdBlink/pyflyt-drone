
import hashlib,json,sys
from pathlib import Path
out,plan_path,mode,counter_path=map(Path,sys.argv[1:])
plan=json.loads(plan_path.read_text())
count=int(counter_path.read_text())+1 if counter_path.exists() else 1
counter_path.write_text(str(count))
if str(mode)=='fail' or str(mode)=='first-fail' and count==1:
    sys.exit(7)
out.mkdir()
case=b'private fixture, not a real case'
(out/'private-case.json').write_bytes(case)
world={'status':'completed','clock_mode':'wallclock_1x','simulated_seconds':600,
 'physics_steps_executed':144000,'physics_hz':240,'control_hz':120,
 'max_clock_lag_seconds':0.02,'private_case_file_sha256':plan['case_file_sha256'],
 'score':{'C':0.5,'S':0.5,'Q':0.5,'Y':0.,'H':1,'E':0,'SR':0}}
(out/'world.json').write_text(json.dumps(world))
run={'status':'completed','source_hashes':plan['source_hashes'],
 'dependency_hashes':plan['dependency_hashes'],'sources_unchanged':True,'dependencies_unchanged':True,
 'private_case_sha256':hashlib.sha256(case).hexdigest(),'policy':plan['condition'],
 'world_sha256':hashlib.sha256((out/'world.json').read_bytes()).hexdigest(),
 'requested_seconds':600,'realtime':True,'private_case_unchanged':True,
 'runtime_thread_environment':plan['thread_environment']}
if str(mode).startswith('bad-source'):run['sources_unchanged']=False
(out/'run.json').write_text(json.dumps(run))
if str(mode)=='bad-source-missing-case':(out/'private-case.json').unlink()

world['private_case_file_sha256']='0'*64
(out/'world.json').write_text(json.dumps(world))
(out/'run.json').unlink()
