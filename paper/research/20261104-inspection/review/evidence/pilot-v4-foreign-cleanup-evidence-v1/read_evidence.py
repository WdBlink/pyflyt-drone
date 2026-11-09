from pathlib import Path
from datetime import datetime,timedelta,timezone
import hashlib,json

REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
R=REPO/'paper/research/20261104-inspection'
OUT=Path(__file__).resolve().parent
SLOTS={24:('dev-L3-01-B3-M1-2',26337),30:('dev-L3-01-B4-M1-1',86715),37:('dev-L2-00-B3-M3-1',65758)}
SIDECAR=REPO/'logs/inspection-eight-condition-pilot.v4.continuation.v2/cleanup-observations/root-60623-48545371959708.json'
JOB='local.home-jarvis.xiaomi-voice'
COMM='/Users/wdblink/Code/my_repo/home-jarvis/.venv/bin/python'
TZ8=timezone(timedelta(hours=8))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def ref(path):return {'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size}
def wall(event):return datetime.fromisoformat(event['timestamp'])
def stamp(path):
 s=path.stat()
 return {key:datetime.fromtimestamp(value,timezone.utc).isoformat() for key,value in [('birth_utc',s.st_birthtime),('mtime_utc',s.st_mtime)]}
sidecar=read(SIDECAR);assert digest(SIDECAR)=='6edd60553b11cf9032bf67e7a5f05f22e05550769c9856c624c3e2b5fa25cee8'
result={'scope':'read-only stored JSON and filesystem timestamp correlation, no new log/launchctl query or original validator','sidecar':ref(SIDECAR),'episodes':[]}
for index,(episode,pid) in SLOTS.items():
 a=REPO/'logs/inspection-eight-condition-dev-flights.v4'/episode/'attempt-00';treepath=a/'process-tree.json';terminalpath=a/'terminal.json';tree=read(treepath);terminal=read(terminalpath);log=Path(f'/tmp/inspection-v4-slot{index:03d}-launchd-matches.v1.json');rows=read(log)
 assert tree['status']=='unknown' and tree['unclaimed_new_python_orphans']==[pid] and tree['remaining_owned']==[] and str(tree['root_pid']) in tree['owned_births']
 assert terminal['status']=='unknown_process_cleanup' and terminal['Y'] is None and terminal['retry_allowed'] is False
 assert all(r['processID']==1 and r['processImagePath']==r['senderImagePath']=='/sbin/launchd' for r in rows)
 boot,=set(r['bootUUID'] for r in rows);label=f'gui/501/{JOB} [{pid}]'
 xpc,=[r for r in rows if r['subsystem']==label and r['eventMessage']==f'xpcproxy spawned with pid {pid}']
 success,=[r for r in rows if r['subsystem']==label and r['eventMessage']==f'Successfully spawned python[{pid}] because semaphore']
 removal,=[r for r in rows if r['subsystem']=='system' and r['eventMessage']==f'removing child: pid/{pid}']
 assert xpc['machTimestamp']<success['machTimestamp']<removal['machTimestamp'] and wall(xpc)<wall(success)<wall(removal)
 times=stamp(treepath);birth=datetime.fromisoformat(times['birth_utc']);mtime=datetime.fromisoformat(times['mtime_utc']);assert wall(success)<birth<=mtime<wall(removal)
 e={'order':index,'episode':episode,'pid':pid,'bootUUID':boot,'subsystem':label,'root_pid':tree['root_pid'],'owned_groups':sorted({r['group'] for r in tree['owned_births'].values()}),'tree':ref(treepath),'terminal':ref(terminalpath),'filtered_log':ref(log),'timestamps':times,'xpc':xpc['timestamp'],'python_spawn':success['timestamp'],'remove':removal['timestamp'],'python_spawn_to_remove_ms':(wall(removal)-wall(success)).total_seconds()*1000,'tree_birth_before_remove_ms':(wall(removal)-birth).total_seconds()*1000,'tree_mtime_before_remove_ms':(wall(removal)-mtime).total_seconds()*1000,'original_unknown':terminal['status'],'raw_grade_Y':terminal['evidence']['frozen_score_recount']['score']['Y'],'original_model_usage':terminal['evidence']['model_meter']['usage']}
 if index==37:
  assert sidecar['original_result']==tree and sidecar['original_exception'] is None and sidecar['observation_errors']==[]
  assert sidecar['adapter_source_sha256']=='83ffb02d0582c243e33421ba0f831d754010c1cb99c70b3022b5a743d96c646c'
  assert sidecar['context']['plan_sha256']==terminal['plan_sha256'] and sidecar['context']['reservation_prefix']=='dev-all8-pilot-v4-037'
  assert str(pid) not in sidecar['baseline_python_rows'] and str(pid) not in tree['owned_births'] and pid not in e['owned_groups']
  scans=[]
  for s in sidecar['global_scans']:
   row=s['rows'][str(pid)];assert row['parent']==1 and row['group']==pid and row['uid']==501 and row['command']==COMM and not row['status'].startswith('Z')
   born=datetime.strptime(row['born'],'%a %b %d %H:%M:%S %Y').replace(tzinfo=TZ8);assert born==wall(success).replace(microsecond=0)
   start=datetime.fromisoformat(s['started_at']);end=start+timedelta(microseconds=(s['monotonic_end_ns']-s['monotonic_start_ns'])/1000)
   assert wall(success)<start<end<wall(removal)
   scans.append({'ordinal':s['ordinal'],'start_utc':start.isoformat(),'end_utc_derived_from_elapsed_monotonic':end.isoformat(),'row':row})
  assert len(scans)==2;e['scans']=scans
  settled=read(a/'settle-owner.json');assert settled['status']=='known' and settled['reservation']=='dev-all8-pilot-v4-037-a0';e['settle_ack']=ref(a/'settle-owner.json')
 result['episodes'].append(e)
assert len({e['bootUUID'] for e in result['episodes']})==1
meta=Path('/tmp/inspection-v4-home-jarvis-launchctl-metadata.v1.txt');assert f'program = {COMM}' in meta.read_text();result['filtered_job_metadata']=ref(meta)
result['time_limit']='source log events share bootUUID. Sidecar carries UTC plus monotonic elapsed, no independent boot UUID. Older tree birth/mtime are filesystem metadata, not exact scans or hashed into tree.'
(OUT/'recount.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'orders':[(e['order'],e['pid'],e['tree_birth_before_remove_ms'],e['tree_mtime_before_remove_ms']) for e in result['episodes']],'scan37':result['episodes'][-1]['scans'],'scope':result['scope']},indent=2))
