from pathlib import Path
import ast,base64,collections,hashlib,importlib.metadata,json,math,zlib
import numpy as np
ROOT=Path('.').resolve();R=ROOT/'paper/research/20261104-inspection';D=R/'qualification/reference/executor-realtime.v2';A=R/'qualification/source/executor-v19-reference-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b,tol=1e-8):assert np.allclose(a,b,atol=tol,rtol=0)
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
run=json.loads((D/'run.json').read_text());w=json.loads((D/'world.json').read_text());q=json.loads((R/'qualification/reference/executor-realtime.v2.qualification.v1.json').read_text());c=json.loads((R/'planning/gate-contract.v19.json').read_text());binding=json.loads((A/'binding.json').read_text())
assert sha(D/'world.json')==run['world_sha256']==q['world_sha256']==binding['world_sha256'];assert sha(D/'run.json')==q['run_sha256']==binding['run_sha256']
assert len(run['sources'])==14==binding['file_count'] and len(run['dependency_hashes'])==61
for p,s in run['sources'].items():
 rel=Path(p).relative_to(ROOT) if Path(p).is_absolute() else Path(p)
 assert (A/rel).read_bytes()==Path(p).read_bytes()==s.encode() and hashlib.sha256(s.encode()).hexdigest()==run['source_hashes'][p]
for p,h in run['dependency_hashes'].items():assert sha(Path(p))==h
for p,h in q['analysis_sources_sha256'].items():assert sha(ROOT/p)==h
for package,version in run['package_versions'].items():assert importlib.metadata.version(package)==version
assert run['status']==w['status']=='completed' and run['requested_seconds']==w['simulated_seconds']==600 and not run['operational_error'];assert run['sources_unchanged'] and run['dependencies_unchanged'];assert run['realtime'] and w['clock_mode']=='wallclock_1x'
assert w['physics_steps_executed']==w['safety']['physics_steps']==w['safety']['physics_audits']==w['execution_tracking_240hz']['physics_samples']==144000
tr=w['execution_tracking_trace'];assert tr['xy_tube_m']==10 and tr['rows']==144000
raw=zlib.decompress(base64.b64decode(tr['data'],validate=True));assert len(raw)==144000*17*8 and hashlib.sha256(raw).hexdigest()==tr['raw_sha256'];a=np.frombuffer(raw,dtype='<f8').reshape(-1,17);assert np.isfinite(a).all();near(a[:,0],np.arange(1,144001)/240,1e-9)
xy,z,lo,hi=(a[:,j::4] for j in [1,2,3,4]);assert (lo==94).all() and (hi==98).all()
counts={'physics_samples':144000,'max_horizontal_error_m':float(xy.max()),'horizontal_tube_breach_steps':int(np.any(xy>10,axis=1).sum()),'altitude_envelope_breach_steps':int(np.any((z<lo)|(z>hi),axis=1).sum())}
assert counts==q['tracking_recount'];assert all(w['execution_tracking_240hz'][k]==v for k,v in counts.items());near(np.abs(z[a[:,0]>=20]-96).max(),w['execution_tracking_240hz']['max_settled_altitude_error_m'])
pos=np.array([r['positions'] for r in w['trace']]);assert pos.shape==(6000,4,3);near([r['t'] for r in w['trace']],np.arange(1,6001)/10);near(z[23::24],pos[:,:,2]);assert np.abs(pos[:,:,:2]).max()<=550
peak=tr['peak_witness'];near(math.dist(peak['actual_xyz'][:2],peak['nominal'][:2]),xy.max());near(xy[round(peak['t']*240)-1,peak['drone']],xy.max())
safe=w['safety'];assert not safe['violations'] and 60<=safe['altitude_m'][0]<=safe['altitude_m'][1]<=100 and 15<=safe['airspeed_m_s'][0]<=safe['airspeed_m_s'][1]<=25 and safe['max_bank_rad']<=math.pi/4 and safe['min_separation_m']>=15
assert z.min()>=safe['altitude_m'][0]-1e-8 and z.max()<=safe['altitude_m'][1]+1e-8
for i in range(4):
 for j in range(i):assert np.linalg.norm(pos[:,i]-pos[:,j],axis=1).min()>=safe['min_separation_m']-1e-8
assert len(safe['exit_dwell_seconds'])==4 and min(safe['exit_dwell_seconds'])>=20
remaining=w['score']['remaining_range_m'];assert len(remaining)==4 and min(remaining)>=1000
for i in range(4):assert np.linalg.norm(np.diff(pos[:,i],axis=0),axis=1).sum()<=18000-remaining[i]+1e-8
terminal=pos[-201:,:,:2]-np.array([[0,-300],[0,-100],[0,100],[0,300]])
radius=float(np.abs(np.linalg.norm(terminal,axis=2)-80).max());course=np.arctan2(np.diff(terminal,axis=0)[:,:,1],np.diff(terminal,axis=0)[:,:,0]);tangent=np.arctan2(terminal[1:,:,1],terminal[1:,:,0])+np.pi/2;course_error=float(np.abs(wrap(course-tangent)).max());assert radius<=10 and course_error<=math.radians(25)
# Reuse only the independent review's OwnState/capture arithmetic, never production helpers.
source=Path('/tmp/audit_v18_full_control_review.py').read_text();tree=ast.parse(source)
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name in {'owncheck','capture'}:exec(compile(ast.Module(body=[n],type_ignores=[]),'<independent-review-helper>','exec'))
events=collections.Counter(r['type'] for r in w['execution_audit']);seam_bad=0;reqs={r['serial']:r for r in w['execution_audit'] if r['type']=='plan_request'}
for r in reqs.values():
 assert len(r['own_states'])==len(r['remaining_range_m'])==4;near(r['commit_at']-r['now'],.5)
 for s in r['own_states']:owncheck(s,r['now'],np.zeros(3))
 assert min(r['remaining_range_m'])>=0
for r in w['execution_audit']:
 if r['type'] in ('program_committed','physical_program_seam'):
  m=r.get('metrics',r.get('capture_metrics'));owncheck(r['own_state'],r['at'],np.zeros(3));assert capture(m)
  near(m['z_m'],r['own_state']['position'][2]);near(m['measured_airspeed_m_s'],r['own_state']['airspeed']);near(m['z_m'],z[round(r['at']*240)-1,r['drone']])
  if r['type']=='physical_program_seam':assert r['capture_valid'];seam_bad+=not r['capture_valid']
assert seam_bad==w['execution_tracking_240hz']['seam_capture_failures']==0 and not w['execution_infrastructure_errors']
for i in range(4):assert np.all(np.diff([r['remaining_range_m'][i] for r in reqs.values()])<=1e-8)
# Clock: recompute stored probe geometry/time deltas and validate all available1Hz samples.
b=run['coordinator_block_probe'];near(b['after'][0]-b['before'][0],b['sim_advance_seconds']);near(np.linalg.norm(np.array(b['after'][1:])-b['before'][1:]),b['displacement_m']);assert abs(b['sim_advance_seconds']-b['actual_wait_seconds'])<=.1 and b['displacement_m']>1
assert w['max_clock_lag_seconds']==w['maximum_clock_lag_witness']['lag_seconds']<=.1 and w['clock_lag_over_0p1_seconds_ticks']==0
progress=[json.loads(line) for line in (D/'progress.jsonl').read_text().splitlines()];assert len(progress)==600;near([p['t'] for p in progress],np.arange(1,601));wall=np.array([p['wall_elapsed'] for p in progress]);assert np.all(np.diff(wall)>0);sample_drift=wall-np.arange(1,601);assert sample_drift.max()<=.1
# Complete saved opportunity denominators and raw RNG draws, separately from emitted/association counts.
emitted={r['evidence_id']:r for r in w['sensor_audit']};assert len(emitted)==len(w['sensor_audit']);referenced=set();keys=set();op=collections.defaultdict(lambda:collections.Counter(visible_opportunities=0,raw_detection_misses=0,emitted_measurements=0,association_suppressed=0));index={e['key']:i for i,e in enumerate(w['hidden_entities'])}
for r in w['sensor_opportunities']:
 key=(r['t'],r['drone_id'],r['entity_key']);assert key not in keys;keys.add(key)
 rng=np.random.default_rng(np.random.SeedSequence([run['seed'],round(r['t']*2),r['drone_id'],index[r['entity_key']],0]));assert bool(rng.random()<.1)==r['missed']
 g=op[r['mode']];g['visible_opportunities']+=1;g['raw_detection_misses']+=r['missed'];eid=r['emitted_evidence_id']
 if eid is not None:
  e=emitted[eid];assert not r['missed'] and eid not in referenced and all(e[k]==r[k] for k in ['t','drone_id','entity_key','mode']);referenced.add(eid);g['emitted_measurements']+=1
 elif not r['missed']:assert r['mode']=='inspect';g['association_suppressed']+=1
bg=collections.Counter(wide_frame_trials=0,raw_triggers=0,emitted_measurements=0);frames=set()
for r in w['background_trials']:
 key=(r['t'],r['drone_id']);assert key not in frames;frames.add(key);rng=np.random.default_rng(np.random.SeedSequence([run['seed'],round(r['t']*2),r['drone_id'],0,1]));assert bool(rng.random()<.01)==r['triggered'];bg['wide_frame_trials']+=1;bg['raw_triggers']+=r['triggered'];eid=r['emitted_evidence_id']
 if eid is not None:
  e=emitted[eid];assert r['triggered'] and e['entity_key'] is None and e['mode']=='survey' and eid not in referenced and all(e[k]==r[k] for k in ['t','drone_id']);referenced.add(eid);bg['emitted_measurements']+=1
assert referenced==set(emitted)
for g in op.values():g['raw_miss_fraction']=g['raw_detection_misses']/g['visible_opportunities']
bg['raw_trigger_fraction']=bg['raw_triggers']/bg['wide_frame_trials']
assert dict(op)==q['sensor_opportunity_recount']['sensor_modes'] and dict(bg)==q['sensor_opportunity_recount']['background']
groups=collections.defaultdict(list);delivered={r['evidence_id']:r for r in w['delivered_observations']}
for r in emitted.values():
 if r['entity_key'] is None:continue
 degraded=run['events'] and r['drone_id']==2 and 300<=r['t']<345;key=r['mode']+('_degraded' if degraded else '_normal');err=np.array(r['estimate_xy'])-r['truth_xy'];groups[key].append((float(err@err),r['predicted_class']==r['true_class'],r['true_class']=='normal',r['true_class']=='normal' and r['predicted_class']!='normal'))
sensor={k:{'measurements':len(vals),'rmse':math.sqrt(sum(v[0] for v in vals)/len(vals)),'accuracy':sum(v[1] for v in vals)/len(vals),'distractors':sum(v[2] for v in vals),'false_distractors':sum(v[3] for v in vals)} for k,vals in groups.items()}
for k,v in sensor.items():near(v['rmse'],q['original_quality_clock_calibration']['sensor_groups'][k]['position_vector_rmse_m']);near(v['accuracy'],q['original_quality_clock_calibration']['sensor_groups'][k]['class_accuracy'])
assert sensor['inspect_normal']['rmse']<=3 and sensor['inspect_normal']['accuracy']>=.9
out={'run_sha256':sha(D/'run.json'),'world_sha256':sha(D/'world.json'),'checker_sha256':sha(ROOT/'eval/qualify_inspection_reference.py'),'raw_sha256':tr['raw_sha256'],'tracking':counts,'steadyZ':w['execution_tracking_240hz']['max_settled_altitude_error_m'],'safety':safe,'remaining_range':remaining,'terminal_radius':radius,'terminal_course_degrees':math.degrees(course_error),'audit_events':dict(events),'block_delta_error':b['sim_advance_seconds']-b['actual_wait_seconds'],'block_distance':b['displacement_m'],'one_hz_max_drift':float(sample_drift.max()),'opportunities':dict(op),'background':dict(bg),'sensor':sensor}
Path('/tmp/audit_g3_v19_control_sensor_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
