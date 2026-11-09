"""Zero-physics independent admission counterexamples."""
import json,math,hashlib
from dataclasses import replace
from pathlib import Path
from eval.inspection_cases import CaseSpec,validate_case,coverage_cells
from envs.inspection.scenario import in_polygon
ROOT=Path.cwd();R=ROOT/'paper/research/20261104-inspection';D=R/'qualification/dev-cases.v1';out={}
cases={l:CaseSpec.from_dict(json.loads((D/f'dev-{l}-00.json').read_text())) for l in ['L1','L2','L3']}
def attempt(name,case):
 try:r=validate_case(case);out[name]={'accepted':True,'validation':r,'case':case.as_dict()}
 except Exception as e:out[name]={'accepted':False,'exception':type(e).__name__,'message':str(e),'case':case.as_dict()}
base=cases['L2'];dyn=next(e for e in base.entities if e.object_type=='dynamic');stat=tuple(e for e in base.entities if e.object_type=='static')
case=replace(base,case_id='counterexample-coincident-demand',entities=stat+tuple(replace(dyn,key=f'duplicate-dynamic-{k:02d}') for k in range(10)))
attempt('coincident_ten_dynamic',case)
a,b=base.roi_polygon[:2];length=math.dist(a,b);inward=(-(b[1]-a[1])/length,(b[0]-a[0])/length);outward=(-inward[0],-inward[1]);mid=((a[0]+b[0])/2,(a[1]+b[1])/2);origin=(mid[0]+.05*inward[0],mid[1]+.05*inward[1]);dt=(600-120)/600;radius=1.5*dt/(2*math.pi)
aliased=replace(dyn,key='aliased-dynamic',origin_xy=origin,appeared_at=120.,speed=1.5,radius=radius,bend=0.,heading=math.atan2(outward[1],outward[0]))
case=replace(base,case_id='counterexample-subsample-roi-exit',entities=stat+(aliased,))
attempt('roi_aliasing',case);out['roi_aliasing'].update(probe_time=120+dt/4,probe_position=aliased.position(120+dt/4),probe_inside=in_polygon(*aliased.position(120+dt/4),base.roi_polygon),sampled_all_inside=all(in_polygon(*aliased.position(120+dt*k),base.roi_polygon) for k in range(601)))
case=replace(cases['L1'],case_id=True,lineage_group=123,entities=(replace(cases['L1'].entities[0],origin_xy=(True,False),radius=True,heading=False,bend=True),)+cases['L1'].entities[1:])
attempt('boolean_numeric_and_nonstring_identity',case)
polygon=((-400.,-400.),(400.,-400.),(400.,400.),(100.,400.),(100.,200.),(-100.,200.),(-100.,400.),(-400.,400.),(-380.,0.))
case=replace(cases['L1'],case_id='counterexample-simple-collinear-separated-edges',roi_polygon=polygon)
attempt('valid_disjoint_collinear_polygon',case)
# Independent correct segment-intersection check includes bounding boxes.
def orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def crosses(a,b,c,d):
 if max(a[0],b[0])<min(c[0],d[0]) or max(c[0],d[0])<min(a[0],b[0]) or max(a[1],b[1])<min(c[1],d[1]) or max(c[1],d[1])<min(a[1],b[1]):return False
 return orient(a,b,c)*orient(a,b,d)<=0 and orient(c,d,a)*orient(c,d,b)<=0
pairs=[]
for i in range(9):
 for j in range(i+1,9):
  if j not in [i+1,(i-1)%9] and crosses(polygon[i],polygon[(i+1)%9],polygon[j],polygon[(j+1)%9]):pairs.append([i,j])
out['valid_disjoint_collinear_polygon']['actual_nonadjacent_intersections']=pairs
out['valid_disjoint_collinear_polygon']['coverage_cells']=len(coverage_cells(polygon))
manifest=json.loads((D/'manifest.json').read_text());records=[]
for r in manifest['records']:
 p=D/r['file'];c=CaseSpec.from_dict(json.loads(p.read_text()));v=validate_case(c)
 records.append({'case_id':c.case_id,'level':c.level,'source_file_sha_match':hashlib.sha256(p.read_bytes()).hexdigest()==r['file_sha256'],'case_digest_match':c.digest()==r['case_sha256'],'validation_match':v==r['validation'],'capacity':v['capacity']['combined_perfect_reports_needed'],'static':sum(e.object_type=='static' and bool(e.rule_id) for e in c.entities),'dynamic':sum(e.object_type=='dynamic' for e in c.entities),'normal':sum(not e.rule_id for e in c.entities),'wind':math.hypot(*c.wind_xy),'coverage':v['coverage_cells']})
out['frozen_suite']={'manifest_sha':hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),'source_sha_matches':manifest['source_sha256']==hashlib.sha256(Path('eval/inspection_cases.py').read_bytes()).hexdigest(),'root_count':len(records),'pilot':[r['case_id'] for r in manifest['records'] if r['first_pilot']],'records':records}
out['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('eval/inspection_cases.py'),Path('eval/inspection_scenarios.py'),Path('eval/run_inspection_reference.py'),Path('tests/test_inspection_cases.py'),Path('envs/inspection/mission.py'),Path('envs/inspection/scoring.py')]}
Path('/tmp/scenario-adapter-review/admission.json').write_text(json.dumps(out,indent=2,allow_nan=False))
for k,v in out.items():
 if k not in ['source_hashes','frozen_suite']:print(k,{z:a for z,a in v.items() if z!='case'})
print('suite',len(records),'all_bindings',all(r['source_file_sha_match'] and r['case_digest_match'] and r['validation_match'] for r in records),'capacity_range',[min(r['capacity'] for r in records),max(r['capacity'] for r in records)])
print('sources',out['source_hashes'])
