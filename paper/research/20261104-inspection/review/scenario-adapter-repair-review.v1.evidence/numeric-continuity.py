"""Zero-physics focused native type and within-envelope curve probes."""
from pathlib import Path
from dataclasses import replace
import json,math
from eval.inspection_cases import CaseSpec,validate_case,CaseEntity
from envs.inspection.scenario import in_polygon
Q=Path('paper/research/20261104-inspection/qualification/dev-cases.v2');base=CaseSpec.from_dict(json.loads((Q/'dev-L2-00.json').read_text()));l1=CaseSpec.from_dict(json.loads((Q/'dev-L1-00.json').read_text()));result={}
def check(name,c):
 try:v=validate_case(c);result[name]={'accepted':True,'validation':v}
 except Exception as e:result[name]={'accepted':False,'type':type(e).__name__,'reason':str(e)}
for field in ['appeared_at','speed','radius','heading','bend']:
 check('entity_bool_'+field,replace(l1,entities=(replace(l1.entities[0],**{field:False}),)+l1.entities[1:]))
check('bool_origin',replace(l1,entities=(replace(l1.entities[0],origin_xy=(True,False)),)+l1.entities[1:]));check('bad_origin_shape',replace(l1,entities=(replace(l1.entities[0],origin_xy=(0.,0.,0.)),)+l1.entities[1:]));check('numeric_key',replace(l1,entities=(replace(l1.entities[0],key=5),)+l1.entities[1:]));check('numeric_case_id',replace(l1,case_id=1));check('numeric_lineage',replace(l1,lineage_group=1))
# A permitted60m sine has its true maximum between601 sample times.
y=0.;cross=[];poly=base.roi_polygon
for a,b in zip(poly,poly[1:]+poly[:1]):
 if (a[1]>y)!=(b[1]>y):cross.append(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]))
x=max(cross);dyn=next(e for e in base.entities if e.object_type=='dynamic');e=replace(dyn,key='between-samples',origin_xy=(x-60+.00001,y),appeared_at=120.,speed=1.5,radius=60.,heading=0.,bend=0.)
c=replace(base,entities=tuple(z for z in base.entities if z.object_type=='static')+(e,));check('within_envelope_between_sample_roi_excursion',c);peak=120+math.pi*60/(2*1.5);result['within_envelope_between_sample_roi_excursion'].update(all_601_inside=all(in_polygon(*e.position(120+.8*k),poly) for k in range(601)),peak_time=peak,peak_position=e.position(peak),peak_inside=in_polygon(*e.position(peak),poly))
assert all(not z['accepted'] for z in result.values()) and result['within_envelope_between_sample_roi_excursion']['all_601_inside'] and not result['within_envelope_between_sample_roi_excursion']['peak_inside']
Path('/tmp/scenario-adapter-repair-review/numeric-continuity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
