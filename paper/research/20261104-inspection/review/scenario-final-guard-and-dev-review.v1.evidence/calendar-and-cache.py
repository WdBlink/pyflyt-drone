"""Offline common-calendar probes and cache-only validation; no new flight."""
import copy,hashlib,json,shutil
from dataclasses import replace
from pathlib import Path
import eval.inspection_episode_receipts as receipts
from eval.inspection_cases import CaseSpec,validate_case
from eval.inspection_common_calendar import rescore_common_calendar
R=Path('paper/research/20261104-inspection');Q=R/'qualification';F=Q/'dev-receipts.v2/dev-L3-00-B0-0/attempt-00/flight';w=json.loads((F/'world.json').read_text());c=CaseSpec.from_dict(w['private_case']);out={}
r=rescore_common_calendar(w,c);saved=json.loads((Q/'dev-receipts.v2/L3-00-common-calendar.v1.json').read_text());out['L3_saved_common_exact']=r==saved
p=Q/'dev-diagnostic.v1/L1-00-full/world.json';l1=json.loads(p.read_text());a=rescore_common_calendar(l1,CaseSpec.from_dict(l1['private_case']));out['L1_original_exact']=a['score']==l1['score']
clean=replace(c,level='L2',priority_at=None,link_drone=None,link_start=None,link_end=None,camera_drone=None,camera_start=None,camera_end=None,retire_drone=None,retire_at=None);validate_case(clean);out['same_L3_trace_clean_demand_rescore']=rescore_common_calendar(w,clean)
# Separate input-admission from original-score/geometry guards.
bad=replace(c,priority_at=0.)
try:validate_case(bad);valid=True
except ValueError:valid=False
try:x=rescore_common_calendar(w,bad);out['invalid_priority_calendar']={'validator_accepts':valid,'rescorer_accepts':True,'score':{k:x['score'][k] for k in ['D','Y','static_denominator_weight','dynamic_denominator_weight']}}
except Exception as e:out['invalid_priority_calendar']={'validator_accepts':valid,'rescorer_accepts':False,'error':str(e)}
bad=replace(c,schema='unsupported-calendar-schema')
try:rescore_common_calendar(w,bad);out['invalid_calendar_schema_accepted']=True
except Exception:out['invalid_calendar_schema_accepted']=False
for name,world,calendar in [('changed_geometry',w,replace(c,wind_xy=(0.,0.))),('wrong_saved_score',copy.deepcopy(w),c)]:
 if name=='wrong_saved_score':world['score']['D']+=.01
 try:rescore_common_calendar(world,calendar);out[name]={'rejected':False}
 except ValueError as e:out[name]={'rejected':True,'reason':str(e)}
# A copied verified completed receipt can only use cache: dispatch is prohibited.
source=Q/'dev-receipts.v2/dev-L3-00-B0-0';dest=Path('/tmp/scenario-final-review/cache-copy');shutil.copytree(source,dest)
plan=json.loads((dest/'plan.json').read_text());factory_calls=[]
def forbid(path):factory_calls.append(str(path));raise AssertionError('review cannot dispatch any flight')
old=receipts.subprocess.Popen
receipts.subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Popen forbidden'))
try:value,receipt=receipts.execute_local_episode(dest,plan,forbid)
finally:receipts.subprocess.Popen=old
out['cache_only']={'Y':value,'receipt':receipt,'factory_calls':factory_calls,'attempt_directories':[p.name for p in dest.glob('attempt-*')]};assert receipt['cache_reused'] and not factory_calls and receipt['actual_attempts']==1
assert out['L3_saved_common_exact'] and out['L1_original_exact'] and out['changed_geometry']['rejected'] and out['wrong_saved_score']['rejected']
Path('/tmp/scenario-final-review/calendar-and-cache.json').write_text(json.dumps(out,indent=2));print('same original',out['L3_saved_common_exact'],out['L1_original_exact']);print('pure_reweight_summary',{k:out['same_L3_trace_clean_demand_rescore']['score'][k] for k in ['C','S','D','Y','H','E','SR','static_denominator_weight','dynamic_denominator_weight']});print('invalid_calendar',out['invalid_priority_calendar'],out['invalid_calendar_schema_accepted']);print('cache_only',out['cache_only'])
