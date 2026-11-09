"""Offline residual-calendar closure, not a physical/causal experiment."""
from pathlib import Path
from dataclasses import replace
import json,hashlib
from eval.inspection_cases import CaseSpec
from eval.inspection_common_calendar import rescore_common_calendar
R=Path('paper/research/20261104-inspection/qualification');out={};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths={'L1':R/'dev-diagnostic.v1/L1-00-full','L3':R/'dev-receipts.v2/dev-L3-00-B0-0/attempt-00/flight'}
for name,p in paths.items():
 w=json.loads((p/'world.json').read_text());c=CaseSpec.from_dict(w['private_case']);v=rescore_common_calendar(w,c);assert v['original_score_exact'] is True and v['score']==w['score'];out[name]={'world_sha':sha(p/'world.json'),'original_score_exact':True,'whole_score_dict_exact':v['score']==w['score'],'summary':{k:w['score'][k] for k in ['C','S','D','Q','H','E','Y','SR']}}
 if name=='L3':
  saved=json.loads((R/'dev-receipts.v2/L3-00-common-calendar.v1.json').read_text());assert v==saved;out[name]['saved_common_json_exact']=True
  for n,bad in [('priority_zero',replace(c,priority_at=0.)),('invalid_schema',replace(c,schema='unsupported-calendar-schema')),('different_geometry',replace(c,wind_xy=(0.,0.)))]:
   try:rescore_common_calendar(w,bad);raise AssertionError('invalid calendar unexpectedly admitted')
   except ValueError as e:out[n]={'rejected':True,'reason':str(e)}
  clean=replace(c,level='L2',priority_at=None,link_drone=None,link_start=None,link_end=None,camera_drone=None,camera_start=None,camera_end=None,retire_drone=None,retire_at=None);v=rescore_common_calendar(w,clean);out['valid_alternate_calendar']={'accepted':True,'role':v['role'],'D':v['score']['D'],'Y':v['score']['Y'],'scope':'same actual event trace rescored; clean sibling not flown'}
Path('/tmp/scenario-residual-review/calendar.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
