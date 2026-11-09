"""Zero-flight doubles of worker read and main finalization guards."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,queue,sys
import envs.inspection.mission as mm
import eval.inspection_scenarios as sm
import eval.run_inspection_reference as rr
D=Path('paper/research/20261104-inspection/qualification/dev-cases.v2/dev-L1-00.json');base=D.read_bytes();expected=hashlib.sha256(base).hexdigest();out={}
class Warmup:
 def __init__(self,*a,**kw):pass
 def step(self):pass
 def close(self):pass
old=mm.InspectionMission;old_s=sm.ScenarioMission
p=Path('/tmp/scenario-residual-review/read-mutated');p.mkdir(exist_ok=True);f=p/'private-case.json';v=json.loads(base);v['seed']+=1;f.write_text(json.dumps(v));calls=[]
try:
 mm.InspectionMission=Warmup;sm.ScenarioMission=lambda c:calls.append(c)
 rr.worker(0,False,False,0,queue.Queue(),queue.Queue(),p,private_case_path=f,expected_case_sha256=expected)
finally:mm.InspectionMission=old;sm.ScenarioMission=old_s
w=json.loads((p/'world.json').read_text());out['worker_read_changed']={'status':w['status'],'no_scenario_created':not calls,'error':w.get('operational_error')};assert w['status']=='operational_failure' and not calls
class FakeContext:
 def __init__(self,mode):self.mode=mode;self.queues=[]
 def Queue(self):q=queue.Queue();self.queues.append(q);return q
 def Array(self,*a):return None
 def Process(self,target,args):
  ctx=self
  class P:
   exitcode=0
   def start(self):
    destination=args[6];copy=args[-2]
    world={'status':'completed','simulated_seconds':0.,'score':{}}
    (destination/'world.json').write_text(json.dumps(world))
    if ctx.mode=='change':copy.write_bytes(copy.read_bytes()+b' ')
    if ctx.mode=='delete':copy.unlink()
    if ctx.mode=='missing_both':copy.unlink();(destination/'world.json').unlink()
    if ctx.mode=='missing_world':(destination/'world.json').unlink()
    if ctx.mode=='invalid_json':(destination/'world.json').write_text('{')
    if ctx.mode=='invalid_schema':(destination/'world.json').write_text('[]')
    if ctx.mode=='invalid_status':(destination/'world.json').write_text(json.dumps({'status':'bad','score':{}}))
    if ctx.mode=='invalid_score':(destination/'world.json').write_text(json.dumps({'status':'completed','score':[]}))
    if ctx.mode=='worker_failure':(destination/'world.json').write_text(json.dumps({'status':'operational_failure','operational_error':'controlled test failure'}))
    ctx.queues[1].put({'kind':'done'})
   def join(self,**kw):pass
   def is_alive(self):return False
  return P()
orig_context,orig_argv=rr.mp.get_context,sys.argv
try:
 for mode in ['unchanged','change','delete','missing_both','missing_world','invalid_json','invalid_schema','invalid_status','invalid_score','worker_failure']:
  ctx=FakeContext(mode);rr.mp.get_context=lambda *a:ctx;p=Path('/tmp/scenario-residual-review/final-'+mode)
  sys.argv=['runner','--case',str(D),'--duration','.01','--out',str(p)]
  try:code=rr.main();exc=None
  except Exception as e:code=None;exc=type(e).__name__+': '+str(e)
  manifest=json.loads((p/'run.json').read_text());out['final_'+mode]={'exit':code,'exception':exc,'manifest_status':manifest['status'],'copy_unchanged':manifest.get('private_case_unchanged'),'copy_verification_error':manifest.get('private_case_verification_error'),'world_verification_error':manifest.get('world_verification_error')}
finally:rr.mp.get_context=orig_context;sys.argv=orig_argv
assert out['final_unchanged']['exit']==0 and out['final_unchanged']['copy_unchanged'] is True
assert out['final_change']['exit']==1 and out['final_change']['manifest_status']=='operational_failure'
Path('/tmp/scenario-residual-review/freeze.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

assert out['final_delete']['exit']==1 and out['final_delete']['exception'] is None and out['final_delete']['manifest_status']=='operational_failure' and out['final_delete']['copy_unchanged'] is False

for mode in ['change','delete','missing_both','missing_world','invalid_json','invalid_schema','invalid_status','invalid_score','worker_failure']:
 assert out['final_'+mode]['exit']==1 and out['final_'+mode]['exception'] is None and out['final_'+mode]['manifest_status']=='operational_failure'
