from pathlib import Path
import importlib.util,json,sys,hashlib,gc
R=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection');F=R/'planning/physics-runtime-window.v1/freeze.v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(F/'binding.json')=='8c08adb80bba6179b78401e9374c9beb3abeb1546a7208dbd07198403117bfac'
b=json.loads((F/'binding.json').read_text())
for n,h in b['files'].items():assert sha(F/n)==h
sys.path.insert(0,str(F/'source'))
import physics_runtime as runtime
import timed_profile as module
class Native:
    numer,denom=125,3
    def __init__(self):self.rt=False;self.values=None
    def get(self,f,n):return {'values':[int(not self.rt)],'default':False} if f==1 else {'values':self.values if self.rt else [0,120000,240000,1],'default':not self.rt}
    def set(self,f,v):self.rt=f==2;self.values=list(v)
class Collector:
    def __init__(self):self.enabled=True
    def isenabled(self):return self.enabled
    def get_threshold(self):return (700,10,10)
    def collect(self):return 0
    def enable(self):self.enabled=True
    def disable(self):self.enabled=False
native=Native();collector=Collector();window=runtime.PhysicsRuntimeWindow(native=native,collector=collector)
module.PhysicsRuntimeWindow=lambda:window
before=list(gc.callbacks);original=gc.get_count
try:
 def count():
  if window.restore_needed and not window.closed:raise RuntimeError('injected post-start metadata fault')
  return original()
 gc.get_count=count
 try:module.TimedClockProfile(1)
 except RuntimeError as e:error=str(e)
 else:raise AssertionError('injected constructor exception must occur')
 result={'exception':error,'base_callback_removed':list(gc.callbacks)==before,'window_closed':window.closed,'native_time_constraint_still_active':native.rt,'collector_enabled_after_failed_constructor':collector.enabled,'audit_errors':window.audit['errors']}
 assert result=={'exception':'injected post-start metadata fault','base_callback_removed':True,'window_closed':True,'native_time_constraint_still_active':False,'collector_enabled_after_failed_constructor':True,'audit_errors':[]}
 print(json.dumps(result,indent=2))
 Path('/tmp/inspection-physics-runtime-window-review-v2/constructor-result.json').write_text(json.dumps(result,indent=2)+'\n')
finally:
 gc.get_count=original
 window.close()
 assert collector.enabled and not native.rt
