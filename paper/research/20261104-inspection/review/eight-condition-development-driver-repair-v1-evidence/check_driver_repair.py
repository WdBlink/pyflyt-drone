from pathlib import Path
import sys,os,json,hashlib,tempfile,shutil,contextlib,io,subprocess,signal
sys.dont_write_bytecode=True
ROOT=Path.cwd().resolve();R=ROOT/'paper/research/20261104-inspection';T=Path('/tmp/inspection-eight-driver-review');sys.path.insert(0,str(ROOT))
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from eval import run_inspection_eight_condition_pilot as m
from eval.inspection_process_cleanup import SubprocessTree
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert H(ROOT/'eval/run_inspection_eight_condition_pilot.py')=='b978b7dd50e68a28128c9101ce5d48d7c2c6591ee17ddca3d86b9faea8cdf59c'
PREP=ROOT/'logs/inspection-eight-condition-pilot-preparation.v1';sdk=Path(next(str(Path(k).parent.parent) for k in json.loads((PREP/'source-hashes.json').read_text()) if '/lib/' in k))
selection=R/'qualification/classic-six-root-cohort.v3/selection.json';suite=R/'qualification/dev-cases.v2'
original_popen=subprocess.Popen
before={str(p):H(p) for p in (R/'.paper/control.json',R/'.paper/checkpoint.json')}

def make_run(mode):
 work=Path(tempfile.mkdtemp(prefix='repair-'+mode+'-',dir=T)); run=work/'run'
 for name in ('.paper/control.json','planning/local-model-registry.v1.json','experiment/design.ral.v3.json','experiment/evaluator.v2.py','experiment/evaluator-source-manifest.v2.json'):
  (run/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,run/name)
 shutil.copytree(R/'experiment/evaluator-source.v2',run/'experiment/evaluator-source.v2')
 return work,run

def invoke(mode):
 work,run=make_run(mode);out=work/'sdk';calls=[];cleanup=[]
 if mode=='initial-manifest-drift':
  p=run/'experiment/evaluator-source-manifest.v2.json';v=json.loads(p.read_text());v['review_fake_extra']=True;p.write_text(json.dumps(v))
 if mode=='initial-entry-drift':
  p=run/'experiment/evaluator.v2.py';p.write_text(p.read_text()+'\n# fake initial drift\n')
 def llm(directory,plan,command,**kw):
  calls.append(plan)
  if len(calls)==1:
   if mode=='entry-drift':
    p=Path(plan['evaluator']);p.write_text(p.read_text()+'\n# fake callback drift\n')
   if mode=='closure-add':(run/'experiment/evaluator-source.v2/extra.py').write_text('# extra')
  return .5,{'cache_reused':False,'fixture_only':True}
 m.execute_development_llm_episode=llm
 class Tree(SubprocessTree):
  def wait(self,timeout):return super().wait(.35)
  def close(self):
   result=super().close();cleanup.append(result);return result
 m.SubprocessTree=Tree
 fakepid=work/'child.pid'
 def popen(cmd,*args,**kw):
  assert len(cmd)>1
  if Path(cmd[1])==ROOT/'eval/run_inspection_dev.py':
   return original_popen([sys.executable,str(T/'classic-fake-supervisor.py'),str(fakepid)],*args,**kw)
  return original_popen(cmd,*args,**kw)
 subprocess.Popen=popen
 argv=['driver','--run',str(run),'--suite',str(suite),'--selection',str(selection),'--selection-sha256',H(selection),
 '--sdk-root',str(sdk),'--flight-receipts',str(work/'receipts'),'--out',str(out),'--schedule-seed','2026110611',
 '--sampler-seed','2026110612','--reservation-prefix','never-reserve-review-fake']
 if mode.startswith('initial-') or mode=='prepare':argv+=['--prepare-only']
 old=sys.argv;sys.argv=argv;error=None;child=None
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):m.main()
 except BaseException as e:error={'type':type(e).__name__,'message':str(e)}
 finally:
  sys.argv=old;subprocess.Popen=original_popen
  if fakepid.exists():
   child=int(fakepid.read_text())
   try:os.kill(child,0);live=True
   except ProcessLookupError:live=False
   # zombies are not live workers; check real inventory if PID lingers
   if live:
    p=subprocess.run(['ps','-o','stat=','-p',str(child)],capture_output=True,text=True);live=p.stdout.strip()!='' and not p.stdout.strip().startswith('Z')
   if live:
    try:os.killpg(child,signal.SIGKILL)
    except ProcessLookupError:pass
 result={'work':str(work),'error':error,'llm_callbacks':len(calls),'cleanup':cleanup,
         'invocations':[json.loads(p.read_text()) for p in sorted((out/'invocations').glob('*.json'))] if (out/'invocations').exists() else [],
         'child_pid':child,'child_live_after_driver':live if child else None,'pilot_completion':(out/'pilot-completion.json').exists()}
 if mode=='initial-manifest-drift':
  p=subprocess.run([sys.executable,str(run/'experiment/evaluator.v2.py'),'--world',str(work/'does-not-exist.json'),'--expected-world-sha256','0'*64],capture_output=True,text=True)
  result['actual_evaluator_returncode']=p.returncode;result['actual_evaluator_stderr']=p.stderr
 if (out/'source-hashes.json').exists():
  hashes=json.loads((out/'source-hashes.json').read_text());result['frozen_source_count']=len(hashes);result['evaluator_frozen']=str(run/'experiment/evaluator.v2.py') in hashes
 if (out/'classic/results.json').exists():
  result['denominators']={g:json.loads((out/g/'results.json').read_text())['reported_denominator'] for g in ('classic','llm','root-means')}
 return result
results={mode:invoke(mode) for mode in ('prepare','initial-entry-drift','entry-drift','closure-add','initial-manifest-drift','classic-timeout')}
results['owner_unchanged']={p:H(p)==h for p,h in before.items()};results['source_sha256']=H(ROOT/'eval/run_inspection_eight_condition_pilot.py');results['cleanup_sha256']=H(ROOT/'eval/inspection_process_cleanup.py')
(T/'repair-results.json').write_text(json.dumps(results,indent=2,sort_keys=True));print(json.dumps(results,indent=2,sort_keys=True))
