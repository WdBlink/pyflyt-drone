from pathlib import Path
import ast,copy,functools,hashlib,importlib,json,pickle,subprocess,sys,base64
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';W=R/'planning/physics-runtime-window.v1';F=W/'freeze.v2';OUT=Path('/tmp/inspection-physics-runtime-window-review-v2');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((F/'binding.json').read_text());assert sha(F/'binding.json')=='8c08adb80bba6179b78401e9374c9beb3abeb1546a7208dbd07198403117bfac'
for n,h in b['files'].items():
 assert sha(F/n)==h
 if n.startswith(('source/','tests/')):assert sha(W/n)==h
oldworker=(REPO/'eval/inspection_diagnostic_worker.py').read_text();worker=(F/'source/timed_worker.py').read_text();needle='                if result["clock_profile"].get("runtime_policy",{}).get("errors"):\n                    result["status"] = "operational_failure"\n';assert worker.count(needle)==1 and worker.replace(needle,'')==oldworker
loop=lambda s:ast.dump(next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.While)),include_attributes=False);assert loop(worker)==loop(oldworker)
old=ast.parse((R/'planning/clock-localization-M2-L3.v1/probe.py').read_text());new=ast.parse((F/'source/probe.py').read_text());values=lambda tree:{n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)};newvals=values(new)
class Normalize(ast.NodeTransformer):
 def visit_Assign(self,n):
  if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('ROLE','EXTRA_SOURCES'):n.value=copy.deepcopy(newvals[n.targets[0].id])
  return self.generic_visit(n)
 def visit_ImportFrom(self,n):
  if n.module=='eval.inspection_diagnostic_worker':n.module='timed_worker'
  elif n.module=='eval.inspection_clock_profile_rusage':n.module='timed_profile';n.names=[ast.alias(name='TimedClockProfile')]
  return n
 def visit_Name(self,n):
  if n.id=='RusageClockProfile':n.id='TimedClockProfile'
  return n
old=Normalize().visit(old)
localpath=ast.parse('sys.path.insert(0,str(Path(__file__).resolve().parent))').body[0]
rootinsert=next(i for i,n in enumerate(old.body) if isinstance(n,ast.Expr) and ast.dump(n).startswith("Expr(value=Call(func=Attribute(value=Attribute(value=Name(id='sys'"))
old.body.insert(rootinsert+1,localpath);assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
role=ast.literal_eval(newvals['ROLE']);extra=ast.literal_eval(newvals['EXTRA_SOURCES']);assert len(extra)==8
assert role=='instrumented_B3_M2_runtime_candidate_not_pilot_retry' and ast.literal_eval(newvals['CASE'])=='qualification/dev-cases.v2/dev-L3-00.json' and ast.literal_eval(newvals['SAMPLER_SEED'])==1096691202
assert set(extra)=={'paper/research/20261104-inspection/planning/physics-runtime-window.v1/source/'+n for n in ['probe.py','timed_worker.py','physics_runtime.py','timed_profile.py']}|{'eval/inspection_clock_profile.py','eval/inspection_clock_profile_rusage.py','eval/inspection_process_cleanup.py','eval/inspection_darwin_inventory.py'}
sys.path.insert(0,str(REPO));sys.path.insert(0,str(W/'source'));from timed_worker import worker as worker_fn;from timed_profile import TimedClockProfile
payload=base64.b64encode(pickle.dumps(functools.partial(worker_fn,profile_factory=TimedClockProfile))).decode()
child='''import sys,pickle,base64,json,hashlib\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])\np=pickle.loads(base64.b64decode(sys.argv[3]))\nprint(json.dumps({'function_module':p.func.__module__,'function_file':str(Path(sys.modules[p.func.__module__].__file__).resolve()),'factory_module':p.keywords['profile_factory'].__module__,'factory_file':str(Path(sys.modules[p.keywords['profile_factory'].__module__].__file__).resolve()),'worker_was_not_called':True,'factory_was_not_constructed':True}))'''
p=subprocess.run([sys.executable,'-c',child,str(REPO),str(W/'source'),payload],capture_output=True,text=True,timeout=10);assert p.returncode==0,p.stderr;pickled=json.loads(p.stdout);assert pickled['function_file']==str(W/'source/timed_worker.py') and pickled['factory_file']==str(W/'source/timed_profile.py')
result={'freeze_binding_sha256':sha(F/'binding.json'),'working_and_frozen_4source_3tests_equal':True,'worker_only_delta_runtime_error_status_guard':True,'while_AST_identical':True,'probe_AST_identical_after_declared_ROLE_sources_factory_worker_path_changes':True,'case':'dev-L3-00','architecture':'B3','model':'M2','sampler_seed':1096691202,'runtime_source_hashes':{n:sha(REPO/n) for n in extra},'fresh_interpreter_pickle_import':pickled,'no_worker_or_factory_execution_in_pickle_probe':True,'future_full_source_and_budget_envelope':'root will separately freeze109source and reserve; not done by this review'}
(OUT/'adapter-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
