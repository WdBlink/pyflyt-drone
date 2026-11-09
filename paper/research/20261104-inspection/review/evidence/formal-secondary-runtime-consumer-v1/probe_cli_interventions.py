"""Frozen actual CLI intervention gate with Fake dispatch, no owner/model/world."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import importlib.util,sys,json,hashlib,tempfile,io,contextlib
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';F=R/'planning/formal-admission-candidate.v3/secondary-consumer-runtime-freeze.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((F/'binding.json').read_text());assert sha(F/'binding.json')=='d0ce314b471a85438493d4399665df598ceb8bc7219e747aaac202d65dd06ef0'
target=F/'source/eval/run_inspection_llm_development.py';assert sha(target)==b['files']['source/eval/run_inspection_llm_development.py'];sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('independent_secondary_actual_cli',target);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
spy={'model_client':0,'physics_context':0}
def client(*args,**kwargs):spy['model_client']+=1;raise AssertionError('no model transport may be created')
def physics(*args,**kwargs):spy['physics_context']+=1;raise AssertionError('no physics context may be created')
module.LocalModelClient=client;module.mp.get_context=physics
fake=ModuleType('eval.inspection_formal_dispatch');sys.modules[fake.__name__]=fake
root=Path(tempfile.mkdtemp(prefix='cli-',dir=OUT));results=[]
for track,arm,actual in [('ablation','periodic',None),('ablation','no-cross-aircraft-memory','periodic'),('latency','delay-05',0),('latency','delay-00',2),('ablation','periodic','periodic'),('latency','delay-20',20),('latency','delay-00',0),('clean','clean',None)]:
 fake.cli_dispatch=lambda *args,track=track,arm=arm,**kwargs:SimpleNamespace(episode=SimpleNamespace(slot=SimpleNamespace(track=track,arm=arm)))
 folder=root/str(len(results));run=folder/'absent-run';out=folder/'output'
 args=['candidate','--run',str(run),'--reservation','Fake-only','--model','M1','--architecture','B4','--case',str(folder/'literal-parent.json'),'--sampler-seed','1','--out',str(out),'--formal-dispatch','literal.json','--formal-dispatch-sha256','a'*64]
 if track=='ablation' and actual is not None:args+=['--ablation',str(actual)]
 elif track=='latency' and actual:args+=['--injected-delay-seconds',str(actual)]
 sys.argv=args
 with contextlib.redirect_stderr(io.StringIO()) as err:
  try:module.main()
  except SystemExit as e:status='rejected_intervention';assert e.code==2 and 'actual intervention differs' in err.getvalue()
  except FileNotFoundError as e:status='passed_intervention_before_missing_fixture_registry';assert str(e.filename).endswith('planning/local-model-registry.v1.json')
  else:raise AssertionError('Fake must stop before any effect')
 expected=(None if track!='ablation' else arm) if track!='latency' else int(arm[6:])
 wrong=actual!=expected
 assert (status=='rejected_intervention')==wrong and not out.exists() and spy=={'model_client':0,'physics_context':0}
 results.append({'track':track,'arm':arm,'actual':actual,'status':status,'output_not_created':True,'effects':dict(spy)})
(OUT/'cli-intervention-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
