from pathlib import Path
import sys, os, json, hashlib, tempfile, importlib.util, contextlib, io, subprocess, collections, ast
ROOT=Path.cwd().resolve(); R=ROOT/'paper/research/20261104-inspection'; TMP=Path('/tmp/inspection-eight-driver-review').resolve()
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
sys.path.insert(0,str(ROOT))
from eval import run_inspection_eight_condition_pilot as driver
from eval import run_inspection_dev as classic
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prepared=ROOT/'logs/inspection-eight-condition-pilot-preparation.v1'
old_sources=json.loads((prepared/'source-hashes.json').read_text())
sdk=Path(next(str(Path(k).parent.parent) for k in old_sources if '/lib/' in k))
selection=R/'qualification/classic-six-root-cohort.v3/selection.json'
suite=R/'qualification/dev-cases.v2'
owner_before={str(p):h(p) for p in (R/'.paper/control.json',R/'.paper/checkpoint.json') if p.exists()}
audit={'process_calls':[],'blocked_writes':[]}
def hook(event,args):
    if event=='open':
        file,mode,flags=args
        if not isinstance(file,(str,bytes,os.PathLike)): return
        p=Path(file).resolve()
        writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and p.is_relative_to(ROOT):
            audit['blocked_writes'].append(str(p));raise AssertionError('repo write forbidden: '+str(p))
    if event in ('os.mkdir','os.remove','os.rename','os.rmdir'):
        for value in args[:2 if event=='os.rename' else 1]:
            if isinstance(value,(str,bytes,os.PathLike)) and Path(value).resolve().is_relative_to(ROOT):
                audit['blocked_writes'].append(str(value));raise AssertionError('repo mutation forbidden: '+str(value))
sys.addaudithook(hook)
def fail_popen(*a,**kw):
    audit['process_calls'].append('Popen');raise AssertionError('real process forbidden')
subprocess.Popen=fail_popen
saved_run=subprocess.run

def argv(out,receipts,prepare=False):
    return ['driver','--run',str(R),'--suite',str(suite),'--selection',str(selection),'--selection-sha256',h(selection),
            '--sdk-root',str(sdk),'--flight-receipts',str(receipts),'--out',str(out),'--schedule-seed','2026110611',
            '--sampler-seed','2026110612','--reservation-prefix','review-fake-never-reserve']+(['--prepare-only'] if prepare else [])

def invoke(mode):
    work=Path(tempfile.mkdtemp(prefix=mode+'-',dir=TMP)); out=work/'sdk'; receipts=work/'receipts'
    captured={'classic':[],'llm':[],'cache_checks':0}; vals={}
    def classic_execute(directory,plan,command_factory):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        (directory/'plan.json').write_text(json.dumps(plan)); term=directory/'attempt-1/terminal.json';term.parent.mkdir();term.write_text('{}')
        ev={'episode_receipt':str(term),'episode_receipt_sha256':h(term),'cache_reused':False}
        cmd=command_factory(directory/'fake-flight')
        captured['classic'].append({'plan':plan,'argv':cmd}); vals[str(directory)]=(.25,ev)
        return .25,ev
    classic.execute_local_episode=classic_execute
    def outer_execute(directory,plan,command_factory):
        captured['cache_checks']+=1
        assert json.loads((Path(directory)/'plan.json').read_text())==plan
        return vals[str(directory)]
    driver.execute_local_episode=outer_execute
    def fake_run(cmd,**kw):
        audit['process_calls'].append('intercept-classic')
        assert cmd[0]==sys.executable and Path(cmd[1])==ROOT/'eval/run_inspection_dev.py'
        prev=sys.argv;sys.argv=[cmd[1],*cmd[2:]]; stream=io.StringIO()
        try:
            with contextlib.redirect_stdout(stream): classic.main()
        finally:sys.argv=prev
        return subprocess.CompletedProcess(cmd,0,stream.getvalue(),'')
    subprocess.run=fake_run
    def llm(directory,plan,command_factory,**kw):
        captured['llm'].append({'directory':str(directory),'plan':plan,'argv':command_factory(Path(directory)/'fake-flight','fake-native-reservation'),
                                'kwargs':{k:str(v) for k,v in kw.items()}})
        if mode=='all' and len(captured['llm'])==1:
            ep=Path(plan['evaluator']);ep.write_text(ep.read_text()+'\n# fake mid-pilot evaluator drift\n')
        if mode=='first-failure':raise RuntimeError('deliberate fake first-slot infrastructure failure')
        return .5,{'cache_reused':False,'fixture_only':True}
    driver.execute_development_llm_episode=llm
    before_process=len(audit['process_calls']); prev=sys.argv;sys.argv=argv(out,receipts,mode=='prepare');error=None
    try:
        with contextlib.redirect_stdout(io.StringIO()):driver.main()
    except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
    finally:sys.argv=prev
    result={'work':str(work),'error':error,'classic_callbacks':len(captured['classic']),'llm_callbacks':len(captured['llm']),
            'cache_checks':captured['cache_checks'],'process_attempts':audit['process_calls'][before_process:]}
    if mode=='prepare':
        assert error is None and not captured['classic'] and not captured['llm'] and not result['process_attempts']
        assert not list(out.rglob('results.json'))
        result['preparation']=json.loads((out/'preparation.json').read_text())
    else:
        result['groups']={}
        for group,n in (('classic',18),('llm',90),('root-means',48)):
            obj=json.loads((out/group/'results.json').read_text())
            result['groups'][group]={'keys':list(obj),'object':obj}
            # keep evidence full outputs on disk; compact below after inspection
        (work/'captures.json').write_text(json.dumps(captured,sort_keys=True,indent=2))
        if mode=='all':
            assert error is None and len(captured['classic'])==18 and len(captured['llm'])==90 and captured['cache_checks']==18
            completion=json.loads((out/'pilot-completion.json').read_text());result['completion']=completion
            assert completion['actual_episode_denominator']==108 and completion['outer_aggregate_rows']==48
            source={str(p):h(p) for p in [*sorted((ROOT/'envs/inspection').glob('*.py')), ROOT/'envs/__init__.py',
                    *[ROOT/'eval'/n for n in ('inspection_cases.py','inspection_scenarios.py','inspection_population.py','run_inspection_reference.py',
                     'inspection_llm_actor.py','inspection_llm_protocol.py','inspection_model_client.py','inspection_model_service.py','inspection_process_cleanup.py','run_inspection_llm_development.py')]]}
            for row in captured['llm']:
                p=row['plan']; assert p['source_hashes']==source
                assert p['repeat_ordinal'] in range(3) and p['max_attempts']==2 and p['max_clock_lag_seconds']==.1
                assert p['physics_python']==row['argv'][0]==sys.executable and p['condition']==row['argv'][row['argv'].index('--architecture')+1]+'-'+row['argv'][row['argv'].index('--model')+1]
                assert str(p['sampler_seed'])==row['argv'][row['argv'].index('--sampler-seed')+1]
                assert p['reservation_usage']=={'calls':p['generation_profile']['max_calls'],'tokens':p['generation_profile']['max_total_tokens']}
            result['llm_source_count']=len(source)
    return result
original_R=R
R=TMP/'fake-run-evaluator-drift'
(R/'.paper').mkdir(parents=True)
(R/'planning').mkdir()
(R/'experiment').mkdir()
for name in ('.paper/control.json','planning/local-model-registry.v1.json','experiment/evaluator.v2.py'):
    (R/name).write_bytes((original_R/name).read_bytes())
results={m:invoke(m) for m in ('all',)}
rows=json.loads((Path(results['all']['work'])/'captures.json').read_text())['llm']
results['evaluator_drift']={'first_sha':rows[0]['plan']['evaluator_sha256'],
    'second_sha':rows[1]['plan']['evaluator_sha256'],
    'both_accepted':rows[0]['plan']['evaluator_sha256']!=rows[1]['plan']['evaluator_sha256'] and results['all']['error'] is None}

results['owner_unchanged']={p:h(p)==v for p,v in owner_before.items()};results['audit']=audit
results['driver_sha256']=h(ROOT/'eval/run_inspection_eight_condition_pilot.py')
(TMP/'evaluator-drift-full.json').write_text(json.dumps(results,indent=2,sort_keys=True))
# print compact schemas for an independent read of SDK denominator/status data
for mode,row in results.items():
    if mode not in ('all','first-failure'):continue
    for group,g in row['groups'].items():
        obj=g.pop('object');g['top_summary']={k:v for k,v in obj.items() if not isinstance(v,(dict,list))};g['nested_keys']={k:(len(v) if isinstance(v,list) else list(v)) for k,v in obj.items() if isinstance(v,(dict,list))}
print(json.dumps(results,indent=2,sort_keys=True))
