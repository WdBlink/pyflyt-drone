from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json,hashlib,tempfile,io,contextlib,os,socket,shutil
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';W=R/'planning/formal-admission-candidate.v3';F=W/'primary-analysis-freeze.v6';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((F/'binding.json').read_text());assert sha(F/'binding.json')=='22affb24af636e16ef77cc0da756255c4717070e025b424a2bbe023248109808'
for n,v in b['members'].items():assert sha(F/n)==v['sha256'] and sha(W/n)==v['sha256']
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('adjacent_analysis_fixture',W/'tests/test_formal_analysis.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h);a=h.analysis

import ast,copy,math
import numpy as np
result={'scope':'Exact frozen v6 scalar_score plus original collect projection AST, actual synthetic L1 grade; no whole SDK, real flight, model, formal RNG or native effects'}
source=(F/'source/eval/analyze_inspection_formal.py').read_text();tree=ast.parse(source)
collect=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect_recording')
loop=next(n for n in collect.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='slot')
start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='recorded_evidence')
end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='rows')
selected=copy.deepcopy(loop.body[start:end+1]);assert len(selected)==5
wrapper=ast.parse('def project_score(slot, raw, y, evidence, schedule_sha, directory, project):\n rows=[]\n return rows[0]\n').body[0];wrapper.body=wrapper.body[:1]+selected+wrapper.body[-1:]
namespace={'scalar_score':a.scalar_score,'canonical_sha':a.canonical_sha}
exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(F/'source/eval/analyze_inspection_formal.py'),'exec'),namespace)
project_score=namespace['project_score'];(OUT/'exact-score-block.txt').write_text('\n'.join(ast.get_source_segment(source,n) for n in selected)+'\n')

def bad(action):
    try:action()
    except (ValueError,KeyError) as error:return {'type':type(error).__name__,'message':str(error)}
    raise AssertionError('must reject')

scenario_path=W/'source/eval/inspection_scenarios.py';assert sha(scenario_path)=='9048fb7b55dde1f5f262c8c60d12357259cc81c0cf9b2b22bec620cc71812638'
spec=importlib.util.spec_from_file_location('independent_scenario_L1_grade_v6',scenario_path);sc=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sc;spec.loader.exec_module(sc)
mission=sc.ScenarioMission.__new__(sc.ScenarioMission);mission.case=SimpleNamespace(level='L1')
mission.entities=[SimpleNamespace(rule_id='r',key='e',origin_xy=(0.,0.),true_class='static-fixture')];mission.ledger=SimpleNamespace(reports={},association_ambiguities=0)
mission.cells=[1];mission.covered={1};mission.weight=lambda *args:1
mission.card={'range_budget_m_per_aircraft':18000.,'exit_range_reserve_m':1000.};mission.distance_m=np.zeros(4);mission.flight=SimpleNamespace(violations=[],exit_dwell=np.full(4,20.))
full=mission.grade();assert full['D'] is None
before=copy.deepcopy(full);full_sha=a.canonical_sha(full)
scalars=a.scalar_score(full,1,full['Y']);assert set(scalars)==set(a.SCORE_FIELDS) and scalars['D'] is None
assert scalars=={k:full[k] for k in a.SCORE_FIELDS} and full==before and a.canonical_sha(full)==full_sha
result['actual_L1_grade_null_D_helper_accepted']=True;result['L1_D_preserved_not_zero']=True
result['full_score_input_unchanged_and_hash_preserved']=True
result['L1_grade_source']={'path':str(scenario_path.relative_to(R)),'sha256':sha(scenario_path)}

with tempfile.TemporaryDirectory(prefix='pa6-') as tmp:
    project=Path(tmp).resolve();directory=project/'logs/receipt';directory.mkdir(parents=True)
    slot=SimpleNamespace(case_id='literal-L1',domain='ID',level=1,condition='B0',repeat_ordinal=0,order_index=0)
    evidence={'role':'synthetic actual L1 grade result, no flight','score':full}
    raw={'value':full['Y'],'evidence':{**copy.deepcopy(evidence),'adapter_repeat_ordinal':0,'adapter_condition':'B0','order_index':0,'schedule_sha256':'s'*64}}
    raw_before=copy.deepcopy(raw)
    row=project_score(slot,raw,full['Y'],evidence,'s'*64,directory,project)
    assert row['score']['D'] is None and row['score']=={k:full[k] for k in a.SCORE_FIELDS}
    assert row['full_score_sha256']==full_sha and raw==raw_before and full==before
    result['actual_collect_block_L1_projection_full_SHA_and_raw_conservation']=True
    changed=copy.deepcopy(raw);changed['evidence']['score']['association_ambiguities']=1
    result['raw_nested_evidence_drift_rejected']=bad(lambda:project_score(slot,changed,full['Y'],evidence,'s'*64,directory,project))
    changed=copy.deepcopy(raw);changed['evidence']['score']['C']=.9
    result['raw_scalar_evidence_drift_rejected']=bad(lambda:project_score(slot,changed,full['Y'],evidence,'s'*64,directory,project))

    nulls={};invalids={};accepted={}
    for level in (1,2,3):
        positive=copy.deepcopy(full)
        if level!=1:positive['D']=.5
        out=a.scalar_score(positive,level,positive['Y']);assert out['D']==positive['D'];accepted[str(level)]=True
        for dynamic in (None,0.,.5,1.,float('nan'),float('inf'),-0.1,1.1,True,'0.5'):
            item=copy.deepcopy(full);item['D']=dynamic
            good=(dynamic is None) if level==1 else type(dynamic) in (int,float) and math.isfinite(dynamic) and 0<=dynamic<=1
            label=f'L{level} D={dynamic!r}'
            if good:
                assert a.scalar_score(item,level,item['Y'])['D']==dynamic
            else:nulls[label]=bad(lambda item=item,level=level:a.scalar_score(item,level,item['Y']))
    result['levels1_2_3_legal_cases_accepted']=accepted;result['D_NA_and_finite_value_negatives']=nulls
    for level in (1,2,3):
        base=copy.deepcopy(full)
        if level!=1:base['D']=.5
        for key in ('C','S','Q','Y'):
            for value in (float('nan'),float('inf'),-.1,1.1,True,None):
                item=copy.deepcopy(base);item[key]=value
                invalids[f'L{level} {key}={value!r}']=bad(lambda item=item,level=level:a.scalar_score(item,level,full['Y']))
        for key in ('H','E','SR'):
            for value in (True,1.,-1,2,None):
                item=copy.deepcopy(base);item[key]=value
                invalids[f'L{level} {key}={value!r}']=bad(lambda item=item,level=level:a.scalar_score(item,level,full['Y']))
        for key in a.SCORE_FIELDS:
            item=copy.deepcopy(base);del item[key]
            invalids[f'L{level} missing {key}']=bad(lambda item=item,level=level:a.scalar_score(item,level,full['Y']))
        mismatch=copy.deepcopy(base);invalids[f'L{level} Y mismatch']=bad(lambda:a.scalar_score(mismatch,level,.123))
    for level in (True,1.,'1',0,4,None):invalids[f'invalid level={level!r}']=bad(lambda level=level:a.scalar_score(full,level,full['Y']))
    result['other_finite_binary_missing_Y_and_level_negatives_count']=len(invalids)
    result['other_finite_binary_missing_Y_and_level_negatives']=invalids
assert full==before and a.canonical_sha(full)==full_sha
for n,v in b['members'].items():assert sha(F/n)==v['sha256'] and sha(W/n)==v['sha256']
result['five_frozen_and_working_member_hashes_stable']=True
(OUT/'delta-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('D_NA_and_finite_value_negatives','other_finite_binary_missing_Y_and_level_negatives')},ensure_ascii=False,indent=2))
