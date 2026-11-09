from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json,hashlib,tempfile,io,contextlib,os,socket,shutil
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';W=R/'planning/formal-admission-candidate.v3';F=W/'primary-analysis-freeze.v5';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((F/'binding.json').read_text());assert sha(F/'binding.json')=='a2d521d166f523b04ebede33eb19ec6bd53f2207ea4d5a7e1c0ec445a082dfb0'
for n,v in b['members'].items():assert sha(F/n)==v['sha256'] and sha(W/n)==v['sha256']
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('adjacent_analysis_fixture',W/'tests/test_formal_analysis.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h);a=h.analysis

import ast,copy,math
import numpy as np
from pytest import MonkeyPatch
result={'scope':'Frozen v5 exact score block and actual component/snapshot/publisher functions with Fake admissions/statistics; no real formal data/model/physics/native effects'}
source=(F/'source/eval/analyze_inspection_formal.py').read_text();tree=ast.parse(source)
collect=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collect_recording')
loop=next(n for n in collect.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='slot')
selected=[copy.deepcopy(n) for n in loop.body if 211<=n.lineno<=225]
assert len(selected)==5 and isinstance(selected[-1],ast.Expr)
wrapper=ast.parse('def project_score(slot, raw, y, evidence, schedule_sha, directory, project):\n rows=[]\n return rows[0]\n').body[0]
wrapper.body=wrapper.body[:1]+selected+wrapper.body[-1:]
namespace={'SCORE_FIELDS':a.SCORE_FIELDS,'math':math,'canonical_sha':a.canonical_sha}
exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(F/'source/eval/analyze_inspection_formal.py'),'exec'),namespace)
project_score=namespace['project_score']
(OUT/'exact-score-block.txt').write_text('\n'.join(ast.get_source_segment(source,n) for n in selected)+'\n')

def bad(action):
    try: action()
    except (ValueError,KeyError) as error: return {'type':type(error).__name__,'message':str(error)}
    raise AssertionError('expected rejection')

with tempfile.TemporaryDirectory(prefix='pa5-') as tmp:
    project=Path(tmp).resolve();run=project/'run';run.mkdir();directory=project/'logs/fixture';directory.mkdir(parents=True)
    slot=SimpleNamespace(case_id='literal-ID-L2',domain='ID',level=2,condition='B3-M1',repeat_ordinal=0,order_index=0)
    full={key:.5 for key in ('C','S','D','Q','Y')};full.update(H=1,E=1,SR=0,dynamic_entities=[{'bins':[{'index':1,'quality':.5}]}],remaining_range_m=[1200]*4)
    evidence={'role':'synthetic validated return for isolated source block','score':full}
    def raw_for(e):return {'value':e['score']['Y'],'evidence':{**copy.deepcopy(e),'adapter_repeat_ordinal':slot.repeat_ordinal,'adapter_condition':slot.condition,'order_index':slot.order_index,'schedule_sha256':'s'*64}}
    row=project_score(slot,raw_for(evidence),.5,evidence,'s'*64,directory,project)
    assert set(row['score'])==set(a.SCORE_FIELDS) and row['score']=={k:full[k] for k in a.SCORE_FIELDS}
    assert row['full_score_sha256']==a.canonical_sha(full)
    assert full['dynamic_entities']==[{'bins':[{'index':1,'quality':.5}]}]
    result['L2_exact_eight_scalar_projection_and_full_hash']=True
    changed=copy.deepcopy(evidence);changed['score']['dynamic_entities'][0]['bins'][0]['quality']=.25
    result['changed_nested_raw_evidence_rejected']=bad(lambda:project_score(slot,raw_for(changed),.5,evidence,'s'*64,directory,project))
    changed_scalar=raw_for(evidence);changed_scalar['evidence']['score']['S']=.4
    result['changed_scalar_raw_evidence_rejected']=bad(lambda:project_score(slot,changed_scalar,.5,evidence,'s'*64,directory,project))
    invalids={}
    for label,key,value in [('missing_C','C','__missing__'),('bool_C','C',True),('nan_D','D',float('nan')),('range_D','D',1.1),('float_H','H',1.0),('value_Y','Y',.4)]:
        item=copy.deepcopy(evidence)
        if value=='__missing__':del item['score'][key]
        else:item['score'][key]=value
        invalids[label]=bad(lambda item=item:project_score(slot,raw_for(item),.5,item,'s'*64,directory,project))
    result['invalid_scalar_cases']=invalids

    # Call only the existing L1 grade method on a synthetic object, never its constructor.
    scenario_path=W/'source/eval/inspection_scenarios.py'
    spec=importlib.util.spec_from_file_location('independent_scenario_L1_grade',scenario_path);sc=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sc;spec.loader.exec_module(sc)
    mission=sc.ScenarioMission.__new__(sc.ScenarioMission)
    mission.case=SimpleNamespace(level='L1');mission.entities=[SimpleNamespace(rule_id='r',key='e',origin_xy=(0.,0.),true_class='static-fixture')]
    mission.ledger=SimpleNamespace(reports={},association_ambiguities=0);mission.cells=[1];mission.covered={1};mission.weight=lambda *args:1
    mission.card={'range_budget_m_per_aircraft':18000.,'exit_range_reserve_m':1000.};mission.distance_m=np.zeros(4)
    mission.flight=SimpleNamespace(violations=[],exit_dwell=np.full(4,20.))
    l1score=mission.grade();assert l1score['D'] is None
    l1=SimpleNamespace(**{**slot.__dict__,'case_id':'literal-ID-L1','level':1});e={'role':'synthetic score from actual L1 grade, no flight','score':l1score}
    original_slot=slot;slot=l1
    result['actual_L1_grade_D_is_None']=True
    result['legal_L1_None_rejected_by_v5']=bad(lambda:project_score(l1,raw_for(e),l1score['Y'],e,'s'*64,directory,project))
    result['L1_grade_corroborating_source']={'path':str(scenario_path.relative_to(R)),'sha256':sha(scenario_path)}
    slot=original_slot
    for level in (2,3):
        invalid=copy.deepcopy(evidence);invalid['score']['D']=None
        t=SimpleNamespace(**{**slot.__dict__,'level':level})
        result['L'+str(level)+'_None_D_rejected']=bad(lambda:project_score(t,raw_for(invalid),.5,invalid,'s'*64,directory,project))

    # Preserve full8 conditions/repeats in a tiny, literal two-level component fixture.
    metadata=[{'case_id':'literal-ID-L1','domain':'ID','level':1},{'case_id':'literal-ID-L2','domain':'ID','level':2}]
    rows=[]
    for m in metadata:
        for condition,n in a.CONDITION_REPEATS.items():
            for repeat in range(n):
                score={k:(1 if k in ('H','E') else 0 if k=='SR' else .5) for k in a.SCORE_FIELDS}
                if m['level']==1:score['D']=None
                rows.append({**m,'condition':condition,'repeat_ordinal':repeat,'status':'completed','SR':0,'score':score})
    components=a.component_summary(metadata,rows)
    assert components['L1_D']=='not_applicable'
    assert all('D' not in value for value in components['root_means']['literal-ID-L1'].values())
    assert all('D' not in value for value in components['stratum_means']['ID.L1'].values())
    assert all(value['D']==.5 for value in components['root_means']['literal-ID-L2'].values())
    result['component_L1_D_NA_and_L2_D_preserved']=True

    # Only the new publication refs are integrated here; numeric/physical/admission stages are Fake.
    from eval.inspection_formal_admission import Artifact
    write=lambda p,value:(p.parent.mkdir(parents=True,exist_ok=True),p.write_text(json.dumps(value,sort_keys=True)+'\n'))
    write(run/'accepted-subject.json',{'design':{'path':'experiment/literal-design.json','sha256':'d'*64}})
    acceptance=Artifact('accepted-subject.json',sha(run/'accepted-subject.json'))
    write(run/'acceptance.json',{'subject':{'path':acceptance.path,'sha256':acceptance.sha256}})
    g4=Artifact('acceptance.json',sha(run/'acceptance.json'))
    namespace_proof=SimpleNamespace(phase='main',g4_acceptance=g4,source_manifest=Artifact('source.json','e'*64))
    proof=SimpleNamespace(dataset=SimpleNamespace(namespace=namespace_proof),classic_best='B1')
    lock=project/'logs/inspection-formal-episodes.v1/.matrix.lock';lock.parent.mkdir(parents=True);lock.touch()
    request={'run':str(run),'sdk':str(R/'runtime/autoresearch-state-reader-v1'),'dataset':{'path':'dataset.json','sha256':'f'*64}}
    world=directory/'world.json';write(world,{'role':'synthetic source world, no physics','score':full});world_before=world.read_bytes()
    original_raw=json.dumps({'value':.5,'evidence':evidence},ensure_ascii=False).encode()
    captured={'llm/raw/literal.json':original_raw}
    rows2=[row for row in rows if row['level']==2];metadata2=metadata[1:];means={metadata2[0]['case_id']:{c:.5 for c in a.CONDITION_REPEATS}}
    with MonkeyPatch.context() as mp:
        mp.setattr(a,'read_request',lambda value:value);mp.setattr(a,'admitted_request',lambda *args:(None,proof))
        mp.setattr(a,'collect_recording',lambda *args,**kwargs:(proof,metadata2,rows2,means,[],captured,{'root-means':{'runs':[]}},[]))
        mp.setattr(a,'RootTable',SimpleNamespace(parse=lambda *args,**kwargs:None))
        mp.setattr(a,'analyze_roots',lambda *args:{'role':'Fake statistics, not resampling'})
        mp.setattr(a,'sr_root_table',lambda *args:None);mp.setattr(a,'sr_cluster_intervals',lambda *args:{'role':'Fake SR statistics'})
        subject=a.analyze_primary(request,project/'logs/unused',project_root=project)
    refs={}
    for name in ('admitted_episode_scores','component_summary','sr_statistics'):
        ref=Artifact.parse(subject[name]);path=ref.checked_path(run);assert sha(path)==ref.sha256
        before=path.read_bytes();path.write_bytes(before+b' ')
        try:ref.checked_path(run)
        except ValueError as error:refs[name]={'bound':True,'tamper_rejected':str(error)}
        else:raise AssertionError('new output ref must bind exact bytes')
        path.write_bytes(before)
    assert (run/'experiment/main-recording.v1/llm/raw/literal.json').read_bytes()==original_raw
    assert world.read_bytes()==world_before
    result['new_subject_semantic_refs']=refs
    result['synthetic_captured_raw_and_source_world_unmodified']=True
    result['publisher_scope']='actual publish function, Fake admissions/collect/statistics and tiny literal grid; not full SDK/actual data'

    prefix_cases={}
    for track,opening in [(None,None),(None,{'literal':'opening'}),('clean',None),('ablation',None),('latency',None),('unadmitted',None),('',None),(True,None)]:
        name=('main' if opening is None else 'holdout') if track is None else str(track)
        case_root=project/('prefix-'+name);case_root.mkdir();r=case_root/'run';r.mkdir();ep=case_root/'logs/receipt';write(ep/'attempt-00/terminal.json',{'role':'fixture'})
        kwargs={'sampler_seed':1,'order_index':2}
        if track is not None:kwargs['track']=track
        typed=SimpleNamespace(**kwargs);req={'run':str(r),'opening':opening,'g4':{'sha256':'a'*64}}
        expected_kind=('main' if opening is None else 'holdout') if track is None else track
        rid=f'formal-{expected_kind}-aaaaaaaaaaaa-0002-a0';op=case_root/'logs/operation'/rid;res=r/'qualification/local-model-episodes'/rid
        write(op/'service-completion.json',{'role':'fixture'});write(res/'usage-receipt.json',{'role':'fixture'})
        for suffix in ('.reserve.json','.settle.json'):write(r/'resource-requests'/(rid+suffix),{'role':'fixture'})
        seen=[]
        with MonkeyPatch.context() as mp:
            mp.setattr(a,'operation_directory',lambda value:(seen.append(value),op)[1])
            if track not in (None,'clean','ablation','latency'):
                reason=bad(lambda:a._receipt_snapshot(ep,typed,req,case_root));assert seen==[]
                prefix_cases[name]={'rejected_before_operation_lookup':True,**reason};continue
            before=a._receipt_snapshot(ep,typed,req,case_root);assert seen==[rid]
            assert str(res.relative_to(case_root)) in before['directories']
            extra=op/'call-9999';extra.mkdir();prefix_cases[name]={'reservation':rid,'empty_directory_guard':bad(lambda:a._check_snapshots([(typed,ep,before)],req,case_root))}
            extra.rmdir();os.mkfifo(op/'bad-fifo')
            prefix_cases[name]['FIFO_guard']=bad(lambda:a._receipt_snapshot(ep,typed,req,case_root));(op/'bad-fifo').unlink()
            a._check_snapshots([(typed,ep,before)],req,case_root)
    result['typed_track_and_primary_prefix_cases']=prefix_cases
for n,v in b['members'].items():assert sha(F/n)==v['sha256'] and sha(W/n)==v['sha256']
result['five_frozen_working_members_stable']=True
(OUT/'delta-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
