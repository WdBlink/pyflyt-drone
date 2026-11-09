from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json,hashlib,tempfile,copy,shutil,zipfile
from pytest import MonkeyPatch
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';W=R/'planning/formal-admission-candidate.v3';F=W/'secondary-analysis-freeze.v2';BOOT=W/'secondary-analysis-freeze.v3';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(F/'binding.json')=='c50e74768028f747d07e6f3ee20805f6b996dfe75e79e6bc63ffa5a6e74f7bb3'
b=json.loads((F/'binding.json').read_bytes());boot=json.loads((BOOT/'binding.json').read_bytes())
assert sha(BOOT/'binding.json')=='0d1fadf64cd4a02d6fc7c0d80ebbca015fbd7924646aa13a9cecf096f16e934a'
for name,ref in b['members'].items():
    assert sha(F/name)==ref['sha256']
    expected=boot['members'][name]['sha256'] if name=='tests/test_formal_secondary_analysis.py' else ref['sha256']
    if name in ('source/eval/analyze_inspection_formal_secondary.py','source/eval/inspection_formal_admission.py','source/eval/inspection_intervention_statistics.py','tests/test_formal_secondary_analysis.py'):assert sha(W/name)==expected
    if name.startswith('source/'):assert boot['members'][name]['sha256']==ref['sha256']
assert sha(W/'source/eval/analyze_inspection_formal.py')=='edeaeb6e76243f90d013bbd0691608e226b1e2d3d1b227ae09fdb6a869fc795b'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_secondary_analysis_fixture',W/'tests/test_formal_secondary_analysis.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h);a=h.analysis
frozen_event=h.load('independent_frozen_event_summary',F/'source/eval/inspection_event_endpoints.py');a.summarize_formal_events=frozen_event.summarize_formal_events
assert Path(a.summarize_formal_events.__code__.co_filename)==F/'source/eval/inspection_event_endpoints.py'
result={'subject':sha(F/'binding.json'),'bootstrap_v3_only_staged_common_calendar_import':sha(BOOT/'binding.json'),'scope':'real original SDK envelopes with fully synthetic native-shaped fixtures, Fake measurement/common-calendar/event/numeric functions; not actual flight/data/G4/statistical validity'}

def rejected(action):
    try:action()
    except (ValueError,KeyError,FileNotFoundError) as error:return {'type':type(error).__name__,'message':str(error)}
    raise AssertionError('expected rejection')

def no_outputs(f,track):
    paths=[f.run/('analysis/secondary-'+track+'.v1'),f.run/('experiment/secondary-'+track+'-recording.v1'),f.run/('experiment/results.secondary-'+track+'.v1.json')]
    assert all(not p.exists() for p in paths)
    return paths

def remove_fake_outputs(f,track):
    # These are only this script's synthetic outputs, never existing run evidence.
    paths=[f.run/('analysis/secondary-'+track+'.v1'),f.run/('experiment/secondary-'+track+'-recording.v1'),f.run/('experiment/results.secondary-'+track+'.v1.json')]
    for p in paths:
        if p.is_dir():shutil.rmtree(p)
        elif p.exists():p.unlink()

with tempfile.TemporaryDirectory(prefix='secondary-consumer-') as tmp:
    base=Path(tmp).resolve();positives={}
    for track,additional,parents,means in [('clean',216,216,96),('ablation',108,36,36),('latency',180,36,60)]:
        td=base/track;td.mkdir()
        with MonkeyPatch.context() as mp:
            h.matching_types.__wrapped__(mp)
            f,request,proof,recording,measure=h.prepared_recording(td,mp,track)
            owner_before=copy.deepcopy(f.owner)
            # Same filename in an unbound directory cannot stand in for locked scores.
            h.write(f.project/'logs/unbound/admitted-episodes.json',[{'not':'bound main scores'}])
            def forbidden(*args,**kwargs):raise AssertionError('analysis attempted an execution/native effect')
            for name in ('execute_local_episode','execute_development_llm_episode','write_secondary_dispatch'):
                mp.setattr(h.helper.driver,name,forbidden)
            subject=a.analyze_secondary(request,recording,project_root=f.project)
            assert subject['additional_actual_episode_denominator']==additional
            assert subject['referenced_primary_episode_denominator']==parents
            assert subject['pending']=='response/resource diagnostic consumer not yet wired'
            assert f.owner==owner_before
            rows=f.read(subject['admitted_episode_scores']);assert len(rows)==additional+parents
            assert len({r['case_id'] for r in rows})==12 and all(r['level']==3 for r in rows)
            index=f.read(subject['recording_index']);copied=f.run/index['SDK_relative_reference_root']
            for ref in index['files']:
                p=f.run/ref['path'];assert p.read_bytes()==(recording/p.relative_to(copied)).read_bytes()
            original=json.loads((recording/'root-means/results.json').read_bytes());projected=f.read(subject['recording_results'])
            assert len(projected['runs'])==means
            assert {k:v for k,v in projected.items() if k not in ('design','runs')}=={k:v for k,v in original.items() if k not in ('design','runs')}
            for old,new in zip(original['runs'],projected['runs']):
                assert {k:v for k,v in old.items() if k!='raw'}=={k:v for k,v in new.items() if k!='raw'}
            # Actual SDK checks the copied raw path projection as well.
            from lib.science import _experiment
            _,parsed,raw=_experiment(f.run,subject['recording_design']['path'],subject['recording_results']['path'])
            assert len(raw)==len(parsed['runs'])==means
            for section in ('closed_receipt_inputs','diagnostic_records'):
                ref=index[section]['archive'];archive_path=f.run/ref['path'];assert sha(archive_path)==ref['sha256']
                with zipfile.ZipFile(archive_path) as archive:
                    assert archive.testzip() is None and len(archive.namelist())==len(rows)
                    for member in index[section]['members']:assert hashlib.sha256(archive.read(member['member'])).hexdigest()==member['sha256']
            if track=='clean':
                with zipfile.ZipFile(f.run/index['diagnostic_records']['archive']['path']) as archive:
                    for row,member in zip(rows,index['diagnostic_records']['members']):
                        detail=json.loads(archive.read(member['member']));assert a.canonical_sha(detail['common_calendar']['score'])==row['common_full_score_sha256']
                        assert row['common_score']=={k:detail['common_calendar']['score'][k] for k in h.primary_analysis.SCORE_FIELDS}
            positives[track]={'additional':additional,'primary_comparators':parents,'rows':len(rows),'outer_means':means,'complete_SDK_projection_original_values_evidence_design_copy':True,'metadata_and_detail_ZIP_members_hashes':True,'unbound_same_name_score_ignored':True,'owner_unchanged_and_dispatch_hooks_forbidden':True,'common_calendar_layout_and_full_score_hash_join':track=='clean'}
            if track!='ablation':continue
            # All counterexamples below reuse only our just-created Fake recording.
            remove_fake_outputs(f,track);negative={}
            def exercise(label,action):
                negative[label]=rejected(action);no_outputs(f,track)
            path=recording/'episodes/results.json';raw_before=path.read_bytes();value=json.loads(raw_before);value['runs'].pop();h.write(path,value)
            exercise('missing_SDK_slot',lambda:a.analyze_secondary(request,recording,project_root=f.project));path.write_bytes(raw_before)
            order=recording/'episode-order.json';order_before=order.read_bytes()
            for key,badvalue in [('arm','not-an-arm'),('sampler_seed',-1)]:
                value=json.loads(order_before);value['slots'][0][key]=badvalue;h.write(order,value)
                exercise('wrong_order_'+key,lambda:a.analyze_secondary(request,recording,project_root=f.project));order.write_bytes(order_before)
            scores=f.run/'analysis/main.v1/admitted-episodes.json';scores_before=scores.read_bytes();scores.write_bytes(scores_before+b' ')
            exercise('bound_main_score_bytes_changed',lambda:a.analyze_secondary(request,recording,project_root=f.project));scores.write_bytes(scores_before)
            original_recount=a._recount
            def wrong_primary(slot,*args,**kwargs):
                row,evidence,snapshot=original_recount(slot,*args,**kwargs)
                if getattr(slot,'track',None) is None:
                    row=copy.deepcopy(row);row['score']['S']=.123
                return row,evidence,snapshot
            with MonkeyPatch.context() as patch:
                patch.setattr(a,'_recount',wrong_primary)
                exercise('primary_recount_not_locked_same_identity_score',lambda:a.analyze_secondary(request,recording,project_root=f.project))
            original_measure=a.measure_completed_episode
            with MonkeyPatch.context() as patch:
                def unknown(*args,**kwargs):raise ValueError('literal unknown operational receipt, never impute0')
                patch.setattr(a,'measure_completed_episode',unknown)
                exercise('unknown_physical_measurement',lambda:a.analyze_secondary(request,recording,project_root=f.project))
            # Invoke the actual receipt join directly; a changed admitted binding must stop before measurement.
            first=proof.slots[0];episode=f.project/'logs/inspection-formal-episodes.v1/secondary-ablation'/f'{first.case_id}-{first.condition}-{first.arm}-{first.repeat_ordinal}'
            original_admit=a.admit_episode_plan;measure_calls=[]
            for key,badvalue in [('arm','not-an-arm'),('sampler_seed',-1),('parent_order_index',-1)]:
                with MonkeyPatch.context() as patch:
                    def bad_binding(plan,project,key=key,badvalue=badvalue):
                        binding=original_admit(plan,project).binding();binding[key]=badvalue;return SimpleNamespace(binding=lambda:binding)
                    patch.setattr(a,'admit_episode_plan',bad_binding)
                    patch.setattr(a,'measure_completed_episode',lambda *args:(measure_calls.append(1),None)[1])
                    negative['wrong_admitted_binding_'+key]=rejected(lambda:a._recount(first,episode,request,f.project))
                    assert measure_calls==[];no_outputs(f,track)
            # Late tree/semantic changes after all receipts validated must be rejected before writes.
            main_slot=proof.primary.slots[first.parent_order_index]
            rid=f"formal-main-{request['g4']['sha256'][:12]}-{main_slot.order_index:04d}-a0"
            late=f.project/'logs/inspection-model-resource-operations.v1'/rid/'call-9999'
            with MonkeyPatch.context() as patch:
                def mutate(track,rows):late.mkdir();return {'role':'Fake numeric result; late directory injection'}
                patch.setattr(a,'analyze_interventions',mutate)
                exercise('primary_empty_call_directory_added_during_statistics',lambda:a.analyze_secondary(request,recording,project_root=f.project))
            late.rmdir()
            main_subject=a._proofs(request,f.project)[4];semantic=f.run/main_subject['component_summary']['path'];semantic_before=semantic.read_bytes()
            with MonkeyPatch.context() as patch:
                def mutate(track,rows):semantic.write_bytes(semantic_before+b' ');return {'role':'Fake numeric result; semantic mutation'}
                patch.setattr(a,'analyze_interventions',mutate)
                exercise('locked_main_semantic_artifact_changed_before_publish',lambda:a.analyze_secondary(request,recording,project_root=f.project))
            semantic.write_bytes(semantic_before)
            # Verify SDK validator receives private captured bytes; mutate original during its call.
            import lib.science as science
            sdk_original=science._experiment;target=recording/'episodes/results.json';target_before=target.read_bytes();seen=[]
            with MonkeyPatch.context() as patch:
                def mutate_original(root,*args,**kwargs):
                    assert Path(root)!=recording and Path(root).name.startswith('inspection-secondary-sdk-validation-')
                    if not seen:target.write_bytes(target_before+b' ')
                    seen.append(str(root));return sdk_original(root,*args,**kwargs)
                patch.setattr(science,'_experiment',mutate_original)
                exercise('original_SDK_changes_during_private_validation',lambda:a.analyze_secondary(request,recording,project_root=f.project))
            target.write_bytes(target_before);assert len(seen)==2
            assert f.owner==owner_before
            result['ablation_counterexamples']=negative
    result['positive_tracks']=positives
for name,ref in b['members'].items():
    assert sha(F/name)==ref['sha256']
    expected=boot['members'][name]['sha256'] if name=='tests/test_formal_secondary_analysis.py' else ref['sha256']
    if name in ('source/eval/analyze_inspection_formal_secondary.py','source/eval/inspection_formal_admission.py','source/eval/inspection_intervention_statistics.py','tests/test_formal_secondary_analysis.py'):assert sha(W/name)==expected
assert sha(W/'source/eval/analyze_inspection_formal.py')=='edeaeb6e76243f90d013bbd0691608e226b1e2d3d1b227ae09fdb6a869fc795b'
result['frozen_v2_consumer_REQ_numeric_and_v3_bootstrap_working_hashes_stable']=True
result['event_summary_function_explicitly_from_frozen_v2']=True
result['event_analyzer_and_common_calendar_numeric_scope']='Fake; external event source/test working drift deliberately not certified'
result['working_event_drift_observed_outside_consumer_scope']={name:sha(W/name) for name in ('source/eval/inspection_event_endpoints.py','tests/test_formal_event_endpoints.py')}
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
