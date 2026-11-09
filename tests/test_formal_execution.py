"""Primary execution-plan fixtures: literal orders/seeds only, no formal RNG."""
import importlib.util
import ast
from pathlib import Path
import sys
from dataclasses import replace

import pytest

HERE=Path(__file__).resolve().parent; STAGE=HERE.parent/'eval'
spec=importlib.util.spec_from_file_location('execution_dataset_helpers',HERE/'test_dataset_admission.py')
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
load=helper.helper.load
sys.modules['eval.inspection_dataset_admission']=helper.module
recording=load('execution_recording_candidate',STAGE/'inspection_recording.py')
sys.modules['eval.inspection_recording']=recording
load('eval.inspection_development_catalog',STAGE/'inspection_development_catalog.py')
load('eval.inspection_formal_population',STAGE/'inspection_formal_population.py')
load('eval.prepare_inspection_formal_population',STAGE/'prepare_inspection_formal_population.py')
module=load('eval.inspection_formal_execution',STAGE/'inspection_formal_execution.py')
producer=load('formal_execution_producer_candidate',STAGE/'prepare_inspection_formal_execution.py')


class ExecutionFixture(helper.DatasetFixture):
    def __init__(self,project):
        super().__init__(project)
        selection=self.put('qualification/synthetic-original-selection.json',
            {'role':'software fixture, not real selection','completed_slots':18,'root_count':6,'classic_best':'B1'})
        subject=self.read(self.subject);subject['classic_selection']=selection
        subject['qualification_evidence'].append(selection)
        self.subject=self.put(self.subject['path'],subject)
        review=self.read(self.review);review['subject_sha256']=self.subject['sha256'];self.review=self.put(self.review['path'],review)
        acceptance=self.read(self.acceptance)
        self.acceptance=self.put(self.acceptance['path'],{**acceptance,'subject':self.subject,'review':self.review})
        self.owner['rounds']=[];self.bind(acceptance['native_finding_id'],[self.acceptance,self.subject,self.review])
        self.selection=selection

    def execution(self,phase='main',opening=None):
        refs=self.dataset(phase,opening); intent=self.read(refs[0]); metadata=intent['root_metadata']
        order=literal_order(metadata,intent['schedule_seed'])
        folder='execution/'+phase+'.v1'
        order_ref=self.put(folder+'/episode-order.json',order)
        seeds={'role':'prospective_formal_sampler_seeds_not_provider_reproducibility_guarantee',
            'master':intent['sampler_master_seed'],'rng':'NumPy PCG64','numpy_version':'1.26.4',
            'by_order_index':{str(r['order_index']):7000+r['order_index'] for r in order['slots']
                              if recording.CONDITION_REPEATS[r['condition']]==3}}
        seed_ref=self.put(folder+'/sampler-seeds.json',seeds)
        count=len(metadata)
        value={'schema':'inspection-primary-execution-plan-v1','phase':phase,'status':'prepared_not_measurements',
            'g4_acceptance':self.acceptance,'namespace_intent':refs[0],'dataset_manifest':refs[1],
            'dataset_review':refs[2],'source_manifest':self.source,
            'model_registry':self.read(self.subject)['registry'],'classic_selection':self.selection,
            'episode_order':order_ref,'sampler_seeds':seed_ref,'root_denominator':count,
            'actual_episode_denominator':18*count,'root_mean_denominator':8*count,
            'native_finding_id':module.EXECUTION_FINDINGS[phase]}
        manifest=self.put(folder+'/manifest.json',value)
        self.bind(module.EXECUTION_FINDINGS[phase],[manifest,order_ref,seed_ref,self.selection])
        return refs,manifest

    def execution_proof(self,refs,execution,opening=None,verification=None):
        return module.read_execution_plan(self.run,self.project,self.acceptance,*refs,execution,
            verification=verification or self.verification,holdout_opening_value=opening)


def literal_order(metadata,seed):
    slots=[]
    for block,row in enumerate(metadata):
        for condition,count in recording.CONDITION_REPEATS.items():
            for repeat in range(count):
                slots.append({**row,'order_index':len(slots),'root_block_index':block,
                              'condition':condition,'repeat_ordinal':repeat})
    return {'role':'frozen_episode_order_not_a_measurement','seed':seed,'rng':'NumPy PCG64',
            'numpy_version':'1.26.4','cases':metadata,'slots':slots}


def test_primary_plan_and_exact_slot_bind_the_existing_full_matrix(tmp_path,monkeypatch):
    f=ExecutionFixture(tmp_path);refs,execution=f.execution()
    def no_rng(*args,**kwargs):raise AssertionError('reader cannot draw')
    monkeypatch.setattr(recording.np.random,'PCG64',no_rng)
    plan=f.execution_proof(refs,execution)
    assert len(plan.slots)==1296 and plan.classic_best=='B1'
    slot=next(row for row in plan.slots if row.condition=='B4-M1' and row.repeat_ordinal==2)
    episode=module.admit_primary_episode(f.run,f.project,f.acceptance,*refs,execution,
        slot.case_id,slot.condition,slot.repeat_ordinal,verification=f.verification)
    assert episode.slot==slot and episode.case.case_id==slot.case_id
    assert episode.worker_input(f.run,f.project/'runtime',f.project/'sdk')['dataset']==refs[1]


@pytest.mark.parametrize('fault',['missing_slot','repeat_bool','index_bool','block_bool','level_bool',
    'header_level_bool','untyped_slot','wrong_stratum','wrong_master','bad_sampler','missing_sampler',
    'extra_sampler','wrong_denominator','foreign_dataset'])
def test_fully_rehashed_bad_primary_execution_is_rejected(tmp_path,fault):
    f=ExecutionFixture(tmp_path);refs,execution=f.execution();value=f.read(execution)
    order=f.read(value['episode_order']);seeds=f.read(value['sampler_seeds'])
    if fault=='missing_slot':order['slots'].pop()
    elif fault=='repeat_bool':order['slots'][0]['repeat_ordinal']=False
    elif fault=='index_bool':order['slots'][0]['order_index']=False
    elif fault=='block_bool':order['slots'][0]['root_block_index']=False
    elif fault=='level_bool':order['slots'][0]['level']=True
    elif fault=='header_level_bool':order['cases'][0]['level']=True
    elif fault=='untyped_slot':order['slots'][0]=None
    elif fault=='wrong_stratum':order['slots'][0]['domain']='OOD'
    elif fault=='wrong_master':seeds['master']+=1
    elif fault=='bad_sampler':seeds['by_order_index'][next(iter(seeds['by_order_index']))]=True
    elif fault=='missing_sampler':seeds['by_order_index'].pop(next(iter(seeds['by_order_index'])))
    elif fault=='extra_sampler':seeds['by_order_index']['0']=8000
    elif fault=='wrong_denominator':value['actual_episode_denominator']=1295
    else:value['dataset_manifest']=f.catalog
    value['episode_order']=f.put(value['episode_order']['path'],order)
    value['sampler_seeds']=f.put(value['sampler_seeds']['path'],seeds)
    execution=f.put(execution['path'],value)
    with pytest.raises(ValueError):f.execution_proof(refs,execution)


def test_holdout_plan_retains12_roots216_slots_and_its_one_opening(tmp_path):
    f=ExecutionFixture(tmp_path);main=f.dataset();opening=f.typed_main_opening(main[1])
    refs,execution=f.execution('holdout',opening)
    proof=f.execution_proof(refs,execution,opening)
    assert len(proof.slots)==216 and proof.dataset.namespace.holdout_opening.path==opening['path']


@pytest.mark.parametrize('target',['manifest','order','seeds','selection','extra'])
def test_final_owner_snapshot_cannot_leave_stale_execution_artifacts(tmp_path,target):
    f=ExecutionFixture(tmp_path);refs,execution=f.execution();value=f.read(execution)
    chosen={'manifest':execution,'order':value['episode_order'],'seeds':value['sampler_seeds'],
            'selection':value['classic_selection']}
    original=f.verification.owner_reader;calls=0
    def change(run):
        nonlocal calls
        owner=original(run);calls+=1
        if calls==5:
            if target=='extra':(f.run/'execution/main.v1/extra.json').write_text('{}')
            else:
                path=f.run/chosen[target]['path'];path.write_bytes(path.read_bytes()+b' ')
        return owner
    with pytest.raises(ValueError):
        f.execution_proof(refs,execution,verification=replace(f.verification,owner_reader=change))
    assert calls==5


def test_primary_producer_uses_registered_masters_and_fake_adapter_after_gate(tmp_path,monkeypatch):
    f=ExecutionFixture(tmp_path);refs=f.dataset();seen=[]
    def fake_order(metadata,seed):seen.append(('order',seed));return literal_order(metadata,seed)
    class FakeGenerator:
        def __init__(self,bitgen):seen.append(('sampler',bitgen))
        def integers(self,low,high):assert (low,high)==(0,2**32);return 9000
    monkeypatch.setattr(producer,'episode_schedule',fake_order)
    monkeypatch.setattr(recording.np.random,'PCG64',lambda seed:('literal-pcg',seed))
    monkeypatch.setattr(recording.np.random,'Generator',FakeGenerator)
    ref=producer.prepare_primary_execution(f.run,f.project,f.acceptance,*refs,verification=f.verification)
    assert seen==[('order',50),('sampler',('literal-pcg',51))]
    with pytest.raises(ValueError):f.execution_proof(refs,ref)
    value=f.read(ref)
    f.bind(module.EXECUTION_FINDINGS['main'],[ref,value['episode_order'],value['sampler_seeds'],value['classic_selection']])
    assert len(f.execution_proof(refs,ref).slots)==1296
    with pytest.raises(ValueError,match='already issued|not regenerated'):
        producer.prepare_primary_execution(f.run,f.project,f.acceptance,*refs,verification=f.verification)
    assert len(seen)==2


def test_changed_sampler_value_cannot_be_accepted_by_rehashing_a_plan(tmp_path):
    f=ExecutionFixture(tmp_path);refs,execution=f.execution();value=f.read(execution)
    seeds=f.read(value['sampler_seeds']);key=next(iter(seeds['by_order_index']))
    seeds['by_order_index'][key]+=1
    value['sampler_seeds']=f.put(value['sampler_seeds']['path'],seeds)
    execution=f.put(execution['path'],value)
    with pytest.raises(ValueError):f.execution_proof(refs,execution)


def test_revoked_execution_cannot_admit_a_previously_valid_slot(tmp_path):
    f=ExecutionFixture(tmp_path);refs,execution=f.execution()
    f.bind('synthetic-execution-revocation',[f.catalog],revises=module.EXECUTION_FINDINGS['main'])
    with pytest.raises(ValueError):f.execution_proof(refs,execution)


def test_unadmitted_producer_does_not_construct_schedule_or_sampler(tmp_path,monkeypatch):
    def reject(*args,**kwargs):raise ValueError('unadmitted dataset')
    def no_rng(*args,**kwargs):raise AssertionError('no schedule or sampler before dataset gate')
    monkeypatch.setattr(producer,'read_dataset_proof',reject)
    monkeypatch.setattr(producer,'episode_schedule',no_rng)
    monkeypatch.setattr(recording.np.random,'PCG64',no_rng)
    with pytest.raises(ValueError,match='unadmitted dataset'):
        producer.prepare_primary_execution(tmp_path,tmp_path,{}, {}, {}, {},verification=object())
    assert not (tmp_path/'execution').exists()


def test_recording_guard_extraction_preserves_existing_scheduler_and_record_matrix_ast():
    baseline=HERE.parent/'paper/research/20261104-inspection/qualification/formal-installation.v1/before'
    old=ast.parse((baseline/'eval/inspection_recording.py').read_text())
    new=ast.parse((STAGE/'inspection_recording.py').read_text())
    functions=lambda tree:{node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    old,new=functions(old),functions(new)
    for name in ('episode_schedule','designs'):
        assert ast.dump(old[name])==ast.dump(new[name])
    body=old['record_matrix'].body
    start=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='slots' for t in node.targets))
    original_guard=body[start:start+4]
    assert [ast.dump(node) for node in original_guard]==[ast.dump(node) for node in new['validate_episode_order'].body[2:-1]]
    expanded=[]
    for node in new['record_matrix'].body:
        if (isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Name) and node.value.func.id=='validate_episode_order'):
            expanded.extend(original_guard)
        else:expanded.append(node)
    assert [ast.dump(node) for node in expanded]==[ast.dump(node) for node in body]
