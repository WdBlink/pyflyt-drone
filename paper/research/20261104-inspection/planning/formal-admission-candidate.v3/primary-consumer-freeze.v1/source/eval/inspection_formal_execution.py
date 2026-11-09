"""Read a fixed primary eight-condition execution plan; no schedule/draw/spawn.

Dataset/native provenance is revalidated at each effect boundary. The plan
binds the existing recorder's order and prospective sampler map, never a score.
Secondary interventions are deliberately separate from this primary schema.
"""
from dataclasses import dataclass, replace
from pathlib import Path

from eval.inspection_dataset_admission import DATASET_FINDINGS, DatasetCase, DatasetProof
from eval.inspection_dataset_admission import read_dataset_proof
from eval.inspection_formal_admission import Artifact, require_native_binding
from eval.inspection_namespace_admission import require_precedes, unsigned
from eval.inspection_recording import CONDITION_REPEATS, validate_episode_order

EXECUTION_FINDINGS={'main':'inspection-main-primary-execution-v1',
                    'holdout':'inspection-holdout-primary-execution-v1'}


@dataclass(frozen=True)
class ExecutionSlot:
    order_index: int
    root_block_index: int
    case_id: str
    domain: str
    level: int
    condition: str
    repeat_ordinal: int
    sampler_seed: int | None


@dataclass(frozen=True)
class ExecutionPlan:
    manifest: Artifact
    dataset: DatasetProof
    registry: Artifact
    classic_selection: Artifact
    classic_best: str
    episode_order: Artifact
    sampler_seeds: Artifact
    slots: tuple


@dataclass(frozen=True)
class FormalEpisode:
    execution: ExecutionPlan
    slot: ExecutionSlot
    case: DatasetCase

    def worker_input(self, run, runtime, sdk):
        ref = lambda value: {'path': value.path, 'sha256': value.sha256}
        data = self.execution.dataset
        return {'schema':'inspection-formal-worker-input-v1', 'run':str(Path(run).resolve()),
            'runtime':str(Path(runtime).resolve()), 'sdk':str(Path(sdk).resolve()),
            'g4':ref(data.namespace.g4_acceptance), 'namespace':ref(data.namespace.intent),
            'dataset':ref(data.manifest), 'dataset_review':ref(data.review),
            'case_id':self.case.case_id,
            'opening':None if data.namespace.holdout_opening is None else ref(data.namespace.holdout_opening)}


def _selection(root, dataset):
    acceptance = dataset.namespace.g4_acceptance.json(root)
    subject = Artifact.parse(acceptance['subject']).json(root)
    selection = Artifact.parse(subject.get('classic_selection'))
    if {'path':selection.path,'sha256':selection.sha256} not in subject['qualification_evidence']:
        raise ValueError('original classical selection must belong to the qualified G4 evidence')
    value = selection.json(root)
    if (type(value.get('completed_slots')) is not int or value['completed_slots'] != 18
            or type(value.get('root_count')) is not int or value['root_count'] != 6
            or value.get('classic_best') not in ('B1','B2')):
        raise ValueError('complete original18-slot B1/B2 selection required')
    return selection, value['classic_best'], Artifact.parse(subject['registry'])


def read_execution_plan(run, project_root, acceptance_value, intent_value, dataset_value,
                        dataset_review_value, execution_value, *, verification,
                        holdout_opening_value=None):
    root = Path(run).resolve()
    data = read_dataset_proof(root, project_root, acceptance_value, intent_value, dataset_value,
        dataset_review_value, verification=verification, holdout_opening_value=holdout_opening_value)
    phase = data.namespace.phase; folder = 'execution/'+phase+'.v1'
    execution = Artifact.parse(execution_value)
    if execution.path != folder+'/manifest.json':
        raise ValueError('one canonical phase primary execution plan required')
    value = execution.json(root)
    fields = {'schema','phase','status','g4_acceptance','namespace_intent','dataset_manifest',
        'dataset_review','source_manifest','model_registry','classic_selection','episode_order',
        'sampler_seeds','root_denominator','actual_episode_denominator','root_mean_denominator','native_finding_id'}
    if (type(value) is not dict or set(value) != fields
            or value['schema'] != 'inspection-primary-execution-plan-v1'
            or value['phase'] != phase or value['status'] != 'prepared_not_measurements'
            or value['native_finding_id'] != EXECUTION_FINDINGS[phase]):
        raise ValueError('typed full primary execution plan required')
    selection, classic_best, registry = _selection(root,data)
    for key, expected in [('g4_acceptance', data.namespace.g4_acceptance),
            ('namespace_intent',data.namespace.intent), ('dataset_manifest',data.manifest),
            ('dataset_review',data.review), ('source_manifest',data.namespace.source_manifest),
            ('model_registry',registry), ('classic_selection',selection)]:
        if Artifact.parse(value[key]) != expected:
            raise ValueError('execution plan changed an admitted dataset/method subject')
    count = len(data.records)
    for key, expected in [('root_denominator',count), ('actual_episode_denominator',count*18),
                           ('root_mean_denominator',count*8)]:
        if type(value[key]) is not int or value[key] != expected:
            raise ValueError('primary execution must retain its complete planned denominator')
    order_ref, seed_ref = Artifact.parse(value['episode_order']), Artifact.parse(value['sampler_seeds'])
    if order_ref.path != folder+'/episode-order.json' or seed_ref.path != folder+'/sampler-seeds.json':
        raise ValueError('canonical prospective order and sampler files required')
    order, seeds = order_ref.json(root), seed_ref.json(root)
    metadata = [{'case_id':row.case_id,'domain':row.domain,'level':row.level} for row in data.records]
    if (type(order) is not dict or set(order) != {'role','seed','rng','numpy_version','cases','slots'}
            or order['role'] != 'frozen_episode_order_not_a_measurement'
            or type(order['seed']) is not int or order['seed'] != data.namespace.schedule_seed
            or order['rng'] != 'NumPy PCG64' or order['numpy_version'] != '1.26.4'
            or type(order['cases']) is not list or order['cases'] != metadata
            or any(type(row) is not dict or set(row) != {'case_id','domain','level'}
                   or type(row['level']) is not int for row in order['cases'])
            or type(order['slots']) is not list
            or any(type(row) is not dict or set(row) != {'order_index','root_block_index','case_id','domain',
                   'level','condition','repeat_ordinal'}
                   or any(type(row[key]) is not int for key in ('order_index','root_block_index','level','repeat_ordinal'))
                   or type(row['case_id']) is not str or type(row['condition']) is not str
                   for row in order['slots'])):
        raise ValueError('prospective order differs from the registered phase/master/metadata')
    rows = validate_episode_order([row.case_id for row in data.records],order)
    if (type(seeds) is not dict or set(seeds) != {'role','master','rng','numpy_version','by_order_index'}
            or seeds['role'] != 'prospective_formal_sampler_seeds_not_provider_reproducibility_guarantee'
            or type(seeds['master']) is not int or seeds['master'] != data.namespace.sampler_master_seed
            or seeds['rng'] != 'NumPy PCG64' or seeds['numpy_version'] != '1.26.4'
            or type(seeds['by_order_index']) is not dict):
        raise ValueError('prospective sampler map differs from its registered master/environment')
    expected_seeds = {str(row['order_index']) for row in rows if CONDITION_REPEATS[row['condition']]==3}
    if set(seeds['by_order_index']) != expected_seeds:
        raise ValueError('one sampler seed for every actual LLM slot, no missing/extra keys')
    case_metadata = {row.case_id:row for row in data.records}; slots=[]
    for row in rows:
        case = case_metadata[row['case_id']]
        if row['domain'] != case.domain or row['level'] != case.level:
            raise ValueError('execution slot changed its admitted root stratum')
        sampler = (None if CONDITION_REPEATS[row['condition']]==1 else
                   unsigned(seeds['by_order_index'][str(row['order_index'])],32))
        slots.append(ExecutionSlot(**row,sampler_seed=sampler))
    owner = verification.owner_reader(root)
    fresh = read_dataset_proof(root, project_root, acceptance_value, intent_value, dataset_value,
        dataset_review_value, verification=replace(verification,owner_reader=lambda _:owner),
        holdout_opening_value=holdout_opening_value)
    if fresh != data:
        raise ValueError('dataset changed during primary execution admission')
    require_native_binding(owner,EXECUTION_FINDINGS[phase],[execution,order_ref,seed_ref,selection])
    require_precedes(owner,DATASET_FINDINGS[phase],EXECUTION_FINDINGS[phase])
    for ref in (execution, selection, registry, order_ref, seed_ref): ref.checked_path(root)
    paths=list((root/folder).rglob('*'))
    if (any(path.is_symlink() for path in paths) or {str(p.relative_to(root)) for p in paths if p.is_file()}
            != {execution.path,order_ref.path,seed_ref.path}):
        raise ValueError('complete exclusive primary plan tree required')
    return ExecutionPlan(execution,data,registry,selection,classic_best,order_ref,seed_ref,tuple(slots))


def admit_primary_episode(run, project_root, acceptance_value, intent_value, dataset_value,
                          dataset_review_value, execution_value, case_id, condition, repeat_ordinal,
                          *, verification, holdout_opening_value=None):
    plan = read_execution_plan(run, project_root, acceptance_value, intent_value, dataset_value,
        dataset_review_value, execution_value, verification=verification,
        holdout_opening_value=holdout_opening_value)
    if (type(case_id) is not str or type(condition) is not str or type(repeat_ordinal) is not int
            or condition not in CONDITION_REPEATS or not 0 <= repeat_ordinal < CONDITION_REPEATS[condition]):
        raise ValueError('exact primary case/condition/repetition required')
    slots = [row for row in plan.slots if (row.case_id,row.condition,row.repeat_ordinal)
             == (case_id,condition,repeat_ordinal)]
    if len(slots) != 1: raise ValueError('primary episode is not in the full frozen execution plan')
    case = next(row for row in plan.dataset.records if row.case_id==case_id)
    return FormalEpisode(plan,slots[0],case)
