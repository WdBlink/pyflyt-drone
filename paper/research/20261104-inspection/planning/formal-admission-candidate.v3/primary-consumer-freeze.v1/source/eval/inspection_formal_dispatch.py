"""Effect-boundary primary dispatch admission; no private decode or execution."""
from dataclasses import dataclass
from pathlib import Path

from eval.inspection_episode_receipts import _mkdir, canonical_digest, durable_once
from eval.inspection_formal_admission import Artifact, context_from_environment, file_sha, parse_json_bytes
from eval.inspection_formal_execution import FormalEpisode, admit_primary_episode


@dataclass(frozen=True)
class DispatchAdmission:
    packet: Artifact
    episode: FormalEpisode
    worker_input: dict

    def binding(self):
        slot=self.episode.slot
        return {'packet':{'path':self.packet.path,'sha256':self.packet.sha256},
            'phase':self.episode.execution.dataset.namespace.phase,
            'execution':{'path':self.episode.execution.manifest.path,
                         'sha256':self.episode.execution.manifest.sha256},
            'order_index':slot.order_index,'case_id':slot.case_id,'condition':slot.condition,
            'repeat_ordinal':slot.repeat_ordinal,'sampler_seed':slot.sampler_seed}

    def frozen_sources(self, run, project_root):
        closure=self.episode.execution.dataset.namespace.source_manifest.json(run)
        project=Path(project_root).resolve()
        return {str(project/name):Artifact(name,sha).checked_path(project).read_text()
                for name,sha in closure['sources'].items()}

    def case_binding(self):
        data=self.episode.execution.dataset;case=self.episode.case;namespace=data.namespace
        ref=lambda value:{'path':value.path,'sha256':value.sha256}
        return {'phase':namespace.phase,'case_id':case.case_id,
            'dataset_manifest':ref(data.manifest),'dataset_review':ref(data.review),
            'namespace_intent':ref(namespace.intent),'g4_acceptance':ref(namespace.g4_acceptance),
            'native_dataset_request_sha256':data.native_request_sha256,
            'native_namespace_request_sha256':namespace.native_request_sha256,
            'private_copy_sha256':case.case.sha256,'case_sha256':case.case_sha256,
            'configuration_sha256':case.configuration_sha256}


def cli_dispatch(run, project_root, path, sha256, *, case_path, condition, sampler_seed=None):
    if path is None and sha256 is None:return None
    if run is None or path is None or sha256 is None:
        raise ValueError('formal CLI requires the run and full dispatch artifact')
    result=admit_dispatch(run,project_root,{'path':path,'sha256':sha256})
    if (Path(case_path).resolve()!=result.episode.case.case.checked_path(Path(run).resolve())
            or condition!=result.episode.slot.condition or sampler_seed!=result.episode.slot.sampler_seed):
        raise ValueError('actual CLI input/condition/sampler differs from its admitted dispatch')
    return result


def admit_episode_plan(plan, project_root):
    """Revalidate before cache reuse, reservation or spawn; never decode a case."""
    value=plan.get('formal_dispatch')
    if (type(value) is not dict or set(value)!={'run','packet'} or type(value['run']) is not str):
        raise ValueError('formal episode requires its exact dispatch/run binding')
    admitted=admit_dispatch(value['run'],project_root,value['packet'])
    slot=admitted.episode.slot;case=admitted.episode.case
    root=Path(value['run']).resolve()
    if (type(plan.get('repeat_ordinal')) is not int
            or slot.sampler_seed is not None and type(plan.get('sampler_seed')) is not int
            or any(plan.get(key)!=getattr(slot,key) for key in ('case_id','condition','repeat_ordinal'))
            or plan.get('sampler_seed')!=slot.sampler_seed
            or Path(plan['case_path']).resolve()!=case.case.checked_path(root)
            or plan.get('case_file_sha256')!=case.case.sha256
            or plan.get('case_sha256')!=case.case_sha256
            or plan.get('suite_manifest_sha256')!=admitted.episode.execution.dataset.manifest.sha256
            or plan.get('max_clock_lag_seconds')!=.1 or type(plan.get('max_attempts')) is not int
            or plan['max_attempts']!=2 or plan.get('ablation') is not None
            or type(plan.get('injected_delay_seconds',0)) is not int or plan.get('injected_delay_seconds',0)!=0
            or plan.get('thread_environment')!={key:'1' for key in
                ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}):
        raise ValueError('episode plan differs from its admitted primary slot/input/protocol')
    if slot.sampler_seed is not None:
        registry=admitted.episode.execution.registry.json(root)
        profile=registry['models'][slot.condition.split('-')[1]]
        if (plan.get('model_registry_sha256')!=admitted.episode.execution.registry.sha256
                or plan.get('model_receipt_sha256')!=profile['download_receipt_sha256']
                or plan.get('generation_profile')!=profile['generation_profile']):
            raise ValueError('formal model identity/profile differs from the fixed registry')
    return admitted


def admit_model_settlement(plan, project_root, reservation, attempt_ordinal, meter):
    """A local settle acknowledgement alone cannot replace the native ledger."""
    admission=admit_episode_plan(plan,project_root)
    run=Path(plan['formal_dispatch']['run']).resolve()
    packet=admission.packet.json(run)
    verification=context_from_environment(run,project_root,packet['runtime'],packet['sdk'])
    owner=verification.owner_reader(run)
    entry=owner.get('usage_ledger',{}).get(reservation,{})
    purpose=parse_json_bytes(entry.get('purpose','{}').encode())
    receipt_path=f'qualification/local-model-episodes/{reservation}/usage-receipt.json'
    expected_purpose={'kind':'inspection_local_generation','phase':admission.binding()['phase'],
        'model_registry_sha256':plan['model_registry_sha256'],
        'model_receipt_sha256':plan['model_receipt_sha256'],
        'condition':plan['condition'],'case_id':plan['case_id'],'repeat_ordinal':plan['repeat_ordinal'],
        'episode_plan_sha256':canonical_digest(plan),'attempt_ordinal':attempt_ordinal,
        'formal_dispatch':plan['formal_dispatch']['packet'],'formal_dispatch_binding':admission.binding()}
    if (entry.get('status')!='known' or entry.get('unreserved') is not False
            or entry.get('reserved')!=plan['reservation_usage']
            or entry.get('reported_usage')!=meter['usage'] or purpose!=expected_purpose
            or entry.get('receipt')!=receipt_path or owner.get('frozen_plan')!=plan['owner_frozen_plan']):
        raise ValueError('formal model usage must be known and identically settled in the sole native owner')
    ref=Artifact.parse({'path':receipt_path,'sha256':entry.get('receipt_sha256')})
    value=ref.json(run)
    if (value.get('reservation')!=reservation or value.get('plan_sha256')!=canonical_digest(plan)
            or value.get('usage')!=meter['usage'] or value.get('resource_recount')!=meter):
        raise ValueError('native settlement receipt differs from the actual model recount')
    return ref


def admit_dispatch(run, project_root, packet_value):
    root=Path(run).resolve();project=Path(project_root).resolve()
    packet=Artifact.parse(packet_value);value=packet.json(root)
    fields={'schema','phase','runtime','sdk','g4','namespace','dataset','dataset_review','execution',
            'opening','case_id','condition','repeat_ordinal'}
    if (type(value) is not dict or set(value)!=fields
            or value['schema']!='inspection-primary-dispatch-v1'
            or value['phase'] not in ('main','holdout')
            or any(type(value[key]) is not str or not value[key] for key in ('runtime','sdk','case_id','condition'))
            or type(value['repeat_ordinal']) is not int):
        raise ValueError('typed fixed primary dispatch packet required')
    expected=f"dispatch/{value['phase']}.v1/{value['case_id']}--{value['condition']}--{value['repeat_ordinal']}.json"
    if packet.path!=expected:raise ValueError('canonical primary dispatch identity required')
    for key in ('g4','namespace','dataset','dataset_review','execution'):Artifact.parse(value[key])
    if value['opening'] is not None:Artifact.parse(value['opening'])
    verification=context_from_environment(root,project,value['runtime'],value['sdk'])
    episode=admit_primary_episode(root,project,value['g4'],value['namespace'],value['dataset'],
        value['dataset_review'],value['execution'],value['case_id'],value['condition'],
        value['repeat_ordinal'],verification=verification,holdout_opening_value=value['opening'])
    if episode.execution.dataset.namespace.phase!=value['phase']:
        raise ValueError('dispatch phase differs from its actual namespace')
    packet.checked_path(root)
    return DispatchAdmission(packet,episode,episode.worker_input(root,value['runtime'],value['sdk']))


def write_primary_dispatch(run, project_root, acceptance_value, intent_value, dataset_value,
                           dataset_review_value, execution_value, case_id, condition, repeat_ordinal,
                           *, runtime, sdk, verification, holdout_opening_value=None):
    root=Path(run).resolve()
    episode=admit_primary_episode(root,project_root,acceptance_value,intent_value,dataset_value,
        dataset_review_value,execution_value,case_id,condition,repeat_ordinal,verification=verification,
        holdout_opening_value=holdout_opening_value)
    phase=episode.execution.dataset.namespace.phase
    path=root/f'dispatch/{phase}.v1/{case_id}--{condition}--{repeat_ordinal}.json'
    value={'schema':'inspection-primary-dispatch-v1','phase':phase,
        'runtime':str(Path(runtime).resolve()),'sdk':str(Path(sdk).resolve()),
        'g4':acceptance_value,'namespace':intent_value,'dataset':dataset_value,
        'dataset_review':dataset_review_value,'execution':execution_value,'opening':holdout_opening_value,
        'case_id':case_id,'condition':condition,'repeat_ordinal':repeat_ordinal}
    _mkdir(path.parent)
    if path.exists():
        old=Artifact(str(path.relative_to(root)),file_sha(path))
        if old.json(root)!=value:raise ValueError('existing dispatch identity changed; preserve it')
    else:durable_once(path,value)
    return {'path':str(path.relative_to(root)),'sha256':file_sha(path)}
