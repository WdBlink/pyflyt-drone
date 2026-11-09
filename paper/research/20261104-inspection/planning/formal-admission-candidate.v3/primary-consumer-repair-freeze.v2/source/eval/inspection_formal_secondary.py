"""Fixed secondary plans derived from the admitted main primary plan.

No new roots, private decode, entropy, sampler draws or scored-data selection.
These plans describe additional actual flights; they do not invent rollouts.
"""
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from eval.inspection_episode_receipts import _mkdir, _sync_directory, durable_once
from eval.inspection_formal_admission import Artifact, file_sha, read_g4_proof, require_native_binding
from eval.inspection_formal_execution import EXECUTION_FINDINGS, ExecutionPlan, ExecutionSlot, read_execution_plan
from eval.inspection_formal_intervention import SECONDARY_ROOTS
from eval.inspection_namespace_admission import require_precedes
from eval.prepare_inspection_formal_population import _require_unissued

ARMS={'clean':('clean',),
    'ablation':('periodic','no-cross-aircraft-memory','no-progressive-confirmation-guidance'),
    'latency':('delay-00','delay-02','delay-05','delay-10','delay-20')}
FINDINGS={track:'inspection-main-secondary-'+track+'-execution-v1' for track in ARMS}
ORDER_POLICY='primary-root-order_then_SHA256-slot-order-v1'
SAMPLER_POLICY='reuse-exact-primary-root-condition-repeat-seed-v1'


@dataclass(frozen=True)
class SecondarySlot(ExecutionSlot):
    parent_order_index: int
    track: str
    arm: str


@dataclass(frozen=True)
class SecondaryPlan(ExecutionPlan):
    primary: ExecutionPlan
    track: str


def derive_secondary_slots(primary,track):
    if track not in ARMS or primary.dataset.namespace.phase!='main':
        raise ValueError('fixed main-only secondary track required')
    records={row.case_id:row for row in primary.dataset.records}
    if not set(SECONDARY_ROOTS)<=set(records) or any(records[root].level!=3 for root in SECONDARY_ROOTS):
        raise ValueError('the twelve preregistered L3 parents must exist in the admitted main dataset')
    groups={}
    for row in primary.slots:
        if row.case_id in SECONDARY_ROOTS and (track=='clean' or row.condition=='B4-M1'):
            groups.setdefault(row.case_id,[]).extend((row,arm) for arm in ARMS[track])
    if set(groups)!=set(SECONDARY_ROOTS):raise ValueError('complete twelve-parent secondary grid required')
    result=[]
    for block,rows in enumerate(groups.values()):
        def key(pair):
            row,arm=pair
            identity=f'{primary.manifest.sha256}/{track}/{row.case_id}/{row.condition}/{row.repeat_ordinal}/{arm}'
            return hashlib.sha256(identity.encode()).hexdigest(),identity
        for row,arm in sorted(rows,key=key):
            result.append(SecondarySlot(len(result),block,row.case_id,row.domain,row.level,row.condition,
                row.repeat_ordinal,row.sampler_seed,row.order_index,track,arm))
    return tuple(result)


def _references(primary):
    ref=lambda value:{'path':value.path,'sha256':value.sha256}
    return {'primary_execution':ref(primary.manifest),'dataset_manifest':ref(primary.dataset.manifest),
        'g4_acceptance':ref(primary.dataset.namespace.g4_acceptance),
        'source_manifest':ref(primary.dataset.namespace.source_manifest),'model_registry':ref(primary.registry)}


def prepare_secondary_execution(run,project_root,g4,namespace,dataset,dataset_review,primary_value,track,
                                *,verification):
    root=Path(run).resolve()
    primary=read_execution_plan(root,project_root,g4,namespace,dataset,dataset_review,primary_value,
                                verification=verification)
    if track not in ARMS:raise ValueError('declared secondary track required')
    proof=read_g4_proof(root,project_root,g4,verification=verification)
    _require_unissued(root,proof,FINDINGS[track],verification)
    folder=root/'execution'/('secondary-'+track+'.v1')
    if folder.exists() or folder.is_symlink() or folder.parent.is_symlink():
        raise ValueError('existing/partial secondary plan is preserved, never replaced')
    _mkdir(folder.parent);folder.mkdir();_sync_directory(folder.parent)
    ref=lambda path:{'path':str(path.relative_to(root)),'sha256':file_sha(path)}
    try:
        slots=derive_secondary_slots(primary,track)
        durable_once(folder/'episode-order.json',{'schema':'inspection-secondary-order-v1','track':track,
            'order_policy':ORDER_POLICY,'sampler_policy':SAMPLER_POLICY,
            'slots':[row.__dict__ for row in slots]})
        fresh=read_execution_plan(root,project_root,g4,namespace,dataset,dataset_review,primary_value,
                                   verification=verification)
        if fresh!=primary:raise ValueError('main primary/source changed during secondary preparation')
        count={'clean':96,'ablation':36,'latency':60}[track]
        durable_once(folder/'manifest.json',{'schema':'inspection-secondary-execution-plan-v1','phase':'main',
            'track':track,'status':'prepared_not_measurements',**_references(primary),
            'root_denominator':12,'actual_episode_denominator':len(slots),'root_mean_denominator':count,
            'episode_order':ref(folder/'episode-order.json'),'native_finding_id':FINDINGS[track]})
    except BaseException as error:
        durable_once(folder/'preparation-failure.json',{'status':'blocked_no_replacement',
            'error_type':type(error).__name__,'reason':str(error)})
        raise
    return ref(folder/'manifest.json')


def read_secondary_execution(run,project_root,g4,namespace,dataset,dataset_review,primary_value,secondary_value,
                             *,verification):
    root=Path(run).resolve()
    primary=read_execution_plan(root,project_root,g4,namespace,dataset,dataset_review,primary_value,
                                verification=verification)
    ref=Artifact.parse(secondary_value);value=ref.json(root)
    if type(value) is not dict:raise ValueError('typed secondary plan object required')
    track=value.get('track')
    fields={'schema','phase','track','status','primary_execution','dataset_manifest','g4_acceptance',
        'source_manifest','model_registry','root_denominator','actual_episode_denominator',
        'root_mean_denominator','episode_order','native_finding_id'}
    if (set(value)!=fields or track not in ARMS or value['schema']!='inspection-secondary-execution-plan-v1'
            or value['phase']!='main' or value['status']!='prepared_not_measurements'
            or value['native_finding_id']!=FINDINGS[track]
            or ref.path!='execution/secondary-'+track+'.v1/manifest.json'):
        raise ValueError('typed canonical main secondary execution plan required')
    if any(value[key]!=expected for key,expected in _references(primary).items()):
        raise ValueError('secondary plan differs from the admitted main parent/method/source')
    slots=derive_secondary_slots(primary,track)
    for key,expected in [('root_denominator',12),('actual_episode_denominator',len(slots)),
                         ('root_mean_denominator',{'clean':96,'ablation':36,'latency':60}[track])]:
        if type(value[key]) is not int or value[key]!=expected:raise ValueError('complete secondary denominator required')
    order=Artifact.parse(value['episode_order'])
    if order.path!=str(Path(ref.path).parent/'episode-order.json'):
        raise ValueError('canonical secondary order required')
    expected={'schema':'inspection-secondary-order-v1','track':track,'order_policy':ORDER_POLICY,
        'sampler_policy':SAMPLER_POLICY,'slots':[row.__dict__ for row in slots]}
    import json
    if json.dumps(order.json(root),sort_keys=True)!=json.dumps(expected,sort_keys=True):
        raise ValueError('secondary order/arm/seed differs from the fixed primary-derived full matrix')
    owner=verification.owner_reader(root)
    fresh=read_execution_plan(root,project_root,g4,namespace,dataset,dataset_review,primary_value,
        verification=replace(verification,owner_reader=lambda _:owner))
    if fresh!=primary:raise ValueError('main primary changed during secondary admission')
    require_native_binding(owner,FINDINGS[track],[ref,order])
    require_precedes(owner,EXECUTION_FINDINGS['main'],FINDINGS[track])
    for item in (ref,order):item.checked_path(root)
    paths=list((root/Path(ref.path).parent).rglob('*'))
    if (any(path.is_symlink() for path in paths) or {str(path.relative_to(root)) for path in paths if path.is_file()}
            !={ref.path,order.path}):raise ValueError('exclusive complete secondary plan tree required')
    return SecondaryPlan(ref,primary.dataset,primary.registry,primary.classic_selection,primary.classic_best,
        order,primary.sampler_seeds,slots,primary,track,owner_snapshot=owner)
