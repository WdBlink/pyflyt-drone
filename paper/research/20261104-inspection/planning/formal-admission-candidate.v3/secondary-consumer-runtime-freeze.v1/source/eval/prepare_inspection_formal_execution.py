"""Prepare one deterministic primary order/sampler plan after dataset admission.

Uses the existing recording adapter. No private generation, model, flight or
extra scheduler. Existing/partial plans are preserved, never replaced.
"""
from pathlib import Path

from eval.inspection_dataset_admission import read_dataset_proof
from eval.inspection_episode_receipts import _mkdir, _sync_directory, durable_once
from eval.inspection_formal_admission import file_sha, read_g4_proof
from eval.inspection_formal_execution import EXECUTION_FINDINGS, _selection
from eval.inspection_recording import CONDITION_REPEATS, episode_schedule
from eval.prepare_inspection_formal_population import _require_unissued


def prepare_primary_execution(run, project_root, acceptance_value, intent_value, dataset_value,
                              dataset_review_value, *, verification, holdout_opening_value=None):
    data = read_dataset_proof(run, project_root, acceptance_value, intent_value, dataset_value,
        dataset_review_value, verification=verification, holdout_opening_value=holdout_opening_value)
    g4=read_g4_proof(run,project_root,acceptance_value,verification=verification)
    _require_unissued(Path(run).resolve(),g4,EXECUTION_FINDINGS[data.namespace.phase],verification)
    # Import/check the numerical recorder environment after the causal gate,
    # before any actual order or sampler construction.
    import numpy as np
    if np.__version__ != '1.26.4':
        raise ValueError('primary schedule/sampler preparation requires NumPy1.26.4')
    root = Path(run).resolve(); folder = root/'execution'/(data.namespace.phase+'.v1')
    if folder.is_symlink() or folder.exists() or folder.parent.is_symlink():
        raise ValueError('existing/partial primary execution plan must be restored, not regenerated')
    selection, best, registry = _selection(root,data)
    _mkdir(folder.parent); folder.mkdir(); _sync_directory(folder.parent)
    ref = lambda path: {'path':str(path.relative_to(root)),'sha256':file_sha(path)}
    art = lambda value: {'path':value.path,'sha256':value.sha256}
    try:
        metadata = [{'case_id':row.case_id,'domain':row.domain,'level':row.level} for row in data.records]
        order = episode_schedule(metadata,data.namespace.schedule_seed)
        rng = np.random.Generator(np.random.PCG64(data.namespace.sampler_master_seed))
        seeds = {str(row['order_index']):int(rng.integers(0,2**32)) for row in order['slots']
                 if CONDITION_REPEATS[row['condition']]==3}
        durable_once(folder/'episode-order.json',order)
        durable_once(folder/'sampler-seeds.json',{
            'role':'prospective_formal_sampler_seeds_not_provider_reproducibility_guarantee',
            'master':data.namespace.sampler_master_seed,'rng':'NumPy PCG64','numpy_version':np.__version__,
            'by_order_index':seeds})
        fresh = read_dataset_proof(run, project_root, acceptance_value, intent_value, dataset_value,
            dataset_review_value, verification=verification,holdout_opening_value=holdout_opening_value)
        if fresh != data: raise ValueError('dataset/source changed during primary plan preparation')
        count=len(data.records)
        value={'schema':'inspection-primary-execution-plan-v1','phase':data.namespace.phase,
            'status':'prepared_not_measurements','g4_acceptance':acceptance_value,'namespace_intent':intent_value,
            'dataset_manifest':dataset_value,'dataset_review':dataset_review_value,
            'source_manifest':art(data.namespace.source_manifest),'model_registry':art(registry),
            'classic_selection':art(selection),'episode_order':ref(folder/'episode-order.json'),
            'sampler_seeds':ref(folder/'sampler-seeds.json'),'root_denominator':count,
            'actual_episode_denominator':count*18,'root_mean_denominator':count*8,
            'native_finding_id':EXECUTION_FINDINGS[data.namespace.phase]}
        durable_once(folder/'manifest.json',value)
        manifest=ref(folder/'manifest.json')
    except BaseException as error:
        durable_once(folder/'preparation-failure.json',{'status':'blocked_no_replacement',
            'error_type':type(error).__name__,'reason':str(error)})
        raise
    return manifest
