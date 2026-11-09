"""Additional clean/ablation/latency flights over the same admitted main roots.

All five latency arms, including zero, use fresh actual flights; no optional
primary-trace reuse is attempted. This stays within the planned180 slots.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from eval.inspection_formal_admission import Artifact, read_json
from eval.inspection_formal_dispatch import admit_dispatch, write_secondary_dispatch
from eval.inspection_formal_secondary import read_secondary_execution
from eval.inspection_measurement_loader import measure_completed_episode
from eval.inspection_recording import record_matrix
from eval.inspection_secondary_recording import record_secondary
from eval.run_inspection_formal import admitted_request, checkpoint_before_dispatch, episode_command, episode_plan, read_request
from eval.inspection_episode_receipts import durable_once, execute_local_episode
from eval.run_inspection_llm_episode import execute_development_llm_episode


def read_secondary_request(value):
    if (type(value) is not dict or value.get('schema')!='inspection-secondary-run-v1'
            or 'secondary_execution' not in value):raise ValueError('typed secondary request required')
    base={key:content for key,content in value.items() if key!='secondary_execution'}
    base['schema']='inspection-primary-run-v1';read_request(base)
    Artifact.parse(value['secondary_execution'])
    if value['opening'] is not None:raise ValueError('secondary track cannot open holdout')
    return value,base


def run_secondary(request,output,*,project_root=ROOT,physics_python=sys.executable):
    request,base=read_secondary_request(request)
    root,project=Path(request['run']).resolve(),Path(project_root).resolve()
    verification,primary=admitted_request(base,project)
    def admit():
        return read_secondary_execution(root,project,request['g4'],request['namespace'],request['dataset'],
            request['dataset_review'],request['execution'],request['secondary_execution'],verification=verification)
    proof=admit()
    if any(os.environ.get(key)!='1' for key in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')):
        raise ValueError('qualified four thread limits required')
    output=Path(output).absolute()
    if output.exists() or not output.resolve().is_relative_to(project/'logs'):
        raise ValueError('new complete secondary SDK envelope inside logs required')
    receipts=project/'logs/inspection-formal-episodes.v1'/('secondary-'+proof.track)
    receipts.mkdir(parents=True,exist_ok=True)
    with (receipts.parent/'.matrix.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if admit()!=proof:raise ValueError('secondary plan changed before recording')
        output.mkdir(parents=True);durable_once(output/'run-request.json',request)
        sys.path.insert(0,str(Path(request['sdk']).resolve()))
        from lib.experiments import ExperimentRecorder
        from lib.production import write_once, json_bytes
        def write(root,name,value):return write_once(root,name,json_bytes(value))
        subject=Artifact.parse(request['g4']).json(root)['subject']
        design=Artifact.parse(Artifact.parse(subject).json(root)['design']).json(root)
        context={key:design[key] for key in ('claim_id','independent_variable','dependent_variable',
                                            'control_variables','data_roles','planned_at')}
        context.update(experiment_id='continuous-inspection-secondary-'+proof.track+'-v1',
            hypothesis='Prespecified descriptive within-parent intervention analysis; no additional confirmatory tests.')
        next_index=0
        def measure(slot):
            nonlocal next_index
            if slot!=proof.slots[next_index]:raise ValueError('secondary callback violates its fixed execution order')
            checkpoint_before_dispatch(output, slot, admit)
            packet=write_secondary_dispatch(root,project,request['g4'],request['namespace'],request['dataset'],
                request['dataset_review'],request['execution'],request['secondary_execution'],
                slot.case_id,slot.condition,slot.repeat_ordinal,slot.track,slot.arm,
                runtime=request['runtime'],sdk=request['sdk'],verification=verification)
            admission=admit_dispatch(root,project,packet)
            plan=episode_plan(admission,root,project,physics_python)
            directory=receipts/f'{slot.case_id}-{slot.condition}-{slot.arm}-{slot.repeat_ordinal}'
            if slot.sampler_seed is None:
                execute_local_episode(directory,plan,lambda out:episode_command(plan,out,project_root=project))
            else:
                prefix=f'formal-{proof.track}-{request["g4"]["sha256"][:12]}-{slot.order_index:04d}'
                execute_development_llm_episode(directory,plan,
                    lambda out,reservation:episode_command(plan,out,project_root=project,reservation=reservation),
                    run=root,reservation_prefix=prefix)
            value,evidence=measure_completed_episode(directory,{**plan,'phase':'main'})
            next_index+=1
            return value,evidence
        if proof.track=='clean':
            slots={(row.case_id,row.repeat_ordinal,row.condition):row for row in proof.slots}
            # Reuse the original full eight-condition recorder unchanged.
            order={'slots':[{key:value for key,value in row.__dict__.items() if key not in
                    ('parent_order_index','track','arm','sampler_seed')} for row in proof.slots],
                'cases':[{'case_id':row.case_id,'domain':row.domain,'level':row.level}
                    for row in proof.dataset.records if row.case_id in {s.case_id for s in proof.slots}]}
            result=record_matrix(output,sorted({row.case_id for row in proof.slots}),
                lambda case,repeat,condition:measure(slots[case,repeat,condition]),context=context,
                classic_best=proof.classic_best,execution_order=order,recorder_type=ExperimentRecorder,write_json=write)
        else:
            result=record_secondary(output,proof,measure,context=context,
                                    recorder_type=ExperimentRecorder,write_json=write)
        if admit()!=proof:raise ValueError('secondary source/plan changed during execution')
        write(output,'secondary-callbacks-complete.json',{**result,'track':proof.track,
            'execution_manifest':request['secondary_execution'],'new_independent_roots':0})
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run_secondary(read_json(args.request),args.out)


if __name__=='__main__':main()
