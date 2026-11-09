"""Primary formal study adapter over the existing recorder and flight loops.

No namespace creation, sampling, G4 decision, primary-analysis lock or holdout
opening occurs here. Every actual callback must belong to a previously admitted
complete execution plan. Operational failures preserve the SDK denominator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from eval.inspection_episode_receipts import digest, durable_once, execute_local_episode
from eval.inspection_formal_admission import Artifact, context_from_environment, read_json
from eval.inspection_formal_dispatch import admit_dispatch, admit_episode_plan, write_primary_dispatch
from eval.inspection_formal_execution import read_execution_plan
from eval.inspection_measurement_loader import measure_completed_episode
from eval.inspection_recording import record_matrix
from eval.run_inspection_llm_episode import execute_development_llm_episode


def read_request(value):
    fields={'schema','run','runtime','sdk','g4','namespace','dataset','dataset_review','execution','opening'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='inspection-primary-run-v1'
            or any(type(value[key]) is not str or not value[key] for key in ('run','runtime','sdk'))):
        raise ValueError('complete primary-run request required')
    for key in ('g4','namespace','dataset','dataset_review','execution'):Artifact.parse(value[key])
    if value['opening'] is not None:Artifact.parse(value['opening'])
    return value


def admitted_request(request, project_root=ROOT):
    request=read_request(request)
    verification=context_from_environment(request['run'],project_root,request['runtime'],request['sdk'])
    proof=read_execution_plan(request['run'],project_root,request['g4'],request['namespace'],
        request['dataset'],request['dataset_review'],request['execution'],verification=verification,
        holdout_opening_value=request['opening'])
    return verification,proof


def episode_plan(admission, run, project_root, physics_python):
    root,project=Path(run).resolve(),Path(project_root).resolve()
    data,slot,case=admission.episode.execution.dataset,admission.episode.slot,admission.episode.case
    sources=admission.frozen_sources(root,project)
    closure=data.namespace.source_manifest.json(root)
    acceptance=data.namespace.g4_acceptance.json(root)
    subject=Artifact.parse(acceptance['subject']).json(root)
    registry=admission.episode.execution.registry.json(root)
    design=Artifact.parse(subject['design']).json(root)
    evaluator=Artifact.parse(design['evaluator'])
    result={'role':'formal_primary_episode_not_scientific_acceptance',
        'formal_dispatch':{'run':str(root),'packet':admission.binding()['packet']},
        'case_id':slot.case_id,'condition':slot.condition,'repeat_ordinal':slot.repeat_ordinal,
        'case_path':str(case.case.checked_path(root)),'case_file_sha256':case.case.sha256,
        'case_sha256':case.case_sha256,'suite_manifest_sha256':data.manifest.sha256,
        'max_attempts':2,'max_clock_lag_seconds':.1,'sampler_seed':slot.sampler_seed,
        'source_hashes':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()},
        'dependency_hashes':closure['dependencies'],
        'thread_environment':{key:'1' for key in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'supervisor_sha256':digest(project/'eval/run_inspection_formal.py'),
        'receipt_adapter_sha256':digest(project/'eval/inspection_episode_receipts.py'),
        'injected_delay_seconds':0,'ablation':None,
        'evaluator':str(evaluator.checked_path(root)),'evaluator_sha256':evaluator.sha256,
        'physics_python':str(Path(physics_python).absolute())}
    if slot.sampler_seed is not None:
        profile=registry['models'][slot.condition.split('-')[1]];limits=profile['generation_profile']
        result.update(model_registry_sha256=admission.episode.execution.registry.sha256,
            model_receipt_sha256=profile['download_receipt_sha256'],generation_profile=limits,
            owner_frozen_plan=subject['owner']['frozen_plan'],
            reservation_usage={'calls':limits['max_calls'],'tokens':limits['max_total_tokens']},
            service_sha256=digest(project/'eval/inspection_model_service.py'))
    admit_episode_plan(result,project)
    return result


def episode_command(plan, out, *, project_root, reservation=None):
    project=Path(project_root).resolve();ref=plan['formal_dispatch']['packet']
    common=['--run',plan['formal_dispatch']['run'],'--formal-dispatch',ref['path'],
        '--formal-dispatch-sha256',ref['sha256'],'--case',plan['case_path'],'--out',str(Path(out).resolve())]
    if plan['sampler_seed'] is None:
        return [plan['physics_python'],str(project/'eval/run_inspection_reference.py'),*common,
                '--policy',plan['condition'],'--layout','generated','--realtime','--duration','600']
    architecture,model=plan['condition'].split('-')
    return [plan['physics_python'],str(project/'eval/run_inspection_llm_development.py'),*common,
        '--reservation',reservation,'--architecture',architecture,'--model',model,
        '--sampler-seed',str(plan['sampler_seed'])]


def run_primary(request, output, *, project_root=ROOT, physics_python=sys.executable):
    request=read_request(request)
    project,root=Path(project_root).resolve(),Path(request['run']).resolve()
    verification,proof=admitted_request(request,project)
    phase=proof.dataset.namespace.phase
    threads={key:'1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
    if any(os.environ.get(key)!=value for key,value in threads.items()):
        raise ValueError('launch primary study with all four qualified thread limits')
    output=Path(output).absolute()
    if output.exists() or not output.resolve().is_relative_to(project/'logs'):
        raise ValueError('choose a new complete SDK envelope inside project logs')
    receipts=project/'logs/inspection-formal-episodes.v1'/phase
    receipts.mkdir(parents=True,exist_ok=True)
    with (receipts/'.matrix.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # The existing SDK executes its frozen order synchronously. A new SDK
        # envelope may only reuse exact completed immutable episode receipts.
        verification,proof=admitted_request(request,project)
        output.mkdir(parents=True)
        durable_once(output/'run-request.json',request)
        sys.path.insert(0,str(Path(request['sdk']).resolve()))
        from lib.experiments import ExperimentRecorder
        from lib.production import json_bytes, write_once
        def write(root,name,value):return write_once(root,name,json_bytes(value))
        cases=[row.case_id for row in proof.dataset.records]
        order=proof.episode_order.json(root)
        acceptance=proof.dataset.namespace.g4_acceptance.json(root)
        subject=Artifact.parse(acceptance['subject']).json(root)
        design_ref=Artifact.parse(subject['design'])
        design=deepcopy(design_ref.json(root))
        evaluator=Artifact.parse(design['evaluator'])
        write_once(output,evaluator.path,evaluator.checked_path(root).read_bytes())
        source_prefix=str((root/'experiment').relative_to(project))+'/'
        closure=proof.dataset.namespace.source_manifest.json(root)
        for name,sha in closure['sources'].items():
            if name.startswith(source_prefix+'evaluator-source'):
                relative=str((project/name).relative_to(root))
                write_once(output,relative,Artifact(name,sha).checked_path(project).read_bytes())
        if phase=='holdout':
            design.update(cases=cases,experiment_id='continuous-inspection-holdout-root-means-v1',
                hypothesis='Once-only descriptive direction and size after locked main analysis; no seventh confirmatory test.')
            design['recording_semantics'].update(root_denominator=12,actual_episode_denominator=216)
            design['recording_semantics']['root_metadata']=[
                {'case_id':row.case_id,'domain':row.domain,'level':row.level} for row in proof.dataset.records]
        elif design['cases']!=cases:
            raise ValueError('main recording cases differ from the accepted design')
        outer_path=design_ref.path if phase=='main' else 'root-means/design.json'
        if phase=='main':write_once(output,outer_path,design_ref.checked_path(root).read_bytes())
        else:write(output,outer_path,design)
        next_index=0
        def measure(case_id,repeat,condition):
            nonlocal next_index
            slot=proof.slots[next_index]
            if (case_id,repeat,condition)!=(slot.case_id,slot.repeat_ordinal,slot.condition):
                raise ValueError('recorder callback differs from the next frozen execution slot')
            packet=write_primary_dispatch(root,project,request['g4'],request['namespace'],request['dataset'],
                request['dataset_review'],request['execution'],case_id,condition,repeat,
                runtime=request['runtime'],sdk=request['sdk'],verification=verification,
                holdout_opening_value=request['opening'])
            admission=admit_dispatch(root,project,packet)
            plan=episode_plan(admission,root,project,physics_python)
            directory=receipts/f'{case_id}-{condition}-{repeat}'
            if slot.sampler_seed is None:
                execute_local_episode(directory,plan,lambda out:episode_command(plan,out,project_root=project))
            else:
                prefix=f'formal-{phase}-{request["g4"]["sha256"][:12]}-{slot.order_index:04d}'
                execute_development_llm_episode(directory,plan,
                    lambda out,reservation:episode_command(plan,out,project_root=project,reservation=reservation),
                    run=root,reservation_prefix=prefix)
            value,evidence=measure_completed_episode(directory,{**plan,'phase':phase})
            next_index+=1
            print(json.dumps({**admission.binding(),'Y':value,'completed_callbacks':next_index}),flush=True)
            return value,evidence
        context={key:design[key] for key in ('claim_id','hypothesis','independent_variable',
            'dependent_variable','control_variables','data_roles','planned_at')}
        context['experiment_id']=design['experiment_id']+'-inner'
        result=record_matrix(output,cases,measure,context=context,classic_best=proof.classic_best,
            execution_order=order,recorder_type=ExperimentRecorder,write_json=write,
            frozen_outer_design=outer_path)
        admitted_request(request,project)
        write(output,'primary-callbacks-complete.json',{**result,'phase':phase,
            'role':'full source-bound callbacks, not primary analysis or scientific acceptance',
            'execution_manifest':request['execution']})
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run_primary(read_request(read_json(args.request)),args.out)


if __name__=='__main__':main()
