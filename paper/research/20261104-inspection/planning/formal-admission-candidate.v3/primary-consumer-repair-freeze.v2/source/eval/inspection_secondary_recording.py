"""Secondary arm/repeat adapter over the provided ExperimentRecorder.

It records only the additional intervention flights and their root means.
Their primary B4-M1 comparator is joined separately in descriptive analysis.
"""
from contextlib import ExitStack
import hashlib
import math
from pathlib import Path
from statistics import mean

from eval.inspection_formal_secondary import ARMS


def record_secondary(root,plan,measure,*,context,recorder_type,write_json):
    if plan.track not in ('ablation','latency'):
        raise ValueError('three ablations or five latency arms required')
    root=Path(root);arms=list(ARMS[plan.track]);cases=sorted({row.case_id for row in plan.slots})
    expected={(case,repeat,arm) for case in cases for repeat in range(3) for arm in arms}
    actual=[(row.case_id,row.repeat_ordinal,row.arm) for row in plan.slots]
    if (len(cases)!=12 or len(actual)!=len(expected) or set(actual)!=expected
            or any(row.order_index!=index or row.condition!='B4-M1' for index,row in enumerate(plan.slots))):
        raise ValueError('complete fixed secondary root x repeat x arm grid required')
    common={**context,'cases':cases,'conditions':arms,'statistical_unit':'case',
        'failure_policy':'penalty','failure_value':0.,'metric':{'name':'Y','unit':'quality','direction':'higher'},
        'statistics':{'estimator':'mean','interval':'none','paired':True,'contrasts':[
            {'id':'descriptive-arm-'+str(i),'minuend':arm,'subtrahend':arms[0]}
            for i,arm in enumerate(arms[1:],1)]}}
    for group,seeds in [('episodes',[0,1,2]),('root-means',[0])]:
        write_json(root,group+'/design.json',{**common,'seeds':seeds,
            'experiment_id':context['experiment_id']+'-'+group,
            'adapter_semantics':{'primary_inference':'none; descriptive secondary interventions only',
                'seed_field':'repeat ordinal' if group=='episodes' else 'one aggregate pass, not a flight'}})
    write_json(root,'episode-order.json',{'role':'frozen secondary execution order, not measurements',
        'execution_manifest':{'path':plan.manifest.path,'sha256':plan.manifest.sha256},
        'slots':[row.__dict__ for row in plan.slots]})
    with ExitStack() as stack:
        records={group:stack.enter_context(recorder_type(root,design=group+'/design.json',
            results=group+'/results.json',raw_directory=group+'/raw')) for group in ('episodes','root-means')}
        values={}
        for slot in plan.slots:
            with records['episodes'].attempt(slot.case_id,slot.repeat_ordinal,slot.arm) as complete:
                write_json(root,f'starts/{slot.order_index:06d}.json',{
                    'role':'secondary attempt start, not completion',**slot.__dict__})
                value,evidence=measure(slot)
                if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                    raise ValueError('actual finite scored Y required; infrastructure failures are exceptions')
                complete(value,evidence);values[slot.case_id,slot.repeat_ordinal,slot.arm]=value
        records['episodes'].finish(outcome='inconclusive')
        for case,_,arm in records['root-means'].grid:
            with records['root-means'].attempt(case,0,arm) as complete:
                paths=[records['episodes'].rows[case,repeat,arm]['raw'] for repeat in range(3)]
                complete(mean(values[case,repeat,arm] for repeat in range(3)),{
                    'aggregation':'arithmetic mean within original independent parent root',
                    'actual_episode_count':3,'not_an_additional_episode':True,
                    'inner_attempts':[{'repeat_ordinal':repeat,'path':path,
                        'sha256':hashlib.sha256((root/path).read_bytes()).hexdigest()} for repeat,path in enumerate(paths)],
                    'inner_results':'episodes/results.json',
                    'inner_results_sha256':hashlib.sha256((root/'episodes/results.json').read_bytes()).hexdigest()})
        records['root-means'].finish(outcome='inconclusive')
    return {'independent_parent_roots':12,'additional_actual_episode_denominator':len(actual),
        'root_mean_rows':12*len(arms),'role':'secondary callbacks only; no confirmatory test or new roots'}
