"""Independent guard/SR arithmetic probes and read-only actual cache audit."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import runpy
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[5]
RESEARCH=REPO/'paper/research/20261104-inspection'
sys.path.insert(0,str(REPO))
from eval import inspection_episode_receipts as receipts
from eval.inspection_recording import CONDITION_REPEATS
from eval.inspection_statistics import CONDITIONS, MAIN_SHAPE, SEEDS
from eval.inspection_secondary_statistics import sr_root_table,sr_cluster_intervals


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def quantile(a,q):
    a=sorted(a);position=(len(a)-1)*q;lo=math.floor(position);hi=math.ceil(position)
    return a[lo]+(position-lo)*(a[hi]-a[lo])


def sr_data(shape):
    metadata=[];episodes=[]
    for domain,levels in shape.items():
        for level,n in levels.items():
            for i in range(n):
                root=f'synthetic-SR-{domain}-{level}-{i:02d}'
                metadata.append(dict(case_id=root,lineage_group=root,domain=domain,level=level))
                for k,(condition,repeats) in enumerate(CONDITION_REPEATS.items()):
                    for j in range(repeats):episodes.append(dict(case_id=root,condition=condition,repeat_ordinal=j,status='completed',SR=(i+j+k+level)%2))
    return metadata,episodes


def sr_reference(metadata,episodes,draws):
    by_root={r['case_id']:{c:[] for c in CONDITIONS} for r in metadata}
    for row in episodes:by_root[row['case_id']][row['condition']].append(row['SR'])
    means={r:{c:math.fsum(v)/len(v) for c,v in conditions.items()} for r,conditions in by_root.items()}
    result={}
    for domain in sorted({r['domain'] for r in metadata}):
        levels=sorted({r['level'] for r in metadata if r['domain']==domain})
        roots=[sorted(r['case_id'] for r in metadata if r['domain']==domain and r['level']==l) for l in levels]
        indices=[np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEEDS[domain],l,1]))).integers(0,len(a),size=(draws,len(a))) for l,a in zip(levels,roots)]
        macro={};ci={}
        for c in CONDITIONS:
            macro[c]=math.fsum(math.fsum(means[r][c] for r in a)/len(a) for a in roots)/len(roots)
            draws_=[]
            for b in range(draws):
                draws_.append(math.fsum(math.fsum(means[a[int(i)]][c] for i in index[b])/len(a) for a,index in zip(roots,indices))/len(roots))
            ci[c]=[quantile(draws_,.025),quantile(draws_,.975)]
        result[domain]=dict(macro=macro,ci=ci)
    return means,result


def actual_cache_audit():
    root=RESEARCH/'qualification/dev-receipts.v2/dev-L3-00-B0-0';attempt=root/'attempt-00'
    names=('intent.json','started.json','stdout.log','flight/run.json','flight/world.json','flight/private-case.json')
    raw={name:sha(attempt/name) for name in names}
    plan=json.loads((root/'plan.json').read_text());terminal=json.loads((attempt/'terminal.json').read_text())
    intent=json.loads((attempt/'intent.json').read_text());started=json.loads((attempt/'started.json').read_text())
    run=json.loads((attempt/'flight/run.json').read_text());world=json.loads((attempt/'flight/world.json').read_text())
    logpath=Path('/tmp/inspection-L3-cache-reuse.v2.log');log=json.loads(logpath.read_text())
    pinned=RESEARCH/'qualification/source/scenario-and-receipts-v2'
    binding=json.loads((pinned/'binding.json').read_text())
    assert all(sha(pinned/p)==h for p,h in binding['files'].items())
    assert sha(pinned/'eval/inspection_episode_receipts.py')==plan['receipt_adapter_sha256']
    assert sha(pinned/'eval/run_inspection_dev.py')==plan['supervisor_sha256']
    assert terminal['record_hashes']==raw and terminal['stdout_sha256']==raw['stdout.log']
    plan_hash=receipts.canonical_digest(plan)
    assert terminal['plan_sha256']==intent['plan_sha256']==started['plan_sha256']==log['plan_sha256']==plan_hash
    assert intent['attempt_ordinal']==0 and terminal['status']=='completed' and not terminal.get('not_started')
    assert sha(attempt/'terminal.json')==log['episode_receipt_sha256'] and log['cache_reused'] is True and log['actual_attempts']==1
    assert len(list(root.glob('attempt-*')))==1
    assert run['source_hashes']==plan['source_hashes']=={p:hashlib.sha256(s.encode()).hexdigest() for p,s in run['sources'].items()}
    assert run['dependency_hashes']==plan['dependency_hashes']
    assert run['runtime_thread_environment']==plan['thread_environment']
    spec=importlib.util.spec_from_file_location('pinned_local_receipts_v2',pinned/'eval/inspection_episode_receipts.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    value,evidence=old.validate_flight(attempt/'flight',plan,terminal['returncode'])
    assert value==terminal['Y']==log['Y'] and evidence==terminal['evidence']
    assert receipts.validate_flight(attempt/'flight',plan,terminal['returncode'])==(value,evidence)
    assert world['simulated_seconds']==600 and world['physics_steps_executed']==144000
    assert datetime.fromisoformat(intent['utc'])<=datetime.fromisoformat(started['utc'])
    return dict(role='read_only_existing_actual_dev_receipt_audit_not_new_flight',case_id=plan['case_id'],condition=plan['condition'],
        terminal_sha256=sha(attempt/'terminal.json'),plan_sha256=plan_hash,actual_receipt_adapter_sha256=plan['receipt_adapter_sha256'],
        actual_attempts=1,cache_log_sha256=sha(logpath),cache_log_reuse=True,Y=value,SR=world['score']['SR'],
        record_hashes=raw,source_snapshot_checked=True,current_guard_only_revalidated_existing_files=True,
        clock={k:world[k] for k in ('clock_mode','simulated_seconds','physics_steps_executed','physics_hz','control_hz','max_clock_lag_seconds')},
        readonly_actual_files_sha256={'plan.json':sha(root/'plan.json'),'attempt-00/terminal.json':sha(attempt/'terminal.json'),**{f'attempt-00/{k}':v for k,v in raw.items()}})


def main():
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=False)
    paths=['eval/inspection_episode_receipts.py','tests/test_inspection_episode_receipts.py','eval/inspection_secondary_statistics.py',
           'tests/test_inspection_secondary_statistics.py','eval/inspection_statistics.py','eval/inspection_recording.py']
    before={p:sha(REPO/p) for p in paths};save(out/'source-snapshot.json',{p:(REPO/p).read_text() for p in paths})
    fixture=runpy.run_path(str(REPO/'tests/test_inspection_episode_receipts.py'))
    guard={}
    for mode in ('missing-run-bad-copy','missing-run-bad-world'):
        base=out/mode;base.mkdir();plan,command,count=fixture['inputs'](base)
        try:receipts.execute_local_episode(base/'episode',plan,command(mode))
        except receipts.UnresolvedEpisode as e:
            terminal=json.loads((base/'episode/attempt-00/terminal.json').read_text())
            assert int(count.read_text())==1 and terminal['retry_allowed'] is False
            guard[mode]=dict(blocked=True,worker_calls=1,message=str(e),failure_reasons=terminal['evidence']['failure_reasons'])
        else:raise AssertionError('visible input fault admitted')
    actual=actual_cache_audit();save(out/'actual-cache-audit.json',actual)
    metadata,episodes=sr_data(MAIN_SHAPE);table=sr_root_table(metadata,episodes)
    assert len(episodes)==1296 and table.values.shape==(72,8)
    reference_means,reference=sr_reference(metadata,episodes,97)
    result=sr_cluster_intervals(table,draws=97,batch_size=7)
    maxerr=0.
    for i,root in enumerate(table.roots):
        for k,c in enumerate(CONDITIONS):assert table.values[i,k]==reference_means[root][c]
    for domain,d in result['domains'].items():
        for c,v in d['level_macro_root_mean_SR'].items():assert abs(v-reference[domain]['macro'][c])<1e-14
        for c,ci in d['ci95_root_cluster_percentile_nominal'].items():
            maxerr=max(maxerr,max(abs(a-b) for a,b in zip(ci,reference[domain]['ci'][c])))
            assert np.allclose(ci,reference[domain]['ci'][c],atol=1e-14,rtol=0)
    reversed_=sr_root_table(list(reversed(metadata)),list(reversed(episodes)))
    assert sr_cluster_intervals(reversed_,draws=97,batch_size=31)==result
    small_meta,small_rows=sr_data({'ID':{1:2,2:3,3:5},'OOD':{2:2,3:5}})
    small=sr_cluster_intervals(sr_root_table(small_meta,small_rows,require_main_shape=False),draws=97,batch_size=7)
    _,small_ref=sr_reference(small_meta,small_rows,97)
    for g,d in small['domains'].items():
        for c,v in d['level_macro_root_mean_SR'].items():assert abs(v-small_ref[g]['macro'][c])<1e-14
    fault_counts=0
    for fault in ('missing','duplicate','failed','bool','nonbinary','repeat-bool'):
        bad=[dict(x) for x in episodes]
        if fault=='missing':bad.pop()
        elif fault=='duplicate':bad.append(bad[0].copy())
        elif fault=='failed':bad[0]['status']='known_operational_failure'
        elif fault=='bool':bad[0]['SR']=True
        elif fault=='nonbinary':bad[0]['SR']=2
        else:bad[0]['repeat_ordinal']=False
        try:sr_root_table(metadata,bad)
        except ValueError:fault_counts+=1
        else:raise AssertionError('bad SR row accepted')
    degenerate={}
    for value in (0,1):
        rows=[{**r,'SR':value} for r in episodes]
        zero=sr_cluster_intervals(sr_root_table(metadata,rows),draws=31)
        assert all(ci==[float(value),float(value)] for d in zero['domains'].values() for ci in d['ci95_root_cluster_percentile_nominal'].values())
        degenerate[str(value)]='empirical point intervals; nominal/descriptive scope only'
    assert before=={p:sha(REPO/p) for p in paths},'source changed during audit'
    save(out/'sr-analysis-synthetic.json',result)
    report=dict(role='independent_guard_and_synthetic_SR_review_not_G4',utc=datetime.now(timezone.utc).isoformat(),source_sha256=before,sources_unchanged=True,
        guard=guard,actual_cache=actual,sr=dict(planned_episode_slots=1296,root_table_shape=[72,8],independent_97draw_quantile_max_error=maxerr,
        input_batch_order_invariant=True,unbalanced_macro_verified=True,invalid_row_cases_rejected=fault_counts,degenerate=degenerate,
        role=result['role'],interval_scope=result['interval_scope']),script_sha256=sha(Path(__file__)))
    save(out/'report.json',report);print(json.dumps({k:report[k] for k in ('guard','sr')},indent=2))
    print(json.dumps({k:actual[k] for k in ('terminal_sha256','plan_sha256','actual_attempts','cache_log_reuse','Y','SR','actual_receipt_adapter_sha256')},indent=2))


if __name__=='__main__':main()
