"""Synthetic SDK compatibility check; never a flight or a method result."""

import argparse
from datetime import datetime,timezone
import json
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.inspection_recording import CONDITION_REPEATS,episode_schedule,record_matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('preserve existing evidence; choose a new output')
    paths=[Path(__file__),ROOT/'eval/inspection_recording.py',*sorted((args.sdk_root/'lib').glob('*.py'))]
    source_files={str(p.resolve()):p.read_text() for p in paths}
    sys.path.insert(0,str(args.sdk_root.resolve()))
    from lib.experiments import ExperimentRecorder
    from lib.production import json_bytes,write_once
    starts=set()
    def write_json(root,name,spec):
        write_once(root,name,json_bytes(spec))
        if spec.get('role')=='episode_attempt_start_not_completion':
            starts.add((spec['case_id'],spec['repeat_ordinal'],spec['condition']))
    context=dict(experiment_id='synthetic-adapter-preflight',claim_id='recording-compatibility-only',
        hypothesis='SYNTHETIC: SDK preserves all eight conditions and unequal episode repeats.',
        independent_variable='fixture condition identity',dependent_variable='fixture Y, not measured flight performance',
        control_variables=['no simulation','no model calls','no scientific outcome'],
        data_roles=[dict(id='synthetic-fixture',role='development',used_for_selection=False)],
        planned_at=datetime.now(timezone.utc).isoformat())
    metadata=[dict(case_id=f'synthetic-ID-L{level}-{i:02d}',domain='ID',level=level) for level in (1,2,3) for i in range(16)]
    metadata += [dict(case_id=f'synthetic-OOD-L{level}-{i:02d}',domain='OOD',level=level) for level in (2,3) for i in range(12)]
    cases=[r['case_id'] for r in metadata]
    order=episode_schedule(metadata,2026110501)
    calls=[]
    def fixture(case,repeat,condition):
        if (case,repeat,condition) not in starts:
            raise AssertionError('measurement ran before durable start marker')
        calls.append((case,repeat,condition))
        return repeat/4,{"role":"synthetic_adapter_fixture_not_scientific_evidence","actual_provider_seed":None}
    summary=record_matrix(args.out/'full-shape',cases,fixture,context=context,classic_best='B1',
        execution_order=order,recorder_type=ExperimentRecorder,write_json=write_json)
    if len(calls)!=1296 or len(set(calls))!=1296 or summary['outer_aggregate_rows']!=576:
        raise AssertionError('wrong full denominator')
    if calls!=[(r['case_id'],r['repeat_ordinal'],r['condition']) for r in order['slots']]:
        raise AssertionError('callback execution did not follow the frozen schedule')
    if order!=episode_schedule(list(reversed(metadata)),2026110501):
        raise AssertionError('input order changed the preregistered schedule')
    outer=json.loads((args.out/'full-shape/root-means/results.json').read_text())
    if any(row['value']!=(0. if CONDITION_REPEATS[row['condition']]==1 else .25) for row in outer['runs']):
        raise AssertionError('unequal repeats were not averaged correctly')
    holdout_cases=[f'synthetic-holdout-L{level}-{i:02d}' for level in (2,3) for i in range(6)]
    holdout_metadata=[dict(case_id=f'synthetic-holdout-L{level}-{i:02d}',domain='OOD',level=level) for level in (2,3) for i in range(6)]
    holdout=record_matrix(args.out/'holdout-shape',holdout_cases,fixture,context=context,classic_best='B1',
        execution_order=episode_schedule(holdout_metadata,2026110502),recorder_type=ExperimentRecorder,write_json=write_json)
    if holdout['actual_episode_denominator']!=216 or holdout['outer_aggregate_rows']!=96:
        raise AssertionError('wrong holdout shape denominator')
    def fault(*identity):raise RuntimeError('deliberate synthetic infrastructure failure')
    small_order=episode_schedule(metadata[:2],2026110503)
    try:
        record_matrix(args.out/'operational-failure',cases[:2],fault,context=context,classic_best='B1',
            execution_order=small_order,recorder_type=ExperimentRecorder,write_json=write_json)
        raise AssertionError('operational failure was silently accepted')
    except RuntimeError as error:
        if 'deliberate synthetic' not in str(error):raise
    failed=[]
    for group in ('classic','llm','root-means'):
        saved=json.loads((args.out/f'operational-failure/{group}/results.json').read_text())
        if saved['execution_status']!='failed' or len(saved['runs'])!=saved['reported_denominator']:
            raise AssertionError('failed/unstarted slots were dropped')
        failed.append(dict(group=group,denominator=len(saved['runs']),attempted=saved['attempted_count']))
    invalid=[]
    for label,bad_value in (('nonfinite',float('nan')),('out-of-range',1.1),('boolean',True)):
        try:
            record_matrix(args.out/label,cases[:2],lambda *identity:(bad_value,{}),context=context,classic_best='B1',
                execution_order=small_order,recorder_type=ExperimentRecorder,write_json=write_json)
            raise AssertionError('invalid Y was silently accepted')
        except ValueError:
            saved=json.loads((args.out/f'{label}/root-means/results.json').read_text())
            if saved['execution_status']!='failed' or len(saved['runs'])!=16:
                raise AssertionError('invalid score lost the planned outer denominator')
            invalid.append(label)
    snapshots={p:hashlib.sha256(s.encode()).hexdigest() for p,s in source_files.items()}
    if not all(Path(p).read_text()==s for p,s in source_files.items()):
        raise AssertionError('SDK or adapter source changed during preflight')
    write_once(args.out,'source-snapshot.json',json_bytes(source_files))
    report={"role":"synthetic_SDK_preflight_not_flight_method_or_inference_evidence","passed":True,
        "positive":summary,"holdout_shape":holdout,"operational_failure_fixture":failed,
        "invalid_score_fixtures_rejected":invalid,"sources_unchanged":True,"source_sha256":snapshots,
        "source_snapshot":"source-snapshot.json",
        "source_snapshot_sha256":hashlib.sha256((args.out/'source-snapshot.json').read_bytes()).hexdigest(),
        "classic_best_is_placeholder_for_fixture_only":True,
        "limitations":["no real episode traces or costs","no real generator/split validation",
            "root-paired stratified inference and multiplicity adapter still required","no formal authorization"]}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
