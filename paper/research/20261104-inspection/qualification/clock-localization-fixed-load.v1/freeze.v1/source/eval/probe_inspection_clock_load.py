"""One unscored B0 clock diagnostic with a fixed, non-controlling MLX workload.

This never retries an adaptive B4 slot. Recorded public prompts/seeds only
produce hardware load; their responses are logged and never parsed as tools.
The sole paper owner must reserve/settle the actual model operation separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.inspection.reference import ReferencePolicy
from eval.inspection_diagnostic_worker import worker
from eval.inspection_model_client import LocalModelClient
from eval.inspection_model_receipts import validate_model_operation
from eval.inspection_model_service import operation_directory
from eval.inspection_process_cleanup import stop_worker_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clock_stamp():
    return {'monotonic': time.monotonic(), 'utc': datetime.now(timezone.utc).isoformat()}


def read_tape(path):
    value = json.loads(Path(path).read_text())
    rows = value['requests']
    if (value.get('role') != 'fixed_public_prompt_workload_not_policy_or_retry'
            or type(rows) is not list or not 1 <= len(rows) <= 32):
        raise ValueError('bounded source-bound workload tape required')
    previous = -1.
    for row in rows:
        at = row['target_sim_time']
        if (type(at) not in (int, float) or not 0 <= at < 600 or at < previous
                or type(row['seed']) is not int or not 0 <= row['seed'] < 2**32
                or type(row['messages']) is not list
                or row['messages_sha256'] != hashlib.sha256(json.dumps(
                    row['messages'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()):
            raise ValueError('workload order/seed/message binding differs')
        previous = at
    for name, expected in value['source_files'].items():
        if sha(name) != expected: raise ValueError('original workload source changed')
    return rows


def model_workload(client, rows, progress, ready, done, records, errors):
    """Strictly serial fixed traffic. Output has no route to the command queue."""
    try:
        if not ready.wait(180): raise TimeoutError('physics did not publish its first state')
        for index, row in enumerate(rows):
            while True:
                with progress.get_lock(): current = float(progress[0])
                if done.is_set():
                    records.append({'kind': 'not_submitted_after_physics_end', 'ordinal': index+1})
                    return
                if current >= row['target_sim_time']-1e-8: break
                time.sleep(.01)
            record = {'ordinal': index+1, 'target_sim_time': row['target_sim_time'],
                'actual_sim_before_submit': current, 'submit': clock_stamp(),
                'seed': row['seed'], 'messages_sha256': row['messages_sha256']}
            client.submit(row['messages'], row['seed'])
            response = None
            while response is None:
                response = client.poll()
                if response is None: time.sleep(.01)
            with progress.get_lock(): current = float(progress[0])
            record.update(response=response, received=clock_stamp(), actual_sim_after_reply=current,
                          role='output retained as load evidence, never executable commands')
            records.append(record)
    except BaseException:
        errors.append(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reservation', required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--tape', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): parser.error('use a fresh bounded diagnostic output')
    threads = {key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    if any(os.environ.get(key) != val for key, val in threads.items()):
        parser.error('all four declared thread environment variables must be1')
    rows = read_tape(args.tape)
    tape = json.loads(args.tape.read_text())
    if sha(args.case) != tape['case_file_sha256']:
        raise ValueError('diagnostic case differs from the frozen original input')
    registry_path = args.run/'planning/local-model-registry.v1.json'
    registry = json.loads(registry_path.read_text()); profile = registry['models']['M1']
    budget = {**profile['generation_profile'], 'max_calls': len(rows)}
    receipt = args.run/profile['download_receipt']
    if sha(receipt) != profile['download_receipt_sha256']: raise ValueError('model receipt changed')
    owner = json.loads((args.run/'.paper/control.json').read_text())
    entry = owner['usage_ledger'][args.reservation]; purpose = json.loads(entry['purpose'])
    expected_purpose = {'kind': 'inspection_local_generation', 'phase': 'development',
        'model_registry_sha256': sha(registry_path), 'model_receipt_sha256': sha(receipt),
        'diagnostic_role': 'fixed_prompt_load_with_unscored_B0_clock_profile', 'tape_sha256': sha(args.tape)}
    if any(purpose.get(key) != value for key, value in expected_purpose.items()):
        raise ValueError('diagnostic purpose differs from the actual native reservation')
    expected = {'reservation': args.reservation, 'owner_frozen_plan': owner['frozen_plan'],
        'model_receipt_sha256': sha(receipt), 'service_sha256': sha(ROOT/'eval/inspection_model_service.py'),
        'reserved': entry['reserved'], 'generation_profile': budget}
    command = [str(ROOT/'.venv-inspection-llm/bin/python'), '-u', str(ROOT/'eval/inspection_model_service.py'),
        '--run', str(args.run.resolve()), '--reservation', args.reservation, '--model-receipt', str(receipt.resolve()),
        '--max-calls', str(len(rows)), '--max-total-tokens', str(budget['max_total_tokens']),
        '--max-output-tokens', str(budget['max_output_tokens']), '--max-input-tokens', str(budget['max_input_tokens']),
        '--max-request-seconds', str(budget['max_request_seconds'])]
    args.out.mkdir(parents=True)
    private = args.out/'private-case.json'; private.write_bytes(args.case.read_bytes())
    case_sha, tape_sha = sha(private), sha(args.tape)
    sources = [*sorted((ROOT/'envs/inspection').glob('*.py')), ROOT/'envs/__init__.py',
        *[ROOT/'eval'/name for name in ('inspection_cases.py', 'inspection_population.py', 'inspection_scenarios.py',
          'run_inspection_reference.py', 'inspection_clock_profile.py', 'inspection_diagnostic_worker.py',
          'inspection_model_client.py', 'inspection_model_service.py', 'inspection_model_receipts.py',
          'inspection_process_cleanup.py')], Path(__file__).resolve()]
    source_hashes = {str(p): sha(p) for p in sources}
    package = Path(importlib.util.find_spec('PyFlyt').origin).parent
    dependencies = [p for p in package.rglob('*') if p.suffix.lower() in ('.py', '.yaml', '.urdf', '.obj', '.stl', '.dae')]
    dependencies.append(Path(importlib.util.find_spec('pybullet').origin))
    deps = {str(p): sha(p) for p in dependencies}
    result = {'role': 'one instrumented engineering clock localization, not a policy comparison, retry or G4 acceptance',
        'started_at': clock_stamp(), 'case_sha256': case_sha, 'tape_sha256': tape_sha,
        'runtime_thread_environment': threads, 'source_hashes': source_hashes, 'dependency_hashes': deps,
        'python_version': sys.version,
        'package_versions': {name: importlib.metadata.version(name) for name in ('PyFlyt', 'pybullet', 'numpy', 'numba')},
        'expected_model_meter': expected, 'model_requests': [], 'errors': [], 'status': 'intent_only'}
    def save():
        (args.out/'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save()
    context = mp.get_context('spawn'); commands, updates = context.Queue(), context.Queue()
    progress = context.Array('d', [0.]*4); ready, done = threading.Event(), threading.Event()
    process = context.Process(target=worker, args=(0, False, True, 600., commands, updates,
        args.out, True, progress, private, case_sha))
    client, workload, awake = None, None, None
    policy = ReferencePolicy()
    try:
        client = LocalModelClient(command, args.out/'model-transport')
        if client.ready.get('reservation') != args.reservation: raise ValueError('wrong service ready identity')
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        result['awake_assertion'] = {'pid': awake.pid, 'scope_supervisor_pid': os.getpid(), 'flags': '-i -w'}
        process.start()
        result['physics_pid'], result['service_pid'] = process.pid, client.process.pid
        workload = threading.Thread(target=model_workload, args=(client, rows, progress, ready, done,
            result['model_requests'], result['errors']), daemon=True)
        workload.start()
        deadline = time.monotonic()+780
        while True:
            if time.monotonic() > deadline: raise TimeoutError('bounded diagnostic supervisor expired')
            try: update = updates.get(timeout=1)
            except queue.Empty:
                if not process.is_alive(): raise RuntimeError('worker exited without done')
                continue
            if update['kind'] == 'done':
                result['worker_status'] = update['status']
                break
            if update['kind'] == 'tool_result': policy.result(update['name'], update['args'], update['response'])
            elif update['kind'] == 'state':
                ready.set()
                for name, params in policy.decide(update['state']): commands.put({'name': name, 'args': params})
        done.set(); workload.join(timeout=125)
        if workload.is_alive(): raise RuntimeError('model workload remains unresolved')
        final = client.finish()
        if final is not None: result['final_model_response'] = final
        process.join(timeout=15)
        if process.is_alive() or process.exitcode != 0: raise RuntimeError('worker did not close cleanly')
        world = json.loads((args.out/'world.json').read_text())
        if (result['worker_status'] != 'completed' or world.get('status') != 'completed'
                or abs(world.get('simulated_seconds', -1)-600.) > 1e-8
                or world.get('physics_steps_executed') != 144000
                or world.get('clock_profile', {}).get('rows') != 72000
                or world['max_clock_lag_seconds'] != world['clock_profile']['max_lag']):
            raise ValueError('diagnostic did not retain the complete unchanged clock denominator')
        result['resource_recount'] = validate_model_operation(operation_directory(args.reservation), expected)
        result['status'] = 'completed_diagnostic_not_qualification' if not result['errors'] else 'diagnostic_workload_error'
    except BaseException:
        result['status'] = 'operational_failure_or_unknown'; result['errors'].append(traceback.format_exc())
    finally:
        done.set(); ready.set()
        if workload is not None and workload.is_alive(): workload.join(timeout=125)
        if client is not None and client.process.poll() is None:
            try: client.finish()
            except BaseException: result['errors'].append(traceback.format_exc())
        if process.pid is not None:
            try: result['physics_cleanup'] = stop_worker_tree(process)
            except BaseException: result['errors'].append(traceback.format_exc())
        if awake is not None and awake.poll() is None: awake.terminate(); awake.wait(timeout=10)
        result['sources_unchanged'] = all(sha(p) == value for p, value in source_hashes.items())
        result['dependencies_unchanged'] = all(sha(p) == value for p, value in deps.items())
        result['private_case_unchanged'] = sha(private) == case_sha
        result['tape_unchanged'] = sha(args.tape) == tape_sha
        result['workload_counts'] = {'planned': len(rows),
            'submitted': sum('submit' in row for row in result['model_requests']),
            'completed': sum(row.get('response', {}).get('status') == 'completed' for row in result['model_requests'])}
        if result['errors'] and result['status'] == 'completed_diagnostic_not_qualification':
            result['status'] = 'diagnostic_cleanup_or_workload_error'
        if not all(result[key] for key in ('sources_unchanged', 'dependencies_unchanged', 'private_case_unchanged', 'tape_unchanged')):
            result['status'] = 'input_or_source_changed_not_valid_diagnostic'
        result['finished_at'] = clock_stamp(); save()
    print(json.dumps({'status': result['status'], 'diagnostic': str(args.out/'diagnostic.json'),
                      'usage': result.get('resource_recount', {}).get('usage')}))
    return 0 if result['status'] == 'completed_diagnostic_not_qualification' else 1


if __name__ == '__main__': raise SystemExit(main())
