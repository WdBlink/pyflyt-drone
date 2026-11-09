"""Read-only representation projection; writes only a requested diagnostic file.

Synthetic records are size templates, never serialized into a real owner store.
No model/physics, generator, reservation, experiment or new research ledger.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--reader-package', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.reader_package.resolve()))
from lib.runtime import Controller, digest, file_hash

start = time.monotonic()
controller = Controller(args.run, args.reader_package)
with controller.lock():
    state = controller.load()
    base_file_sha = file_hash(controller.path)
base_sha = digest(state)

def size(value):
    return len(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())+1

base_bytes = size(state)
template = max(state['usage_ledger'].values(), key=size)
reference = {'format': 'controller-operation-result-v1',
    'path': '.paper/operation-results/'+'a'*64+'.json', 'sha256': 'a'*64}
for index in range(13000):
    key = f'synthetic-capacity-operation-{index:05d}-'+'x'*50
    row = {'request_sha256': 'b'*64, 'result_ref': reference}
    state['operations'][key] = row
    state['history'].append({'operation': key, 'at': '2000-01-01T00:00:00Z', **row})
for index in range(5000):
    key = f'synthetic-capacity-reservation-{index:05d}-'+'x'*48
    state['usage_ledger'][key] = copy.deepcopy(template)
envelope = copy.deepcopy(state['rounds'][-1])
envelope['request']['round'].pop('documents', None)
envelope['request']['round']['documents_ref'] = {'format': 'controller-round-documents-v1',
    'path': '.paper/round-documents/'+'c'*64+'.json', 'sha256': 'c'*64,
    'documents_sha256': 'd'*64, 'count': 30000}
for _ in range(2000):
    state['rounds'].append(envelope)
projected = size(state)
checkpoint = size({'state': state, 'sha256': 'e'*64})
result = {'role': 'synthetic representation capacity projection, not experiments or resource usage',
    'base_state_sha256': base_sha, 'base_file_sha256': base_file_sha, 'base_compact_bytes': base_bytes,
    'future_operation_records': 13000, 'operation_allocation': {'reserve_settle': 10000, 'round_apply': 2000, 'other': 1000},
    'future_known_llm_attempts': 5000, 'future_round_envelopes': 2000,
    'recent_envelope_per_round_compact_bytes': size(envelope), 'usage_entry_template_bytes': size(template),
    'projected_control_bytes': projected, 'projected_checkpoint_bytes': checkpoint,
    'bound_bytes': 128*1024*1024, 'within_bound': max(projected, checkpoint) < 128*1024*1024,
    'elapsed_seconds': time.monotonic()-start,
    'limits': ['Actual recent envelope reused as a size template; larger future workloads can grow more.',
              'Object payload storage is outside live state; each object is still bounded to16MiB.',
              'No unlimited-growth guarantee; pre-save capacity must continue failing closed.',
              'No actual owner operation/reservation/round or formal seed was created; no unknown cost became zero.']}
with args.out.open('x') as stream:
    json.dump(result, stream, indent=2); stream.write('\n')
print(json.dumps(result))
