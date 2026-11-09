"""Isolated software tests; no real owner, model, flight or formal population."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch
import zipfile

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from lib.runtime import Controller, Refusal, atomic, digest, file_hash, read_json
from lib import round_documents as snapshots
from lib.budget import summary
from tests.test_runtime import brief


@pytest.fixture
def controller(tmp_path):
    ctl = Controller(tmp_path)
    initialized = ctl.init(brief())
    ctl.approve(initialized['brief_sha256'], 'software fixture')
    (tmp_path/'evidence.json').write_text('{"software_fixture":true}')
    return ctl


def sample(count=5):
    return {f'folder/document-{i}.md': {'sha256': hashlib.sha256(str(i).encode()).hexdigest(), 'lines': i}
            for i in range(count)}


def test_round_ref_reuse_and_inline_compatibility(controller):
    ctl = controller
    a = snapshots.write(ctl.root, sample())
    assert a == snapshots.write(ctl.root, sample())
    assert len(list((ctl.home/'round-documents').glob('*.json'))) == 1
    assert snapshots.resolve(ctl.root, {'documents_ref': a}) == sample()
    assert snapshots.resolve(ctl.root, {'documents': sample()}) == sample()
    result = ctl.begin_round(session_id='fixture-ref')
    assert 'documents' not in result['round']
    assert 'documents_ref' in result['round']
    request = ctl.prepare_round(['evidence.json'])
    decision = {'request_sha256': request['request_sha256'], 'progress': 'no_progress',
                'reason': 'Software only', 'status': 'active'}
    completed = ctl.apply_round(request, decision, 'ref-round')
    assert completed == ctl.apply_round(request, decision, 'ref-round')
    assert ctl.load()['rounds'][0]['request'] == request
    with pytest.raises(Refusal):
        ctl.apply_round(request, {**decision, 'reason': 'changed'}, 'ref-round')


@pytest.mark.parametrize('fault', ['both', 'neither', 'format', 'sha', 'count', 'canonical', 'path', 'missing', 'bytes', 'symlink', 'hardlink'])
def test_invalid_snapshot_fails_without_state_change(controller, fault):
    ctl = controller
    ref = snapshots.write(ctl.root, sample())
    baseline = {'documents_ref': copy.deepcopy(ref)}
    before = file_hash(ctl.path)
    path = ctl.root/ref['path']
    if fault == 'both': baseline['documents'] = sample()
    elif fault == 'neither': baseline = {}
    elif fault == 'format': baseline['documents_ref']['format'] = 'unknown'
    elif fault == 'sha': baseline['documents_ref']['sha256'] = '0'*64
    elif fault == 'count': baseline['documents_ref']['count'] += 1
    elif fault == 'canonical': baseline['documents_ref']['documents_sha256'] = '0'*64
    elif fault == 'path': baseline['documents_ref']['path'] = '../escape.json'
    elif fault == 'missing': path.unlink()
    elif fault == 'bytes': path.write_bytes(b'{}')
    elif fault == 'symlink':
        target = ctl.root/'other'; path.rename(target); path.symlink_to(target)
    elif fault == 'hardlink': os.link(path, ctl.root/'other')
    with pytest.raises((ValueError, OSError)):
        snapshots.resolve(ctl.root, baseline)
    assert before == file_hash(ctl.path)


@pytest.mark.parametrize('value', [True, -1, 1.5, '4'])
def test_metadata_line_type_is_strict(controller, value):
    documents = sample(1); documents[next(iter(documents))]['lines'] = value
    with pytest.raises(Refusal): snapshots.write(controller.root, documents)


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'])
def test_snapshot_strict_json(raw):
    with pytest.raises(Refusal): snapshots.strict(raw)


def test_durable_order_and_orphan_does_not_change_latest(controller):
    ctl = controller; before = file_hash(ctl.path)
    original = snapshots.sync_directory
    calls = []
    def fail_final_directory(path):
        calls.append(path)
        original(path)
        if Path(path).name == 'round-documents': raise OSError('fixture final fsync failure')
    with patch.object(snapshots, 'sync_directory', fail_final_directory):
        with pytest.raises(OSError): snapshots.write(ctl.root, sample())
    assert file_hash(ctl.path) == before
    assert ctl.load()['active_round'] is None
    assert len(list((ctl.home/'round-documents').glob('*.json'))) == 1
    snapshots.write(ctl.root, sample())  # Reuse exact orphan, never elect it latest.


def test_checkpoint_interruption_recovers_only_latest_valid_ref(controller):
    ctl = controller; before = read_json(ctl.path)
    ref = snapshots.write(ctl.root, sample())
    state = ctl.load(); state['active_round'] = {'id': 'fixture', 'documents_ref': ref}
    import lib.runtime as runtime
    original = runtime.atomic
    def fail_live(path, value, **kwargs):
        if path == ctl.path: raise OSError('fixture live projection failure')
        original(path, value, **kwargs)
    with patch.object(runtime, 'atomic', fail_live):
        with pytest.raises(OSError): ctl.save(state)
    assert read_json(ctl.path) == before
    with pytest.raises(Refusal): ctl.load()
    assert ctl.recover()['status'] == 'recovered'
    assert ctl.load()['active_round'] == state['active_round']
    # A missing latest reference cannot be reconstructed from current documents.
    (ctl.root/ref['path']).unlink()
    with pytest.raises(OSError): ctl.load()
    with pytest.raises(OSError): snapshots.closure(ctl.root, ctl.load())


def test_rewind_creates_referenced_baseline(controller):
    ctl = controller
    state = ctl.load(); state['receipts'] = [{'stage': 'intake', 'packet': {}, 'hashes': {}}]
    ctl.save(state)
    ctl.rewind('intake', 'Software fixture only', 'rewind-ref')
    assert 'documents_ref' in ctl.load()['active_round']


def test_many_rounds_do_not_embed_full_maps(controller):
    ctl = controller
    big = sample(10000)
    with patch('lib.rounds.documents', return_value=big):
        sizes = []
        for i in range(6):
            ctl.begin_round()
            request = ctl.prepare_round(['evidence.json'])
            assert 'documents' not in request['round']
            decision = {'request_sha256': request['request_sha256'], 'progress': 'no_progress',
                        'reason': 'Software representation only', 'status': 'active'}
            ctl.apply_round(request, decision, f'round-{i}')
            sizes.append(ctl.path.stat().st_size)
    assert max(b-a for a, b in zip(sizes, sizes[1:])) < 16000
    assert len(list((ctl.home/'round-documents').glob('*.json'))) == 1


def test_relocated_export_has_exact_snapshot_closure_and_zip(controller, tmp_path):
    ctl = controller; ref = snapshots.write(ctl.root, sample())
    state = ctl.load(); state['operations'] = {}; state['history'] = []; state['active_round'] = {'id': 'fixture', 'documents_ref': ref}
    dest = ctl.root/'delivery/fixture'; dest.mkdir(parents=True)
    atomic(dest/'control.json', state)
    (dest/'artifacts').mkdir(); (dest/'artifacts/evidence.json').write_text('{}')
    target = dest/ref['path']; target.parent.mkdir(parents=True); shutil.copyfile(ctl.root/ref['path'], target)
    payload = {'schema_version': 1, 'artifacts': {'evidence.json': file_hash(dest/'artifacts/evidence.json')},
               'software': {}, 'state_sha256': file_hash(dest/'control.json'), 'owner_objects': {ref['path']: ref['sha256']}}
    atomic(dest/'manifest.json', payload)
    assert ctl.verify_export('delivery/fixture/manifest.json')['status'] == 'verified'
    relocated = tmp_path/'relocated'; shutil.copytree(dest, relocated/'delivery/fixture')
    verifier = Controller(relocated)
    assert verifier.verify_export('delivery/fixture/manifest.json')['status'] == 'verified'
    archive = dest.parent/'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        for path in dest.rglob('*'):
            if path.is_file(): zipped.write(path, str(path.relative_to(dest)))
    assert ctl.verify_export('delivery/fixture/manifest.json')['archive']['files'] == 4
    target.unlink()
    with pytest.raises(OSError): ctl.verify_export('delivery/fixture/manifest.json')


def test_external_packet_bound_unchanged(controller):
    path = controller.root/'large.json'; path.write_bytes(b' '*(16*1024*1024+1))
    with pytest.raises(Refusal, match='16 MiB'): read_json(path)


def test_result_ref_preserves_exact_wire_payload_and_idempotent_response(controller):
    ctl = controller
    result = {'status': 'known', 'budget': {'unreported_fields': {'fixture-call': ['cost']},
              'known_usage': {'calls': 2, 'tokens': 17, 'cost': {}}}}
    state = ctl.load(); before = copy.deepcopy(state['history'])
    assert ctl.record(state, 'result-fixture', {'command': 'fixture'}, result) == result
    state = ctl.load(); stored = state['operations']['result-fixture']
    assert 'result' not in stored and 'result_ref' in stored
    assert state['history'][:len(before)] == before
    assert state['history'][-1]['result_ref'] == stored['result_ref']
    assert snapshots.resolve_result(ctl.root, stored) == result
    assert ctl.operation(state, 'result-fixture', {'command': 'fixture'}) == result
    assert snapshots.closure(ctl.root, state)[stored['result_ref']['path']] == stored['result_ref']['sha256']
    (ctl.root/stored['result_ref']['path']).write_text('{}')
    # History is lazy on ordinary load, and fail-closed when actually consumed,
    # recovered or exported. No historical result is silently reconstructed.
    state = ctl.load()
    with pytest.raises(Refusal): ctl.operation(state, 'result-fixture', {'command': 'fixture'})
    with pytest.raises(Refusal): snapshots.closure(ctl.root, state)


@pytest.mark.parametrize('fault', ['both', 'path', 'format', 'missing', 'symlink', 'hardlink'])
def test_operation_result_reference_counterexamples(controller, fault):
    ctl = controller
    ref = snapshots.write_result(ctl.root, {'status': 'synthetic'})
    record = {'result_ref': copy.deepcopy(ref)}; path = ctl.root/ref['path']
    if fault == 'both': record['result'] = {'status': 'synthetic'}
    elif fault == 'path': record['result_ref']['path'] = '../escape'
    elif fault == 'format': record['result_ref']['format'] = 'unknown'
    elif fault == 'missing': path.unlink()
    elif fault == 'symlink':
        other = ctl.root/'target'; path.rename(other); path.symlink_to(other)
    elif fault == 'hardlink': os.link(path, ctl.root/'other')
    with pytest.raises((ValueError, OSError)): snapshots.resolve_result(ctl.root, record)


def test_budget_result_refs_avoid_quadratic_live_state_growth(controller):
    ctl = controller; state = ctl.load(); sizes = []
    for i in range(12):
        result = {'status': 'known', 'budget': {'unreported_fields': {f'fixture-call-{j:04d}': ['cost'] for j in range(200*i)},
                  'known_usage': {'calls': i, 'tokens': 0, 'cost': {}}}}
        ctl.record(state, f'budget-result-{i}', {'command': 'fixture', 'i': i}, result)
        sizes.append(ctl.path.stat().st_size)
    assert max(b-a for a, b in zip(sizes, sizes[1:])) < 1600
    assert len(snapshots.closure(ctl.root, ctl.load())) == 12  # Approval is an event rather than a record() operation.


def test_owned_pair_compact_format_and_capacity_refuse_before_either_write(controller):
    ctl = controller
    raw = ctl.path.read_bytes(); checkpoint = (ctl.home/'checkpoint.json').read_bytes()
    assert raw.count(b'\n') == checkpoint.count(b'\n') == 1
    state = ctl.load(); state['software_fixture_padding'] = 'x'*9000
    with patch('lib.runtime.OWNER_STATE_MAX_BYTES', 4096):
        with pytest.raises(Refusal, match='before save'): ctl.save(state)
    assert ctl.path.read_bytes() == raw
    assert (ctl.home/'checkpoint.json').read_bytes() == checkpoint


def test_old_reader_command_only_replays_preserved_operation_never_bypasses_new_gate(controller):
    ctl = controller; packet = {'role': 'synthetic historical reader repair packet'}
    with pytest.raises(Refusal, match='historical reader repair'):
        ctl.migrate_state_reader(packet, 'unreviewed-new-migration')
    state = ctl.load()
    request = {'command': 'migrate-state-reader', 'packet': packet}
    result = {'status': 'synthetic preserved historical result'}
    state['operations']['old-reader-repair'] = {'request_sha256': digest(request), 'result': result}
    ctl.save(state); before = file_hash(ctl.path)
    assert ctl.migrate_state_reader(packet, 'old-reader-repair') == result
    assert file_hash(ctl.path) == before
    with pytest.raises(Refusal): ctl.migrate_state_reader({'changed': True}, 'old-reader-repair')


@pytest.fixture
def migration(tmp_path, request):
    prior = tmp_path/'runtime/autoresearch-state-reader-v1'; target = tmp_path/'runtime/autoresearch-document-snapshots-v2'
    shutil.copytree(PACKAGE.parent/'autoresearch-state-reader-v1', prior,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(PACKAGE, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    script = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from lib.runtime import Controller
c=Controller(sys.argv[2]);v=c.init(json.loads(sys.argv[3]));c.approve(v['brief_sha256'],'software fixture')
(c.root/'evidence.json').write_text('{"software_fixture":true}')
c.begin_round();q=c.prepare_round(['evidence.json'])
d={'request_sha256':q['request_sha256'],'progress':'no_progress','reason':'Software only','status':'active'}
c.apply_round(q,d,'legacy-round')
'''
    if getattr(request, 'param', None) == 'planned':
        script = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,str(Path(sys.argv[1])/'tests'))
from lib.runtime import Controller
from run_fixture import run,FixturePaused
try: run(Path(sys.argv[2]),stop_after='plan',budget={})
except FixturePaused: pass
c=Controller(sys.argv[2]);c.begin_round();q=c.prepare_round([])
d={'request_sha256':q['request_sha256'],'progress':'no_progress','reason':'Software only','status':'active'}
c.apply_round(q,d,'legacy-round')
'''
    initialized = subprocess.run([sys.executable, '-c', script, str(prior), str(tmp_path), json.dumps(brief())],
                                 capture_output=True, text=True)
    assert initialized.returncode == 0, initialized.stderr
    ctl = Controller(tmp_path, target); before = read_json(ctl.path)
    # Compatibility entry does not import or initialize an old Controller.
    wrapper = (PACKAGE.parents[1]/'recovery/owner-document-snapshots-v2/compat-research.py').read_text()
    (prior/'scripts/research.py').write_text(wrapper)
    transport = {'schema': 'controller-owner-transport-v1', 'target_package': 'runtime/autoresearch-document-snapshots-v2',
        'target_implementation': ctl.implementation(), 'compatibility_entry': 'runtime/autoresearch-state-reader-v1/scripts/research.py',
        'sources': {str(p.relative_to(tmp_path)): file_hash(p)
                    for p in [*ctl.implementation_paths(), prior/'scripts/research.py']}}
    def write(name, value):
        path = tmp_path/name; path.write_text(json.dumps(value))
        return {'path': name, 'sha256': file_hash(path)}
    transport_ref = write('transport.json', transport)
    review = {'decision': 'approve_round_documents_migration', 'reviewer': 'synthetic-reviewer',
        'from_implementation': before['implementation_sha256'], 'to_implementation': ctl.implementation(),
        'transport': transport_ref, 'preserves_contract': True, 'open_findings': []}
    budget = {k: v for k, v in summary(before).items() if k != 'elapsed_task_minutes'}
    gate = {'schema': 'controller-between-episodes-gate-v1', 'closed_round_count': len(before['rounds']),
        'active_round': None, 'pending_round': None, 'no_active_request_model_flight': True,
        'usage_summary': budget, 'state_sha256': digest(before)}
    packet = {'kind': 'controller-round-documents-v2', 'from_implementation': before['implementation_sha256'],
        'to_implementation': ctl.implementation(), 'prior_package': 'runtime/autoresearch-state-reader-v1',
        'transport': transport_ref, 'review': write('review.json', review), 'gate': write('gate.json', gate)}
    return ctl, before, packet, prior


def test_future_only_migration_exact_preservation_idempotence_and_compatibility(migration):
    ctl, before, packet, prior = migration
    result = ctl.migrate_round_documents(packet, 'migrate-ref')
    after = ctl.load()
    assert after['history'][:len(before['history'])] == before['history']
    assert {key: after['operations'][key] for key in before['operations']} == before['operations']
    excluded = {'implementation_sha256', 'history', 'operations'}
    assert {k: v for k, v in before.items() if k not in excluded} == {k: v for k, v in after.items() if k not in excluded}
    assert result == ctl.migrate_round_documents(packet, 'migrate-ref')
    assert after == ctl.load()
    legacy = before['rounds'][0]
    assert ctl.apply_round(legacy['request'], legacy['decision'], 'legacy-round') == before['operations']['legacy-round']['result']
    completed = subprocess.run([sys.executable, str(prior/'scripts/research.py'), '--run', str(ctl.root), 'status'],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    assert json.loads(completed.stdout)['stage'] == 0
    ctl.begin_round()
    assert 'documents_ref' in ctl.load()['active_round']
    # Every admission checks the compatibility entry; no old owner fallback.
    with (prior/'scripts/research.py').open('a') as stream: stream.write('# drift\n')
    with pytest.raises(Refusal, match='source changed'): ctl.load()


@pytest.mark.parametrize('fault', ['kind', 'target', 'from', 'review', 'gate', 'transport', 'active', 'unknown'])
def test_migration_counterexamples_never_change_state(migration, fault):
    ctl, before, packet, prior = migration
    packet = copy.deepcopy(packet)
    if fault == 'kind': packet['kind'] = 'controller-json-reader-v1'
    elif fault == 'target': packet['to_implementation'] = '0'*64
    elif fault == 'from': packet['from_implementation'] = '0'*64
    elif fault in ('review', 'gate', 'transport'): packet[fault]['sha256'] = '0'*64
    elif fault == 'active':
        state = copy.deepcopy(before); state['active_round'] = {'id': 'not-closed', 'documents': {}}
        ctl.save(state)
    elif fault == 'unknown':
        state = copy.deepcopy(before)
        state['usage_ledger']['fixture-unknown'] = {'status': 'unknown', 'reserved': {'calls': 1}, 'reported_usage': {}}
        ctl.save(state)
    sha = file_hash(ctl.path)
    with pytest.raises(Refusal): ctl.migrate_round_documents(packet, 'bad-migration')
    assert file_hash(ctl.path) == sha


@pytest.mark.parametrize('migration', ['planned'], indirect=True)
def test_compatibility_cli_reserve_settle_preserves_accounting_and_replay(migration):
    ctl, before, packet, prior = migration
    ctl.migrate_round_documents(packet, 'migrate-planned')
    ctl.begin_round()
    cli = prior/'scripts/research.py'
    def invoke(action, value, operation, reservation=None):
        path = ctl.root/(operation+'-packet.json')
        path.write_text(json.dumps(value))
        argv = [sys.executable, str(cli), '--run', str(ctl.root), action,
                '--packet', str(path), '--operation-id', operation]
        if reservation is not None: argv += ['--reservation', reservation]
        result = subprocess.run(argv, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
        return json.loads(result.stdout)
    request = {'usage': {'calls': 1, 'tokens': 40}, 'purpose': 'Synthetic compatibility test; no model requested'}
    reserved = invoke('budget-reserve', request, 'compat-resource')
    assert reserved['status'] == 'reserved'
    assert invoke('budget-reserve', request, 'compat-resource') == reserved
    receipt = ctl.root/'synthetic-usage.json'
    receipt.write_text(json.dumps({'kind': 'software_fixture', 'request_id': 'compat-resource',
                                   'calls': 1, 'tokens': 17}))
    settlement = {'receipt': receipt.name, 'status': 'known', 'usage': {'calls': 1, 'tokens': 17}}
    settled = invoke('budget-settle', settlement, 'compat-settle', 'compat-resource')
    assert invoke('budget-settle', settlement, 'compat-settle', 'compat-resource') == settled
    assert settled['budget']['known_usage'] == {'calls': 1, 'tokens': 17, 'cost': {}}
    assert settled['budget']['reserved'] == {'calls': 0, 'tokens': 0, 'cost': {}}
    assert settled['budget']['unreported_fields'] == {'compat-resource': ['cost']}


def malformed_closure_state(ctl, fault):
    state = ctl.load()
    documents = snapshots.write(ctl.root, sample())
    result = snapshots.write_result(ctl.root, {'status': 'synthetic-regression'})
    baseline = {'id': 'synthetic-baseline', 'documents_ref': documents}
    state['rounds'] = [{'request': {'round': copy.deepcopy(baseline)}} for _ in range(2)]
    record = {'request_sha256': 'a'*64, 'result_ref': result}
    state['operations'] = {'first': copy.deepcopy(record), 'second': copy.deepcopy(record)}
    state['history'] += [{'operation': name, **copy.deepcopy(record)} for name in ('first', 'second')]
    if fault == 'dual_baseline': state['rounds'][1]['request']['round']['documents'] = sample()
    elif fault == 'dual_operation': state['operations']['second']['result'] = {'status': 'synthetic-regression'}
    elif fault == 'dual_history': state['history'][-1]['result'] = {'status': 'synthetic-regression'}
    elif fault == 'missing_operation': state['operations']['second'].pop('result_ref')
    elif fault == 'missing_history': state['history'][-1].pop('result_ref')
    else: raise AssertionError('unknown synthetic fault')
    return state, {documents['path']: documents['sha256'], result['path']: result['sha256']}


@pytest.mark.parametrize('fault', ['dual_baseline', 'dual_operation', 'dual_history', 'missing_operation', 'missing_history'])
def test_closure_checks_every_record_before_deduplicating_object_bytes(controller, fault):
    state, _ = malformed_closure_state(controller, fault)
    with pytest.raises(Refusal, match='exactly one'):
        snapshots.closure(controller.root, state)


@pytest.mark.parametrize('fault', ['dual_baseline', 'dual_operation', 'dual_history', 'missing_operation', 'missing_history'])
def test_recovery_rejects_bad_later_record_without_overwriting_live_state(controller, fault):
    ctl = controller; state, _ = malformed_closure_state(ctl, fault)
    before = file_hash(ctl.path)
    atomic(ctl.home/'checkpoint.json', {'state': state, 'sha256': digest(state)})
    with pytest.raises(Refusal, match='exactly one'): ctl.recover()
    assert file_hash(ctl.path) == before


@pytest.mark.parametrize('fault', ['dual_baseline', 'dual_operation', 'dual_history', 'missing_operation', 'missing_history'])
def test_portable_export_and_zip_reject_bad_later_record(controller, fault):
    ctl = controller; state, objects = malformed_closure_state(ctl, fault)
    dest = ctl.root/'delivery/malformed'; dest.mkdir(parents=True)
    atomic(dest/'control.json', state, compact=True)
    (dest/'artifacts').mkdir(); (dest/'artifacts/evidence.json').write_text('{}')
    for name in objects:
        target = dest/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ctl.root/name, target)
    atomic(dest/'manifest.json', {'schema_version': 1,
        'artifacts': {'evidence.json': file_hash(dest/'artifacts/evidence.json')}, 'software': {},
        'state_sha256': file_hash(dest/'control.json'), 'owner_objects': objects})
    with zipfile.ZipFile(dest.parent/'malformed.zip', 'w') as archive:
        for path in dest.rglob('*'):
            if path.is_file(): archive.write(path, str(path.relative_to(dest)))
    with pytest.raises(Refusal, match='exactly one'):
        ctl.verify_export('delivery/malformed/manifest.json')
