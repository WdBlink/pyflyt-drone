"""Worker-content tests use /tmp serialization fixtures, no formal draws.

One selected fixture is copied from an existing dev input solely to exercise
content decoding under a synthetic native owner. It is not prospective data.
"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parent/'paper/research/20261104-inspection'
STAGE = HERE.parent/'eval'
spec = importlib.util.spec_from_file_location('formal_case_dataset_helpers', HERE/'test_dataset_admission.py')
helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
load = helper.helper.load
sys.modules['eval.inspection_dataset_admission'] = helper.module
catalog = load('eval.inspection_development_catalog', STAGE/'inspection_development_catalog.py')
shared = load('formal_case_shared_law', STAGE/'inspection_population.py')
previous = sys.modules.get('eval.inspection_population')
sys.modules['eval.inspection_population'] = shared
try:
    module = load('formal_case_candidate', STAGE/'inspection_formal_case.py')
finally:
    if previous is not None: sys.modules['eval.inspection_population'] = previous
    else: sys.modules.pop('eval.inspection_population', None)


def selected_fixture(tmp_path):
    f = helper.DatasetFixture(tmp_path); intent, manifest, review = f.dataset()
    value = f.read(manifest)
    row = next(row for row in value['records'] if row['case_id'] == 'main-ID-L2-00')
    body = json.loads((RUN/'qualification/dev-population.v1/devext-ID-L2-00.json').read_text())
    body.update(split='main', case_id=row['case_id'], lineage_group=row['case_id'], seed=row['sensor_seed'])
    case = shared.decode_formal_population_case(body, 'main', row['case_id'])
    row['case'] = f.put(row['case']['path'], case.as_dict())
    row['case_sha256'] = case.digest()
    row['configuration_sha256'] = catalog.physical_configuration_digest(case)
    manifest, review = f.bind_dataset(value)
    private = f.project/'flight/private-case.json'; private.parent.mkdir()
    private.write_bytes((f.run/row['case']['path']).read_bytes())
    return f, (intent, manifest, review), row, private, case


def read_selected(f, refs, row, private, **changes):
    return module.load_formal_case(f.run, f.project, f.acceptance, *refs,
        changes.get('case_id', row['case_id']), changes.get('copy_path', private),
        changes.get('expected', row['case']['sha256']), verification=f.verification)


def worker_payload(f, refs, row):
    return {'schema': 'inspection-formal-worker-input-v1', 'run': str(f.run),
        'runtime': str(f.project/'native-runtime'), 'sdk': str(f.project/'recording-sdk'),
        'g4': f.acceptance, 'namespace': refs[0], 'dataset': refs[1],
        'dataset_review': refs[2], 'case_id': row['case_id'], 'opening': None}


def test_worker_payload_constructs_its_verifier_in_process_then_checks_private_content(tmp_path, monkeypatch):
    f, refs, row, private, case = selected_fixture(tmp_path); payload = worker_payload(f, refs, row)
    seen = []
    def verifier(run, project, runtime, sdk):
        seen.append((run, project, runtime, sdk)); return f.verification
    monkeypatch.setattr(module, 'context_from_environment', verifier)
    result = module.load_formal_worker_input(f.project, payload, private, row['case']['sha256'])
    assert seen == [(payload['run'], f.project, payload['runtime'], payload['sdk'])]
    assert result.case.as_dict() == case.as_dict()
    binding = result.binding()
    assert binding['phase'] == 'main' and binding['case_id'] == row['case_id']
    assert binding['dataset_manifest'] == refs[1] and 'entities' not in json.dumps(binding)


@pytest.mark.parametrize('fault', ['missing_field', 'extra_field', 'schema', 'runtime_type',
                                  'case_id_type', 'bad_g4_path', 'bad_opening'])
def test_malformed_worker_payload_is_rejected_before_context_factory(tmp_path, monkeypatch, fault):
    f, refs, row, private, case = selected_fixture(tmp_path); payload = worker_payload(f, refs, row)
    if fault == 'missing_field': payload.pop('dataset')
    elif fault == 'extra_field': payload['unknown'] = True
    elif fault == 'schema': payload['schema'] = 'unknown'
    elif fault == 'runtime_type': payload['runtime'] = True
    elif fault == 'case_id_type': payload['case_id'] = True
    elif fault == 'bad_g4_path': payload['g4'] = {'path': '../foreign.json', 'sha256': '0'*64}
    else: payload['opening'] = {'path': '../foreign.json', 'sha256': '0'*64}
    def no_context(*args, **kwargs): raise AssertionError('shape check must precede factory')
    monkeypatch.setattr(module, 'context_from_environment', no_context)
    with pytest.raises(ValueError):
        module.load_formal_worker_input(f.project, payload, private, row['case']['sha256'])


def test_wrong_content_numpy_environment_fails_before_dataset_access(tmp_path, monkeypatch):
    f, refs, row, private, case = selected_fixture(tmp_path)
    monkeypatch.setattr(module.np, '__version__', 'synthetic-other-version')
    def no_dataset(*args, **kwargs): raise AssertionError('version check must precede dataset')
    monkeypatch.setattr(module, 'read_dataset_proof', no_dataset)
    with pytest.raises(ValueError, match='NumPy1.26.4'): read_selected(f, refs, row, private)


@pytest.mark.parametrize('fault', ['stop', 'source_drift', 'dataset_revocation'])
def test_admission_is_checked_again_after_content_validation_before_warmup(tmp_path, monkeypatch, fault):
    f, refs, row, private, case = selected_fixture(tmp_path)
    original = module.decode_formal_population_case
    def change_after_decode(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == 'stop':
            f.owner['task_stop'] = {'role':'synthetic stop during pure content validation'}; f.save()
        elif fault == 'source_drift':
            (f.project/'eval/inspection_formal_case.py').write_text('synthetic changed source')
        else:
            f.bind('synthetic-dataset-revocation-during-content', [f.catalog],
                   revises=helper.module.DATASET_FINDINGS['main'])
        return result
    monkeypatch.setattr(module, 'decode_formal_population_case', change_after_decode)
    with pytest.raises(ValueError): read_selected(f, refs, row, private)


def test_configuration_digest_dependency_drift_cannot_accept_a_bad_configuration(tmp_path, monkeypatch):
    f, refs, row, private, case = selected_fixture(tmp_path)
    intent, manifest, review = refs; value = f.read(manifest)
    target = next(item for item in value['records'] if item['case_id'] == row['case_id'])
    target['configuration_sha256'] = '0'*64
    manifest, review = f.bind_dataset(value)
    with pytest.raises(ValueError, match='configuration/noise identity'):
        read_selected(f, (intent,manifest,review), target, private)
    helper_path = f.project/'eval/prepare_inspection_development_population.py'
    helper_path.write_text('def configuration_digest(case):\n    return "'+('0'*64)+'"\n')
    # Model a fresh worker importing the changed helper, without executing
    # arbitrary fixture Python. Its identity function would return this value.
    monkeypatch.setattr(catalog, 'configuration_digest', lambda case: '0'*64)
    with pytest.raises(ValueError):
        read_selected(f, (intent,manifest,review), target, private)


def test_selected_content_is_decoded_after_real_synthetic_dataset_predicate_without_any_draw(tmp_path, monkeypatch):
    f, refs, row, private, case = selected_fixture(tmp_path)
    def no_random(*args, **kwargs): raise AssertionError('content reader must not draw')
    monkeypatch.setattr(shared.np.random, 'SeedSequence', no_random)
    monkeypatch.setattr(shared.np.random, 'default_rng', no_random)
    result = read_selected(f, refs, row, private)
    assert result.case.as_dict() == case.as_dict()
    assert result.private_copy_sha256 == row['case']['sha256']
    assert result.record.case_id == row['case_id']


@pytest.mark.parametrize('fault', ['wrong_root', 'wrong_expected', 'changed_private', 'symlink', 'hardlink', 'outside_project'])
def test_bad_private_copy_or_identity_is_rejected_before_decoding(tmp_path, monkeypatch, fault):
    f, refs, row, private, case = selected_fixture(tmp_path)
    kwargs = {}
    if fault == 'wrong_root': kwargs['case_id'] = 'main-ID-L2-99'
    elif fault == 'wrong_expected': kwargs['expected'] = '0'*64
    elif fault == 'changed_private': private.write_text('{}')
    elif fault == 'symlink':
        other = private.with_name('alias.json'); other.symlink_to(private); kwargs['copy_path'] = other
    elif fault == 'hardlink': private.with_name('hard-alias.json').hardlink_to(private)
    else:
        other = f.project.parent/(f.project.name+'-outside.json')
        other.write_bytes(private.read_bytes()); kwargs['copy_path'] = other
    def no_decode(*args, **kwargs): raise AssertionError('identity/copy rejection must precede decoding')
    monkeypatch.setattr(module, 'decode_formal_population_case', no_decode)
    with pytest.raises(ValueError): read_selected(f, refs, row, private, **kwargs)


@pytest.mark.parametrize('fault', ['phase', 'root', 'sensor', 'semantic', 'configuration', 'geometry'])
def test_native_bound_descriptor_cannot_override_actual_private_content(tmp_path, fault):
    f, refs, row, private, case = selected_fixture(tmp_path)
    intent, manifest, review = refs; value = f.read(manifest)
    target = next(item for item in value['records'] if item['case_id'] == row['case_id'])
    changed = case
    if fault == 'phase': changed = replace(case, split='dev')
    elif fault == 'root': changed = replace(case, case_id='main-ID-L2-99', lineage_group='main-ID-L2-99')
    elif fault == 'sensor': changed = replace(case, seed=case.seed+1)
    elif fault == 'geometry': changed = replace(case, wind_xy=(9.0,0.0))
    target['case'] = f.put(target['case']['path'], changed.as_dict())
    target['case_sha256'] = '0'*64 if fault == 'semantic' else changed.digest()
    target['configuration_sha256'] = ('0'*64 if fault == 'configuration'
                                      else catalog.physical_configuration_digest(changed))
    manifest, review = f.bind_dataset(value)
    private.write_bytes((f.run/target['case']['path']).read_bytes())
    with pytest.raises(ValueError): read_selected(f, (intent, manifest, review), target, private)


def test_private_json_duplicate_keys_are_rejected_even_when_bytes_are_native_bound(tmp_path):
    f, refs, row, private, case = selected_fixture(tmp_path)
    intent, manifest, review = refs; value = f.read(manifest)
    target = next(item for item in value['records'] if item['case_id'] == row['case_id'])
    raw = ('{"seed":'+str(case.seed)+','+json.dumps(case.as_dict())[1:]).encode()
    path = f.run/target['case']['path']; path.write_bytes(raw)
    target['case']['sha256'] = helper.helper.admission.file_sha(path)
    manifest, review = f.bind_dataset(value); private.write_bytes(raw)
    with pytest.raises(ValueError, match='duplicate'):
        read_selected(f, (intent, manifest, review), target, private)


def test_hash_and_decode_use_one_bounded_buffer_even_if_copy_changes_after_read(tmp_path, monkeypatch):
    f, refs, row, private, case = selected_fixture(tmp_path)
    original = module.bounded_bytes; reads = []
    def mutate_after_read(path, maximum):
        raw = original(path, maximum); reads.append(path)
        Path(path).write_bytes(b'{}')
        return raw
    monkeypatch.setattr(module, 'bounded_bytes', mutate_after_read)
    result = read_selected(f, refs, row, private)
    assert reads == [private] and result.case.as_dict() == case.as_dict()
    assert private.read_bytes() == b'{}'
    # Only the returned input buffer is certified. The later run/end-copy
    # unchanged check must reject the changed file; this is not that check.
