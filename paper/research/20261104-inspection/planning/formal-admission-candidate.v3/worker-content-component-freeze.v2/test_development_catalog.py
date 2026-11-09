"""Read existing development inputs or isolated copies; no draws or new cases."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[2]
spec = importlib.util.spec_from_file_location('catalog_namespace_test_helpers', HERE/'test_namespace_admission.py')
helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
module = helper.load('eval.inspection_development_catalog', HERE.parent/'source/eval/inspection_development_catalog.py')


def ref(root, name):
    return {'path': name, 'sha256': helper.admission.file_sha(root/name)}


def test_actual_development_catalog_is_deterministic_and_never_draws_or_prepares_formal_data(monkeypatch):
    def no_random(*args, **kwargs): raise AssertionError('catalog must not draw any seed or geometry')
    monkeypatch.setattr(module.np.random, 'default_rng', no_random)
    monkeypatch.setattr(module.np.random, 'SeedSequence', no_random)
    manifests = [ref(RUN, name) for name in ('qualification/dev-cases.v2/manifest.json',
                                            'qualification/dev-population.v1/manifest.json')]
    extras = [ref(RUN, f'qualification/dev-same-level-event-pair.v2/{variant}/devext-OOD-L3-00.json')
              for variant in ('event', 'clean')]
    before = (RUN/'datasets').exists()
    first = module.build_development_catalog(RUN, manifests, extras)
    assert first == module.build_development_catalog(RUN, manifests, extras)
    assert first['schema'] == 'inspection-development-identity-catalog-v1'
    assert len(first['case_records']) == 31
    assert len(first['seed_values']) == len(set(first['seed_values']))
    assert all(row['case_id'].startswith(('dev-', 'devext-')) for row in first['case_records'])
    for value in manifests:
        manifest = json.loads((RUN/value['path']).read_text())
        if 'master_seed' in manifest: assert manifest['master_seed'] in first['seed_values']
        for row in manifest['records']:
            if 'root_generation_seed' in row: assert row['root_generation_seed'] in first['seed_values']
    assert (RUN/'datasets').exists() == before


def test_physical_digest_ignores_clean_parent_identity_provenance_but_keeps_intervention():
    path = RUN/'qualification/dev-same-level-event-pair.v2/clean/devext-OOD-L3-00.json'
    clean = module.load_development_case(json.loads(path.read_text()))
    changed = replace(clean, case_id='synthetic-metadata-alias', lineage_group='synthetic-lineage',
                      seed=123, parent_case={'role': 'synthetic changed provenance'}, parent_sha256='0'*64)
    assert module.physical_configuration_digest(clean) == module.physical_configuration_digest(changed)
    event = module.load_development_case(json.loads((RUN/'qualification/dev-same-level-event-pair.v2/event/devext-OOD-L3-00.json').read_text()))
    assert module.physical_configuration_digest(event) != module.physical_configuration_digest(clean)


def miniature(root):
    source_manifest = json.loads((RUN/'qualification/dev-cases.v2/manifest.json').read_text())
    row = source_manifest['records'][0]
    body = (RUN/'qualification/dev-cases.v2'/row['file']).read_bytes()
    (root/'case.json').write_bytes(body)
    record = {**row, 'file': 'case.json', 'file_sha256': helper.admission.file_sha(root/'case.json')}
    value = {'split': 'dev', 'master_seed': 999, 'records': [record]}
    return value


@pytest.mark.parametrize('fault', ['formal_manifest', 'escaped_case', 'changed_case_bytes', 'semantic_mismatch',
                                  'formal_case', 'boolean_master', 'boolean_generation_seed', 'duplicate_manifest'])
def test_bad_development_inventory_inputs_fail_closed(tmp_path, fault):
    value = miniature(tmp_path)
    if fault == 'formal_manifest': value['split'] = 'main'
    elif fault == 'escaped_case': value['records'][0]['file'] = '../case.json'
    elif fault == 'changed_case_bytes': (tmp_path/'case.json').write_text('{}')
    elif fault == 'semantic_mismatch': value['records'][0]['case_sha256'] = '0'*64
    elif fault == 'formal_case':
        path = tmp_path/'case.json'; body = json.loads(path.read_text()); body['split'] = 'main'
        path.write_text(json.dumps(body)); value['records'][0]['file_sha256'] = helper.admission.file_sha(path)
    elif fault == 'boolean_master': value['master_seed'] = True
    elif fault == 'boolean_generation_seed': value['records'][0]['root_generation_seed'] = True
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(value))
    manifest = ref(tmp_path, path.name)
    values = [manifest, manifest] if fault == 'duplicate_manifest' else [manifest]
    with pytest.raises(ValueError): module.build_development_catalog(tmp_path, values)


def test_duplicate_explicit_case_keeps_richer_generation_provenance_and_conflict_rejects(tmp_path):
    value = miniature(tmp_path); value['records'][0]['root_generation_seed'] = 10000
    (tmp_path/'manifest.json').write_text(json.dumps(value))
    manifest = ref(tmp_path, 'manifest.json'); case = ref(tmp_path, 'case.json')
    result = module.build_development_catalog(tmp_path, [manifest], [case])
    assert len(result['case_records']) == 1 and result['case_records'][0]['root_generation_seed'] == 10000
    value['records'][0]['root_generation_seed'] = 10001
    (tmp_path/'second.json').write_text(json.dumps(value))
    with pytest.raises(ValueError, match='contradictory'):
        module.build_development_catalog(tmp_path, [manifest, ref(tmp_path, 'second.json')])
    # An older inventory can lack this metadata; adding known provenance is
    # different from contradicting an already known generation seed.
    value['records'][0].pop('root_generation_seed')
    (tmp_path/'unknown.json').write_text(json.dumps(value))
    richer = module.build_development_catalog(tmp_path, [ref(tmp_path, 'unknown.json'), manifest])
    assert richer['case_records'][0]['root_generation_seed'] == 10000


def test_wrong_producer_numpy_version_refuses_before_input_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(module.np, '__version__', 'synthetic-wrong-version')
    with pytest.raises(ValueError, match='NumPy1.26.4'):
        module.build_development_catalog(tmp_path, [])


def test_old_invalid_and_retired_inputs_are_inventoried_without_readmission(monkeypatch):
    def no_random(*args,**kwargs):raise AssertionError('archive inventory cannot draw')
    monkeypatch.setattr(module.np.random,'default_rng',no_random)
    monkeypatch.setattr(module.np.random,'SeedSequence',no_random)
    manifests=[ref(RUN,name) for name in ('qualification/dev-cases.v1/manifest.json',
        'qualification/dev-cases.v2/manifest.json','qualification/dev-population.v1/manifest.json')]
    extras=[ref(RUN,'qualification/dev-clean-sibling.v1/devext-OOD-L3-00-clean.json'),
        *[ref(RUN,f'qualification/dev-same-level-event-pair.v2/{part}/devext-OOD-L3-00.json') for part in ('event','clean')]]
    result=module.build_development_catalog(RUN,manifests,extras)
    assert len(result['case_records'])==56  # artifact versions/aliases, not56 roots.
    old=next(r for r in result['case_records'] if r['case']['path']=='qualification/dev-cases.v1/dev-L2-03.json')
    assert old['case_sha256']=='ed93254799d5b4634b9ed50c974f21f67a62b387f189f35a1d1f38c3e4d38947'
    assert old['case_sha256'] in result['case_sha256_values']
    assert old['configuration_sha256'] in result['configuration_sha256_values']
    assert old['current_mission_validation']['status']=='rejected_by_current_mission_validator'
    assert 'separated between samples' in old['current_mission_validation']['reason']
    value=json.loads((RUN/old['case']['path']).read_text())
    with pytest.raises(ValueError,match='separated between samples'):module.load_development_case(value)
    assert all(r['identity_role']=='archived_input_not_mission_qualification' for r in result['case_records'])


@pytest.mark.parametrize('fault',['untyped_schema','boolean_coordinate','nonfinite_coordinate'])
def test_archived_identity_keeps_structural_type_and_finite_guards(fault):
    value=json.loads((RUN/'qualification/dev-cases.v1/dev-L2-03.json').read_text())
    if fault=='untyped_schema':value['schema']=[]
    elif fault=='boolean_coordinate':value['roi_polygon'][0][0]=True
    else:value['roi_polygon'][0][0]=float('nan')
    with pytest.raises(ValueError):module._archived_identity_case(value)
