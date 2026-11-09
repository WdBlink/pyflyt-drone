"""Shared-law component checks; no actual main/holdout draw or file writer."""
import importlib.util
import json
from pathlib import Path
import sys
from dataclasses import replace

import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parent/'paper/research/20261104-inspection'
STAGE = HERE.parent/'eval'
old_path = RUN/'qualification/formal-installation.v1/before/eval/inspection_population.py'
old_spec = importlib.util.spec_from_file_location('population_pre_installation', old_path)
previous = importlib.util.module_from_spec(old_spec)
sys.modules[old_spec.name] = previous
old_spec.loader.exec_module(previous)
spec = importlib.util.spec_from_file_location('population_shared_law_candidate', STAGE/'inspection_population.py')
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)


@pytest.mark.parametrize('index', range(5))
def test_existing_five_development_draws_and_all_rejections_are_exactly_preserved(index):
    folder = RUN/'qualification/dev-population.v1'
    record = json.loads((folder/'manifest.json').read_text())['records'][index]
    args = (record['case_id'], record['domain'], record['level'], record['root_generation_seed'])
    old_case, old_audit = previous.generate_development_population_case(*args)
    new_case, new_audit = candidate.generate_development_population_case(*args)
    assert new_case.as_dict() == old_case.as_dict()
    assert json.loads(json.dumps(new_case.as_dict())) == json.loads((folder/record['file']).read_text())
    assert new_audit == old_audit == json.loads((folder/record['generation_audit']).read_text())


@pytest.mark.parametrize('phase', ['main', 'holdout'])
def test_content_validation_preserves_geometry_but_does_not_open_development_reader(phase):
    value = json.loads((RUN/'qualification/dev-population.v1/devext-ID-L2-00.json').read_text())
    original = candidate.PopulationCase.from_dict(value)
    name = phase+'-ID-L2-00'
    # A local serialization fixture is not a generated/formally admitted root.
    content = replace(original, split=phase, case_id=name, lineage_group=name)
    decoded = candidate.decode_formal_population_case(content.as_dict(), phase, name)
    assert decoded.as_dict() == content.as_dict()
    assert candidate._validate_population_contents(content)['area_m2'] == candidate.validate_population_case(original)['area_m2']
    with pytest.raises(ValueError, match='only generates new development'):
        candidate.load_development_case(content.as_dict())
    bad = replace(content, wind_xy=(3.0, 0.0))
    with pytest.raises(ValueError, match='wind exceeds'):
        candidate.validate_formal_population_case(bad, phase, name)


@pytest.mark.parametrize('fault', ['phase', 'id', 'lineage', 'schema', 'boolean_seed', 'domain', 'level'])
def test_formal_content_identity_mismatch_is_rejected(fault):
    value = json.loads((RUN/'qualification/dev-population.v1/devext-ID-L2-00.json').read_text())
    value.update(split='main', case_id='main-ID-L2-00', lineage_group='main-ID-L2-00')
    changes = {'phase': {'split': 'dev'}, 'id': {'case_id': 'main-ID-L2-01'},
               'lineage': {'lineage_group': 'main-ID-L2-01'}, 'schema': {'schema': 'unknown'},
               'boolean_seed': {'seed': True}, 'domain': {'domain': 'OOD'}, 'level': {'level': 'L1'}}
    value.update(changes[fault])
    with pytest.raises(ValueError):
        candidate.decode_formal_population_case(value, 'main', 'main-ID-L2-00')


def test_development_identity_guard_runs_before_any_rng(monkeypatch):
    def no_rng(*args, **kwargs):
        raise AssertionError('identity guard must precede any random construction')
    monkeypatch.setattr(candidate.np.random, 'SeedSequence', no_rng)
    with pytest.raises(ValueError, match='development-only'):
        candidate.generate_development_population_case('main-ID-L2-00', 'ID', 'L2', 7)
