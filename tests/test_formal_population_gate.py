"""Check pre-draw ordering with a fake generator; create no formal inputs."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('formal_population_namespace_helpers', HERE/'test_namespace_admission.py')
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
module = helper.load('formal_population_gate_candidate', HERE.parent/'eval/inspection_formal_population.py')


def fixture(monkeypatch, phase='main'):
    seen = []
    name = phase+'-ID-L2-00'
    intent = SimpleNamespace(path='synthetic-intent', sha256='0'*64,
        json=lambda _: {'root_metadata': [{'case_id': name, 'domain': 'ID', 'level': 2}]})
    proof = SimpleNamespace(intent=intent, phase=phase,
        root_generation_seeds=((name, 71),), native_request_sha256='1'*64)
    def fake_generation(*args):
        seen.append(('generation', args))
        return 'synthetic-no-draw-result', {'role': 'fake generator, no random call'}
    monkeypatch.setitem(sys.modules, 'eval.inspection_population',
                        SimpleNamespace(_generate_population_case=fake_generation))
    return seen, name, proof


def test_namespace_failure_prevents_generator_import_and_any_draw(monkeypatch, tmp_path):
    seen, name, proof = fixture(monkeypatch)
    def reject(*args, **kwargs):
        seen.append('namespace_rejected')
        raise ValueError('native namespace not admitted')
    monkeypatch.setattr(module, 'read_namespace_proof', reject)
    with pytest.raises(ValueError, match='native namespace not admitted'):
        module.generate_formal_population_case(tmp_path, tmp_path, {}, {}, name, verification=object())
    assert seen == ['namespace_rejected']


@pytest.mark.parametrize('phase', ['main', 'holdout'])
def test_verified_namespace_supplies_exact_registered_seed_without_sampling(monkeypatch, tmp_path, phase):
    seen, name, proof = fixture(monkeypatch, phase)
    verification, acceptance, intent, opening = object(), {'synthetic': 'acceptance'}, {'synthetic': 'intent'}, {'synthetic': 'opening'}
    def admit(*args, **kwargs):
        assert args == (tmp_path, tmp_path, acceptance, intent)
        assert kwargs == {'verification': verification, 'holdout_opening_value': opening}
        seen.append('namespace_accepted')
        return proof
    monkeypatch.setattr(module, 'read_namespace_proof', admit)
    case, audit = module.generate_formal_population_case(tmp_path, tmp_path, acceptance, intent,
        name, verification=verification, holdout_opening_value=opening)
    assert case == 'synthetic-no-draw-result'
    assert seen == ['namespace_accepted', ('generation', (name, phase, 'ID', 'L2', 71))]
    assert audit['native_namespace_request_sha256'] == proof.native_request_sha256


@pytest.mark.parametrize('fault', ['unknown_root', 'missing_seed', 'duplicate_metadata'])
def test_root_membership_failure_prevents_generation(monkeypatch, tmp_path, fault):
    seen, name, proof = fixture(monkeypatch)
    if fault == 'unknown_root': name = 'main-ID-L2-01'
    elif fault == 'missing_seed': proof.root_generation_seeds = ()
    else:
        row = {'case_id': name, 'domain': 'ID', 'level': 2}
        proof.intent.json = lambda _: {'root_metadata': [row, row]}
    monkeypatch.setattr(module, 'read_namespace_proof', lambda *args, **kwargs: proof)
    with pytest.raises(ValueError, match='one accepted namespace root'):
        module.generate_formal_population_case(tmp_path, tmp_path, {}, {}, name, verification=object())
    assert seen == []


def test_wrong_numpy_environment_prevents_even_namespace_read(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError('must reject environment first')
    monkeypatch.setattr(module.np, '__version__', 'synthetic-wrong')
    monkeypatch.setattr(module, 'read_namespace_proof', unexpected)
    with pytest.raises(ValueError, match='NumPy1.26.4'):
        module.generate_formal_population_case(tmp_path, tmp_path, {}, {}, 'main-ID-L2-00', verification=object())
