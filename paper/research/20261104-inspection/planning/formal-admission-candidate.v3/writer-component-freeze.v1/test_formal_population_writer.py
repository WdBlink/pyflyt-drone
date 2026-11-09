"""One-shot effects only in /tmp synthetic fixtures, with real RNG forbidden."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('writer_dataset_helpers', HERE/'test_dataset_admission.py')
helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
load = helper.helper.load
sys.modules['eval.inspection_dataset_admission'] = helper.module
load('eval.inspection_development_catalog', HERE.parent/'source/eval/inspection_development_catalog.py')
load('eval.inspection_formal_population', HERE.parent/'source/eval/inspection_formal_population.py')
module = load('formal_population_writer_candidate', HERE.parent/'source/eval/prepare_inspection_formal_population.py')


def forbid_random(*args, **kwargs):
    raise AssertionError('no actual formal RNG is permitted in these software fixtures')


def literal_entropy(monkeypatch, f, first_root=10000, fail=False, phase='main'):
    seen = []
    path = f.run/('datasets/'+phase+'.v1/namespace-intent.json')
    def literal_bits(bits):
        seen.append(('literal_bits', bits))
        if bits == 128:
            assert path.exists() and path.stat().st_size == 0  # Exclusive claim before entropy.
            if fail: raise RuntimeError('injected entropy failure')
            return 131072
        return 313 if len(seen) == 2 else 414
    class LiteralChild:
        def __init__(self, value): self.value = value
        def generate_state(self, count):
            assert count == 4
            return module.np.array([self.value, 0, 0, 0], dtype='<u4')
    class LiteralSequence:
        def __init__(self, value):
            assert value == 131072; seen.append('literal_sequence')
        def spawn(self, count):
            assert count == (72 if phase == 'main' else 12)
            return [LiteralChild(first_root+i) for i in range(count)]
    monkeypatch.setattr(module.secrets, 'randbits', literal_bits)
    monkeypatch.setattr(module.np.random, 'SeedSequence', LiteralSequence)
    monkeypatch.setattr(module.np.random, 'default_rng', forbid_random)
    return seen


def test_namespace_claim_precedes_literal_seed_construction_and_cannot_be_replaced(tmp_path, monkeypatch):
    f = helper.DatasetFixture(tmp_path); seen = literal_entropy(monkeypatch, f)
    ref = module.write_namespace_intent(f.run, f.project, f.acceptance, 'main', verification=f.verification)
    value = f.read(ref)
    assert len(value['root_generation_seeds']) == 72
    assert value['master_seed'] == 131072 and len(seen) == 4
    f.seal_intent(value, ref)
    assert len(f.proof(ref).root_generation_seeds) == 72
    before = list(seen)
    with pytest.raises(ValueError, match='cannot be replaced or reseeded'):
        module.write_namespace_intent(f.run, f.project, f.acceptance, 'main', verification=f.verification)
    assert seen == before


@pytest.mark.parametrize('failure', ['entropy', 'collision'])
def test_failed_namespace_preserves_claim_and_never_redraws(tmp_path, monkeypatch, failure):
    f = helper.DatasetFixture(tmp_path)
    seen = literal_entropy(monkeypatch, f, first_root=7 if failure == 'collision' else 10000,
                           fail=failure == 'entropy')
    with pytest.raises((RuntimeError, ValueError)):
        module.write_namespace_intent(f.run, f.project, f.acceptance, 'main', verification=f.verification)
    folder = f.run/'datasets/main.v1'
    data = json.loads((folder/'namespace-failure.json').read_text())
    assert data['status'] == 'blocked_no_reseed'
    assert data['complete_intent_available'] == (failure == 'collision')
    before = list(seen)
    with pytest.raises(ValueError):
        module.write_namespace_intent(f.run, f.project, f.acceptance, 'main', verification=f.verification)
    assert seen == before


@pytest.mark.parametrize('failure', ['missing_g4', 'source_drift', 'missing_holdout_opening'])
def test_unadmitted_namespace_never_claims_a_file_or_touches_entropy(tmp_path, monkeypatch, failure):
    f = helper.DatasetFixture(tmp_path)
    monkeypatch.setattr(module.secrets, 'randbits', forbid_random)
    monkeypatch.setattr(module.np.random, 'SeedSequence', forbid_random)
    acceptance, phase = f.acceptance, 'main'
    if failure == 'missing_g4': acceptance = {'path': 'not-found.json', 'sha256': '0'*64}
    elif failure == 'source_drift': (f.project/'eval/inspection_formal_admission.py').write_text('changed synthetic source')
    else: phase = 'holdout'
    with pytest.raises((ValueError, FileNotFoundError)):
        module.write_namespace_intent(f.run, f.project, acceptance, phase, verification=f.verification)
    assert not (f.run/'datasets').exists()


def before_opening_fixture(tmp_path):
    f = helper.DatasetFixture(tmp_path); main = f.dataset()
    old_opening = f.typed_main_opening(main[1]); lock = f.read(old_opening)['main_analysis_lock']
    # Prepare an earlier synthetic /tmp state, before the one-time opening.
    path = f.run/old_opening['path']; path.unlink(); path.parent.rmdir()
    f.owner['rounds'] = [row for row in f.owner['rounds']
        if row['decision']['finding']['id'] != helper.helper.module.HOLDOUT_FINDING]
    f.save()
    return f, lock


def test_holdout_opening_requires_typed_locked_main_and_never_reopens(tmp_path, monkeypatch):
    f, lock = before_opening_fixture(tmp_path)
    monkeypatch.setattr(module.secrets, 'randbits', forbid_random)
    opening = module.write_holdout_opening(f.run, f.project, f.acceptance, lock, verification=f.verification)
    assert f.read(opening)['main_analysis_lock'] == lock
    assert not (f.run/'datasets/holdout.v1/namespace-intent.json').exists()
    with pytest.raises(ValueError, match='never reopen'):
        module.write_holdout_opening(f.run, f.project, f.acceptance, lock, verification=f.verification)


def test_changed_main_dataset_review_blocks_opening_before_any_holdout_file(tmp_path, monkeypatch):
    f, lock = before_opening_fixture(tmp_path)
    path = f.run/'datasets/main.v1/independent-review.json'
    path.write_text(path.read_text()+' ')
    monkeypatch.setattr(module.secrets, 'randbits', forbid_random)
    with pytest.raises(ValueError):
        module.write_holdout_opening(f.run, f.project, f.acceptance, lock, verification=f.verification)
    assert not (f.run/'datasets/holdout.v1').exists()


@pytest.mark.parametrize('collision', [False, True])
def test_holdout_namespace_excludes_locked_main_sensor_seeds_before_geometry(tmp_path, monkeypatch, collision):
    f, lock = before_opening_fixture(tmp_path)
    opening = module.write_holdout_opening(f.run, f.project, f.acceptance, lock, verification=f.verification)
    f.bind(helper.helper.module.HOLDOUT_FINDING, [opening])
    seen = literal_entropy(monkeypatch, f, first_root=3000 if collision else 10000, phase='holdout')
    if collision:
        with pytest.raises(ValueError, match='namespace seed collision'):
            module.write_namespace_intent(f.run, f.project, f.acceptance, 'holdout',
                verification=f.verification, holdout_opening_value=opening)
        assert (f.run/'datasets/holdout.v1/namespace-failure.json').exists()
    else:
        ref = module.write_namespace_intent(f.run, f.project, f.acceptance, 'holdout',
            verification=f.verification, holdout_opening_value=opening)
        body = f.read(ref); f.seal_intent(body, ref)
        assert len(f.proof(ref, opening).root_generation_seeds) == 12
    assert len(seen) == 4
    assert not (f.run/'datasets/holdout.v1/cases').exists()


def geometry_fixture(tmp_path, monkeypatch, failure=None):
    f = helper.DatasetFixture(tmp_path); intent, intent_ref = f.intent(); f.seal_intent(intent, intent_ref)
    seen = []
    monkeypatch.setattr(module.secrets, 'randbits', forbid_random)
    monkeypatch.setattr(module.np.random, 'SeedSequence', forbid_random)
    monkeypatch.setattr(module.np.random, 'default_rng', forbid_random)
    def fake_generate(run, project, acceptance, ref, name, **kwargs):
        assert (run, project, acceptance, ref) == (f.run, f.project, f.acceptance, intent_ref)
        assert (f.run/'datasets/main.v1/cases').is_dir()
        index = len(seen); seen.append(name)
        if failure == 'exhaustion':
            raise module.PopulationGenerationExhausted([{'candidate_index': 0, 'status': 'synthetic_rejection'}])
        if failure == 'final_source' and index == 71:
            (f.project/'eval/inspection_formal_admission.py').write_text('changed after final fake')
        case = SimpleNamespace(seed=3000+index, case_id=name,
            as_dict=lambda: {'role': 'fake serializer, no geometry draw', 'id': name},
            digest=lambda: helper.digest('semantic:'+name))
        return case, {'role': 'fake audit, no generation occurred', 'root_generation_seed': intent['root_generation_seeds'][name]}
    monkeypatch.setattr(module, 'generate_formal_population_case', fake_generate)
    monkeypatch.setattr(module, 'physical_configuration_digest',
        lambda case: helper.digest('same-config' if failure == 'collision' else 'config:'+case.case_id))
    return f, intent_ref, seen


def test_geometry_fake_effects_keep_complete_fixed_denominator_and_do_not_grant_dataset_proof(tmp_path, monkeypatch):
    f, intent, seen = geometry_fixture(tmp_path, monkeypatch)
    ref = module.write_geometry(f.run, f.project, f.acceptance, intent, verification=f.verification)
    value = f.read(ref)
    assert len(seen) == len(value['records']) == 72
    assert (value['root_denominator'], value['actual_episode_denominator'], value['root_mean_denominator']) == (72,1296,576)
    assert not (f.run/'datasets/main.v1/independent-review.json').exists()
    assert not any(r['decision']['finding']['id'] == helper.module.DATASET_FINDINGS['main'] for r in f.owner['rounds'])
    before = list(seen)
    with pytest.raises(ValueError, match='no automatic regeneration'):
        module.write_geometry(f.run, f.project, f.acceptance, intent, verification=f.verification)
    assert seen == before


@pytest.mark.parametrize('failure,raw_cases,completed', [('collision',2,1), ('exhaustion',0,0), ('final_source',72,72)])
def test_geometry_failure_keeps_all_partial_evidence_and_blocks_retry(tmp_path, monkeypatch, failure, raw_cases, completed):
    f, intent, seen = geometry_fixture(tmp_path, monkeypatch, failure)
    with pytest.raises((ValueError, module.PopulationGenerationExhausted)):
        module.write_geometry(f.run, f.project, f.acceptance, intent, verification=f.verification)
    folder = f.run/'datasets/main.v1'
    assert len(list((folder/'cases').glob('*.json'))) == raw_cases
    value = json.loads((folder/'generation-failure.json').read_text())
    assert value['completed_root_records'] == completed
    assert not (folder/'manifest.json').exists()
    if failure == 'exhaustion':
        audit = json.loads(next((folder/'generation').glob('*.json')).read_text())
        assert audit['status'] == 'finite_generation_exhausted' and len(audit['attempts']) == 1
    before = list(seen)
    with pytest.raises(ValueError):
        module.write_geometry(f.run, f.project, f.acceptance, intent, verification=f.verification)
    assert seen == before
