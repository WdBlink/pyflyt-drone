"""Synthetic native-proof fixtures only; literal seeds, no RNG/model/physics.

Nothing here touches the actual run's .paper or prepares actual formal data.
Public root names are copied from its accepted design, never private instances.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[5]))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


helper = load('namespace_synthetic_g4_fixture', HERE/'test_g4_proof_boundary.py')
admission = helper.module
sys.modules['eval.inspection_formal_admission'] = admission
phase = load('eval.inspection_formal_namespace', HERE.parent/'source/eval/inspection_formal_namespace.py')
module = load('eval.inspection_namespace_admission', HERE.parent/'source/eval/inspection_namespace_admission.py')


class Fixture:
    def __init__(self, project):
        self.project = project
        self.run, old_acceptance, self.verification = helper.synthetic_g4_fixture(project)
        self.owner = json.loads((self.run/'.paper/control.json').read_text())
        old = self.read(old_acceptance)
        subject = self.read(old['subject'])
        old_design = self.read(subject['design'])
        self.design = json.loads((RUN/'experiment/design.ral.v3.json').read_text())
        self.design['evaluator'] = old_design['evaluator']
        self.design['split_contract']['population'] = old_design['split_contract']['population']
        design_ref = self.put('design.json', self.design)
        plan = self.owner['receipts'][0]['packet']
        plan['experiment_design_sha256'] = design_ref['sha256']
        self.owner['frozen_plan'] = admission.canonical_sha(plan)
        subject['owner']['frozen_plan'] = self.owner['frozen_plan']
        subject['design'] = design_ref
        self.catalog = self.put('synthetic-dev-catalog.json', {
            'schema': 'inspection-development-identity-catalog-v1', 'seed_values': [7, 8, 9],
            'role': 'synthetic software fixture, not a real collision inventory'})
        subject['development_identity_catalog'] = self.catalog
        subject['qualification_evidence'].append(self.catalog)
        self.subject = self.put('subject.json', subject)
        self.source = subject['source_manifest']
        review = self.read(old['review']); review['subject_sha256'] = self.subject['sha256']
        self.review = self.put('review.json', review)
        self.acceptance = self.put('acceptance.json', {**old, 'subject': self.subject, 'review': self.review})
        self.owner['rounds'] = []
        self.bind('synthetic-finding', [self.acceptance, self.subject, self.review])

    def put(self, name, value):
        path = self.run/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {'path': name, 'sha256': admission.file_sha(path)}

    def read(self, ref):
        return json.loads((self.run/ref['path']).read_text())

    def save(self):
        (self.run/'.paper/control.json').write_text(json.dumps(self.owner))
        (self.run/'.paper/checkpoint.json').write_text(json.dumps({
            'state': self.owner, 'sha256': admission.canonical_sha(self.owner)}))

    def bind(self, finding_id, artifacts, revises=None):
        request = {'round': {'id': 'synthetic-proof-'+str(len(self.owner['rounds']))},
                   'evidence': {ref['path']: ref['sha256'] for ref in artifacts}}
        issued = admission.canonical_sha(request); request['request_sha256'] = issued
        finding = {'id': finding_id}
        if revises is not None: finding['revises'] = revises
        self.owner['rounds'].append({'request': request, 'validation': 'structure_and_binding_only',
            'decision': {'request_sha256': issued, 'progress': 'verified_progress',
                         'status': 'active', 'finding': finding}})
        self.save()

    def opening(self):
        main_intent, main_ref = self.intent('main')
        self.seal_intent(main_intent, main_ref)
        dataset = self.put('datasets/main.v1/manifest.json', {
            'role': 'synthetic main dataset artifact; no actual cases generated'})
        self.bind(module.MAIN_DATASET_FINDING, [dataset])
        evidence = self.put('synthetic-main-evidence.json', {'role': 'invented software fixtures, not experiment results'})
        subject = self.put('synthetic-lock-subject.json', {
            'schema': 'inspection-main-primary-analysis-subject-v1', 'phase': 'main',
            'root_denominator': 72, 'actual_episode_denominator': 1296,
            'root_mean_denominator': 576, 'confirmatory_family_size': 6,
            'design': self.read(self.subject)['design'], 'source_manifest': self.source,
            'main_dataset': dataset,
            **dict.fromkeys(('recording_results', 'primary_statistics', 'all_model_settlements'), evidence)})
        review = self.put('synthetic-lock-review.json', {
            'schema': 'inspection-main-primary-analysis-review-v1', 'decision': 'accept_main_primary_analysis_lock',
            'subject_sha256': subject['sha256'], 'unresolved_blocking_findings': [],
            'reviewer': 'synthetic-fixture-not-a-real-review'})
        lock = self.put('analysis/main-primary-lock.v1.json', {
            'schema': 'inspection-main-primary-analysis-lock-v1', 'status': 'locked',
            'subject': subject, 'review': review, 'native_finding_id': module.MAIN_LOCK_FINDING})
        self.bind(module.MAIN_LOCK_FINDING, [lock, subject, review])
        opening = self.put('datasets/holdout.v1/opening.json', {
            'schema': 'inspection-holdout-opening-v1', 'role': 'once_only_descriptive_direction_and_size',
            'g4_acceptance': self.acceptance, 'source_manifest': self.source, 'main_analysis_lock': lock,
            'holdout_root_ids': list(phase.accepted_phase_layout(self.design, 'holdout').roots),
            'native_finding_id': module.HOLDOUT_FINDING})
        self.bind(module.HOLDOUT_FINDING, [opening])
        return opening

    def intent(self, name='main', opening=None):
        layout = phase.accepted_phase_layout(self.design, name)
        value = {'schema': 'inspection-formal-namespace-intent-v1', 'phase': name,
            'g4_acceptance': self.acceptance, 'source_manifest': self.source,
            'population': self.read(self.subject)['population'], 'prior_catalog': self.catalog,
            'master_seed': 999 if name == 'main' else 1999,
            'root_generation_seeds': {root: (1000 if name == 'main' else 2000)+i for i, root in enumerate(layout.roots)},
            'schedule_seed': 50, 'sampler_master_seed': 51,
            'root_metadata': [{'case_id': root, 'domain': domain, 'level': level}
                              for root, domain, level in layout.metadata],
            'rng': module.RNG, 'holdout_opening': opening,
            'native_finding_id': module.NAMESPACE_FINDINGS[name]}
        return value, self.put(layout.intent_path, value)

    def seal_intent(self, value, ref):
        self.bind(value['native_finding_id'], [ref])

    def proof(self, intent, opening=None, verification=None):
        return module.read_namespace_proof(self.run, self.project, self.acceptance, intent,
            verification=verification or self.verification, holdout_opening_value=opening)


def test_main_namespace_literal_fixture_has_all72_roots_and_native_binding(tmp_path):
    f = Fixture(tmp_path); value, intent = f.intent(); f.seal_intent(value, intent)
    result = f.proof(intent)
    assert result.phase == 'main' and len(result.root_generation_seeds) == 72
    assert result.holdout_opening is None


@pytest.mark.parametrize('fault', ['no_native_binding', 'alias_id', 'extra_root', 'missing_root', 'duplicate_seed',
                                  'boolean_seed', 'master_collision', 'development_root_collision', 'development_master_collision',
                                  'bad_schedule', 'rng_version', 'metadata_order', 'metadata_bool', 'foreign_source', 'foreign_catalog', 'main_opens_holdout'])
def test_fully_rehashed_bad_namespace_intents_still_fail(tmp_path, fault):
    f = Fixture(tmp_path); value, _ = f.intent(); roots = list(value['root_generation_seeds'])
    if fault == 'alias_id': value['native_finding_id'] = 'new-main-namespace-alias'
    elif fault == 'extra_root': value['root_generation_seeds']['main-ID-L1-99'] = 9000
    elif fault == 'missing_root': value['root_generation_seeds'].pop(roots[0])
    elif fault == 'duplicate_seed': value['root_generation_seeds'][roots[1]] = value['root_generation_seeds'][roots[0]]
    elif fault == 'boolean_seed': value['root_generation_seeds'][roots[0]] = True
    elif fault == 'master_collision': value['master_seed'] = value['root_generation_seeds'][roots[0]]
    elif fault == 'development_root_collision': value['root_generation_seeds'][roots[0]] = 7
    elif fault == 'development_master_collision': value['master_seed'] = 8
    elif fault == 'bad_schedule': value['schedule_seed'] = True
    elif fault == 'rng_version': value['rng'] = {**value['rng'], 'version': 'different'}
    elif fault == 'metadata_order': value['root_metadata'] = list(reversed(value['root_metadata']))
    elif fault == 'metadata_bool': value['root_metadata'][0]['level'] = True
    elif fault == 'foreign_source': value['source_manifest'] = f.catalog
    elif fault == 'foreign_catalog': value['prior_catalog'] = f.put('new-catalog.json', {'schema': 'inspection-development-identity-catalog-v1', 'seed_values': [1]})
    elif fault == 'main_opens_holdout': value['holdout_opening'] = f.catalog
    intent = f.put('datasets/main.v1/namespace-intent.json', value)
    if fault != 'no_native_binding': f.seal_intent(value, intent)
    with pytest.raises(ValueError): f.proof(intent)


def test_explicit_namespace_revision_rejects_old_intent(tmp_path):
    f = Fixture(tmp_path); value, intent = f.intent(); f.seal_intent(value, intent)
    f.bind('synthetic-revocation', [f.catalog], revises=module.NAMESPACE_FINDINGS['main'])
    with pytest.raises(ValueError, match='superseded'): f.proof(intent)


def test_holdout_parent_lock_and_opening_are_read_before_any_holdout_seed_bytes(tmp_path, monkeypatch):
    f = Fixture(tmp_path); opening = f.opening(); value, intent = f.intent('holdout', opening); f.seal_intent(value, intent)
    reads = []; original = admission.Artifact.json
    def observe(ref, root): reads.append(ref.path); return original(ref, root)
    monkeypatch.setattr(admission.Artifact, 'json', observe)
    result = f.proof(intent, opening)
    assert result.phase == 'holdout' and len(result.root_generation_seeds) == 12
    assert reads.index('analysis/main-primary-lock.v1.json') < reads.index(intent['path'])


@pytest.mark.parametrize('fault', ['missing_opening', 'unapplied_opening', 'unapplied_lock', 'foreign_opening', 'seventh_test', 'new_opening_alias'])
def test_holdout_refuses_before_reading_seed_bytes_without_valid_one_time_parent(tmp_path, monkeypatch, fault):
    f = Fixture(tmp_path); opening = f.opening(); value, intent = f.intent('holdout', opening); f.seal_intent(value, intent)
    if fault == 'missing_opening': opening = None
    elif fault in ('unapplied_opening', 'unapplied_lock'):
        target = module.HOLDOUT_FINDING if fault == 'unapplied_opening' else module.MAIN_LOCK_FINDING
        f.owner['rounds'] = [row for row in f.owner['rounds'] if row['decision']['finding']['id'] != target]; f.save()
    else:
        v = f.read(opening)
        if fault == 'foreign_opening': v['source_manifest'] = f.catalog
        elif fault == 'seventh_test': v['role'] = 'another_confirmatory_test'
        else: v['native_finding_id'] = 'second-opening-alias'
        opening = f.put(opening['path'], v)
    reads = []; original = admission.Artifact.json
    def observe(ref, root): reads.append(ref.path); return original(ref, root)
    monkeypatch.setattr(admission.Artifact, 'json', observe)
    with pytest.raises(ValueError): f.proof(intent, opening)
    assert intent['path'] not in reads


def test_changed_latest_owner_stops_before_reading_namespace_seeds(tmp_path, monkeypatch):
    f = Fixture(tmp_path); value, intent = f.intent(); f.seal_intent(value, intent)
    original_reader = f.verification.owner_reader; calls = 0
    def stop_second_read(run):
        nonlocal calls
        owner = original_reader(run); calls += 1
        if calls >= 2: owner['task_stop'] = {'role': 'synthetic user-stop state transition'}
        return owner
    verification = admission.VerificationContext(f.verification.project_root, stop_second_read,
        f.verification.source_names, f.verification.dependency_names, f.verification.absent_source_names)
    reads = []; original = admission.Artifact.json
    def observe(ref, root): reads.append(ref.path); return original(ref, root)
    monkeypatch.setattr(admission.Artifact, 'json', observe)
    with pytest.raises(ValueError, match='current active owner'): f.proof(intent, verification=verification)
    assert intent['path'] not in reads


@pytest.mark.parametrize('fault', ['plan_drift', 'source_drift'])
def test_latest_native_valid_snapshot_rechecks_the_full_g4_predicate_before_seed_read(tmp_path, monkeypatch, fault):
    f = Fixture(tmp_path); value, intent = f.intent(); f.seal_intent(value, intent)
    original_reader = f.verification.owner_reader; calls = 0
    def drift_second_read(run):
        nonlocal calls
        calls += 1
        if calls == 2:
            if fault == 'plan_drift':
                f.owner['receipts'][0]['packet']['synthetic_plan_drift'] = True
                f.save()  # Valid pair, Brief and pin; stale frozen-plan digest.
            else:
                with (f.project/'envs/__init__.py').open('a') as stream:
                    stream.write('# synthetic qualified-source drift\n')
        return original_reader(run)
    verification = admission.VerificationContext(f.verification.project_root, drift_second_read,
        f.verification.source_names, f.verification.dependency_names, f.verification.absent_source_names)
    reads = []; original = admission.Artifact.json
    def observe(ref, root): reads.append(ref.path); return original(ref, root)
    monkeypatch.setattr(admission.Artifact, 'json', observe)
    with pytest.raises(ValueError): f.proof(intent, verification=verification)
    assert intent['path'] not in reads


@pytest.mark.parametrize('parent,child', [
    ('synthetic-finding', module.NAMESPACE_FINDINGS['main']),
    (module.NAMESPACE_FINDINGS['main'], module.MAIN_DATASET_FINDING),
    (module.MAIN_DATASET_FINDING, module.MAIN_LOCK_FINDING),
    (module.MAIN_LOCK_FINDING, module.HOLDOUT_FINDING),
    (module.HOLDOUT_FINDING, module.NAMESPACE_FINDINGS['holdout'])])
def test_native_application_order_cannot_retroactively_authorize_a_child(tmp_path, parent, child):
    f = Fixture(tmp_path); opening = f.opening(); value, intent = f.intent('holdout', opening); f.seal_intent(value, intent)
    def position(name):
        return next(i for i, row in enumerate(f.owner['rounds']) if row['decision']['finding']['id'] == name)
    a, b = position(parent), position(child)
    f.owner['rounds'][a], f.owner['rounds'][b] = f.owner['rounds'][b], f.owner['rounds'][a]
    f.save()
    with pytest.raises(ValueError, match='strictly before'): f.proof(intent, opening)


@pytest.mark.parametrize('fault', ['superseded', 'changed_intent', 'missing_evidence'])
def test_invalid_main_namespace_ancestor_refuses_before_holdout_seed_read(tmp_path, monkeypatch, fault):
    f = Fixture(tmp_path); opening = f.opening(); value, intent = f.intent('holdout', opening); f.seal_intent(value, intent)
    path = 'datasets/main.v1/namespace-intent.json'
    if fault == 'superseded':
        f.bind('synthetic-revoke-main-namespace', [f.catalog], revises=module.NAMESPACE_FINDINGS['main'])
    elif fault == 'changed_intent':
        with (f.run/path).open('a') as stream: stream.write(' ')
    else:
        row = next(r for r in f.owner['rounds'] if r['decision']['finding']['id']==module.NAMESPACE_FINDINGS['main'])
        row['request']['evidence'].pop(path)
        core = {k:v for k,v in row['request'].items() if k!='request_sha256'}
        row['request']['request_sha256'] = admission.canonical_sha(core)
        row['decision']['request_sha256'] = row['request']['request_sha256']; f.save()
    reads = []; original = admission.Artifact.json
    def observe(ref, root): reads.append(ref.path); return original(ref, root)
    monkeypatch.setattr(admission.Artifact, 'json', observe)
    with pytest.raises(ValueError): f.proof(intent, opening)
    assert intent['path'] not in reads


def test_main_namespace_revoked_in_final_snapshot_invalidates_holdout(tmp_path):
    f = Fixture(tmp_path); opening = f.opening(); value, intent = f.intent('holdout', opening); f.seal_intent(value, intent)
    original_reader = f.verification.owner_reader; calls = 0
    def revoke_final_read(run):
        nonlocal calls
        calls += 1
        if calls == 3:
            f.bind('synthetic-final-main-namespace-revocation', [f.catalog], revises=module.NAMESPACE_FINDINGS['main'])
        return original_reader(run)
    verification = admission.VerificationContext(f.verification.project_root, revoke_final_read,
        f.verification.source_names, f.verification.dependency_names, f.verification.absent_source_names)
    with pytest.raises(ValueError, match='superseded'): f.proof(intent, opening, verification)
    assert calls == 3
