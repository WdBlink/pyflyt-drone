"""Literal /tmp dataset-proof fixtures; no actual formal RNG or physics."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from dataclasses import replace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dataset_namespace_fixture_helpers', HERE/'test_namespace_admission.py')
helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
module = helper.load('dataset_admission_candidate', HERE.parent/'source/eval/inspection_dataset_admission.py')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class DatasetFixture(helper.Fixture):
    def __init__(self, project):
        super().__init__(project)
        old_catalog = self.catalog
        self.catalog = self.put(old_catalog['path'], {'schema': 'inspection-development-identity-catalog-v1',
            'seed_values': [7, 8, 9], 'case_sha256_values': ['a'*64], 'configuration_sha256_values': ['b'*64]})
        subject = self.read(self.subject)
        subject['development_identity_catalog'] = self.catalog
        subject['qualification_evidence'] = [self.catalog if item == old_catalog else item
                                             for item in subject['qualification_evidence']]
        self.subject = self.put(self.subject['path'], subject)
        review = self.read(self.review); review['subject_sha256'] = self.subject['sha256']
        self.review = self.put(self.review['path'], review)
        acceptance = self.read(self.acceptance)
        self.acceptance = self.put(self.acceptance['path'], {**acceptance, 'subject': self.subject, 'review': self.review})
        self.owner['rounds'] = []
        self.bind(acceptance['native_finding_id'], [self.acceptance, self.subject, self.review])

    def dataset(self, phase='main', opening=None):
        intent, intent_ref = self.intent(phase, opening); self.seal_intent(intent, intent_ref)
        records = []
        folder = 'datasets/'+phase+'.v1'
        for index, row in enumerate(intent['root_metadata']):
            name = row['case_id']
            case = self.put(folder+'/cases/'+name+'.json', {'role': 'synthetic bytes, not qualified mission content', 'id': name})
            audit = self.put(folder+'/generation/'+name+'.json', {'role': 'synthetic audit, no geometry generated'})
            records.append({**row, 'case': case, 'generation_audit': audit,
                'case_sha256': digest('semantic:'+name), 'configuration_sha256': digest('config:'+name),
                'sensor_seed': (3000 if phase == 'main' else 5000)+index,
                'root_generation_seed': intent['root_generation_seeds'][name]})
        count = len(records)
        value = {'schema': 'inspection-formal-dataset-v1', 'phase': phase,
            'status': 'generation_complete_not_episode_results',
            'g4_acceptance': self.acceptance, 'namespace_intent': intent_ref,
            'source_manifest': self.source, 'population': intent['population'], 'prior_catalog': self.catalog,
            'root_denominator': count, 'actual_episode_denominator': count*18, 'root_mean_denominator': count*8,
            'records': records, 'native_finding_id': module.DATASET_FINDINGS[phase]}
        manifest, review = self.bind_dataset(value)
        return intent_ref, manifest, review

    def bind_dataset(self, value):
        phase = value['phase']; folder = 'datasets/'+phase+'.v1'
        manifest = self.put(folder+'/manifest.json', value)
        report = self.put('review/synthetic-'+phase+'-dataset-report.json', {'role': 'software fixture, not a scientific review'})
        review = self.put(folder+'/independent-review.json', {
            'schema': 'inspection-formal-dataset-review-v1',
            'decision': 'accept_generation_identity_and_collision_scope',
            'manifest_sha256': manifest['sha256'], 'reviewer': 'synthetic-test-only',
            'unresolved_blocking_findings': [], 'report': report})
        self.owner['rounds'] = [r for r in self.owner['rounds']
            if r['decision']['finding']['id'] != module.DATASET_FINDINGS[phase]]
        self.bind(module.DATASET_FINDINGS[phase], [manifest, review])
        return manifest, review

    def proof_dataset(self, intent, manifest, review, opening=None):
        return module.read_dataset_proof(self.run, self.project, self.acceptance, intent,
            manifest, review, verification=self.verification, holdout_opening_value=opening)

    def typed_main_opening(self, main):
        evidence = self.put('synthetic-main-results.json', {'role': 'literal fixture, no experiment'})
        subject = self.put('synthetic-primary-subject.json', {
            'schema': 'inspection-main-primary-analysis-subject-v1', 'phase': 'main',
            'root_denominator': 72, 'actual_episode_denominator': 1296, 'root_mean_denominator': 576,
            'confirmatory_family_size': 6, 'design': self.read(self.subject)['design'],
            'source_manifest': self.source, 'main_dataset': main,
            **dict.fromkeys(('recording_results', 'primary_statistics', 'all_model_settlements'), evidence)})
        review = self.put('synthetic-primary-review.json', {'schema': 'inspection-main-primary-analysis-review-v1',
            'decision': 'accept_main_primary_analysis_lock', 'subject_sha256': subject['sha256'],
            'reviewer': 'synthetic-only', 'unresolved_blocking_findings': []})
        lock = self.put('analysis/main-primary-lock.v1.json', {'schema': 'inspection-main-primary-analysis-lock-v1',
            'status': 'locked', 'subject': subject, 'review': review,
            'native_finding_id': helper.module.MAIN_LOCK_FINDING})
        self.bind(helper.module.MAIN_LOCK_FINDING, [lock, subject, review])
        opening = self.put('datasets/holdout.v1/opening.json', {'schema': 'inspection-holdout-opening-v1',
            'role': 'once_only_descriptive_direction_and_size', 'g4_acceptance': self.acceptance,
            'source_manifest': self.source, 'main_analysis_lock': lock,
            'holdout_root_ids': list(helper.phase.accepted_phase_layout(self.design, 'holdout').roots),
            'native_finding_id': helper.module.HOLDOUT_FINDING})
        self.bind(helper.module.HOLDOUT_FINDING, [opening])
        return opening


def test_complete72root_fixture_checks_private_hashes_without_deserializing_case_content(tmp_path, monkeypatch):
    f = DatasetFixture(tmp_path); refs = f.dataset()
    original = helper.admission.Artifact.json
    def no_private_json(ref, root):
        if '/cases/' in ref.path or '/generation/' in ref.path:
            raise AssertionError('private inputs must be hashed without deserialization here')
        return original(ref, root)
    monkeypatch.setattr(helper.admission.Artifact, 'json', no_private_json)
    proof = f.proof_dataset(*refs)
    assert len(proof.records) == 72 and proof.namespace.phase == 'main'
    # Deliberately non-mission JSON above demonstrates the distinction between
    # byte/provenance validation and the separate worker content predicate.


@pytest.mark.parametrize('fault', ['missing_root', 'metadata_bool', 'wrong_order', 'root_seed', 'sensor_bool',
    'duplicate_sensor', 'sensor_root_collision', 'dev_sensor_collision', 'dev_semantic', 'dev_config',
    'duplicate_config', 'wrong_case_path', 'wrong_source', 'wrong_denominator'])
def test_fully_rehashed_and_native_bound_bad_dataset_still_fails(tmp_path, fault):
    f = DatasetFixture(tmp_path); intent, manifest, review = f.dataset(); value = f.read(manifest)
    rows = value['records']
    if fault == 'missing_root': rows.pop()
    elif fault == 'metadata_bool': rows[0]['level'] = True
    elif fault == 'wrong_order': rows.reverse()
    elif fault == 'root_seed': rows[0]['root_generation_seed'] = 99999
    elif fault == 'sensor_bool': rows[0]['sensor_seed'] = True
    elif fault == 'duplicate_sensor': rows[1]['sensor_seed'] = rows[0]['sensor_seed']
    elif fault == 'sensor_root_collision': rows[0]['sensor_seed'] = rows[-1]['root_generation_seed']
    elif fault == 'dev_sensor_collision': rows[0]['sensor_seed'] = 7
    elif fault == 'dev_semantic': rows[0]['case_sha256'] = 'a'*64
    elif fault == 'dev_config': rows[0]['configuration_sha256'] = 'b'*64
    elif fault == 'duplicate_config': rows[1]['configuration_sha256'] = rows[0]['configuration_sha256']
    elif fault == 'wrong_case_path': rows[0]['case'] = f.catalog
    elif fault == 'wrong_source': value['source_manifest'] = f.catalog
    else: value['actual_episode_denominator'] = 1295
    manifest, review = f.bind_dataset(value)
    with pytest.raises(ValueError): f.proof_dataset(intent, manifest, review)


@pytest.mark.parametrize('fault', ['raw_case_changed', 'audit_changed', 'extra_output', 'no_dataset_binding', 'revoked_namespace', 'revoked_dataset', 'wrong_review'])
def test_mutation_or_revocation_cannot_use_the_old_proof(tmp_path, fault):
    f = DatasetFixture(tmp_path); intent, manifest, review = f.dataset(); row = f.read(manifest)['records'][0]
    if fault == 'raw_case_changed': (f.run/row['case']['path']).write_text('{}')
    elif fault == 'audit_changed': (f.run/row['generation_audit']['path']).write_text('{}')
    elif fault == 'extra_output': f.put('datasets/main.v1/failed-generation.json', {'role': 'partial'})
    elif fault == 'no_dataset_binding':
        f.owner['rounds'] = [r for r in f.owner['rounds'] if r['decision']['finding']['id'] != module.DATASET_FINDINGS['main']]; f.save()
    elif fault.startswith('revoked_'):
        target = helper.module.NAMESPACE_FINDINGS['main'] if fault == 'revoked_namespace' else module.DATASET_FINDINGS['main']
        f.bind('synthetic-revocation', [f.catalog], revises=target)
    else:
        value = f.read(review); value['manifest_sha256'] = '0'*64
        review = f.put(review['path'], value)
    with pytest.raises(ValueError): f.proof_dataset(intent, manifest, review)


def test_holdout_typed_dataset_requires_the_exact_locked_main_dataset(tmp_path):
    f = DatasetFixture(tmp_path); main_refs = f.dataset(); opening = f.typed_main_opening(main_refs[1])
    refs = f.dataset('holdout', opening)
    proof = f.proof_dataset(*refs, opening)
    assert len(proof.records) == 12 and proof.namespace.phase == 'holdout'


@pytest.mark.parametrize('field', ['sensor_seed', 'case_sha256', 'configuration_sha256'])
def test_holdout_dataset_cannot_collide_with_locked_main_inputs(tmp_path, field):
    f = DatasetFixture(tmp_path); main_refs = f.dataset(); main_row = f.read(main_refs[1])['records'][0]
    opening = f.typed_main_opening(main_refs[1]); intent, manifest, review = f.dataset('holdout', opening)
    value = f.read(manifest); value['records'][0][field] = main_row[field]
    manifest, review = f.bind_dataset(value)
    with pytest.raises(ValueError, match='collision'): f.proof_dataset(intent, manifest, review, opening)


def test_holdout_root_seed_cannot_reuse_a_locked_main_sensor_seed(tmp_path):
    f = DatasetFixture(tmp_path); main_refs = f.dataset(); opening = f.typed_main_opening(main_refs[1])
    intent_ref, manifest, review = f.dataset('holdout', opening)
    intent = f.read(intent_ref); name = intent['root_metadata'][0]['case_id']
    intent['root_generation_seeds'][name] = 3000  # First main sensor, not a main root/master.
    intent_ref = f.put(intent_ref['path'], intent)
    remove = {helper.module.NAMESPACE_FINDINGS['holdout'], module.DATASET_FINDINGS['holdout']}
    f.owner['rounds'] = [row for row in f.owner['rounds'] if row['decision']['finding']['id'] not in remove]
    f.seal_intent(intent, intent_ref)
    value = f.read(manifest); value['namespace_intent'] = intent_ref
    value['records'][0]['root_generation_seed'] = 3000
    manifest, review = f.bind_dataset(value)
    with pytest.raises(ValueError, match='prior namespace/root seed collision'):
        f.proof_dataset(intent_ref, manifest, review, opening)


@pytest.mark.parametrize('fault', ['stop', 'source_drift'])
def test_latest_native_snapshot_rechecks_source_and_liveness_before_dataset_effect(tmp_path, fault):
    f = DatasetFixture(tmp_path); refs = f.dataset(); original = f.verification.owner_reader
    calls = 0
    def change_on_final_read(run):
        nonlocal calls
        owner = original(run); calls += 1
        if calls == 4:
            if fault == 'stop': owner['task_stop'] = {'role': 'synthetic stop'}
            else: (f.project/'eval/inspection_formal_admission.py').write_text('synthetic changed bytes')
        return owner
    verification = replace(f.verification, owner_reader=change_on_final_read)
    with pytest.raises(ValueError):
        module.read_dataset_proof(f.run, f.project, f.acceptance, *refs, verification=verification)
    assert calls == 4
