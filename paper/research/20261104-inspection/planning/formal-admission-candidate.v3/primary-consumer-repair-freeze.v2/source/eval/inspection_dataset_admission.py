"""Read-only, native-bound formal dataset provenance; no draw or dispatch.

Hash private case/audit bytes without deserializing them. The worker's separate
content reader must check the selected semantic/configuration identities before
warmup. A genuine independent dataset review supplies the content judgment;
these structural checks cannot manufacture that judgment.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re

from eval.inspection_formal_admission import Artifact, file_sha, require_native_binding
from eval.inspection_formal_namespace import CONDITION_REPEATS
from eval.inspection_namespace_admission import (
    NAMESPACE_FINDINGS, NamespaceProof, read_namespace_proof, require_precedes, unsigned,
)

DATASET_FINDINGS = {'main': 'inspection-main-dataset-v1',
                    'holdout': 'inspection-holdout-dataset-v1'}


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    domain: str
    level: int
    case: Artifact
    case_sha256: str
    configuration_sha256: str
    sensor_seed: int
    root_generation_seed: int
    generation_audit: Artifact


@dataclass(frozen=True)
class DatasetProof:
    manifest: Artifact
    review: Artifact
    namespace: NamespaceProof
    records: tuple
    native_request_sha256: str
    report: Artifact
    main_dataset: DatasetProof | None = None


def _check_dataset_tree(root, namespace, manifest, review, records):
    expected = {namespace.intent.path, manifest.path, review.path,
                *[ref.path for row in records for ref in (row.case, row.generation_audit)]}
    if namespace.holdout_opening is not None:
        expected.add(namespace.holdout_opening.path)
    folder = 'datasets/'+namespace.phase+'.v1'
    paths = list((root/folder).rglob('*'))
    if (any(path.is_symlink() for path in paths)
            or {str(path.relative_to(root)) for path in paths if path.is_file()} != expected):
        raise ValueError('complete fixed dataset tree required; no partial/failed/extra outputs')


def _recheck_dataset_closure(root, proof):
    # Reclose the SAME frozen versions after the final owner/source check.
    # This is a bounded admission check, not an atomic filesystem capability.
    if proof.main_dataset is not None:
        _recheck_dataset_closure(root, proof.main_dataset)
    for ref in (proof.manifest, proof.review, proof.report):
        ref.checked_path(root)
    for row in proof.records:
        row.case.checked_path(root)
        row.generation_audit.checked_path(root)
    _check_dataset_tree(root, proof.namespace, proof.manifest, proof.review, proof.records)


def hex_digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('native lowercase SHA256 required')
    return value


def digest_inventory(value):
    if type(value) is not list or not value:
        raise ValueError('nonempty qualified identity inventory required')
    result = [hex_digest(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError('identity inventory contains duplicates')
    return set(result)


def read_dataset_proof(run, project_root, acceptance_value, intent_value, manifest_value,
                       review_value, *, verification, holdout_opening_value=None):
    root = Path(run).resolve()
    namespace = read_namespace_proof(root, project_root, acceptance_value, intent_value,
        verification=verification, holdout_opening_value=holdout_opening_value)
    phase = namespace.phase
    folder = 'datasets/'+phase+'.v1'
    manifest, review = Artifact.parse(manifest_value), Artifact.parse(review_value)
    if (manifest.path != folder+'/manifest.json'
            or review.path != folder+'/independent-review.json'):
        raise ValueError('canonical dataset and independent review paths required')
    value, judgment = manifest.json(root), review.json(root)
    fields = {'schema', 'phase', 'status', 'g4_acceptance', 'namespace_intent', 'source_manifest',
              'population', 'prior_catalog', 'root_denominator', 'actual_episode_denominator',
              'root_mean_denominator', 'records', 'native_finding_id'}
    if (type(value) is not dict or set(value) != fields
            or value['schema'] != 'inspection-formal-dataset-v1'
            or value['phase'] != phase or value['status'] != 'generation_complete_not_episode_results'
            or value['native_finding_id'] != DATASET_FINDINGS[phase]):
        raise ValueError('complete typed formal dataset required')
    intent = namespace.intent.json(root)
    for field, expected in [('g4_acceptance', namespace.g4_acceptance),
                            ('namespace_intent', namespace.intent),
                            ('source_manifest', namespace.source_manifest),
                            ('population', Artifact.parse(intent['population'])),
                            ('prior_catalog', namespace.prior_catalog)]:
        if Artifact.parse(value[field]) != expected:
            raise ValueError('dataset changed an admitted namespace subject')
    expected_metadata = intent['root_metadata']
    count = 72 if phase == 'main' else 12
    for name, expected in [('root_denominator', count),
                           ('actual_episode_denominator', count*sum(CONDITION_REPEATS.values())),
                           ('root_mean_denominator', count*len(CONDITION_REPEATS))]:
        if type(value[name]) is not int or value[name] != expected:
            raise ValueError('complete accepted dataset denominator required')
    if (type(judgment) is not dict or set(judgment) != {'schema', 'decision', 'manifest_sha256',
            'reviewer', 'unresolved_blocking_findings', 'report'}
            or judgment['schema'] != 'inspection-formal-dataset-review-v1'
            or judgment['decision'] != 'accept_generation_identity_and_collision_scope'
            or judgment['manifest_sha256'] != manifest.sha256
            or judgment['unresolved_blocking_findings'] != []
            or type(judgment['reviewer']) is not str or not judgment['reviewer']):
        raise ValueError('independent review must bind this exact complete dataset')
    report = Artifact.parse(judgment['report'])
    if not report.path.startswith('review/'):
        raise ValueError('separate dataset review report required')
    report.checked_path(root)
    catalog = namespace.prior_catalog.json(root)
    prior_semantics = digest_inventory(catalog.get('case_sha256_values'))
    prior_configs = digest_inventory(catalog.get('configuration_sha256_values'))
    prior_seeds = {unsigned(seed, 128) for seed in catalog['seed_values']}
    main = None
    if phase == 'holdout':
        # The opening/primary-lock chain was verified BEFORE holdout intent.
        # Now include the exact locked main dataset, not an arbitrary file.
        opening = namespace.holdout_opening.json(root)
        lock = Artifact.parse(opening['main_analysis_lock']).json(root)
        locked_subject = Artifact.parse(lock['subject']).json(root)
        main_manifest = Artifact.parse(locked_subject['main_dataset'])
        main_value = main_manifest.json(root)
        main_review_path = 'datasets/main.v1/independent-review.json'
        main_review = Artifact(main_review_path, file_sha(root/main_review_path))
        main = read_dataset_proof(root, project_root, acceptance_value,
            main_value['namespace_intent'], locked_subject['main_dataset'],
            {'path': main_review.path, 'sha256': main_review.sha256}, verification=verification)
        prior_semantics.update(row.case_sha256 for row in main.records)
        prior_configs.update(row.configuration_sha256 for row in main.records)
        prior_seeds.update([main.namespace.master_seed,
            *[seed for _, seed in main.namespace.root_generation_seeds],
            *[row.sensor_seed for row in main.records]])
    root_seeds = dict(namespace.root_generation_seeds)
    if namespace.master_seed in prior_seeds or set(root_seeds.values()) & prior_seeds:
        raise ValueError('prior namespace/root seed collision')
    used_seeds = prior_seeds | {namespace.master_seed, *root_seeds.values()}
    if type(value['records']) is not list or len(value['records']) != count:
        raise ValueError('one exact record per accepted root required')
    records = []
    for row, metadata in zip(value['records'], expected_metadata):
        if (type(row) is not dict or set(row) != {'case_id', 'domain', 'level', 'case',
                'case_sha256', 'configuration_sha256', 'sensor_seed', 'root_generation_seed', 'generation_audit'}
                or {key: row[key] for key in ('case_id', 'domain', 'level')} != metadata
                or type(row['level']) is not int):
            raise ValueError('dataset root order/metadata differs from its accepted namespace')
        seed = unsigned(row['sensor_seed'], 128)
        root_seed = unsigned(row['root_generation_seed'], 128)
        semantic, config = hex_digest(row['case_sha256']), hex_digest(row['configuration_sha256'])
        if (root_seed != root_seeds[row['case_id']] or seed in used_seeds
                or semantic in prior_semantics or config in prior_configs):
            raise ValueError('seed/semantic/configuration collision or wrong generation provenance')
        used_seeds.add(seed); prior_semantics.add(semantic); prior_configs.add(config)
        case, audit = Artifact.parse(row['case']), Artifact.parse(row['generation_audit'])
        if (case.path != folder+'/cases/'+row['case_id']+'.json'
                or audit.path != folder+'/generation/'+row['case_id']+'.json'):
            raise ValueError('canonical per-root case/audit artifacts required')
        case.checked_path(root); audit.checked_path(root)  # Hash only; no private deserialization.
        records.append(DatasetCase(row['case_id'], row['domain'], row['level'], case,
            semantic, config, seed, root_seed, audit))
    _check_dataset_tree(root, namespace, manifest, review, records)
    owner = verification.owner_reader(root)
    snapshot = replace(verification, owner_reader=lambda _: owner)
    fresh = read_namespace_proof(root, project_root, acceptance_value, intent_value,
        verification=snapshot, holdout_opening_value=holdout_opening_value)
    if fresh != namespace:
        raise ValueError('namespace changed during dataset admission')
    issued = require_native_binding(owner, DATASET_FINDINGS[phase], [manifest, review])
    require_precedes(owner, NAMESPACE_FINDINGS[phase], DATASET_FINDINGS[phase])
    result = DatasetProof(manifest, review, namespace, tuple(records), issued, report, main)
    _recheck_dataset_closure(root, result)
    return result
