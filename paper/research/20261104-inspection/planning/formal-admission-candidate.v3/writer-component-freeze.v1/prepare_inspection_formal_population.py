"""One-shot formal data effects behind native G4/namespace proofs.

No scheduler or owner mutation. Each returned artifact must receive its real
independent review/native finding before the next gate admits it. Existing,
failed or partial namespaces are preserved; this entry never reseeds them.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.inspection_dataset_admission import DATASET_FINDINGS, digest_inventory, read_dataset_proof
from eval.inspection_development_catalog import physical_configuration_digest
from eval.inspection_episode_receipts import _mkdir, _sync_directory, durable_once
from eval.inspection_formal_admission import Artifact, context_from_environment, file_sha, read_g4_proof
from eval.inspection_formal_namespace import accepted_phase_layout, CONDITION_REPEATS
from eval.inspection_formal_population import generate_formal_population_case
from eval.inspection_namespace_admission import (HOLDOUT_FINDING, NAMESPACE_FINDINGS, RNG,
    current_g4_binding, read_holdout_opening, read_namespace_proof, read_primary_analysis_lock)
from eval.inspection_population import PopulationGenerationExhausted


def artifact(root, path):
    return {'path': str(path.relative_to(root)), 'sha256': file_sha(path)}


def _environment():
    if np.__version__ != '1.26.4' or sys.byteorder != 'little':
        raise ValueError('qualified NumPy1.26.4/little-endian environment required before any draw')


def _folder(root, phase):
    path = root/'datasets'/(phase+'.v1')
    for parent in (root/'datasets', path):
        if parent.is_symlink() or parent.exists() and not parent.is_dir():
            raise ValueError('canonical dataset directories cannot be aliases or non-directories')
    return path


def _typed_main_dataset(root, project, acceptance_value, lock_value, *, verification):
    g4 = read_g4_proof(root, project, acceptance_value, verification=verification)
    owner = verification.owner_reader(root)
    current_g4_binding(root, g4, owner, verification)
    lock = read_primary_analysis_lock(root, lock_value, g4, owner, verification)
    subject = Artifact.parse(lock.json(root)['subject']).json(root)
    manifest = Artifact.parse(subject['main_dataset'])
    value = manifest.json(root)
    review_path = root/'datasets/main.v1/independent-review.json'
    return read_dataset_proof(root, project, acceptance_value, value['namespace_intent'],
        subject['main_dataset'], artifact(root, review_path), verification=verification)


def write_holdout_opening(run, project_root, acceptance_value, lock_value, *, verification):
    root, project = Path(run).resolve(), Path(project_root).resolve()
    g4 = read_g4_proof(root, project, acceptance_value, verification=verification)
    design = g4.design.json(root)
    layout = accepted_phase_layout(design, 'holdout')
    _typed_main_dataset(root, project, acceptance_value, lock_value, verification=verification)
    folder = _folder(root, 'holdout')
    if folder.exists():
        raise ValueError('holdout directory already exists; recover its exact evidence, never reopen')
    _mkdir(folder.parent)
    folder.mkdir()  # Exclusive claim before any opening write; no RNG here.
    _sync_directory(folder.parent)
    value = {'schema': 'inspection-holdout-opening-v1',
        'role': 'once_only_descriptive_direction_and_size',
        'g4_acceptance': acceptance_value,
        'source_manifest': {'path': g4.source_manifest.path, 'sha256': g4.source_manifest.sha256},
        'main_analysis_lock': lock_value, 'holdout_root_ids': list(layout.roots),
        'native_finding_id': HOLDOUT_FINDING}
    durable_once(folder/'opening.json', value)
    return artifact(root, folder/'opening.json')


def write_namespace_intent(run, project_root, acceptance_value, phase, *, verification,
                           holdout_opening_value=None):
    _environment()
    root, project = Path(run).resolve(), Path(project_root).resolve()
    g4 = read_g4_proof(root, project, acceptance_value, verification=verification)
    design = g4.design.json(root)
    layout = accepted_phase_layout(design, phase)
    catalog_ref = Artifact.parse(g4.subject.json(root)['development_identity_catalog'])
    catalog = catalog_ref.json(root)
    prior_seeds = set(catalog['seed_values'])
    folder = _folder(root, phase)
    if phase == 'holdout':
        owner = verification.owner_reader(root)
        current_g4_binding(root, g4, owner, verification)
        opening = read_holdout_opening(root, holdout_opening_value, g4, owner, design, verification)
        opening_body = opening.json(root)
        main = _typed_main_dataset(root, project, acceptance_value,
            opening_body['main_analysis_lock'], verification=verification)
        prior_seeds.update([main.namespace.master_seed,
            *[seed for _, seed in main.namespace.root_generation_seeds],
            *[row.sensor_seed for row in main.records]])
        if {str(p.relative_to(root)) for p in folder.rglob('*') if p.is_file()} != {opening.path}:
            raise ValueError('holdout namespace must contain only its bound opening before seed claim')
    else:
        if holdout_opening_value is not None:
            raise ValueError('main namespace cannot carry a holdout opening')
        if folder.exists():
            raise ValueError('existing main namespace cannot be replaced or reseeded')
        _mkdir(folder.parent)
        folder.mkdir()
        _sync_directory(folder.parent)
    path = folder/'namespace-intent.json'
    # Claim the final canonical file BEFORE touching entropy. A crash leaves
    # an empty/partial exclusive file; a later invocation cannot silently draw
    # a replacement. Full intent is flushed before geometry is ever admitted.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    intent_durable = False
    try:
        with os.fdopen(fd, 'w') as stream:
            _sync_directory(folder)
            master = secrets.randbits(128)
            branches = np.random.SeedSequence(master).spawn(len(layout.roots))
            seeds = [int.from_bytes(child.generate_state(4).tobytes(), 'little') for child in branches]
            value = {'schema': 'inspection-formal-namespace-intent-v1', 'phase': phase,
                'g4_acceptance': acceptance_value,
                'source_manifest': {'path': g4.source_manifest.path, 'sha256': g4.source_manifest.sha256},
                'population': {'path': g4.population.path, 'sha256': g4.population.sha256},
                'prior_catalog': {'path': catalog_ref.path, 'sha256': catalog_ref.sha256},
                'master_seed': master, 'root_generation_seeds': dict(zip(layout.roots, seeds)),
                'schedule_seed': secrets.randbits(32), 'sampler_master_seed': secrets.randbits(32),
                'root_metadata': [{'case_id': name, 'domain': domain, 'level': level}
                                  for name, domain, level in layout.metadata],
                'rng': RNG, 'holdout_opening': holdout_opening_value,
                'native_finding_id': NAMESPACE_FINDINGS[phase]}
            stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        _sync_directory(folder)
        intent_durable = True
        if (len(set(seeds)) != len(seeds) or master in seeds or master in prior_seeds
                or set(seeds) & prior_seeds):
            raise ValueError('namespace seed collision; persisted intent is blocked, no replacement draw')
        fresh = read_g4_proof(root, project, acceptance_value, verification=verification)
        if fresh != g4:
            raise ValueError('qualification changed during seed registration')
    except BaseException as error:
        durable_once(folder/'namespace-failure.json', {'status': 'blocked_no_reseed',
            'error_type': type(error).__name__, 'reason': str(error),
            'complete_intent_available': intent_durable})
        raise
    return artifact(root, path)


def write_geometry(run, project_root, acceptance_value, intent_value, *, verification,
                   holdout_opening_value=None):
    _environment()
    root, project = Path(run).resolve(), Path(project_root).resolve()
    proof = read_namespace_proof(root, project, acceptance_value, intent_value,
        verification=verification, holdout_opening_value=holdout_opening_value)
    intent = proof.intent.json(root)
    folder = _folder(root, proof.phase)
    expected = {proof.intent.path}
    if proof.holdout_opening is not None: expected.add(proof.holdout_opening.path)
    if ({str(p.relative_to(root)) for p in folder.rglob('*') if p.is_file()} != expected
            or (folder/'cases').exists() or (folder/'generation').exists()):
        raise ValueError('existing/partial geometry is preserved; no automatic regeneration')
    catalog = proof.prior_catalog.json(root)
    semantics = digest_inventory(catalog.get('case_sha256_values'))
    configs = digest_inventory(catalog.get('configuration_sha256_values'))
    prior_seeds = set(catalog['seed_values'])
    if proof.phase == 'holdout':
        main = _typed_main_dataset(root, project, acceptance_value,
            proof.holdout_opening.json(root)['main_analysis_lock'], verification=verification)
        semantics.update(row.case_sha256 for row in main.records)
        configs.update(row.configuration_sha256 for row in main.records)
        prior_seeds.update([main.namespace.master_seed,
            *[seed for _, seed in main.namespace.root_generation_seeds],
            *[row.sensor_seed for row in main.records]])
    roots = dict(proof.root_generation_seeds)
    if proof.master_seed in prior_seeds or set(roots.values()) & prior_seeds:
        raise ValueError('prior namespace/root seed collision before geometry draw')
    used_seeds = prior_seeds | {proof.master_seed, *roots.values()}
    # Exclusive directories are the one-shot generation claim, before any RNG.
    (folder/'cases').mkdir(); _sync_directory(folder)
    (folder/'generation').mkdir(); _sync_directory(folder)
    records = []
    try:
        for metadata in intent['root_metadata']:
            name = metadata['case_id']; audit_path = folder/'generation'/(name+'.json')
            try:
                case, audit = generate_formal_population_case(root, project, acceptance_value,
                    intent_value, name, verification=verification,
                    holdout_opening_value=holdout_opening_value)
            except PopulationGenerationExhausted as error:
                durable_once(audit_path, {'status': 'finite_generation_exhausted',
                    'root_generation_seed': roots[name], 'attempts': error.audit,
                    'namespace_intent': intent_value})
                raise
            durable_once(audit_path, audit)
            case_path = folder/'cases'/(name+'.json')
            durable_once(case_path, case.as_dict())
            semantic, config = case.digest(), physical_configuration_digest(case)
            if case.seed in used_seeds or semantic in semantics or config in configs:
                raise ValueError('generated seed/semantic/configuration collision; raw case/audit retained')
            used_seeds.add(case.seed); semantics.add(semantic); configs.add(config)
            records.append({**metadata, 'case': artifact(root, case_path), 'case_sha256': semantic,
                'configuration_sha256': config, 'sensor_seed': case.seed,
                'root_generation_seed': roots[name], 'generation_audit': artifact(root, audit_path)})
        fresh = read_namespace_proof(root, project, acceptance_value, intent_value,
            verification=verification, holdout_opening_value=holdout_opening_value)
        if fresh != proof:
            raise ValueError('namespace/source changed before completing geometry manifest')
        count = len(records)
        value = {'schema': 'inspection-formal-dataset-v1', 'phase': proof.phase,
            'status': 'generation_complete_not_episode_results', 'g4_acceptance': acceptance_value,
            'namespace_intent': intent_value,
            'source_manifest': {'path': proof.source_manifest.path, 'sha256': proof.source_manifest.sha256},
            'population': intent['population'],
            'prior_catalog': {'path': proof.prior_catalog.path, 'sha256': proof.prior_catalog.sha256},
            'root_denominator': count, 'actual_episode_denominator': count*sum(CONDITION_REPEATS.values()),
            'root_mean_denominator': count*len(CONDITION_REPEATS), 'records': records,
            'native_finding_id': DATASET_FINDINGS[proof.phase]}
        durable_once(folder/'manifest.json', value)
    except BaseException as error:
        durable_once(folder/'generation-failure.json', {'status': 'blocked_no_replacement',
            'error_type': type(error).__name__, 'reason': str(error),
            'completed_root_records': len(records), 'namespace_intent': intent_value})
        raise
    return artifact(root, folder/'manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--sdk', required=True, type=Path)
    parser.add_argument('--action', choices=('namespace', 'geometry', 'holdout-opening'), required=True)
    parser.add_argument('--phase', choices=('main', 'holdout'), required=True)
    for name in ('g4', 'intent', 'opening', 'main-lock'):
        parser.add_argument('--'+name)
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    def ref(name, required=False):
        path, digest = getattr(args, name.replace('-', '_')), getattr(args, name.replace('-', '_')+'_sha256')
        if path is None and digest is None and not required: return None
        if path is None or digest is None: parser.error('both path and SHA256 required: '+name)
        value = {'path': path, 'sha256': digest}; Artifact.parse(value); return value
    project = ROOT
    verification = context_from_environment(args.run, project, args.runtime, args.sdk)
    g4, opening = ref('g4', True), ref('opening')
    if args.action == 'holdout-opening':
        if args.phase != 'holdout': parser.error('opening requires holdout phase')
        result = write_holdout_opening(args.run, project, g4, ref('main-lock', True), verification=verification)
    elif args.action == 'namespace':
        result = write_namespace_intent(args.run, project, g4, args.phase,
            verification=verification, holdout_opening_value=opening)
    else:
        intent = ref('intent', True)
        if intent['path'] != 'datasets/'+args.phase+'.v1/namespace-intent.json':
            parser.error('intent path must match the explicit phase')
        result = write_geometry(args.run, project, g4, intent,
            verification=verification, holdout_opening_value=opening)
    print(json.dumps({'status': 'artifact_written_pending_independent_native_binding', 'artifact': result}))


if __name__ == '__main__':
    main()
