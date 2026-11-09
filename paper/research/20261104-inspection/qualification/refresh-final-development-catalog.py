"""Inventory persisted development identities without drawing or running cases.

The reviewed installed producer owns identity decoding. This adapter binds the
actual manifests, reconciles all persisted private copies and ZIP members, and
records the separate procedural qualification inputs. It grants no G4 access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    project = run.parents[2]
    if not out.is_relative_to(run) or out.exists():
        parser.error('fresh run-relative output required')
    datasets_before = (run/'datasets').exists()
    sys.path.insert(0, str(project))
    stage = project/'eval'
    from eval import inspection_development_catalog as producer

    def no_draw(*_args, **_kwargs):
        raise AssertionError('identity inventory cannot draw seeds or cases')
    producer.np.random.default_rng = no_draw
    producer.np.random.SeedSequence = no_draw
    secrets.randbits = no_draw
    manifests = [f'qualification/{folder}/manifest.json' for folder in (
        'dev-cases.v1', 'dev-cases.v2', 'dev-population.v1',
        'dev-clean-sibling.v1', 'dev-same-level-event-pair.v2/event',
        'dev-same-level-event-pair.v2/clean')]
    refs = [{'path': name, 'sha256': sha(run/name)} for name in manifests]
    catalog = producer.build_development_catalog(run, refs)
    assert catalog == producer.build_development_catalog(run, refs)
    known = {}
    for row in catalog['case_records']:
        known.setdefault(row['case']['sha256'], []).append(row['case']['path'])
    copies, archives, procedural = [], [], []
    for directory in (project/'logs', run/'qualification'):
        for path in sorted(directory.rglob('private-case.json')):
            if path.is_relative_to(out) or 'freeze.v1' in path.parts or 'source' in path.parts:
                continue
            if path.is_symlink():
                raise ValueError('private input copy cannot be a symbolic link')
            digest = sha(path)
            if digest not in known:
                raise ValueError('unmapped persisted private input: '+str(path))
            parent = path.parent/'run.json'
            if parent.exists():
                value = json.loads(parent.read_text())
                if value.get('private_case_sha256') not in (None, digest):
                    raise ValueError('private copy differs from its actual run manifest')
            copies.append({'path': str(path.relative_to(project)), 'sha256': digest,
                'catalog_paths': known[digest],
                'run_manifest_sha256': sha(parent) if parent.exists() else None})
    for path in sorted((run/'qualification').rglob('*.zip')):
        members = []
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.filename.endswith('/private-case.json') or info.filename == 'private-case.json':
                    digest = hashlib.sha256(archive.read(info)).hexdigest()
                    if digest not in known:
                        raise ValueError('unmapped archived private input: '+str(path)+'!'+info.filename)
                    members.append({'member': info.filename, 'sha256': digest,
                                    'catalog_paths': known[digest]})
        archives.append({'path': str(path.relative_to(run)), 'sha256': sha(path), 'private_members': members})
    # These parameter roots belong to earlier control/executor/reference laws,
    # not independent benchmark cases. Include their actual integer seeds in
    # the exclusion set as a conservative extra guard; never relabel as cases.
    additional_seeds = set()
    paths = list((run/'qualification/reference').glob('*/run.json'))
    paths += [p for p in (run/'qualification').glob('*/manifest.json')
              if p.parent.name.startswith(('g1-', 'executor-'))]
    for path in sorted(paths):
        value = json.loads(path.read_text())
        seeds = []
        if type(value.get('seed')) is int:
            seeds.append(producer.seed(value['seed']))
        for row in value.get('grid', []):
            if type(row.get('root')) is int and row.get('status') != 'unstarted':
                seeds.append(producer.seed(row['root']))
        additional_seeds.update(seeds)
        procedural.append({'path': str(path.relative_to(run)), 'sha256': sha(path),
                           'parameter_seeds': sorted(set(seeds))})
    catalog['seed_values'] = sorted(set(catalog['seed_values']) | additional_seeds)
    catalog['procedural_seed_scope'] = 'conservative extra seed exclusion only; control/executor/reference laws are not benchmark CaseSpec roots'
    old_path = run/'qualification/development-identity-catalog.v1/results.v2/catalog.json'
    old = json.loads(old_path.read_text())
    for key in ('seed_values','case_sha256_values','configuration_sha256_values'):
        if catalog[key] != old[key]:
            raise ValueError('independently reviewed exclusion identities changed: '+key)
    plans = []
    source_count = 0
    for directory in (project/'logs',run/'qualification'):
        for path in sorted(directory.rglob('plan.json')):
            if 'source' in path.parts or 'freeze.v1' in path.parts:
                continue
            value = json.loads(path.read_bytes())
            if 'case_file_sha256' not in value or 'case_id' not in value:
                continue
            case_sha = value['case_file_sha256']
            if case_sha not in known or not value['case_id'].startswith(('dev-','devext-')):
                raise ValueError('unmapped typed development plan: '+str(path))
            case_source = value.get('case_path')
            if case_source is not None:
                case_source = Path(case_source)
                if (not case_source.is_absolute() or not case_source.is_relative_to(project)
                        or case_source.is_symlink() or sha(case_source)!=case_sha):
                    raise ValueError('typed plan case source differs: '+str(path))
                source_count += 1
            plans.append({'path':str(path.relative_to(project)),'sha256':sha(path),
                'case_id':value['case_id'],'case_file_sha256':case_sha,'catalog_paths':known[case_sha],
                'source_path':None if case_source is None else str(case_source.relative_to(project)),
                'policy_sampler_seed_not_environment_seed':value.get('sampler_seed')})
    coverage = {'role': 'persisted development input coverage, not universal historical RNG audit or G4',
        'previous_catalog_sha256':sha(old_path),'typed_case_plans':plans,'typed_plan_actual_source_paths':source_count,
        'private_copies': copies, 'qualification_archives': archives,
        'procedural_inputs': procedural, 'additional_procedural_seed_values': sorted(additional_seeds),
        'unmapped_private_inputs': [], 'datasets_exists_before': datasets_before,
        'no_seed_draws': True, 'source_hashes': {str(p.relative_to(project)): sha(p) for p in (
            Path(__file__).resolve(), stage/'inspection_formal_admission.py',
            stage/'inspection_development_catalog.py', project/'eval/inspection_cases.py',
            project/'eval/inspection_population.py', project/'eval/prepare_inspection_development_population.py')}}
    out.mkdir(parents=True)
    for name, value in [('catalog.json', catalog), ('coverage.json', coverage)]:
        with (out/name).open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    assert (run/'datasets').exists() == datasets_before
    binding = {'role': coverage['role'], 'files': {str((out/name).relative_to(run)): sha(out/name)
                for name in ('catalog.json', 'coverage.json')}, 'producer_sha256': sha(__file__)}
    (out/'binding.json').write_text(json.dumps(binding, indent=2)+'\n')
    print(json.dumps({'case_artifact_versions': len(catalog['case_records']),
        'semantic_identities': len(catalog['case_sha256_values']),
        'physical_configurations': len(catalog['configuration_sha256_values']),
        'private_copies': len(copies), 'archives': len(archives),
        'archived_private_copies': sum(len(a['private_members']) for a in archives),
        'seed_values': len(catalog['seed_values']),'typed_plans':len(plans),'typed_plan_source_paths':source_count,
        'binding_sha256': sha(out/'binding.json')}))


if __name__ == '__main__':
    main()
