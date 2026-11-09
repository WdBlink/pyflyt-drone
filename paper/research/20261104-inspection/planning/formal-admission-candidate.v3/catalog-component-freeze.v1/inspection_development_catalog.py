"""Build a read-only collision inventory from explicitly bound development data.

No seed draw or writer. The caller freezes the resulting catalog as G4 evidence
after independently checking coverage of the development inputs actually used.
The NumPy1.26.4 producer is separate from the standard-library-only consumers.
"""
from pathlib import Path

import numpy as np

from eval.inspection_cases import CaseSpec
from eval.inspection_formal_admission import Artifact
from eval.inspection_population import load_development_case
from eval.prepare_inspection_development_population import configuration_digest


def seed(value):
    if type(value) is not int or not 0 <= value < 2**128:
        raise ValueError('development catalog seeds must be native unsigned128 integers')
    return value


def physical_configuration_digest(case):
    # Explicit clean siblings carry parent provenance. Their physical identity
    # must not depend on nested parent IDs/seed/schema/provenance fields.
    base = CaseSpec(**{name: getattr(case, name) for name in CaseSpec.__dataclass_fields__})
    return configuration_digest(base)


def build_development_catalog(run, manifest_values, extra_case_values=()):
    if np.__version__ != '1.26.4':
        raise ValueError('catalog producer uses the qualified NumPy1.26.4 environment')
    root = Path(run).resolve()
    manifests = tuple(Artifact.parse(value) for value in manifest_values)
    if not manifests or len({ref.path for ref in manifests}) != len(manifests):
        raise ValueError('explicit distinct development manifests required')
    cases, seeds, intents = {}, set(), []

    def case_record(ref, expected_semantic=None, root_generation_seed=None):
        value = ref.json(root)
        case = load_development_case(value)
        if case.split != 'dev':
            raise ValueError('catalog cannot import main or holdout cases')
        if expected_semantic is not None and case.digest() != expected_semantic:
            raise ValueError('development semantic digest differs from its manifest')
        previous = cases.get(ref.path)
        if previous is not None:
            old_seed = previous['root_generation_seed']
            if root_generation_seed is not None and old_seed is not None and root_generation_seed != old_seed:
                raise ValueError('contradictory development generation seed provenance')
            if root_generation_seed is None: root_generation_seed = old_seed
            elif old_seed is None: previous = {**previous, 'root_generation_seed': root_generation_seed}
        seeds.add(seed(case.seed))
        if root_generation_seed is not None: seeds.add(seed(root_generation_seed))
        record = {'case_id': case.case_id, 'lineage_group': case.lineage_group,
            'case': {'path': ref.path, 'sha256': ref.sha256}, 'case_sha256': case.digest(),
            'configuration_sha256': physical_configuration_digest(case), 'sensor_seed': case.seed,
            'root_generation_seed': root_generation_seed}
        if previous is not None and previous != record:
            raise ValueError('same development artifact has contradictory provenance')
        cases[ref.path] = record

    for ref in manifests:
        value = ref.json(root)
        if (type(value) is not dict or value.get('split') != 'dev'
                or type(value.get('records')) is not list or not value['records']):
            raise ValueError('only nonempty declared development manifests admitted')
        if 'master_seed' in value: seeds.add(seed(value['master_seed']))
        folder = Path(ref.path).parent
        for row in value['records']:
            name = str(folder/row['file'])
            case_ref = Artifact.parse({'path': name, 'sha256': row['file_sha256']})
            if Path(case_ref.path).parent != folder:
                raise ValueError('case must remain beside its development manifest')
            case_record(case_ref, row['case_sha256'], row.get('root_generation_seed'))
        if 'namespace_intent_sha256' in value:
            intent_ref = Artifact.parse({'path': str(folder/'namespace-intent.json'),
                                        'sha256': value['namespace_intent_sha256']})
            intent = intent_ref.json(root)
            seeds.add(seed(intent['master_seed']))
            for value in intent['root_generation_seeds']: seeds.add(seed(value))
            intents.append({'path': intent_ref.path, 'sha256': intent_ref.sha256})
    for value in extra_case_values:
        case_record(Artifact.parse(value))
    records = [cases[name] for name in sorted(cases)]
    return {'schema': 'inspection-development-identity-catalog-v1',
        'scope': 'explicit development manifests/cases only; caller independently verifies used-input coverage before G4',
        'manifest_refs': [{'path': ref.path, 'sha256': ref.sha256} for ref in manifests],
        'namespace_intent_refs': intents, 'case_records': records,
        'seed_values': sorted(seeds), 'case_sha256_values': sorted({row['case_sha256'] for row in records}),
        'configuration_sha256_values': sorted({row['configuration_sha256'] for row in records})}
