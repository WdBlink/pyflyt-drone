"""Namespace-verified population generation component; no file writer.

This staged component creates no namespace, samples no master/root seed and
grants no dataset/episode admission. Its caller must retain generation errors
and complete the dataset-wide collision and final provenance checks.
"""
from pathlib import Path

import numpy as np

from eval.inspection_namespace_admission import read_namespace_proof


def generate_formal_population_case(run, project_root, acceptance_value, intent_value,
                                   case_id, *, verification, holdout_opening_value=None):
    if np.__version__ != '1.26.4':
        raise ValueError('formal population uses the qualified NumPy1.26.4 environment')
    # The same native-applied namespace/G4 predicate is checked before the
    # shared law can construct a SeedSequence or any generator.
    proof = read_namespace_proof(run, project_root, acceptance_value, intent_value,
        verification=verification, holdout_opening_value=holdout_opening_value)
    root = Path(run).resolve()
    intent = proof.intent.json(root)
    matches = [row for row in intent['root_metadata'] if row['case_id'] == case_id]
    seeds = dict(proof.root_generation_seeds)
    if type(case_id) is not str or len(matches) != 1 or case_id not in seeds:
        raise ValueError('generation requires one accepted namespace root; no alias or replacement')
    metadata = matches[0]
    from eval.inspection_population import _generate_population_case
    case, audit = _generate_population_case(case_id, proof.phase, metadata['domain'],
                                            'L'+str(metadata['level']), seeds[case_id])
    audit = {**audit, 'phase': proof.phase,
             'namespace_intent': {'path': proof.intent.path, 'sha256': proof.intent.sha256},
             'native_namespace_request_sha256': proof.native_request_sha256}
    return case, audit
