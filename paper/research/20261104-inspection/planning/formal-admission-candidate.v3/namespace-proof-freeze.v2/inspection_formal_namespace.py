"""Accepted phase identities and fixed namespace paths; no random/data effects.

Public identities already occur in the accepted plan. This component never
creates a namespace, draws seeds/geometry, opens holdout or grants dispatch.
"""
from dataclasses import dataclass


CONDITION_REPEATS = {'B0': 1, 'B1': 1, 'B2': 1, 'B3-M1': 3, 'B3-M2': 3,
                     'B3-M3': 3, 'B4-M1': 3, 'B4-M2': 3}
MAIN_COUNTS = {('ID', 1): 16, ('ID', 2): 16, ('ID', 3): 16,
               ('OOD', 2): 12, ('OOD', 3): 12}
HOLDOUT_COUNTS = {('ID', 2): 6, ('ID', 3): 6}


@dataclass(frozen=True)
class PhaseLayout:
    phase: str
    directory: str
    roots: tuple
    metadata: tuple
    actual_episode_denominator: int
    root_mean_denominator: int

    @property
    def intent_path(self):
        return self.directory+'/namespace-intent.json'


def _identities(phase, counts):
    return tuple(sorted((f'{phase}-{domain}-L{level}-{index:02d}', domain, level)
                        for (domain, level), count in counts.items() for index in range(count)))


def accepted_phase_layout(design, phase):
    """Check this study's full accepted denominator before any phase is prepared.

    Changes to the actual scientific plan require an explicit reviewed adapter
    revision; this function cannot quietly drop a condition or a population.
    """
    if phase not in ('main', 'holdout'):
        raise ValueError('exact main or holdout phase required')
    if (type(design.get('conditions')) is not list
            or design['conditions'] != list(CONDITION_REPEATS)
            or any(type(value) is not str for value in design['conditions'])
            or type(design.get('seeds')) is not list or design['seeds'] != [0]
            or any(type(value) is not int for value in design['seeds'])):
        raise ValueError('the accepted outer eight-condition single-root-mean seed grid is required')
    semantics = design['recording_semantics']
    split = design['split_contract']
    repeats = semantics['condition_repeats']
    if (type(repeats) is not dict or repeats != CONDITION_REPEATS
            or any(type(value) is not int for value in repeats.values())
            or semantics['root_denominator'] != 72 or type(semantics['root_denominator']) is not int
            or semantics['actual_episode_denominator'] != 1296
            or type(semantics['actual_episode_denominator']) is not int):
        raise ValueError('accepted eight-condition72root1296episode contract required')
    main = _identities('main', MAIN_COUNTS)
    records = semantics['root_metadata']
    if (type(records) is not list or len(records) != 72
            or any(type(row) is not dict or set(row) != {'case_id', 'domain', 'level'}
                   or type(row['level']) is not int for row in records)
            or tuple(sorted((row['case_id'], row['domain'], row['level']) for row in records)) != main
            or type(design['cases']) is not list or sorted(design['cases']) != [row[0] for row in main]):
        raise ValueError('accepted main identities/metadata differ')
    expected_counts = {domain: {'L'+str(level): count for (d, level), count in MAIN_COUNTS.items() if d == domain}
                       for domain in ('ID', 'OOD')}
    if (split['main_counts'] != expected_counts
            or any(type(v) is not int for group in split['main_counts'].values() for v in group.values())):
        raise ValueError('accepted main stratification differs')
    holdout = _identities('holdout', HOLDOUT_COUNTS)
    if (type(split['holdout_ids']) is not list or sorted(split['holdout_ids']) != [row[0] for row in holdout]
            or split['holdout_episode_denominator'] != 216
            or type(split['holdout_episode_denominator']) is not int):
        raise ValueError('accepted twelve ID holdout identities/216episode contract required')
    metadata = main if phase == 'main' else holdout
    return PhaseLayout(phase, 'datasets/'+phase+'.v1', tuple(row[0] for row in metadata), metadata,
                       len(metadata)*sum(repeats.values()), len(metadata)*len(repeats))


def require_canonical_namespace_path(layout, value):
    if not isinstance(layout, PhaseLayout) or type(value) is not str or value != layout.intent_path:
        raise ValueError('one exact phase namespace intent; no replacement output or reseeding namespace')
    return value
