"""Read-only native-bound formal seed intents and one-time holdout opening.

No generator, filesystem writer or dispatcher. Consumers must call this at
their side-effect boundary, rather than treating a returned object as a permit
that survives changed qualification, sources or native findings.
"""
from dataclasses import dataclass
from pathlib import Path

from eval.inspection_formal_admission import Artifact, read_g4_proof, require_native_binding
from eval.inspection_formal_namespace import accepted_phase_layout, require_canonical_namespace_path

NAMESPACE_FINDINGS = {'main': 'inspection-main-namespace-v1',
                      'holdout': 'inspection-holdout-namespace-v1'}
HOLDOUT_FINDING = 'inspection-holdout-open-once-v1'
MAIN_LOCK_FINDING = 'inspection-main-primary-analysis-lock-v1'
RNG = {'library': 'numpy', 'version': '1.26.4', 'root_rng': 'SeedSequence',
       'draw_rng': 'PCG64', 'byte_order': 'little'}


@dataclass(frozen=True)
class NamespaceProof:
    intent: Artifact
    g4_acceptance: Artifact
    source_manifest: Artifact
    prior_catalog: Artifact
    phase: str
    master_seed: int
    root_generation_seeds: tuple
    schedule_seed: int
    sampler_master_seed: int
    native_request_sha256: str
    holdout_opening: Artifact | None


def same_artifact(value, expected):
    if Artifact.parse(value) != expected:
        raise ValueError('namespace changed an already accepted scientific subject')


def unsigned(value, bits):
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ValueError('native unsigned seed in the declared range required')
    return value


def current_g4_binding(root, g4, owner):
    """Check qualification again in the state used to validate the child intent."""
    active = owner.get('active_round')
    expected_owner = g4.subject.json(root)['owner']
    if (owner.get('task_stop') is not None or type(active) is not dict
            or type(active.get('id')) is not str or not active['id']
            or type(active.get('stage')) is not str or not active['stage']
            or {key: owner.get(key) for key in expected_owner} != expected_owner):
        raise ValueError('current active owner and unchanged qualified contract required')
    qualification = g4.acceptance.json(root)
    return require_native_binding(owner, qualification['native_finding_id'],
                                  [g4.acceptance, g4.subject, g4.review])


def read_primary_analysis_lock(root, lock_value, g4, owner):
    """Validate the immutable, independently reviewed main-analysis lock chain.

    Its producer/reviewer must actually validate the full raw recording and
    primary statistics. These hashes bind that work, not a claim that arbitrary
    status labels establish completion. No holdout input belongs to this lock.
    """
    lock = Artifact.parse(lock_value)
    if lock.path != 'analysis/main-primary-lock.v1.json':
        raise ValueError('one canonical main primary analysis lock required')
    value = lock.json(root)
    if (type(value) is not dict or set(value) != {'schema', 'status', 'subject', 'review', 'native_finding_id'}
            or value['schema'] != 'inspection-main-primary-analysis-lock-v1'
            or value['status'] != 'locked' or value['native_finding_id'] != MAIN_LOCK_FINDING):
        raise ValueError('native-bound typed main analysis lock required')
    subject_ref, review_ref = Artifact.parse(value['subject']), Artifact.parse(value['review'])
    subject, review = subject_ref.json(root), review_ref.json(root)
    if (type(subject) is not dict or type(review) is not dict
            or subject.get('schema') != 'inspection-main-primary-analysis-subject-v1'
            or subject.get('phase') != 'main'
            or subject.get('root_denominator') != 72 or type(subject.get('root_denominator')) is not int
            or subject.get('actual_episode_denominator') != 1296
            or type(subject.get('actual_episode_denominator')) is not int
            or subject.get('root_mean_denominator') != 576
            or type(subject.get('root_mean_denominator')) is not int
            or subject.get('confirmatory_family_size') != 6
            or type(subject.get('confirmatory_family_size')) is not int):
        raise ValueError('complete accepted main recording and six-test analysis subject required')
    same_artifact(subject.get('design'), g4.design)
    same_artifact(subject.get('source_manifest'), g4.source_manifest)
    # Exact raw/analysis evidence, not only denominator claims, must be bound.
    for key in ('main_dataset', 'recording_results', 'primary_statistics', 'all_model_settlements'):
        Artifact.parse(subject.get(key)).checked_path(root)
    if (review.get('schema') != 'inspection-main-primary-analysis-review-v1'
            or review.get('decision') != 'accept_main_primary_analysis_lock'
            or review.get('subject_sha256') != subject_ref.sha256
            or review.get('unresolved_blocking_findings') != []
            or type(review.get('reviewer')) is not str or not review['reviewer']):
        raise ValueError('actual independent main analysis review must bind the exact subject')
    require_native_binding(owner, MAIN_LOCK_FINDING, [lock, subject_ref, review_ref])
    return lock


def read_holdout_opening(root, opening_value, g4, owner, design):
    opening = Artifact.parse(opening_value)
    if opening.path != 'datasets/holdout.v1/opening.json':
        raise ValueError('one canonical holdout opening required')
    value = opening.json(root)
    if (type(value) is not dict or set(value) != {'schema', 'role', 'g4_acceptance', 'source_manifest', 'main_analysis_lock',
                       'holdout_root_ids', 'native_finding_id'}
            or value['schema'] != 'inspection-holdout-opening-v1'
            or value['role'] != 'once_only_descriptive_direction_and_size'
            or value['native_finding_id'] != HOLDOUT_FINDING):
        raise ValueError('one-time descriptive holdout opening, not another confirmatory test')
    same_artifact(value['g4_acceptance'], g4.acceptance)
    same_artifact(value['source_manifest'], g4.source_manifest)
    layout = accepted_phase_layout(design, 'holdout')
    if value['holdout_root_ids'] != list(layout.roots):
        raise ValueError('holdout opening changed the accepted twelve ID roots')
    read_primary_analysis_lock(root, value['main_analysis_lock'], g4, owner)
    # A fixed finding ID is the durable once latch. A new file/new finding alias
    # cannot authorize another opening; native duplicate IDs are not accepted.
    require_native_binding(owner, HOLDOUT_FINDING, [opening])
    return opening


def read_namespace_proof(run, project_root, acceptance_value, intent_value, *, verification, holdout_opening_value=None):
    root = Path(run).resolve()
    # This occurs before reading any proposed formal seed intent.
    g4 = read_g4_proof(root, project_root, acceptance_value, verification=verification, for_dispatch=True)
    intent = Artifact.parse(intent_value)
    phases = {'datasets/main.v1/namespace-intent.json': 'main',
              'datasets/holdout.v1/namespace-intent.json': 'holdout'}
    if intent.path not in phases:
        raise ValueError('one canonical phase intent; no replacement namespace')
    phase = phases[intent.path]
    design = g4.design.json(root)
    layout = accepted_phase_layout(design, phase)
    owner = verification.owner_reader(root)
    current_g4_binding(root, g4, owner)
    opening = None
    if phase == 'holdout':
        # Verify main lock and opening BEFORE reading any holdout seed bytes.
        opening = read_holdout_opening(root, holdout_opening_value, g4, owner, design)
    elif holdout_opening_value is not None:
        raise ValueError('main cannot carry a holdout opening')
    value = intent.json(root)
    fields = {'schema', 'phase', 'g4_acceptance', 'source_manifest', 'population', 'prior_catalog',
              'master_seed', 'root_generation_seeds', 'schedule_seed', 'sampler_master_seed',
              'root_metadata', 'rng', 'holdout_opening', 'native_finding_id'}
    if type(value) is not dict or set(value) != fields or value['schema'] != 'inspection-formal-namespace-intent-v1':
        raise ValueError('typed, complete formal namespace seed intent required')
    if value['phase'] != phase:
        raise ValueError('intent phase differs from its canonical namespace')
    require_canonical_namespace_path(layout, intent.path)
    same_artifact(value['g4_acceptance'], g4.acceptance)
    same_artifact(value['source_manifest'], g4.source_manifest)
    same_artifact(value['population'], g4.population)
    subject = g4.subject.json(root)
    catalog = Artifact.parse(subject.get('development_identity_catalog'))
    same_artifact(value['prior_catalog'], catalog)
    # It is part of the actually qualified evidence, not an arbitrary new list
    # supplied by the generator to hide known development collisions.
    if {'path': catalog.path, 'sha256': catalog.sha256} not in subject['qualification_evidence']:
        raise ValueError('development collision catalog must belong to the G4 evidence')
    catalog_value = catalog.json(root)
    if type(catalog_value) is not dict or catalog_value.get('schema') != 'inspection-development-identity-catalog-v1':
        raise ValueError('qualified development identity catalog required')
    expected = [{'case_id': case_id, 'domain': domain, 'level': level}
                for case_id, domain, level in layout.metadata]
    if (value['root_metadata'] != expected or value['rng'] != RNG
            or type(value['root_metadata']) is not list
            or any(type(row) is not dict or type(row.get('level')) is not int for row in value['root_metadata'])):
        raise ValueError('fixed root order and accepted RNG environment required')
    master = unsigned(value['master_seed'], 128)
    seeds = value['root_generation_seeds']
    if type(seeds) is not dict or set(seeds) != set(layout.roots):
        raise ValueError('exactly one generation seed per accepted root required')
    pairs = tuple((name, unsigned(seeds[name], 128)) for name in layout.roots)
    all_seeds = [seed for _, seed in pairs]
    if len(set(all_seeds)) != len(all_seeds) or master in all_seeds:
        raise ValueError('namespace/root seed collision; no silent replacement')
    prior_seeds = catalog_value.get('seed_values')
    if type(prior_seeds) is not list or not prior_seeds:
        raise ValueError('qualified development seed catalog cannot be empty')
    prior_seeds = [unsigned(seed, 128) for seed in prior_seeds]
    if len(prior_seeds) != len(set(prior_seeds)) or master in prior_seeds or set(all_seeds) & set(prior_seeds):
        raise ValueError('development/namespace seed collision; no replacement seed')
    schedule = unsigned(value['schedule_seed'], 32)
    sampler = unsigned(value['sampler_master_seed'], 32)
    owner = verification.owner_reader(root)
    current_g4_binding(root, g4, owner)
    finding_id = NAMESPACE_FINDINGS[layout.phase]
    if value['native_finding_id'] != finding_id:
        raise ValueError('fixed native namespace identity required; no reseeding alias')
    if layout.phase == 'main':
        if value['holdout_opening'] is not None:
            raise ValueError('main namespace cannot open holdout')
    else:
        same_artifact(value['holdout_opening'], opening)
        # Check the same parent chain against the final owner snapshot as well.
        read_holdout_opening(root, value['holdout_opening'], g4, owner, design)
    issued = require_native_binding(owner, finding_id, [intent])
    return NamespaceProof(intent, g4.acceptance, g4.source_manifest, catalog, layout.phase,
                          master, pairs, schedule, sampler, issued, opening)
