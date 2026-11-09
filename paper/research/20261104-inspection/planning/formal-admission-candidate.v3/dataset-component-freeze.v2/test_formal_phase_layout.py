"""Public accepted identities only; never draw/open formal data."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

path = Path(__file__).resolve().parents[1]/'source/eval/inspection_formal_namespace.py'
spec = importlib.util.spec_from_file_location('staged_phase_layout', path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
spec.loader.exec_module(module)
RUN = Path(__file__).resolve().parents[3]


def design():
    return json.loads((RUN/'experiment/design.ral.v3.json').read_text())


@pytest.mark.parametrize('phase,roots,episodes,means', [('main', 72, 1296, 576), ('holdout', 12, 216, 96)])
def test_actual_accepted_phase_contract_contains_every_slot_without_preparing_data(phase, roots, episodes, means):
    before = (RUN/'datasets').exists()
    layout = module.accepted_phase_layout(design(), phase)
    assert len(layout.roots) == len(set(layout.roots)) == roots
    assert layout.actual_episode_denominator == episodes
    assert layout.root_mean_denominator == means
    assert layout.intent_path == f'datasets/{phase}.v1/namespace-intent.json'
    assert module.require_canonical_namespace_path(layout, layout.intent_path) == layout.intent_path
    assert (RUN/'datasets').exists() == before
    if phase == 'holdout':
        assert {domain for _, domain, _ in layout.metadata} == {'ID'}
        assert {level for _, _, level in layout.metadata} == {2, 3}


@pytest.mark.parametrize('fault', ['dropped_arm', 'boolean_repeat', 'flattened_repeat', 'denominator', 'root_metadata',
                                  'duplicate_main', 'stratum', 'holdout_ood', 'duplicate_holdout', 'holdout_denominator'])
def test_changed_denominators_or_identities_fail_closed(fault):
    value = design(); semantics = value['recording_semantics']; split = value['split_contract']
    if fault == 'dropped_arm': semantics['condition_repeats'].pop('B3-M3')
    elif fault == 'boolean_repeat': semantics['condition_repeats']['B0'] = True
    elif fault == 'flattened_repeat': semantics['condition_repeats']['B4-M1'] = 1
    elif fault == 'denominator': semantics['actual_episode_denominator'] = 1280
    elif fault == 'root_metadata': semantics['root_metadata'][0]['level'] = True
    elif fault == 'duplicate_main': value['cases'][-1] = value['cases'][0]
    elif fault == 'stratum': split['main_counts']['OOD']['L3'] = 11
    elif fault == 'holdout_ood': split['holdout_ids'][0] = 'holdout-OOD-L2-00'
    elif fault == 'duplicate_holdout': split['holdout_ids'][-1] = split['holdout_ids'][0]
    elif fault == 'holdout_denominator': split['holdout_episode_denominator'] = 215
    with pytest.raises(ValueError): module.accepted_phase_layout(value, 'main')


@pytest.mark.parametrize('value', ['datasets/main.v2/namespace-intent.json', '../main', '/tmp/main',
                                 'datasets/holdout.v1/namespace-intent.json'])
def test_replacement_namespace_paths_do_not_get_admitted(value):
    layout = module.accepted_phase_layout(design(), 'main')
    with pytest.raises(ValueError): module.require_canonical_namespace_path(layout, value)


def test_phase_name_cannot_relabel_development_or_other_holdout():
    for phase in ('dev', 'devext', 'test', 'holdout-OOD', '', None):
        with pytest.raises(ValueError): module.accepted_phase_layout(design(), phase)


@pytest.mark.parametrize('fault', ['missing_arm', 'extra_arm', 'duplicate_arm', 'missing_seed', 'extra_seed', 'boolean_seed'])
def test_actual_outer_sdk_grid_must_agree_with_the_recording_semantics(fault):
    value = design()
    if fault == 'missing_arm': value['conditions'].pop()
    elif fault == 'extra_arm': value['conditions'].append('unplanned')
    elif fault == 'duplicate_arm': value['conditions'][-1] = value['conditions'][0]
    elif fault == 'missing_seed': value['seeds'] = []
    elif fault == 'extra_seed': value['seeds'] = [0, 1]
    else: value['seeds'] = [False]
    with pytest.raises(ValueError, match='single-root-mean seed grid'):
        module.accepted_phase_layout(value, 'main')
