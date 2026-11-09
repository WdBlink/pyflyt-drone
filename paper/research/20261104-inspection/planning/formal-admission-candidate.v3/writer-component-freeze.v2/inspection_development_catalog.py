"""Build a read-only collision inventory from explicitly bound development data.

No seed draw or writer. The caller freezes the resulting catalog as G4 evidence
after independently checking coverage of the development inputs actually used.
The NumPy1.26.4 producer is separate from the standard-library-only consumers.
"""
from pathlib import Path
import json
import math

import numpy as np

from eval.inspection_cases import CaseSpec, CaseEntity, SCHEMA as LEGACY_SCHEMA
from eval.inspection_formal_admission import Artifact
from eval.inspection_population import (load_development_case, PopulationCase, EventCleanSibling,
    SCHEMA as POPULATION_SCHEMA, CLEAN_SIBLING_SCHEMA)
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


def _archived_identity_case(value):
    """Decode only finite typed identity data, never grant mission admission.

    Historical cases can fail today's geometry/event laws. Reuse the declared
    dataclasses and hash helpers without invoking their executable loaders;
    callers receive catalog dictionaries, never this serialization container.
    """
    classes={LEGACY_SCHEMA:CaseSpec,POPULATION_SCHEMA:PopulationCase,CLEAN_SIBLING_SCHEMA:EventCleanSibling}
    if type(value) is not dict or type(value.get('schema')) is not str or value['schema'] not in classes:
        raise ValueError('known development archive schema required')
    cls=classes[value['schema']]
    if set(value)!=set(cls.__dataclass_fields__) or value['split']!='dev':
        raise ValueError('complete development archive fields required')
    if any(type(value[k]) is not str or not value[k] for k in ('case_id','lineage_group','level')):
        raise ValueError('typed archive identity required')
    seed(value['seed']);json.dumps(value,allow_nan=False)
    number=lambda v:type(v) is int or type(v) is float and math.isfinite(v)
    pair=lambda v:type(v) in (list,tuple) and len(v)==2 and all(number(x) for x in v)
    if (type(value['roi_polygon']) is not list or not all(pair(p) for p in value['roi_polygon'])
            or not pair(value['wind_xy']) or type(value['entities']) is not list):
        raise ValueError('typed archive geometry required')
    converted=dict(value)
    try:
        entities=[]
        for row in value['entities']:
            if type(row) is not dict or set(row)-set(CaseEntity.__dataclass_fields__) or not pair(row.get('origin_xy')):
                raise ValueError('typed archive entity required')
            entity=CaseEntity(**{**row,'origin_xy':tuple(row['origin_xy'])})
            if (any(type(getattr(entity,k)) is not str for k in ('key','object_type','true_class','rule_id'))
                    or any(not number(getattr(entity,k)) for k in ('appeared_at','speed','heading','radius','bend'))):
                raise ValueError('typed archive entity data required')
            entities.append(entity)
        for key in ('priority_at','link_start','link_end','camera_start','camera_end','retire_at'):
            if value[key] is not None and not number(value[key]):raise ValueError('typed archive event time required')
        for key in ('link_drone','camera_drone','retire_drone'):
            if value[key] is not None and type(value[key]) is not int:raise ValueError('typed archive aircraft required')
        if cls is not CaseSpec and (type(value['domain']) is not str or not value['domain']):
            raise ValueError('typed archive domain required')
        if cls is EventCleanSibling:
            if type(value['parent_case']) is not dict or value['parent_case'].get('schema')!=POPULATION_SCHEMA:
                raise ValueError('archive clean parent must be a population case')
            parent=_archived_identity_case(value['parent_case'])
            if (type(parent) is not PopulationCase or type(value['parent_sha256']) is not str
                    or len(value['parent_sha256'])!=64
                    or any(c not in '0123456789abcdef' for c in value['parent_sha256'])):
                raise ValueError('typed archive clean parent identity required')
            converted['parent_case']=parent.as_dict()
        converted.update(roi_polygon=tuple(tuple(p) for p in value['roi_polygon']),
                         wind_xy=tuple(value['wind_xy']),entities=tuple(entities))
        return cls(**converted)
    except (TypeError,KeyError,IndexError) as error:
        raise ValueError('malformed development archive identity') from error


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
        case = _archived_identity_case(value)
        try:
            executable=load_development_case(value)
        except (ValueError,TypeError,KeyError,IndexError,OverflowError) as error:
            admission={'status':'rejected_by_current_mission_validator','reason':str(error)}
        else:
            if executable.digest()!=case.digest():
                raise ValueError('archive identity normalization differs from executable decoder')
            admission={'status':'accepted_by_current_mission_validator','reason':None}
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
            'root_generation_seed': root_generation_seed,
            'identity_role':'archived_input_not_mission_qualification','current_mission_validation':admission}
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
