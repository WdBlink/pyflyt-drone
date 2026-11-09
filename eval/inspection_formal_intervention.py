"""Pure formal clean-sibling transformation after parent dataset admission.

This content helper grants no namespace, dispatch or new independent root.
Geometry, target motion, level and exogenous sensor seed remain the parent’s.
"""
from dataclasses import dataclass
import json

from eval.inspection_cases import CaseEntity
from eval.inspection_population import EVENT_FIELDS, PopulationCase, decode_formal_population_case
from eval.inspection_population import validate_formal_population_case

CLEAN_SCHEMA='inspection-formal-event-clean-sibling-v1'
SECONDARY_ROOTS=tuple(f'main-{domain}-L3-{index:02d}' for domain in ('ID','OOD') for index in range(6))


@dataclass(frozen=True)
class FormalCleanSibling(PopulationCase):
    parent_case: dict
    parent_sha256: str

    @classmethod
    def from_dict(cls,value):
        if type(value) is not dict or set(value)!=set(cls.__dataclass_fields__):
            raise ValueError('exact formal clean-sibling fields required')
        try:
            converted=dict(value)
            converted['roi_polygon']=tuple(tuple(point) for point in value['roi_polygon'])
            converted['wind_xy']=tuple(value['wind_xy'])
            converted['entities']=tuple(CaseEntity(**{**row,'origin_xy':tuple(row['origin_xy'])})
                                        for row in value['entities'])
            case=cls(**converted)
            validate_formal_clean_sibling(case)
        except (TypeError,KeyError,IndexError) as error:
            raise ValueError('malformed formal clean sibling') from error
        return case


def validate_formal_clean_sibling(case):
    if (type(case) is not FormalCleanSibling or case.schema!=CLEAN_SCHEMA
            or case.split!='main' or case.case_id not in SECONDARY_ROOTS):
        raise ValueError('formal clean intervention is limited to its twelve main L3 parents')
    parent=decode_formal_population_case(case.parent_case,'main',case.case_id)
    if parent.level!='L3' or type(case.parent_sha256) is not str or case.parent_sha256!=parent.digest():
        raise ValueError('clean sibling must bind the intact admitted L3 parent')
    expected={**parent.as_dict(),'schema':CLEAN_SCHEMA,**dict.fromkeys(EVENT_FIELDS)}
    actual={key:value for key,value in case.as_dict().items() if key not in ('parent_case','parent_sha256')}
    if json.dumps(expected,sort_keys=True,allow_nan=False)!=json.dumps(actual,sort_keys=True,allow_nan=False):
        raise ValueError('clean sibling may suppress only the four declared event interventions')
    return parent


def formal_clean_sibling(parent):
    validate_formal_population_case(parent,'main',parent.case_id)
    if parent.case_id not in SECONDARY_ROOTS or parent.level!='L3':
        raise ValueError('fixed twelve L3 parents required before clean intervention')
    case=FormalCleanSibling(**{**{key:getattr(parent,key) for key in PopulationCase.__dataclass_fields__},
        'schema':CLEAN_SCHEMA,**dict.fromkeys(EVENT_FIELDS)},
        parent_case=parent.as_dict(),parent_sha256=parent.digest())
    validate_formal_clean_sibling(case)
    return case


def check_clean_world_identity(world,parent_sha256):
    """Offline receipt audit, after flight; never passed into the actor context."""
    case=FormalCleanSibling.from_dict(world['private_case'])
    parent=validate_formal_clean_sibling(case)
    if (parent.digest()!=parent_sha256 or world['case_sha256']!=case.digest()
            or world['hidden_entities']!=case.as_dict()['entities']):
        raise ValueError('effective clean world differs from its admitted parent/transform')
    return case
