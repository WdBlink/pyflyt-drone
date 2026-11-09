"""Literal secondary plans and content transforms only; no actual formal RNG."""
import importlib.util
import json
from pathlib import Path
import sys
from dataclasses import replace

import pytest

from eval import inspection_population as shared

HERE=Path(__file__).resolve().parent;STAGE=HERE.parent/'eval'
spec=importlib.util.spec_from_file_location('secondary_execution_helpers',HERE/'test_formal_dispatch.py')
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
load=helper.helper.load
intervention=load('eval.inspection_formal_intervention',STAGE/'inspection_formal_intervention.py')
module=load('eval.inspection_formal_secondary',STAGE/'inspection_formal_secondary.py')


def literal_parent():
    project=HERE.parent
    # A renamed in-memory software fixture, never a generated/registered test root.
    raw=json.loads((project/'paper/research/20261104-inspection/qualification/dev-population.v1/devext-ID-L3-00.json').read_text())
    raw.update(case_id='main-ID-L3-00',lineage_group='main-ID-L3-00',split='main')
    return shared.decode_formal_population_case(raw,'main',raw['case_id'])


def test_clean_transform_keeps_level_geometry_motion_noise_and_explicit_parent():
    parent=literal_parent();clean=intervention.formal_clean_sibling(parent)
    assert clean.level==parent.level=='L3' and clean.seed==parent.seed
    assert (clean.roi_polygon,clean.wind_xy,clean.entities)==(parent.roi_polygon,parent.wind_xy,parent.entities)
    assert all(getattr(clean,key) is None for key in shared.EVENT_FIELDS)
    assert clean.parent_sha256==parent.digest() and clean.digest()!=parent.digest()
    assert intervention.FormalCleanSibling.from_dict(clean.as_dict())==clean


@pytest.mark.parametrize('field,value',[('level','L2'),('seed',9),('priority_at',180.),
    ('wind_xy',(1.,1.)),('parent_sha256','0'*64),('case_id','main-ID-L3-06')])
def test_clean_content_cannot_change_more_than_the_declared_events(field,value):
    clean=intervention.formal_clean_sibling(literal_parent())
    with pytest.raises(ValueError):intervention.validate_formal_clean_sibling(replace(clean,**{field:value}))


@pytest.mark.parametrize('track,count,means',[('clean',216,96),('ablation',108,36),('latency',180,60)])
def test_secondary_plan_derives_full_fixed_grid_and_exact_parent_sampler(tmp_path,monkeypatch,track,count,means):
    f=helper.helper.ExecutionFixture(tmp_path);refs,execution=f.execution()
    primary=f.execution_proof(refs,execution)
    def forbidden(*args,**kwargs):raise AssertionError('no new RNG in secondary adapter')
    monkeypatch.setattr(helper.helper.recording.np.random,'PCG64',forbidden)
    ref=module.prepare_secondary_execution(f.run,f.project,f.acceptance,*refs,execution,track,
                                           verification=f.verification)
    with pytest.raises(ValueError):module.read_secondary_execution(f.run,f.project,f.acceptance,*refs,
        execution,ref,verification=f.verification)
    value=f.read(ref);f.bind(module.FINDINGS[track],[ref,value['episode_order']])
    plan=module.read_secondary_execution(f.run,f.project,f.acceptance,*refs,execution,ref,
                                         verification=f.verification)
    assert len(plan.slots)==count and len({row.case_id for row in plan.slots})==12
    assert value['root_mean_denominator']==means
    for row in plan.slots:
        parent=primary.slots[row.parent_order_index]
        assert (row.case_id,row.condition,row.repeat_ordinal,row.sampler_seed)==(
            parent.case_id,parent.condition,parent.repeat_ordinal,parent.sampler_seed)
    with pytest.raises(ValueError,match='already issued|preserved'):
        module.prepare_secondary_execution(f.run,f.project,f.acceptance,*refs,execution,track,
                                             verification=f.verification)


def test_secondary_reader_rejects_a_rehashed_changed_arm(tmp_path):
    f=helper.helper.ExecutionFixture(tmp_path);refs,execution=f.execution()
    ref=module.prepare_secondary_execution(f.run,f.project,f.acceptance,*refs,execution,'latency',
                                           verification=f.verification)
    value=f.read(ref);order=f.read(value['episode_order']);order['slots'][0]['arm']='delay-99'
    value['episode_order']=f.put(value['episode_order']['path'],order);ref=f.put(ref['path'],value)
    f.bind(module.FINDINGS['latency'],[ref,value['episode_order']])
    with pytest.raises(ValueError,match='order/arm/seed'):
        module.read_secondary_execution(f.run,f.project,f.acceptance,*refs,execution,ref,verification=f.verification)
