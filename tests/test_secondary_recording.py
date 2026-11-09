"""Actual provided SDK with literal secondary Fake callbacks, no flight/RNG."""
import importlib.util
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent;STAGE=HERE.parent/'eval';PROJECT=HERE.parent
spec=importlib.util.spec_from_file_location('secondary_recording_helpers',HERE/'test_formal_secondary.py')
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
module=helper.load('secondary_recording_candidate',STAGE/'inspection_secondary_recording.py')
SDK=PROJECT/'paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1'
sys.path.insert(0,str(SDK))
from lib.experiments import ExperimentRecorder
from lib.production import json_bytes,write_once


def fixture(track):
    slots=[]
    for block,case in enumerate(helper.intervention.SECONDARY_ROOTS):
        for repeat in range(3):
            for arm in helper.module.ARMS[track]:
                slots.append(helper.module.SecondarySlot(len(slots),block,case,
                    'OOD' if '-OOD-' in case else 'ID',3,'B4-M1',repeat,7000+repeat,repeat,track,arm))
    plan=SimpleNamespace(track=track,slots=tuple(slots),manifest=SimpleNamespace(path='literal-plan.json',sha256='a'*64))
    context={'experiment_id':'synthetic-secondary-'+track,'claim_id':'software-adapter-only',
        'hypothesis':'SYNTHETIC software preflight, no measured method outcome',
        'independent_variable':'fixture arms','dependent_variable':'Fake Y',
        'control_variables':['no model','no simulation'],
        'data_roles':[{'id':'literal-fixture','role':'development','used_for_selection':False}],
        'planned_at':datetime.now(timezone.utc).isoformat()}
    return plan,context


def write(root,name,value):return write_once(root,name,json_bytes(value))


@pytest.mark.parametrize('track,count,means',[('ablation',108,36),('latency',180,60)])
def test_secondary_sdk_preserves_full_grid_order_and_mean_provenance(tmp_path,track,count,means):
    plan,context=fixture(track);seen=[]
    def measure(slot):
        assert (tmp_path/f'starts/{slot.order_index:06d}.json').is_file()
        seen.append(slot);return slot.repeat_ordinal/4,{'role':'literal software Fake, no flight'}
    result=module.record_secondary(tmp_path,plan,measure,context=context,
                                   recorder_type=ExperimentRecorder,write_json=write)
    assert result['additional_actual_episode_denominator']==count and seen==list(plan.slots)
    outer=json.loads((tmp_path/'root-means/results.json').read_text())
    assert len(outer['runs'])==means and all(row['value']==.25 for row in outer['runs'])
    for row in outer['runs']:
        raw=json.loads((tmp_path/row['raw']).read_text())
        assert len(raw['evidence']['inner_attempts'])==3
        assert raw['evidence']['not_an_additional_episode'] is True


def test_secondary_infrastructure_failure_keeps_all_unstarted_denominators(tmp_path):
    plan,context=fixture('latency')
    def fail(slot):raise RuntimeError('intentional software failure')
    with pytest.raises(RuntimeError,match='intentional software'):
        module.record_secondary(tmp_path,plan,fail,context=context,
                                recorder_type=ExperimentRecorder,write_json=write)
    for group,count in [('episodes',180),('root-means',60)]:
        value=json.loads((tmp_path/group/'results.json').read_text())
        assert value['execution_status']=='failed' and len(value['runs'])==count
        assert all(row['value'] is None for row in value['runs'])
