"""Counter telemetry fixtures only: no model, physics, RNG or QoS mutation."""
from array import array
import gc
import json
from types import SimpleNamespace
import pytest

from eval.inspection_clock_profile import STAGES
from eval.inspection_clock_profile_rusage import RusageClockProfile,COUNTERS,RU_FIELDS
from tests.test_inspection_clock_profile import FakeClock


def test_counter_snapshots_exclude_sleep_and_preserve_exact_fault_switch_deltas(tmp_path):
    clock=FakeClock();calls=[]
    def usage(scope):
        calls.append(scope);clock.advance(.00001)
        return SimpleNamespace(**{key:float(len(calls)) for key in COUNTERS})
    profile=RusageClockProfile(1,clock=clock,rusage_reader=usage,qos_reader=lambda:{'fixture':True})
    try:
        profile.anchor(100.);profile.initial_publication_end();profile.begin(0.);profile.requested_sleep(.005)
        clock.advance(.005);profile.mark('wake');profile.rpc(0)
        for stage in STAGES[2:]:clock.advance(.001);profile.mark(stage)
        profile.lag(.01,False);profile.end(1/120);result=profile.save(tmp_path)
        assert len(calls)==3 and result['rusage_profile']['rows']==1
        values=array('d');values.frombytes((tmp_path/'clock-rusage.f64').read_bytes())
        row=dict(zip(RU_FIELDS,values));assert len(values)==len(RU_FIELDS)
        assert row['log_end_ru_majflt']-row['wake_ru_majflt']==1.
        assert row['wake_ru_nvcsw']-row['begin_ru_nvcsw']==1.
        assert 0<row['wake_wall_after']-row['wake_wall_before']<.00002
        assert profile.callback not in gc.callbacks
    finally:profile.stop()


def test_qos_query_failure_removes_installed_gc_callback():
    before=tuple(gc.callbacks)
    def fail():raise RuntimeError('qos probe fixture failure')
    with pytest.raises(RuntimeError,match='qos probe'):RusageClockProfile(1,qos_reader=fail)
    assert tuple(gc.callbacks)==before


@pytest.mark.parametrize('fault',[None,'settled','short_calls','wrong_case','wrong_variant'])
def test_adaptive_diagnostic_requires_its_exact_live_reserved_source_variant(tmp_path,fault):
    from eval.probe_inspection_adaptive_clock import admission,sha,ROLE,CASE,SAMPLER_SEED,ROOT,EXTRA_SOURCES
    (tmp_path/'planning').mkdir();(tmp_path/'.paper').mkdir()
    receipt=tmp_path/'model-receipt.json';receipt.write_text('{}')
    registry=tmp_path/'planning/local-model-registry.v1.json'
    registry.write_text(json.dumps({'models':{'M1':{'download_receipt':'model-receipt.json',
        'download_receipt_sha256':sha(receipt),'generation_profile':{'max_calls':32,'max_total_tokens':180000}}}}))
    case=tmp_path/CASE;case.parent.mkdir(parents=True);case.write_text('{}')
    purpose={'kind':'inspection_local_generation','phase':'development','diagnostic_role':ROLE,
        'model_registry_sha256':sha(registry),'model_receipt_sha256':sha(receipt),
        'case_file_sha256':sha(case),'sampler_seed':SAMPLER_SEED,
        'worker_variant_source_hashes':{str(ROOT/p):sha(ROOT/p) for p in EXTRA_SOURCES}}
    entry={'status':'reserved','purpose':None,'reserved':{'calls':32,'tokens':180000}}
    if fault=='settled':entry['status']='known'
    elif fault=='short_calls':entry['reserved']['calls']=31
    elif fault=='wrong_case':purpose['case_file_sha256']='0'*64
    elif fault=='wrong_variant':purpose['worker_variant_source_hashes']={}
    entry['purpose']=json.dumps(purpose)
    (tmp_path/'.paper/control.json').write_text(json.dumps({'frozen_plan':'synthetic-not-accepted-plan',
        'usage_ledger':{'test':entry}}))
    if fault:
        with pytest.raises(ValueError,match='live diagnostic reservation'):admission(tmp_path,'test')
    else:assert admission(tmp_path,'test')['reserved']=={'calls':32,'tokens':180000}
