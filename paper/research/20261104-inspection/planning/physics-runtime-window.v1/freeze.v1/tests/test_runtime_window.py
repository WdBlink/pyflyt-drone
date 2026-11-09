"""Policy/GC lifecycle fixtures plus one short self-thread native check."""
import importlib.util
from pathlib import Path
import sys

import pytest

SOURCE=Path(__file__).resolve().parents[1]/'source'
sys.path.insert(0,str(SOURCE))
from physics_runtime import DarwinThread, PhysicsRuntimeWindow


class Collector:
    def __init__(self,enabled=True):self.enabled=enabled;self.calls=[]
    def isenabled(self):return self.enabled
    def get_threshold(self):return (700,10,10)
    def collect(self):self.calls.append('collect');return 0
    def disable(self):self.calls.append('disable');self.enabled=False
    def enable(self):self.calls.append('enable');self.enabled=True


class Native:
    numer,denom=125,3
    def __init__(self):self.rt=False;self.values=None;self.fail=None
    def get(self,flavor,size):
        if flavor==1:return {'values':[0 if self.rt else 1],'default':False}
        return {'values':self.values if self.rt else [0,120000,240000,1],'default':not self.rt}
    def set(self,flavor,values):
        if self.fail==flavor:raise RuntimeError('intentional native error')
        self.rt=flavor==2;self.values=list(values)


@pytest.mark.parametrize('enabled',[True,False])
def test_success_restores_exact_gc_and_original_thread_mode(enabled):
    native,collector=Native(),Collector(enabled);window=PhysicsRuntimeWindow(native=native,collector=collector)
    audit=window.start()
    assert native.rt and not collector.enabled
    assert audit['requested_ticks']==[200000,96000,192000,1]
    with pytest.raises(RuntimeError,match='single-use'):window.start()
    assert native.rt
    window.close();calls=list(collector.calls)
    assert not native.rt and collector.enabled is enabled and not audit['errors']
    assert window.close()==audit and collector.calls==calls


def test_start_failure_restores_thread_and_gc_without_entering_window():
    native,collector=Native(),Collector();native.fail=2
    window=PhysicsRuntimeWindow(native=native,collector=collector)
    with pytest.raises(RuntimeError,match='intentional'):window.start()
    assert not native.rt and collector.enabled and window.closed


def test_demoted_thread_is_reported_and_all_other_state_restored():
    native,collector=Native(),Collector();window=PhysicsRuntimeWindow(native=native,collector=collector)
    window.start();native.rt=False;window.close()
    assert window.audit['errors'] and collector.enabled and not native.rt


def test_restore_failure_remains_visible_and_gc_is_restored():
    native,collector=Native(),Collector();window=PhysicsRuntimeWindow(native=native,collector=collector)
    window.start();native.fail=1;window.close()
    assert window.audit['errors'] and collector.enabled


@pytest.mark.skipif(sys.platform!='darwin',reason='fixed qualification host is Darwin')
def test_short_native_self_thread_round_trip():
    collector=Collector();native=DarwinThread()
    window=PhysicsRuntimeWindow(native=native,collector=collector)
    try:
        window.start()
        assert native.get(2,4)=={'values':window.request,'default':False}
    finally:window.close()
    assert native.get(1,1)=={'values':[1],'default':False}
    assert native.get(2,4)['default'] and not window.audit['errors']
