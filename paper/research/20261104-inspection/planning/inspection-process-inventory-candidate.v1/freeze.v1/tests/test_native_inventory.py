"""Bounded process recovery tests; no simulation, model or scientific RNG."""
import ctypes
import importlib.util
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest

BASE=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
native=load('eval.inspection_darwin_inventory',BASE/'source/eval/inspection_darwin_inventory.py')
cleanup=load('staged_process_cleanup',BASE/'source/eval/inspection_process_cleanup.py')


def info(pid,parent=1,seconds=1700000000):
    value=native.ProcBSDInfo();value.pid=pid;value.ppid=parent;value.uid=os.getuid()
    value.status=3;value.pgid=pid;value.start_seconds=seconds;value.name=b'Python'
    return value


class FakeAPI:
    def __init__(self,values,children):self.values=values;self.links=children;self.reads=[];self.counts={}
    def info(self,pid):
        self.reads.append(pid);self.counts[pid]=self.counts.get(pid,0)+1
        value=self.values.get(pid)
        if callable(value):value=value(self.counts[pid])
        if isinstance(value,Exception):raise value
        return native.ProcBSDInfo.from_buffer_copy(bytes(value)) if value is not None else None
    def path(self,pid):return '/example/Python'
    def children(self,pid):return self.links.get(pid,[])


def owned(*pids):return {p:{'born':' '.join(time.ctime(1700000000).split())} for p in pids}


def test_exact_live_forest_without_inspecting_unrelated_processes():
    api=FakeAPI({10:info(10),11:info(11,10),12:info(12,11),99:info(99)}, {10:[11],11:[12]})
    result=native.descendant_inventory(owned(10),api)
    assert set(result)=={10,11,12} and 99 not in api.reads
    assert result[11]['parent']==10 and result[12]['parent']==11


def test_already_owned_reparented_descendant_and_its_children_remain_observed():
    api=FakeAPI({10:info(10),11:info(11,1),12:info(12,11)}, {11:[12]})
    assert set(native.descendant_inventory(owned(10,11),api))=={10,11,12}


def test_reused_root_pid_is_not_a_parent_authority():
    api=FakeAPI({10:info(10,seconds=1700000001),11:info(11,10)}, {10:[11]})
    result=native.descendant_inventory(owned(10),api)
    assert set(result)=={10} and result[10]['born']!=owned(10)[10]['born']


def test_confirmed_exited_root_does_not_hide_an_already_owned_child():
    api=FakeAPI({11:info(11,1),12:info(12,11)}, {11:[12]})
    assert set(native.descendant_inventory(owned(10,11),api))=={11,12}


@pytest.mark.parametrize('fault',['permission','child_reparented','parent_reused','cycle'])
def test_uncertain_native_forest_requires_full_legacy_fallback(fault):
    values={10:info(10),11:info(11,10)};links={10:[11]}
    if fault=='permission':values[11]=native.RecordUnavailable('fixture permission')
    elif fault=='child_reparented':values[11]=info(11,1)
    elif fault=='parent_reused':values[10]=lambda n:info(10,seconds=1700000000 if n<4 else 1700000001)
    else:links[11]=[10]
    with pytest.raises(native.NativeInventoryUnavailable):native.descendant_inventory(owned(10),FakeAPI(values,links))


def test_unreadable_descendant_falls_back_to_unchanged_full_inventory(monkeypatch):
    def fail(*args):raise native.RecordUnavailable('fixture permission')
    monkeypatch.setattr(native,'descendant_inventory',fail)
    sentinel={19:{'role':'full legacy snapshot includes unrelated/orphan data'}}
    monkeypatch.setattr(cleanup,'_inventory',lambda:sentinel)
    assert cleanup._observed_inventory(owned(10)) is sentinel
    assert cleanup.inventory_diagnostics()['hot_legacy']>=1


class Function:
    def __init__(self,fn):self.fn=fn
    def __call__(self,*args):return self.fn(*args)


def test_child_buffer_uses_count_and_bytes_without_quarter_table_truncation():
    class Lib:
        proc_pidinfo=Function(lambda *args:0)
        proc_pidpath=Function(lambda *args:0)
        def __init__(self):self.proc_listchildpids=Function(self.children)
        def children(self,pid,buffer,size):
            if buffer is None:return 100
            assert size>=100*ctypes.sizeof(ctypes.c_int)
            for i in range(100):ctypes.cast(buffer,ctypes.POINTER(ctypes.c_int))[i]=2000+i
            return 100
    assert native.Libproc(Lib()).children(10)==list(range(2000,2100))


def test_empty_child_list_is_valid_but_native_api_failure_is_not():
    class Lib:
        proc_pidinfo=Function(lambda *args:0);proc_pidpath=Function(lambda *args:0)
        proc_listchildpids=Function(lambda *args:0)
    assert native.Libproc(Lib()).children(10)==[]
    def denied(*args):ctypes.set_errno(1);return 0
    lib=Lib();lib.proc_listchildpids=Function(denied)
    with pytest.raises(native.NativeInventoryUnavailable):native.Libproc(lib).children(10)


@pytest.mark.skipif(sys.platform!='darwin',reason='actual libproc process forest check')
def test_actual_native_hot_observation_and_birth_checked_tree_cleanup():
    watch=cleanup.SubprocessTree()
    child_code="import subprocess,sys,time; sys.stdin.readline(); p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)"
    process=subprocess.Popen([sys.executable,'-u','-c',child_code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        watch.attach(process)
        assert set(watch.owned)=={process.pid}
        process.stdin.write('start\n');process.stdin.flush()
        assert select.select([process.stdout],[],[],5)[0]
        child=int(process.stdout.readline());watch.observe()
        assert child in watch.owned and cleanup.inventory_diagnostics()['hot_native']>=1
        legacy=cleanup._inventory();observed=native.descendant_inventory(watch.owned)
        for pid in (process.pid,child):
            assert all(observed[pid][k]==legacy[pid][k] for k in ('parent','group','uid','born'))
        result=watch.close()
        assert result['status']=='closed' and result['remaining_owned']==[]
        assert process.poll() is not None
    finally:
        watch.close();process.stdin.close();process.stdout.close()


@pytest.mark.skipif(sys.platform!='darwin',reason='actual direct-child recovery')
def test_failed_attach_still_recovers_and_stops_our_unbound_child(monkeypatch):
    watch=cleanup.SubprocessTree();original=cleanup._inventory
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    try:
        monkeypatch.setattr(cleanup,'_inventory',lambda:{})
        with pytest.raises(RuntimeError,match='birth'):watch.attach(process)
        monkeypatch.setattr(cleanup,'_inventory',original)
        result=watch.close()
        assert result['status']=='closed' and process.poll() is not None
    finally:
        monkeypatch.setattr(cleanup,'_inventory',original)
        watch.close()
