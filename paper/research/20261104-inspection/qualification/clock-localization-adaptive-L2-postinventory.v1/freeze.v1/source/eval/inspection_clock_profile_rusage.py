"""Opt-in engineering counters around sleep and work; no scheduler changes."""
from array import array
import ctypes
import hashlib
import json
import resource
import sys
import time

from eval.inspection_clock_profile import ClockProfile

COUNTERS=('ru_utime','ru_stime','ru_minflt','ru_majflt','ru_nswap','ru_nvcsw','ru_nivcsw','ru_maxrss')
SNAPSHOTS=('begin','wake','log_end')
RU_FIELDS=tuple(stage+'_'+field for stage in SNAPSHOTS for field in ('wall_before','wall_after',*COUNTERS))


def requested_qos():
    if sys.platform!='darwin':
        return {'status':'not_darwin','scope':'requested QoS not queried'}
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    lib.pthread_self.argtypes=[];lib.pthread_self.restype=ctypes.c_void_p
    lib.pthread_get_qos_class_np.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_int)]
    lib.pthread_get_qos_class_np.restype=ctypes.c_int
    qos,relative=ctypes.c_uint(),ctypes.c_int()
    code=lib.pthread_get_qos_class_np(lib.pthread_self(),ctypes.byref(qos),ctypes.byref(relative))
    return {'return_code':code,'requested_qos_class':qos.value if code==0 else None,
            'requested_relative_priority':relative.value if code==0 else None,
            'scope':'requested current physics-main-thread QoS, not effective priority or a scheduling guarantee',
            'no_scheduling_settings_changed':True}


class RusageClockProfile(ClockProfile):
    def __init__(self,duration,*,clock=time,rusage_reader=resource.getrusage,qos_reader=requested_qos):
        super().__init__(duration,clock=clock)
        try:
            self.rusage_reader=rusage_reader
            self.rusage_rows=array('d',[float('nan')])*(self.capacity*len(RU_FIELDS))
            self.qos=qos_reader()  # Before the unchanged worker epoch.
        except BaseException:
            self.stop();raise

    def mark(self,stage):
        super().mark(stage)
        if stage not in SNAPSHOTS:return
        before=self.clock.monotonic();usage=self.rusage_reader(resource.RUSAGE_SELF);after=self.clock.monotonic()
        base=self.index*len(RU_FIELDS)+SNAPSHOTS.index(stage)*(len(COUNTERS)+2)
        self.rusage_rows[base]=before;self.rusage_rows[base+1]=after
        for index,key in enumerate(COUNTERS):self.rusage_rows[base+2+index]=getattr(usage,key)

    def save(self,directory):
        result=super().save(directory)
        from pathlib import Path
        directory=Path(directory);binary=directory/'clock-rusage.f64'
        with binary.open('xb') as stream:self.rusage_rows[:self.completed*len(RU_FIELDS)].tofile(stream)
        metadata={'schema':'inspection-engineering-clock-rusage-v1','role':'diagnostic counter snapshots, not qualification',
            'rows':self.completed,'fields':RU_FIELDS,'dtype':'native_float64','byte_order':sys.byteorder,
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'binary_bytes':binary.stat().st_size,
            'base_clock_profile_sha256':hashlib.sha256((directory/'clock-profile.json').read_bytes()).hexdigest(),
            'requested_qos_before_epoch':self.qos,
            'scope':'RUSAGE_SELF includes all physics-process threads; counter definitions/availability are OS-specific; maxrss is a cumulative high-water mark',
            'limits':['No scheduler, GC, epoch or lag-threshold changes.','Counter calls and allocations add finite overhead.',
                      'Snapshots bracket getrusage, not individual kernel events; cannot uniquely assign all waits.',
                      'Unavailable or unchanging OS counters do not prove absence of an event.']}
        path=directory/'clock-rusage-profile.json'
        with path.open('x') as stream:json.dump(metadata,stream,indent=2,allow_nan=False);stream.write('\n')
        return {**result,'rusage_profile':{'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':self.completed}}
