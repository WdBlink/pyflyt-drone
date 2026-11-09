"""Bounded physics-process GC deferral and current-thread Darwin scheduling.

The caller enters after warmup, before its unchanged epoch, and closes after
the <=600s control loop or an exception. This is soft real time, not a guarantee.
"""
import ctypes
import gc
import sys
import time


class DarwinThread:
    def __init__(self):
        if sys.platform!='darwin':raise RuntimeError('this qualification targets Darwin')
        self.lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        self.lib.pthread_self.argtypes=[];self.lib.pthread_self.restype=ctypes.c_void_p
        self.lib.pthread_mach_thread_np.argtypes=[ctypes.c_void_p];self.lib.pthread_mach_thread_np.restype=ctypes.c_uint
        self.lib.thread_policy_get.argtypes=[ctypes.c_uint,ctypes.c_uint,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_int)]
        self.lib.thread_policy_get.restype=ctypes.c_int
        self.lib.thread_policy_set.argtypes=[ctypes.c_uint,ctypes.c_uint,ctypes.POINTER(ctypes.c_int),ctypes.c_uint]
        self.lib.thread_policy_set.restype=ctypes.c_int
        class Timebase(ctypes.Structure):
            _fields_=[('numer',ctypes.c_uint),('denom',ctypes.c_uint)]
        self.lib.mach_timebase_info.argtypes=[ctypes.POINTER(Timebase)];self.lib.mach_timebase_info.restype=ctypes.c_int
        base=Timebase()
        if self.lib.mach_timebase_info(ctypes.byref(base)) or not base.numer or not base.denom:
            raise RuntimeError('valid native Mach timebase required')
        self.numer,self.denom=base.numer,base.denom
        self.thread=self.lib.pthread_mach_thread_np(self.lib.pthread_self()) # borrowed port

    def get(self,flavor,size):
        data=(ctypes.c_int*size)();count=ctypes.c_uint(size);default=ctypes.c_int(0)
        code=self.lib.thread_policy_get(self.thread,flavor,data,ctypes.byref(count),ctypes.byref(default))
        if code or count.value!=size:raise RuntimeError(f'thread_policy_get failed:{code}/{count.value}')
        return {'values':list(data),'default':bool(default.value)}

    def set(self,flavor,values):
        data=(ctypes.c_int*len(values))(*values)
        code=self.lib.thread_policy_set(self.thread,flavor,data,len(values))
        if code:raise RuntimeError(f'thread_policy_set failed:{code}')


class PhysicsRuntimeWindow:
    def __init__(self,*,native=None,collector=gc):
        self.native=native if native is not None else DarwinThread()
        self.gc=collector;self.closed=False;self.restore_needed=False
        self.audit={'schema':'inspection-physics-runtime-window-v1','errors':[],
            'scope':'current physics main thread only; GC deferral covers its process, not model/coordinator processes',
            'requested_ns':{'period':8_333_333,'computation':4_000_000,'constraint':8_000_000},
            'soft_realtime_not_a_guarantee':True}

    def start(self):
        if self.restore_needed or self.closed:raise RuntimeError('runtime windows are single-use')
        self.before=self.native.get(1,1);old_rt=self.native.get(2,4)
        if self.before!={'values':[1],'default':False} or old_rt['default'] is not True:
            raise RuntimeError('a fresh standard-scheduled worker thread is required')
        self.enabled=self.gc.isenabled();self.threshold=self.gc.get_threshold()
        ticks=[(self.audit['requested_ns'][key]*self.native.denom+self.native.numer//2)//self.native.numer
               for key in ('period','computation','constraint')]
        if not 0<ticks[1]<=ticks[2]<=ticks[0]<2**31:raise RuntimeError('invalid native time constraint')
        self.request=[*ticks,1]
        self.audit.update(before=self.before,prior_rt=old_rt,mach_timebase={'numer':self.native.numer,'denom':self.native.denom},
            requested_ticks=self.request,gc_enabled_before=self.enabled,gc_threshold_before=list(self.threshold))
        try:
            self.audit['pre_epoch_collected']=self.gc.collect()
            self.restore_needed=True
            self.native.set(2,self.request)
            self.audit['active_start']=self.native.get(2,4)
            if self.audit['active_start']!={'values':self.request,'default':False}:
                raise RuntimeError('requested real-time policy not active')
            self.gc.disable();self.audit['gc_enabled_at_epoch']=self.gc.isenabled()
            if self.gc.isenabled():raise RuntimeError('physics cyclic-GC deferral failed')
            self.started=time.monotonic()
        except BaseException:
            self.close();raise
        return self.audit

    def close(self):
        if self.closed:return self.audit
        self.closed=True
        if not self.restore_needed:return self.audit
        self.audit['window_wall_seconds']=time.monotonic()-self.started if hasattr(self,'started') else 0.
        try:
            self.audit['active_end']=self.native.get(2,4)
            if self.audit['active_end']!={'values':self.request,'default':False}:
                raise RuntimeError('physics thread was demoted or its policy changed')
        except Exception as error:self.audit['errors'].append(repr(error))
        try:
            self.native.set(1,self.before['values'])
            self.audit['restored']=self.native.get(1,1)
            self.audit['rt_after_restore']=self.native.get(2,4)
            if self.audit['restored']!=self.before or not self.audit['rt_after_restore']['default']:
                raise RuntimeError('original thread scheduling was not restored')
        except Exception as error:self.audit['errors'].append(repr(error))
        finally:
            try:
                self.audit['post_loop_collected']=self.gc.collect()
            except Exception as error:self.audit['errors'].append(repr(error))
            finally:
                self.gc.enable() if self.enabled else self.gc.disable()
                self.audit['gc_enabled_after']=self.gc.isenabled()
                if self.gc.get_threshold()!=self.threshold or self.gc.isenabled()!=self.enabled:
                    self.audit['errors'].append('original GC configuration was not restored')
        return self.audit
