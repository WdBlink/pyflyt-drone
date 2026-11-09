"""Read Darwin's process table without spawning a full ps subprocess.

Uses the installed SDK's libproc ABI. No signalling, priority, cached process
identity or reduced scan cadence. Any incomplete/uncertain read raises so the
caller can use its original ps inventory. Birth identity retains the original
second-resolution local-time representation for backend interoperability.
"""
import ctypes
import errno
import os
import sys
import time


class NativeInventoryUnavailable(RuntimeError):
    pass


class ProcBSDInfo(ctypes.Structure):
    _fields_ = [(name,ctypes.c_uint32) for name in
        ('flags','status','xstatus','pid','ppid','uid','gid','ruid','rgid','svuid','svgid','reserved')]+[
        ('comm',ctypes.c_char*16),('name',ctypes.c_char*32)]+[
        (name,ctypes.c_uint32) for name in ('nfiles','pgid','pjobc','tdev','tpgid')]+[
        ('nice',ctypes.c_int32),('start_seconds',ctypes.c_uint64),('start_microseconds',ctypes.c_uint64)]


class Libproc:
    def __init__(self,lib=None):
        if ctypes.sizeof(ProcBSDInfo)!=136:
            raise NativeInventoryUnavailable('unsupported proc_bsdinfo ABI')
        try:self.lib=lib or ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
        except OSError as error:raise NativeInventoryUnavailable('libproc unavailable') from error
        self.lib.proc_listallpids.argtypes=[ctypes.c_void_p,ctypes.c_int]
        self.lib.proc_listallpids.restype=ctypes.c_int
        self.lib.proc_pidinfo.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_int]
        self.lib.proc_pidinfo.restype=ctypes.c_int
        self.lib.proc_pidpath.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]
        self.lib.proc_pidpath.restype=ctypes.c_int

    def pids(self):
        # proc_listallpids returns PID COUNTS, while buffer size is BYTES.
        # Apple xnu/libsyscall/wrappers/libproc/libproc.c confirms this wrapper.
        count=self.lib.proc_listallpids(None,0)
        if count<=0:raise NativeInventoryUnavailable('process sizing failed')
        capacity=max(128,count+count//8+64)
        for _ in range(4):
            if capacity>131072:raise NativeInventoryUnavailable('bounded PID buffer exceeded')
            buffer=(ctypes.c_int*capacity)();ctypes.set_errno(0)
            count=self.lib.proc_listallpids(buffer,ctypes.sizeof(buffer))
            if count<=0 or count>capacity:raise NativeInventoryUnavailable('process enumeration failed')
            if count==capacity:capacity*=2;continue
            values=[int(pid) for pid in buffer[:count] if pid>0]
            if len(values)!=len(set(values)):raise NativeInventoryUnavailable('duplicate PID snapshot')
            return values
        raise NativeInventoryUnavailable('process enumeration remained saturated')

    @staticmethod
    def gone(pid):
        try:os.kill(pid,0)
        except ProcessLookupError:return True
        except PermissionError:return False
        return False

    def info(self,pid):
        value=ProcBSDInfo();ctypes.set_errno(0)
        size=self.lib.proc_pidinfo(pid,3,0,ctypes.byref(value),ctypes.sizeof(value))
        if size==0 and (ctypes.get_errno()==errno.ESRCH or self.gone(pid)):return None
        if size!=ctypes.sizeof(value) or value.pid!=pid:
            raise NativeInventoryUnavailable('incomplete BSD process record')
        return value

    def path(self,pid):
        buffer=ctypes.create_string_buffer(4096);ctypes.set_errno(0)
        size=self.lib.proc_pidpath(pid,buffer,ctypes.sizeof(buffer))
        if size==0 and (ctypes.get_errno()==errno.ESRCH or self.gone(pid)):return None
        if not 0<size<ctypes.sizeof(buffer) or not buffer.value:
            raise NativeInventoryUnavailable('executable path unavailable or truncated')
        return buffer.value.decode('utf-8','replace')


_api=None


def inventory(api=None):
    global _api
    if api is None:
        if sys.platform!='darwin':raise NativeInventoryUnavailable('Darwin backend only')
        if _api is None:_api=Libproc()
        api=_api
    rows={};own_uid=os.getuid()
    for pid in api.pids():
        info=api.info(pid)
        if info is None:continue  # A confirmed exit, not a permission failure.
        status={1:'I',2:'R',3:'S',4:'T',5:'Z'}.get(info.status)
        if status is None:raise NativeInventoryUnavailable('unknown BSD process status')
        command=(bytes(info.name) or bytes(info.comm)).decode('utf-8','replace')
        if info.uid==own_uid and status!='Z':
            command=api.path(pid)
            if command is None:continue
            # Re-read identity after the path query: never combine two reused
            # PIDs' fields into one apparently owned process record.
            check=api.info(pid)
            if check is None:continue
            if (check.start_seconds,check.start_microseconds,check.ppid,check.uid)!=(
                    info.start_seconds,info.start_microseconds,info.ppid,info.uid):
                raise NativeInventoryUnavailable('process identity changed during read')
        rows[pid]={'parent':int(info.ppid),'group':int(info.pgid),'uid':int(info.uid),
            'status':status,'born':' '.join(time.ctime(info.start_seconds).split()),'command':command}
    if os.getpid() not in rows:
        raise NativeInventoryUnavailable('own process missing from inventory')
    return rows
