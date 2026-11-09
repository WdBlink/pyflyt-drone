"""Read the observed process forest without spawning full-table ps in wait.

Uses the SDK libproc ABI. Delivers no signals, changes no priority, caches no
process identity and changes no scan cadence. Any uncertain read raises for
the caller's original full ps fallback. This is NOT a global orphan inventory;
the caller keeps its original full baseline and shutdown/orphan scans.
Birth strings retain the original second-resolution representation.
"""
import ctypes
import errno
import os
import sys
import time


class NativeInventoryUnavailable(RuntimeError):
    pass


class RecordUnavailable(NativeInventoryUnavailable):
    """One PID needs the original ps reader, rather than being omitted."""


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
        self.lib.proc_listchildpids.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
        self.lib.proc_listchildpids.restype=ctypes.c_int
        self.lib.proc_pidinfo.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_int]
        self.lib.proc_pidinfo.restype=ctypes.c_int
        self.lib.proc_pidpath.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]
        self.lib.proc_pidpath.restype=ctypes.c_int

    def children(self,pid):
        # proc_listchildpids returns PID COUNTS, buffer size is BYTES.
        ctypes.set_errno(0);count=self.lib.proc_listchildpids(pid,None,0)
        if count==0 and ctypes.get_errno()==0:return []
        if count<=0:raise NativeInventoryUnavailable('child process sizing failed')
        capacity=max(8,count+count//8+8)
        for _ in range(4):
            if capacity>131072:raise NativeInventoryUnavailable('bounded PID buffer exceeded')
            buffer=(ctypes.c_int*capacity)();ctypes.set_errno(0)
            count=self.lib.proc_listchildpids(pid,buffer,ctypes.sizeof(buffer))
            if count==0 and ctypes.get_errno()==0:return []
            if count<0 or count>capacity or ctypes.get_errno()!=0:
                raise NativeInventoryUnavailable('child process enumeration failed')
            if count==capacity:capacity*=2;continue
            values=[int(pid) for pid in buffer[:count] if pid>0]
            if len(values)!=len(set(values)) or pid in values:
                raise NativeInventoryUnavailable('invalid child PID snapshot')
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
        if size==0:raise RecordUnavailable('BSD record not readable')
        if size!=ctypes.sizeof(value) or value.pid!=pid:
            raise NativeInventoryUnavailable('incomplete BSD process record')
        return value

    def path(self,pid):
        buffer=ctypes.create_string_buffer(4096);ctypes.set_errno(0)
        size=self.lib.proc_pidpath(pid,buffer,ctypes.sizeof(buffer))
        if size==0 and (ctypes.get_errno()==errno.ESRCH or self.gone(pid)):return None
        if size==0:raise RecordUnavailable('executable path not readable')
        if not 0<size<ctypes.sizeof(buffer) or not buffer.value:
            raise NativeInventoryUnavailable('executable path unavailable or truncated')
        return buffer.value.decode('utf-8','replace')


_api=None


def descendant_inventory(owned,api=None):
    global _api
    if api is None:
        if sys.platform!='darwin':raise NativeInventoryUnavailable('Darwin backend only')
        if _api is None:_api=Libproc()
        api=_api
    rows={};pending=list(owned);parents={};own_uid=os.getuid()
    identity=lambda info:(info.start_seconds,info.start_microseconds,info.ppid,info.uid)
    while pending:
        pid=pending.pop()
        if pid in rows:continue
        info=api.info(pid)
        if info is None:continue
        if info.uid!=own_uid:
            raise RecordUnavailable('foreign observed process requires full legacy inventory')
        status={1:'I',2:'R',3:'S',4:'T',5:'Z'}.get(info.status)
        if status is None:raise NativeInventoryUnavailable('unknown BSD process status')
        command=(bytes(info.name) or bytes(info.comm)).decode('utf-8','replace')
        if info.uid==own_uid and status!='Z':
            command=api.path(pid)
            if command is None:continue
            check=api.info(pid)
            if check is None or identity(check)!=identity(info):
                raise NativeInventoryUnavailable('process identity changed during path read')
        if pid in parents:
            parent,expected=parents[pid];check=api.info(parent)
            if info.ppid!=parent or check is None or identity(check)!=expected:
                raise NativeInventoryUnavailable('parent changed before child binding')
        rows[pid]={'parent':int(info.ppid),'group':int(info.pgid),'uid':int(info.uid),
            'status':status,'born':' '.join(time.ctime(info.start_seconds).split()),'command':command}
        if pid in owned and rows[pid]['born']!=owned[pid]['born']:continue
        if status=='Z':continue
        children=api.children(pid);check=api.info(pid)
        if check is None or identity(check)!=identity(info):
            raise NativeInventoryUnavailable('parent changed during child enumeration')
        for child in children:
            if child in rows and rows[child]['parent']!=pid:
                raise NativeInventoryUnavailable('already read child has a different parent')
            if child in parents and parents[child]!=(pid,identity(info)):
                raise NativeInventoryUnavailable('contradictory child parent snapshot')
            parents[child]=(pid,identity(info));pending.append(child)
    return rows
