"""Recount saved240Hz executor errors without running a new physical episode."""

import base64
import hashlib
import math
import zlib

import numpy as np


COLUMNS = ["t"]+[f"drone-{i}-{field}" for i in range(4)
    for field in ("xy_error_m","z_m","z_low_m","z_high_m")]


def recount_tracking(record,*,expected_xy_tube_m=None):
    trace = record["execution_tracking_trace"]
    if trace["encoding"]!="base64-zlib-little-endian-float64" or trace["columns"]!=COLUMNS:
        raise ValueError("unsupported physical error trace schema")
    count = record["execution_tracking_240hz"]["physics_samples"]
    # Legacy traces predate the field and were expressly recorded under6m.
    # A new contract cannot reinterpret their counters using a wider default.
    tube = trace.get("xy_tube_m",6.)
    if (type(tube) not in (int,float) or not math.isfinite(tube) or tube<=0
            or expected_xy_tube_m is not None and tube!=expected_xy_tube_m):
        raise ValueError("trace and qualification tracking envelope differ")
    if type(count) is not int or not 0<count<=600*240 or trace["rows"]!=count:
        raise ValueError("nonempty complete bounded trace required")
    raw = zlib.decompress(base64.b64decode(trace["data"],validate=True))
    if len(raw)!=count*len(COLUMNS)*8 or hashlib.sha256(raw).hexdigest()!=trace["raw_sha256"]:
        raise ValueError("physical trace bytes/hash mismatch")
    rows = np.frombuffer(raw,dtype='<f8').reshape(count,len(COLUMNS))
    if not np.isfinite(rows).all() or not np.allclose(rows[:,0],np.arange(1,count+1)/240,atol=1e-9,rtol=0):
        raise ValueError("invalid physical trace time or numeric coverage")
    xy,z,low,high = (rows[:,offset::4] for offset in (1,2,3,4))
    if np.any(xy<0) or np.any(low>high):
        raise ValueError("invalid physical trace domains")
    return {"physics_samples":count,"max_horizontal_error_m":float(xy.max()),
        "horizontal_tube_breach_steps":int(np.any(xy>tube,axis=1).sum()),
        "altitude_envelope_breach_steps":int(np.any((z<low)|(z>high),axis=1).sum())}
