"""Bounded numeric clock telemetry for an explicitly instrumented engineering run.

No pacing, epoch, GC configuration, simulation or policy changes. Per-tick and
GC storage is preallocated; binary/JSON output is written after control ends.
"""
from array import array
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

STAGES = ('begin', 'wake', 'rpc_end', 'step_end', 'progress_end', 'stats_end', 'pub_end', 'log_end')
FIELDS = ('sim_before', 'sim_after', 'sleep_requested', 'lag', 'rpc_count', 'sensor_tick') + tuple(
    stage+'_'+clock for stage in STAGES for clock in ('wall', 'thread_cpu', 'process_cpu'))
WIDTH = len(FIELDS)
GC_FIELDS = ('event', 'generation', 'wall', 'thread_cpu', 'process_cpu', 'thread_id',
             'tick_index', 'stage_index', 'control_active', 'collected', 'uncollectable')


class ClockProfile:
    def __init__(self, duration, *, gc_capacity=16384, clock=time):
        if (type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 600
                or type(gc_capacity) is not int or gc_capacity <= 0):
            raise ValueError('bounded diagnostic duration/storage required')
        self.clock = clock
        self.capacity = math.ceil(duration*120)+2
        self.rows = array('d', [math.nan])*(self.capacity*WIDTH)
        self.gc_rows = array('d', [math.nan])*(gc_capacity*len(GC_FIELDS))
        self.gc_capacity, self.gc_count, self.gc_overflow = gc_capacity, 0, 0
        self.index, self.completed, self.stage = -1, 0, -1
        self.control_active = False
        self.main_thread = threading.get_ident()
        self.configuration = {'enabled': gc.isenabled(), 'threshold': gc.get_threshold(),
                              'count': gc.get_count(), 'frozen_count': gc.get_freeze_count()}
        self.callback = self._gc_event
        gc.callbacks.append(self.callback)

    def anchor(self, epoch):
        before = self.clock.monotonic()
        utc = self.clock.time_ns()
        after = self.clock.monotonic()
        self.epoch = epoch
        self.anchor_value = {'worker_monotonic_epoch': epoch, 'monotonic_before_utc': before,
            'utc_time_ns': utc, 'monotonic_after_utc': after,
            'scope': 'same worker clocks; bracket retained, no clock rebasing'}
        self.initial_publication_begin = self.clock.monotonic()
        self.initial_publication_finish = None

    def initial_publication_end(self):
        self.initial_publication_finish = self.clock.monotonic()

    def begin(self, simulated):
        self.index += 1
        if self.index >= self.capacity:
            raise RuntimeError('diagnostic tick storage exhausted')
        self.control_active = True
        self.rows[self.index*WIDTH] = simulated
        self.mark('begin')

    def mark(self, stage):
        self.stage = STAGES.index(stage)
        offset = self.index*WIDTH+6+self.stage*3
        self.rows[offset] = self.clock.monotonic()
        self.rows[offset+1] = self.clock.thread_time()
        self.rows[offset+2] = self.clock.process_time()

    def requested_sleep(self, value):
        self.rows[self.index*WIDTH+2] = max(0., value)

    def rpc(self, count):
        self.rows[self.index*WIDTH+4] = count

    def lag(self, value, sensor_tick):
        self.rows[self.index*WIDTH+3] = value
        self.rows[self.index*WIDTH+5] = int(sensor_tick)

    def end(self, simulated):
        self.rows[self.index*WIDTH+1] = simulated
        self.completed = self.index+1

    def _gc_event(self, event, info):
        if self.gc_count >= self.gc_capacity:
            self.gc_overflow += 1
            return
        base = self.gc_count*len(GC_FIELDS)
        self.gc_rows[base] = 1 if event == 'start' else 2
        self.gc_rows[base+1] = info.get('generation', -1)
        self.gc_rows[base+2] = self.clock.monotonic()
        self.gc_rows[base+3] = self.clock.thread_time()
        self.gc_rows[base+4] = self.clock.process_time()
        self.gc_rows[base+5] = threading.get_ident()
        self.gc_rows[base+6] = self.index
        self.gc_rows[base+7] = self.stage
        self.gc_rows[base+8] = int(self.control_active)
        self.gc_rows[base+9] = info.get('collected', -1)
        self.gc_rows[base+10] = info.get('uncollectable', -1)
        self.gc_count += 1

    def stop(self):
        self.control_active = False
        if self.callback in gc.callbacks:
            gc.callbacks.remove(self.callback)

    def save(self, directory):
        self.stop()
        directory = Path(directory)
        files = {}
        for name, values, count in [('clock-ticks.f64', self.rows, self.completed*WIDTH),
                                    ('clock-gc-events.f64', self.gc_rows, self.gc_count*len(GC_FIELDS))]:
            path = directory/name
            with path.open('xb') as stream:
                values[:count].tofile(stream)
            files[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        lags = [self.rows[index*WIDTH+3] for index in range(self.completed)]
        result = {'role': 'instrumented engineering clock localization; not a scored policy sample or G4 acceptance',
            'schema': 'inspection-worker-clock-profile-v1', 'rows': self.completed, 'capacity': self.capacity,
            'dtype': 'native_float64', 'byte_order': sys.byteorder, 'fields': FIELDS,
            'gc_rows': self.gc_count, 'gc_capacity': self.gc_capacity, 'gc_overflow': self.gc_overflow,
            'gc_fields': GC_FIELDS, 'gc_configuration_at_start': self.configuration,
            'recorder_changes_gc_configuration': False,
            'main_thread_id': self.main_thread, 'anchor': self.anchor_value,
            'initial_publication_wall_seconds': (self.initial_publication_finish-self.initial_publication_begin
                if self.initial_publication_finish is not None else None),
            'max_lag': max(lags) if lags else None,
            'p95_lag': sorted(lags)[int(.95*(len(lags)-1))] if lags else None,
            'ticks_over_0p1': sum(value > .1 for value in lags), 'files': files,
            'limits': ['Instrumentation has finite timing/heap overhead; cannot certify the old failures cause.',
                       'Thread CPU is the physics main thread; process CPU includes its feeder threads.',
                       'Stage flags describe intervals after a marker; GC records identify their actual executing thread.',
                       'No epoch reset, GC disabling/freezing, threshold relaxation or retry disposition change.']}
        path = directory/'clock-profile.json'
        with path.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        return result
