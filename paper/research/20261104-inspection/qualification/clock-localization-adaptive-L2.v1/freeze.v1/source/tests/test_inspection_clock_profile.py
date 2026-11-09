"""Numeric telemetry tests only; no model, physics or system clock pacing."""
from array import array
import gc
import json
import math

import pytest

from eval.inspection_clock_profile import ClockProfile, FIELDS, STAGES, WIDTH


class FakeClock:
    def __init__(self): self.wall, self.thread, self.process = 100., 2., 3.
    def monotonic(self): return self.wall
    def thread_time(self): return self.thread
    def process_time(self): return self.process
    def time_ns(self): return 1234567890000000000
    def advance(self, wall, cpu=0.):
        self.wall += wall; self.thread += cpu; self.process += cpu


def test_numeric_capture_preserves_epoch_lag_and_cpu_wait_distinction(tmp_path):
    clock = FakeClock(); profile = ClockProfile(1, clock=clock)
    before = gc.isenabled(), gc.get_threshold(), gc.get_freeze_count()
    try:
        profile.anchor(99.); profile.initial_publication_end()
        for index, lag in enumerate((.01, .3, -.001)):
            profile.begin(index/120); profile.requested_sleep(.005)
            clock.advance(.02, .0001); profile.mark('wake')
            profile.rpc(2)
            for stage in STAGES[2:]: clock.advance(.001, .0005); profile.mark(stage)
            profile.lag(lag, index == 0); profile.end((index+1)/120)
        result = profile.save(tmp_path)
        assert result['rows'] == 3 and result['max_lag'] == .3 and result['ticks_over_0p1'] == 1
        assert result['anchor']['worker_monotonic_epoch'] == 99.
        data = array('d'); data.frombytes((tmp_path/'clock-ticks.f64').read_bytes())
        assert len(data) == 3*WIDTH
        wake, begin = FIELDS.index('wake_wall'), FIELDS.index('begin_wall')
        cpu_wake, cpu_begin = FIELDS.index('wake_thread_cpu'), FIELDS.index('begin_thread_cpu')
        assert data[wake]-data[begin] > .019
        assert data[cpu_wake]-data[cpu_begin] < .001
        assert (gc.isenabled(), gc.get_threshold(), gc.get_freeze_count()) == before
        assert profile.callback not in gc.callbacks
    finally: profile.stop()


def test_gc_events_are_bounded_and_have_thread_and_stage_identity(tmp_path):
    profile = ClockProfile(1, gc_capacity=1, clock=FakeClock())
    try:
        profile.anchor(100.); profile.initial_publication_end(); profile.begin(0.)
        profile._gc_event('start', {'generation': 2})
        profile._gc_event('stop', {'generation': 2, 'collected': 1, 'uncollectable': 0})
        profile.lag(.0, False)
        for stage in STAGES[1:]: profile.mark(stage)
        profile.end(1/120)
        value = profile.save(tmp_path)
        assert value['gc_rows'] == 1 and value['gc_overflow'] >= 1
        assert value['main_thread_id'] > 0
    finally: profile.stop()


@pytest.mark.parametrize('duration', [True, 0, -1, 601, math.inf, math.nan])
def test_profile_bound_rejects_invalid_duration(duration):
    with pytest.raises(ValueError): ClockProfile(duration)


def test_incomplete_initial_publication_does_not_mask_failure(tmp_path):
    profile = ClockProfile(1, clock=FakeClock())
    try:
        profile.anchor(100.)
        value = profile.save(tmp_path)
        assert value['rows'] == 0 and value['initial_publication_wall_seconds'] is None
        assert value['max_lag'] is None
    finally: profile.stop()


def test_fixed_model_load_outputs_never_become_control_commands():
    import threading
    from eval.probe_inspection_clock_load import model_workload
    class Progress:
        def get_lock(self): return threading.Lock()
        def __getitem__(self, key): return 10.
    class Client:
        def __init__(self): self.submitted = []
        def submit(self, messages, seed): self.submitted.append((messages, seed))
        def poll(self):
            return {'kind': 'response', 'text': '{"commands":[{"name":"unsafe","args":{}}]}'}
    client = Client(); ready, done = threading.Event(), threading.Event(); ready.set()
    records, errors = [], []
    rows = [{'target_sim_time': 0, 'seed': 1, 'messages': [], 'messages_sha256': 'fixture'}]
    model_workload(client, rows, Progress(), ready, done, records, errors)
    assert not errors and client.submitted == [([], 1)]
    assert records[0]['response']['text'].startswith('{"commands"')
    assert records[0]['role'] == 'output retained as load evidence, never executable commands'


def test_fixed_model_load_does_not_submit_after_physics_end():
    import threading
    from eval.probe_inspection_clock_load import model_workload
    class Progress:
        def get_lock(self): return threading.Lock()
        def __getitem__(self, key): return 600.
    class Client:
        def submit(self, *args): raise AssertionError('no new request after physics ends')
    ready, done = threading.Event(), threading.Event(); ready.set(); done.set()
    records, errors = [], []
    model_workload(Client(), [{'target_sim_time': 0}], Progress(), ready, done, records, errors)
    assert not errors and records == [{'kind': 'not_submitted_after_physics_end', 'ordinal': 1}]


@pytest.mark.parametrize('failure', ['submit', 'poll'])
def test_fixed_load_retains_dispatch_anchor_when_transport_fails(failure):
    import threading
    from eval.probe_inspection_clock_load import model_workload
    class Progress:
        def get_lock(self): return threading.Lock()
        def __getitem__(self, key): return 10.
    class Client:
        def submit(self, *args):
            if failure == 'submit': raise RuntimeError('pipe outcome unknown')
        def poll(self): raise RuntimeError('response outcome unknown')
    ready, done = threading.Event(), threading.Event(); ready.set()
    records, errors = [], []
    model_workload(Client(), [{'target_sim_time': 0, 'seed': 12,
        'messages': [], 'messages_sha256': 'fixture'}], Progress(), ready, done, records, errors)
    assert len(errors) == 1 and len(records) == 1
    record = records[0]
    assert record['seed'] == 12 and record['submit']['monotonic'] > 0
    assert record['dispatch_status'] == 'dispatch_or_response_unknown'
    assert 'response' not in record and 'error' in record
