"""Real pipe/process faults using a fake service, never a language model."""

import json
import sys
import time

import pytest

from eval.inspection_model_client import LocalModelClient, ModelOperationUnknown


FAKE = r'''
import json,sys,time
print(json.dumps({'kind':'ready','reservation':'synthetic'}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    mode=request['messages'][0]['content']
    if mode=='exit':sys.exit(7)
    if mode=='hang':time.sleep(20)
    if mode=='bad':
        print('invalid protocol',flush=True)
        continue
    time.sleep(.03)
    print(json.dumps({'kind':'response','status':'completed','text':'{"commands":[]}',
                     'wall_seconds':.03,'deadline_exceeded':False,'usage':{'calls':1,'tokens':2}}),flush=True)
'''


def client(tmp_path):
    return LocalModelClient([sys.executable, "-u", "-c", FAKE], tmp_path/"fake-service", ready_timeout=2)


def submit(c, content):
    c.submit([{"role": "user", "content": content}], 123)


def test_poll_is_nonblocking_and_single_inflight_preserved(tmp_path):
    c = client(tmp_path)
    try:
        submit(c, "normal")
        assert c.poll() is None
        with pytest.raises(ModelOperationUnknown, match="prior request"):
            submit(c, "another")
        turns, result, deadline = 0, None, time.monotonic()+2
        while result is None and time.monotonic() < deadline:
            result = c.poll()
            turns += 1
            time.sleep(.001)
        assert result["submitted_ordinal"] == 1 and result["usage"]["calls"] == 1
        assert result["parent_end_to_end_wall_seconds"] >= .02 and turns > 1
        assert c.finish() is None
    finally:
        if c.process.poll() is None:
            c.stop_unknown()


def test_close_retains_last_inflight_response_instead_of_dropping_it(tmp_path):
    c = client(tmp_path)
    submit(c, "normal")
    result = c.finish()
    assert result["kind"] == "response" and result["submitted_ordinal"] == 1
    assert c.inflight is None and c.submitted == 1


@pytest.mark.parametrize("mode", ["exit", "bad"])
def test_broken_child_or_stdout_is_unknown_without_retry(tmp_path, mode):
    c = client(tmp_path)
    try:
        submit(c, mode)
        deadline = time.monotonic()+2
        with pytest.raises(ModelOperationUnknown):
            while time.monotonic() < deadline:
                c.poll()
                time.sleep(.001)
        assert c.submitted == 1 and c.inflight is not None
        with pytest.raises(ModelOperationUnknown):
            submit(c, "normal")
    finally:
        c.stop_unknown()


def test_hard_stop_does_not_claim_timeout_as_completed_or_zero_usage(tmp_path):
    import subprocess
    c = client(tmp_path)
    submit(c, "hang")
    with pytest.raises(subprocess.TimeoutExpired):
        c.finish(timeout=.05)
    assert c.process.poll() is not None and c.submitted == 1 and c.inflight is not None
    assert json.loads((c.directory/"stdout.log").read_text().splitlines()[0])["kind"] == "ready"


def test_large_stdin_to_unread_child_is_async_and_watchdog_keeps_unknown(tmp_path):
    script = "import json,time;print(json.dumps({'kind':'ready'}),flush=True);time.sleep(20)"
    c = LocalModelClient([sys.executable, "-u", "-c", script], tmp_path/"blocked-input",
        ready_timeout=2, request_watchdog_seconds=.08)
    started = time.monotonic()
    submit(c, "x"*70000)
    assert time.monotonic()-started < .05
    with pytest.raises(ModelOperationUnknown, match="watchdog"):
        while True:
            c.poll()
            time.sleep(.005)
    assert c.submitted == 1 and c.inflight is not None
    assert c.process.poll() is not None and not c.writer.is_alive()
