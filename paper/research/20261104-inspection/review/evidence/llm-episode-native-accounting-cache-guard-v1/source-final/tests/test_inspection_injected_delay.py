"""Wall-clock coordinator wait with an independent timer, no MLX/PyFlyt."""

import copy
import multiprocessing as mp
import time

import pytest

import eval.run_inspection_llm_development as runner


def timer(shared, ready):
    epoch = time.monotonic()
    ready.set()
    while time.monotonic()-epoch < 3:
        with shared.get_lock():
            shared[0] = time.monotonic()-epoch
        time.sleep(.01)


def test_two_second_injection_does_not_pause_the_independent_worker_or_reprice_service_time():
    context = mp.get_context("spawn")
    shared, ready = context.Array("d", [0.]*4), context.Event()
    child = context.Process(target=timer, args=(shared, ready))
    child.start()
    try:
        assert ready.wait(10)
        response = {"kind": "response", "status": "completed", "wall_seconds": 1.,
            "parent_end_to_end_wall_seconds": 1.1, "deadline_exceeded": False, "text": '{"commands":[]}'}
        original = copy.deepcopy(response)
        before, injected = runner.inject_reply_wait(response, 2, lambda: list(shared[:]))
        assert injected >= 2 and shared[0]-before[0] > 1.5
        assert response == original
    finally:
        if child.is_alive(): child.terminate()
        child.join(5)
        assert not child.is_alive()


@pytest.mark.parametrize("result,stamp,final", [
    ({"kind": "not_started"}, 100, False), ({"kind": "response", "status": "runtime_failure"}, 100, False),
    ({"kind": "response", "status": "completed"}, 600, False),
    ({"kind": "response", "status": "completed"}, 590, True),
])
def test_no_injected_wait_for_non_generation_or_after_episode(monkeypatch, result, stamp, final):
    def forbidden(_): raise AssertionError("no wait permitted")
    monkeypatch.setattr(runner.time, "sleep", forbidden)
    assert runner.inject_reply_wait(result, 20, lambda: [stamp, 0, 0, 80], final=final) == ([stamp, 0, 0, 80], 0.)
