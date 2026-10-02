"""A blocked callback cannot block rendering; stale/refused work cannot act."""

import threading
import time

import pytest

from cadence.experimental.equilibrium.runtime import LiveController, slew


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def completed(controller, count=1):
    until = time.monotonic() + 2.0
    while time.monotonic() < until:
        result = controller.inspect()
        if result["completed"] >= count:
            return result
        time.sleep(0.001)
    pytest.fail(f"callback did not finish: {controller.inspect()}")


def test_blocked_callback_has_one_serial_owner_and_latest_pending_wins():
    entered, release, submitted = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    seen, owners, commands = [], [], []

    def callback(observation):
        seen.append(observation)
        owners.append(threading.get_ident())
        if observation == 1:
            entered.set()
            assert release.wait(2.0)
        return {"qualified": True, "command": [observation]}

    controller = LiveController(callback, fallback=[0], max_age=10)

    def render_loop():
        controller.submit(2)
        controller.submit(3)
        commands.append(controller.read())
        submitted.set()

    renderer = threading.Thread(target=render_loop)
    try:
        assert controller.submit(1) == 1
        assert entered.wait(1.0)
        renderer.start()
        assert submitted.wait(1.0), "submit/read waited for callback"
        assert commands == [(0.0,)]
        assert controller.inspect()["dropped"] == 1
        release.set()
        metrics = completed(controller, 2)
        assert seen == [1, 3]
        assert len(set(owners)) == 1
        assert owners[0] != threading.get_ident()
        assert controller.read() == (3.0,)
        assert (metrics["submitted"], metrics["started"], metrics["completed"]) == (
            3,
            2,
            2,
        )
        assert metrics["last_result_id"] == 3
    finally:
        release.set()
        renderer.join(2.0)
        assert controller.close()


def test_age_begins_at_submission_not_completion():
    clock, entered, release = Clock(), threading.Event(), threading.Event()

    def callback(observation):
        entered.set()
        assert release.wait(2.0)
        return {"qualified": True, "command": observation}

    controller = LiveController(callback, fallback=[0], max_age=0.5, clock=clock)
    try:
        assert controller.read() == (0.0,)
        assert controller.inspect()["status"] == "waiting"
        controller.submit([7])
        assert entered.wait(1.0)
        clock.value = 0.75
        release.set()
        metrics = completed(controller)
        assert metrics["qualified"] == 1
        assert metrics["status"] == "stale"
        assert metrics["result_age"] == 0.75
        assert metrics["last_solve_seconds"] == 0.75
        assert metrics["last_latency_seconds"] == 0.75
        assert controller.read() == (0.0,)
    finally:
        release.set()
        assert controller.close()


def test_fresh_previous_command_remains_while_new_work_runs_then_expires():
    clock, entered, release = Clock(), threading.Event(), threading.Event()

    def callback(observation):
        if observation == 2:
            entered.set()
            assert release.wait(2.0)
        return {"qualified": True, "command": [observation]}

    controller = LiveController(callback, fallback=[0], max_age=1, clock=clock)
    try:
        controller.submit(1)
        completed(controller)
        clock.value = 0.5
        controller.submit(2)
        assert entered.wait(1.0)
        assert controller.read() == (1.0,)
        clock.value = 1.0
        assert controller.read() == (1.0,)
        clock.value = 1.1
        assert controller.read() == (0.0,)
        release.set()
        completed(controller, 2)
        assert controller.read() == (2.0,)
    finally:
        release.set()
        assert controller.close()


@pytest.mark.parametrize(
    ("result", "status"),
    [
        ({"qualified": False}, "refused"),
        ({"qualified": False, "command": [float("nan")]}, "refused"),
        ({"accepted": True, "command": [1]}, "error"),
        ({"qualified": 1, "command": [1]}, "error"),
        ({"qualified": True}, "error"),
        ({"qualified": True, "command": [1, 2]}, "error"),
        ({"qualified": True, "command": [float("inf")]}, "error"),
        ({"qualified": True, "command": [True]}, "error"),
        ({"qualified": True, "command": "1"}, "error"),
        ({"qualified": True, "command": []}, "error"),
        (None, "error"),
        (RuntimeError("callback failed"), "error"),
    ],
)
def test_refusal_or_error_replaces_prior_command_and_worker_recovers(result, status):
    def callback(observation):
        if observation == "bad":
            if isinstance(result, Exception):
                raise result
            return result
        return {"qualified": True, "command": [observation]}

    controller = LiveController(callback, fallback=[-1], clock=Clock())
    try:
        controller.submit(1)
        completed(controller)
        assert controller.read() == (1.0,)
        controller.submit("bad")
        metrics = completed(controller, 2)
        assert metrics["status"] == status
        assert metrics["errors" if status == "error" else "refused"] == 1
        assert (metrics["last_error"] is not None) == (status == "error")
        assert controller.read() == (-1.0,)
        controller.submit(2)
        metrics = completed(controller, 3)
        assert controller.read() == (2.0,)
        assert metrics["last_error"] is None
        assert metrics["qualified"] == 2
    finally:
        assert controller.close()


def test_input_and_output_are_owned_copies():
    entered, release = threading.Event(), threading.Event()
    source = {"sensor": [1, {"inner": (2, None, True, "sample")}]}
    command = [3]
    seen = []

    def callback(observation):
        entered.set()
        assert release.wait(2.0)
        seen.append(observation)
        return {"qualified": True, "command": command}

    fallback = [0]
    controller = LiveController(callback, fallback=fallback, clock=Clock())
    try:
        fallback[0] = 99
        assert controller.read() == (0.0,)
        controller.submit(source)
        assert entered.wait(1.0)
        source["sensor"][1]["inner"] = "mutated"
        source["sensor"].append(9)
        release.set()
        completed(controller)
        assert seen == [{"sensor": [1, {"inner": (2, None, True, "sample")}]}]
        command[0] = 99
        assert controller.read() == (3.0,)
        metrics = controller.inspect()
        metrics["submitted"] = 99
        assert controller.inspect()["submitted"] == 1
    finally:
        release.set()
        assert controller.close()


@pytest.mark.parametrize(
    "observation", [float("nan"), {"x": float("inf")}, {1: 2}, {1, 2}, object()]
)
def test_invalid_observation_is_refused_before_enqueue(observation):
    controller = LiveController(lambda _: None, fallback=[0])
    try:
        with pytest.raises(ValueError):
            controller.submit(observation)
        assert controller.inspect()["submitted"] == 0
    finally:
        assert controller.close()


def test_cyclic_observation_is_refused_without_consuming_submission_id():
    cyclic = []
    cyclic.append(cyclic)
    controller = LiveController(lambda _: {"qualified": False}, fallback=[0])
    try:
        with pytest.raises(ValueError, match="cycles"):
            controller.submit(cyclic)
        assert controller.submit(None) == 1
    finally:
        assert controller.close()


def test_close_drops_pending_and_never_publishes_inflight_command():
    entered, release = threading.Event(), threading.Event()
    seen = []

    def callback(observation):
        seen.append(observation)
        entered.set()
        assert release.wait(2.0)
        return {"qualified": True, "command": [9]}

    controller = LiveController(callback, fallback=[0])
    try:
        controller.submit(1)
        assert entered.wait(1.0)
        controller.submit(2)
        assert not controller.close(timeout=0)
        assert controller.read() == (0.0,)
        assert controller.inspect()["status"] == "closed"
        assert controller.inspect()["dropped"] == 1
        with pytest.raises(ValueError, match="closed"):
            controller.submit(3)
        assert not controller.close(wait=False)
        release.set()
        assert controller.close(timeout=2)
        assert seen == [1]
        assert controller.read() == (0.0,)
        assert controller.inspect()["completed"] == 1
        assert controller.close()
    finally:
        release.set()
        assert controller.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fallback": []},
        {"fallback": [True]},
        {"fallback": [float("nan")]},
        {"fallback": "x"},
        {"max_age": 0},
        {"max_age": -1},
        {"max_age": float("inf")},
        {"max_age": True},
        {"clock": None},
        {"clock": lambda: float("nan")},
    ],
)
def test_invalid_configuration_fails_before_thread_creation(kwargs):
    with pytest.raises(ValueError):
        LiveController(lambda _: None, **{"fallback": [0], **kwargs})


def test_invalid_callback_and_close_arguments():
    with pytest.raises(ValueError, match="callable"):
        LiveController(None, fallback=[0])
    controller = LiveController(lambda _: None, fallback=[0])
    try:
        for kwargs in ({"wait": 1}, {"timeout": -1}, {"timeout": float("inf")}):
            with pytest.raises(ValueError):
                controller.close(**kwargs)
        assert not controller.inspect()["closed"]
    finally:
        assert controller.close()


def test_clock_must_be_nondecreasing():
    clock = Clock()
    controller = LiveController(lambda _: {"qualified": False}, fallback=[0], clock=clock)
    try:
        clock.value = -1
        with pytest.raises(ValueError, match="nondecreasing"):
            controller.read()
        assert controller.close()
    finally:
        controller.close()


def test_slew_limits_each_axis_without_overshoot_or_direction_selection():
    assert slew([0, 4, 1], [5, -4, 1], rate=2, dt=0.5) == (1.0, 3.0, 1.0)
    assert slew([0, 4], [0.2, -4], rate=[1, 2], dt=1) == (0.2, 2.0)
    assert slew([2, -3], [-9, 9], rate=0, dt=5) == (2.0, -3.0)
    assert slew([2], [-9], rate=7, dt=0) == (2.0,)
    assert slew([-1e308], [1e308], rate=1e308, dt=1) == (0.0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current": []},
        {"current": [True]},
        {"current": [float("nan")]},
        {"target": [float("inf")]},
        {"target": [1, 2]},
        {"target": [[1]]},
        {"rate": -1},
        {"rate": [1, 2]},
        {"rate": [True]},
        {"rate": [-1]},
        {"rate": float("inf")},
        {"dt": -1},
        {"dt": True},
        {"dt": float("inf")},
        {"rate": 1e308, "dt": 1e308},
    ],
)
def test_slew_rejects_nonfinite_or_mismatched_inputs(kwargs):
    with pytest.raises(ValueError):
        slew(**{"current": [0], "target": [1], "rate": 1, "dt": 1, **kwargs})
