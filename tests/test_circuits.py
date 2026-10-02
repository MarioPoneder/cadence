import numpy as np
import pytest

from cadence.circuits import ActivityMonitor, imagine, reflex_arc


def test_imagination_preserves_nested_live_state_and_responds_to_adversary():
    live = {"trail": [], "values": np.zeros(2)}

    def move(s, a):
        s["trail"].append(a)
        s["values"][a] += 1
        return s

    def value(s):
        # First route is tempting but opponent can refute it; second is safe.
        return (10 if s["trail"][1] == 0 else -10) if s["trail"][0] == 0 else 2

    result = imagine(
        live,
        lambda s: (0, 1),
        move,
        value,
        lambda s: len(s["trail"]) == 2,
        depth=2,
        adversarial=True,
    )
    assert result.futures[0].action == 1
    assert result.nodes == 6 and live["trail"] == []
    np.testing.assert_array_equal(live["values"], [0, 0])
    result.futures[0].state["values"][0] = 99
    assert result.futures[1].state["values"][0] != 99
    with pytest.raises(ValueError, match="budget"):
        imagine(
            live,
            lambda s: (0, 1),
            move,
            value,
            lambda s: len(s["trail"]) == 2,
            depth=2,
            max_nodes=1,
        )


def test_monitor_requests_work_for_ambiguous_choices_but_respects_budget():
    monitor = ActivityMonitor()
    uncertain = monitor.read(np.zeros(4), np.array([0.1, 0.1]))
    assert uncertain.request_more
    pressured = monitor.read(np.ones(4), np.array([0.1, 0.1]), pressure=1)
    assert pressured.activity_change == 1 and not pressured.request_more
    certain = ActivityMonitor().read(np.zeros(4), np.array([-100.0, 100.0]))
    assert not certain.request_more
    with pytest.raises(ValueError):
        monitor.read(np.zeros(4), np.array([np.nan]))


def test_sensor_motor_population_counts():
    connectome = reflex_arc(3)
    assert connectome.n == 9 and connectome.synapses == 6
    assert connectome.populations["sensory"] == (0, 1, 2)
    assert connectome.populations["positive"] == (3, 5, 7)
    assert connectome.populations["negative"] == (4, 6, 8)
    with pytest.raises(ValueError):
        reflex_arc(0)
