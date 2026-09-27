"""The orienting instruments: capture, latency, return, the habituation curve, dishabituation."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import dishabituation, orienting


def _trace():
    g = np.full(200, 0.5)
    events = []
    # ten clock ticks whose capture falls from 0.4 to 0.04, each one frame, twelve frames apart
    for i in range(10):
        start = 10 + 12 * i
        g[start] = 0.5 + 0.4 * (1 - i / 10)
        g[start + 1] = 0.5 + 0.2 * (1 - i / 10)   # the return
        events.append(("clock", start, 1))
    # a cry of three frames at 150 that rises over two frames and stays up for four after
    g[150:153] = [0.6, 0.9, 1.0]
    g[153:157] = [0.9, 0.8, 0.7, 0.6]
    events.append(("cry", 150, 3))
    # a clock after the cry, back to a large capture (dishabituation)
    g[170] = 0.9
    events.append(("clock", 170, 1))
    return g, events


def test_capture_latency_and_return_of_one_event_by_hand():
    g, events = _trace()
    out = orienting(g, events, pre=4, post=12)
    cry = [r for r in out["rows"] if r["kind"] == "cry"][0]
    assert cry["baseline"] == pytest.approx(0.5) and cry["peak"] == pytest.approx(1.0)
    assert cry["capture"] == pytest.approx(0.5)
    assert cry["latency"] == 1                       # half the capture is reached one frame after the onset
    assert cry["return"] == 3                        # back within a fifth of the capture three frames after the event
    first = [r for r in out["rows"] if r["kind"] == "clock"][0]
    assert first["capture"] == pytest.approx(0.4) and first["latency"] == 0 and first["return"] == 1


def test_the_habituation_curve_and_the_kinds_summary():
    g, events = _trace()
    out = orienting(g, events, bins=5)
    clock = out["kinds"]["clock"]
    assert clock["count"] == 11
    assert clock["capture_first"] > clock["capture_last"]       # the clock habituates
    assert len(clock["curve"]) == 3 and clock["curve"][0] > clock["curve"][1]
    assert clock["curve"][-1] == pytest.approx(0.4)  # the final partial bin is not discarded
    assert clock["latency_zero"] == 1.0 and clock["latency_none"] == 0.0
    assert out["kinds"]["cry"]["capture_mean"] == pytest.approx(0.5)


def test_events_without_a_quiet_baseline_or_past_the_end_are_skipped():
    g = np.full(30, 1.0)
    out = orienting(g, [("a", 2, 1), ("b", 28, 5), ("c", 0, 1)], pre=2)
    kinds = [r["kind"] for r in out["rows"]]
    assert kinds == ["a"]                    # b runs past the trace, c has no frame before it
    quiet = np.zeros(30, dtype=bool)
    assert orienting(g, [("a", 2, 1)], quiet=quiet)["rows"] == []   # no usable baseline frame
    with pytest.raises(ValueError, match="frames"):
        orienting(np.ones((3, 3)), [])
    with pytest.raises(ValueError, match="flag"):
        orienting(g, [], quiet=np.ones(3, dtype=bool))


def test_dishabituation_compares_the_ticks_around_the_cry():
    g, events = _trace()
    rows = orienting(g, events)["rows"]
    d = dishabituation(rows, consequential="cry", kind="clock", window=60, count=2)
    assert d["events"] == 3 and d["after"] > d["before"]    # the tick after the cry captures more than the two before
    none = dishabituation(rows, consequential="phone", kind="clock")
    assert none == {"before": None, "after": None, "events": 0}
