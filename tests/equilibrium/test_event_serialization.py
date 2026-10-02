"""Admission identity must remain serializable before any proposal is solved."""

import sys

import pytest

from cadence.experimental.equilibrium import Brain, Cortex


@pytest.fixture
def digit_limit():
    original = sys.get_int_max_str_digits()
    limit = sys.int_info.str_digits_check_threshold
    sys.set_int_max_str_digits(limit)
    try:
        yield limit
    finally:
        sys.set_int_max_str_digits(original)


def small_brain():
    layout = Cortex()
    signal = layout.input("signal", shape=1)
    features = layout.column("features", patches=2, inputs=signal)
    state = layout.column("state", patches=1, inputs=features)
    layout.output("answer", shape=1, reads=state)
    return layout.build()


def test_oversized_explicit_identity_is_rejected_before_solving(digit_limit, monkeypatch):
    brain = small_brain()
    before = brain.snapshot()
    solve = brain._solve
    monkeypatch.setattr(
        brain, "_solve", lambda *args, **kwargs: pytest.fail("invalid ID reached solve")
    )
    with pytest.raises(ValueError, match="event_id.*JSON integer"):
        brain.observe({"signal": [0.0]}, {"answer": [0.0]}, event_id=10**digit_limit)
    assert brain.snapshot() == before
    monkeypatch.setattr(brain, "_solve", solve)
    result = brain.observe({"signal": [0.0]}, {"answer": [0.0]}, event_id=0)
    assert result["accepted"] and result["event_id"] == 0
    assert brain.inspect()["admissions"] == 1


def test_automatic_identity_overflow_preserves_continuation(digit_limit, monkeypatch):
    brain = small_brain()
    largest = 10**digit_limit - 1
    assert brain.observe({"signal": [0.0]}, {"answer": [0.0]}, event_id=largest)["accepted"]
    before = brain.snapshot()
    monkeypatch.setattr(
        brain, "_solve", lambda *args, **kwargs: pytest.fail("invalid ID reached solve")
    )
    with pytest.raises(ValueError, match="event_id.*JSON integer"):
        brain.observe({"signal": [0.0]}, {"answer": [0.0]})
    assert brain.snapshot() == before
    assert brain.inspect()["last_event_id"] == largest
    assert brain.inspect()["admissions"] == 1


def test_large_serializable_identity_roundtrips_and_recognizes_retry(digit_limit):
    brain = small_brain()
    identity = 10**128 + 7
    result = brain.observe({"signal": [0.0]}, {"answer": [0.0]}, event_id=identity)
    assert result["accepted"]
    saved = brain.snapshot()
    restored = Brain.from_snapshot(saved)
    assert restored.snapshot() == saved
    retry = restored.observe({"signal": [0.0]}, {"answer": [0.0]}, event_id=identity)
    assert retry["duplicate"] and not retry["accepted"]
    next_event = restored.observe({"signal": [0.0]}, {"answer": [0.0]})
    assert next_event["accepted"] and next_event["event_id"] == identity + 1


def test_same_identity_can_be_retried_after_serialization_limit_changes(digit_limit):
    brain = small_brain()
    identity = 10**digit_limit
    before = brain.snapshot()
    with pytest.raises(ValueError, match="event_id.*JSON integer"):
        brain.observe({"signal": [0.0]}, {"answer": [0.0]}, event_id=identity)
    assert brain.snapshot() == before
    sys.set_int_max_str_digits(digit_limit + 1)
    result = brain.observe({"signal": [0.0]}, {"answer": [0.0]}, event_id=identity)
    assert result["accepted"] and not result["duplicate"]
    assert brain.inspect()["admissions"] == 1
