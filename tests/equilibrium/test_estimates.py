"""Derived targets use ordinary repair without acquiring witness provenance."""

import copy
import json

import pytest

from cadence.experimental.equilibrium import Brain, Cortex


def learner(*, recursive=False):
    cortex = Cortex(seed=7)
    signal = cortex.input("signal", shape=1)
    base = cortex.column("base", patches=2, inputs=signal)
    top = (
        cortex.observer("top", patches=2, inputs=signal, observes=base)
        if recursive
        else cortex.column("top", patches=2, inputs=(signal, base))
    )
    cortex.output("answer", shape=1, reads=top)
    return cortex.build()


ROWS = [({"signal": [value]}, {"answer": [0.6 * value]}) for value in (-0.7, 0.5)]


def admit(brain, batch, **options):
    if batch:
        return brain.observe_batch(ROWS, **options)
    return brain.observe(*ROWS[0], **options)


@pytest.mark.parametrize("batch", (False, True))
@pytest.mark.parametrize("recursive", (False, True))
def test_estimates_and_witnesses_have_identical_repair_but_distinct_custody(batch, recursive):
    witness = learner(recursive=recursive)
    assert witness.step({"signal": [0.2]})["accepted"]
    estimate = Brain.from_snapshot(witness.snapshot())
    before = witness.state
    actual = admit(witness, batch, source="witness", event_id=3)
    derived = admit(estimate, batch, source="estimate", event_id=3)
    assert actual["accepted"] and derived["accepted"]
    assert actual.pop("source") == "witness"
    assert derived.pop("source") == "estimate"
    assert actual == derived
    assert witness.state == estimate.state
    if batch:
        assert estimate.state == before
    else:
        assert estimate.state != before
    assert witness.weights == estimate.weights
    assert witness.biases == estimate.biases
    assert witness.predict({"signal": [0.4]}) == estimate.predict({"signal": [0.4]})
    actual_record = json.loads(witness.snapshot())
    estimate_record = json.loads(estimate.snapshot())
    assert actual_record.pop("event_digest") != estimate_record.pop("event_digest")
    assert actual_record == estimate_record


@pytest.mark.parametrize("batch", (False, True))
def test_default_source_is_explicit_witness_with_identical_continuation(batch):
    default, explicit = learner(), learner()
    result = admit(default, batch)
    assert result["source"] == "witness"
    assert result == admit(explicit, batch, source="witness")
    assert default.snapshot() == explicit.snapshot()


@pytest.mark.parametrize("batch", (False, True))
@pytest.mark.parametrize("source", ("witness", "estimate"))
def test_snapshot_and_restore_preserve_source_bound_retries(batch, source):
    brain = learner(recursive=True)
    assert admit(brain, batch, source=source, event_id=5)["accepted"]
    saved = brain.snapshot()
    restored = Brain.from_snapshot(saved)
    assert restored.snapshot() == saved
    # Neither an intervening live-state update nor restore can relabel the event.
    assert restored.step({"signal": [0.4]})["accepted"]
    before = restored.snapshot()
    retry = admit(restored, batch, source=source, event_id=5, budget="unused for retry")
    assert retry == {
        "accepted": False,
        "qualified": True,
        "duplicate": True,
        "event_id": 5,
        "source": source,
        **({"batch_size": len(ROWS)} if batch else {}),
    }
    assert restored.snapshot() == before
    opposite = "estimate" if source == "witness" else "witness"
    with pytest.raises(ValueError, match="conflicts"):
        admit(restored, batch, source=opposite, event_id=5)
    assert restored.snapshot() == before
    restored.restore(saved)
    with pytest.raises(ValueError, match="conflicts"):
        admit(restored, batch, source=opposite, event_id=5)
    assert restored.snapshot() == saved
    # A new event may have different provenance in the same ordered stream.
    update = admit(restored, batch, source=opposite)
    assert update["accepted"] and update["event_id"] == 6
    assert update["source"] == opposite
    assert restored.inspect()["admissions"] == 2


@pytest.mark.parametrize("batch", (False, True))
@pytest.mark.parametrize(
    "source", (None, True, 1, 0.0, "", "Witness", "derived", b"estimate", [], {})
)
def test_invalid_source_is_atomic_and_precedes_argument_parsing(batch, source, monkeypatch):
    brain = learner()
    assert admit(brain, batch, source="estimate", event_id=2)["accepted"]
    before = brain.snapshot()
    monkeypatch.setattr(brain, "_solve", lambda *a, **k: pytest.fail("invalid solve"))
    monkeypatch.setattr(
        brain, "_arguments", lambda *a, **k: pytest.fail("invalid argument parsing")
    )
    with pytest.raises(ValueError, match="source must be 'witness' or 'estimate'"):
        admit(brain, batch, source=source, event_id=2)
    assert brain.snapshot() == before


@pytest.mark.parametrize("batch", (False, True))
@pytest.mark.parametrize("source", ("witness", "estimate"))
def test_refusal_returns_source_without_consuming_event_or_parameters(batch, source):
    brain = learner(recursive=True)
    before = brain.snapshot()
    result = admit(brain, batch, source=source, event_id=4, budget=0)
    assert not result["accepted"] and not result["qualified"]
    assert not result["duplicate"] and result["reason"] == "budget"
    assert result["source"] == source
    assert brain.snapshot() == before
    # A refused estimate was never admitted, so it owns no retry identity.
    opposite = "witness" if source == "estimate" else "estimate"
    result = admit(brain, batch, source=opposite, event_id=4)
    assert result["accepted"] and result["source"] == opposite
    assert brain.inspect()["admissions"] == 1


@pytest.mark.parametrize("batch", (False, True))
def test_estimates_keep_boundary_validation_and_exception_atomicity(batch, monkeypatch):
    brain = learner()
    before = brain.snapshot()
    for targets in ({}, {"answer": [float("nan")]}, {"answer": [100]}):
        with pytest.raises(ValueError):
            if batch:
                brain.observe_batch([*ROWS, (ROWS[0][0], targets)], source="estimate")
            else:
                brain.observe(ROWS[0][0], targets, source="estimate")
        assert brain.snapshot() == before

    def failed_solve(*args, **kwargs):
        raise ValueError("nonrepresentable solve")

    monkeypatch.setattr(brain, "_solve", failed_solve)
    with pytest.raises(ValueError, match="nonrepresentable"):
        admit(brain, batch, source="estimate")
    assert brain.snapshot() == before


@pytest.mark.parametrize("batch", (False, True))
def test_estimate_retries_still_bind_values_order_and_call_kind(batch):
    brain = learner()
    rows = copy.deepcopy(ROWS)
    if batch:
        first = brain.observe_batch(rows, source="estimate", event_id=2)
    else:
        first = brain.observe(*rows[0], source="estimate", event_id=2)
    assert first["accepted"]
    assert rows == ROWS
    before = brain.snapshot()
    rows[0][1]["answer"][0] += 0.1
    with pytest.raises(ValueError, match="conflicts"):
        if batch:
            brain.observe_batch(rows, source="estimate", event_id=2)
        else:
            brain.observe(*rows[0], source="estimate", event_id=2)
    with pytest.raises(ValueError, match="conflicts"):
        admit(brain, not batch, source="estimate", event_id=2)
    if batch:
        with pytest.raises(ValueError, match="conflicts"):
            brain.observe_batch(ROWS[::-1], source="estimate", event_id=2)
    with pytest.raises(ValueError, match="older"):
        admit(brain, batch, source="estimate", event_id=1)
    assert brain.snapshot() == before


def test_estimated_labels_can_teach_unclamped_predictions_without_becoming_facts():
    brain = learner()
    for _ in range(20):
        assert brain.observe_batch(ROWS, source="estimate")["accepted"]
    for value in (-0.3, 0.3):
        output = brain.predict({"signal": [value]})["answer"][0]
        assert abs(output - 0.6 * value) < 0.08
    # This checks acquisition of the supplied relation, not its truth in a world.
    assert brain.inspect()["admissions"] == 20
    assert brain.state == (0.0, 0.0, 0.0, 0.0)
