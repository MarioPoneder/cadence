"""Small public-API acquisition checks, scored only on unclamped predictions.

These signed relations check usable defaults and persistence, not a benefit
from depth. Hidden and observer readouts have no direct sensory skip: their
answers must use the representations in the jointly settling populations.
"""

from itertools import product

import pytest

from cadence import Brain, Cortex

TRAINING = tuple(product((-0.8, 0.8), repeat=2))
QUERIES = tuple(product((-0.4, 0.4), repeat=2))


def learner(kind, seed, **config):
    cortex = Cortex(seed=seed, **config)
    left = cortex.input("left", shape=1)
    right = cortex.input("right", shape=1)
    base = cortex.column("base", patches=4, inputs=(left, right))
    readout = base
    if kind in ("ordinary", "deep"):
        hidden = cortex.column("integration", patches=3, inputs=base)
        readout = hidden
        if kind == "deep":
            readout = cortex.observer("reflection", patches=2, observes=(base, hidden))
    elif kind == "recursive":
        readout = cortex.observer("reflection", patches=2, observes=base)
    cortex.output("horizontal", shape=1, reads=readout, indices=(0,))
    cortex.output("vertical", shape=1, reads=readout, indices=(1,))
    return cortex.build()


def teach(brain, cases):
    for left, right in cases:
        result = brain.observe(
            {"left": [left], "right": [right]},
            {"horizontal": [left], "vertical": [-right]},
        )
        assert result["accepted"], (result["reason"], result["stationarity"])


def response(brain, left, right):
    # predict raises on an unqualified query; refused outputs cannot score.
    outputs = brain.predict({"left": [left], "right": [right]})
    return outputs["horizontal"][0], outputs["vertical"][0]


def responses(brain):
    checkpoint = brain.snapshot()
    predictions = tuple(response(brain, *inputs) for inputs in QUERIES)
    assert brain.snapshot() == checkpoint
    return predictions


def squared_error(predictions):
    return sum(
        (horizontal - left) ** 2 + (vertical + right) ** 2
        for (left, right), (horizontal, vertical) in zip(
            QUERIES, predictions, strict=True
        )
    ) / (2 * len(QUERIES))


def assert_acquired(predictions):
    for (left, right), (horizontal, vertical) in zip(QUERIES, predictions, strict=True):
        assert horizontal * left > 0.08
        assert vertical * -right > 0.08
    # Toggling either input should change its own output, without also
    # reversing the independent response to the other held-fixed input.
    for a, b in ((0, 2), (1, 3)):
        assert predictions[b][0] - predictions[a][0] > 0.4
        assert abs(predictions[b][1] - predictions[a][1]) < 0.15
    for a, b in ((0, 1), (2, 3)):
        assert predictions[a][1] - predictions[b][1] > 0.4
        assert abs(predictions[a][0] - predictions[b][0]) < 0.15


@pytest.mark.parametrize("kind", ("flat", "ordinary", "recursive"))
@pytest.mark.parametrize("seed", (0, 2, 7))
def test_defaults_acquire_independent_relations_and_retain_them_on_replay(kind, seed):
    brain = learner(kind, seed)
    initial_error = squared_error(responses(brain))
    teach(brain, TRAINING * 12)

    acquired = responses(brain)
    assert_acquired(acquired)
    assert squared_error(acquired) < initial_error / 4

    # Different final witness/order: success must not be a held motor state.
    teach(brain, tuple(reversed(TRAINING)) * 8)
    retained = responses(brain)
    assert_acquired(retained)
    assert squared_error(retained) < initial_error / 4

    # Remove one cue at a time while preserving the other cue's response.
    for value in (-0.4, 0.4):
        horizontal, vertical = response(brain, 0.0, value)
        assert abs(horizontal) < 0.15
        assert vertical * -value > 0.08
        horizontal, vertical = response(brain, value, 0.0)
        assert horizontal * value > 0.08
        assert abs(vertical) < 0.15


def test_checkpoint_continues_the_same_learning_and_unclamped_answers():
    original = learner("recursive", seed=2)
    teach(original, TRAINING * 6)
    restored = Brain.from_snapshot(original.snapshot())
    assert responses(restored) == responses(original)

    for brain in (original, restored):
        teach(brain, TRAINING * 6)
        assert_acquired(responses(brain))
        assert brain.inspect()["admissions"] == 48
    assert restored.snapshot() == original.snapshot()


@pytest.mark.parametrize("seed", (0, 2, 7, 11, 29, 73))
def test_deeper_observer_acquires_relations_with_default_settings(seed):
    brain = learner("deep", seed)
    initial_error = squared_error(responses(brain))
    teach(brain, TRAINING * 12)
    acquired = responses(brain)
    assert_acquired(acquired)
    assert squared_error(acquired) < initial_error / 4


def test_capped_deep_witness_rolls_back_and_can_retry_with_default_budget():
    brain = learner("deep", seed=0)
    inputs = {"left": [-0.8], "right": [-0.8]}
    targets = {"horizontal": [-0.8], "vertical": [0.8]}
    checkpoint = brain.snapshot()

    refused = brain.observe(inputs, targets, event_id=17, budget=512)
    assert not refused["accepted"]
    assert refused["reason"] == "budget"
    assert brain.snapshot() == checkpoint

    accepted = brain.observe(inputs, targets, event_id=17)
    assert accepted["accepted"]
    assert brain.inspect()["admissions"] == 1
    assert brain.inspect()["last_event_id"] == 17
