"""Independent numerical oracles and continuation properties for predictors."""

import json
import math
import random
from fractions import Fraction

import pytest

from cadence import Cortex, grid


@pytest.mark.parametrize("decay", [1.0, 0.5, 0.9])
@pytest.mark.parametrize("height", [1, 2, 3])
def test_weighted_stream_matches_independent_evidence_and_scalar_fixed_point(
    decay, height
):
    net = Cortex(decay=decay, height=height, optimism=0)
    weight, linear, square = Fraction(1), Fraction(0), Fraction(0)
    for event, (value, mass) in enumerate(((2, 1), (-1, 2), (3, 1), (-2, 2)), 1):
        factor = Fraction(str(decay)) ** mass
        weight, linear, square = (
            factor * weight + mass,
            factor * linear + mass * value,
            factor * square + mass * value * value,
        )
        assert net.observe("sensor", value, weight=mass, event=event)["accepted"]
        belief = net.value("sensor")
        assert belief["qualified"]
        assert belief["mean"] == pytest.approx(float(linear / weight), abs=1e-12)
        if height == 1:
            scatter = square - linear * linear / weight
            precision = (1 + (weight - 1) / 2) / (1 + scatter / 2)
            expected_variance = float(1 / (weight * precision))
            assert belief["variance"] == pytest.approx(expected_variance, abs=2e-10)


def test_coarse_prior_generalizes_only_between_shared_contexts():
    deep = Cortex(feature_maps=grid([(-1, 1)], bins=4, depth=1), decay=1)
    flat = Cortex(feature_maps=grid([(-1, 1)], bins=4, depth=0), decay=1)
    for event in range(1, 5):
        deep.observe((-0.75,), 2, event=event)
        flat.observe((-0.75,), 2, event=event)
    assert deep.predict((0.75,))[0] > 0
    assert flat.predict((0.75,)) == (0.0,)
    deep.prior_ports_cut = True
    assert deep.predict((0.75,)) == (0.0,)
    # Cutting the prior is a readout intervention, not erasure of experience.
    deep.prior_ports_cut = False
    assert deep.predict((0.75,))[0] > 0


def test_zero_coupling_agrees_with_flat_evidence_under_matched_witness_mass():
    coupled = Cortex(
        feature_maps=grid([(-1, 1)], bins=4, depth=2),
        level_weights=(1, 1, 1),
        coupling=0,
    )
    flat = Cortex(feature_maps=grid([(-1, 1)], bins=4, depth=0))
    for event, (observation, target) in enumerate(
        (((-0.8,), 2), ((0.8,), -2), ((-0.7,), 1), ((0.7,), 0)), 1
    ):
        coupled.observe(observation, target, event=event)
        flat.observe(observation, target, event=event)
    for observation in ((-0.75,), (0.75,), (0.0,)):
        assert coupled.predict(observation) == flat.predict(observation)
        assert coupled.value(observation)["variance"] == pytest.approx(
            flat.value(observation)["variance"]
        )


@pytest.mark.parametrize("seed", [0, 13, 99])
def test_serialized_lives_continue_with_identical_evidence_and_actions(seed):
    rng = random.Random(seed)
    net = Cortex.from_dimensions(
        2, n_outputs=3, bins=4, depth=1, decay=0.9, seed=seed, epsilon=0.4
    )
    for _ in range(5):
        net.observe(
            (rng.choice((-0.5, 0.5)), rng.choice((-0.5, 0.5))),
            [rng.uniform(-1, 1) for _ in range(3)],
        )
    net.learn((-0.5, 0.5), 1, 0.5, (0.5, -0.5))
    twin = Cortex.from_snapshot(net.snapshot())
    assert twin.snapshot() == net.snapshot()
    assert twin.flush() == net.flush()
    for event in range(7, 12):
        context = (rng.choice((-0.5, 0.5)), rng.choice((-0.5, 0.5)))
        targets = [rng.uniform(-1, 1) for _ in range(3)]
        assert twin.act(context) == net.act(context)
        assert twin.observe(context, targets, event=event) == net.observe(
            context, targets, event=event
        )
        assert twin.predict(context) == net.predict(context)
    left, right = json.loads(net.snapshot()), json.loads(twin.snapshot())
    # Caches are deliberately derived state, so their work counts may differ.
    left.pop("counters")
    right.pop("counters")
    assert left == right


def test_freezing_preserves_pending_events_and_resumes_without_recounting():
    net = Cortex(optimism=0, discount=0, decay=1)
    net.learn("first", 0, 3, "second", event=1)
    net.learning_enabled = False
    before = net.snapshot()
    assert net.flush() == 0
    assert not net.learn("second", 0, 2, None, terminal=True, event=2)["admitted"]
    assert net.snapshot() == before
    net.learning_enabled = True
    assert net.flush() == 1
    assert net.predict("first") == (1.5,)
    assert net.learn("first", 0, 3, "second", event=1)["duplicate"]
    assert net.flush() == 0


@pytest.mark.parametrize("field", ["weight", "linear", "square"])
def test_large_target_arithmetic_cannot_poison_retained_state(field):
    net = Cortex()
    net.observe("context", 1)
    before = json.loads(net.snapshot())["rows"]
    kwargs = {"weight": 1e308} if field == "weight" else {}
    target = 1e308 if field != "weight" else 2.0
    with pytest.raises(ValueError):
        net.observe("context", target, **kwargs)
    assert json.loads(net.snapshot())["rows"] == before
    assert all(math.isfinite(v) for v in net.predict("context"))
