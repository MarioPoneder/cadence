import random

import pytest

from cadence import Cortex


def make(**options):
    return Cortex.from_dimensions(2, 1, bounds=(0, 1), bins=2, decay=1.0, **options)


def run(model, steps=900):
    rng, errors = random.Random(0), []
    for t in range(steps):
        b = [rng.randint(0, 1), rng.randint(0, 1)]
        y = float(b[0] & b[1] if (t // 150) % 2 == 0 else b[0] ^ b[1])
        if t >= 300:
            errors.append(abs(model.predict(b)[0] - y))
        model.observe(b, y)
    return sum(errors) / len(errors)


def test_self_observation_is_the_default_and_needs_height():
    assert make().self_observation is True
    assert make().reflective is False  # height one has no modes to select
    assert make(height=3).reflective is True
    assert make(height=3, self_observation=False).reflective is False


def test_height_one_is_unchanged_by_the_flag():
    assert run(make(height=1)) == run(make(height=1, self_observation=False))


def test_height_changes_answers_under_hidden_rule_switches():
    plain = run(make(height=3, self_observation=False))
    reflective = make(height=3)
    assert run(reflective) < plain / 5
    assert reflective.stats()["mode_switches"] > 0


def test_reflection_state_round_trips():
    model = make(height=3)
    run(model, 400)
    clone = Cortex.from_snapshot(model.snapshot())
    assert clone.snapshot() == model.snapshot() and clone.mode == model.mode
    assert clone.predict([1, 0]) == model.predict([1, 0])


def test_invalid_reflection_parameters_are_rejected():
    with pytest.raises(ValueError):
        make(height=2, reflect_threshold=1.0)
    with pytest.raises(ValueError):
        make(height=2, reflect_rate=0.0)
