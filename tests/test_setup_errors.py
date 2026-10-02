"""Common first-run mistakes produce actionable errors without changing memory."""

from numbers import Integral

import pytest

from cadence import Cortex, SettlementError


def learner():
    layout = Cortex(seed=7)
    eyes = layout.input("eyes", shape=2)
    ears = layout.input("ears", shape=1)
    sense = layout.column("sense", patches=2, inputs=(eyes, ears))
    body = layout.column("body", patches=1, inputs=sense)
    layout.output("move", shape=1, reads=body)
    layout.output("alias", shape=1, reads=body)
    return layout.build(), eyes


def test_registered_integral_scalar_shapes_match_sequence_shapes():
    # Array-library integer scalars use Integral without subclassing builtin int.
    # Exercise that public numeric contract without requiring an array library.
    @Integral.register
    class IntegerScalar:
        def __int__(self):
            return 2

        def __lt__(self, other):
            return 2 < other

    brains = []
    for shape in (2, IntegerScalar(), (IntegerScalar(),)):
        layout = Cortex()
        signal = layout.input("signal", shape=shape)
        features = layout.column("features", patches=2, inputs=signal)
        state = layout.column("state", patches=2, inputs=features)
        layout.output("answer", shape=shape, reads=state)
        brains.append(layout.build())
    assert brains[0].snapshot() == brains[1].snapshot() == brains[2].snapshot()


def test_larger_state_bound_does_not_rescale_terminal_predictions():
    layout = Cortex(state_bound=4.0)
    signal = layout.input("signal", shape=1)
    features = layout.column("features", patches=2, inputs=signal)
    state = layout.column("state", patches=1, inputs=features)
    layout.output("answer", shape=1, reads=state)
    brain = layout.build()
    for _ in range(20):
        assert brain.observe({"signal": [0.0]}, {"answer": [2.0]})["accepted"]
    answer = brain.predict({"signal": [0.0]})["answer"][0]
    # For a terminal patch, dE/dx = (1 + prior)*x - tanh(drive).
    # Admitting the larger clamp cannot turn its free answer into target 2.
    ceiling = 1 / (1 + brain.config["state_prior"])
    assert 0.9 < answer <= ceiling + brain.config["tolerance"]


@pytest.mark.parametrize(
    ("inputs", "targets", "message"),
    [
        ({"eyes": [0.2, -0.3]}, {"move": [0.5]}, "missing: ears"),
        (
            {"eyes": [0.2, -0.3], "ear": [0.1]},
            {"move": [0.5]},
            "'ear'; expected names: eyes, ears",
        ),
        (
            {"eyes": [0.2, -0.3], "ears": [0.1]},
            {"motor": [0.5]},
            "'motor'; expected names: move, alias",
        ),
        (
            {"eyes": [0.2, -0.3], "ears": [0.1]},
            {"move": [2.0]},
            "'move' clamp 2.0 exceeds state_bound=1.0",
        ),
        (
            {"eyes": [0.2, -0.3], "ears": [0.1]},
            {"move": [0.5], "alias": [-0.5]},
            "'alias' conflicts with another clamp on patch 2",
        ),
    ],
)
def test_invalid_experience_names_the_problem_and_preserves_continuation(
    inputs, targets, message
):
    brain, _ = learner()
    before = brain.snapshot()
    with pytest.raises(ValueError) as error:
        brain.observe(inputs, targets, event_id=0)
    assert message in str(error.value)
    assert brain.snapshot() == before
    assert brain.observe(
        {"eyes": [0.2, -0.3], "ears": [0.1]}, {"move": [0.5]}, event_id=0
    )["accepted"]


def test_duplicate_sensor_name_and_handle_are_identified():
    brain, eyes = learner()
    before = brain.snapshot()
    with pytest.raises(ValueError, match="Duplicate values for 'eyes'"):
        brain.predict({"eyes": [0.2, -0.3], eyes: [0.2, -0.3], "ears": [0.1]})
    assert brain.snapshot() == before


def test_refusal_explains_numerical_threshold_without_admitting_outputs():
    brain, _ = learner()
    inputs = {"eyes": [0.9, -0.7], "ears": [0.4]}
    before = brain.snapshot()
    refused = brain.settle(inputs, budget=0)
    assert not refused["qualified"]
    with pytest.raises(SettlementError) as error:
        brain.predict(inputs, budget=0)
    message = str(error.value)
    assert "budget" in message and "sweeps=0" in message
    assert f"stationarity={refused['stationarity']:.6g}" in message
    assert f"tolerance={brain.config['tolerance']:.6g}" in message
    assert "settle()" in message
    assert brain.snapshot() == before
    assert brain.settle(inputs)["qualified"]
    assert brain.snapshot() == before
