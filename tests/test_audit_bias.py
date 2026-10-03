"""Audit 2026-09-15, class 1: the rectified neuron is exactly zero at rest.

By default, every builder starts the biases at zero. Under a sign-symmetric random projection about
half of the free hidden neurons then sit at or below rest, publish at most ``-leak``, and
carry an order of magnitude less contrast than the active half. The strict xfail below
records the defect: it turns into a failure the day a builder or ``NeuralGraph`` handles it, so
the test is then updated rather than forgotten.
"""

import json

import numpy as np
import pytest

import cadence as cd


def _drive_embedded(connectome: cd.Connectome, rng: np.random.Generator, batch: int) -> np.ndarray:
    d = np.zeros((batch, connectome.n))
    for p in range(3):
        d[np.arange(batch), p * 12 + rng.integers(0, 12, batch)] = 1.0
    return d


def _silent_fraction(brain: cd.NeuralGraph, drive: np.ndarray, members: list[int]) -> float:
    state = brain.settle_batch(drive, steps=300, tolerance=1e-6)
    return float((np.asarray(state.activation)[:, members] <= 0.0).mean())


def _builders() -> list[tuple[str, cd.NeuralGraph, np.ndarray, list[int]]]:
    rng = np.random.default_rng(0)
    out = []
    c = cd.layered(20, 40, 5, seed=0)
    d = np.zeros((16, c.n))
    d[:, :20] = rng.random((16, 20))
    out.append(("layered", cd.NeuralGraph(c, cd.learning_neuron_model()), d, list(c.populations["hidden"])))
    c, _ = cd.embedded(12, 3, 6, 24, 5, seed=0)
    brain = cd.NeuralGraph(c, cd.learning_neuron_model(dt=1.0))
    out.append(("embedded", brain, _drive_embedded(c, rng, 16), list(c.populations["hidden"])))
    c, _ = cd.stateful(12, 3, 6, 24, 5, seed=0)
    brain = cd.NeuralGraph(c, cd.learning_neuron_model(dt=1.0))
    out.append(("stateful", brain, _drive_embedded(c, rng, 16), list(c.populations["hidden"])))
    g = cd.Brain.build(10, 4, hidden=64, seed=0)
    drive = g.stimulus(rng.random((16, 10)), memory=False)
    out.append(("generic", g.brain, drive, list(g.association_index)))
    g = cd.Brain.build((8, 8), 4, hidden=64, seed=0)
    drive = g.stimulus(rng.random((16, 8, 8)), memory=False)
    out.append(("generic-image", g.brain, drive, list(g.connectome.populations["visual/output"])))
    return out


def test_every_builder_starts_with_zero_bias() -> None:
    for name, brain, _, _ in _builders():
        assert not brain.bias.any(), name


@pytest.mark.xfail(
    reason="builders start biases at zero: about half of the free hidden neurons sit at or "
    "below rest and publish nothing (audit 2026-09-15, open)",
    strict=True,
)
def test_free_hidden_neurons_are_mostly_responsive() -> None:
    for name, brain, drive, members in _builders():
        assert _silent_fraction(brain, drive, members) < 0.25, name


def test_silent_hidden_neurons_carry_little_contrast_and_a_resting_bias_repairs_it() -> None:
    rng = np.random.default_rng(0)
    c = cd.layered(20, 40, 5, seed=0)
    d = np.zeros((16, c.n))
    d[:, :20] = rng.random((16, 20))
    labels = rng.integers(0, 5, 16)
    hidden = np.asarray(c.populations["hidden"])
    config = cd.LearnerConfig(tolerance=1e-6, free_steps=300, nudged_steps=300)

    def contrast(bias: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        brain = cd.NeuralGraph(c, cd.learning_neuron_model(), bias=np.full(c.n, bias))
        learner = cd.Learner(brain, c.populations["output"], config)
        target = learner.targets(labels)
        free = learner.free(d)
        plus = learner.nudged(d, free, target)
        minus = learner.nudged(d, free, target, sign=-1.0)
        _, neurons = learner.contrast(free, plus, minus)
        return free.activation[:, hidden], np.abs(neurons[hidden]), np.asarray(free.v)[:, hidden]

    activation, contrast_zero, _ = contrast(0.0)
    silent = activation.max(axis=0) <= 0.0
    assert 0.2 < silent.mean() < 0.65  # silent in every row; about half are silent per row
    assert contrast_zero[silent].mean() < 0.25 * contrast_zero[~silent].mean()
    activation_biased, contrast_biased, _ = contrast(0.5)
    assert (activation_biased.max(axis=0) <= 0.0).mean() < 0.2
    assert contrast_biased.mean() > 2.0 * contrast_zero.mean()


def test_resting_bias_is_a_selectable_initialization_gene() -> None:
    """Issue 106: the composed brain can start its processing regions above rest. The default
    stays at zero (the xfail above records that), sensory, working-memory and motor
    populations keep zero bias, and the option is a plain plastic bias from there on."""
    rng = np.random.default_rng(0)
    x = rng.random((16, 10))
    for resting in (0.0, 0.5):  # where the bias lands in a deep layout with an observer
        g = cd.Brain.compose(10, 4, modules=(64, 16), observers=(8,), seed=0, resting_bias=resting)
        pops = g.connectome.populations
        bias = g.brain.bias
        for name in ("sensory", "prefrontal", "motor"):
            assert not bias[np.asarray(pops[name], dtype=np.int64)].any(), name
        for name in ("module_0", "association", "observer_0"):
            members = np.asarray(pops[name], dtype=np.int64)
            assert np.allclose(bias[members], resting), name
        assert g.resting_bias == resting
    silent = {}
    for resting in (0.0, 0.5):  # the responsive fraction on the default single-module layout
        g = cd.Brain.compose(10, 4, modules=(64,), seed=0, resting_bias=resting)
        silent[resting] = _silent_fraction(g.brain, g.stimulus(x, memory=False), list(g.association_index))
    assert silent[0.0] > 0.25 > silent[0.5]
    image = cd.Brain.build((8, 8), 4, hidden=64, seed=0, resting_bias=0.5)
    assert not image.brain.bias[np.asarray(image.connectome.populations["visual/input"], dtype=np.int64)].any()
    assert np.allclose(image.brain.bias[image.association_index], 0.5)
    with pytest.raises(ValueError):
        cd.Brain.compose(10, 4, seed=0, resting_bias=-0.1)


@pytest.mark.parametrize("value", [
    True, np.bool_(False), None, "0.5", 0.5 + 0j, [0.5], (0.5,),
    np.array(0.5), np.array([0.5]), np.array([0.0, 0.5]),
    -0.1, np.nan, np.inf, -np.inf, 10**400,
])
def test_resting_bias_rejects_invalid_scalars(value) -> None:
    with pytest.raises(ValueError, match="resting_bias.*finite nonnegative real scalar"):
        cd.Brain.compose(2, 2, modules=(4,), resting_bias=value)


@pytest.mark.parametrize("value", [0, 1, np.int64(1), np.float32(0.5), np.float64(0.5)])
def test_resting_bias_accepts_real_numeric_scalars(value) -> None:
    brain = cd.Brain.compose(2, 2, modules=(4,), resting_bias=value)
    assert type(brain.resting_bias) is float
    np.testing.assert_array_equal(brain.brain.bias[brain.association_index], float(value))


@pytest.mark.parametrize("sensory_name", ["sensory", "visual/input"])
@pytest.mark.parametrize("reverse", [False, True])
def test_resting_bias_excludes_boundary_neurons_even_through_custom_aliases(
    sensory_name, reverse,
) -> None:
    populations = {
        sensory_name: (0,), "association": (1,), "motor": (2,), "prefrontal": (3,),
        "custom_processing": (4,), "visual/features": (5,), "motor/other": (6,),
        "custom_sense": (0,), "all_named": tuple(range(7)),
    }
    if reverse:
        populations = dict(reversed(list(populations.items())))
    connectome = cd.Connectome.from_synapses(
        8, pre=[0, 1, 2], post=[1, 2, 1], sign=[1.0, 1.0, 1.0], populations=populations,
    )
    brain = cd.Brain(connectome, resting_bias=0.5)
    # Excluded aliases win in either insertion order; unnamed neuron 7 stays at zero.
    np.testing.assert_array_equal(brain.brain.bias, [0, 0.5, 0, 0, 0.5, 0, 0, 0])


def test_explicit_zero_resting_bias_preserves_default_actions_and_diagnostics() -> None:
    default = cd.Brain.compose(2, 2, modules=(4,), seed=3)
    explicit = cd.Brain.compose(2, 2, modules=(4,), seed=3, resting_bias=0.0)
    observation = np.array([[0.2, -0.1]])
    np.testing.assert_array_equal(default.step(observation), explicit.step(observation))
    np.testing.assert_array_equal(default.brain.bias, explicit.brain.bias)
    assert default.last_settlement == explicit.last_settlement
    assert default.last_settlement["qualified"]


def test_resting_bias_checkpoint_preserves_learned_bias_and_pending_continuation(tmp_path) -> None:
    brain = cd.Brain.compose(2, 2, modules=(4,), seed=3, resting_bias=0.5)
    initial = brain.brain.bias.copy()
    brain.step([[0.2, -0.1]], teacher=np.array([1]))
    assert not np.array_equal(brain.brain.bias, initial)
    saved = brain.save(tmp_path / "pending.npz")
    restored = cd.Brain.load(saved)
    assert restored.resting_bias == brain.resting_bias == 0.5
    np.testing.assert_array_equal(restored.brain.bias, brain.brain.bias)
    assert restored.last_settlement is None
    reward, done, following = np.array([0.5]), np.array([False]), [[-0.1, 0.2]]
    np.testing.assert_array_equal(
        brain.step(following, reward=reward, done=done),
        restored.step(following, reward=reward, done=done),
    )
    assert brain.last_settlement == restored.last_settlement
    with np.load(brain.save(tmp_path / "continued.npz"), allow_pickle=False) as first:
        with np.load(restored.save(tmp_path / "restored.npz"), allow_pickle=False) as second:
            assert set(first.files) == set(second.files)
            for name in first.files:
                np.testing.assert_array_equal(first[name], second[name], err_msg=name)


@pytest.mark.parametrize("value", [None, True, "0.5", [0.5], -0.1, np.nan, np.inf])
def test_resting_bias_checkpoint_rejects_invalid_initialization_metadata(tmp_path, value) -> None:
    brain = cd.Brain.compose(2, 2, modules=(4,), resting_bias=0.5)
    with np.load(brain.save(tmp_path / "valid.npz"), allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    metadata = json.loads(str(data["generic"]))
    metadata["resting_bias"] = value
    data["generic"] = np.array(json.dumps(metadata))
    np.savez(tmp_path / "invalid.npz", **data)
    with pytest.raises(ValueError, match="resting_bias.*finite nonnegative real scalar"):
        cd.Brain.load(tmp_path / "invalid.npz")


def test_checkpoint_without_resting_bias_metadata_keeps_stored_bias(tmp_path) -> None:
    brain = cd.Brain.compose(2, 2, modules=(4,), resting_bias=0.5)
    with np.load(brain.save(tmp_path / "current.npz"), allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    metadata = json.loads(str(data["generic"]))
    del metadata["resting_bias"]
    data["generic"] = np.array(json.dumps(metadata))
    np.savez(tmp_path / "prior.npz", **data)
    restored = cd.Brain.load(tmp_path / "prior.npz")
    assert restored.resting_bias == 0.0
    np.testing.assert_array_equal(restored.brain.bias, brain.brain.bias)
