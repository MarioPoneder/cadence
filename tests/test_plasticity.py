from __future__ import annotations

import numpy as np

import cadence as cd
from cadence.learning import SCALE_CAP


def _contextual_bandit(rng: np.random.Generator, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Two contexts, two actions; action 0 pays in context 0 and action 1 in context 1."""
    context = rng.integers(0, 2, size=batch)
    x = np.zeros((batch, 4))
    x[np.arange(batch), context * 2] = 1.0
    x[np.arange(batch), context * 2 + 1] = 1.0
    return x, context


def test_actor_critic_learns_a_contextual_bandit_from_dopamine() -> None:
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    learner = cd.Learner(
        cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3),
        seed=0,
    )
    rng = np.random.default_rng(0)
    batch = 32

    def drive_of(x: np.ndarray) -> np.ndarray:
        return learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4))))

    def hit_rate() -> float:
        x, context = _contextual_bandit(rng, 200)
        ac.reset()
        action = ac.act(drive_of(x), greedy=True)
        return float((action == context).mean())

    before = hit_rate()
    ac.reset()
    x, context = _contextual_bandit(rng, batch)
    drive = drive_of(x)
    for _ in range(150):
        action = ac.act(drive)
        reward = (action == context).astype(float)
        x, context = _contextual_bandit(rng, batch)
        drive = drive_of(x)
        ac.learn(reward, np.ones(batch, dtype=bool), drive)
    after = hit_rate()
    assert after >= 0.9 and after > before


def test_traces_reset_on_done_and_updates_are_local() -> None:
    connectome = cd.layered(4, 6, 2, density=1.0, seed=1)
    learner = cd.Learner(
        cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(eta=1.0),
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(gamma=0.9, lam=0.5, eta=0.1),
        seed=1,
    )
    rng = np.random.default_rng(1)
    x, _ = _contextual_bandit(rng, 4)
    drive = learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4))))
    ac.act(drive)
    ac.learn(np.zeros(4), np.array([True, False, False, True]), drive)
    assert ac.trace is not None
    assert np.all(ac.trace[[0, 3]] == 0.0)
    assert np.any(ac.trace[[1, 2]] != 0.0)
    # every synapse's step is delta times its own trace: recompute one step by hand
    learner.reciprocal = False
    learner.reverse[:] = -1
    ac.act(drive)
    kind, plus, minus, value = ac._pending
    plus, minus = plus.activation, minus.activation  # the pending phases are states
    w = connectome
    contrast = (plus[:, w.pre] * plus[:, w.post] - minus[:, w.pre] * minus[:, w.post]) / (
        2.0 * learner.config.beta
    )
    before = learner.brain.efficacy.copy()
    trace_before = ac.trace.copy()
    w_critic, b_critic = ac.w_critic.copy(), ac.b_critic
    reward = np.array([1.0, 0.0, 0.5, 0.0])
    ac.learn(reward, np.zeros(4, dtype=bool), drive)
    next_value = (
        ac._free.activation[:, ac.critic_index] @ w_critic + b_critic
    )  # the critic as it was
    delta = reward + 0.9 * next_value - value
    expected_trace = 0.9 * 0.5 * trace_before + contrast
    expected = 0.1 * (delta[:, None] * expected_trace).mean(axis=0)
    got = learner.brain.efficacy - before
    clipped = np.abs(before + expected) > SCALE_CAP
    assert np.allclose(got[~clipped], expected[~clipped])


def test_grouped_softmax_nudge_agrees_between_kernels_and_bins_learn_a_continuous_bandit() -> None:

    from cadence import brain as S

    bins = cd.Bins(dims=2, size=5)
    connectome = cd.layered(4, 8, bins.dims * bins.size, density=1.0, seed=3)
    brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0))
    rng = np.random.default_rng(3)
    drive = brain.stimulus_levels(np.pad(rng.random((6, 4)), ((0, 0), (0, connectome.n - 4))))
    out = np.asarray(connectome.populations["output"])
    target = np.zeros((6, connectome.n))
    target[:, out[[0, 7]]] = 1.0
    mask = np.zeros(connectome.n)
    mask[out] = 1.0
    nudge = cd.Nudge(
        target, mask, 0.1, softmax_temperature=0.2, groups=bins.groups(out, connectome.n)
    )
    free = brain.settle_batch(drive, steps=60, tolerance=3e-3)
    fused = brain.settle_batch(drive, steps=12, state=free, nudge=nudge, tolerance=3e-3)
    was = S._FUSED
    S._FUSED = False
    try:
        plain = brain.settle_batch(drive, steps=12, state=free, nudge=nudge, tolerance=3e-3)
    finally:
        S._FUSED = was
    assert np.abs(fused.activation - plain.activation).max() < 1e-12

    learner = cd.Learner(
        brain,
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0, eta_critic=0.3),
        seed=3,
        population=bins,
    )
    wanted = np.array([[0.5, -1.0], [-0.5, 1.0]])

    def batch(k: int) -> tuple[np.ndarray, np.ndarray]:
        c = rng.integers(0, 2, k)
        x = np.zeros((k, 4))
        x[np.arange(k), 2 * c] = 1.0
        x[np.arange(k), 2 * c + 1] = 1.0
        return brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4)))), c

    def error() -> float:
        d, c = batch(200)
        ac.reset()
        return float(np.abs(ac.act(d, greedy=True) - wanted[c]).mean())

    before = error()
    ac.reset()
    d, c = batch(32)
    for _ in range(200):
        a = ac.act(d)
        reward = -((a - wanted[c]) ** 2).sum(axis=1)
        d, c = batch(32)
        ac.learn(reward, np.ones(32, dtype=bool), d)
    after = error()
    assert after < 0.2 and after < before


def test_one_stream_learns_the_same_on_the_device_as_on_the_host() -> None:
    """With one stream settled on the torch kernel, the trace and the step stay on the device
    and the synapses end where the host path puts them."""
    import pytest

    if "torch" not in cd.available_backends():
        pytest.skip("no torch")
    connectome = cd.layered(6, 10, 4, density=1.0, seed=1)
    config = cd.LearnerConfig(
        beta=0.1, eta=1.0, temperature=0.3, tolerance=1e-6, nudged_steps=30, free_steps=200
    )
    ac_config = cd.ActorCriticConfig(
        gamma=0.9, lam=0.8, eta=0.3, eta_bias=0.03, eta_critic=0.1, dopamine_cap=1.0
    )
    rng = np.random.default_rng(3)
    drives = [np.concatenate([rng.random(6), np.zeros(14)])[None] for _ in range(6)]
    rewards = [0.5, -0.2, 1.0, 0.0, 0.3, -1.0]
    results = []
    for backend in ("cpu", "torch"):
        brain = cd.Brain(
            connectome,
            cd.learning_neuron_model(dt=1.0),
            backend=backend,
            device="cpu" if backend == "torch" else None,
            precision="float64" if backend == "torch" else None,
        )  # type: ignore[arg-type]
        learner = cd.Learner(brain, connectome.populations["output"], config, slots=2)
        ac = cd.ActorCritic(
            learner,
            connectome.populations["hidden"],
            ac_config,
            seed=0,
            population=cd.Bins(dims=2, size=2),
        )
        actions = []
        for k in range(5):
            actions.append(ac.act(drives[k]).copy())
            ac.learn(np.array([rewards[k]]), np.array([k == 3]), drives[k + 1])
        results.append(
            (
                np.stack(actions),
                np.asarray(ac.learner.brain.efficacy),
                np.asarray(ac.learner.brain.bias),
                ac.w_critic.copy(),
            )
        )
    (a_host, s_host, b_host, c_host), (a_dev, s_dev, b_dev, c_dev) = results
    assert np.array_equal(a_host, a_dev)
    assert np.allclose(s_host, s_dev, atol=1e-6) and np.abs(s_host - connectome.sign).max() > 1e-4
    assert np.allclose(b_host, b_dev, atol=1e-6)
    assert np.allclose(c_host, c_dev, atol=1e-6)


def test_a_batch_of_streams_learns_the_same_on_the_device_as_on_the_host() -> None:
    import pytest

    if "torch" not in cd.available_backends():
        pytest.skip("no torch")
    connectome = cd.layered(6, 10, 4, density=1.0, seed=2)
    config = cd.LearnerConfig(
        beta=0.1, eta=1.0, temperature=0.3, tolerance=1e-6, nudged_steps=30, free_steps=200
    )
    ac_config = cd.ActorCriticConfig(
        gamma=0.9, lam=0.8, eta=0.3, eta_bias=0.03, eta_critic=0.1, dopamine_cap=1.0
    )
    rng = np.random.default_rng(5)
    batch = 3
    drives = [
        np.concatenate([rng.random((batch, 6)), np.zeros((batch, 14))], axis=1) for _ in range(6)
    ]
    rewards = [rng.normal(size=batch) for _ in range(6)]
    dones = [np.zeros(batch, dtype=bool) for _ in range(6)]
    dones[2][1] = True
    results = []
    for backend in ("cpu", "torch"):
        brain = cd.Brain(
            connectome,
            cd.learning_neuron_model(dt=1.0),
            backend=backend,
            device="cpu" if backend == "torch" else None,
            precision="float64" if backend == "torch" else None,
        )  # type: ignore[arg-type]
        learner = cd.Learner(brain, connectome.populations["output"], config, slots=2)
        ac = cd.ActorCritic(
            learner,
            connectome.populations["hidden"],
            ac_config,
            seed=0,
            population=cd.Bins(dims=2, size=2),
        )
        actions = []
        for k in range(5):
            actions.append(ac.act(drives[k]).copy())
            ac.learn(rewards[k], dones[k], drives[k + 1])
        results.append(
            (
                np.stack(actions),
                np.asarray(ac.learner.brain.efficacy),
                np.asarray(ac.learner.brain.bias),
                ac.w_critic.copy(),
            )
        )
    (a_host, s_host, b_host, c_host), (a_dev, s_dev, b_dev, c_dev) = results
    assert np.array_equal(a_host, a_dev)
    assert np.allclose(s_host, s_dev, atol=1e-6) and np.abs(s_host - connectome.sign).max() > 1e-4
    assert np.allclose(b_host, b_dev, atol=1e-6)
    assert np.allclose(c_host, c_dev, atol=1e-6)


def test_actor_critic_centres_the_dopamine_per_stream() -> None:
    """Two streams with rewards of different sizes each keep their own level: the agent's
    valence is per stream, so the small-reward stream is not always below the mean."""
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    learner = cd.Learner(
        cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=0.1, eta_critic=0.0, dopamine_center=0.5),
        seed=0,
    )
    rng = np.random.default_rng(1)
    x = rng.random((2, 4))
    drive = np.pad(x, ((0, 0), (0, connectome.n - 4)))
    rewards = np.array([10.0, 0.1])  # stream 0 is paid a hundred times stream 1
    ac.act(drive)
    for _ in range(20):
        ac.learn(rewards + rng.normal(0.0, 0.01, size=2), np.zeros(2, dtype=bool), drive)
        ac.act(drive)
    assert isinstance(ac.delta_mean, np.ndarray) and ac.delta_mean.shape == (2,)
    assert ac.delta_mean[0] > ac.delta_mean[1] + 5.0  # each stream's level is its own reward's


def test_actor_critic_dopamine_floor_is_quiet_for_the_usual_reward() -> None:
    """With a floor, a reward at its usual level gives no dopamine and no step; a surprise does."""
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    learner = cd.Learner(
        cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(
            gamma=0.0, lam=0.0, eta=0.1, eta_critic=0.0, dopamine_center=0.5, dopamine_floor=1.0
        ),
        seed=0,
    )
    rng = np.random.default_rng(2)
    drive = np.pad(rng.random((2, 4)), ((0, 0), (0, connectome.n - 4)))
    ac.act(drive)
    for _ in range(30):
        ac.learn(np.array([1.0, 1.0]), np.zeros(2, dtype=bool), drive)
        ac.act(drive)
    before = learner.brain.weights.copy()
    ac.learn(np.array([1.0, 1.0]), np.zeros(2, dtype=bool), drive)  # the usual reward: quiet
    ac.act(drive)
    assert np.array_equal(learner.brain.weights, before)
    ac.learn(np.array([50.0, 50.0]), np.zeros(2, dtype=bool), drive)  # a surprise: a step
    assert not np.array_equal(learner.brain.weights, before)


def test_actor_critic_centre_without_the_scale_keeps_the_rewards_size() -> None:
    """Unscaled, the centred dopamine is in the reward's own units: a big surprise is big."""
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)

    def make(scale: bool) -> cd.ActorCritic:
        learner = cd.Learner(
            cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
            connectome.populations["output"],
            cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
        )
        return cd.ActorCritic(
            learner,
            connectome.populations["hidden"],
            cd.ActorCriticConfig(
                gamma=0.0,
                lam=0.0,
                eta=0.0,
                eta_critic=0.0,
                dopamine_center=0.5,
                dopamine_cap=0.0,
                center_scale=scale,
            ),
            seed=0,
        )

    scaled, raw = make(True), make(False)
    for ac in (scaled, raw):
        for _ in range(20):
            ac._centre(np.array([1.0, 1.0]))
    big_scaled = scaled._centre(np.array([11.0, 11.0]))
    big_raw = raw._centre(np.array([11.0, 11.0]))
    # the running mean and scale take the new reward in first (half of it at this forgetting):
    # a reward ten above the usual reads five in the reward's own units, and in scales it
    # reads root two whatever its size, since the surprise inflates its own scale
    assert abs(big_raw[0] - 5.0) < 0.1
    assert abs(big_scaled[0] - np.sqrt(2.0)) < 0.05
    assert (
        abs(raw._centre(np.array([101.0, 101.0]))[0] - 47.5) < 0.5
    )  # ten times the surprise, ten times the signal


def test_padding_cannot_teach_the_actor_critic_or_change_the_active_update() -> None:
    import pytest

    connectome = cd.layered(2, 4, 2, seed=4)
    for backend in ("cpu", "torch"):
        if backend == "torch" and "torch" not in cd.available_backends():
            continue
        results = []
        for batch in (1, 2):
            dynamics = cd.Brain(
                connectome,
                cd.learning_neuron_model(dt=1.0),
                backend=backend,
                device="cpu" if backend == "torch" else None,
                precision="float64" if backend == "torch" else None,
            )
            learner = cd.Learner(
                dynamics,
                connectome.populations["output"],
                cd.LearnerConfig(tolerance=1e-8, free_steps=200, nudged_steps=40),
            )
            actor = cd.ActorCritic(
                learner,
                connectome.populations["hidden"],
                cd.ActorCriticConfig(eta=0.1, dopamine_center=0.9),
                seed=3,
            )
            drive = np.zeros((batch, connectome.n))
            drive[:, :2] = [0.3, 0.7]
            actor.act(drive)
            pending = actor._pending
            with pytest.raises(ValueError, match="observed"):
                actor.learn(
                    np.ones(batch), np.zeros(batch, bool), drive, observed=np.zeros(batch, bool)
                )
            assert actor._pending is pending
            observed = np.arange(batch) == 0
            actor.learn(
                np.array([1.0, 999.0])[:batch], np.ones(batch, bool), drive, observed=observed
            )
            assert actor._pending is None
            if actor.trace is not None:
                assert not actor.trace.any()
            if actor._trace_device is not None:
                assert not actor._trace_device[0].any().item()
            results.append(
                (
                    np.asarray(learner.brain.efficacy),
                    np.asarray(learner.brain.bias),
                    actor.w_critic,
                    actor.b_critic,
                )
            )
        for alone, padded in zip(results[0], results[1], strict=True):
            assert np.allclose(alone, padded, atol=1e-8), backend


def test_unobserved_rewards_do_not_enter_shared_or_per_stream_valence() -> None:
    for per_stream in (False, True):
        alone = cd.Valence(level=0.9, per_stream=per_stream)
        padded = cd.Valence(level=0.9, per_stream=per_stream)
        for reward in (1.0, -0.5, 2.0):
            expected = alone(np.array([reward]))
            actual = padded(np.array([reward, 10000.0]), observed=np.array([True, False]))
            assert actual[1] == 0 and np.allclose(actual[:1], expected)
        before = np.array(padded.mean, copy=True)
        assert not padded(np.array([1.0, 2.0]), observed=np.zeros(2, bool)).any()
        assert np.array_equal(before, padded.mean)


def test_scale_cap_bounds_the_plastic_efficacy_and_is_validated() -> None:
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0))
    learner = cd.Learner(brain, connectome.populations["output"], cd.LearnerConfig(scale_cap=0.5))
    learner.apply(np.full(connectome.synapses, 10.0), np.zeros(connectome.n))
    assert float(np.abs(learner.brain.efficacy).max()) <= 0.5
    wide = cd.Learner(
        cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)), connectome.populations["output"]
    )
    wide.apply(np.full(connectome.synapses, 10.0), np.zeros(connectome.n))
    assert float(np.abs(wide.brain.efficacy).max()) == SCALE_CAP
    assert cd.LearnerConfig().to_dict()["scale_cap"] == SCALE_CAP
    for bad in (0.0, -1.0, float("nan")):
        try:
            cd.LearnerConfig(scale_cap=bad)
        except ValueError:
            continue
        raise AssertionError(f"scale_cap={bad} was accepted")


def test_the_critic_learns_from_the_raw_error_whenever_the_dopamine_is_centred() -> None:
    assert cd.ActorCriticConfig().critic_target == "modulated"
    assert cd.ActorCriticConfig(dopamine_center=0.5).critic_target == "td"
    assert (
        cd.ActorCriticConfig(dopamine_center=0.5, critic_signal="modulated").critic_target
        == "modulated"
    )
    assert cd.ActorCriticConfig(critic_signal="td").critic_target == "td"
    assert cd.ActorCriticConfig().to_dict()["critic_signal"] == "auto"


def test_the_report_reads_the_outputs_saturation_and_the_traces_size() -> None:
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    plastic = np.zeros(connectome.synapses, dtype=bool)
    plastic[: connectome.synapses // 2] = True
    learner = cd.Learner(
        cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
        plastic_synapses=plastic,
    )
    ac = cd.ActorCritic(
        learner,
        connectome.populations["hidden"],
        cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=1.0),
        seed=0,
    )
    rng = np.random.default_rng(1)
    x, context = _contextual_bandit(rng, 8)
    drive = learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4))))
    action = ac.act(drive)
    report = ac.learn((action == context).astype(float), np.ones(8, dtype=bool), drive)
    assert 0.0 <= report["saturation"] <= 1.0
    assert report["trace"] >= 0.0
    assert report["trace"] == float(np.abs(ac.trace[:, plastic]).mean())
