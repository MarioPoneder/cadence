"""The child's capabilities, each out of simpler components already in the core.

A child learning games keeps a permanent world model (the synapses), sees one frame and keeps
each moment reverberating for seconds before it vanishes (the settled state and its
afterglow), and changes the world model only when a moment carried a positive or negative
signal, in proportion (the eligibility trace times a dopamine that is quiet otherwise).
These tests are the small experiments for each capability."""

from __future__ import annotations

import dataclasses

import numpy as np

import cadence as cd


def _bandit_drive(
    rng: np.random.Generator, batch: int, connectome: cd.Connectome, learner: cd.Learner
) -> tuple[np.ndarray, np.ndarray]:
    """Two contexts, two actions; action 0 pays in context 0 and action 1 in context 1."""
    context = rng.integers(0, 2, size=batch)
    x = np.zeros((batch, 4))
    x[np.arange(batch), context * 2] = 1.0
    x[np.arange(batch), context * 2 + 1] = 1.0
    return learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 4)))), context


def _actor(connectome: cd.Connectome, **config: float) -> tuple[cd.ActorCritic, cd.Learner]:
    learner = cd.Learner(
        cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
    )
    return cd.ActorCritic(
        learner, connectome.populations["hidden"], cd.ActorCriticConfig(**config), seed=0
    ), learner


def test_the_trace_credits_an_action_paid_three_moments_later() -> None:
    """A press whose consequence comes later: the eligibility trace carries the moment to the
    signal. With no trace only the last moment is credited and nothing is learned."""
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)

    def hit_rate(lam: float) -> float:
        rng = np.random.default_rng(0)
        ac, learner = _actor(
            connectome,
            gamma=1.0,
            lam=lam,
            eta=1.0,
            eta_critic=0.0,
            dopamine_center=0.0,
            dopamine_cap=0.0,
        )
        batch = 32
        blank = learner.brain.stimulus_levels(np.zeros((batch, connectome.n)))
        drive, context = _bandit_drive(rng, batch, connectome, learner)
        for _ in range(200):
            action = ac.act(drive)  # the moment that matters
            ac.learn(np.zeros(batch), np.zeros(batch, dtype=bool), blank)
            ac.act(blank)  # two moments of nothing, with presses of their own
            ac.learn(np.zeros(batch), np.zeros(batch, dtype=bool), blank)
            ac.act(blank)
            reward = (action == context).astype(float)  # paid now, three moments on
            drive, context = _bandit_drive(rng, batch, connectome, learner)
            ac.learn(reward, np.ones(batch, dtype=bool), drive)
        ac.reset()
        x, ctx = _bandit_drive(rng, 400, connectome, learner)
        return float((ac.act(x, greedy=True) == ctx).mean())

    with_trace, without = hit_rate(0.9), hit_rate(0.0)
    print(f"delayed credit: hit rate {with_trace:.2f} with the trace, {without:.2f} without")
    assert with_trace >= 0.8
    assert without <= 0.65


def test_the_dopamine_is_quiet_for_the_usual_and_speaks_for_more_or_less() -> None:
    """The child's rule: no critic, the reward in its own units less the usual level, nothing
    within the usual; a reward that is what it always was moves no synapse, one that is missing
    or larger does, in proportion."""
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    ac, learner = _actor(
        connectome,
        gamma=0.0,
        lam=0.0,
        eta=0.5,
        eta_critic=0.0,
        dopamine_center=0.5,
        center_scale=False,
        dopamine_floor=1.0,
        dopamine_cap=10.0,
    )
    rng = np.random.default_rng(3)
    drive, _ = _bandit_drive(rng, 8, connectome, learner)
    ones = np.ones(8)
    for _ in range(30):  # the usual: a reward of one every moment
        ac.act(drive)
        ac.learn(ones, np.zeros(8, dtype=bool), drive)

    def change(reward: float) -> float:
        before = learner.brain.weights.copy()
        ac.act(drive)
        ac.learn(np.full(8, reward), np.zeros(8, dtype=bool), drive)
        return float(np.abs(learner.brain.weights - before).sum())

    usual, missing, bigger = change(1.0), change(0.0), change(11.0)
    print(
        f"synapse change for the usual {usual:.4f}, a missing reward {missing:.4f}, a reward ten larger {bigger:.4f}"
    )
    assert usual == 0.0
    assert missing > 0.0
    assert bigger > missing


def _one_moment_ago(
    rng: np.random.Generator,
    connectome: cd.Connectome,
    glow: cd.Afterglow,
    tie: np.ndarray | None,
    v: int,
    background: int,
) -> float:
    """Predict the symbol seen one moment ago from one moment's input, ``background`` neurons always on."""
    streams, length = 64, 60
    seq = rng.integers(0, v, (streams, length))
    cfg = cd.LearnerConfig(
        eta=2.0, beta=0.1, temperature=0.1, tolerance=3e-3, nudged_steps=12, free_steps=60
    )
    learner = cd.Learner(
        cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cfg,
        tie_groups=tie,
    )

    def run(learn: bool) -> float:
        glow.reset(streams)
        hits = total = 0
        for t in range(1, length):
            drive = np.zeros((streams, connectome.n))
            drive[np.arange(streams), seq[:, t]] = 1.0
            drive[:, v : v + background] = 1.0  # the background, the same every moment
            target = seq[:, t - 1]
            drive = glow.stimulate(drive)
            if learn:
                learned, _ = learner.step(drive, target)
                free = learned.free
            else:
                free = learner.free(drive)
            if t > 5:
                hits += int(
                    (free.activation[:, learner.output_index].argmax(axis=1) == target).sum()
                )
                total += streams
            glow.update(free)
        return hits / total

    for epoch in range(6):
        learner.config = dataclasses.replace(cfg, eta=2.0 * 0.8**epoch)
        run(learn=True)
    learner.config = dataclasses.replace(cfg, tolerance=1e-4)
    return run(learn=False)


def test_the_afterglow_remembers_a_cue_against_a_static_background() -> None:
    """Predict the symbol seen one moment ago, from one moment's input, while twelve neurons of
    background are always on. The echo of the interpretation keeps everything and the
    background swamps the cue; the afterglow keeps what changed. Two afterglows: of the
    interpretation (the hidden neurons) and of the picture itself (an afterimage of the
    input neurons, a fading picture of what changed on the retina)."""
    v, background = 4, 12
    scores = {}
    for name, focus in (
        ("echo", 0.0),
        ("afterglow of the interpretation", 1.0),
        ("afterglow of the interpretation, focus 2", 2.0),
    ):
        w, tie = cd.stateful(v + background, 1, 4, 24, v, seed=0)
        w = w.with_populations(afterglow=w.populations["context"])
        scores[name] = _one_moment_ago(
            np.random.default_rng(0), w, cd.Afterglow(w, decay=0.5, focus=focus), tie, v, background
        )
    n = v + background
    picture = cd.Genome(
        regions=(
            cd.Region("input", n),
            cd.Region("afterglow", n),
            cd.Region("hidden", 24),
            cd.Region("output", v),
        ),
        projections=(
            cd.Projection("input", "hidden", density=1.0, sign=0.0, scale=1.0, reciprocal=False),
            cd.Projection(
                "afterglow", "hidden", density=1.0, sign=0.0, scale=1.0, reciprocal=False
            ),
            cd.Projection("hidden", "output", density=1.0, sign=0.0, scale=1.0),
        ),
        label="afterimage",
    )
    for name, focus in (
        ("afterimage of the picture, no focus", 0.0),
        ("afterimage of the picture", 1.0),
    ):
        w = cd.develop(picture, seed=0)
        scores[name] = _one_moment_ago(
            np.random.default_rng(0),
            w,
            cd.Afterglow(w, decay=0.5, focus=focus, source="input"),
            None,
            v,
            background,
        )
    print(
        "one moment ago against a static background (chance 0.25): "
        + ", ".join(f"{k} {s:.2f}" for k, s in scores.items())
    )
    assert max(scores.values()) > 0.5
    assert scores["afterimage of the picture"] >= scores["echo"]


def test_a_capped_activity_change_carries_the_moment_before_and_a_full_one_forgets_it() -> None:
    """Settling as the memory: settled for a few steps from the previous equilibrium, the
    state still holds the moment before; settled to convergence, it does not."""
    w, _ = cd.stateful(4, 1, 4, 24, 4, seed=0)
    brain = cd.Brain(w, cd.learning_neuron_model(dt=1.0))
    hidden = list(w.populations["hidden"])
    a = np.zeros((1, w.n))
    a[:, 0] = 1.0
    b = np.zeros((1, w.n))
    b[:, 1] = 1.0
    eq_a = brain.settle_batch(a, steps=300, tolerance=1e-5)
    cold_b = brain.settle_batch(b, steps=300, tolerance=1e-5)
    partial = brain.settle_batch(b, steps=3, state=eq_a)
    full = brain.settle_batch(b, steps=300, state=eq_a, tolerance=1e-5)
    d = lambda s: float(np.abs(s.activation[:, hidden] - eq_a.activation[:, hidden]).sum())  # noqa: E731
    print(
        f"distance from the moment before: partial settling {d(partial):.4f}, full settling {d(full):.4f}, cold {d(cold_b):.4f}"
    )
    assert d(partial) < d(cold_b)
    assert abs(d(full) - d(cold_b)) < 0.05 * d(cold_b) + 1e-6


def test_the_eligibility_weighted_by_the_afterimage_credits_the_cue_not_the_background() -> None:
    """A context seen for one moment with the press it calls for, paid three moments later, and
    twelve neurons of background always on: the synapses out of the background carry eligibility
    at every moment, the synapses out of the context only at the one that mattered. With the
    eligibility weighted by the afterimage (what is still ringing is what gets written) the
    signal lands on the context's synapses. Both learn it here: the background's eligibility is
    noise that averages out over rounds, and the weighting is a faster start on some seeds
    (1.00 against 0.57 at fifty rounds on one, no difference on another), not a gain the
    small experiment can promise."""
    v, background, hidden, batch, rounds = 4, 12, 8, 32, 100
    n_in = v + background
    genome = cd.Genome(
        regions=(
            cd.Region("input", n_in),
            cd.Region("afterglow", n_in),
            cd.Region("hidden", hidden),
            cd.Region("output", 2),
        ),
        projections=(
            cd.Projection("input", "hidden", density=1.0, sign=0.0, scale=1.0, reciprocal=False),
            cd.Projection(
                "afterglow", "hidden", density=1.0, sign=0.0, scale=1.0, reciprocal=False
            ),
            cd.Projection("hidden", "output", density=1.0, sign=0.0, scale=1.0),
        ),
        label="cue",
    )

    def hit_rate(weighted: bool) -> float:
        rng = np.random.default_rng(0)
        connectome = cd.develop(genome, seed=0)
        learner = cd.Learner(
            cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
            connectome.populations["output"],
            cd.LearnerConfig(beta=0.1, eta=1.0, temperature=0.2, tolerance=3e-3, nudged_steps=12),
        )
        ac = cd.ActorCritic(
            learner,
            connectome.populations["hidden"],
            cd.ActorCriticConfig(
                gamma=1.0, lam=0.9, eta=1.0, eta_critic=0.0, dopamine_center=0.0, dopamine_cap=0.0
            ),
            seed=0,
        )
        glow = cd.Afterglow(connectome, decay=0.7, focus=1.0, source="input")
        glow.reset(batch)

        def raw(context: np.ndarray | None) -> np.ndarray:
            d = np.zeros((batch, connectome.n))
            d[:, v:n_in] = 1.0  # the background, every moment
            if context is not None:
                d[np.arange(batch), context * 2] = 1.0
                d[np.arange(batch), context * 2 + 1] = 1.0
            return d

        def moment(drive: np.ndarray) -> np.ndarray:
            action = ac.act(drive)
            glow.update(ac._free)
            ac.salience = glow.ringing() if weighted else None
            return action

        context = rng.integers(0, 2, size=batch)
        drive = glow.stimulate(raw(context))
        for _ in range(rounds):
            action = moment(drive)  # the moment that matters, with the context in view
            drive = glow.stimulate(raw(None))
            ac.learn(np.zeros(batch), np.zeros(batch, dtype=bool), drive)
            moment(drive)  # two moments of background only, with presses of their own
            drive = glow.stimulate(raw(None))
            ac.learn(np.zeros(batch), np.zeros(batch, dtype=bool), drive)
            moment(drive)
            reward = (action == context).astype(float)
            context = rng.integers(0, 2, size=batch)
            drive = glow.stimulate(
                raw(context)
            )  # the afterimage runs on: the background has long stopped ringing, the new context rings
            ac.learn(reward, np.ones(batch, dtype=bool), drive)
        ac.reset()
        ac.salience = None
        hits = []
        for _ in range(8):
            ctx = rng.integers(0, 2, size=batch)
            hits.append((ac.act(glow.stimulate(raw(ctx)), greedy=True) == ctx).mean())
            glow.update(ac._free)
            ac.reset()
        return float(np.mean(hits))

    plain, weighted = hit_rate(False), hit_rate(True)
    print(
        f"a cue paid three moments later against twelve neurons of background: hit rate {plain:.2f} with the plain trace, {weighted:.2f} weighted by the afterimage"
    )
    assert weighted > 0.9
    assert plain > 0.9


def test_the_valence_is_one_element() -> None:
    """The Valence alone: level, units, floor, cap; and the agent's dopamine is the same object."""
    v = cd.Valence(level=0.5, floor=1.0, cap=10.0, units=True)
    for _ in range(30):
        v(np.array([1.0, 1.0]))
    assert np.array_equal(v(np.array([1.0, 1.0])), [0.0, 0.0])  # the usual: quiet
    missing, bigger = v(np.array([0.0, 0.0])), v(np.array([11.0, 11.0]))
    assert missing[0] < 0 < bigger[0] and abs(bigger[0]) > abs(missing[0])
    assert v(np.array([1000.0, 1000.0]))[0] == 10.0  # the cap
    connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
    ac, _ = _actor(
        connectome, dopamine_center=0.5, dopamine_floor=1.0, dopamine_cap=10.0, center_scale=False
    )
    assert isinstance(ac.valence, cd.Valence) and ac.valence.floor == 1.0 and ac.valence.units
