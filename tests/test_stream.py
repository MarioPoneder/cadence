"""Owned state as a stimulus: the stateful connectome, the echo, and a task that needs carried state."""

from __future__ import annotations

import dataclasses

import numpy as np

import cadence as cd


def test_stateful_connectome_has_a_context_source_range_into_the_hidden_neurons() -> None:
    w, tie = cd.stateful(5, 2, 3, 4, 5, seed=0)
    assert w.n == 2 * 5 + 4 + 2 * 3 + 4 + 5
    ctx, hid = w.populations["context"], w.populations["hidden"]
    assert len(ctx) == len(hid) == 4 and len(tie) == w.synapses
    degree = w.in_degree()
    assert degree[list(ctx)].sum() == 0  # context neurons hear nothing
    assert all(degree[i] >= 4 for i in hid)  # every hidden neuron hears every context neuron
    brain = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0))
    lay = brain.layout
    assert 1 in lay.sources()  # the context range is a source of the block transport
    assert (
        cd.conformance(brain, list(w.populations["input"])[:2], steps=20)["max_abs_deviation"]
        < 1e-12
    )


def test_echo_decays_toward_the_hidden_activation_and_enters_the_clamp() -> None:
    w, _ = cd.stateful(3, 1, 2, 3, 3, seed=1)
    brain = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0))
    echo = cd.Echo(w, decay=0.5)
    drive = np.zeros((2, w.n))
    drive[:, 0] = 1.0
    state = brain.settle_batch(echo.stimulate(drive), steps=30)
    echo.update(state)
    h = state.activation[:, list(w.populations["hidden"])]
    assert np.allclose(echo.trace, 0.5 * h)
    echo.update(state)
    assert np.allclose(echo.trace, 0.75 * h)
    stimulated = echo.stimulate(drive)
    assert np.allclose(stimulated[:, list(w.populations["context"])], echo.trace)
    assert stimulated[:, 0].sum() == 2.0  # the input stimulus is untouched


def test_trace_remembers_previous_activation_when_state_storage_is_reused() -> None:
    connectome = cd.Connectome.from_synapses(
        4, pre=[], post=[], populations={"hidden": [0, 1], "context": [2, 3]}
    )
    trace = cd.Trace(connectome, decay=0.0, focus=1.0)
    activation = np.array([[0.1, 0.7, 0.0, 0.0]])
    state = cd.BrainState(
        v=activation.copy(), activation=activation, adaptation=np.zeros_like(activation), steps=1
    )
    trace.update(state)
    activation[0, 0] = 0.5
    np.testing.assert_array_equal(trace.last, [[0.1, 0.7]])
    trace.update(state)
    moved = np.array([[0.4, 0.0]])
    expected = moved / (moved.mean(axis=1, keepdims=True) + 1e-9) * [[0.5, 0.7]]
    np.testing.assert_allclose(trace.trace, expected)


def _carried_state_accuracy(seed: int) -> float:
    """Predict the symbol seen one input ago from a window of one: impossible without state."""
    rng = np.random.default_rng(seed)
    v, streams, length = 4, 64, 60
    seq = rng.integers(0, v, (streams, length))
    w, tie = cd.stateful(v, 1, 4, 24, v, seed=seed)
    cfg = cd.LearnerConfig(
        eta=2.0, beta=0.1, temperature=0.1, tolerance=3e-3, nudged_steps=12, free_steps=60
    )
    learner = cd.Learner(
        cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0)), w.populations["output"], cfg, tie_groups=tie
    )
    echo = cd.Echo(w, decay=0.0)  # the previous equilibrium, undiluted

    def run(learn: bool) -> float:
        echo.reset(streams)
        hits = total = 0
        for t in range(1, length):
            drive = np.zeros((streams, w.n))
            drive[np.arange(streams), seq[:, t]] = 1.0
            target = seq[:, t - 1]
            drive = echo.stimulate(drive)
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
            echo.update(free)
        return hits / total

    for epoch in range(6):
        learner.config = dataclasses.replace(cfg, eta=2.0 * 0.8**epoch)
        run(learn=True)
    learner.config = dataclasses.replace(cfg, tolerance=1e-4)
    return run(learn=False)


def test_carried_state_learns_what_no_window_can_see() -> None:
    # Chance is 0.25 and a window of one gives exactly chance. A single short trajectory at
    # this loose tolerance is sensitive to rounding (one seed moves from 0.85 to 0.56 between
    # two exact contrast formulas), so two seeds are scored: each clearly above chance, and
    # their mean well above it.
    scores = [_carried_state_accuracy(seed) for seed in (0, 2)]
    assert all(score > 0.45 for score in scores), scores
    assert sum(scores) / len(scores) > 0.6, scores


def test_fast_synapses_bind_a_cue_to_what_was_active_and_fade() -> None:
    # neurons 0..3 are cues, 4..6 are contents; a stream that saw cue 1 with content 6 recalls 6
    connectome = cd.layered(4, 2, 3, density=1.0, seed=0)
    fast = cd.FastSynapses(np.arange(4), np.asarray(connectome.populations["output"]), decay=0.5)
    fast.reset(2)
    s = np.zeros((2, connectome.n))
    out = np.asarray(connectome.populations["output"])
    s[0, 1] = 1.0
    s[0, out[2]] = 1.0  # stream 0: cue 1 with content 2
    s[1, 3] = 1.0
    s[1, out[0]] = 1.0  # stream 1: cue 3 with content 0
    state = cd.BrainState(v=s, activation=s, adaptation=np.zeros_like(s), steps=1)
    fast.update(state, np.array([True, True]))
    drive = np.zeros((2, connectome.n))
    drive[0, 1] = 1.0
    drive[1, 3] = 1.0
    read = fast.read(drive)
    assert read[0].argmax() == 2 and read[1].argmax() == 0
    assert np.isclose(read[0, 2], 1.0)
    fast.update(state, None)  # fades
    assert np.isclose(fast.read(drive)[0, 2], 0.5)
    drive[0, 1] = 0.0
    drive[0, 2] = 1.0  # an unseen cue reads nothing
    assert np.allclose(fast.read(drive)[0], 0.0)


def test_normalized_fast_synapses_read_an_average_of_what_followed() -> None:
    connectome = cd.layered(4, 2, 3, density=1.0, seed=0)
    out = np.asarray(connectome.populations["output"])
    fast = cd.FastSynapses(np.arange(4), out, decay=1.0, normalize=True)
    fast.reset(1)
    s = np.zeros((1, connectome.n))
    s[0, :4] = [3.0, 0.0, 0.0, 0.0]  # a key of any length is written as a unit vector
    state = cd.BrainState(v=s, activation=s, adaptation=np.zeros_like(s), steps=1)
    post = np.zeros((1, 3))
    post[0, 2] = 1.0
    fast.update(state, np.array([True]), post=post)
    fast.update(state, np.array([True]), post=post)  # written twice: still an average of one
    drive = np.zeros((1, connectome.n))
    drive[0, 0] = 0.1  # a cue of any length
    read = fast.read(drive)
    assert np.allclose(read[0], [0.0, 0.0, 1.0])
    other = np.zeros((1, 3))
    other[0, 0] = 1.0
    fast.update(
        state, np.array([True]), post=other
    )  # a third write with the same key, another follower
    assert np.allclose(fast.read(drive)[0], [1 / 3, 0.0, 2 / 3])


def test_replacing_fast_synapses_keep_one_note_per_key() -> None:
    connectome = cd.layered(3, 2, 2, density=1.0, seed=0)
    out = np.asarray(connectome.populations["output"])
    fast = cd.FastSynapses(np.arange(3), out, replace=True)
    fast.reset(1)
    s = np.zeros((1, connectome.n))
    s[0, 1] = 1.0
    state = cd.BrainState(v=s, activation=s, adaptation=np.zeros_like(s), steps=1)
    first = np.array([[1.0, 0.0]])
    second = np.array([[0.0, 1.0]])
    fast.update(state, np.array([True]), post=first)
    fast.update(state, np.array([True]), post=second)  # the same key again: the first note is gone
    drive = np.zeros((1, connectome.n))
    drive[0, 1] = 1.0
    assert np.allclose(fast.read(drive)[0], [0.0, 1.0])


def test_afterglow_is_brightest_where_the_moment_changed() -> None:
    """The afterglow weighs each hidden neuron's trace by its movement since the last moment:
    a neuron that changed glows, one that stood still fades; with focus 0 it is the Echo."""
    w, _ = cd.stateful(3, 1, 2, 3, 3, seed=1)
    w = w.with_populations(afterglow=w.populations["context"])
    brain = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0))
    glow = cd.Afterglow(w, decay=0.5, focus=1.0)
    echo = cd.Afterglow(w, decay=0.5, focus=0.0)
    hidden = list(w.populations["hidden"])
    drive_a = np.zeros((1, w.n))
    drive_a[:, 0] = 1.0
    drive_b = np.zeros((1, w.n))
    drive_b[:, 1] = 1.0
    first = brain.settle_batch(glow.stimulate(drive_a), steps=30)
    glow.update(first)
    echo.update(first)
    h1 = first.activation[:, hidden]
    assert np.allclose(glow.trace, 0.5 * h1)  # a cold stream's first moment weighs one
    assert np.allclose(echo.trace, 0.5 * h1)
    second = brain.settle_batch(glow.stimulate(drive_b), steps=30)
    glow.update(second)
    echo.update(second)
    h2 = second.activation[:, hidden]
    moved = np.abs(h2 - h1)
    weight = moved / (moved.mean(axis=1, keepdims=True) + 1e-9)
    assert np.allclose(glow.trace, 0.25 * h1 + 0.5 * weight * h2)
    assert np.allclose(echo.trace, 0.25 * h1 + 0.5 * h2)
    assert not np.allclose(glow.trace, echo.trace)
    glow.reset(1, rows=np.array([True]))
    assert glow.cold.all() and not glow.trace.any()


def test_the_trace_reads_a_state_on_the_device_the_same_as_on_the_host() -> None:
    """A settled state that rests on the torch device is sliced there; the trace it feeds is the
    host trace to the last digit."""
    pytest = __import__("pytest")
    if "torch" not in cd.available_backends():
        pytest.skip("no torch")
    w, _ = cd.stateful(3, 1, 2, 3, 3, seed=1)
    w = w.with_populations(afterglow=w.populations["context"])
    drive = np.zeros((2, w.n))
    drive[:, 0] = 1.0
    host = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0)).settle_batch(drive, steps=30)
    dev = cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0), backend="torch", device="cpu").settle_batch(
        drive, steps=30
    )
    a, b = cd.Afterglow(w, decay=0.5, source="input"), cd.Afterglow(w, decay=0.5, source="input")
    a.update(host)
    b.update(dev)
    assert np.allclose(a.trace, b.trace)
    drive[:, 1] = 1.0
    a.update(
        cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0)).settle_batch(drive, steps=30, state=host)
    )
    b.update(
        cd.NeuralGraph(w, cd.learning_neuron_model(dt=1.0), backend="torch", device="cpu").settle_batch(
            drive, steps=30, state=dev
        )
    )
    assert np.allclose(a.trace, b.trace, atol=1e-6)
