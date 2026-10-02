"""Thinking has its own clock: isolated hypotheses never become observed outcomes."""

from itertools import product

import numpy as np
import pytest

import cadence as cd
from cadence.circuits import Deliberator, imagine


def planner(**options):
    def move(state, action):
        state.append(action)  # Deliberately mutate the supplied branch.
        return state

    return Deliberator(lambda _: (0, 1), move, lambda s: sum(s), lambda s: len(s) == 8, **options)


def finish(p, chunk=3):
    while p.pending:
        before = p.nodes
        p.tick(chunk)
        assert 0 <= p.nodes - before <= chunk
    return p.result


def test_bounded_ticks_pause_resume_and_input_supersedes_old_work():
    p = planner(depth=3)
    live = [0]
    p.start(live)
    live.append(99)
    p.tick(1)
    assert p.nodes == 1 and p.result is None
    p.pause()
    for _ in range(10):
        p.tick()
    assert p.nodes == 1 and p.pending
    p.resume()
    result = finish(p)
    assert result.depth == 3 and result.futures[0].state == [0, 1, 1, 1]
    before = p.nodes
    p.tick()
    assert p.nodes == before  # A completed search does not invent more work.
    p.start([3])
    assert p.result is None and p.nodes == 0
    result = finish(p)
    assert result.futures[0].state == [3, 1, 1, 1]
    p.start([])
    p.tick()
    p.cancel()
    assert not p.pending and p.result is None


def test_budget_keeps_only_a_completed_depth_and_never_overruns_transitions():
    for budget, completed in ((1, None), (2, 1), (4, 1), (8, 2)):
        p = planner(depth=4, max_nodes=budget)
        p.start([])
        result = finish(p)
        assert p.nodes == budget and p.budget_exhausted
        assert (None if result is None else result.depth) == completed
    p = planner(depth=1, max_nodes=2)
    p.start([])
    assert finish(p).depth == 1 and not p.budget_exhausted


@pytest.mark.parametrize("prune", [False, True])
def test_matches_independent_minimax_and_synchronous_comparison(prune):
    rng = np.random.default_rng(731)
    values = {path: float(rng.normal()) for path in product(range(3), repeat=4)}

    def exact(path):
        if len(path) == 4:
            return values[path]
        child = [exact((*path, action)) for action in range(3)]
        return (max if len(path) % 2 == 0 else min)(child)

    callbacks = (
        lambda _: range(3),
        lambda s, a: (*s, a),
        lambda s: values.get(s, 0.0),
        lambda s: len(s) == 4,
    )
    p = Deliberator(*callbacks, depth=4, adversarial=True, prune=prune)
    p.start(())
    result = finish(p)
    eager = imagine((), *callbacks, depth=4, adversarial=True, prune=prune)
    assert result.futures == eager.futures
    assert {f.action: f.score for f in result.futures} == {a: exact((a,)) for a in range(3)}
    assert result.nodes > eager.nodes  # Count earlier depths too.


def test_returned_future_is_isolated_and_invalid_evaluators_stop_work():
    p = planner(depth=2)
    p.start([])
    while p.result is None:
        p.tick(1)
    returned = p.tick(1)
    returned.futures[0].state.append(999)
    assert 999 not in p.result.futures[0].state
    assert 999 not in finish(p).futures[0].state
    bad = Deliberator(lambda _: [1], lambda s, a: a, lambda _: float("nan"), lambda _: False)
    bad.start(0)
    with pytest.raises(ValueError, match="nonfinite"):
        bad.tick()
    assert not bad.pending and bad.result is None


def test_terminal_state_and_numpy_action_sequences():
    p = Deliberator(lambda _: np.array([0, 1]), lambda s, a: s + 1, float, lambda s: s >= 2)
    p.start(2)
    assert finish(p).futures == () and p.nodes == 0
    p.start(0)
    assert finish(p).futures[0].score == 2


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budgets_do_not_advance_or_replace_a_search(value):
    with pytest.raises(ValueError):
        planner(depth=value)
    with pytest.raises(ValueError):
        planner(max_nodes=value)
    p = planner()
    p.start([])
    with pytest.raises(ValueError):
        p.tick(value)
    assert p.nodes == 0 and p.pending


def test_thinking_preserves_real_action_credit_and_default_memory(tmp_path):
    brain = cd.GenericBrain.build(2, 2, hidden=4, seed=4)
    assert isinstance(brain.hippocampus, cd.SynapticMemory)
    brain.step([[1.0, 0.0]])
    baseline = cd.GenericBrain.load(brain.save(tmp_path / "pending.npz"))
    p = Deliberator(
        lambda _: (0, 1),
        lambda state, action: np.eye(2)[[action]],
        lambda state: float(brain.basal_ganglia.value_of(brain.stimulus(state))[0]),
        lambda _: False,
        depth=4,
    )
    p.start(np.eye(2)[[0]])
    finish(p, chunk=1)
    assert brain.hippocampus.writes == 0
    np.testing.assert_array_equal(brain.hippocampus.consolidated, 0)
    for b in (brain, baseline):
        b.step([[0.0, 1.0]], reward=np.array([1.0]))
    np.testing.assert_array_equal(
        brain.learner.brain.connectome.sign, baseline.learner.brain.connectome.sign
    )
    np.testing.assert_array_equal(brain.basal_ganglia.w_critic, baseline.basal_ganglia.w_critic)
    np.testing.assert_array_equal(brain.hippocampus.consolidated, baseline.hippocampus.consolidated)
    assert brain.hippocampus.writes == 1
    assert np.linalg.norm(brain.hippocampus.consolidated) > 0
    before = brain.hippocampus.consolidated.copy()
    brain.reset()
    brain.hippocampus.reset(1)
    np.testing.assert_array_equal(brain.hippocampus.consolidated, before)


def test_explicit_memory_opt_out_and_checkpoint_preserve_configuration(tmp_path):
    brain = cd.GenericBrain.build(2, 2, hidden=4, episodic=False)
    assert brain.hippocampus is None
    assert cd.GenericBrain.load(brain.save(tmp_path / "without.npz")).hippocampus is None
    assert isinstance(cd.GenericBrain(brain.connectome).hippocampus, cd.SynapticMemory)


def test_custom_state_cloner_also_isolates_published_results():
    class State:
        def __init__(self, value):
            self.value = value

        def __deepcopy__(self, memo):
            raise AssertionError("a caller-supplied state cloner must be used")

    p = Deliberator(
        lambda _: (1,),
        lambda s, a: State(s.value + a),
        lambda s: s.value,
        lambda s: s.value >= 2,
        depth=2,
        clone=lambda s: State(s.value),
    )
    p.start(State(0))
    finish(p)
    published = p.tick()
    assert published.futures[0].state.value == 2
    published.futures[0].state.value = 99
    assert p.tick().futures[0].state.value == 2
