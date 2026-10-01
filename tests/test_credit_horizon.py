"""Independent return arithmetic, sequence custody and admitted delayed targets."""

import json
from collections import Counter

import pytest

from cadence import Cortex, Reinforcement


def learner(horizon=8, **options):
    c = Cortex(seed=2)
    sensor = c.input("observation", shape=1)
    values = c.column(patches=2, inputs=sensor)
    for action in range(2):
        c.output(f"q{action}", shape=(), reads=values, indices=(action,))
    return Reinforcement(
        c.build(),
        actions=2,
        action_input=None,
        value_output=("q0", "q1"),
        credit_horizon=horizon,
        capacity=256,
        batch_size=16,
        discount=0.9,
        value_scale=0.8,
        **options,
    )


def context(tick):
    return {"observation": [tick / 256]}


def values(pair=(0.4, 0.2), qualified=True):
    return [{"qualified": qualified, "work": {"evaluations": 1}}], pair


def collect(agent, rewards, monkeypatch, *, terminal=True):
    monkeypatch.setattr(agent, "_values", lambda *a: values())
    for tick, reward in enumerate(rewards):
        choice = agent.act(context(tick), explore=False)
        assert choice["action"] == 0
        done = terminal and tick == len(rewards) - 1
        agent.feedback(
            reward,
            None if done else context(tick + 1),
            terminal=done,
            learn=False,
            decision_id=choice["decision_id"],
            executed_action=choice["action"],
        )


@pytest.mark.parametrize("horizon", (1, 8, 32, 128))
@pytest.mark.parametrize("sign", (-1, 1))
def test_terminal_outcome_reaches_earliest_eligible_decision(
    horizon, sign, monkeypatch
):
    agent = learner(horizon)
    collect(agent, [0] * (horizon - 1) + [sign], monkeypatch)
    before = agent.brain.snapshot()
    target, length, stop = agent._target(0, None, Counter())
    # Independently expanded normalized discounted return, no recursive formula.
    assert target == pytest.approx(0.1 * 0.8 * sign * 0.9 ** (horizon - 1))
    assert length == horizon and stop == "terminal"
    assert agent.brain.snapshot() == before


def test_non_greedy_record_cuts_before_its_reward(monkeypatch):
    agent = learner(8)
    collect(agent, [0, 1, 1], monkeypatch)
    monkeypatch.setattr(agent, "_values", lambda *a: values((-0.2, 0.5)))
    target, length, reason = agent._target(0, None, Counter())
    assert (length, reason) == (1, "off_policy")
    assert target == pytest.approx(0.9 * 0.5)


@pytest.mark.parametrize("kind", ("reset", "context_gap"))
def test_credit_never_bridges_reset_or_missing_context(kind, monkeypatch):
    agent = learner(8)
    collect(agent, [0], monkeypatch, terminal=False)
    if kind == "reset":
        agent.reset()
    observed = context(1) if kind == "reset" else context(2)
    choice = agent.act(observed, explore=False)
    assert choice["accepted"]
    agent.feedback(
        1,
        terminal=True,
        learn=False,
        decision_id=choice["decision_id"],
        executed_action=choice["action"],
    )
    target, length, reason = agent._target(0, None, Counter())
    assert (length, reason) == (1, "discontinuity")
    assert target == pytest.approx(0.9 * 0.4)


def test_collection_cut_bootstraps_instead_of_inventing_terminal_zero(monkeypatch):
    agent = learner(8)
    collect(agent, [0.2, -0.4], monkeypatch, terminal=False)
    target, length, reason = agent._target(0, None, Counter())
    expected = 0.1 * 0.8 * 0.2 + 0.9 * 0.1 * 0.8 * -0.4 + 0.9**2 * 0.4
    assert target == pytest.approx(expected)
    assert (length, reason) == (2, "pending_future")


def test_finite_horizon_bootstraps_before_later_reward(monkeypatch):
    agent = learner(2)
    collect(agent, [0, 0, 1], monkeypatch)
    target, length, reason = agent._target(0, None, Counter())
    assert target == pytest.approx(0.9**2 * 0.4)
    assert (length, reason) == (2, "horizon")


def test_equal_best_actions_preserve_the_trace(monkeypatch):
    agent = learner(8)
    collect(agent, [0, 1], monkeypatch)
    monkeypatch.setattr(agent, "_values", lambda *a: values((0.4, 0.4)))
    assert agent._target(0, None, Counter()) == pytest.approx((0.072, 2, "terminal"))


def test_long_targets_use_public_repair_and_keep_actual_outcome_on_refusal(monkeypatch):
    agent = learner(8)
    collect(agent, [0, 0, 1], monkeypatch)
    before = agent.snapshot()
    result = agent.replay(budget=0)
    assert not result["accepted"] and result["source"] == "estimate"
    assert agent.snapshot() == before
    result = agent.replay()
    assert result["accepted"] and result["source"] == "estimate"
    targets = dict(zip(result["indices"], result["targets"], strict=True))
    assert targets == pytest.approx({0: 0.0648, 1: 0.072, 2: 0.08})


def test_late_bootstrap_refusal_changes_neither_replay_rng_nor_parameters(monkeypatch):
    agent = learner(8)
    collect(agent, [0, 0, 1], monkeypatch)
    before = agent.snapshot()
    monkeypatch.setattr(
        agent, "_values", lambda c, b: values(qualified=c != agent._context(context(2)))
    )
    result = agent.replay()
    assert result["reason"] == "bootstrap_refused"
    assert agent.snapshot() == before


def test_pending_sequence_and_reset_survive_exact_checkpoint_continuation():
    original = learner(8, seed=9)
    for tick in range(4):
        choice = original.act(context(tick))
        assert choice["accepted"]
        original.feedback(
            0.2,
            context(tick + 1),
            learn=False,
            decision_id=choice["decision_id"],
            executed_action=choice["action"],
        )
    choice = original.act(context(4))
    assert choice["accepted"]
    clone = Reinforcement.from_snapshot(original.snapshot())
    assert clone.snapshot() == original.snapshot()
    acknowledgment = dict(
        decision_id=choice["decision_id"],
        executed_action=choice["action"],
    )
    assert original.feedback(-1, terminal=True, **acknowledgment) == clone.feedback(
        -1,
        terminal=True,
        **acknowledgment,
    )
    assert original.snapshot() == clone.snapshot()
    for owner in (original, clone):
        owner.reset()
        choice = owner.act(context(0))
        assert choice["accepted"]
        owner.feedback(
            0.5,
            terminal=True,
            decision_id=choice["decision_id"],
            executed_action=choice["action"],
        )
    assert original.snapshot() == clone.snapshot()


@pytest.mark.parametrize("horizon", (0, True, 1.5, 257))
def test_invalid_horizon_is_rejected(horizon):
    with pytest.raises(ValueError):
        learner(horizon)


@pytest.mark.parametrize(
    "mutation", ("terminal_segment", "backward", "future", "boolean")
)
def test_forged_episode_links_are_rejected(monkeypatch, mutation):
    agent = learner(8)
    collect(agent, [1], monkeypatch)
    collect(agent, [1], monkeypatch)
    saved = json.loads(agent.snapshot())
    if mutation == "terminal_segment":
        saved["records"][1][4] = saved["records"][0][4]
    elif mutation == "backward":
        saved["records"][0][4], saved["records"][1][4] = 1, 0
    elif mutation == "future":
        saved["records"][0][4] = saved["episode"] + 1
    else:
        saved["episode"] = True
    with pytest.raises(ValueError):
        Reinforcement.from_snapshot(json.dumps(saved))
