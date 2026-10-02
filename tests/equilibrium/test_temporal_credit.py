"""Behavior and evidence controls for the bounded temporal qualification."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "examples" / "equilibrium" / "temporal_credit.py"
SPEC = importlib.util.spec_from_file_location("temporal_credit", PATH)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


@pytest.mark.parametrize("delay", (2, 4, 8))
def test_paired_suffixes_have_no_cue_leak_and_history_expires(delay):
    negative = experiment.cue_sequence(-0.6, delay, 100)
    positive = experiment.cue_sequence(0.6, delay, 100)
    assert negative[1:] == positive[1:]
    assert experiment.encode(negative) != experiment.encode(positive)
    assert experiment.encode(negative, memoryless=True) == experiment.encode(
        positive, memoryless=True
    )
    left = experiment.History(2, steps=delay)
    right = experiment.History(2, steps=delay)
    for a, b in zip(negative, positive, strict=True):
        lhs, rhs = left.push(a), right.push(b)
    assert lhs == rhs  # cue is genuinely unavailable after eviction


@pytest.mark.parametrize("delay", (2, 4, 8))
@pytest.mark.parametrize("architecture", ("ordinary", "observer"))
def test_reserved_cue_recall_and_resumed_life(delay, architecture):
    result = experiment.memory_case(0, delay, architecture)
    assert result["passed"], result
    assert len(result["rows"]) == 32
    assert result["work"]["solver"]["evaluations"] > 0
    for control in ("memoryless_retained_state", "memoryless_reset_state"):
        for suffix in (100, 101):
            predictions = [
                row["prediction"]
                for row in result["rows"]
                if row["control"] == control and row["suffix"] == suffix
            ]
            # Warm starts alone cannot reproduce the two different target signs.
            assert max(predictions) - min(predictions) < 0.002


@pytest.mark.parametrize("preferred", (0, 1))
def test_reward_eight_ticks_later_changes_executed_policy(preferred):
    result = experiment.credit_case(0, 8, "td_replay", preferred=preferred)
    assert result["passed"], result
    assert len(result["outcomes"]) == 60
    assert result["resumed_equal"]
    for episode in result["outcomes"]:
        assert len(episode["actions"]) == 9
        assert episode["reward"] == (1 if episode["actions"][0] == preferred else -1)
    assert result["success"] == sum(row["chosen"] == preferred for row in result["evaluation"]) / 40
    assert result["success"] >= 0.65
    # Two query calls per action and replay work are all charged by the helper.
    assert result["work"]["solver"]["evaluations"] > 2 * 9 * 100
    assert result["experience"]["transitions"] == 9 * 100
    assert result["experience"]["updates"] == 9 * 60
    assert result["ideal_value_magnitude"] == pytest.approx(0.18 * 0.8**8)


def test_hypothetical_or_abandoned_choice_cannot_receive_credit():
    brain = experiment.layout(5, 0, "ordinary", ("q0", "q1"))
    learner = experiment.Reinforcement(
        brain, actions=2, action_input=None, value_output=("q0", "q1")
    )
    initial = learner.snapshot()
    assert brain.settle(experiment.context(0, 0, 2))["qualified"]
    assert learner.snapshot() == initial
    with pytest.raises(ValueError, match="pending accepted act"):
        learner.feedback(1, terminal=True, decision_id=1, executed_action=0)
    decision = learner.act(experiment.context(0, 0, 2))
    assert decision["accepted"]
    learner.reset()  # body rejected the command; no executed transition
    with pytest.raises(ValueError, match="pending accepted act"):
        learner.feedback(
            1,
            terminal=True,
            decision_id=decision["decision_id"],
            executed_action=decision["action"],
        )
    assert learner.inspect()["transitions"] == 0
    assert learner.inspect()["updates"] == 0


@pytest.mark.parametrize("episodes", (0, 101, True, 1.2))
def test_collection_is_bounded(episodes):
    with pytest.raises(ValueError, match="episodes"):
        experiment.credit_case(0, 8, "td_replay", episodes=episodes)
