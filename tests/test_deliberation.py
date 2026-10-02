"""Compare futures with a learned evaluator; learn only from observed outcomes.

Regression fixture for the core search and local learning operations. The transition
model here is supplied. Each imagined branch gets its own copied state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import cadence as cd
from cadence.circuits import Deliberator


def compare_futures(
    live: dict[str, Any],
    actions: tuple[int, ...],
    transition: Callable[[dict[str, Any], int], tuple[dict[str, Any], float, bool]],
    value: Callable[[dict[str, Any]], float],
    *,
    horizon: int = 2,
    discount: float = 0.9,
) -> list[dict[str, Any]]:
    """Use the core bounded planner, keeping each root action's best future.

    A copied branch carries cumulative reward and elapsed time. Its leaf value
    adds the discounted remaining utility. A supplied terminal evaluator may
    include outcome utility that the transition has not already returned.
    """
    if not actions or horizon < 1 or not 0 <= discount <= 1:
        raise ValueError("nonempty actions, positive horizon and discount in [0, 1] required")
    if len(actions) ** horizon > 4096:
        raise ValueError("this example is limited to 4096 leaf branches")

    def advance(branch: dict[str, Any], action: int) -> dict[str, Any]:
        state, reward, done = transition(branch["world"], action)
        return {
            "world": state,
            "reward": branch["reward"] + discount ** branch["elapsed"] * reward,
            "elapsed": branch["elapsed"] + 1,
            "done": done,
        }

    thought = Deliberator(
        lambda branch: actions,
        advance,
        lambda branch: branch["reward"] + discount ** branch["elapsed"] * value(branch["world"]),
        lambda branch: branch["done"],
        depth=horizon,
        max_nodes=8192,
    )
    thought.start({"world": live, "reward": 0.0, "elapsed": 0, "done": False})
    while thought.pending:
        # A UI schedules one bounded slice while waiting for input.
        thought.tick(nodes=16)
    result = thought.result
    if result is None or result.depth != horizon:
        raise ValueError("the budget could not finish the requested horizon")
    return [
        {"actions": future.sequence, "score": future.score, "state": future.state["world"]}
        for future in result.futures
    ]


class Evaluator:
    """A tiny brain learns whether a predicted terminal observation is good or bad."""

    def __init__(self, seed: int = 0) -> None:
        connectome = cd.layered(2, 8, 2, density=1, seed=seed)
        config = cd.LearnerConfig(eta=3, eta_bias=0.03, temperature=0.1, tolerance=1e-4)
        self.learner = cd.Learner(
            cd.Brain(connectome, cd.learning_neuron_model(dt=1)),
            connectome.populations["output"],
            config,
        )
        self.n = connectome.n

    def drive(self, features: np.ndarray) -> np.ndarray:
        return np.pad(features, ((0, 0), (0, self.n - 2)))

    def teach(self, features: np.ndarray, outcomes: np.ndarray) -> None:
        self.learner.step(self.drive(features), (outcomes < 0).astype(int))

    def value(self, state: dict[str, Any]) -> float:
        free = self.learner.free(self.drive(state["features"][None]))
        logits = free.activation[0, self.learner.output_index] / 0.1
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return float(p[0] - p[1])

    def observe_outcome(self, state: dict[str, Any], actual: float) -> float:
        """Report a signed prediction error and teach the observed terminal outcome."""
        error = actual - self.value(state)
        self.teach(state["features"][None], np.array([actual]))
        return error


def transition(state: dict[str, Any], action: int) -> tuple[dict[str, Any], float, bool]:
    """A supplied two-step world: the tempting route hides a bad terminal outcome."""
    state["trail"].append(action)
    if state["phase"] == 0:
        state["chosen"] = action
        state["phase"] = 1
        return state, float(state["temptation"] if action != state["safe"] else 0), False
    good = state["chosen"] == state["safe"]
    state["features"] = np.array(
        [state["intensity"] if good else 0, 0 if good else state["intensity"]]
    )
    state["phase"] = 2
    return state, 0.0, True  # terminal utility is evaluated separately


def run(seed: int = 0, trials: int = 100) -> dict[str, Any]:
    brain = Evaluator(seed)
    rng = np.random.default_rng(seed)
    features = np.eye(2)[np.arange(64) % 2] * rng.uniform(0.5, 1, (64, 1))
    outcomes = np.where(np.arange(64) % 2 == 0, 1, -1)
    for _ in range(100):
        brain.teach(features, outcomes)
    wins = {"one_step": 0, "two_steps": 0, "exact_value_control": 0, "wrong_model": 0}
    errors = []
    for _ in range(trials):
        live = {
            "phase": 0,
            "safe": int(rng.integers(2)),
            "features": np.zeros(2),
            "intensity": float(rng.uniform(0.6, 1)),
            "temptation": float(rng.uniform(0.05, 0.3)),
            "trail": [],
        }
        wrong = {**live, "safe": 1 - live["safe"]}

        def exact(state: dict[str, Any]) -> float:
            return float(np.sign(state["features"][0] - state["features"][1]))

        for arm, initial, horizon, evaluator in (
            ("one_step", live, 1, brain.value),
            ("two_steps", live, 2, brain.value),
            ("exact_value_control", live, 2, exact),
            ("wrong_model", wrong, 2, brain.value),
        ):
            candidate = compare_futures(initial, (0, 1), transition, evaluator, horizon=horizon)[0]
            action = candidate["actions"][0]
            wins[arm] += action == live["safe"]
            if arm == "two_steps":
                actual = 1.0 if action == live["safe"] else -1.0
                predicted = brain.value(candidate["state"])
                errors.append(actual - predicted)  # signed outcome prediction error
        assert live["phase"] == 0 and live["trail"] == []
    # Only an observed terminal outcome supplies a teaching update, never a hypothetical one.
    updates_before = brain.learner.updates
    observed = {"features": np.array([0.8, 0.0])}
    positive_error = brain.observe_outcome(observed, 1.0)
    negative_error = brain.observe_outcome(observed, -1.0)
    return {
        "trials": trials,
        "correct": wins,
        "accuracy": {k: v / trials for k, v in wins.items()},
        "observed_feedback": {
            "better_than_expected": positive_error,
            "worse_than_expected": negative_error,
        },
        "mean_outcome_prediction_error": float(np.mean(errors)),
        "real_outcome_updates": brain.learner.updates - updates_before,
        "boundary": (
            "Supplied transition model; learned terminal evaluator; "
            "exact-value and wrong-model controls. "
            "Not a learned world model or a general planning advantage."
        ),
    }



def test_delayed_consequence_and_wrong_world_model():
    result = run(trials=20)
    assert result["accuracy"]["two_steps"] == 1
    assert result["accuracy"]["one_step"] == 0
    assert result["accuracy"]["exact_value_control"] == 1
    assert result["accuracy"]["wrong_model"] == 0
    assert result["real_outcome_updates"] == 2
    assert result["observed_feedback"]["better_than_expected"] > 0
    assert result["observed_feedback"]["worse_than_expected"] < 0


def test_branches_do_not_share_live_or_other_branch_memory():
    live = {"memory": np.zeros(2), "trail": []}

    def step(state, action):
        state["memory"][action] += 1
        state["trail"].append(action)
        return state, 0.0, False

    futures = compare_futures(live, (0, 1), step, lambda s: float(s["memory"][1]), horizon=2)
    np.testing.assert_array_equal(live["memory"], [0, 0])
    assert live["trail"] == []
    assert len(futures) == 2
    assert {tuple(f["state"]["trail"]) for f in futures} == {(0, 1), (1, 1)}
    assert futures[0]["actions"] == (1, 1)


def test_imagination_reads_without_changing_learner():
    brain = Evaluator()
    before = brain.learner.brain.dense().copy()
    state = {
        "phase": 0,
        "safe": 1,
        "features": np.zeros(2),
        "intensity": 0.8,
        "temptation": 0.2,
        "trail": [],
    }
    compare_futures(state, (0, 1), transition, brain.value)
    np.testing.assert_array_equal(brain.learner.brain.dense(), before)
    assert brain.learner.updates == 0
    with pytest.raises(ValueError):
        compare_futures(state, (0, 1), transition, brain.value, horizon=20)


def test_unexpected_bad_outcome_reduces_its_predicted_value():
    brain = Evaluator()
    state = {"features": np.array([0.8, 0.0])}
    for _ in range(20):
        brain.observe_outcome(state, 1.0)
    before = brain.value(state)
    error = brain.observe_outcome(state, -1.0)
    assert error < 0
    assert brain.value(state) < before
