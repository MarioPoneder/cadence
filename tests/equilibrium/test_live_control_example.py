"""The live example uses acquired consequences and measured outcome error."""

import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "equilibrium"
sys.path.insert(0, str(EXAMPLES))
from live_control import (  # noqa: E402
    DT,
    ModelController,
    PointBody,
    body_reward,
    experiences,
    prepare,
    run,
)


def test_body_witnesses_and_reward_come_from_observed_changes_and_effort():
    body = PointBody(0.5)
    assert body.advance(-0.4) == pytest.approx(0.4)
    assert body.advance(0.4) == pytest.approx(0.5)
    assert experiences((0.5,), (-0.4,))[0] == (
        {"body": [0.5, -0.4]},
        {"next_position": [0.4]},
    )
    assert body_reward(0.5, 0.4, -0.4) > 0
    assert body_reward(0.5, 0.6, 0.4) < 0
    assert body_reward(0.5, 0.5, 0) == 0
    assert body_reward(0.5, 0.5, 0.4) == pytest.approx(-0.01 * 0.4**2 * DT)


def test_acquired_model_predicts_fresh_action_conditioned_consequences():
    brain, report = prepare()
    assert report["passed"]
    assert report["history"][-1]["checks"]["max_error"] < (
        report["history"][0]["checks"]["max_error"] / 4
    )
    snapshot = brain.snapshot()
    for position in (-0.3, 0.2):
        predictions = []
        for action in (-0.5, 0.1):
            predicted = brain.predict({"body": [position, action]})["next_position"][0]
            measured = PointBody(position).advance(action)
            assert abs(predicted - measured) < 0.1
            predictions.append(predicted)
        assert predictions[1] > predictions[0] + 0.1
    assert brain.snapshot() == snapshot


def test_action_comparison_uses_model_consequences_not_a_supplied_body_policy():
    class Model:
        def __init__(self):
            self.queries = []

        def settle(self, inputs):
            self.queries.append(inputs)
            action = inputs["body"][1]
            # Counterfactual model says positive motion restores the body.
            return {
                "qualified": True,
                "outputs": {"next_position": [0 if action > 0 else 0.8]},
            }

    brain = Model()
    controller = ModelController(brain)
    command = controller({"position": 0.55, "velocity": 0, "transition": None})
    assert command == {"qualified": True, "command": [0.5]}
    assert len(brain.queries) == 3


def test_unqualified_candidate_cannot_become_a_command():
    class RefusingModel:
        def settle(self, inputs):
            return {"qualified": False, "outputs": {"next_position": [0]}}

    assert ModelController(RefusingModel())(
        {"position": 0.55, "velocity": 0, "transition": None}
    ) == {"qualified": False}


def test_live_learning_reduces_need_and_progress_uses_prediction_error():
    # A generous expiry isolates deterministic behavior from machine load.
    report = run(deadline=30)
    assert report["bootstrap"]["passed"]
    assert report["final_need"] < report["initial_need"] / 10
    rows = report["transitions"]
    assert len(rows) == 20
    previous_velocity, means = 0.0, {}
    for row in rows:
        assert row["accepted"]
        assert abs(row["action"] - previous_velocity) <= 2 * DT
        assert row["measured"] == pytest.approx(row["before"] + row["action"] * DT)
        error = abs(row["predicted"] - row["measured"])
        assert row["prediction_error"] == error
        key = f"{row['action']:.2f}"
        previous = means.get(key, error)
        means[key] = 0.75 * previous + 0.25 * error
        expected = max(0, previous - means[key]) / max(1, previous)
        assert row["learning_progress"] == pytest.approx(expected)
        previous_velocity = row["action"]
    assert any(row["learning_progress"] > 0 for row in rows)
    effort = sum(0.01 * row["action"] ** 2 * DT for row in rows)
    assert sum(row["body_reward"] for row in rows) == pytest.approx(
        report["initial_need"] - report["final_need"] - effort
    )
    assert report["runtime"]["completed"] == 20
    assert report["runtime"]["errors"] == 0
    assert report["runtime"]["closed"]
    assert not report["runtime"]["worker_alive"]
    timings = report["submission_to_command_seconds"]
    assert len(timings) == 20
    assert all(value >= 0 for value in timings)
    # No wall-clock threshold assertion: the machine is not a real-time system.
    assert report["deadline_misses"] == sum(value > 30 for value in timings)


@pytest.mark.parametrize("decisions", [0, -1, 101, True, 1.5])
def test_example_rejects_unbounded_work(decisions):
    with pytest.raises(ValueError, match="decisions"):
        run(decisions=decisions)


@pytest.mark.parametrize("deadline", [0, -1, float("inf"), float("nan"), True])
def test_example_rejects_invalid_deadline(deadline):
    with pytest.raises(ValueError, match="deadline"):
        run(deadline=deadline)
