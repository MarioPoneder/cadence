"""Independent arithmetic and actual-experience custody for the diagnostic."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "examples" / "credit_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("credit_diagnostic", PATH)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def segment(rewards, episode=0):
    return [
        dict(
            episode=episode,
            stage=i,
            first_action=0,
            action=i % 2,
            context={"senses": [i]},
            reward=reward,
            following=None if i == len(rewards) - 1 else {"senses": [i + 1]},
        )
        for i, reward in enumerate(rewards)
    ]


@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("length", (1, 8, 32, 128))
def test_observed_return_matches_expanded_discounted_sum(sign, length):
    rewards = [sign * ((i % 5) - 2) / 2 for i in range(length)]
    result, steps, stop = experiment.observed_return(segment(rewards), 0, 0.9, 0.8, 2)
    expected = sum(0.9**i * 0.1 * 0.8 * reward / 2 for i, reward in enumerate(rewards))
    assert result == pytest.approx(expected)
    assert steps == length and stop == "terminal"
    assert abs(result) <= 0.8


def test_terminal_stops_before_another_episodes_reward():
    records = segment([0, -1]) + segment([1, 1], episode=1)
    before = json.dumps(records)
    assert experiment.observed_return(records, 0, 0.5, 0.8, 1) == (-0.2, 2, "terminal")
    assert json.dumps(records) == before


@pytest.mark.parametrize("gap", ("episode", "context", "unfinished"))
def test_no_return_is_invented_across_missing_evidence(gap):
    records = segment([0, 1])
    if gap == "episode":
        records[1]["episode"] = 1
    elif gap == "context":
        records[1]["context"] = {"senses": [99]}
    else:
        records.pop()
    with pytest.raises(ValueError):
        experiment.observed_return(records, 0, 0.9, 0.8, 1)


def test_full_exploration_preserves_actual_actions_and_counterbalanced_outcomes():
    agents = [experiment.make(2, 1) for _ in range(2)]
    collections = [
        experiment.collect(agent, 1, preferred)
        for preferred, agent in enumerate(agents)
    ]
    for records, outcomes, work, status in collections:
        assert status == "complete" and len(outcomes) == 24
        assert len(records) == 48 and work["evaluations"] > 0
        for row in records:
            body = outcomes[row["episode"]]
            assert row["action"] == body["actions"][row["stage"]]
            assert row["first_action"] == body["actions"][0]
            assert row["reward"] == (body["reward"] if row["stage"] == 1 else 0)
    assert [row["actions"] for row in collections[0][1]] == [
        row["actions"] for row in collections[1][1]
    ]
    assert all(
        a["reward"] == -b["reward"]
        for a, b in zip(collections[0][1], collections[1][1], strict=True)
    )
    assert all(agent.brain.inspect()["admissions"] == 0 for agent in agents)


def test_fixed_rows_cover_both_root_actions_and_never_duplicate_a_row():
    agent = experiment.make(2, 1)
    records, _, _, _ = experiment.collect(agent, 1, 0)
    covered = experiment.coverage(records, 1)
    assert covered["root_adequate"]
    batches = experiment.schedule(records, 99)
    assert batches == experiment.schedule(records, 99)
    assert len(batches) == 32
    for rows in batches:
        assert len(rows) == len(set(rows)) == 16
        assert [records[i]["action"] for i in rows[:2]] == [0, 1]
        assert all(records[i]["stage"] == 0 for i in rows[:2])
        assert all(records[i]["stage"] > 0 for i in rows[2:])
        assert rows[-1] == len(records) - 1
    incomplete = [row for row in records if row["first_action"] == 0]
    assert not experiment.coverage(incomplete, 1)["root_adequate"]
    with pytest.raises(ValueError, match="Both executed root actions"):
        experiment.schedule(incomplete, 99)


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_each_arm_uses_public_estimated_repair_and_preserves_record_custody(arm):
    founder = experiment.make(2, 1)
    records, _, _, _ = experiment.collect(founder, 1, 0)
    saved = json.loads(founder.snapshot())
    saved["config"]["credit_horizon"] = 1 if arm == "one_step" else 2
    agent = experiment.Reinforcement.from_snapshot(json.dumps(saved))
    rows = experiment.schedule(records, 99)[0]
    before = json.loads(agent.snapshot())
    result = experiment.fit(agent, records, rows, arm)
    after = json.loads(agent.snapshot())
    assert result["accepted"] and result["source"] == "estimate"
    assert result["indices"] == rows and len(result["targets"]) == 16
    assert result["work"]["evaluations"] > 0
    for key in ("records", "rng", "pending", "episode", "transitions"):
        assert before[key] == after[key]
    assert agent.brain.inspect()["admissions"] == 1
    assert json.loads(before["brain"])["state"] == json.loads(after["brain"])["state"]
    query_before = agent.snapshot()
    assert experiment.root_query(agent, 1, 0)["query_pure"]
    evaluation = experiment.evaluate(agent, 1, 0)
    assert evaluation["status"] == "complete" and len(evaluation["actions"]) == 2
    assert evaluation["reward"] == (1 if evaluation["actions"][0] == 0 else -1)
    assert agent.snapshot() == query_before


def test_complete_runner_freezes_actual_records_and_identical_rows(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(experiment, "UPDATES", 2)
    monkeypatch.setattr(experiment, "CHECKPOINTS", (0, 1, 2))
    destination = tmp_path / "case"
    report = experiment.run(
        SimpleNamespace(seed=2, delay=1, preferred=0, out=destination)
    )
    assert report["status"] == "complete"
    assert report["matched_rows"] and report["sources_unchanged"]
    assert report["collection_sha256"] == experiment.digest(
        destination / "collection.json"
    )
    assert report["batches_sha256"] == experiment.digest(destination / "batches.json")
    batches = json.loads((destination / "batches.json").read_text())
    for arm in report["arms"]:
        assert [row["indices"] for row in arm["updates"]] == batches
        assert arm["total_work"]["evaluations"] > arm["work"]["evaluations"]
        assert [row["update"] for row in arm["queries"]] == [0, 1, 2]
    assert report["arms"][2]["root_terminal_targets"] == 4
