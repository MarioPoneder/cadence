"""Action-value targets, outcome custody and genuine ordinary-repair learning."""

import copy
import json
import random
from collections import Counter

import pytest

from cadence import Cortex
from cadence import reinforcement as reinforcement_module
from cadence.reinforcement import Reinforcement


def controller(*, shape=(1,), recursive=False, **options):
    cortex = Cortex(seed=3)
    signal = cortex.input("signal", shape=shape)
    action = cortex.input("action", shape=2)
    base = cortex.column("base", patches=3, inputs=(signal, action))
    top = (
        cortex.observer("top", patches=2, inputs=(signal, action), observes=base)
        if recursive
        else base
    )
    cortex.output("value", shape=shape, reads=top)
    return Reinforcement(cortex.build(), actions=2, **options)


def context(value=0.3):
    return {"signal": [value]}


def record(agent, reward=0.5, *, terminal=True, learn=False, value=0.3):
    assert agent.act(context(value))["accepted"]
    return agent.feedback(
        reward, None if terminal else context(-value), terminal=terminal, learn=learn
    )


def fake_values(values, qualified=True):
    # Unit fixtures isolate target arithmetic; acquisition tests use real solves.
    return [
        {"qualified": qualified, "work": {"evaluations": i + 1}}
        for i in range(len(values))
    ], tuple(values)


@pytest.mark.parametrize("shape", ((), (1,), (1, 1)))
def test_scalar_output_shapes_and_boundaries(shape):
    agent = controller(shape=shape, batch_size=1)
    supplied = {"signal": 0.3 if not shape else [0.3]}
    result = agent.act(supplied)
    assert result["accepted"] and result["action"] in (0, 1)
    update = agent.feedback(1, terminal=True)
    assert update["accepted"] and update["source"] == "estimate"
    assert update["targets"] == pytest.approx((0.045,))


@pytest.mark.parametrize("reward", (-2.0, 0.0, 1.0, 2.0))
def test_terminal_targets_have_reward_scaling_but_no_bootstrap(reward, monkeypatch):
    agent = controller(discount=0.25, reward_scale=2, value_scale=0.8)
    record(agent, reward)
    monkeypatch.setattr(agent, "_values", lambda *a: pytest.fail("terminal query"))
    before = agent.brain.state
    update = agent.replay()
    assert update["accepted"] and update["source"] == "estimate"
    assert update["targets"] == pytest.approx(((1 - 0.25) * 0.8 * reward / 2,))
    assert agent.brain.state == before
    assert agent.inspect()["transitions"] == agent.inspect()["updates"] == 1


@pytest.mark.parametrize("values", ((0.1, 0.5), (-0.9, -0.7), (1.5, 2), (-2, -1.5)))
def test_nonterminal_targets_use_max_next_value_with_explicit_projection(
    values, monkeypatch
):
    agent = controller(discount=0.4, reward_scale=2, value_scale=0.8)
    record(agent, reward=-1, terminal=False)
    calls = []

    def next_values(inputs, budget):
        calls.append(inputs)
        return fake_values(values)

    monkeypatch.setattr(agent, "_values", next_values)
    observe_batch = agent.brain.observe_batch
    fits = []

    def fit(examples, **options):
        fits.append((copy.deepcopy(examples), dict(options)))
        return observe_batch(examples, **options)

    monkeypatch.setattr(agent.brain, "observe_batch", fit)
    result = agent.replay()
    expected = (1 - 0.4) * 0.8 * (-1 / 2) + 0.4 * max(-0.8, min(0.8, max(values)))
    assert result["accepted"]
    assert result["targets"] == pytest.approx((expected,))
    assert calls == [{"signal": (-0.3,)}]
    assert fits[0][1]["source"] == "estimate"
    assert fits[0][0][0][1] == {"value": (expected,)}
    assert result["work"]["evaluations"] > 3


def test_replay_recomputes_delayed_credit_without_admitting_predicted_outcomes(
    monkeypatch,
):
    agent = controller(discount=0.5, value_scale=0.8)
    record(agent, reward=0, terminal=False)
    monkeypatch.setattr(agent, "_values", lambda *a: fake_values((0.0, 0.0)))
    first = agent.replay()
    monkeypatch.setattr(agent, "_values", lambda *a: fake_values((0.2, 0.4)))
    second = agent.replay()
    assert first["targets"] == (0.0,)
    assert second["targets"] == (0.2,)
    assert first["source"] == second["source"] == "estimate"
    assert agent.inspect()["records"] == agent.inspect()["transitions"] == 1
    assert agent.inspect()["updates"] == 2


def test_latest_record_included_fifo_bounded_and_targets_share_preupdate_brain(
    monkeypatch,
):
    agent = controller(capacity=3, batch_size=2)
    for value in (0.1, 0.2, 0.3, 0.4):
        record(agent, terminal=False, value=value)
    assert agent.inspect()["records"] == 3
    assert agent.inspect()["transitions"] == 4
    saved = json.loads(agent.snapshot())
    assert [row[0]["signal"][0] for row in saved["records"]] == [0.2, 0.3, 0.4]
    seen = []
    checkpoint = agent.brain.snapshot()

    def next_values(inputs, budget):
        assert agent.brain.snapshot() == checkpoint
        seen.append(inputs)
        return fake_values((0.1, 0.2))

    monkeypatch.setattr(agent, "_values", next_values)
    update = agent.replay()
    assert update["accepted"]
    assert len(set(update["indices"])) == 2 and update["indices"][-1] == 2
    assert len(seen) == 2 and seen[-1] == {"signal": (-0.4,)}
    assert agent.inspect()["transitions"] == 4


@pytest.mark.parametrize("explore", (False, True))
def test_action_selection_matches_local_rng_and_greedy_ties(explore, monkeypatch):
    agent = controller(seed=11, exploration=1)
    monkeypatch.setattr(agent, "_values", lambda *a: fake_values((0.2, 0.2)))
    rng = random.Random(11)
    expected = []
    actual = []
    for _ in range(12):
        if explore:
            rng.random()
        expected.append(rng.choice([0, 1]))
        result = agent.act(context(), explore=explore)
        assert result["accepted"]
        assert result["exploratory"] is explore
        actual.append(result["action"])
        agent.reset()
    assert actual == expected
    assert set(actual) == {0, 1}


def test_greedy_selects_maximum_and_query_action_vectors_are_one_hot(monkeypatch):
    agent = controller(exploration=0)
    settle = agent.brain.settle
    seen = []

    def query(inputs, **options):
        seen.append(inputs["action"])
        return settle(inputs, **options)

    monkeypatch.setattr(agent.brain, "settle", query)
    result = agent.act(context(), explore=False)
    assert result["values"][result["action"]] == max(result["values"])
    assert seen[:2] == [(1.0, 0.0), (0.0, 1.0)]
    assert seen[-1] == tuple(float(i == result["action"]) for i in range(2))


@pytest.mark.parametrize("failure", ("query_refused", "step_refused", "step_error"))
def test_action_refusal_and_exceptions_preserve_rng_pending_and_continuation(
    failure, monkeypatch
):
    agent = controller()
    before = agent.snapshot()
    if failure == "query_refused":
        result = agent.act(context(), budget=0)
    else:

        def step(*args, **kwargs):
            if failure == "step_error":
                raise ValueError("failed step")
            return {"accepted": False, "work": {"evaluations": 2}}

        monkeypatch.setattr(agent.brain, "step", step)
        if failure == "step_error":
            with pytest.raises(ValueError, match="failed step"):
                agent.act(context())
        else:
            result = agent.act(context())
    if failure != "step_error":
        assert not result["accepted"] and result["reason"] == failure
        assert result["action"] is None and result["work"]["evaluations"] > 0
    assert agent.snapshot() == before


@pytest.mark.parametrize(
    "failure", ("bootstrap_refused", "query_error", "fit_error", "fit_refused")
)
def test_replay_refusal_and_exceptions_preserve_rng_and_parameters(
    failure, monkeypatch
):
    agent = controller(capacity=4, batch_size=2)
    for i in range(3):
        record(agent, terminal=False, value=0.1 * (i + 1))
    before = agent.snapshot()
    if failure == "bootstrap_refused":
        monkeypatch.setattr(agent, "_values", lambda *a: fake_values((0, 0), False))
    if failure == "query_error":

        def failed_query(*args):
            raise ValueError("failed query")

        monkeypatch.setattr(agent, "_values", failed_query)
    if failure == "fit_error":

        def failed_fit(*args, **kwargs):
            raise ValueError("failed fit")

        monkeypatch.setattr(agent.brain, "observe_batch", failed_fit)
    if failure == "fit_refused":
        fit = agent.brain.observe_batch
        monkeypatch.setattr(
            agent.brain,
            "observe_batch",
            lambda examples, **options: fit(examples, source="estimate", budget=0),
        )
    if failure.endswith("error"):
        with pytest.raises(ValueError, match="failed"):
            agent.replay()
    else:
        result = agent.replay()
        assert not result["accepted"] and result["work"]["evaluations"] > 0
    assert agent.snapshot() == before


def test_feedback_keeps_actual_outcome_after_refused_fit_and_replay_can_retry():
    agent = controller(discount=0, batch_size=1)
    assert agent.act(context())["accepted"]
    before = agent.brain.snapshot()
    refused = agent.feedback(1, terminal=True, budget=0)
    assert refused["stored"] and not refused["accepted"]
    assert refused["source"] == "estimate"
    assert agent.brain.snapshot() == before
    assert agent.inspect() == {
        "config": dict(agent.config),
        "records": 1,
        "transitions": 1,
        "updates": 0,
        "pending": False,
    }
    with pytest.raises(ValueError, match="preceding"):
        agent.feedback(1, terminal=True)
    update = agent.replay()
    assert update["accepted"] and agent.inspect()["updates"] == 1
    assert agent.inspect()["transitions"] == 1


def test_feedback_numerical_exception_keeps_valid_record_but_restores_replay_rng(
    monkeypatch,
):
    agent = controller()
    record(agent, terminal=False)
    assert agent.act(context())["accepted"]
    prior = json.loads(agent.snapshot())

    def fail(*args, **kwargs):
        raise ValueError("numeric failure")

    monkeypatch.setattr(agent, "_values", fail)
    with pytest.raises(ValueError, match="numeric failure"):
        agent.feedback(0.3, context(-0.2))
    after = json.loads(agent.snapshot())
    assert after["rng"] == prior["rng"]
    assert after["brain"] == prior["brain"]
    assert after["transitions"] == 2 and len(after["records"]) == 2
    assert after["pending"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward": 2},
        {"reward": True},
        {"reward": float("nan")},
        {"reward": 0, "terminal": 1},
        {"reward": 0, "learn": 1},
        {"reward": 0, "terminal": True, "next_inputs": context()},
        {"reward": 0},
        {"reward": 0, "next_inputs": {}},
        {"reward": 0, "next_inputs": context(), "budget": -1},
    ],
)
def test_invalid_feedback_is_atomic_including_pending_action(kwargs):
    agent = controller()
    assert agent.act(context())["accepted"]
    before = agent.snapshot()
    with pytest.raises(ValueError):
        agent.feedback(**kwargs)
    assert agent.snapshot() == before


def test_pending_action_blocks_second_act_and_reset_does_not_invent_experience():
    agent = controller()
    assert agent.act(context())["accepted"]
    before = agent.snapshot()
    with pytest.raises(ValueError, match="pending"):
        agent.act(context())
    assert agent.snapshot() == before
    activity = agent.brain.snapshot()
    agent.reset()
    assert agent.brain.snapshot() == activity
    assert agent.inspect()["transitions"] == agent.inspect()["records"] == 0
    assert agent.inspect()["pending"] is False


def test_snapshot_pending_replay_rng_and_continuation_are_exact_and_independent():
    agent = controller(seed=91, capacity=4, batch_size=2)
    for value in (0.1, 0.2, 0.3):
        record(agent, value=value)
    assert agent.act(context(-0.4))["accepted"]
    saved = agent.snapshot()
    restored = Reinforcement.from_snapshot(saved)
    assert restored.snapshot() == saved
    for owner in (agent, restored):
        assert owner.feedback(-0.5, terminal=True)["accepted"]
    assert agent.snapshot() == restored.snapshot()
    assert agent.act(context(0.5)) == restored.act(context(0.5))
    assert agent.snapshot() == restored.snapshot()
    prior = restored.snapshot()
    agent.reset()
    assert restored.snapshot() == prior


def test_inputs_replay_and_inspection_are_owned_and_rng_is_local():
    agent = controller()
    rng = random.getstate()
    supplied = context()
    assert agent.act(supplied)["accepted"]
    supplied["signal"][0] = 99
    following = context(-0.4)
    agent.feedback(0.2, following, learn=False)
    following["signal"][0] = 99
    record = json.loads(agent.snapshot())["records"][0]
    assert record[0] == {"signal": [0.3]} and record[3] == {"signal": [-0.4]}
    information = agent.inspect()
    information["config"]["discount"] = 0
    assert agent.config["discount"] == 0.95
    with pytest.raises(TypeError):
        agent.config["discount"] = 0
    assert random.getstate() == rng


def test_foreign_sensor_and_injected_action_are_rejected_without_advancement():
    agent = controller()
    foreign = Cortex().input("signal", shape=1)
    before = agent.snapshot()
    for supplied in ({foreign: [0.3]}, {**context(), "action": [1, 0]}):
        with pytest.raises(ValueError):
            agent.act(supplied)
        assert agent.snapshot() == before
    own = next(sensor for sensor in agent.brain._inputs if sensor.name == "signal")
    assert agent.act({own: [0.3]})["accepted"]


@pytest.mark.parametrize(
    "option,value",
    [
        ("actions", 1),
        ("actions", True),
        ("discount", 1),
        ("discount", -0.1),
        ("exploration", 1.1),
        ("exploration", -0.1),
        ("reward_scale", 0),
        ("reward_scale", float("inf")),
        ("value_scale", 0),
        ("value_scale", 1),
        ("capacity", 0),
        ("capacity", 3),
        ("capacity", 10**100),
        ("batch_size", 0),
        ("seed", -1),
        ("action_input", "absent"),
        ("value_output", "absent"),
    ],
)
def test_configuration_rejected_without_touching_brain(option, value):
    brain = controller().brain
    before = brain.snapshot()
    with pytest.raises(ValueError):
        Reinforcement(brain, **{"actions": 2, option: value})
    assert brain.snapshot() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("implementation", "0" * 64),
        ("transitions", True),
        ("updates", 1),
        ("records", []),
        ("pending", [{"signal": [0.1]}, 2]),
        ("rng", [3, [0] * 625, 0.0]),
    ],
)
def test_checkpoint_rejects_bad_structure_counts_actions_and_rng(field, value):
    agent = controller()
    record(agent)
    data = json.loads(agent.snapshot())
    data[field] = value
    with pytest.raises(ValueError):
        Reinforcement.from_snapshot(json.dumps(data))


def test_action_and_replay_work_include_every_real_candidate_query_and_fit(monkeypatch):
    agent = controller()
    work = Counter()
    settle = agent.brain.settle
    fit = agent.brain.observe_batch

    def query(*args, **kwargs):
        result = settle(*args, **kwargs)
        work.update(result["work"])
        return result

    def update(*args, **kwargs):
        result = fit(*args, **kwargs)
        work.update(result["work"])
        return result

    monkeypatch.setattr(agent.brain, "settle", query)
    monkeypatch.setattr(agent.brain, "observe_batch", update)
    action = agent.act(context())
    assert action["work"] == dict(work)
    work.clear()
    feedback = agent.feedback(0.5, context(-0.4))
    assert feedback["accepted"] and feedback["work"] == dict(work)


@pytest.mark.parametrize("recursive", (False, True))
def test_actual_reward_experience_acquires_greedy_choice_with_ordinary_repair(
    recursive,
):
    agent = controller(recursive=recursive, discount=0, exploration=1, batch_size=8)
    for _ in range(32):
        action = agent.act(context())
        assert action["accepted"]
        result = agent.feedback(1 if action["action"] == 1 else -1, terminal=True)
        assert result["accepted"] and result["source"] == "estimate"
    assessment = agent.act(context(), explore=False)
    assert assessment["accepted"] and assessment["action"] == 1
    assert assessment["values"][1] > 0.5
    assert assessment["values"][0] < -0.5
    assert agent.inspect()["transitions"] == agent.inspect()["updates"] == 32


def test_actual_two_step_credit_reaches_the_unrewarded_predecessor():
    agent = controller(discount=0.6, value_scale=0.8, exploration=1, batch_size=8)
    for _ in range(40):
        assert agent.act(context(-0.8))["accepted"]
        assert agent.feedback(0, context(0.8))["accepted"]
        assert agent.act(context(0.8))["accepted"]
        assert agent.feedback(1, terminal=True)["accepted"]
    # The preceding state never received immediate reward. The analytic
    # discounted returns are 0.6 * (1 - 0.6) * 0.8 and (1 - 0.6) * 0.8.
    for state, expected in ((-0.8, 0.192), (0.8, 0.32)):
        settled, values = agent._values(context(state), None)
        assert all(result["qualified"] for result in settled)
        assert values == pytest.approx((expected, expected), abs=0.015)
    assert agent.inspect()["transitions"] == agent.inspect()["updates"] == 80


def test_checkpoint_cannot_claim_updates_without_any_recorded_transition():
    agent = controller()
    assert record(agent, learn=True)["accepted"]
    data = json.loads(agent.snapshot())
    data.update(transitions=0, records=[])
    with pytest.raises(ValueError, match="Updates require recorded transitions"):
        Reinforcement.from_snapshot(json.dumps(data))


def test_action_only_brain_accepts_an_empty_context():
    cortex = Cortex()
    action = cortex.input("action", shape=2)
    node = cortex.column(patches=1, inputs=action)
    cortex.output("value", shape=(), reads=node)
    agent = Reinforcement(cortex.build(), actions=2)
    assert agent.act({})["accepted"]
    assert agent.feedback(1, terminal=True)["accepted"]


def test_wrong_action_width_or_multiple_values_rejected_at_construction():
    cortex = Cortex()
    action = cortex.input("action", shape=2)
    node = cortex.column(patches=2, inputs=action)
    cortex.output("value", shape=2, reads=node)
    brain = cortex.build()
    with pytest.raises(ValueError, match="scalar-valued"):
        Reinforcement(brain, actions=2)
    with pytest.raises(ValueError, match="one coordinate per action"):
        Reinforcement(brain, actions=3)
    with pytest.raises(ValueError, match="compiled Brain"):
        Reinforcement(cortex, actions=2)


def test_snapshot_identity_is_bound_to_imported_source_not_later_disk_edits(
    monkeypatch,
):
    agent = controller()
    before = agent.snapshot()
    monkeypatch.setattr(
        reinforcement_module, "files", lambda *a: pytest.fail("reread source")
    )
    assert agent.snapshot() == before
    assert Reinforcement.from_snapshot(before).snapshot() == before


def vector_controller(**options):
    cortex = Cortex(seed=3)
    signal = cortex.input("signal", shape=1)
    node = cortex.column(patches=2, inputs=signal)
    cortex.output("left", shape=(), reads=node, indices=(0,))
    cortex.output("right", shape=1, reads=node, indices=(1,))
    return Reinforcement(
        cortex.build(),
        actions=2,
        action_input=None,
        value_output=("left", "right"),
        **options,
    )


def test_vector_values_use_one_joint_query_and_retain_qualified_activity(monkeypatch):
    agent = vector_controller(exploration=0)
    settle = agent.brain.settle
    calls = []

    def query(inputs, **options):
        assert set(inputs) == {"signal"}
        result = settle(inputs, **options)
        calls.append(result)
        return result

    monkeypatch.setattr(agent.brain, "settle", query)
    before = agent.brain.snapshot()
    selected = agent.act(context(), explore=False)
    assert selected["accepted"]
    assert len(calls) == 2  # One joint candidate query, then selected live step.
    assert selected["values"] == tuple(
        calls[0]["outputs"][n][0] for n in ("left", "right")
    )
    assert selected["action"] == max(range(2), key=selected["values"].__getitem__)
    assert agent.brain.state == calls[1]["state"]
    assert agent.brain.snapshot() != before
    expected = Counter(calls[0]["work"])
    expected.update(calls[1]["work"])
    assert selected["work"] == dict(expected)


@pytest.mark.parametrize("terminal", (False, True))
def test_vector_reward_teaches_only_selected_scalar_output(terminal, monkeypatch):
    agent = vector_controller(discount=0.4, reward_scale=2, value_scale=0.8)
    action = agent.act(context())["action"]
    value = agent._value_outputs[action]
    following = context(-0.4)
    before_values = agent._values(following, None)[1]
    expected = 0.6 * 0.8 * 0.5
    if not terminal:
        expected += 0.4 * max(-0.8, min(0.8, max(before_values)))
    fit = agent.brain.observe_batch
    seen = []

    def update(examples, **options):
        seen.extend(copy.deepcopy(examples))
        assert options["source"] == "estimate"
        return fit(examples, **options)

    monkeypatch.setattr(agent.brain, "observe_batch", update)
    result = agent.feedback(1, None if terminal else following, terminal=terminal)
    assert result["accepted"] and result["source"] == "estimate"
    assert result["targets"] == pytest.approx((expected,))
    assert len(seen) == 1 and set(seen[0][1]) == {value.name}
    assert seen[0][0] == {"signal": (0.3,)}


def test_vector_snapshot_pending_and_replay_restore_exactly():
    agent = vector_controller(seed=19, batch_size=2)
    record(agent, reward=0.3, terminal=False)
    assert agent.act(context(-0.5))["accepted"]
    saved = agent.snapshot()
    clone = Reinforcement.from_snapshot(saved)
    assert clone.config["value_output"] == ("left", "right")
    assert clone.snapshot() == saved
    assert clone.feedback(-0.2, terminal=True) == agent.feedback(-0.2, terminal=True)
    assert clone.snapshot() == agent.snapshot()


@pytest.mark.parametrize(
    "names",
    ("left", (), ("left",), ("left", "missing"), ("left", "left"), ("left", [])),
)
def test_invalid_vector_outputs_are_rejected(names):
    brain = vector_controller().brain
    before = brain.snapshot()
    with pytest.raises(ValueError):
        Reinforcement(brain, actions=2, action_input=None, value_output=names)
    assert brain.snapshot() == before


def test_vector_outputs_cannot_alias_the_same_physical_patch():
    cortex = Cortex()
    node = cortex.column(patches=1)
    cortex.output("first", shape=(), reads=node)
    cortex.output("second", shape=(), reads=node)
    with pytest.raises(ValueError, match="distinct patches"):
        Reinforcement(
            cortex.build(),
            actions=2,
            action_input=None,
            value_output=("first", "second"),
        )


def test_vector_values_learn_context_dependent_opposite_choices():
    agent = vector_controller(discount=0, exploration=1, batch_size=8)
    for i in range(48):
        state = -0.7 if i % 2 else 0.7
        action = agent.act(context(state))["action"]
        correct = 0 if state < 0 else 1
        assert agent.feedback(1 if action == correct else -1, terminal=True)["accepted"]
    for state, expected in ((-0.4, 0), (0.4, 1)):
        result = agent.act(context(state), explore=False)
        assert result["accepted"] and result["action"] == expected
        agent.reset()
