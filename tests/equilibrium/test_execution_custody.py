"""Executed-action acknowledgments must survive retries and failed learning."""

import copy
import json

import pytest

from cadence.experimental.equilibrium import Reinforcement

from .test_reinforcement import context, controller, vector_controller


@pytest.mark.parametrize("vector", (False, True))
def test_body_override_teaches_only_the_action_actually_executed(monkeypatch, vector):
    agent = (vector_controller if vector else controller)(discount=0, batch_size=1)
    proposed = agent.act(context())
    executed = 1 - proposed["action"]
    seen = []
    fit = agent.brain.observe_batch

    def captured(examples, **options):
        seen.extend(copy.deepcopy(examples))
        return fit(examples, **options)

    monkeypatch.setattr(agent.brain, "observe_batch", captured)
    result = agent.feedback(
        0.6,
        terminal=True,
        decision_id=proposed["decision_id"],
        executed_action=executed,
    )
    assert result["accepted"] and result["stored"] and not result["duplicate"]
    assert result["targets"] == pytest.approx((0.54,))
    row = json.loads(agent.snapshot())["records"][0]
    assert row[1] == executed and row[5] == proposed["decision_id"]
    if vector:
        assert set(seen[0][1]) == {agent._value_outputs[executed].name}
    else:
        assert seen[0][0]["action"] == tuple(float(i == executed) for i in range(2))


def test_delayed_retry_cannot_credit_or_consume_the_new_pending_action(monkeypatch):
    agent = controller()
    first = agent.act(context())
    evidence = dict(decision_id=first["decision_id"], executed_action=first["action"])
    assert agent.feedback(0.3, context(-0.2), learn=False, **evidence)["stored"]
    second = agent.act(context(-0.2))
    before = agent.snapshot()

    def unexpected(*args, **kwargs):
        pytest.fail("Duplicate evidence attempted learning")

    monkeypatch.setattr(agent, "replay", unexpected)
    duplicate = agent.feedback(0.3, context(-0.2), learn=True, budget=0, **evidence)
    assert duplicate == dict(
        accepted=False,
        stored=False,
        duplicate=True,
        decision_id=first["decision_id"],
        reason="duplicate_feedback",
        transitions=1,
        work={},
    )
    assert agent.snapshot() == before
    assert agent.inspect()["pending_decision_id"] == second["decision_id"]


@pytest.mark.parametrize("changed", ("action", "reward", "context", "terminal"))
def test_latest_identity_rejects_conflicting_evidence_atomically(changed):
    agent = controller()
    first = agent.act(context())
    evidence = dict(
        decision_id=first["decision_id"],
        executed_action=first["action"],
        reward=0.3,
        next_inputs=context(-0.2),
        learn=False,
    )
    agent.feedback(**evidence)
    agent.act(context(-0.2))
    before = agent.snapshot()
    if changed == "action":
        evidence["executed_action"] = 1 - first["action"]
    elif changed == "reward":
        evidence["reward"] = -0.3
    elif changed == "context":
        evidence["next_inputs"] = context(0.2)
    else:
        evidence.update(terminal=True, next_inputs=None)
    with pytest.raises(ValueError, match="conflicts"):
        agent.feedback(**evidence)
    assert agent.snapshot() == before


def test_reset_cancels_execution_credit_without_reusing_decision_identity():
    agent = controller()
    canceled = agent.act(context())
    agent.reset()
    active = agent.act(context())
    assert (canceled["decision_id"], active["decision_id"]) == (1, 2)
    before = agent.snapshot()
    for identity in (canceled["decision_id"], active["decision_id"] + 1):
        with pytest.raises(ValueError, match="pending accepted act"):
            agent.feedback(
                0.4,
                terminal=True,
                decision_id=identity,
                executed_action=active["action"],
            )
        assert agent.snapshot() == before
    result = agent.feedback(
        0.4,
        terminal=True,
        decision_id=active["decision_id"],
        executed_action=active["action"],
        learn=False,
    )
    assert result["stored"] and result["transitions"] == 1
    saved = agent.snapshot()
    assert Reinforcement.from_snapshot(saved).snapshot() == saved


def test_refused_proposal_consumes_no_execution_identity():
    agent = controller()
    before = agent.snapshot()
    refused = agent.act(context(), budget=0)
    assert not refused["accepted"] and refused["decision_id"] is None
    assert agent.snapshot() == before
    accepted = agent.act(context())
    assert accepted["decision_id"] == 1


@pytest.mark.parametrize("failure", ("refused", "exception"))
def test_record_and_retry_receipt_survive_failed_fit_and_checkpoint(failure, monkeypatch):
    agent = controller(discount=0, batch_size=1)
    proposed = agent.act(context())
    brain = agent.brain.snapshot()
    rng = json.loads(agent.snapshot())["rng"]
    evidence = dict(
        reward=-0.5,
        terminal=True,
        decision_id=proposed["decision_id"],
        executed_action=proposed["action"],
    )
    if failure == "refused":
        result = agent.feedback(**evidence, budget=0)
        assert result["stored"] and not result["accepted"]
    else:
        with monkeypatch.context() as patch:

            def failed(*args, **kwargs):
                raise ValueError("failed fit")

            patch.setattr(agent.brain, "observe_batch", failed)
            with pytest.raises(ValueError, match="failed fit"):
                agent.feedback(**evidence)
    assert agent.brain.snapshot() == brain
    saved = agent.snapshot()
    assert json.loads(saved)["rng"] == rng
    clone = Reinforcement.from_snapshot(saved)
    for owner in (agent, clone):
        assert owner.feedback(**evidence)["duplicate"]
        assert owner.snapshot() == saved
        assert owner.replay()["accepted"]
    assert clone.snapshot() == agent.snapshot()
    assert agent.inspect()["transitions"] == agent.inspect()["updates"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision_id", None),
        ("decision_id", True),
        ("decision_id", 0),
        ("decision_id", -1),
        ("decision_id", 1.0),
        ("decision_id", "1"),
        ("executed_action", None),
        ("executed_action", True),
        ("executed_action", -1),
        ("executed_action", 2),
        ("executed_action", 0.0),
        ("executed_action", "0"),
    ],
)
def test_malformed_execution_acknowledgment_is_atomic(field, value):
    agent = controller()
    proposed = agent.act(context())
    evidence = dict(decision_id=proposed["decision_id"], executed_action=proposed["action"])
    evidence[field] = value
    before = agent.snapshot()
    with pytest.raises(ValueError):
        agent.feedback(0.4, terminal=True, **evidence)
    assert agent.snapshot() == before


@pytest.mark.parametrize("omitted", ("decision_id", "executed_action"))
def test_execution_acknowledgment_cannot_be_inferred_from_the_proposal(omitted):
    agent = controller()
    proposed = agent.act(context())
    evidence = dict(decision_id=proposed["decision_id"], executed_action=proposed["action"])
    del evidence[omitted]
    before = agent.snapshot()
    with pytest.raises(TypeError):
        agent.feedback(0.4, terminal=True, **evidence)
    assert agent.snapshot() == before


def test_latest_retry_and_order_survive_replay_eviction_and_new_pending_snapshot():
    agent = controller(capacity=1, batch_size=1)
    history = []
    for _ in range(3):
        proposed = agent.act(context())
        evidence = dict(
            decision_id=proposed["decision_id"],
            executed_action=proposed["action"],
        )
        agent.feedback(0.2, terminal=True, learn=False, **evidence)
        history.append(evidence)
    next_proposal = agent.act(context())
    saved = agent.snapshot()
    clone = Reinforcement.from_snapshot(saved)
    assert clone.snapshot() == saved
    for owner in (agent, clone):
        assert owner.feedback(0.2, terminal=True, **history[-1])["duplicate"]
        with pytest.raises(ValueError, match="pending accepted act"):
            owner.feedback(0.2, terminal=True, **history[0])
        assert owner.snapshot() == saved
        assert owner.inspect()["pending_decision_id"] == next_proposal["decision_id"]


def test_learning_admission_identity_is_independent_of_executed_decisions():
    agent = controller(discount=0, batch_size=1)
    first = agent.act(context())
    agent.feedback(
        0.3,
        terminal=True,
        decision_id=first["decision_id"],
        executed_action=first["action"],
    )
    assert agent.replay()["accepted"]
    assert agent.brain.observe({**context(), "action": [1, 0]}, {"value": [0.3]}, event_id=100)[
        "accepted"
    ]
    second = agent.act(context())
    result = agent.feedback(
        0.2,
        terminal=True,
        decision_id=second["decision_id"],
        executed_action=second["action"],
    )
    assert result["accepted"] and result["event_id"] == 101
    assert result["decision_id"] == 2 and result["transitions"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "reinforcement/2"),
        ("issued_decisions", True),
        ("issued_decisions", 0),
        ("last_feedback", None),
        ("last_feedback", [True, "0" * 64]),
        ("last_feedback", [1, "0" * 64]),
        ("last_feedback", [2, "0" * 64]),
        ("pending", [{"signal": [0.3]}, 0, 1]),
        ("pending", [{"signal": [0.3]}, 0, 3]),
    ],
)
def test_checkpoint_rejects_inconsistent_execution_custody(field, value):
    agent = controller()
    proposed = agent.act(context())
    agent.feedback(
        0.3,
        terminal=True,
        decision_id=proposed["decision_id"],
        executed_action=proposed["action"],
        learn=False,
    )
    agent.act(context())
    data = json.loads(agent.snapshot())
    data[field] = value
    with pytest.raises(ValueError):
        Reinforcement.from_snapshot(json.dumps(data))


def test_checkpoint_rejects_reused_recorded_decision_id():
    agent = controller()
    for _ in range(2):
        proposed = agent.act(context())
        agent.feedback(
            0.3,
            terminal=True,
            decision_id=proposed["decision_id"],
            executed_action=proposed["action"],
            learn=False,
        )
    data = json.loads(agent.snapshot())
    data["records"][1][5] = data["records"][0][5]
    with pytest.raises(ValueError, match="ordering"):
        Reinforcement.from_snapshot(json.dumps(data))


@pytest.mark.parametrize("pending", (False, True))
def test_loader_rejects_issued_identity_gap_without_an_intervening_reset(pending):
    agent = controller()
    first = agent.act(context())
    agent.feedback(
        0.3,
        context(),
        decision_id=first["decision_id"],
        executed_action=first["action"],
        learn=False,
    )
    if pending:
        agent.act(context())
    data = json.loads(agent.snapshot())
    data["issued_decisions"] += 1
    if pending:
        data["pending"][2] += 1
    with pytest.raises(ValueError, match="gaps require reset episodes"):
        Reinforcement.from_snapshot(json.dumps(data))


@pytest.mark.parametrize("pending", (False, True))
def test_loader_rejects_unexplained_decision_gap_in_empty_history(pending):
    agent = controller()
    if pending:
        agent.act(context())
    data = json.loads(agent.snapshot())
    data["issued_decisions"] += 1
    if pending:
        data["pending"][2] += 1
    with pytest.raises(ValueError, match="gaps require reset episodes"):
        Reinforcement.from_snapshot(json.dumps(data))


@pytest.mark.parametrize("terminal", (False, True))
def test_real_cancellation_gap_roundtrips_before_and_after_feedback(terminal):
    agent = controller(capacity=1, batch_size=1)
    first = agent.act(context())
    agent.feedback(
        0.3,
        None if terminal else context(),
        terminal=terminal,
        decision_id=first["decision_id"],
        executed_action=first["action"],
        learn=False,
    )
    for _ in range(2):
        agent.act(context())
        agent.reset()
    agent.reset()  # Empty reset is also a valid credit-segment boundary.
    assert Reinforcement.from_snapshot(agent.snapshot()).snapshot() == agent.snapshot()
    next_decision = agent.act(context())
    assert next_decision["decision_id"] == 4
    clone = Reinforcement.from_snapshot(agent.snapshot())
    for owner in (agent, clone):
        owner.feedback(
            0.4,
            context(),
            decision_id=next_decision["decision_id"],
            executed_action=next_decision["action"],
            learn=False,
        )
        saved = owner.snapshot()
        assert Reinforcement.from_snapshot(saved).snapshot() == saved
    assert clone.snapshot() == agent.snapshot()


def test_empty_resets_and_canceled_proposals_preserve_valid_empty_history():
    agent = controller()
    agent.reset()
    for _ in range(2):
        agent.act(context())
        agent.reset()
    for pending in (False, True):
        if pending:
            agent.act(context())
        saved = agent.snapshot()
        assert Reinforcement.from_snapshot(saved).snapshot() == saved


def test_terminal_boundary_does_not_also_explain_a_canceled_decision():
    agent = controller()
    first = agent.act(context())
    agent.feedback(
        0.3,
        terminal=True,
        decision_id=first["decision_id"],
        executed_action=first["action"],
        learn=False,
    )
    agent.act(context())
    data = json.loads(agent.snapshot())
    data["issued_decisions"] += 1
    data["pending"][2] += 1
    with pytest.raises(ValueError, match="gaps require reset episodes"):
        Reinforcement.from_snapshot(json.dumps(data))


@pytest.mark.parametrize("capacity", (1, 4))
def test_loader_checks_reset_count_for_gaps_in_retained_records(capacity):
    agent = controller(capacity=capacity, batch_size=1)
    for _ in range(2):
        proposed = agent.act(context())
        agent.feedback(
            0.3,
            context(),
            decision_id=proposed["decision_id"],
            executed_action=proposed["action"],
            learn=False,
        )
        agent.act(context())
        agent.reset()
    data = json.loads(agent.snapshot())
    # Remove every reset episode while preserving distinct issued identities.
    # With capacity one, the first retained record must account for the lost
    # prefix; with capacity four, the gap is visible between retained records.
    for row in data["records"]:
        row[4] = 0
    data["episode"] = 0
    with pytest.raises(ValueError, match="gaps require reset episodes|ordering"):
        Reinforcement.from_snapshot(json.dumps(data))
