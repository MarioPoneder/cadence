"""Independent Cortex admission, ownership and continuation regressions.

Small deterministic fixtures exercise the public generic learner.  They do
not establish task competence or turn fixed iteration counts into a proof.
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from cadence import Cortex
from cadence import cortex as cortex_module


def model(outputs=2, **kwargs):
    options = dict(
        optimism=0.0,
        epsilon=0.0,
        decay=1.0,
        discount=0.0,
        wiring_id="audit-contexts-v1",
    )
    options.update(kwargs)
    return Cortex(
        n_outputs=outputs, feature_maps=(lambda x: (x[0],), lambda x: ()), **options
    )


def evidence(net):
    """Include allocation and versions: a rejected proposal must not leak them."""
    return tuple(
        tuple(
            (key, c.weight, c.linear, c.square, c.version) for key, c in table.items()
        )
        for table in net._columns
    )


def fail_settlement(monkeypatch, *, call=None, residual=1.0):
    original = cortex_module.el.settle
    calls = []

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        if call is None or len(calls) == call:
            result = dict(
                result,
                converged=False,
                full_residual=residual,
                executed_residual=residual,
            )
        return result

    monkeypatch.setattr(cortex_module.el, "settle", wrapped)
    return calls


@pytest.mark.parametrize(
    "key,value",
    [
        ("n_outputs", 0),
        ("n_outputs", True),
        ("n_outputs", 1.5),
        ("height", True),
        ("height", 0),
        ("height", 1.5),
        ("settle_budget", True),
        ("settle_budget", -1),
        ("settle_budget", 8.5),
        ("decay", 0),
        ("decay", math.nan),
        ("decay", math.inf),
        ("discount", -0.1),
        ("discount", 1),
        ("discount", math.nan),
        ("epsilon", -0.1),
        ("epsilon", 1.1),
        ("epsilon", math.nan),
        ("optimism", -1),
        ("optimism", math.nan),
        ("optimism", math.inf),
        ("coupling", -1),
        ("coupling", math.nan),
        ("target_bound", 0),
        ("target_bound", math.nan),
        ("tolerance", 0),
        ("tolerance", math.nan),
        ("tolerance", math.inf),
        ("damping", 0),
        ("damping", 1.1),
        ("damping", math.nan),
        ("prior_weight", 0),
        ("prior_shape", -1),
        ("prior_rate", math.inf),
        ("meta_shape", 0),
        ("meta_rate", math.nan),
        ("max_columns", -1),
        ("max_columns", 1.5),
        ("max_cache", -1),
        ("max_pending", True),
        ("learning_enabled", "false"),
        ("update_mode", "eventually"),
    ],
)
def test_constructor_rejects_invalid_parameters(key, value):
    with pytest.raises((TypeError, ValueError)):
        Cortex(**{key: value})


def test_empty_feature_maps_rejected():
    with pytest.raises((TypeError, ValueError)):
        Cortex(feature_maps=())
    assert len(Cortex(n_outputs=1).predict((0,))) == 1


@pytest.mark.parametrize("shape,weight", [(0.25, 0.5), (0.1, 0.1)])
def test_prior_parameters_require_a_finite_positive_fixed_precision(shape, weight):
    with pytest.raises(ValueError):
        Cortex(prior_shape=shape, prior_weight=weight)


@pytest.mark.parametrize(
    "parameters",
    [
        {"prior_shape": 1e308, "prior_rate": 1e-308},
        {"prior_weight": 1e-308, "prior_shape": 1.0, "prior_rate": 1e308},
        {"meta_shape": 1e308, "meta_rate": 1e-308, "height": 2},
        {"meta_shape": 1e-308, "meta_rate": 1e308, "height": 3},
        {
            "prior_shape": 1e308,
            "prior_rate": 1.2e308,
            "meta_shape": 1.6e308,
            "meta_rate": 1.0,
            "height": 2,
        },
    ],
)
def test_nonrepresentable_derived_prior_is_rejected_at_construction(parameters):
    with pytest.raises(ValueError):
        Cortex(**parameters)


@pytest.mark.parametrize(
    "name",
    ["coupling", "tolerance", "n_outputs", "decay", "feature_maps", "level_weights"],
)
def test_model_configuration_cannot_change_behind_cached_readouts(name):
    net = model()
    before = net.value((0,), output=0)
    with pytest.raises(AttributeError):
        setattr(net, name, getattr(net, name))
    assert net.value((0,), output=0) == before


def test_default_scalar_regression_acquires_witnessed_value():
    net = Cortex(optimism=0.0, decay=1.0)
    before = net.predict((0.25,))[0]
    for _ in range(6):
        net.observe((0.25,), 2.0)
    after = net.predict((0.25,))[0]
    assert before == 0.0
    assert 0.0 < after < 2.0
    assert abs(after - 2.0) < abs(before - 2.0)
    assert net.value((0.25,))["qualified"] is True


def test_multitarget_and_sparse_admission_preserve_unobserved_output():
    net = model(outputs=3)
    net.observe((4,), (2.0, -2.0, 0.5))
    before = net.predict((4,))
    assert before[0] > 0 > before[1]
    assert before[2] > 0
    net.observe((4,), {1: 2.0})
    after = net.predict((4,))
    assert after[0] == before[0]
    assert after[2] == before[2]
    assert after[1] > before[1]


def test_generic_context_types_remain_distinct_and_mapping_order_is_irrelevant():
    net = Cortex(decay=1.0, optimism=0.0)
    contexts = [True, 1, 1.0, [1], (1,), {"a": 1, "b": 2}]
    for i, context in enumerate(contexts):
        net.observe(context, float(i + 1))
    predictions = [net.predict(x)[0] for x in contexts]
    assert len(set(predictions)) == len(contexts)
    assert net.predict({"b": 2, "a": 1})[0] == predictions[-1]


def test_event_retry_is_idempotent_and_conflicts_or_gaps_are_rejected():
    net = model()
    net.observe((0,), [1, -1], event=1)
    before = net.snapshot()
    net.observe((0,), [1, -1], event=1)
    assert net.snapshot() == before
    for observation, targets, event in [
        ((1,), [1, -1], 1),
        ((0,), [2, -1], 1),
        ((0,), [1, -1], 3),
        ((0,), [1, -1], True),
    ]:
        with pytest.raises((TypeError, ValueError)):
            net.observe(observation, targets, event=event)
        assert net.snapshot() == before
    net.observe((0,), [2, -2], event=2)
    with pytest.raises(ValueError):
        net.observe((0,), [1, -1], event=1)


def test_queued_retry_is_not_a_second_witness_and_direct_updates_cannot_pass_it():
    net, control = model(), model()
    net.learn((0,), 0, 1, (1,), event=1)
    before = net.snapshot()
    net.learn((0,), 0, 1, (1,), event=1)
    assert net.snapshot() == before
    with pytest.raises((ValueError, RuntimeError)):
        net.observe((0,), [1, 2], event=2)
    assert net.snapshot() == before
    with pytest.raises(ValueError):
        net.observe((0,), [1, 2], event=1)
    control.learn((0,), 0, 1, (1,), event=1)
    net.flush()
    control.flush()
    assert evidence(net) == evidence(control)


def test_failed_direct_admission_does_not_consume_event(monkeypatch):
    net = model()
    fail_settlement(monkeypatch)
    try:
        net.observe((0,), [1, -1], event=1)
    except cortex_module.SettlementError:
        pass
    assert net.counters["updates"] == 0
    assert not any(net._columns)
    monkeypatch.undo()
    net.observe((0,), [1, -1], event=1)
    assert net.predict((0,))[0] > 0


def test_step_update_mode_is_immediate_and_terminal_truncation_are_distinct():
    terminal = model(update_mode="step", discount=0.5)
    truncated = model(update_mode="step", discount=0.5)
    for net in (terminal, truncated):
        net.observe((1,), [4, 4])
    terminal.learn((0,), 0, 0, (1,), terminal=True)
    truncated.learn((0,), 0, 0, (1,), truncated=True)
    assert truncated.predict((0,))[0] > terminal.predict((0,))[0]


@pytest.mark.parametrize("options", [{"update_mode": "step"}, {}])
def test_failed_flush_cannot_cross_required_transition_boundary(monkeypatch, options):
    net = model(**options)
    fail_settlement(monkeypatch)
    try:
        net.learn((0,), 0, 1, (1,), truncated=True, event=1)
    except cortex_module.SettlementError:
        pass
    before = net.snapshot()
    with pytest.raises((ValueError, RuntimeError)):
        net.learn((2,), 1, 2, (3,), event=2)
    assert net.snapshot() == before
    monkeypatch.undo()
    assert net.flush() > 0
    net.learn((2,), 1, 2, (3,), event=2)


@pytest.mark.parametrize(
    "override",
    [
        {"reward": math.nan},
        {"reward": math.inf},
        {"reward": "1"},
        {"terminal": 1},
        {"truncated": "false"},
        {"action": True},
        {"action": -1},
        {"action": 2},
    ],
)
def test_transition_validation_is_atomic(override):
    net = model()
    kwargs = dict(
        observation=(0,),
        action=0,
        reward=1.0,
        next_observation=(1,),
        terminal=False,
        truncated=False,
    )
    kwargs.update(override)
    before = net.snapshot()
    with pytest.raises((TypeError, ValueError)):
        net.learn(**kwargs)
    assert net.snapshot() == before


@pytest.mark.parametrize(
    "targets",
    [
        (1,),
        (1, 2, 3),
        {2: 1},
        {-1: 1},
        {True: 1},
        {0: math.nan},
        (1, math.inf),
        (1, "2"),
    ],
)
def test_bad_targets_reject_before_mutation(targets):
    net = model()
    before = net.snapshot()
    with pytest.raises((TypeError, ValueError)):
        net.observe((1,), targets)
    assert net.snapshot() == before


@pytest.mark.parametrize("weight", [0, -1, True, math.nan, math.inf])
def test_bad_witness_mass_rejects_atomically(weight):
    net = model()
    before = net.snapshot()
    with pytest.raises((TypeError, ValueError)):
        net.observe((0,), [1, 2], weight=weight)
    assert net.snapshot() == before


@pytest.mark.parametrize("action", [True, -1, 2, 0.5])
def test_output_index_validation(action):
    net = model()
    before = net.snapshot()
    with pytest.raises((TypeError, ValueError)):
        net.value((0,), action)
    assert net.snapshot() == before


def test_cached_belief_is_detached_from_caller_mutation():
    net = model()
    belief = net.value((0,), 0)
    expected = copy.deepcopy(belief)
    belief["mean"] = 999.0
    belief["qualified"] = False
    assert net.value((0,), 0) == expected


def test_read_only_queries_do_not_allocate_persistent_evidence():
    net = model(max_columns=1, max_cache=2)
    for i in range(7):
        net.predict((i,))
        assert len(net._cache) <= 2
    assert not any(net._columns)
    assert net.counters["updates"] == 0


@pytest.mark.parametrize("method", ["predict", "act"])
def test_unqualified_readout_cannot_predict_or_act(monkeypatch, method):
    net = model(epsilon=1.0)
    rng_before = net._rng.getstate()
    fail_settlement(monkeypatch)
    with pytest.raises(cortex_module.SettlementError):
        getattr(net, method)((0,))
    assert net.counters["updates"] == 0
    assert net._rng.getstate() == rng_before


def test_actual_zero_budget_cannot_act_or_admit_a_nonfixed_state():
    net = model(settle_budget=0, epsilon=1.0)
    value = net.value((0,), output=0)
    assert not value["qualified"]
    assert math.isfinite(value["residual"]) and value["residual"] > net.tolerance
    with pytest.raises(cortex_module.SettlementError):
        net.act((0,))
    result = net.observe((0,), [1, -1], event=1)
    assert not result["accepted"]
    assert net.counters["updates"] == 0
    assert not any(net._columns)


def test_unqualified_diagnostic_is_explicit_and_retry_is_not_cached(monkeypatch):
    net = model()
    calls = fail_settlement(monkeypatch, call=1)
    first = net.value((0,), 0)
    assert first["qualified"] is False and first["residual"] > net.tolerance
    second = net.value((0,), 0)
    assert second["qualified"] is True
    assert len(calls) == 2


def test_late_output_failure_rolls_back_every_level_and_output(monkeypatch):
    net = model(outputs=3)
    net.observe((0,), [1, -1, 0.5])
    before = evidence(net)
    updates = net.counters["updates"]
    calls = fail_settlement(monkeypatch, call=2)
    try:
        net.observe((0,), [2, -2, 1])
    except cortex_module.SettlementError:
        pass
    assert len(calls) >= 2
    assert all("readback1" in result["messages"] for result in calls)
    assert evidence(net) == before
    assert net.counters["updates"] == updates


def test_bootstrap_failure_keeps_pending_experience_and_evidence(monkeypatch):
    net = model()
    net.learn([0], 0, 1.0, [1], terminal=False)
    before = evidence(net)
    fail_settlement(monkeypatch)
    try:
        net.flush()
    except cortex_module.SettlementError:
        pass
    assert evidence(net) == before
    assert net.counters["updates"] == 0
    monkeypatch.undo()
    net.flush()
    assert net.counters["updates"] > 0
    assert net.predict((0,))[0] > 0


def test_pending_observations_are_owned_at_submission():
    net, control = model(), model()
    observation, successor = [0], [1]
    net.learn(observation, 0, 1.0, successor)
    control.learn([0], 0, 1.0, [1])
    observation[0], successor[0] = 99, 100
    net.flush()
    control.flush()
    assert evidence(net) == evidence(control)


def test_failed_backward_flush_retries_only_the_unadmitted_transition(monkeypatch):
    net, control = model(), model()
    for item in (net, control):
        item.learn((0,), 0, 1.0, (1,), event=1)
        item.learn((2,), 1, 2.0, (3,), event=2)
    original = net._admit
    attempts = []

    def reject_second(*args, **kwargs):
        attempts.append(args)
        return 0 if len(attempts) == 2 else original(*args, **kwargs)

    monkeypatch.setattr(net, "_admit", reject_second)
    with pytest.raises(cortex_module.SettlementError):
        net.flush()
    assert net.stats()["pending"] == 1
    assert net.counters["updates"] == net.levels
    monkeypatch.undo()
    net.flush()
    control.flush()
    assert evidence(net) == evidence(control)
    assert net.counters["updates"] == control.counters["updates"]


def test_disabling_learning_preserves_pending_for_explicit_reenable():
    net = model()
    net.learn((0,), 0, 1.0, (1,))
    net.learning_enabled = False
    before = net.snapshot()
    assert net.flush() == 0
    assert net.snapshot() == before
    net.learning_enabled = True
    assert net.flush() > 0
    assert net.predict((0,))[0] > 0


def test_weighted_generic_evidence_matches_repeated_equal_witnesses():
    weighted, repeated = model(), model()
    weighted.observe((0,), [1.5, -0.5], weight=2.0)
    repeated.observe((0,), [1.5, -0.5])
    repeated.observe((0,), [1.5, -0.5])
    assert weighted.predict((0,)) == repeated.predict((0,))
    for weighted_table, repeated_table in zip(
        weighted._columns, repeated._columns, strict=True
    ):
        assert weighted_table.keys() == repeated_table.keys()
        for key in weighted_table:
            assert weighted_table[key].stats() == repeated_table[key].stats()


@pytest.mark.parametrize("meta_rate", [None, 2.0])
def test_observer_precision_accounts_for_displaced_live_belief(meta_rate):
    # Independently sum E[(Y-y_j)^2] over weighted witnessed points.  The
    # incoming mean is displaced by a coarse prior from the evidence mean.
    points = [(-1.0, 1.0), (2.0, 2.0), (3.0, 0.5)]
    weight = sum(w for _, w in points)
    own_mean = sum(y * w for y, w in points) / weight
    scatter = sum(w * (y - own_mean) ** 2 for y, w in points)
    live_mean, live_variance = -0.25, 0.3
    observer = cortex_module.LevelObserver(
        weight, scatter, 0, own_mean, prior_shape=1.7, prior_rate=0.9
    )
    inbox = {"readback0": (live_mean, live_variance)}
    if meta_rate is not None:
        inbox["0:meta_feedback"] = meta_rate
    expected_square_error = sum(
        w * (live_variance + (live_mean - y) ** 2) for y, w in points
    )
    expected = (1.7 + weight / 2) / (
        (0.9 if meta_rate is None else meta_rate) + expected_square_error / 2
    )
    assert observer.emit(None, inbox) == pytest.approx(expected, rel=1e-14)


def test_feature_failure_cannot_allocate_a_partial_context_chain():
    def broken(_):
        raise ValueError("feature failure")

    net = Cortex(
        n_outputs=1,
        feature_maps=(lambda x: (x[0],), broken),
        wiring_id="broken-fixture",
    )
    before = net.snapshot()
    with pytest.raises(ValueError, match="feature failure"):
        net.observe([0], 1.0)
    assert net.snapshot() == before


def test_context_and_cache_capacity_are_bounded_without_evidence_eviction():
    net = Cortex(max_columns=3, max_cache=1, optimism=0.0)
    for i in range(3):
        net.observe((i,), 1.0)
        net.predict((i,))
        assert len(net._cache) <= 1
    before = evidence(net)
    with pytest.raises((ValueError, RuntimeError)):
        net.observe((3,), 1.0)
    assert evidence(net) == before
    assert net.predict((0,))[0] > 0


def test_multilevel_allocation_limit_is_checked_before_any_commit():
    net = model(max_columns=3)
    before = evidence(net)
    with pytest.raises((ValueError, RuntimeError)):
        net.observe((0,), [1, -1])
    assert evidence(net) == before
    assert net.counters["updates"] == 0


def test_pending_capacity_does_not_drop_existing_transition():
    net = model(max_pending=1)
    net.learn((0,), 0, 1.0, (1,))
    before = net.snapshot()
    with pytest.raises((ValueError, RuntimeError)):
        net.learn((2,), 1, -1.0, (3,))
    assert net.snapshot() == before
    net.flush()
    assert net.predict((0,))[0] > 0
    assert net.predict((2,))[1] == 0


def test_snapshot_exactly_continues_random_actions_and_pending_learning():
    net = model(epsilon=1.0, seed=57)
    for _ in range(7):
        net.act((0,))
    net.learn([2], 1, 2.0, [3])
    net.prior_ports_cut = True
    saved = net.snapshot()
    twin = model(epsilon=1.0, seed=57)
    twin.restore(saved)
    assert twin.snapshot() == saved
    assert [net.act((0,)) for _ in range(20)] == [twin.act((0,)) for _ in range(20)]
    net.flush()
    twin.flush()
    assert evidence(net) == evidence(twin)


@pytest.mark.parametrize("context", ["named", 3, ("nested", (1, 2)), (None, True, 1.5)])
def test_custom_context_type_survives_checkpoint(context):
    maps = (lambda _: context,)
    net = Cortex(feature_maps=maps, wiring_id="typed-context")
    net.observe((0,), 1.0)
    saved = net.snapshot()
    twin = Cortex(feature_maps=maps, wiring_id="typed-context")
    twin.restore(saved)
    assert twin.predict((0,)) == net.predict((0,))
    assert evidence(twin) == evidence(net)


def test_custom_wiring_needs_explicit_checkpoint_identity():
    net = Cortex(feature_maps=(lambda _: (),))
    with pytest.raises(ValueError):
        net.snapshot()


def test_restore_rejects_configuration_or_wiring_mismatch_atomically():
    source = model()
    source.observe((1,), [1.0, -1.0])
    for destination in (
        model(coupling=0.5),
        model(wiring_id="another-mapping"),
        model(seed=99),
    ):
        destination.learn((3,), 1, 2.0, (4,))
        before = destination.snapshot()
        with pytest.raises(ValueError):
            destination.restore(source.snapshot())
        assert destination.snapshot() == before


def test_malformed_checkpoint_is_atomic_even_with_pending_experience():
    net = model()
    net.observe((1,), [1.0, -1.0])
    net.learn((3,), 1, 2.0, (4,))
    before = net.snapshot()
    state = json.loads(before)
    state["schema"] = "unknown/100"
    for malformed in ("{", "[]", "null", json.dumps(state)):
        with pytest.raises((TypeError, ValueError)):
            net.restore(malformed)
        assert net.snapshot() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "impossible_moments",
        "negative_weight",
        "bool_version",
        "duplicate_row",
        "bool_cursor",
        "empty_latest",
        "unknown_latest_kind",
        "pending_reward",
        "pending_action",
        "pending_event",
        "rng_boolean",
        "negative_counter",
        "invalid_context",
        "unknown_field",
        "inconsistent_updates",
        "zero_cursor_with_evidence",
    ],
)
def test_checkpoint_rejects_inconsistent_state_atomically(mutation):
    net = model()
    net.observe((1,), [1.0, -1.0], event=1)
    net.learn((3,), 1, 2.0, (4,), event=2)
    before = net.snapshot()
    state = json.loads(before)
    if mutation == "impossible_moments":
        state["rows"][0][3:6] = [1.0, 10.0, 0.0]
    elif mutation == "negative_weight":
        state["rows"][0][3] = -1.0
    elif mutation == "bool_version":
        state["rows"][0][6] = True
    elif mutation == "duplicate_row":
        state["rows"].append(copy.deepcopy(state["rows"][0]))
    elif mutation == "bool_cursor":
        state["cursor"] = True
    elif mutation == "empty_latest":
        state["last_record"] = "{}"
    elif mutation == "unknown_latest_kind":
        last = json.loads(state["last_record"])
        last["kind"] = "invented"
        state["last_record"] = json.dumps(last, sort_keys=True, separators=(",", ":"))
    elif mutation == "pending_reward":
        state["pending"][-1][2] = 6.0
    elif mutation == "pending_action":
        state["pending"][-1][1] = 0
    elif mutation == "pending_event":
        state["pending"][-1][-1] = 3
    elif mutation == "rng_boolean":
        state["rng"][1][0] = True
    elif mutation == "negative_counter":
        state["counters"]["updates"] = -1
    elif mutation == "invalid_context":
        state["rows"][0][1] = ["unknown", 0]
    elif mutation == "inconsistent_updates":
        state["counters"]["updates"] += 1
    elif mutation == "zero_cursor_with_evidence":
        state["cursor"] = 0
        state["last_record"] = None
        state["pending"] = []
    else:
        state["unrecognized"] = 1
    with pytest.raises((TypeError, ValueError)):
        net.restore(json.dumps(state))
    assert net.snapshot() == before


def test_checkpoint_duplicate_json_fields_and_nonfinite_literals_rejected():
    net = model()
    net.observe((0,), [1, -1])
    before = net.snapshot()
    # A normal JSON parser silently accepts the last duplicate field.
    duplicate = '{"cursor":0,' + before[1:]
    nonfinite = before.replace('"cursor":1', '"cursor":NaN')
    for text in (duplicate, nonfinite):
        assert text != before
        with pytest.raises(ValueError):
            net.restore(text)
        assert net.snapshot() == before


def test_builtin_wiring_from_snapshot_is_self_contained():
    net = Cortex.from_dimensions(
        2, n_outputs=2, bounds=(-1, 1), bins=4, depth=2, optimism=0.0, seed=21
    )
    net.observe((0.25, -0.75), [1.5, -1.5])
    net.learn((0, 0), 1, 0.5, (0.5, -0.5))
    saved = net.snapshot()
    twin = Cortex.from_snapshot(saved)
    assert twin.snapshot() == saved
    assert twin.predict((0.25, -0.75)) == net.predict((0.25, -0.75))
    net.flush()
    twin.flush()
    assert evidence(twin) == evidence(net)
