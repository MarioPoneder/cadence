"""Batch learning is joint, atomic and distinct from ordered live experience."""

import copy
import math
import random

import pytest

from cadence import Brain, Cortex, bootstrap


def learner(seed=2, layout="observer", device="python"):
    cortex = Cortex(seed=seed, device=device)
    signal = cortex.input("signal", shape=1)
    base = cortex.column("base", patches=3, inputs=signal)
    if layout == "flat":
        top = base
    elif layout == "composed":
        top = cortex.column("top", patches=2, inputs=(signal, base))
    else:
        top = cortex.observer("top", patches=2, inputs=signal, observes=base)
    cortex.output("answer", shape=1, reads=top)
    return cortex.build()


def cases(values):
    return [({"signal": [x]}, {"answer": [x]}) for x in values]


def test_singleton_matches_observe_parameters_but_preserves_live_activity():
    brain = learner()
    assert brain.step({"signal": [0.3]})["accepted"]
    single = Brain.from_snapshot(brain.snapshot())
    before = brain.state
    example = cases((0.7,))[0]
    batch_result = brain.observe_batch([example])
    single_result = single.observe(*example)
    assert batch_result["accepted"] and single_result["accepted"]
    for key in ("weights", "biases", "energy", "sweeps", "work", "stationarity"):
        assert batch_result[key] == single_result[key]
    assert batch_result["states"] == (single_result["state"],)
    assert batch_result["outputs"] == (single_result["outputs"],)
    assert brain.state == before
    assert single.state != before
    assert brain.weights == single.weights and brain.biases == single.biases


def test_batch_has_private_clamps_and_one_shared_parameter_set():
    brain = learner()
    original = brain.snapshot()
    examples = cases((-0.8, 0.8))
    untouched = copy.deepcopy(examples)
    result = brain.observe_batch(examples, event_id=9)
    assert result["accepted"] and result["batch_size"] == 2
    assert result["outputs"] == ({"answer": (-0.8,)}, {"answer": (0.8,)})
    assert result["states"][0] != result["states"][1]
    assert "state" not in result
    assert len(result["weights"]) == len(brain.graph.edges)
    assert len(result["biases"]) == brain.graph.n_patches
    assert brain.inspect()["admissions"] == 1
    assert brain.inspect()["last_event_id"] == 9
    assert examples == untouched
    assert brain.snapshot() != original
    saved = brain.snapshot()
    result["outputs"][0]["answer"] = (99,)
    assert brain.snapshot() == saved


def test_reordered_rows_produce_same_parameters_not_last_example_wins():
    left = learner()
    right = Brain.from_snapshot(left.snapshot())
    examples = cases((-0.7, 0.2, 0.6))
    first, second = left.observe_batch(examples), right.observe_batch(examples[::-1])
    assert first["accepted"] and second["accepted"]
    assert left.weights == pytest.approx(right.weights, abs=2e-10)
    assert left.biases == pytest.approx(right.biases, abs=2e-10)
    for a, b in zip(first["states"], reversed(second["states"]), strict=True):
        assert a == pytest.approx(b, abs=2e-10)


def test_inconsistent_targets_are_a_shared_compromise_not_two_independent_models():
    cortex = Cortex()
    node = cortex.column(patches=1)
    cortex.output("answer", shape=1, reads=node)
    brain = cortex.build()
    result = brain.observe_batch([({}, {"answer": [0.8]}), ({}, {"answer": [-0.8]})])
    assert result["accepted"]
    assert result["stationarity"] <= brain.config["tolerance"]
    assert result["prediction_residual"] == pytest.approx(0.8)
    assert brain.biases == (0.0,)
    assert brain.predict({})["answer"] == (0.0,)


def test_latest_batch_retry_is_idempotent_across_checkpoint_and_interleaved_calls():
    brain = learner()
    examples = cases((-0.8, 0.8))
    assert brain.observe_batch(examples, event_id=5)["accepted"]
    restored = Brain.from_snapshot(brain.snapshot())
    assert restored.step({"signal": [0.3]})["accepted"]
    before = restored.snapshot()
    retry = restored.observe_batch(examples, event_id=5, budget="ignored on retry")
    assert retry == {
        "accepted": False,
        "qualified": True,
        "duplicate": True,
        "event_id": 5,
        "batch_size": 2,
    }
    assert restored.snapshot() == before
    for changed, identity in ((examples[::-1], 5), (examples, 4), (examples[:1], 5)):
        with pytest.raises(ValueError, match="conflicts|older"):
            restored.observe_batch(changed, event_id=identity)
        assert restored.snapshot() == before
    with pytest.raises(ValueError, match="conflicts|older"):
        restored.observe(*examples[0], event_id=5)
    assert restored.observe(*examples[0])["event_id"] == 6
    assert restored.observe_batch(examples)["event_id"] == 7
    assert restored.inspect()["admissions"] == 3


def test_refusal_and_late_invalid_row_cannot_partially_admit_or_consume_identity():
    brain = learner()
    examples = cases((-0.8, 0.8))
    before = brain.snapshot()
    refused = brain.observe_batch(examples, event_id=7, budget=0)
    assert not refused["accepted"] and refused["reason"] == "budget"
    assert brain.snapshot() == before
    assert brain.observe_batch(examples, event_id=7)["accepted"]
    before = brain.snapshot()
    with pytest.raises(ValueError, match=r"examples\[2\]"):
        brain.observe_batch([*examples, ({"signal": [math.nan]}, {"answer": [0.4]})])
    assert brain.snapshot() == before


@pytest.mark.parametrize(
    "bad",
    [
        [],
        (),
        None,
        "rows",
        iter(cases((0.3,))),
        [({}, {})],
        [cases((0.3,))[0], ((),)],
        [({"signal": [True]}, {"answer": [0.2]})],
    ],
)
def test_invalid_batch_rejected_before_any_solve(bad, monkeypatch):
    brain = learner()
    before = brain.snapshot()
    monkeypatch.setattr(brain, "_solve", lambda *a, **k: pytest.fail("invalid solve"))
    with pytest.raises(ValueError):
        brain.observe_batch(bad)
    assert brain.snapshot() == before


@pytest.mark.parametrize("budget", (-1, 1.5, True))
def test_invalid_budget_precedes_optional_device_initialization(budget):
    brain = learner(device="cuda")
    before = brain.snapshot()
    with pytest.raises(ValueError, match="budget"):
        brain.observe_batch(cases((-0.8, 0.8)), budget=budget)
    assert brain._engine is None
    assert brain.snapshot() == before


def test_shapes_partial_targets_aliases_and_foreign_handles():
    cortex = Cortex()
    sensor = cortex.input("signal", shape=())
    node = cortex.column(patches=2, inputs=sensor)
    grid = cortex.output("grid", shape=(1, 2), reads=node)
    alias = cortex.output("alias", shape=(), reads=node, indices=(1,))
    brain = cortex.build()
    rows = [
        ({sensor: 0.2}, {grid: [[0.1, 0.2]], alias: 0.2}),
        ({sensor: -0.2}, {alias: -0.2}),
    ]
    result = brain.observe_batch(rows)
    assert result["accepted"]
    assert result["outputs"][0]["grid"] == (0.1, 0.2)
    assert result["outputs"][1]["alias"] == (-0.2,)
    before = brain.snapshot()
    with pytest.raises(ValueError, match="conflicts"):
        brain.observe_batch([({sensor: 0.2}, {grid: [[0.1, 0.2]], alias: -0.2})])
    foreign = Cortex().input("signal", shape=())
    with pytest.raises(ValueError, match="foreign"):
        brain.observe_batch([({foreign: 0.2}, {alias: 0.2})])
    assert brain.snapshot() == before


@pytest.mark.parametrize("layout", ("flat", "composed", "observer"))
@pytest.mark.parametrize("seed", (0, 2, 7))
def test_batch_bootstrap_acquires_and_generalizes_with_atomic_update_accounting(
    layout, seed
):
    brain = learner(seed, layout)
    before = brain.state
    report = bootstrap(
        brain,
        cases((-0.8, 0.8)),
        checks=cases((-0.4, 0.4)),
        max_error=0.15,
        batch_size=2,
    )
    assert report["passed"], report
    assert report["accepted"] == report["presentations"] == 2 * report["updates"]
    assert brain.inspect()["admissions"] == report["updates"] == report["epochs"]
    assert brain.state == before
    assert brain.predict({"signal": [-0.4]})["answer"][0] < -0.25
    assert brain.predict({"signal": [0.4]})["answer"][0] > 0.25


def test_minibatch_replay_and_short_tail_match_manual_admission():
    brain = learner()
    shadow = Brain.from_snapshot(brain.snapshot())
    examples, checks = cases((-0.7, 0.1, 0.6)), cases((-0.3, 0.3))
    report = bootstrap(
        brain, examples, checks=checks, max_error=0, epochs=2, seed=17, batch_size=2
    )
    rng = random.Random(17)
    for _ in range(2):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for offset in range(0, len(order), 2):
            assert shadow.observe_batch(
                [examples[i] for i in order[offset : offset + 2]]
            )["accepted"]
    assert report["updates"] == 4 and report["accepted"] == 6
    assert brain.snapshot() == shadow.snapshot()
    recreated = learner()
    assert bootstrap(recreated, examples, checks=checks, **report["options"]) == report


@pytest.mark.parametrize("size", (0, -1, 1.5, True, None))
def test_bad_batch_size_cannot_start_bootstrap(size):
    brain = learner()
    before = brain.snapshot()
    with pytest.raises(ValueError, match="batch_size"):
        bootstrap(
            brain,
            cases((-0.8, 0.8)),
            checks=cases((-0.4, 0.4)),
            max_error=0.15,
            batch_size=size,
        )
    assert brain.snapshot() == before


def test_bootstrap_batch_refusal_preserves_prior_batches_and_reports_every_row(
    monkeypatch,
):
    brain = learner()
    examples = cases((-0.8, -0.4, 0.4, 0.8))
    observe = brain.observe_batch
    snapshots = []

    def capped_second(rows, **kwargs):
        if snapshots:
            kwargs["budget"] = 0
        result = observe(rows, **kwargs)
        snapshots.append(brain.snapshot())
        return result

    monkeypatch.setattr(brain, "observe_batch", capped_second)
    report = bootstrap(
        brain, examples, checks=cases((-0.2, 0.2)), max_error=0, batch_size=2, seed=0
    )
    assert report["reason"] == "refused"
    assert report["presentations"] == 4 and report["accepted"] == 2
    assert report["updates"] == 1
    assert report["failure"]["stage"] == "observe_batch"
    assert len(report["failure"]["indices"]) == 2
    assert snapshots[0] == snapshots[1] == brain.snapshot()
