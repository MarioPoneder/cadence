"""Bootstrap measures free recall, preserves custody and remains ordinary learning."""

import copy
import math
import random
from collections import Counter

import pytest

from cadence import Brain, Cortex, bootstrap


def learner(seed=0):
    cortex = Cortex(seed=seed)
    sensor = cortex.input("signal", shape=1)
    base = cortex.column("base", patches=4, inputs=sensor)
    readout = cortex.observer("readout", patches=2, inputs=sensor, observes=base)
    cortex.output("answer", shape=1, reads=readout)
    return cortex.build()


def cases(values):
    return [({"signal": [value]}, {"answer": [value]}) for value in values]


def replay(brain, examples, epochs, seed):
    rng = random.Random(seed)
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for index in order:
            assert brain.observe(*examples[index])["accepted"]


@pytest.mark.parametrize("seed", (0, 2, 7))
def test_bootstrap_acquires_two_examples_and_unseen_checks_without_teaching_checks(
    seed,
):
    brain = learner(seed)
    shadow = Brain.from_snapshot(brain.snapshot())
    examples, checks = cases((-0.8, 0.8)), cases((-0.4, 0.4))
    before = [brain.predict(inputs)["answer"][0] for inputs, _ in checks]
    report = bootstrap(brain, examples, checks=checks, max_error=0.15, seed=11)

    assert report["passed"]
    assert report["reason"] == "passed"
    assert report["failure"] is None
    assert report["examples"] == report["checks"] == 2
    assert 0 < report["epochs"] <= 20
    assert report["presentations"] == report["accepted"] == 2 * report["epochs"]
    assert brain.inspect()["admissions"] == report["accepted"]
    after = [brain.predict(inputs)["answer"][0] for inputs, _ in checks]
    assert after[0] < -0.25 and after[1] > 0.25
    assert sum(
        abs(a - target) for a, target in zip(after, (-0.4, 0.4), strict=True)
    ) < sum(abs(a - target) for a, target in zip(before, (-0.4, 0.4), strict=True))
    assert report["history"][-1]["recall"]["max_error"] <= 0.15
    assert report["history"][-1]["checks"]["max_error"] <= 0.15

    # Independent manual presentation of ONLY examples reaches the identical
    # full continuation. Evaluation must neither teach checks nor retain state.
    replay(shadow, examples, report["epochs"], seed=11)
    assert brain.snapshot() == shadow.snapshot()

    # Bootstrap adds no closed training phase or lock on the brain.
    later = brain.observe({"signal": [0.2]}, {"answer": [-0.3]})
    assert later["accepted"]
    assert brain.inspect()["admissions"] == report["accepted"] + 1


def test_shuffling_is_private_deterministic_and_checkpoint_continuable():
    original = learner(seed=7)
    clone = Brain.from_snapshot(original.snapshot())
    examples, checks = cases((-0.8, 0.8)), cases((-0.4, 0.4))
    random_state = random.getstate()
    first = bootstrap(
        original, examples, checks=checks, max_error=0.0, epochs=3, seed=29
    )
    second = bootstrap(clone, examples, checks=checks, max_error=0.0, epochs=3, seed=29)
    assert random.getstate() == random_state
    assert first == second
    assert first["options"] == {
        "max_error": 0.0,
        "epochs": 3,
        "seed": 29,
        "budget": original.config["settle_budget"],
    }
    assert not first["passed"] and first["reason"] == "epochs"
    assert first["epochs"] == 3 and first["presentations"] == 6
    assert [entry["epoch"] for entry in first["history"]] == [0, 1, 2, 3]
    assert original.snapshot() == clone.snapshot()

    # The report retains the gate and shuffle/work settings for a fresh replay.
    repeated = learner(seed=7)
    assert bootstrap(repeated, examples, checks=checks, **first["options"]) == first
    assert repeated.snapshot() == original.snapshot()

    restored = Brain.from_snapshot(original.snapshot())
    for brain in (original, restored):
        bootstrap(brain, examples, checks=checks, max_error=0.0, epochs=2, seed=73)
    assert original.snapshot() == restored.snapshot()


def test_zero_epochs_evaluates_without_learning_and_a_ready_brain_stops_early():
    brain = learner()
    checkpoint = brain.snapshot()
    examples, checks = cases((-0.8, 0.8)), cases((-0.4, 0.4))
    failed = bootstrap(brain, examples, checks=checks, max_error=0.0, epochs=0)
    assert not failed["passed"] and failed["reason"] == "epochs"
    assert failed["epochs"] == failed["presentations"] == failed["accepted"] == 0
    assert len(failed["history"]) == 1
    assert brain.snapshot() == checkpoint

    exact = cases((0.0,))
    passed = bootstrap(brain, exact, checks=exact, max_error=0.0, epochs=20, budget=0)
    assert passed["passed"] and passed["reason"] == "passed"
    assert passed["epochs"] == passed["presentations"] == passed["accepted"] == 0
    assert brain.snapshot() == checkpoint


@pytest.mark.parametrize("stage", ("recall", "checks"))
def test_a_refused_query_never_passes_even_with_a_generous_error_threshold(stage):
    brain = learner()
    checkpoint = brain.snapshot()
    examples = cases((0.8,)) if stage == "recall" else cases((0.0,))
    report = bootstrap(brain, examples, checks=cases((0.4,)), max_error=100, budget=0)
    assert not report["passed"]
    assert report["reason"] == "refused"
    assert report["failure"]["stage"] == stage
    assert report["failure"]["index"] == 0
    assert report["failure"]["reason"] == "budget"
    assert report["failure"]["stationarity"] > brain.config["tolerance"]
    assert report["presentations"] == report["accepted"] == 0
    assert report["history"][0][stage] == {
        "evaluated": 1,
        "qualified": 0,
        "max_error": None,
    }
    if stage == "recall":
        assert report["history"][0]["checks"] is None
    assert brain.snapshot() == checkpoint


@pytest.mark.parametrize("failing_split", ("examples", "checks"))
def test_both_recall_and_checks_must_meet_the_error_threshold(failing_split):
    brain = learner()
    parts = {"examples": cases((0.0,)), "checks": cases((0.0,))}
    parts[failing_split][0][1]["answer"][0] = 0.8
    report = bootstrap(
        brain, parts["examples"], checks=parts["checks"], max_error=0.1, epochs=0
    )
    assert not report["passed"] and report["reason"] == "epochs"
    failed_metric = "recall" if failing_split == "examples" else "checks"
    assert report["history"][0][failed_metric]["max_error"] == 0.8


def test_work_includes_queries_and_teaching_and_evaluation_has_no_clamps(monkeypatch):
    brain = learner()
    counted = Counter()
    queried = taught = 0
    settle, observe = brain.settle, brain.observe

    def query(inputs, **kwargs):
        nonlocal queried
        assert "targets" not in kwargs and "interventions" not in kwargs
        result = settle(inputs, **kwargs)
        counted.update(result["work"])
        queried += 1
        return result

    def teach(*args, **kwargs):
        nonlocal taught
        result = observe(*args, **kwargs)
        counted.update(result["work"])
        taught += 1
        return result

    monkeypatch.setattr(brain, "settle", query)
    monkeypatch.setattr(brain, "observe", teach)
    report = bootstrap(
        brain, cases((-0.8, 0.8)), checks=cases((-0.4, 0.4)), max_error=0.0, epochs=2
    )
    assert report["work"] == dict(counted)
    assert taught == report["presentations"] == 4
    assert queried == 4 * (report["epochs"] + 1)


def test_post_epoch_evaluation_refusal_stops_without_discarding_learning(monkeypatch):
    brain = learner()
    settle = brain.settle

    def exhaust_after_teaching(inputs, **kwargs):
        if brain.inspect()["admissions"]:
            kwargs["budget"] = 0
        return settle(inputs, **kwargs)

    monkeypatch.setattr(brain, "settle", exhaust_after_teaching)
    report = bootstrap(
        brain, cases((-0.8, 0.8)), checks=cases((-0.4, 0.4)), max_error=0.0
    )
    assert not report["passed"] and report["reason"] == "refused"
    assert report["epochs"] == 1
    assert report["accepted"] == report["presentations"] == 2
    assert report["failure"]["stage"] == "recall"
    assert brain.inspect()["admissions"] == 2


def test_admission_refusal_preserves_previous_successful_witnesses():
    cortex = Cortex(seed=2)
    left = cortex.input("left", shape=1)
    right = cortex.input("right", shape=1)
    base = cortex.column("base", patches=4, inputs=(left, right))
    hidden = cortex.column("hidden", patches=3, inputs=base)
    observer = cortex.observer("observer", patches=2, observes=(base, hidden))
    cortex.output("horizontal", shape=1, reads=observer, indices=(0,))
    cortex.output("vertical", shape=1, reads=observer, indices=(1,))
    brain = cortex.build()
    shadow = Brain.from_snapshot(brain.snapshot())
    examples = [
        (
            {"left": [-0.8], "right": [value]},
            {"horizontal": [-0.8], "vertical": [-value]},
        )
        for value in (-0.8, 0.8)
    ]
    checks = [({"left": [0.4], "right": [0.4]}, {"horizontal": [0.4]})]
    report = bootstrap(
        brain, examples, checks=checks, max_error=0.01, seed=0, budget=512
    )
    assert not report["passed"] and report["reason"] == "refused"
    assert report["failure"]["stage"] == "observe"
    assert report["failure"]["index"] == 1
    assert report["presentations"] == 2 and report["accepted"] == 1
    assert report["epochs"] == 0
    assert shadow.observe(*examples[0], budget=512)["accepted"]
    assert brain.snapshot() == shadow.snapshot()


def test_scalar_multidimensional_partial_targets_and_aliases_score_actual_coordinates():
    cortex = Cortex(seed=7)
    sensor = cortex.input("image", shape=(1, 2))
    column = cortex.column("coordinates", patches=2, inputs=sensor)
    grid = cortex.output("grid", shape=(1, 2), reads=column, indices=(1, 0))
    alias = cortex.output("alias", shape=(), reads=column, indices=(0,))
    brain = cortex.build()
    inputs = {sensor: [[0.2, -0.5]]}
    predicted = brain.predict(inputs)
    targets = {grid: [[0.25, -0.3]], alias: -0.3}
    expected = max(abs(predicted["grid"][0] - 0.25), abs(predicted["grid"][1] + 0.3))
    partial = {alias: predicted["alias"][0]}
    checkpoint = brain.snapshot()
    report = bootstrap(
        brain, [(inputs, targets)], checks=[(inputs, partial)], max_error=1, epochs=0
    )
    assert report["passed"]
    assert report["history"][0]["recall"]["max_error"] == pytest.approx(expected)
    assert report["history"][0]["checks"]["max_error"] == 0
    assert brain.snapshot() == checkpoint

    conflict = {grid: [[0.25, -0.3]], alias: 0.3}
    with pytest.raises(ValueError):
        bootstrap(
            brain,
            [(inputs, targets), (inputs, conflict)],
            checks=[(inputs, partial)],
            max_error=1,
        )
    assert brain.snapshot() == checkpoint


def test_scalar_sensor_and_witness_remain_scalars_during_learning():
    cortex = Cortex(seed=2)
    sensor = cortex.input("scalar", shape=())
    column = cortex.column("relation", patches=1, inputs=sensor)
    output = cortex.output("scalar_answer", shape=(), reads=column)
    brain = cortex.build()
    examples = [({sensor: value}, {output: value}) for value in (-0.6, 0.6)]
    checks = [({sensor: value}, {output: value}) for value in (-0.3, 0.3)]
    report = bootstrap(brain, examples, checks=checks, max_error=0.08)
    assert report["passed"]
    assert brain.predict({sensor: -0.3})["scalar_answer"][0] < -0.22
    assert brain.predict({sensor: 0.3})["scalar_answer"][0] > 0.22


@pytest.mark.parametrize(
    "argument,value",
    [
        ("max_error", -0.1),
        ("max_error", math.nan),
        ("max_error", math.inf),
        ("max_error", True),
        ("epochs", -1),
        ("epochs", 1.5),
        ("epochs", True),
        ("seed", -1),
        ("seed", 1.5),
        ("seed", True),
        ("budget", -1),
        ("budget", 1.5),
        ("budget", True),
    ],
)
def test_invalid_options_are_rejected_before_mutation(argument, value):
    brain = learner()
    checkpoint = brain.snapshot()
    options = {"checks": cases((0.4,)), "max_error": 0.1, argument: value}
    with pytest.raises(ValueError):
        bootstrap(brain, cases((-0.8, 0.8)), **options)
    assert brain.snapshot() == checkpoint


@pytest.mark.parametrize("split", ("examples", "checks"))
@pytest.mark.parametrize(
    "bad",
    [
        ({"signal": [0.2]}, {}),
        ({"signal": [0.2, 0.3]}, {"answer": [0.2]}),
        ({"signal": [math.nan]}, {"answer": [0.2]}),
        ({"signal": [0.2]}, {"answer": [2.0]}),
        ({"missing": [0.2]}, {"answer": [0.2]}),
        ({"signal": [0.2]}, {"missing": [0.2]}),
        ({"signal": [0.2]},),
    ],
)
def test_a_late_invalid_example_or_check_cannot_partially_train(
    split, bad, monkeypatch
):
    brain = learner()
    checkpoint = brain.snapshot()

    def query_before_validation(*args, **kwargs):
        pytest.fail("All samples must be validated before even pure numerical queries")

    monkeypatch.setattr(brain, "settle", query_before_validation)
    parts = {"examples": cases((-0.8, 0.8)), "checks": cases((-0.4, 0.4))}
    parts[split].append(bad)
    with pytest.raises(ValueError):
        bootstrap(brain, parts["examples"], checks=parts["checks"], max_error=0.1)
    assert brain.snapshot() == checkpoint


@pytest.mark.parametrize("split", ("examples", "checks"))
@pytest.mark.parametrize("container", ([], (), "rows", b"rows", None))
def test_dataset_containers_must_be_nonempty_finite_sequences(split, container):
    brain = learner()
    checkpoint = brain.snapshot()
    parts = {"examples": cases((-0.8, 0.8)), "checks": cases((-0.4, 0.4))}
    parts[split] = container
    with pytest.raises(ValueError):
        bootstrap(brain, parts["examples"], checks=parts["checks"], max_error=0.1)
    assert brain.snapshot() == checkpoint


@pytest.mark.parametrize("split", ("examples", "checks"))
def test_single_use_iterators_are_rejected_before_learning(split):
    brain = learner()
    checkpoint = brain.snapshot()
    parts = {"examples": cases((-0.8, 0.8)), "checks": cases((-0.4, 0.4))}
    parts[split] = iter(parts[split])
    with pytest.raises(ValueError):
        bootstrap(brain, parts["examples"], checks=parts["checks"], max_error=0.1)
    assert brain.snapshot() == checkpoint


def test_samples_are_copied_before_the_first_admission(monkeypatch):
    brain = learner()
    shadow = Brain.from_snapshot(brain.snapshot())
    examples, checks = cases((-0.8, 0.8)), cases((-0.4, 0.4))
    pristine_examples, pristine_checks = copy.deepcopy((examples, checks))
    observe = brain.observe

    def mutate_callers_data(*args, **kwargs):
        examples[1][0].clear()
        checks[0][1]["answer"][0] = math.nan
        return observe(*args, **kwargs)

    monkeypatch.setattr(brain, "observe", mutate_callers_data)
    actual = bootstrap(brain, examples, checks=checks, max_error=0.0, epochs=1)
    expected = bootstrap(
        shadow, pristine_examples, checks=pristine_checks, max_error=0.0, epochs=1
    )
    assert actual == expected
    assert brain.snapshot() == shadow.snapshot()
    assert not examples[1][0]
