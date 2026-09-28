"""Bounded supervised preparation of the same brain used in the live phase."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from ._validation import integer, number
from .brain import Brain
from .ports import _values


def _pairs(brain, examples, name):
    """Validate and own every sample before any admission can occur."""
    if (
        isinstance(examples, (str, bytes))
        or not isinstance(examples, Sequence)
        or not examples
    ):
        raise ValueError(f"{name} must be a nonempty finite sequence of pairs")

    def boundary(supplied, nodes):
        result = {}
        for node in nodes:
            if node.name in supplied:
                values = _values(supplied[node.name], node.shape, node.name)
                result[node.name] = values if node.shape else values[0]
        return result

    records = []
    for index, pair in enumerate(examples):
        try:
            if (
                isinstance(pair, (str, bytes))
                or not isinstance(pair, Sequence)
                or len(pair) != 2
            ):
                raise ValueError("Each example must be an (inputs, targets) pair")
            inputs = boundary(brain._mapping(pair[0], brain._inputs), brain._inputs)
            supplied = brain._mapping(pair[1], brain._outputs, partial=True)
            if not supplied:
                raise ValueError("Supply at least one actual output target")
            targets = boundary(supplied, brain._outputs)
            _, clamps = brain._arguments(inputs, targets)
        except ValueError as error:
            raise ValueError(f"{name}[{index}]: {error}") from error
        records.append((inputs, targets, clamps))
    return tuple(records)


def bootstrap(brain, examples, *, checks, max_error, epochs=20, seed=0, budget=None):
    """Replay supervised witnesses until target-free readiness checks pass.

    ``examples`` and ``checks`` are nonempty finite sequences of
    ``(input_mapping, target_mapping)`` pairs. All samples are validated and
    copied before work. Only examples are admitted; checks are repeatedly used
    for readiness, so they are a development set, not an untouched final test.

    A local seeded RNG shuffles examples each epoch. Before teaching and after
    each complete epoch, pure ``settle`` queries measure maximum absolute error
    against each unique targeted patch coordinate. Readiness requires every
    query to qualify and both recall and checks to meet ``max_error`` in output
    units. ``epochs=0`` only assesses readiness. No inputs or targets are scaled.

    Any refusal stops the helper with ``passed=False``. Earlier accepted
    witnesses remain committed; this is not a batch transaction. The report
    retains work, presentation counts, validation history and failure details.
    Replays are counted as presentations, not newly collected experiences.
    The same brain continues into the live phase without a mode change.
    """
    if not isinstance(brain, Brain):
        raise ValueError("brain must be a Brain")
    epochs = integer(epochs, "epochs")
    seed = integer(seed, "seed")
    budget = None if budget is None else integer(budget, "budget")
    max_error = number(max_error, "max_error")
    if max_error < 0:
        raise ValueError("max_error must be nonnegative")
    examples = _pairs(brain, examples, "examples")
    checks = _pairs(brain, checks, "checks")
    report = {
        "options": {
            "max_error": max_error,
            "epochs": epochs,
            "seed": seed,
            "budget": brain.config["settle_budget"] if budget is None else budget,
        },
        "passed": False,
        "reason": "epochs",
        "epochs": 0,
        "presentations": 0,
        "accepted": 0,
        "examples": len(examples),
        "checks": len(checks),
        "history": [],
        "work": {},
        "failure": None,
    }
    work = Counter()

    def finish(reason):
        report["reason"] = reason
        report["passed"] = reason == "passed"
        report["work"] = dict(work)
        return report

    def refusal(stage, index, result):
        report["failure"] = {
            "stage": stage,
            "index": index,
            "reason": result["reason"],
            "stationarity": result["stationarity"],
        }

    def evaluate(records, stage):
        metric = {"evaluated": 0, "qualified": 0, "max_error": 0.0}
        for index, (inputs, _, clamps) in enumerate(records):
            result = brain.settle(inputs, budget=budget)
            work.update(result["work"])
            metric["evaluated"] += 1
            if not result["qualified"]:
                metric["max_error"] = None
                refusal(stage, index, result)
                break
            metric["qualified"] += 1
            metric["max_error"] = max(
                metric["max_error"],
                max(abs(result["state"][i] - target) for i, target in clamps.items()),
            )
        return metric

    def ready(epoch):
        entry = {"epoch": epoch, "recall": None, "checks": None}
        report["history"].append(entry)
        for stage, records in (("recall", examples), ("checks", checks)):
            entry[stage] = evaluate(records, stage)
            if report["failure"] is not None:
                return False
        return all(
            entry[stage]["max_error"] <= max_error for stage in ("recall", "checks")
        )

    if ready(0):
        return finish("passed")
    if report["failure"] is not None:
        return finish("refused")
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for index in order:
            inputs, targets, _ = examples[index]
            result = brain.observe(inputs, targets, budget=budget)
            work.update(result["work"])
            report["presentations"] += 1
            if not result["accepted"]:
                refusal("observe", index, result)
                return finish("refused")
            report["accepted"] += 1
        report["epochs"] = epoch
        if ready(epoch):
            return finish("passed")
        if report["failure"] is not None:
            return finish("refused")
    return finish("epochs")
