"""The element's contracts: admission custody, readback, federation."""

from fractions import Fraction as F
from itertools import product

import pytest

from cadence import solve_cluster_forest
from cadence.element import CorticalColumn


def factor(scope, rule):
    return {
        "scope": tuple(scope),
        "values": {a: F(rule(a)) for a in product((0, 1), repeat=len(scope))},
    }


def test_admission_is_transactional_and_ordered():
    column = CorticalColumn()
    before = column.snapshot()
    rejected = column.observe(1, -2, budget=0)  # zero budget cannot commit
    assert not rejected["accepted"] and column.snapshot() == before
    assert column.observe(1, -2)["accepted"] is True
    retry = column.observe(1, -2)  # identical latest retry is a duplicate
    assert retry["duplicate"] is True and not retry["accepted"]
    with pytest.raises(ValueError):
        column.observe(1, 2)  # conflicting retry
    with pytest.raises(ValueError):
        column.observe(3, 0)  # skipped identifier
    with pytest.raises(ValueError):
        column.observe(2, 0, kind="derived")  # not owned experience


def test_capacity_and_checkpoint_custody():
    column = CorticalColumn(capacity=24)
    for event in range(1, 25):
        assert column.observe(event, 0)["accepted"] is True
    with pytest.raises(ValueError):
        column.observe(25, 0)  # declared capacity
    saved = column.snapshot()
    assert len(saved.encode()) <= 4096
    twin = CorticalColumn(capacity=24)
    twin.restore(saved)
    assert twin.snapshot() == saved and twin.query() == column.query()
    with pytest.raises(ValueError):
        twin.restore(saved.replace('"cursor":24', '"cursor":-1'))


def test_history_discrimination_and_read_purity():
    outcomes = []
    for sign in (-1, 1):
        column = CorticalColumn()
        for event, value in enumerate([sign * 2] * 3 + [0], 1):
            column.observe(event, value)
        saved = column.snapshot()
        outcomes.append(column.query()["answer"])
        for _ in range(5):
            column.query()
        assert column.snapshot() == saved  # queries acquire no witnesses
    assert outcomes[0] < -0.5 and outcomes[1] > 0.5


def test_observer_readback_is_reciprocal_and_recoverable():
    column = CorticalColumn()
    for event, value in enumerate([-2, 2, -2], 1):
        column.observe(event, value)
    result = column.observer_intervention()
    assert abs(result["observer_after"] - result["observer_before"]) > 1e-7
    assert abs(result["lower_after"] - result["lower_before"]) > 1e-7
    assert result["recovered_qualified"] is True
    assert result["recovered_residual"] <= 1e-10
    assert len(result["lesions"]) == 2
    for lesion in result["lesions"].values():
        assert lesion["converged"] is True and lesion["full_residual"] > 1e-7


def test_exact_frustrated_cycle_and_contradiction():
    soft = factor(
        ("a", "b", "c"),
        lambda a: (
            (3 if a[0] != a[1] else 1)
            * (3 if a[1] != a[2] else 1)
            * (3 if a[2] != a[0] else 1)
        ),
    )
    boundary = factor(("a", "b"), lambda a: 1)
    beliefs = CorticalColumn().solve_factors([soft, boundary], [(0, 1)])
    unequal = sum(v for k, v in beliefs[1]["values"].items() if k[0] != k[1])
    assert unequal == F(9, 14)  # exact frustrated posterior, not 3/4
    hard = factor(
        ("a", "b", "c"), lambda a: int(a[0] != a[1] and a[1] != a[2] and a[2] != a[0])
    )
    with pytest.raises(ValueError):
        CorticalColumn().solve_factors([hard, boundary], [(0, 1)])


def test_uncertified_structures_are_rejected():
    def pair(p):
        return factor(p, lambda a: 3 if a[0] != a[1] else 1)

    with pytest.raises(ValueError):
        solve_cluster_forest(
            [pair(("a", "b")), pair(("b", "c")), pair(("c", "a"))],
            [(0, 1), (1, 2), (2, 0)],
        )  # cycle
    bad = factor(("a",), lambda a: 1)
    bad["values"][(0,)] = 0.5
    with pytest.raises(ValueError):
        solve_cluster_forest([bad], [])  # float potential
