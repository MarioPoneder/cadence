"""Adversarial checks of equation qualification and exact evidence custody."""

import json
import math
from fractions import Fraction
from itertools import product

import pytest

from cadence import element as el


class Constant:
    def __init__(self, value):
        self.value = value

    def emit(self, port, inbox):
        return self.value


@pytest.mark.parametrize("budget", [-1, True, 0.5, math.nan, math.inf, "8"])
def test_budget_is_not_silently_coerced(budget):
    with pytest.raises(ValueError):
        el.settle([], budget=budget)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1"])
def test_invalid_scalar_message_never_qualifies(value):
    source = Constant(value)
    port = el.Port("test", source, object(), "scalar", 0.0)
    with pytest.raises(ValueError):
        el.settle([port])


def test_zero_budget_and_damping_cannot_hide_unsolved_equations():
    patch = Constant(1.0)
    port = el.Port("test", patch, object(), "scalar", 0.0)
    assert not el.settle([port], budget=0)["converged"]
    result = el.settle([port], budget=2, damping=1e-30)
    assert not result["converged"]
    assert result["full_residual"] > 0.9
    assert el.settle([port], budget=2)["converged"]


def test_duplicate_ports_and_invalid_moments_are_rejected():
    source = Constant(1.0)
    ports = [el.Port("same", source, object(), "scalar", 0.0) for _ in range(2)]
    with pytest.raises(ValueError):
        el.settle(ports)
    with pytest.raises(ValueError):
        el.Port("bad", source, object(), "moments", (0.0, -1.0))
    with pytest.raises(ValueError):
        el.settle(None)


def test_table_messages_remain_exact_under_large_numeric_tolerance():
    class Toggle:
        def emit(self, port, inbox):
            return {(0,): 1 - inbox["loop"][(0,)], (1,): inbox["loop"][(0,)]}

    patch = Toggle()
    port = el.Port("loop", patch, patch, "table", {(0,): 0, (1,): 1})
    assert not el.settle([port], budget=2, tolerance=100)["converged"]
    with pytest.raises(ValueError):
        el.settle([port], damping=0.5)


@pytest.mark.parametrize(
    "args", [(0, 0, 0), (1, 2, 1), (1, 0, -1), (math.inf, 0, 0), (True, 0, 0)]
)
def test_impossible_or_nonfinite_moments_rejected(args):
    with pytest.raises(ValueError):
        el.settle_scalar(*args)


def test_float_roundoff_allowed_but_real_moment_violation_rejected():
    el._stats(1.0, 1.0, math.nextafter(1.0, 0.0))
    with pytest.raises(ValueError):
        el._stats(1.0, 1.0, 0.9)
    with pytest.raises(ValueError):
        el._stats(Fraction(1), Fraction(1), Fraction(999, 1000))


@pytest.mark.parametrize(
    "text", ["1e999999999", "1e-999999999", "1.0", "01", "-0", "2/2", "1/0", "+1", " 1"]
)
def test_checkpoint_rational_parser_rejects_noncanonical_and_exponent_forms(text):
    with pytest.raises(ValueError):
        el._parse_fraction(text)


def test_large_canonical_rational_roundtrip_without_global_int_limit_changes():
    value = Fraction(1, 2**15000 + 1)
    text = el._fraction_text(value)
    assert el._parse_fraction(text) == value
    with pytest.raises(ValueError):
        el._parse_fraction(text, max_bits=128)


def test_element_atomic_restore_checks_cursor_latest_and_moments():
    column = el.CorticalColumn(capacity=24)
    column.add(2)
    before = column.snapshot()
    mutations = (
        {"s1": "999"},
        {"s2": "-1"},
        {"w": "5"},
        {"cursor": True},
        {"last": "1"},
        {"s2": "1e999999"},
    )
    for change in mutations:
        state = {**json.loads(before), **change}
        with pytest.raises(ValueError):
            column.restore(json.dumps(state))
        assert column.snapshot() == before
    with pytest.raises(ValueError):
        column.restore(before[:-1] + ',"cursor":1}')
    assert column.snapshot() == before


def test_checkpoints_require_current_class_schema():
    from cadence.column import CorticalColumn

    scalar = el.CorticalColumn(capacity=24)
    column = CorticalColumn(capacity=24)
    for receiver, source in ((scalar, column), (column, scalar)):
        saved = receiver.snapshot()
        with pytest.raises(ValueError):
            receiver.restore(source.snapshot())
        assert receiver.snapshot() == saved
    malformed = json.loads(scalar.snapshot())
    malformed["schema"] = "unrecognized/1"
    with pytest.raises(ValueError):
        scalar.restore(json.dumps(malformed))


def test_invalid_cluster_assignment_and_edges_fail_closed():
    factor = {"scope": ("x",), "values": {(False,): 1, (True,): 1}}
    with pytest.raises(ValueError):
        el.solve_cluster_forest([factor], [])
    factor["values"] = {(0,): 1, (1,): 1}
    with pytest.raises(ValueError):
        el.solve_cluster_forest([factor], [None])


def test_cluster_forest_matches_independent_global_enumeration():
    scopes = (("x", "y"), ("y", "z"), ("z", "q"), ("u",))
    rules = (
        lambda a: 1 + 3 * int(a[0] == a[1]),
        lambda a: 2 + 5 * int(a[0] != a[1]),
        lambda a: 1 + a[0] + 2 * a[1],
        lambda a: 1 + 2 * a[0],
    )
    factors = [
        {
            "scope": scope,
            "values": {
                a: Fraction(rule(a)) for a in product((0, 1), repeat=len(scope))
            },
        }
        for scope, rule in zip(scopes, rules, strict=True)
    ]
    beliefs = el.solve_cluster_forest(factors, [(0, 1), (1, 2)])
    variables = ("x", "y", "z", "q", "u")
    joint = {}
    for assignment in product((0, 1), repeat=len(variables)):
        world = dict(zip(variables, assignment, strict=True))
        weight = Fraction(1)
        for factor in factors:
            weight *= factor["values"][tuple(world[v] for v in factor["scope"])]
        joint[assignment] = weight
    total = sum(joint.values())
    for belief in beliefs:
        for values, probability in belief["values"].items():
            oracle = (
                sum(
                    weight
                    for assignment, weight in joint.items()
                    if tuple(assignment[variables.index(v)] for v in belief["scope"])
                    == values
                )
                / total
            )
            assert probability == oracle
    factors[-1]["values"] = {(0,): Fraction(0), (1,): Fraction(0)}
    with pytest.raises(ValueError, match="Zero joint support"):
        el.solve_cluster_forest(factors, [(0, 1), (1, 2)])


def test_source_can_emit_on_multiple_ports_without_aliasing_an_inbox():
    class Reader:
        def emit(self, port, inbox):
            return sum(inbox.values())

    source, left, right = Constant(2.0), Reader(), Reader()
    ports = [
        el.Port("a", source, left, "scalar", 0.0),
        el.Port("b", source, right, "scalar", 0.0),
        el.Port("c", left, right, "scalar", 0.0),
    ]
    result = el.settle(ports)
    assert result["converged"]
    assert result["messages"] == {"a": 2.0, "b": 2.0, "c": 2.0}
