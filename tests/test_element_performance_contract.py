"""Numeric and payload contracts that hot-loop optimizations must preserve."""

import math
from decimal import Decimal
from fractions import Fraction
from numbers import Real

import pytest

from cadence import element as el


class FloatSubclass(float):
    def __float__(self):
        return -2.5


class IntSubclass(int):
    def __float__(self):
        return 3.25


class FractionSubclass(Fraction):
    pass


class RegisteredReal:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return self.value


Real.register(RegisteredReal)


class FloatLike:
    def __float__(self):
        return 1.0


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0.0),
        (-11, -11.0),
        (10**300, 1e300),
        (1.25, 1.25),
        (Fraction(1, 3), 1.0 / 3.0),
        (FractionSubclass(3, 8), 0.375),
        (Decimal("0.125"), 0.125),
        (FloatSubclass(9.0), -2.5),
        (IntSubclass(9), 3.25),
        (RegisteredReal(2.25), 2.25),
    ],
)
def test_finite_conversion_preserves_numeric_subclass_and_real_protocol(
    value, expected
):
    result = el._finite(value, "fixture")
    assert type(result) is float
    assert result == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "1.0",
        None,
        1 + 0j,
        FloatLike(),
        10**400,
        -(10**400),
        math.nan,
        math.inf,
        -math.inf,
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        Decimal("1e400"),
        RegisteredReal(math.nan),
        RegisteredReal(math.inf),
    ],
)
def test_finite_rejects_unsupported_or_unrepresentable_values(value):
    with pytest.raises(ValueError):
        el._finite(value, "fixture")


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        Decimal("-0"),
        -1,
        Fraction(-1, 2),
        FloatSubclass(1.0),
        Fraction(1, 10**400),
    ],
)
def test_positive_flag_rejects_signed_zero_negative_and_underflow(value):
    with pytest.raises(ValueError):
        el._finite(value, "fixture", positive=True)


@pytest.mark.parametrize("value", [0.0, -0.0, Decimal("-0")])
def test_unrestricted_zero_keeps_its_conversion_sign(value):
    result = el._finite(value, "fixture")
    assert result == 0.0
    assert math.copysign(1.0, result) == math.copysign(1.0, float(value))


def test_positive_accepts_smallest_float_and_does_not_use_subclass_payload():
    smallest = math.nextafter(0.0, 1.0)
    assert el._finite(smallest, "fixture", positive=True) == smallest
    assert el._finite(IntSubclass(-99), "fixture", positive=True) == 3.25


class Constant:
    def __init__(self, payload):
        self.payload = payload

    def emit(self, port, inbox):
        return self.payload


def test_mutating_table_inbox_cannot_poison_current_messages_or_next_emission():
    # Emitters are required to be pure. Even a violating custom emitter must
    # not gain an alias to retained messages through its supplied inbox.
    expected = {(0,): Fraction(1, 3), (1,): Fraction(2, 3)}

    class Mutator:
        def emit(self, port, inbox):
            assert inbox["feed"] == expected
            observed = inbox["feed"][(0,)]
            inbox["feed"][(0,)] = Fraction(999)
            inbox["feed"].pop((1,))
            inbox.clear()
            return float(observed)

    mutator = Mutator()
    source = Constant(dict(expected))
    ports = [
        el.Port("first", mutator, object(), "scalar", 0.0),
        el.Port("feed", source, mutator, "table", dict(expected)),
        el.Port("second", mutator, object(), "scalar", 0.0),
    ]
    first = el.settle(ports, budget=4)
    assert first["converged"] and first["full_residual"] == 0.0
    assert first["messages"]["feed"] == expected
    assert first["messages"]["first"] == first["messages"]["second"] == 1.0 / 3
    assert ports[1].message == source.payload == expected
    assert el.settle(ports, budget=4) == first
    first["messages"]["feed"][(0,)] = Fraction(99)
    assert ports[1].message == source.payload == expected
    assert el.settle(ports, budget=4)["messages"]["feed"] == expected


def test_moments_are_canonical_and_each_emitter_gets_a_fresh_inbox_dict():
    class Mutator:
        def emit(self, port, inbox):
            assert inbox["feed"] == (1.25, 0.5)
            assert type(inbox["feed"]) is tuple
            assert all(type(value) is float for value in inbox["feed"])
            answer = inbox["feed"]
            inbox["feed"] = (999.0, 999.0)
            inbox["extra"] = 1
            return answer

    mutator = Mutator()
    raw = [Decimal("1.25"), Fraction(1, 2)]
    source = Constant(raw)
    feed = el.Port("feed", source, mutator, "moments", raw)
    ports = [
        feed,
        el.Port("first", mutator, object(), "moments", (0, 1)),
        el.Port("second", mutator, object(), "moments", (0, 1)),
    ]
    result = el.settle(ports, budget=4)
    assert result["converged"]
    assert result["messages"] == {
        key: (1.25, 0.5) for key in ("feed", "first", "second")
    }
    assert feed.message == (1.25, 0.5)
    assert raw == [Decimal("1.25"), Fraction(1, 2)]
    assert el.settle(ports, budget=4) == result


@pytest.mark.parametrize(
    "family,invalid",
    [
        ("scalar", True),
        ("scalar", math.nan),
        ("moments", [0.0, math.inf]),
        ("table", {(0,): -1}),
    ],
)
def test_mutated_public_port_seed_is_revalidated_on_entry(family, invalid):
    initial = {"scalar": 1.0, "moments": (0.0, 1.0), "table": {(0,): 1}}[family]
    port = el.Port("test", Constant(initial), object(), family, initial)
    port.message = invalid
    with pytest.raises(ValueError):
        el.settle([port])


@pytest.mark.parametrize("damping", [1.0, 0.5])
def test_optimized_scalar_settlement_matches_closed_form_and_repeated_run(damping):
    ports = el.scalar_ports(3, 1, 5)
    before = tuple(port.message for port in ports)
    result = el.settle(ports, damping=damping)
    # Independent fixed point: mean=1/3, centered scatter=14/3,
    # tau=(1+(3-1)/2)/(1+(14/3)/2)=3/5, variance=1/(3*tau).
    assert result["converged"]
    assert result["messages"]["readback"] == pytest.approx((1 / 3, 5 / 9), abs=1e-11)
    assert result["messages"]["feedback"] == pytest.approx(3 / 5, abs=1e-11)
    assert result["full_residual"] <= 1e-11
    assert tuple(port.message for port in ports) == before
    assert el.settle(ports, damping=damping) == result
