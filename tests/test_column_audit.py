"""Configuration, real observations, analytic qualification and continuation."""

import json
import math
from decimal import Decimal
from fractions import Fraction

import pytest

from cadence.column import CorticalColumn, RateObserver, stage_ports


def test_real_data_add_and_explicit_ids_have_identical_continuations():
    auto, explicit = CorticalColumn(), CorticalColumn()
    for event, value in enumerate((0.125, Decimal("1.25"), Fraction(-7, 4), 0), 1):
        assert auto.add(value) == explicit.observe(event, value)
    assert auto.snapshot() == explicit.snapshot()
    assert auto.observe(4, 0.0)["duplicate"]
    before = auto.snapshot()
    auto.query()
    auto.observer_intervention()
    assert auto.snapshot() == before


def test_default_capacity_is_open_but_explicit_event_cap_is_available():
    column = CorticalColumn()
    for _ in range(32):
        assert column.add(0)["accepted"]
    assert column.count == 32
    bounded = CorticalColumn(capacity=1)
    bounded.add(0)
    with pytest.raises(ValueError):
        bounded.add(0)


def test_configured_scalar_fixed_point_matches_closed_form():
    column = CorticalColumn(
        decay=1,
        prior_weight=2,
        prior_shape=3,
        prior_rate=4,
        value_bound=100,
        damping=0.75,
    )
    for value in (10.5, -3.25):
        assert column.add(value)["accepted"]
    # Independent stationary equations: tau=(a+(w-1)/2)/(b+scatter/2).
    weight = 4
    mean = 7.25 / weight
    scatter = 10.5**2 + (-3.25) ** 2 - weight * mean**2
    precision = (3 + (weight - 1) / 2) / (4 + scatter / 2)
    belief = column.query()
    assert belief["qualified"]
    assert belief["answer"] == pytest.approx(mean)
    assert belief["precision"] == pytest.approx(precision, abs=1e-10)
    assert belief["variance"] == pytest.approx(1 / (weight * precision), abs=1e-10)


def test_config_bound_snapshot_factory_and_resume():
    column = CorticalColumn(
        height=3,
        decay=0.9,
        prior_shape=2,
        prior_rate=3,
        meta_shape=5,
        meta_rate=6,
        capacity=40,
        value_bound=20,
    )
    column.add(1.5)
    column.add(-0.25)
    twin = CorticalColumn.from_snapshot(column.snapshot())
    assert twin.config == column.config and twin.height == 3
    assert twin.query() == column.query()
    column.add(2.125)
    twin.add(2.125)
    assert twin.snapshot() == column.snapshot()
    with pytest.raises(TypeError):
        twin.config["decay"] = 0.5
    with pytest.raises(AttributeError):
        twin.height = 1


@pytest.mark.parametrize(
    "config",
    [
        dict(decay=0),
        dict(decay=math.nan),
        dict(capacity=True),
        dict(value_bound=math.inf),
        dict(settle_budget=1.5),
        dict(tolerance=0),
        dict(damping=0),
        dict(damping=1.1),
        dict(prior_weight=0),
        dict(prior_shape=-1),
        dict(prior_rate=0),
        dict(meta_shape=math.nan),
        dict(meta_rate=True),
        dict(max_statistic_bits=32),
        dict(prior_weight=0.1, prior_shape=0.1),
    ],
)
def test_bad_configuration_fails_before_use(config):
    with pytest.raises(ValueError):
        CorticalColumn(**config)


@pytest.mark.parametrize(
    "value", [math.nan, math.inf, True, "1.5", complex(1, 0), Decimal("1e-999999")]
)
def test_invalid_real_input_is_atomic(value):
    column = CorticalColumn()
    before = column.snapshot()
    with pytest.raises(ValueError):
        column.add(value)
    assert column.snapshot() == before


def test_budget_and_arithmetic_limit_rejections_do_not_consume_event():
    column = CorticalColumn(max_statistic_bits=64)
    before = column.snapshot()
    assert not column.add(0.5, budget=0)["accepted"]
    assert column.count == 0 and column.snapshot() == before
    with pytest.raises(ValueError):
        column.add(Fraction(1, 2**100))
    assert column.snapshot() == before
    assert column.add(0.5)["accepted"] and column.count == 1


def test_arithmetic_growth_limit_keeps_previous_valid_checkpoint():
    column = CorticalColumn(max_statistic_bits=64)
    for _ in range(63):
        assert column.add(0)["accepted"]
    before = column.snapshot()
    with pytest.raises(ValueError):
        column.add(0)
    assert column.snapshot() == before
    assert CorticalColumn.from_snapshot(before).snapshot() == before


def test_restore_rejects_changed_configuration_or_boolean_height_atomically():
    column = CorticalColumn(height=2)
    column.add(0.5)
    saved = column.snapshot()
    state = json.loads(saved)
    for change in ({"height": True}, {"decay": "9/10"}, {"prior_shape": True}):
        mutated = {**state, "config": {**state["config"], **change}}
        with pytest.raises(ValueError):
            column.restore(json.dumps(mutated))
        assert column.snapshot() == saved


def test_restore_rejects_exponential_reconstruction_before_allocating_it():
    column = CorticalColumn(decay=Fraction(1, 2**200), max_statistic_bits=1024)
    saved = column.snapshot()
    state = json.loads(saved)
    state.update(cursor=10**100, last="0", w="1")
    with pytest.raises(ValueError, match="reconstruction budget"):
        column.restore(json.dumps(state))
    assert column.snapshot() == saved


def test_exact_scatter_retains_small_spread_on_large_centers():
    from cadence import element as el

    mean = Fraction(10**12)
    weight = Fraction(2)
    _, center, scatter = el._stats(weight, 2 * mean, 2 * mean * mean + 2)
    assert center == 10**12 and scatter == 2


@pytest.mark.parametrize(
    "config",
    [
        dict(prior_weight=1e308, prior_shape=1e308),
        dict(prior_shape=1e-308, prior_rate=1e308),
        dict(prior_weight=1e-308, prior_shape=1, prior_rate=1e308),
        dict(prior_shape=1e-320, prior_rate=1),
    ],
)
def test_prior_configuration_rejects_derived_overflow_or_underflow(config):
    with pytest.raises(ValueError):
        CorticalColumn(**config)


@pytest.mark.parametrize("shape,rate", [(1e308, 1e-308), (1e-308, 1e308)])
def test_meta_ratio_overflow_and_underflow_fail_before_any_query(shape, rate):
    from cadence import element as el

    with pytest.raises(ValueError, match="initial meta precision"):
        CorticalColumn(height=3, meta_shape=shape, meta_rate=rate)
    observer = el.PrecisionObserver(1.0, 0.0)
    with pytest.raises(ValueError, match="initial meta precision"):
        stage_ports(observer, 3, 1.0, meta_shape=shape, meta_rate=rate)
    with pytest.raises(ValueError, match="initial meta precision"):
        RateObserver(1.0, "read", shape=shape, rate=rate)


def test_initial_meta_feedback_overflow_fails_before_port_construction():
    from cadence import element as el

    settings = dict(meta_shape=1.6e308, meta_rate=1.0)
    with pytest.raises(ValueError, match="initial meta feedback"):
        CorticalColumn(height=2, prior_shape=1e308, prior_rate=1.2e308, **settings)
    observer = el.PrecisionObserver(1.0, 0.0)
    with pytest.raises(ValueError, match="initial meta feedback"):
        stage_ports(observer, 2, 1.0, prior_rate=1.2e308, **settings)


@pytest.mark.parametrize("key", ["", 0, False])
def test_rate_observer_rejects_invalid_upper_port_key(key):
    with pytest.raises(ValueError, match="nonempty strings"):
        RateObserver(1.0, "read", key)


def test_rate_observer_rejects_derived_nonfinite_runtime_rate():
    observer = RateObserver(1.0, "read", "up")
    with pytest.raises(ValueError, match="total incoming rate"):
        observer.emit(None, {"read": 1e308, "up": 1e308})
