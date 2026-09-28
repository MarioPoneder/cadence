"""The public CorticalColumn: height semantics and element identity."""

import pytest

from cadence import CorticalColumn
from cadence import element as el


def test_height_validates():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            CorticalColumn(height=bad)


def test_height_one_is_the_element_exactly():
    public, scalar = CorticalColumn(height=1), el.CorticalColumn()
    for event, value in enumerate([-2, 2, -2, 0, 2], 1):
        assert public.observe(event, value) == scalar.observe(event, value)
    ours, theirs = public.query(), scalar.query()
    assert ours["stages"] == {}
    assert {k: ours[k] for k in theirs} == theirs  # identical shared fields
    ours_i, theirs_i = public.observer_intervention(), scalar.observer_intervention()
    assert {k: ours_i[k] for k in theirs_i} == theirs_i


def test_taller_columns_settle_and_read_back():
    for height in (2, 3):
        column = CorticalColumn(height=height)
        for event, value in enumerate([-2, 2, -2], 1):
            assert column.observe(event, value)["accepted"] is True
        answer = column.query()
        assert answer["qualified"] is True
        assert sorted(answer["stages"]) == list(range(2, height + 1))
        result = column.observer_intervention()
        assert abs(result["observer_after"] - result["observer_before"]) > 1e-7
        assert abs(result["lower_after"] - result["lower_before"]) > 1e-7
        assert result["recovered_qualified"] is True
        assert result["recovered_residual"] <= 1e-10
        for lesion in result["lesions"].values():
            assert lesion["converged"] is True and lesion["full_residual"] > 1e-7


def test_height_changes_the_settled_state():
    short, tall = CorticalColumn(height=1), CorticalColumn(height=2)
    for event, value in enumerate([-2, 2, -2], 1):
        short.observe(event, value)
        tall.observe(event, value)
    assert short.query()["precision"] != tall.query()["precision"]


def test_checkpoint_binds_height():
    tall = CorticalColumn(height=2)
    tall.observe(1, 2)
    saved = tall.snapshot()
    twin = CorticalColumn(height=2)
    twin.restore(saved)
    assert twin.snapshot() == saved
    with pytest.raises(ValueError):
        CorticalColumn(height=3).restore(saved)
