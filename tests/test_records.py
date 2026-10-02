"""Records: a reading touches few records, and the witnessed outcome is written into exactly those."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def test_batched_draws_equal_single_draws_and_normals_are_standard() -> None:
    a, b = cd.Mulberry32(12345), cd.Mulberry32(12345)
    single = np.array([a.random() for _ in range(1000)])
    assert np.array_equal(single, b.batch(1000)) and a.state == b.state
    z = cd.Mulberry32(7).normals(20001)
    assert z.shape == (20001,) and abs(z.mean()) < 0.03 and abs(z.std() - 1.0) < 0.03


def test_a_code_keeps_the_active_cells_at_unit_length() -> None:
    records = cd.Records(20, {"y": 3}, cells=500, active=10, seed=1)
    code = records.code(np.random.default_rng(0).random((4, 20)))
    assert code.shape == (2, 4, 500)
    assert ((code[0] > 0).sum(axis=1) <= 10).all()
    np.testing.assert_allclose(np.linalg.norm(code[0], axis=1), 1.0)


def test_one_write_is_read_back_and_other_readings_move_by_their_overlap() -> None:
    rng = np.random.default_rng(0)
    records = cd.Records(40, {"y": 4}, cells=4000, active=20, rate=1.0, habituation=0.0, seed=3)
    a = (rng.random(40) < 0.3).astype(float)
    b = 1.0 - a
    code_a, code_b = records.code(a)[:, 0], records.code(b)[:, 0]
    target = np.array([0.0, 1.0, 0.0, 0.0])
    assert records.write(code_a, {"y": target}) == 1
    np.testing.assert_allclose(records.read(code_a)["y"], target, atol=1e-12)
    np.testing.assert_allclose(records.read(code_b)["y"], (code_a[0] @ code_b[0]) * target, atol=1e-12)


def test_records_learn_a_walk_from_its_own_correlated_stream() -> None:
    size, rng = 24, np.random.default_rng(1)
    records = cd.Records(size + 2, {"next": size}, cells=3000, active=12, rate=0.5, seed=4)
    here = 0
    for _ in range(3000):
        action = int(rng.integers(2))
        reading = np.zeros(size + 2)
        reading[here], reading[size + action] = 2.0, 2.0
        nxt = (here + (1 if action else -1)) % size
        records.write(records.code(reading, adapt=True)[:, 0], {"next": np.eye(size)[nxt]})
        here = nxt
    correct = 0
    for cell in range(size):
        for action in range(2):
            reading = np.zeros(size + 2)
            reading[cell], reading[size + action] = 2.0, 2.0
            predicted = int(np.argmax(records.read(records.code(reading)[:, 0])["next"]))
            correct += predicted == (cell + (1 if action else -1)) % size
    assert correct == 2 * size


def test_habituation_lets_a_small_cue_move_the_code() -> None:
    def overlap(habituation: float) -> float:
        records = cd.Records(32, {"y": 1}, cells=2000, active=20, habituation=habituation, seed=5)
        first, second = np.full(32, 2.0), np.full(32, 2.0)
        first[30], second[31] = 3.0, 3.0
        for _ in range(3000):
            records.code(np.stack([first, second]), adapt=True)
        codes = records.code(np.stack([first, second]))[0]
        return float(codes[0] @ codes[1])

    assert overlap(0.01) < 0.5 < overlap(0.0)


def test_valued_fields_read_the_normalised_code_at_their_own_rate() -> None:
    records = cd.Records(
        10, {"y": 2, "value": 1}, cells=300, active=10, rate=0.1, valued=["value"],
        valued_rate=1.0, habituation=0.0, pathways=[np.arange(8), np.arange(8, 10)],
        pathway_rate=1.0, seed=2,
    )
    reading = np.zeros(10)
    reading[:8], reading[8] = 1.0, 1.0
    code = records.code(reading, adapt=True)[:, 0]
    assert not np.array_equal(code[0], code[1])
    records.write(code, {"value": np.array([1.0]), "y": np.array([1.0, 0.0])})
    read = records.read(code)
    np.testing.assert_allclose(read["value"], [1.0], atol=1e-12)
    np.testing.assert_allclose(read["y"], [0.1, 0.0], atol=1e-12)


def test_known_masks_unobserved_entries_and_imagined_readings_do_not_adapt() -> None:
    records = cd.Records(5, {"y": 2}, cells=50, active=5, rate=1.0, seed=6)
    code = records.code(np.ones(5))[:, 0]
    assert not records.mean.any()
    records.write(code, {"y": np.array([1.0, 1.0])}, {"y": np.array([True, False])})
    np.testing.assert_allclose(records.read(code)["y"], [1.0, 0.0], atol=1e-12)
    records.code(np.ones(5), adapt=True)
    assert records.mean.any()


def test_the_cells_are_rebuilt_from_the_seed_and_arguments_are_checked() -> None:
    a = cd.Records(6, {"y": 1}, cells=40, active=4, seed=9)
    b = cd.Records(**{k: v for k, v in a.to_dict().items() if k not in ("inputs", "fields")},
                   inputs=6, fields={"y": 1})
    assert np.array_equal(a.projection, b.projection) and np.array_equal(a.offset, b.offset)
    assert a.parameters() == 40
    with pytest.raises(ValueError):
        cd.Records(0, {"y": 1})
    with pytest.raises(ValueError):
        cd.Records(5, {"y": 1}, cells=10, active=11)
    with pytest.raises(ValueError):
        cd.Records(5, {"y": 1}, valued=["z"])
    with pytest.raises(ValueError):
        a.code(np.ones(5))
    with pytest.raises(ValueError):
        a.write(a.code(np.ones(6))[:, 0], {"y": np.ones(2)})



def test_the_mean_settles_to_the_average_of_the_readings() -> None:
    records = cd.Records(3, {"y": 1}, cells=30, active=3, habituation=1e-6, seed=7)
    readings = np.random.default_rng(4).random((500, 3))
    for row in readings:
        records.code(row, adapt=True)
    np.testing.assert_allclose(records.mean, readings.mean(axis=0), atol=1e-9)
    assert records.seen == 500


def test_task_sets_give_each_task_its_own_value_cells_and_share_the_plain_code() -> None:
    rng = np.random.default_rng(3)
    options = {"cells": 2000, "active": 20, "valued": ["value"], "habituation": 0.0, "seed": 8}
    gated = cd.Records(12, {"y": 1, "value": 1}, tasks=[10, 11], **options)
    plain = cd.Records(12, {"y": 1, "value": 1}, **options)
    assert np.array_equal(gated.projection, plain.projection)
    assert sorted(np.bincount(gated.task_of_cell)) == [1000, 1000]
    first, second = np.zeros(12), np.zeros(12)
    first[:10] = second[:10] = rng.random(10)
    first[10], second[11] = 1.0, 1.0
    codes, reference = gated.code(np.stack([first, second])), plain.code(np.stack([first, second]))
    assert np.array_equal(codes[0], reference[0])
    assert (gated.task_of_cell[codes[1, 0] > 0] == 0).all() and (gated.task_of_cell[codes[1, 1] > 0] == 1).all()
    assert codes[1, 0] @ codes[1, 1] == 0.0
    silent = np.zeros(12)
    silent[:10] = first[:10]
    assert np.array_equal(gated.code(silent)[1], plain.code(silent)[1])
    gated.write(gated.code(first)[:, 0], {"value": np.array([1.0])})
    assert gated.read(gated.code(second)[:, 0])["value"][0] == 0.0
    again = cd.Records(12, {"y": 1, "value": 1}, tasks=[10, 11], **options)
    assert np.array_equal(again.task_of_cell, gated.task_of_cell)
    with pytest.raises(ValueError):
        cd.Records(12, {"y": 1}, cells=30, active=20, tasks=[10, 11])


def test_fan_in_restricts_each_cell_to_its_pathways_and_rebuilds_from_the_seed() -> None:
    pathways = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]
    records = cd.Records(14, {"y": 2}, cells=300, active=10, pathways=pathways, fan_in=2, seed=5)
    reads = np.stack([(records.projection[p] != 0).any(axis=0) for p in pathways])
    assert reads.shape == (3, 300) and (reads.sum(axis=0) == 2).all()
    assert (records.projection[12:] != 0).all()  # inputs outside the pathways reach every cell
    columns = np.sqrt((records.projection**2).sum(axis=0))
    assert abs(columns.mean() - 1.0) < 0.1  # ten of fourteen inputs per cell, rescaled to unit drive variance
    again = cd.Records(**records.to_dict())
    assert records.to_dict()["fan_in"] == 2
    assert np.array_equal(again.projection, records.projection)
    dense = cd.Records(14, {"y": 2}, cells=300, active=10, pathways=pathways, seed=5)
    assert dense.fan_in == 0 and (dense.projection != 0).all()
    with pytest.raises(ValueError):
        cd.Records(14, {"y": 2}, cells=300, active=10, fan_in=2, seed=5)
    with pytest.raises(ValueError):
        cd.Records(14, {"y": 2}, cells=300, active=10, pathways=pathways, fan_in=4, seed=5)


def test_a_consequence_only_code_leaves_the_valued_code_unset() -> None:
    records = cd.Records(6, {"y": 1, "value": 1}, valued=["value"], cells=200, active=8, pathways=[[0, 1, 2], [3, 4, 5]], seed=1)
    reading = np.arange(6, dtype=float) / 6
    full = records.code(reading)
    partial = records.code(reading, valued=False)
    assert np.array_equal(full[0], partial[0]) and np.isnan(partial[1]).all()
    read = records.read(partial[:, 0])
    assert np.isfinite(read["y"]).all() and np.isnan(read["value"]).all()
    norms = records.pathway_norm.copy()
    records.code(reading, adapt=True, valued=False)
    assert (records.pathway_norm != norms).all()  # witnessing adapts the norms either way


def test_homeostasis_spreads_the_code_over_the_cells():
    rng = np.random.default_rng(9)
    # Readings whose variance lies in a few directions, which hub cells would otherwise own.
    basis = rng.normal(size=(3, 20))
    readings = rng.normal(size=(600, 3)) @ basis * 4.0 + rng.normal(size=(600, 20)) * 0.1
    def usage(records):
        used = np.zeros(records.cells)
        for row in readings:
            used += records.code(row[None], valued=False)[0][0] > 0
        return used
    fixed = cd.Records(20, {"y": 2}, cells=400, active=8, seed=5)
    balanced = cd.Records(20, {"y": 2}, cells=400, active=8, seed=5, homeostasis=0.05)
    for _ in range(3):
        balanced.witness(readings)
    top_fixed = np.sort(usage(fixed))[::-1][:20].sum() / (8 * 600)
    top_balanced = np.sort(usage(balanced))[::-1][:20].sum() / (8 * 600)
    assert top_balanced < 0.75 * top_fixed
    assert (usage(balanced) > 0).sum() > 1.5 * (usage(fixed) > 0).sum()
    assert np.all(np.abs(balanced.boost) <= 3.0 * balanced.drive_scale + 1e-12)


def test_averaging_takes_the_first_outcome_whole_and_then_averages():
    records = cd.Records(6, {"y": 1}, cells=50, active=4, rate=0.05, averaging=True, seed=2)
    reading = np.array([1.0, -0.5, 0.25, 0.0, 0.7, -0.1])
    code = records.code(reading[None], valued=False)
    code = np.stack((code[0][0], code[0][0]))
    records.write(code, {"y": np.array([2.0])})
    assert abs(float(records.read(code)["y"][0]) - 2.0) < 1e-9  # the first write is exact
    for value in (4.0, 6.0, 8.0):
        records.write(code, {"y": np.array([value])})
    read = float(records.read(code)["y"][0])
    assert 2.0 < read < 8.0  # later outcomes are averaged in, not taken whole
    plain = cd.Records(6, {"y": 1}, cells=50, active=4, rate=0.05, seed=2)
    plain.write(code, {"y": np.array([2.0])})
    assert abs(float(plain.read(code)["y"][0]) - 0.1) < 1e-9  # a fixed rate takes a twentieth


def test_state_round_trip_carries_the_counts_usage_and_boost():
    rng = np.random.default_rng(1)
    records = cd.Records(5, {"y": 2}, cells=60, active=3, averaging=True, homeostasis=0.1, seed=3)
    readings = rng.normal(size=(30, 5))
    records.witness(readings)
    code = records.code(readings[:1], valued=False)[0][0]
    records.write(np.stack((code, code)), {"y": np.array([1.0, -1.0])})
    twin = cd.Records(5, {"y": 2}, cells=60, active=3, averaging=True, homeostasis=0.1, seed=3)
    twin.load_state(records.state())
    assert np.array_equal(twin.count, records.count) and np.array_equal(twin.boost, records.boost)
    assert np.array_equal(twin.usage, records.usage) and twin.drive_scale == records.drive_scale
    assert twin.to_dict() == records.to_dict()
