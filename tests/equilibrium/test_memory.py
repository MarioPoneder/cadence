"""Explicit temporal context and bounded error progress retain honest semantics."""

import json
import math
import random
import sys

import pytest

from cadence.experimental.equilibrium import Cortex
from cadence.experimental.equilibrium.memory import History, LearningProgress


def test_history_padding_mask_shape_preview_and_oldest_drop():
    history = History(2, steps=3)
    assert history.input_size == 2
    assert history.steps == 3
    assert history.size == 9
    assert history.shape == (9,)
    initial = history.snapshot()
    expected = (0.0,) * 6 + (0.0, 0.0, 1.0)
    assert history.preview([0, 0]) == expected
    assert history.snapshot() == initial
    assert history.push([0, 0]) == expected
    assert history.push([2, 3]) == (0, 0, 0, 0, 0, 1, 2, 3, 1)
    assert history.push([4, 5]) == (0, 0, 1, 2, 3, 1, 4, 5, 1)
    assert history.push([6, 7]) == (2, 3, 1, 4, 5, 1, 6, 7, 1)
    history.reset()
    assert history.snapshot() == initial


def test_same_occluded_present_retains_different_explicit_histories():
    left, right = History(2, steps=2), History(2, steps=2)
    # Raw sample: position and visibility. The slot mask does not pretend to
    # know whether a sensor was occluded; visibility is explicitly supplied.
    left.push([-0.8, 1])
    right.push([0.8, 1])
    a, b = left.push([0, 0]), right.push([0, 0])
    assert a[-3:] == b[-3:] == (0, 0, 1)
    assert a[:3] != b[:3]
    cortex = Cortex(seed=7)
    sensor = cortex.input("history", shape=left.shape)
    features = cortex.column("features", patches=2, inputs=sensor)
    output = cortex.column("response", patches=1, inputs=features)
    cortex.output("answer", shape=1, reads=output)
    brain = cortex.build()
    before = brain.snapshot()
    for encoded in (a, b):
        assert brain.settle({"history": encoded})["qualified"]
    assert brain.snapshot() == before


def test_history_owns_inputs_and_checkpoint_continuation():
    history = History(2, steps=2)
    values = [1, 2]
    history.push(values)
    saved = history.snapshot()
    values[0] = 100
    assert history.snapshot() == saved
    restored = History.from_snapshot(saved)
    assert restored.snapshot() == saved
    for values in ([3, 4], [0, 0], [-1, -2]):
        assert history.preview(values) == restored.preview(values)
        assert history.push(values) == restored.push(values)
        assert history.snapshot() == restored.snapshot()


@pytest.mark.parametrize(
    "values",
    ([1], [1, 2, 3], [[1], [2]], [True, 0], [math.nan, 0], [math.inf, 0], "12", None),
)
def test_invalid_history_push_and_preview_are_atomic(values):
    history = History(2, steps=1)
    history.push([0.5, 0.6])
    before = history.snapshot()
    for operation in (history.push, history.preview):
        with pytest.raises(ValueError):
            operation(values)
        assert history.snapshot() == before


@pytest.mark.parametrize(
    "size, steps",
    [(0, 4), (True, 4), (1.5, 4), (2, False), (2, 0), (1, 500001), (1_000_000, 1)],
)
def test_history_dimension_limits(size, steps):
    with pytest.raises(ValueError):
        History(size, steps=steps)


def test_learning_progress_formula_constant_errors_and_decay():
    progress = LearningProgress(rate=0.5)
    assert progress.update("signal", 2) == 0
    assert progress.update("signal", 0) == 0.5
    assert progress.update("signal", 0) == 0.5
    assert progress.update("signal", 0) == 0.25
    assert progress.update("signal", 2) == 0
    for error in (0, 1, 100):
        key = f"constant {error}"
        assert [progress.update(key, error) for _ in range(20)] == [0] * 20
    assert progress.update("increasing", 1) == 0
    assert progress.update("increasing", 2) == 0
    assert progress.update("increasing", 3) == 0


def test_context_eviction_is_lru_and_survives_checkpoints():
    progress = LearningProgress(rate=0.25, capacity=2)
    progress.update("a", 1)
    progress.update("b", 2)
    progress.update("a", 0.5)
    restored = LearningProgress.from_snapshot(progress.snapshot())
    for key, error in (("c", 3), ("b", 0.1), ("a", 0.2), ("c", 0)):
        assert progress.update(key, error) == restored.update(key, error) == 0
        assert progress.snapshot() == restored.snapshot()
        assert len(json.loads(progress.snapshot())["errors"]) == 2


def test_progress_restoration_continues_exact_scores_and_error_means():
    progress = LearningProgress(rate=0.1)
    for error in (1, 0.8, 0.5):
        progress.update("visible", error)
    restored = LearningProgress.from_snapshot(progress.snapshot())
    assert restored.rate == progress.rate == 0.1
    assert restored.capacity == progress.capacity == 128
    for error in (0.4, 0.6, 0.1, 0):
        assert restored.update("visible", error) == progress.update("visible", error)
        assert restored.snapshot() == progress.snapshot()


def test_noise_is_not_silently_reported_as_information_gain():
    # Random observation error with no improving predictor can score positively.
    # This test preserves the documented limitation instead of claiming that an
    # error-only statistic can distinguish every noise fluctuation from learning.
    progress = LearningProgress()
    rng = random.Random(7)
    scores = [progress.update("noise", rng.random()) for _ in range(100)]
    assert scores[0] == 0
    assert any(score > 0 for score in scores)
    assert any(score == 0 for score in scores)
    assert all(0 <= score <= 1 for score in scores)
    # Mere novelty (different context labels) supplies no positive score.
    assert all(progress.update(f"new {i}", rng.random()) == 0 for i in range(100))


@pytest.mark.parametrize("error", [-1, True, math.nan, math.inf, -math.inf, "1", None])
def test_progress_rejects_bad_errors_without_update_or_eviction(error):
    progress = LearningProgress(capacity=1)
    progress.update("known", 1)
    saved = progress.snapshot()
    with pytest.raises(ValueError):
        progress.update("other", error)
    assert progress.snapshot() == saved


@pytest.mark.parametrize("key", ["", True, 1, None, (), "x" * 257, "é" * 129, "\ud800"])
def test_progress_rejects_bad_keys_atomically(key):
    progress = LearningProgress(capacity=1)
    progress.update("known", 1)
    saved = progress.snapshot()
    with pytest.raises(ValueError):
        progress.update(key, 0)
    assert progress.snapshot() == saved


@pytest.mark.parametrize(
    "options",
    [
        {"rate": 0},
        {"rate": True},
        {"rate": 1.1},
        {"rate": math.inf},
        {"capacity": 0},
        {"capacity": True},
        {"capacity": 1.5},
        {"capacity": 4097},
    ],
)
def test_progress_configuration_limits(options):
    with pytest.raises(ValueError):
        LearningProgress(**options)


@pytest.mark.parametrize("rate", [1, 0.1, sys.float_info.min])
def test_progress_extreme_finite_errors_remain_finite(rate):
    progress = LearningProgress(rate=rate)
    for error in (sys.float_info.max, 0, sys.float_info.max, sys.float_info.max, 0):
        score = progress.update("extreme", error)
        assert math.isfinite(score) and 0 <= score <= 1
        assert LearningProgress.from_snapshot(progress.snapshot()).snapshot() == progress.snapshot()


@pytest.mark.parametrize("small_error", [1.0, sys.float_info.min, 5e-324])
def test_full_rate_preserves_new_error_after_an_extreme_downward_jump(small_error):
    progress = LearningProgress(rate=1)
    progress.update("signal", 1e308)
    progress.update("signal", small_error)
    assert json.loads(progress.snapshot())["errors"] == [["signal", small_error]]
    assert progress.update("signal", 0) == small_error


@pytest.mark.parametrize("rate", [1e-308, 5e-324])
def test_tiny_rate_retains_representable_progress_toward_large_error(rate):
    progress = LearningProgress(rate=rate)
    assert progress.update("signal", 0) == 0
    assert progress.update("signal", 1e308) == 0
    mean = json.loads(progress.snapshot())["errors"][0][1]
    assert mean == rate * 1e308
    assert math.isfinite(mean) and mean > 0
    assert progress.update("signal", 0) == 0  # Reduction below float resolution.
    assert json.loads(progress.snapshot())["errors"][0][1] == mean


@pytest.mark.parametrize("factory", [lambda: History(2), LearningProgress])
@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=1),
        lambda d: d.update(version=True),
        lambda d: d.update(version=2),
        lambda d: d.update(kind="other"),
        lambda d: d.pop("kind"),
    ],
)
def test_checkpoint_schema_is_exact(factory, mutation):
    instance = factory()
    data = json.loads(instance.snapshot())
    mutation(data)
    with pytest.raises(ValueError):
        type(instance).from_snapshot(json.dumps(data))


@pytest.mark.parametrize("cls", [History, LearningProgress])
@pytest.mark.parametrize(
    "text",
    [
        "null",
        "[]",
        "{",
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '"\ud800"',
        "[" * 1100 + "]" * 1100,
        None,
        b"{}",
    ],
)
def test_checkpoint_json_is_strict(cls, text):
    with pytest.raises(ValueError):
        cls.from_snapshot(text)


@pytest.mark.parametrize(
    "patch",
    [
        {"rows": [[1, 2]] * 5},
        {"rows": {}},
        {"rows": [[1]]},
        {"rows": [[True, 1]]},
        {"input_size": True},
        {"steps": 1_000_001},
    ],
)
def test_history_checkpoint_validates_rows_and_dimensions(patch):
    data = json.loads(History(2).snapshot())
    data.update(patch)
    with pytest.raises(ValueError):
        History.from_snapshot(json.dumps(data))


@pytest.mark.parametrize(
    "rows",
    [
        [[]],
        [["a", 0, 1]],
        [["a", -1]],
        [["a", True]],
        [["a", 0], ["a", 1]],
        [["", 0]],
        [["a", 0], ["b", 1], ["c", 2]],
        {},
    ],
)
def test_progress_checkpoint_validates_contexts_means_and_capacity(rows):
    data = json.loads(LearningProgress(capacity=2).snapshot())
    data["errors"] = rows
    with pytest.raises(ValueError):
        LearningProgress.from_snapshot(json.dumps(data))
