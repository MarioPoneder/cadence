"""Shared-parameter batch math, complete work and actual CPU/GPU execution."""

import math

import pytest

from cadence import Brain, Cortex, _repair, bootstrap
from cadence._tensor import TensorEngine

torch = pytest.importorskip("torch")

DEVICES = [
    pytest.param("cpu", "float64", id="cpu64"),
    *(
        pytest.param(
            "cuda:0",
            dtype,
            id=f"cuda{dtype.removeprefix('float')}",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA hardware unavailable"
            ),
        )
        for dtype in ("float64", "float32")
    ),
    pytest.param(
        "mps",
        "float32",
        id="mps32",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(),
            reason="MPS hardware unavailable",
        ),
    ),
]


def batch_case():
    graph = _repair.Graph(
        2,
        4,
        (
            ("input", 0, 0),
            ("input", 1, 1),
            ("state", 3, 0),
            ("state", 0, 1),
            ("state", 1, 1),
            ("residual", 0, 2),
            ("residual", 1, 2),
            ("residual", 2, 3),
            ("state", 0, 3),
        ),
    )
    batch = 3
    inputs = [0.6, -0.2, -0.7, 0.8, 0.1, 0.4]
    state = [0.1, -0.3, 0.2, 0.4, -0.2, 0.6, 0.3, -0.1, 0.5, 0.2, -0.4, 0.1]
    weights = [0.2, -0.3, 0.4, 0.1, -0.2, 0.3, -0.4, 0.2, 0.15]
    biases = [0.05, -0.2, 0.1, 0.15]
    return graph, batch, inputs, [state, weights, biases]


def energy_oracle(graph, batch, inputs, state, weights, biases, alpha, anchors, beta):
    """Scalar forward equation, independent of both production derivative paths."""
    total = 0.0
    for row in range(batch):
        x = state[row * graph.n_patches : (row + 1) * graph.n_patches]
        u = inputs[row * graph.n_inputs : (row + 1) * graph.n_inputs]
        errors = []
        # This fixture's explicit order is 0,1,2,3; never consume a production
        # residual-order or gradient implementation as the oracle.
        for target in range(graph.n_patches):
            drive = biases[target]
            for edge, weight in zip(graph.edges, weights, strict=True):
                kind, source, destination = edge
                if destination == target:
                    values = u if kind == "input" else x if kind == "state" else errors
                    drive += weight * values[source]
            errors.append(x[target] - math.tanh(drive))
        total += (sum(e * e for e in errors) + alpha * sum(v * v for v in x)) / 2
    total /= batch
    if anchors is not None:
        total += (
            beta
            / 2
            * sum(
                (value - anchor) ** 2
                for group, anchor_group in zip((weights, biases), anchors, strict=True)
                for value, anchor in zip(group, anchor_group, strict=True)
            )
        )
    return total


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
@pytest.mark.parametrize("anchored", (False, True))
def test_batch_derivatives_match_independent_finite_differences(
    device, dtype, anchored
):
    graph, batch, inputs, groups = batch_case()
    alpha, beta = 0.13, 0.29
    anchors = [[-0.12] * len(group) for group in groups[1:]] if anchored else None
    engine = TensorEngine(graph, device, dtype)
    energy, gradients = engine.evaluate(
        engine.tensor(inputs),
        *(engine.tensor(group) for group in groups),
        alpha,
        tuple(engine.tensor(group) for group in anchors) if anchored else None,
        beta,
        True,
        batch_size=batch,
    )
    tolerance = 4e-6 if dtype == "float32" else 2e-9
    assert energy.device.type == torch.device(device).type
    if device.startswith("cuda:"):
        assert energy.device == torch.device(device)
    assert float(energy) == pytest.approx(
        energy_oracle(graph, batch, inputs, *groups, alpha, anchors, beta),
        abs=tolerance,
    )
    for group_index, gradient in enumerate(gradients):
        assert gradient.device == energy.device
        assert gradient.numel() == len(groups[group_index])
        for coordinate, actual in enumerate(gradient.cpu().tolist()):
            plus, minus = (
                [list(group) for group in groups],
                [list(group) for group in groups],
            )
            h = 2e-6
            plus[group_index][coordinate] += h
            minus[group_index][coordinate] -= h
            expected = (
                energy_oracle(graph, batch, inputs, *plus, alpha, anchors, beta)
                - energy_oracle(graph, batch, inputs, *minus, alpha, anchors, beta)
            ) / (2 * h)
            assert actual == pytest.approx(expected, abs=tolerance, rel=tolerance)


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_repeated_rows_do_not_multiply_or_dilute_parameter_anchor(device, dtype):
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    engine = TensorEngine(graph, device, dtype)
    weights, biases = engine.tensor([0.3]), engine.tensor([-0.1])
    anchors = engine.tensor([-0.4]), engine.tensor([0.2])
    first = engine.evaluate(
        engine.tensor([0.7]),
        engine.tensor([0.2]),
        weights,
        biases,
        0.1,
        anchors,
        0.4,
        True,
    )
    explicit_one = engine.evaluate(
        engine.tensor([0.7]),
        engine.tensor([0.2]),
        weights,
        biases,
        0.1,
        anchors,
        0.4,
        True,
        batch_size=1,
    )
    assert torch.equal(first[0], explicit_one[0])
    assert all(
        torch.equal(a, b) for a, b in zip(first[1], explicit_one[1], strict=True)
    )
    repeated = engine.evaluate(
        engine.tensor([0.7] * 5),
        engine.tensor([0.2] * 5),
        weights,
        biases,
        0.1,
        anchors,
        0.4,
        True,
        batch_size=5,
    )
    tolerance = 2e-7 if dtype == "float32" else 1e-14
    assert float(repeated[0]) == pytest.approx(float(first[0]), abs=tolerance)
    assert (repeated[1][0] * 5).cpu().tolist() == pytest.approx(
        first[1][0].cpu().tolist() * 5,
        abs=tolerance,
    )
    for index in (1, 2):
        assert repeated[1][index].cpu().tolist() == pytest.approx(
            first[1][index].cpu().tolist(),
            abs=tolerance,
        )


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_batch_query_freezes_exact_parameters_and_refines_every_row(device, dtype):
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    inputs = (0.3, -0.8, 0.9)
    weights, biases = (0.123456789012345,), (0.234567890123456,)
    result = _repair.settle(
        graph,
        inputs,
        (0.0,) * 3,
        weights,
        biases,
        _batch_size=3,
        tolerance=1e-11,
        _engine=TensorEngine(graph, device, dtype),
    )
    assert result["qualified"]
    assert result["weights"] == weights and result["biases"] == biases
    expected = tuple(math.tanh(weights[0] * x + biases[0]) / 1.01 for x in inputs)
    assert result["state"] == pytest.approx(expected, abs=1e-10)
    checked = _repair.settle(
        graph,
        inputs,
        result["state"],
        weights,
        biases,
        _batch_size=3,
        tolerance=1e-11,
        budget=0,
    )
    assert checked["qualified"]
    assert result["stationarity"] == checked["stationarity"]
    assert result["energy"] == checked["energy"]
    assert result["execution"]["device"] == device


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_shared_learning_qualifies_original_witnesses_and_single_anchor(device, dtype):
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    inputs = (-0.8, 0.1, 0.9)
    witnesses = (-0.5, 0.2, 0.65)
    options = dict(
        clamps=dict(enumerate(witnesses)),
        learn=True,
        _batch_size=3,
        tolerance=1e-10,
        parameter_prior=0.3,
        anchor_weights=(-0.3,),
        anchor_biases=(0.4,),
    )
    result = _repair.settle(
        graph,
        inputs,
        (0.0,) * 3,
        (0.2,),
        (-0.1,),
        _engine=TensorEngine(graph, device, dtype),
        **options,
    )
    assert result["qualified"]
    assert result["state"] == witnesses
    w, b = result["weights"][0], result["biases"][0]
    predictions = [math.tanh(w * x + b) for x in inputs]
    signals = [
        (p - y) * (1 - p * p) for p, y in zip(predictions, witnesses, strict=True)
    ]
    dw = sum(s * x for s, x in zip(signals, inputs, strict=True)) / 3 + 0.3 * (w + 0.3)
    db = sum(signals) / 3 + 0.3 * (b - 0.4)
    assert max(abs(dw), abs(db)) <= 1e-10
    expected = energy_oracle(
        graph,
        3,
        inputs,
        witnesses,
        [w],
        [b],
        0.01,
        ([-0.3], [0.4]),
        0.3,
    )
    assert result["energy"] == pytest.approx(expected, abs=1e-14)
    python = _repair.settle(graph, inputs, (0.0,) * 3, (0.2,), (-0.1,), **options)
    assert python["qualified"]
    assert result["weights"] == pytest.approx(python["weights"], abs=1e-8)
    assert result["biases"] == pytest.approx(python["biases"], abs=1e-8)


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_averaged_state_gradient_cannot_hide_unsettled_row(device, dtype):
    graph = _repair.Graph(0, 1, ())
    batch = 8
    result = _repair.settle(
        graph,
        (),
        (4e-6,) + (0.0,) * (batch - 1),
        (),
        (0.0,),
        budget=0,
        tolerance=1e-6,
        _batch_size=batch,
        _engine=TensorEngine(graph, device, dtype),
    )
    assert not result["qualified"] and result["reason"] == "budget"
    assert result["stationarity"] == pytest.approx(4.04e-6)
    assert result["work"]["evaluations"] == 3
    assert result["work"]["patch_visits"] == 2 * batch * 3


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_batch_work_counts_vector_evaluations_and_all_row_visits(
    monkeypatch, device, dtype
):
    graph = _repair.Graph(1, 2, (("input", 0, 0), ("residual", 0, 1)))
    batch = 3
    engine = TensorEngine(graph, device, dtype)
    counts = {"tensor": 0, "reference": 0}
    tensor_evaluate, reference_evaluate = engine.evaluate, _repair._evaluate_batch

    def counted_tensor(*args, **kwargs):
        assert kwargs["batch_size"] == batch
        assert args[1].numel() == batch * graph.n_patches
        counts["tensor"] += 1
        return tensor_evaluate(*args, **kwargs)

    def counted_reference(*args, **kwargs):
        assert kwargs["batch_size"] == batch
        counts["reference"] += 1
        return reference_evaluate(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate", counted_tensor)
    monkeypatch.setattr(_repair, "_evaluate_batch", counted_reference)
    result = _repair.settle(
        graph,
        (0.3, -0.4, 0.8),
        (0.0,) * 6,
        (0.2, 0.4),
        (0.1, -0.1),
        _batch_size=batch,
        tolerance=1e-9,
        _engine=engine,
    )
    assert result["qualified"]
    assert counts["tensor"] > 0
    assert result["work"]["evaluations"] == sum(counts.values())
    assert result["work"]["patch_visits"] == 2 * batch * graph.n_patches * sum(
        counts.values()
    )
    assert result["work"]["edge_visits"] == 2 * batch * len(graph.edges) * sum(
        counts.values()
    )
    assert result["execution"]["reference_evaluations"] == counts["reference"]
    assert result["sweeps"] <= 2048


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_batch_query_omits_unused_parameter_gradients(device, dtype):
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    engine = TensorEngine(graph, device, dtype)
    big = 1e308 if dtype == "float64" else 1e38
    args = (
        engine.tensor([big] * 2),
        engine.tensor([4.0] * 2),
        engine.tensor([0.0]),
        engine.tensor([0.0]),
        0.01,
        None,
        0.1,
    )
    energy, gradients = engine.evaluate(*args, False, batch_size=2)
    assert math.isfinite(float(energy))
    assert gradients[1].numel() == gradients[2].numel() == 0
    with pytest.raises(ValueError, match="gradient exceeds"):
        engine.evaluate(*args, True, batch_size=2)


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_one_overflowing_batch_row_is_not_hidden_by_tanh(device, dtype):
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    engine = TensorEngine(graph, device, dtype)
    big = 1e308 if dtype == "float64" else 1e38
    with pytest.raises(ValueError, match="prediction.*finite numeric range"):
        engine.evaluate(
            engine.tensor([0.2, big]),
            engine.tensor([0.0, 0.0]),
            engine.tensor([4.0]),
            engine.tensor([0.0]),
            0.01,
            None,
            0.1,
            False,
            batch_size=2,
        )


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_batch_refinement_preserves_strict_tolerance(device, dtype):
    graph = _repair.Graph(0, 1, ())
    result = _repair.settle(
        graph,
        (),
        (1e-7, -2e-7),
        (),
        (0.0,),
        tolerance=1e-12,
        _batch_size=2,
        _engine=TensorEngine(graph, device, dtype),
    )
    assert result["qualified"] and result["stationarity"] <= 1e-12
    if dtype == "float32":
        assert result["execution"]["tensor_sweeps"] == 0
        assert result["execution"]["reference_sweeps"] > 0


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_subnormal_row_stationarity_is_checked_before_tensor_admission(device, dtype):
    graph = _repair.Graph(0, 1, ())
    result = _repair.settle(
        graph,
        (),
        (1e-323,) * 8,
        (),
        (0.0,),
        budget=0,
        tolerance=5e-324,
        _batch_size=8,
        _engine=TensorEngine(graph, device, dtype),
    )
    assert not result["qualified"]
    assert result["stationarity"] == 1e-323
    assert result["execution"]["tensor_sweeps"] == 0


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_batch_reductions_do_not_overflow_finite_mean_objective(device, dtype):
    engine = TensorEngine(_repair.Graph(0, 1, ()), device, dtype)
    large = math.sqrt(torch.finfo(engine.dtype).max * 0.6)
    energy, gradients = engine.evaluate(
        engine.tensor([]),
        engine.tensor([large, -large]),
        engine.tensor([]),
        engine.tensor([0.0]),
        0.1,
        None,
        0.1,
        False,
        batch_size=2,
    )
    assert math.isfinite(float(energy))
    assert float(energy) == pytest.approx(0.55 * large**2, rel=2e-6)
    assert gradients[0].cpu().tolist() == pytest.approx(
        [0.55 * large, -0.55 * large],
        rel=2e-6,
    )
    # Each row's parameter derivative is finite; summing them before dividing
    # would overflow. Zero weights keep the input drive itself finite.
    engine = TensorEngine(_repair.Graph(1, 1, (("input", 0, 0),)), device, dtype)
    large_input = torch.finfo(engine.dtype).max * 0.6
    _, gradients = engine.evaluate(
        engine.tensor([large_input] * 4),
        engine.tensor([1.0] * 4),
        engine.tensor([0.0]),
        engine.tensor([0.0]),
        0.1,
        None,
        0.1,
        True,
        batch_size=4,
    )
    assert float(gradients[1][0]) == pytest.approx(-large_input, rel=2e-6)


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_no_input_all_clamped_contradictory_batch_keeps_shared_compromise(
    device, dtype
):
    graph = _repair.Graph(0, 1, ())
    options = dict(
        learn=True,
        tolerance=1e-10,
        parameter_prior=0.2,
        anchor_weights=(),
        anchor_biases=(0.0,),
        _engine=TensorEngine(graph, device, dtype),
    )
    # Opposing witnesses require one shared bias, not independent per-row
    # fitted parameters and not a prediction-error-zero claim.
    answers = []
    for copies in (1, 3):
        witnesses = (-0.6, 0.6) * copies
        result = _repair.settle(
            graph,
            (),
            (0.0,) * len(witnesses),
            (),
            (0.3,),
            clamps=dict(enumerate(witnesses)),
            _batch_size=len(witnesses),
            **options,
        )
        assert result["qualified"]
        assert result["state"] == witnesses
        assert result["weights"] == ()
        assert result["biases"] == pytest.approx((0.0,), abs=1e-9)
        assert result["prediction_residual"] == pytest.approx(0.6, abs=1e-9)
        assert result["energy"] == pytest.approx(0.1818, abs=1e-13)
        answers.append(result)
    assert answers[0]["biases"] == pytest.approx(answers[1]["biases"], abs=1e-9)


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
@pytest.mark.parametrize("budget", (1, 2))
def test_duplicate_rows_preserve_preconditioned_state_steps(device, dtype, budget):
    graph = _repair.Graph(0, 1, ())
    engine = TensorEngine(graph, device, dtype)
    single = _repair.settle(
        graph,
        (),
        (0.4,),
        (),
        (0.1,),
        budget=budget,
        tolerance=1e-12,
        _engine=engine,
    )
    batch = _repair.settle(
        graph,
        (),
        (0.4,) * 5,
        (),
        (0.1,),
        budget=budget,
        tolerance=1e-12,
        _batch_size=5,
        _engine=engine,
    )
    tolerance = 4e-8 if dtype == "float32" else 1e-14
    assert batch["state"] == pytest.approx(single["state"] * 5, abs=tolerance)
    assert batch["energy"] == pytest.approx(single["energy"], abs=tolerance)
    assert batch["sweeps"] <= budget and single["sweeps"] <= budget


def test_faulty_batch_proposal_is_restarted_under_original_objective(monkeypatch):
    graph = _repair.Graph(0, 1, ())
    engine = TensorEngine(graph, "cpu", "float32")

    def wrong_objective(
        inputs, state, weights, biases, alpha, anchors, beta, learn, *, batch_size
    ):
        assert batch_size == 3
        delta = state - 0.5
        return delta.square().sum() / batch_size, (
            2 * delta / batch_size,
            weights.new_empty(0),
            biases.new_empty(0),
        )

    monkeypatch.setattr(engine, "evaluate", wrong_objective)
    result = _repair.settle(
        graph,
        (),
        (0.1, -0.15, 0.2),
        (),
        (0.0,),
        budget=12,
        _batch_size=3,
        _engine=engine,
    )
    assert result["execution"]["reference_restart"]
    assert result["qualified"] and result["state"] == pytest.approx(
        (0.0,) * 3, abs=1e-6
    )
    assert result["sweeps"] <= 12
    assert result["work"]["patch_visits"] == 6 * result["work"]["evaluations"]


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_public_batch_bootstrap_recall_checkpoint_transfer_and_live_learning(
    device, dtype
):
    cortex = Cortex(seed=2, device=device, dtype=dtype)
    sensor = cortex.input("signal", shape=1)
    base = cortex.column("base", patches=2, inputs=sensor)
    observer = cortex.observer("observer", patches=1, inputs=sensor, observes=base)
    cortex.output("answer", shape=1, reads=observer)
    brain = cortex.build()
    assert brain.step({"signal": [0.2]})["accepted"]
    original_state, original_weights = brain.state, brain.weights
    examples = [({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.8, 0.8)]
    checks = [({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.4, 0.4)]
    report = bootstrap(brain, examples, checks=checks, max_error=0.12, batch_size=2)
    assert report["passed"] and report["failure"] is None
    assert report["accepted"] == report["presentations"]
    assert report["accepted"] > 0
    assert brain.weights != original_weights
    assert brain.state == original_state
    snapshot = brain.snapshot()
    for x in (-0.6, 0.6):
        assert abs(brain.predict({"signal": [x]})["answer"][0] - 0.6 * x) < 0.12
    assert brain.snapshot() == snapshot
    admitted = brain.observe_batch(examples, event_id=100)
    assert admitted["accepted"] and admitted["execution"]["device"] == device
    assert admitted["execution"]["tensor_sweeps"] > 0
    assert admitted["outputs"] == ({"answer": (-0.48,)}, {"answer": (0.48,)})
    assert brain.state == original_state
    snapshot = brain.snapshot()
    restored = Brain.from_snapshot(snapshot)
    transferred = Brain.from_snapshot(snapshot, device="python")
    assert restored.snapshot() == snapshot
    for continuation in (restored, transferred):
        assert continuation.weights == brain.weights
        assert continuation.biases == brain.biases
        assert continuation.state == original_state
        before_retry = continuation.snapshot()
        assert continuation.observe_batch(examples, event_id=100)["duplicate"]
        assert continuation.snapshot() == before_retry
        assert abs(continuation.predict({"signal": [0.6]})["answer"][0] - 0.36) < 0.12
    inputs, targets = {"signal": [0.2]}, {"answer": [-0.3]}
    previous = restored.predict(inputs)["answer"][0]
    live = restored.observe(inputs, targets)
    assert live["accepted"] and live["event_id"] == 101
    assert abs(restored.predict(inputs)["answer"][0] + 0.3) < abs(previous + 0.3)
    assert restored.weights != brain.weights
    assert brain.snapshot() == snapshot


@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_public_batch_refusal_and_retry_preserve_order_and_source_custody(
    device, dtype
):
    cortex = Cortex(seed=7, device=device, dtype=dtype)
    sensor = cortex.input("signal", shape=1)
    base = cortex.column("base", patches=2, inputs=sensor)
    observer = cortex.observer("observer", patches=1, inputs=sensor, observes=base)
    cortex.output("answer", shape=1, reads=observer)
    brain = cortex.build()
    assert brain.step({"signal": [0.3]})["accepted"]
    examples = [({"signal": [x]}, {"answer": [x]}) for x in (-0.8, 0.8)]
    before = brain.snapshot()
    live_state = brain.state
    refused = brain.observe_batch(examples, event_id=7, source="estimate", budget=0)
    assert not refused["accepted"] and refused["reason"] == "budget"
    assert brain.snapshot() == before
    admitted = brain.observe_batch(examples, event_id=7, source="estimate")
    assert admitted["accepted"] and admitted["execution"]["device"] == device
    assert brain.state == live_state
    assert brain.inspect()["admissions"] == 1
    saved = brain.snapshot()
    assert brain.observe_batch(examples, event_id=7, source="estimate")["duplicate"]
    assert brain.snapshot() == saved
    for rows, source in ((examples[::-1], "estimate"), (examples, "witness")):
        with pytest.raises(ValueError, match="conflicts"):
            brain.observe_batch(rows, event_id=7, source=source)
        assert brain.snapshot() == saved
