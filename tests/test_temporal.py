"""Independent derivatives, frozen-kernel parity and live-state boundaries."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cadence import TemporalPatchNet


def same_snapshot(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert_array_equal(left[key], right[key])


def differences(function, point, step=1e-6):
    result = np.empty_like(point)
    for index in np.ndindex(point.shape):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2 * step)
    return result


def independent_energy(parameters, inputs, boundary, hidden, output, target, beta, precision=1.0):
    value = 0.0
    batch, horizon, _ = inputs.shape
    for row in range(batch):
        previous = boundary[row]
        for t in range(horizon):
            seam = (
                hidden[row, t]
                - parameters["A"] @ np.tanh(previous)
                - parameters["B"] @ inputs[row, t]
            )
            readout = output[row, t] - parameters["C"] @ np.tanh(hidden[row, t])
            value += 0.5 * (seam @ seam + readout @ readout)
            previous = hidden[row, t]
    return value / batch + 0.5 * beta * np.mean(precision * (output - target) ** 2)


def independent_rollout(parameters, inputs, boundary):
    hidden = boundary.copy()
    output = []
    for t in range(inputs.shape[1]):
        hidden = np.stack(
            [
                parameters["A"] @ np.tanh(h) + parameters["B"] @ u
                for h, u in zip(hidden, inputs[:, t], strict=True)
            ]
        )
        output.append(np.tanh(hidden) @ parameters["C"].T)
    return np.stack(output, axis=1)


def test_frozen_scalar_kernel_parity():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/temporal_reference_v1.json").read_text()
    )
    assert (
        fixture["source_sha256"]
        == "ff82abd80adcfb06e816f85aa582e1d7f073f7193f7f338b7ce9db8fed46fcca"
    )
    for case in fixture["cases"]:
        net = TemporalPatchNet(1, case["width"], 1, seed=case["seed"], tolerance=case["tolerance"])
        for key, value in net.parameters().items():
            # QR initialization can differ by a few ulps across BLAS builds.
            # Check that initialization agrees numerically, then replay the
            # kernel from the exact archived arrays rather than a seed proxy.
            assert_allclose(value, case["initial"][key], atol=1e-15, rtol=1e-15)
        net.set_parameters({key: np.array(value) for key, value in case["initial"].items()})
        inputs = np.zeros((2, case["horizon"], 1))
        inputs[:, 0, 0] = case["cue"]
        result = net.observe(inputs, np.array(case["target"]), beta=case["beta"], rate=case["rate"])
        assert result.updated
        for name in ["free", "plus", "minus"]:
            phase = getattr(result, name)
            expected = case["phases"][name]
            assert phase.converged and phase.iterations == expected["iterations"]
            assert_allclose(phase.hidden, expected["hidden"], atol=2e-12, rtol=2e-12)
            assert_allclose(phase.output, expected["output"], atol=2e-12, rtol=2e-12)
        for key, value in net.parameters().items():
            assert_allclose(value, case["final"][key], atol=2e-12, rtol=2e-12)
            assert_allclose(result.delta[key], case["delta"][key], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("precision", [np.ones(2), np.array([0.2, 3.0])])
@pytest.mark.parametrize("beta", [-0.12, 0.12])
def test_state_gradient_and_exact_hessian_with_time_varying_ports(beta, precision):
    rng = np.random.default_rng(31)
    net = TemporalPatchNet(2, 3, 2, seed=37, output_precision=precision)
    inputs, boundary = rng.normal(size=(2, 4, 2)) * 0.2, rng.normal(size=(2, 3)) * 0.2
    hidden, output = rng.normal(size=(2, 4, 3)) * 0.2, rng.normal(size=(2, 4, 2)) * 0.2
    target = rng.normal(size=output.shape) * 0.2
    parameters = net.parameters()

    def energy(h, y):
        return independent_energy(parameters, inputs, boundary, h, y, target, beta, precision)

    gh, gy = net._state_gradient(inputs, hidden, output, boundary, target, beta)
    assert_allclose(gh / 2, differences(lambda h: energy(h, output), hidden), atol=2e-10)
    assert_allclose(gy / 2, differences(lambda y: energy(hidden, y), output), atol=2e-10)
    b = beta * precision / (4 * 2)

    def reduced(h):
        return energy(h, (np.tanh(h) @ parameters["C"].T + b * target) / (1 + b))

    gradient, diagonal, lower = net._reduced_derivatives(inputs, hidden, boundary, target, beta)
    assert_allclose(gradient / 2, differences(reduced, hidden), atol=2e-10)
    analytic = np.zeros((hidden.size, hidden.size))
    for row in range(2):
        for t in range(4):
            offset = (row * 4 + t) * 3
            analytic[offset : offset + 3, offset : offset + 3] = diagonal[row, t] / 2
            if t:
                analytic[offset : offset + 3, offset - 3 : offset] = lower[row, t - 1] / 2
                analytic[offset - 3 : offset, offset : offset + 3] = lower[row, t - 1].T / 2
    numeric = np.empty_like(analytic)
    for index in range(hidden.size):
        plus, minus = hidden.copy(), hidden.copy()
        plus.flat[index] += 1e-6
        minus.flat[index] -= 1e-6
        gp = net._reduced_derivatives(inputs, plus, boundary, target, beta)[0]
        gm = net._reduced_derivatives(inputs, minus, boundary, target, beta)[0]
        numeric[:, index] = ((gp - gm) / 4e-6).ravel()
    assert_allclose(analytic, numeric, atol=3e-10)


@pytest.mark.parametrize("precision", [np.ones(2), np.array([0.3, 2.7])])
def test_centered_local_update_matches_independent_free_loss_derivative(precision):
    rng = np.random.default_rng(97)
    net = TemporalPatchNet(2, 3, 2, seed=101, tolerance=1e-12, output_precision=precision)
    net.advance(rng.normal(size=(2, 2, 2)) * 0.1)
    boundary = net.state
    inputs, targets = rng.normal(size=(2, 5, 2)) * 0.2, rng.normal(size=(2, 5, 2)) * 0.2
    parameters = net.parameters()
    numeric = {}
    for key, values in parameters.items():

        def loss(replacement, key=key):
            changed = {**parameters, key: replacement}
            return 0.5 * np.mean(
                precision * (independent_rollout(changed, inputs, boundary) - targets) ** 2
            )

        numeric[key] = differences(loss, values)
    before = net.snapshot()
    free = net.imagine(inputs)
    result = net.observe(inputs, targets, beta=1e-4, rate=0.2)
    assert result.updated and net.updates == 1
    assert_array_equal(net.state, free.final_state)
    assert not np.array_equal(net.state, result.plus.final_state)
    assert net.readback().state_parameter_revision == 0
    assert net.readback().parameter_revision == 1
    for key in parameters:
        assert_allclose(result.delta[key], numeric[key], rtol=2e-5, atol=2e-9)
        assert_allclose(net.parameters()[key], before[key] - 0.2 * numeric[key], atol=4e-10)
    for phase in [result.plus, result.minus]:
        assert phase.minimum_pivot_eigenvalue > 0
        assert phase.residual <= net.tolerance
        assert np.all(np.diff(phase.energy_history) <= 2e-15)


def test_private_imagination_state_carry_and_target_independence():
    rng = np.random.default_rng(13)
    inputs = rng.normal(size=(2, 9, 2)) * 0.1
    net = TemporalPatchNet(2, 4, 3, seed=7)
    initial = net.snapshot()
    complete = net.imagine(inputs)
    ignored = net.settle(inputs, target=np.array([np.nan]), beta=0)
    assert_array_equal(complete.output, ignored.output)
    same_snapshot(initial, net.snapshot())
    first, second = net.advance(inputs[:, :4]), net.advance(inputs[:, 4:])
    assert_allclose(
        np.concatenate([first.output, second.output], axis=1), complete.output, atol=1e-16
    )
    assert_allclose(net.state, complete.final_state, atol=1e-16)
    live = net.snapshot()
    branch = net.imagine(inputs, state=np.ones((2, 4)))
    branch.hidden.fill(99)
    net.readback().state.fill(99)
    net.state.fill(99)
    net.parameters()["A"].fill(99)
    net.snapshot()["state"].fill(99)
    same_snapshot(live, net.snapshot())
    net.reset()
    assert net.state is None and net.readback().energy is None
    for name in ["A", "B", "C"]:
        assert_array_equal(net.parameters()[name], live[name])


def test_failed_nudge_and_negative_curvature_never_commit_weights():
    inputs, target = np.zeros((1, 1, 1)), np.zeros((1, 1, 1))
    net = TemporalPatchNet(1, 1, 1)
    net.set_parameters({"A": np.zeros((1, 1)), "B": np.zeros((1, 1)), "C": np.full((1, 1), 2.0)})
    before = net.parameters()
    result = net.observe(inputs, target, beta=0.5)
    assert not result.updated and result.minus.reason == "nonminimum_stationary_point"
    assert result.minus.residual == 0 and result.minus.block_chain_attempts == 1
    assert net.updates == 0
    assert_array_equal(net.state, result.free.final_state)
    for key, value in before.items():
        assert_array_equal(net.parameters()[key], value)
    capped = TemporalPatchNet(1, 2, 1, max_iterations=0)
    result = capped.observe(np.ones((1, 3, 1)), np.ones((1, 3, 1)))
    assert not result.updated and result.plus.reason == "iteration_cap"


def test_finite_overflow_is_failed_free_phase_without_state_or_weight_update():
    net = TemporalPatchNet(1, 2, 1)
    net.advance(np.zeros((1, 1, 1)))
    parameters = net.parameters()
    parameters["B"].fill(1e308)
    net.set_parameters(parameters)
    before = net.snapshot()
    result = net.observe(np.full((1, 2, 1), 1e308), np.zeros((1, 2, 1)))
    assert not result.updated and result.free.reason == "nonfinite_free_state"
    same_snapshot(before, net.snapshot())


def test_save_restore_resumes_live_inference_and_next_update(tmp_path):
    rng = np.random.default_rng(83)
    net = TemporalPatchNet(2, 3, 2, seed=89)
    inputs, target = rng.normal(size=(2, 4, 2)) * 0.1, rng.normal(size=(2, 4, 2)) * 0.1
    assert net.observe(inputs, target).updated
    path = net.save(tmp_path / "nested/temporal")
    assert path.suffix == ".npz"
    restored = TemporalPatchNet.load(path)
    same_snapshot(net.snapshot(), restored.snapshot())
    assert_array_equal(net.imagine(inputs).output, restored.imagine(inputs).output)
    assert net.observe(inputs, target).updated and restored.observe(inputs, target).updated
    same_snapshot(net.snapshot(), restored.snapshot())
    cold = TemporalPatchNet.restore(TemporalPatchNet(2, 3, 2).snapshot())
    assert cold.state is None


@pytest.mark.parametrize(
    "invalid", [np.full((1, 2, 1), np.nan), np.ones((1, 0, 1)), np.ones((1, 2, 2)), np.ones((2, 1))]
)
def test_invalid_observation_is_atomic(invalid):
    net = TemporalPatchNet(1, 2, 1)
    before = net.snapshot()
    with pytest.raises(ValueError):
        net.observe(invalid, np.zeros((1, 2, 1)))
    same_snapshot(before, net.snapshot())


def test_validation_of_dimensions_batch_changes_and_checkpoint():
    for args in [(True, 2, 1), (1, 0, 1), (1, 2, 1.5)]:
        with pytest.raises(ValueError):
            TemporalPatchNet(*args)
    net = TemporalPatchNet(1, 2, 1)
    net.advance(np.zeros((2, 2, 1)))
    with pytest.raises(ValueError, match="reset"):
        net.advance(np.zeros((1, 2, 1)))
    before = net.snapshot()
    broken = net.parameters()
    broken["C"][0, 0] = np.nan
    with pytest.raises(ValueError):
        net.set_parameters(broken)
    same_snapshot(before, net.snapshot())
    broken = net.snapshot()
    broken["state"][0, 0] = np.nan
    with pytest.raises(ValueError):
        TemporalPatchNet.restore(broken)
    broken = net.snapshot()
    meta = json.loads(str(broken["meta"]))
    meta["state_parameter_revision"] = meta["parameter_revision"] + 1
    broken["meta"] = np.array(json.dumps(meta))
    with pytest.raises(ValueError):
        TemporalPatchNet.restore(broken)


def test_precision_changes_only_teaching_loss_and_survives_checkpoint(tmp_path):
    net = TemporalPatchNet(2, 3, 2)
    path = np.ones((1, 3, 2)) * 0.1
    net.advance(path)
    state, before, parameters = net.state, net.readback(), net.parameters()
    free = net.imagine(path)
    supplied = np.array([0.2, 3.0])
    net.set_output_precision(supplied)
    supplied[:] = 8
    net.output_precision[:] = 9
    assert_array_equal(net.output_precision, [0.2, 3.0])
    assert_array_equal(net.state, state)
    assert_array_equal(net.imagine(path).output, free.output)
    assert net.readback().parameter_revision == before.parameter_revision
    assert net.readback().energy == before.energy and net.readback().residual == before.residual
    same_snapshot(net.parameters(), parameters)
    restored = TemporalPatchNet.load(net.save(tmp_path / "weighted"))
    same_snapshot(net.snapshot(), restored.snapshot())
    assert net.observe(path, path).updated and restored.observe(path, path).updated
    same_snapshot(net.snapshot(), restored.snapshot())


def test_legacy_checkpoint_defaults_to_unit_precision_and_default_parity():
    net = TemporalPatchNet(2, 3, 2)
    legacy = net.snapshot()
    del legacy["output_precision"]
    restored = TemporalPatchNet.restore(legacy)
    assert_array_equal(restored.output_precision, np.ones(2))
    explicit = TemporalPatchNet(2, 3, 2, output_precision=np.ones(2))
    path = np.ones((1, 3, 2)) * 0.1
    result = restored.observe(path, path)
    other = explicit.observe(path, path)
    assert result.updated and other.updated
    same_snapshot(restored.snapshot(), explicit.snapshot())
    for key in ["free", "plus", "minus"]:
        assert_array_equal(getattr(result, key).hidden, getattr(other, key).hidden)
        assert_array_equal(getattr(result, key).output, getattr(other, key).output)


def test_precision_validation_and_negative_phase_domain_are_atomic():
    net = TemporalPatchNet(1, 2, 2)
    before = net.snapshot()
    for invalid in [
        np.array([0.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([np.inf, 1.0]),
        np.array([np.nan, 1.0]),
        np.ones(3),
        np.ones((1, 2)),
        np.ones(2, dtype=bool),
    ]:
        with pytest.raises(ValueError):
            net.set_output_precision(invalid)
        same_snapshot(before, net.snapshot())
    net.set_output_precision(np.array([10.0, 1.0]))
    before = net.snapshot()
    with pytest.raises(ValueError, match="precision"):
        net.observe(np.zeros((1, 1, 1)), np.zeros((1, 1, 2)), beta=0.2)
    with pytest.raises(ValueError, match="precision"):
        net.settle(np.zeros((1, 1, 1)), target=np.zeros((1, 1, 2)), beta=-0.2)
    same_snapshot(before, net.snapshot())
    assert net.settle(np.zeros((1, 1, 1)), target=np.array([np.nan]), beta=0).converged


def test_precision_change_keeps_protected_response_binding_valid():
    from cadence import TemporalMemory

    net = TemporalPatchNet(2, 4, 2)
    memory = TemporalMemory()
    old, new = np.array([[[1.0, 0.0]]]), np.array([[[0.0, 1.0]]])
    expected = net.imagine(old).output
    memory.protect(net, old)
    net.set_output_precision(np.array([0.2, 3.0]))
    net.reset()
    assert memory.observe(net, new, np.ones((1, 1, 2)) * 0.2).updated
    assert_allclose(net.imagine(old, state=np.zeros((1, 4))).output, expected, atol=1e-14)
    assert_array_equal(net.output_precision, [0.2, 3.0])


def test_off_center_detuned_paths_halve_beta_before_a_parameter_update():
    from cadence.temporal import contrast_asymmetry

    rng = np.random.default_rng(2)
    net = TemporalPatchNet(2, 4, 1, seed=2, initial_radius=1.5)
    inputs = rng.normal(size=(1, 24, 2))
    target = np.sign(rng.normal(size=(1, 24, 1)))
    boundary = np.zeros((1, 4))
    before = net.parameters()

    refused = net.observe(inputs, target, rate=0.1, max_halvings=0)
    assert not refused.updated and refused.reason == "contrast_asymmetric"
    assert refused.beta == 0.01 and refused.contrast_halvings == 0 and net.updates == 0
    assert refused.plus is not None and refused.minus is not None
    assert contrast_asymmetry(refused.free, refused.plus, refused.minus) > 0.3
    assert contrast_asymmetry(refused.free, refused.free, refused.free) == 0.0
    for key, value in before.items():
        assert_array_equal(net.parameters()[key], value)

    net.reset()
    checked = net.observe(inputs, target, rate=0.0, symmetry_tolerance=0.1)
    assert checked.updated and checked.beta == 0.0025 and checked.contrast_halvings == 2
    assert contrast_asymmetry(checked.free, checked.plus, checked.minus) <= 0.1

    def free_loss(parameters):
        state, outputs = boundary.copy(), []
        for moment in range(inputs.shape[1]):
            state = np.tanh(state) @ parameters["A"].T + inputs[:, moment] @ parameters["B"].T
            outputs.append(np.tanh(state) @ parameters["C"].T)
        return float(0.5 * np.mean((np.stack(outputs, axis=1) - target) ** 2))

    for key, value in before.items():
        expected = np.empty_like(value)
        for index in np.ndindex(value.shape):
            plus = {k: v.copy() for k, v in before.items()}
            minus = {k: v.copy() for k, v in before.items()}
            plus[key][index] += 1e-6
            minus[key][index] -= 1e-6
            expected[index] = (free_loss(plus) - free_loss(minus)) / 2e-6
        assert np.abs(checked.delta[key] - expected).max() < 0.01 * np.abs(expected).max()

    net.reset()
    backtracked = net.observe(inputs, target, rate=0.1, backtrack=True, symmetry_tolerance=0.1)
    assert backtracked.beta == 0.0025 and backtracked.contrast_halvings == 2
