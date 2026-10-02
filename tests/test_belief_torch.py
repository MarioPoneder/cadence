"""Direct tests for cadence.belief_torch: TorchPort and TorchBelief.

Forward and gradient parity with the NumPy ``BeliefPatch`` are checked in ``test_belief.py``;
these tests cover the module's own contracts. All tests skip when torch is not installed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cadence import BeliefPatch, DenseBlock, MapBlock, StructuredPort  # noqa: E402
from cadence.belief_torch import TorchBelief, TorchPort  # noqa: E402


@pytest.fixture(autouse=True)
def float64() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _port() -> StructuredPort:
    return StructuredPort(
        1 * 5 * 5 + 3, [MapBlock(0, 1, 5, 5, 2, 3, 2), DenseBlock(25, 3, 4)], broadcast=(25, 3)
    )


def _patch(seed: int = 0) -> BeliefPatch:
    return BeliefPatch(
        _port(), actions=3, belief=6, outputs=4, cells=64, active=4, record_width=8, seed=seed
    )


def _twin(patch: BeliefPatch) -> TorchBelief:
    twin = TorchBelief(
        patch.port,
        patch.actions,
        patch.belief,
        patch.outputs,
        iterations=patch.iterations,
        damping=patch.damping,
        record_width=patch.record_width,
    )
    twin.load(patch.parameters())
    return twin


def _inputs(patch: BeliefPatch, n: int = 2, t: int = 3, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    return (
        torch.as_tensor(rng.random((n, t, patch.inputs))),
        torch.as_tensor(rng.random((n, t, patch.actions))),
    )


def test_torch_port_packs_in_the_library_layout_and_loads_back() -> None:
    port = _port()
    tp = TorchPort(port)
    size = sum(int(np.prod(port.weight_shape(b))) for b in port.blocks)
    flat = np.random.default_rng(1).normal(size=size)
    tp.load_packed(flat)
    np.testing.assert_array_equal(tp.packed(), flat)
    assert tp(torch.zeros(2, port.inputs)).shape == (2, port.outputs)


def test_export_carries_every_slow_array_and_matches_the_patch() -> None:
    patch = _patch()
    exported = _twin(patch).export()
    params = patch.parameters()
    assert set(exported) == {"E", "e_b", "T", "t_b", "G", "g_b", "F", "f_b", "C", "c"}
    for key, value in exported.items():
        assert isinstance(value, np.ndarray) and value.dtype == np.float64, key
        np.testing.assert_allclose(value, params[key], err_msg=key)


def test_export_and_load_roundtrip_between_twins() -> None:
    patch = _patch(seed=2)
    first = _twin(patch)
    with torch.no_grad():
        first.C.normal_()
    second = _twin(_patch(seed=3))
    second.load(first.export())
    for key, value in first.export().items():
        np.testing.assert_array_equal(second.export()[key], value, err_msg=key)


def test_forward_shapes_and_imagination() -> None:
    patch = _patch()
    twin = _twin(patch)
    o, a = _inputs(patch, n=2, t=5)
    beliefs, outputs = twin(o, a)
    assert beliefs.shape == (2, 5, patch.belief)
    assert outputs.shape == (2, 5, patch.outputs)
    imagined, _ = twin(None, a, state=beliefs[:, -1])
    assert imagined.shape == beliefs.shape
    assert torch.isfinite(imagined).all()


def test_an_unobserved_row_keeps_the_expectation() -> None:
    patch = _patch()
    twin = _twin(patch)
    o, a = _inputs(patch, n=2, t=1)
    observed = torch.tensor([[True], [False]])
    beliefs, _ = twin(o, a, observed=observed)
    expected = twin.expect(torch.zeros(2, patch.belief), a[:, 0])
    torch.testing.assert_close(beliefs[1, 0], expected[1])
    assert not torch.allclose(beliefs[0, 0], expected[0])


def test_gains_accept_every_documented_shape() -> None:
    patch = _patch()
    twin = _twin(patch)
    o, a = _inputs(patch, n=2, t=3)
    blocks = len(patch.port.blocks)
    full = torch.full((2, 3, blocks), 0.5)
    reference = twin(o, a, gains=full)[0]
    torch.testing.assert_close(twin(o, a, gains=torch.full((blocks,), 0.5))[0], reference)
    torch.testing.assert_close(twin(o, a, gains=torch.full((2, blocks), 0.5))[0], reference)


def test_malformed_gains_and_masks_are_refused() -> None:
    patch = _patch()
    twin = _twin(patch)
    o, a = _inputs(patch)
    with pytest.raises(ValueError, match="gains"):
        twin(o, a, gains=torch.ones(len(patch.port.blocks) + 1))
    with pytest.raises(ValueError, match="observed"):
        twin(o, a, observed=torch.ones(2, 99, dtype=torch.bool))


def test_the_readout_learns_by_autograd() -> None:
    patch = _patch()
    twin = _twin(patch)
    o, a = _inputs(patch)
    target = torch.ones(2, 3, patch.outputs)
    optimiser = torch.optim.SGD(twin.parameters(), lr=0.5)
    losses = []
    for _ in range(5):
        optimiser.zero_grad()
        loss = ((twin(o, a)[1] - target) ** 2).mean()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
