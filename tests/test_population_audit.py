"""The population kernel's observe stages every change and commits none on a bad moment."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cadence.population import PopulationPatch  # noqa: E402


def _patch(streams=3, dtype=torch.float64, device="cpu", optimizer="sgd", seed=0):
    patch = PopulationPatch(
        12, 8, 3, instances=2, streams=streams, seed=seed, cells=64, active=4, device=device, dtype=dtype
    )
    patch.optimizer = optimizer
    return patch


def _as_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _snapshot(patch):
    state = {}
    for key, value in patch.state().items():
        if isinstance(value, dict):
            state.update({f"{key}:{k}": _as_array(v) for k, v in value.items()})
        else:
            state[key] = _as_array(value)
    state.update({f"param:{k}": v.detach().cpu().numpy().copy() for k, v in patch.parameters().items()})
    state.update({f"adam:{k}": (m.cpu().numpy().copy(), v.cpu().numpy().copy()) for k, (m, v) in patch._adam.items()})
    state["adam_t"] = patch._adam_t.cpu().numpy().copy()
    state["written"] = patch.written.cpu().numpy().copy()
    state["settles"] = patch.settles
    return state


def _assert_unchanged(patch, before):
    after = _snapshot(patch)
    assert after.keys() == before.keys()
    for key, value in before.items():
        if isinstance(value, tuple):
            for a, b in zip(after[key], value, strict=True):
                np.testing.assert_array_equal(a, b, err_msg=key)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(after[key], value, err_msg=key)
        else:
            assert after[key] == value, key


def _moment(patch, rng, moments=None):
    n = patch.B if moments is None else moments
    x = torch.as_tensor(rng.normal(size=(patch.P, n, 12)), dtype=patch.dtype)
    t = torch.as_tensor(rng.normal(size=(patch.P, n, 3)), dtype=patch.dtype)
    return x, t


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
@pytest.mark.parametrize("folded", [False, True])
@pytest.mark.parametrize("bad", ["target", "input", "rate", "weight"])
def test_a_nonfinite_moment_is_rejected_before_any_change(device, optimizer, folded, bad):
    patch = _patch(optimizer=optimizer, device=device, dtype=torch.float64 if device == "cpu" else torch.float32)
    rng = np.random.default_rng(1)
    stream_of = torch.tensor([0, 2, 2, 1]) if folded else None
    x, t = _moment(patch, rng, 4 if folded else None)
    patch.observe(x, t, rate=0.1, stream_of=stream_of)  # a first, valid moment fills the moments and stores
    x, t = _moment(patch, rng, 4 if folded else None)
    rate, weight = 0.1, None
    if bad == "target":
        t[1, 2, 1] = torch.nan
    elif bad == "input":
        x[0, 1, 5] = torch.inf
    elif bad == "rate":
        rate = torch.tensor([0.1, torch.nan], dtype=patch.dtype)
    else:
        weight = torch.tensor([1.0, torch.inf, 1.0], dtype=patch.dtype)
    before = _snapshot(patch)
    with pytest.raises(ValueError, match="finite"):
        patch.observe(x, t, rate=rate, stream_of=stream_of, weight=weight)
    _assert_unchanged(patch, before)


def test_a_masked_out_moment_may_carry_a_nonfinite_target():
    """A masked moment carries no lesson: its target is ignored, so NaN there is a legal way to
    say 'nothing', and the outcome equals the same observe with any finite dummy target."""
    rng = np.random.default_rng(2)
    a, b = _patch(), _patch()
    x, t = _moment(a, rng)
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=a.dtype)
    t_nan = t.clone()
    t_nan[0, 1] = torch.nan
    t_nan[1, 2] = torch.inf
    out_a = a.observe(x, t_nan, rate=0.3, mask=mask)
    out_b = b.observe(x, t, rate=0.3, mask=mask)
    for key in ("loss", "slow", "hidden", "residual", "code"):
        np.testing.assert_allclose(out_a[key].numpy(), out_b[key].numpy(), atol=1e-12, err_msg=key)
    assert np.isfinite(out_a["residual"].numpy()).all()
    assert (out_a["residual"][0, 1].abs().sum() == 0) and (out_a["residual"][1, 2].abs().sum() == 0)
    for name, value in a.parameters().items():
        np.testing.assert_allclose(value.numpy(), b.parameters()[name].numpy(), atol=1e-12, err_msg=name)
    np.testing.assert_array_equal(a.tables.numpy(), b.tables.numpy())
    assert np.isfinite(a.tables.numpy()).all() and np.isfinite(a.mean.numpy()).all()


@pytest.mark.parametrize("device", DEVICES)
def test_an_overflowing_write_is_rejected_atomically(device):
    """Four queued moments of one stream, each moving a row already near the top of float32,
    would sum past it: the write is summed on a copy of its rows, seen to overflow, and
    refused with the store, its statistics and the parameters untouched."""
    patch = _patch(streams=2, dtype=torch.float32, device=device)
    rng = np.random.default_rng(3)
    patch.tables.fill_(2e38)
    x = torch.as_tensor(rng.normal(size=(patch.P, 1, 12)), dtype=torch.float32, device=device).expand(-1, 4, -1).clone()
    t = torch.full((patch.P, 4, 3), 3e38, dtype=torch.float32, device=device)
    before = _snapshot(patch)
    with pytest.raises(ValueError, match="nonfinite table"):
        patch.observe(x, t, rate=0.0, stream_of=torch.tensor([0, 0, 0, 0], device=device))
    _assert_unchanged(patch, before)


def test_an_overflowing_slow_step_is_rejected_with_its_write():
    """A target that overflows Adam's second moment in float32 is refused before the moments,
    the parameters or the store change."""
    patch = _patch(dtype=torch.float32, optimizer="adam")
    rng = np.random.default_rng(4)
    x, t = _moment(patch, rng)
    patch.observe(x, t, rate=0.01)
    x, t = _moment(patch, rng)
    t = t * 1e30
    before = _snapshot(patch)
    with pytest.raises(ValueError, match="nonfinite parameter or moment"):
        patch.observe(x, t, rate=0.01)
    _assert_unchanged(patch, before)


def test_a_large_finite_moment_still_lands():
    patch = _patch(dtype=torch.float32)
    rng = np.random.default_rng(5)
    x, t = _moment(patch, rng)
    out = patch.observe(x, t * 1e6, rate=0.0, stream_of=None)
    assert patch.writes == 1 and np.isfinite(patch.tables.numpy()).all()
    assert np.isfinite(out["residual"].numpy()).all()


def test_folded_writes_match_the_unfolded_path_and_run_on_the_accelerator():
    """The staged path with duplicate streams equals sequential single-stream observes at rate
    zero (writes against the store as it stood), on the cpu and, when present, on the
    machine's accelerator in float32."""
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2, 4, 12))
    t = rng.normal(size=(2, 4, 3))
    for device in devices:
        dtype = torch.float64 if device == "cpu" else torch.float32
        folded = _patch(streams=3, dtype=dtype, device=device)
        plain = _patch(streams=3, dtype=dtype, device=device)
        xs, ts = torch.as_tensor(x, dtype=dtype, device=device), torch.as_tensor(t, dtype=dtype, device=device)
        folded.observe(xs, ts, rate=0.0, stream_of=torch.tensor([1, 1, 0, 1], device=device))
        # the same lessons one stream at a time, every write against the store before the call
        held = plain.imagine(xs[:, [0, 1, 3]], stream_of=torch.tensor([1, 1, 1], device=device))["read"]
        assert held.abs().sum() == 0
        for moment, stream in ((2, 0),):
            plain.observe(xs[:, [moment]], ts[:, [moment]], rate=0.0, stream_of=torch.tensor([stream], device=device))
        out = folded.imagine(xs, stream_of=torch.tensor([1, 1, 0, 1], device=device))["output"]
        assert torch.isfinite(out).all() and folded.writes == 1
        assert folded.tables[:, 2].abs().sum() == 0  # the untouched stream stays empty
        np.testing.assert_allclose(
            folded.tables[:, 0].cpu().numpy(), plain.tables[:, 0].cpu().numpy(), atol=1e-6 if dtype == torch.float32 else 1e-12
        )
