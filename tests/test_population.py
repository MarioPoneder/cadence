import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cadence import RecordPatchNet  # noqa: E402
from cadence.population import PopulationPatch  # noqa: E402


def _net(groups=None, seed=3):
    return RecordPatchNet(
        12, 8, 3, seed=seed, cells=64, active=4, record_rate=0.5, slowest=2.0, groups=groups
    )


@pytest.mark.parametrize("groups", [None, (3,)])
def test_prediction_parity_with_the_numpy_patch(groups) -> None:
    net = _net(groups)
    twin = PopulationPatch.from_patch(
        net, instances=2, streams=3, device="cpu", dtype=torch.float64
    )
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 12))
    out = twin.imagine(torch.as_tensor(x))
    for p in range(2):
        net.reset()
        ref = net.imagine(x[p][:, None, :])
        np.testing.assert_allclose(out["output"][p].numpy(), ref.output[:, 0], atol=1e-9)
        np.testing.assert_allclose(out["hidden"][p].numpy(), ref.hidden[:, -1], atol=1e-9)


@pytest.mark.parametrize("groups", [None, (3,)])
def test_slow_step_parity_with_the_adjoint(groups) -> None:
    net = _net(groups)
    twin = PopulationPatch.from_patch(
        net, instances=1, streams=1, device="cpu", dtype=torch.float64
    )
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 1, 12))
    t = rng.normal(size=(1, 1, 3))
    if groups is not None:
        t = np.eye(3)[rng.integers(3, size=(1, 1))]
    net.reset()
    net.observe(x, t, rate=0.3, backtrack=False, write=False)
    twin.observe(torch.as_tensor(x), torch.as_tensor(t), rate=0.3, write=False)
    ref = net.parameters()
    for name, key in (("G", "G"), ("g", "g"), ("B", "B"), ("b", "b"), ("C", "C"), ("c", "c")):
        np.testing.assert_allclose(
            twin.parameters()[key][0].numpy(), ref[name], atol=1e-9, err_msg=name
        )


def test_write_and_read_parity() -> None:
    net = _net()
    twin = PopulationPatch.from_patch(
        net, instances=1, streams=1, device="cpu", dtype=torch.float64
    )
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 1, 12))
    t = rng.normal(size=(1, 1, 3))
    net.reset()
    net.observe(x, t, rate=0.0, backtrack=False, write=True)
    twin.observe(torch.as_tensor(x), torch.as_tensor(t), rate=0.0, write=True)
    net.reset()
    ref = net.imagine(x).output[0, 0]
    out = twin.imagine(torch.as_tensor(x))["output"][0, 0].numpy()
    np.testing.assert_allclose(out, ref, atol=1e-9)
    slow = twin.imagine(torch.as_tensor(x))["slow"][0, 0].numpy()
    assert np.abs(out - t[0, 0]).max() < np.abs(slow - t[0, 0]).max()


def test_instances_and_streams_are_independent() -> None:
    twin = PopulationPatch(
        12,
        8,
        3,
        instances=3,
        streams=2,
        seed=0,
        cells=64,
        active=4,
        device="cpu",
        dtype=torch.float64,
    )
    rng = np.random.default_rng(3)
    x = torch.as_tensor(rng.normal(size=(3, 2, 12)))
    t = torch.as_tensor(rng.normal(size=(3, 2, 3)))
    before = twin.imagine(x)["output"].clone()
    twin.observe(x, t, rate=torch.tensor([0.5, 0.0, 0.5], dtype=torch.float64), write=False)
    after = twin.imagine(x)["output"]
    assert torch.allclose(before[1], after[1])  # rate zero: untouched
    assert not torch.allclose(before[0], after[0])
    # a write in stream 0 leaves stream 1's store alone
    twin.observe(x, t, rate=0.0, write=True)
    assert twin.tables[:, 0].abs().sum() > 0
    twin.tables[:, 1].zero_()
    out = twin.imagine(x)
    assert torch.allclose(out["read"][:, 1], torch.zeros_like(out["read"][:, 1]))


def test_inherit_copies_parents() -> None:
    twin = PopulationPatch(
        12,
        8,
        3,
        instances=4,
        streams=1,
        seed=0,
        cells=64,
        active=4,
        device="cpu",
        dtype=torch.float64,
    )
    rng = np.random.default_rng(4)
    x = torch.as_tensor(rng.normal(size=(4, 1, 12)))
    t = torch.as_tensor(rng.normal(size=(4, 1, 3)))
    twin.observe(x, t, rate=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64), write=False)
    twin.inherit(torch.tensor([0, 0, 0, 0]))
    out = twin.imagine(x[:1].expand(4, -1, -1))["output"]
    assert torch.allclose(out[0], out[3])


def test_masked_streams_neither_learn_nor_write() -> None:
    """A masked observe equals an observe over the masked streams alone: the slow step is
    their mean, the other streams' stores and statistics stay put."""
    full = PopulationPatch(
        12, 8, 3, instances=1, streams=3, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64
    )
    part = PopulationPatch(
        12, 8, 3, instances=1, streams=2, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64
    )
    rng = np.random.default_rng(5)
    x = torch.as_tensor(rng.normal(size=(1, 3, 12)))
    t = torch.as_tensor(rng.normal(size=(1, 3, 3)))
    full.observe(x, t, rate=0.3, write=True, mask=torch.tensor([[1.0, 0.0, 1.0]]))
    part.observe(x[:, [0, 2]], t[:, [0, 2]], rate=0.3, write=True)
    for name in ("G", "g", "B", "b", "C", "c"):
        np.testing.assert_allclose(full.parameters()[name].numpy(), part.parameters()[name].numpy(), atol=1e-12, err_msg=name)
    assert full.tables[:, 1].abs().sum() == 0
    assert full.seen[0, 1] == 0
    np.testing.assert_allclose(full.tables[:, [0, 2]].numpy(), part.tables.numpy(), atol=1e-12)
    np.testing.assert_allclose(full.input_norm[:, [0, 2]].numpy(), part.input_norm.numpy(), atol=1e-12)


def test_moments_folded_into_the_batch_by_stream() -> None:
    """Several moments per stream in one call: the imagined readings equal the per-stream
    imaginations, and an observe over moments on distinct streams equals the plain one."""
    twin = PopulationPatch(
        12, 8, 3, instances=2, streams=3, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64
    )
    rng = np.random.default_rng(6)
    x = torch.as_tensor(rng.normal(size=(2, 3, 12)))
    t = torch.as_tensor(rng.normal(size=(2, 3, 3)))
    twin.observe(x, t, rate=0.2, write=True)  # stores with content
    y = torch.as_tensor(rng.normal(size=(2, 6, 12)))
    stream_of = torch.tensor([0, 0, 1, 1, 2, 2])
    folded = twin.imagine(y, stream_of=stream_of)["output"]
    for k in range(6):
        one = twin.imagine(y[:, k : k + 1].expand(-1, 3, -1))["output"][:, stream_of[k]]
        np.testing.assert_allclose(folded[:, k].numpy(), one.numpy(), atol=1e-12)
    other = PopulationPatch(
        12, 8, 3, instances=2, streams=3, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64
    )
    other.observe(x, t, rate=0.2, write=True)
    perm = torch.tensor([2, 0, 1])
    twin.observe(x[:, perm], t[:, perm], rate=0.3, write=True, stream_of=perm)
    other.observe(x, t, rate=0.3, write=True)
    for name in ("G", "g", "B", "b", "C", "c"):
        np.testing.assert_allclose(twin.parameters()[name].numpy(), other.parameters()[name].numpy(), atol=1e-12, err_msg=name)
    np.testing.assert_allclose(twin.tables.numpy(), other.tables.numpy(), atol=1e-12)
    np.testing.assert_allclose(twin.mean.numpy(), other.mean.numpy(), atol=1e-12)
    np.testing.assert_allclose(twin.seen.numpy(), other.seen.numpy(), atol=1e-12)
    np.testing.assert_allclose(twin.input_norm.numpy(), other.input_norm.numpy(), atol=1e-12)


def test_output_weights_reweight_the_squared_errors() -> None:
    """A weight of the outputs' count on one output and zero elsewhere trains that output
    alone; all ones is the plain mean."""
    a = PopulationPatch(12, 8, 3, instances=1, streams=2, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64)
    b = PopulationPatch(12, 8, 3, instances=1, streams=2, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64)
    rng = np.random.default_rng(7)
    x = torch.as_tensor(rng.normal(size=(1, 2, 12)))
    t = torch.as_tensor(rng.normal(size=(1, 2, 3)))
    a.observe(x, t, rate=0.3, write=False)
    b.observe(x, t, rate=0.3, write=False, weight=torch.ones(3, dtype=torch.float64))
    np.testing.assert_allclose(a.parameters()["C"].numpy(), b.parameters()["C"].numpy(), atol=1e-12)
    c = PopulationPatch(12, 8, 3, instances=1, streams=2, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64)
    before = c.parameters()["C"].clone()
    c.observe(x, t, rate=0.3, write=False, weight=torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64))
    after = c.parameters()["C"]
    assert torch.allclose(before[0, 1:], after[0, 1:])  # the unweighted outputs' readout rows did not move
    assert not torch.allclose(before[0, 0], after[0, 0])


def test_record_weight_damps_an_outputs_records() -> None:
    twin = PopulationPatch(12, 8, 3, instances=1, streams=1, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64)
    twin.record_weight = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    rng = np.random.default_rng(8)
    x = torch.as_tensor(rng.normal(size=(1, 1, 12)))
    t = torch.as_tensor(rng.normal(size=(1, 1, 3)))
    twin.observe(x, t, rate=0.0, write=True)
    read = twin.imagine(x)["read"][0, 0]
    assert read[1] == 0.0 and read[0] != 0.0 and read[2] != 0.0


def test_familiarity_rises_where_the_store_has_written() -> None:
    twin = PopulationPatch(12, 8, 3, instances=1, streams=2, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64)
    rng = np.random.default_rng(9)
    x = torch.as_tensor(rng.normal(size=(1, 2, 12)))
    t = torch.as_tensor(rng.normal(size=(1, 2, 3)))
    assert float(twin.imagine(x)["familiarity"].max()) == 0.0
    twin.observe(x, t, rate=0.0, write=True, mask=torch.tensor([[1.0, 0.0]]))
    fam = twin.imagine(x)["familiarity"]
    assert float(fam[0, 0]) > 0.99 and float(fam[0, 1]) == 0.0  # the written stream knows the reading, the other does not
    y = torch.as_tensor(rng.normal(size=(1, 2, 12)))
    assert float(twin.imagine(y)["familiarity"][0, 0]) < 1.0


def test_adam_moves_the_weights_and_respects_a_zero_rate() -> None:
    sgd = PopulationPatch(12, 8, 3, instances=2, streams=4, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64)
    adam = PopulationPatch(12, 8, 3, instances=2, streams=4, seed=0, cells=64, active=4, device="cpu", dtype=torch.float64)
    adam.optimizer = "adam"
    rng = np.random.default_rng(10)
    x = torch.as_tensor(rng.normal(size=(2, 4, 12)) * 0.05)  # small readings: a tiny context, the regime where the plain step stalls
    t = torch.as_tensor(rng.normal(size=(2, 4, 3)))
    c0 = adam.parameters()["C"].clone()
    for _ in range(20):
        sgd.observe(x, t, rate=torch.tensor([0.3, 0.0], dtype=torch.float64), write=False)
        adam.observe(x, t, rate=torch.tensor([0.01, 0.0], dtype=torch.float64), write=False)
    moved_sgd = float((sgd.parameters()["C"][0] - c0[0]).abs().mean())
    moved_adam = float((adam.parameters()["C"][0] - c0[0]).abs().mean())
    assert moved_adam > 3 * moved_sgd  # the moments lift a tiny gradient to the rate's scale (here 8x at a thirtieth of the rate)
    assert torch.equal(adam.parameters()["C"][1], c0[1])  # the instance at rate zero did not move
