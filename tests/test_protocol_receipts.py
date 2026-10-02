"""Protocols and receipts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence.protocol import PREDICATES, Levels
from cadence.receipts import canonical_sha256, source_manifest


def reading(mean: float, fraction: float) -> dict[str, float]:
    return {"mean": mean, "fraction": fraction}


def test_every_predicate_evaluates() -> None:
    high, low = reading(0.9, 0.9), reading(0.05, 0.02)
    assert cd.evaluate_predicate("active", high, low)
    assert cd.evaluate_predicate("inactive", low, high)
    assert cd.evaluate_predicate("reduced", low, high)
    assert cd.evaluate_predicate("retained", high, high)
    assert cd.evaluate_predicate("released", high, low)
    assert cd.evaluate_predicate("exceeds", high, low, levels=Levels())
    assert cd.evaluate_predicate("lateralized", high, low)
    sparse, dense = reading(0.5, 0.01), reading(0.5, 0.5)
    assert cd.evaluate_predicate("sparse", sparse, dense)
    assert cd.evaluate_predicate("densified", dense, sparse)
    assert not cd.evaluate_predicate("active", low, high)
    assert set(PREDICATES) >= {
        "active",
        "inactive",
        "reduced",
        "retained",
        "released",
        "exceeds",
        "lateralized",
        "sparse",
        "densified",
    }
    with pytest.raises(KeyError):
        cd.evaluate_predicate("levitates", high, low)


def ring_protocol() -> tuple[cd.Connectome, cd.Protocol]:
    n = 8
    connectome = cd.Connectome.from_synapses(
        n,
        pre=list(range(n)),
        post=[(i + 1) % n for i in range(n)],
        count=[100] * n,
        populations={"head": [0, 1], "tail": [6, 7], "none": []},
    )
    rows = [
        cd.Row("head drives tail", "touch", "tail", "active"),
        cd.Row("quiet stays quiet", "rest", "tail", "inactive"),
        cd.Row(
            "cut head, tail drops",
            "touch",
            "tail",
            "reduced",
            relative_to="touch",
            ablate=("head",),
        ),
    ]
    protocol = cd.Protocol(
        stimuli={"touch": ("head",), "rest": ("none",)},
        rows=rows,
        training=[("touch", "head", "active")],
        steps=40,
    )
    return connectome, protocol


def test_protocol_scores_and_shuffled_control_and_gain_selection() -> None:
    connectome, protocol = ring_protocol()
    brain = cd.NeuralGraph(connectome, cd.NeuronModel(gain=0.03, dt=0.5))
    result = protocol.score(brain)
    assert set(result) >= {"passed", "rows"} and 0 <= result["passed"] <= len(protocol.rows)
    assert all("id" in r and "passed" in r for r in result["rows"])
    assert protocol.neurons_for(connectome, "touch") == (0, 1)
    d = protocol.to_dict()
    assert len(d["rows"]) == 3
    gain, table = cd.select_gain(
        lambda g: cd.NeuralGraph(connectome, cd.NeuronModel(gain=g, dt=0.5)),
        protocol,
        [0.01, 0.03, 0.1],
        sparsity_cap=1.0,
    )
    assert gain in (0.01, 0.03, 0.1) and len(table) == 3
    control = cd.shuffled(connectome, seed=0)
    assert control.synapses == connectome.synapses
    assert protocol.score(cd.NeuralGraph(control, cd.NeuronModel(gain=gain, dt=0.5)))["passed"] <= len(
        protocol.rows
    )


def test_gain_selection_breaks_ties_by_gain_not_grid_order() -> None:
    connectome, protocol = ring_protocol()
    gain, table = cd.select_gain(
        lambda g: cd.NeuralGraph(connectome, cd.NeuronModel(gain=g, dt=0.5)),
        protocol,
        [0.1, 0.03, 0.01],
        sparsity_cap=None,
    )
    assert len({row["facts_passed"] for row in table}) == 1
    assert gain == 0.01


@pytest.mark.parametrize("predicate", ["exceeds", "lateralized"])
def test_readout_comparison_uses_readout_when_stimulus_has_the_same_name(predicate: str) -> None:
    connectome = cd.Connectome.from_synapses(
        2, pre=[], post=[], populations={"left": [0], "right": [1]}
    )
    protocol = cd.Protocol(
        stimuli={"trial": ("right",), "left": ("right",)},
        rows=[cd.Row("compare", "trial", "right", predicate, relative_to="left")],
        steps=10,
    )
    result = protocol.score(cd.NeuralGraph(connectome, cd.NeuronModel(dt=1.0)))["rows"][0]
    assert result["reference"]["mean"] == 0.0
    assert result["passed"]


def test_protocol_rejects_unknown_explicit_reference() -> None:
    connectome = cd.Connectome.from_synapses(2, pre=[], post=[], populations={"right": [1]})
    protocol = cd.Protocol(
        stimuli={"trial": ("right",)},
        rows=[cd.Row("retained", "trial", "right", "retained", relative_to="missing")],
        steps=10,
    )
    with pytest.raises(KeyError, match="missing"):
        protocol.score(cd.NeuralGraph(connectome, cd.NeuronModel(dt=1.0)))


def test_gain_selection_rejects_an_empty_or_inadmissible_grid() -> None:
    connectome, protocol = ring_protocol()

    def make_brain(gain: float) -> cd.NeuralGraph:
        return cd.NeuralGraph(connectome, cd.NeuronModel(gain=gain, dt=0.5))

    with pytest.raises(ValueError, match="empty"):
        cd.select_gain(make_brain, protocol, [])
    with pytest.raises(ValueError, match="sparsity cap"):
        cd.select_gain(make_brain, protocol, [0.01, 0.03, 0.1], sparsity_cap=0.0)


@pytest.mark.parametrize("n", [3, 4, 8])
def test_shuffled_preserves_endpoint_multiset_and_attached_edge_data(n: int) -> None:
    connectome = cd.Connectome.from_synapses(
        n,
        pre=np.arange(n),
        post=np.roll(np.arange(n), -1),
        count=np.arange(1, n + 1),
        sign=np.linspace(-1.0, 1.0, n),
    )
    for seed in range(20):
        shuffled = cd.shuffled(connectome, seed)
        np.testing.assert_array_equal(shuffled.in_degree(), connectome.in_degree())
        np.testing.assert_array_equal(shuffled.out_degree(), connectome.out_degree())
        assert np.all(shuffled.pre != shuffled.post)
        # Counts identify edges: shuffling must keep each source, count, and sign together.
        order = np.argsort(shuffled.count)
        original = np.argsort(connectome.count)
        for name in ("pre", "count", "sign"):
            np.testing.assert_array_equal(
                getattr(shuffled, name)[order], getattr(connectome, name)[original]
            )


def test_shuffled_keeps_selected_edges_and_validates_mask_shape() -> None:
    connectome = cd.layered(6, 4, 2, density=0.6, seed=2)
    keep = np.zeros(connectome.synapses, dtype=bool)
    keep[::2] = True
    shuffled = cd.shuffled(connectome, seed=0, keep=keep)
    original = list(
        zip(
            connectome.pre[keep],
            connectome.post[keep],
            connectome.count[keep],
            connectome.sign[keep],
            strict=True,
        )
    )
    result = list(zip(shuffled.pre, shuffled.post, shuffled.count, shuffled.sign, strict=True))
    for edge in original:
        result.remove(edge)
    np.testing.assert_array_equal(shuffled.in_degree(), connectome.in_degree())
    with pytest.raises(ValueError, match="keep"):
        cd.shuffled(connectome, seed=0, keep=np.zeros(2, dtype=bool))


def test_receipt_build_write_read_verify(tmp_path: Path) -> None:
    src = tmp_path / "code.py"
    src.write_text("x = 1\n")
    body = {"accuracy": 0.5, "b": [1, 2, {"c": np.float64(0.25)}], "arr": np.arange(3)}
    r = cd.Receipt.build("test/v1", body, sources=[("code.py", src)])
    path = r.write(tmp_path / "out" / "receipt.json")
    back = cd.Receipt.read(path)
    assert (
        back.digest == r.digest and back.body["accuracy"] == 0.5 and back.body["arr"] == [0, 1, 2]
    )
    ok, message = cd.Receipt.verify(
        path, sources=[("code.py", src)], check=lambda b: None if b["accuracy"] <= 1 else "too good"
    )
    assert ok, message
    ok, message = cd.Receipt.verify(
        path, sources=[("code.py", src)], check=lambda b: "always wrong"
    )
    assert not ok and "always wrong" in message
    src.write_text("x = 2\n")  # the code changed: the receipt no longer binds it
    ok, message = cd.Receipt.verify(path, sources=[("code.py", src)])
    assert not ok
    tampered = path.read_text().replace("0.5", "0.9")
    path.write_text(tampered)
    ok, message = cd.Receipt.verify(path)
    assert not ok
    assert canonical_sha256({"b": 1, "a": 2}) == canonical_sha256({"a": 2, "b": 1})
    with pytest.raises(ValueError):
        cd.canonical_json({"nan": float("nan")})
    assert source_manifest([("code.py", src)])["files"][0]["path"] == "code.py"
