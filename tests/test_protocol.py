"""Direct tests for cadence.protocol: levels, codes, the shuffled control, and serialisation."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd
from cadence.protocol import PREDICATES, Levels, shared_code


def reading(mean: float, fraction: float, **extra: float) -> dict[str, float]:
    return {"mean": mean, "fraction": fraction, **extra}


def test_levels_defaults_and_immutability() -> None:
    levels = Levels()
    assert (levels.active, levels.inactive, levels.margin) == (0.5, 0.2, 0.15)
    assert levels.code_level == 0.5
    with pytest.raises(AttributeError):
        levels.active = 0.9  # type: ignore[misc]


def test_predicates_hold_at_their_declared_boundaries() -> None:
    L = Levels()
    assert cd.evaluate_predicate("active", reading(L.active, 0), reading(0, 0))
    assert cd.evaluate_predicate("inactive", reading(L.inactive, 0), reading(0, 0))
    assert cd.evaluate_predicate("sparse", reading(0, L.sparse_min), reading(0, 0))
    assert cd.evaluate_predicate("sparse", reading(0, L.sparse_max), reading(0, 0))
    assert not cd.evaluate_predicate("sparse", reading(0, L.sparse_max + 1e-9), reading(0, 0))


def test_a_dead_net_cannot_pass_reference_predicates_vacuously() -> None:
    dead = reading(0.0, 0.0)
    for predicate in ("reduced", "retained", "densified", "sparse"):
        assert not cd.evaluate_predicate(predicate, dead, dead), predicate


def test_specific_needs_both_codes_and_a_small_overlap() -> None:
    apart = reading(0.5, 0.1, shared=0.1)
    together = reading(0.5, 0.1, shared=0.9)
    other = reading(0.5, 0.1)
    assert cd.evaluate_predicate("specific", apart, other)
    assert not cd.evaluate_predicate("specific", together, other)
    assert not cd.evaluate_predicate("specific", reading(0.5, 0.1), other)  # no overlap reading
    assert not cd.evaluate_predicate("specific", apart, reading(0.0, 0.0))  # versus is dead
    assert "specific" in PREDICATES


def test_shared_code_is_the_share_of_the_union() -> None:
    a = np.array([1.0, 1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 1.0, 0.0])
    assert shared_code(a, b, range(4), 0.5) == pytest.approx(1 / 3)
    assert shared_code(a, a, range(4), 0.5) == 1.0
    assert shared_code(np.zeros(4), np.zeros(4), range(4), 0.5) == 0.0
    assert shared_code(a, b, [3], 0.5) == 0.0  # members restrict the code


def ring(n: int = 12) -> cd.Connectome:
    return cd.Connectome.from_synapses(
        n,
        pre=list(range(n)) * 2,
        post=[(i + 1) % n for i in range(n)] + [(i + 3) % n for i in range(n)],
        count=list(range(1, 2 * n + 1)),
        populations={"all": range(n)},
    )


def edges(c: cd.Connectome) -> list[tuple[int, int]]:
    return sorted(zip(c.pre.tolist(), c.post.tolist(), strict=True))


def test_shuffled_keeps_counts_signs_degrees_and_populations() -> None:
    c = ring()
    s = cd.shuffled(c, seed=3)
    assert s.synapses == c.synapses
    np.testing.assert_array_equal(np.bincount(s.pre, minlength=c.n), np.bincount(c.pre, minlength=c.n))
    np.testing.assert_array_equal(np.bincount(s.post, minlength=c.n), np.bincount(c.post, minlength=c.n))
    assert sorted(s.count.tolist()) == sorted(c.count.tolist())
    assert sorted(s.sign.tolist()) == sorted(c.sign.tolist())
    assert not (s.pre == s.post).any()  # no autapse is made
    assert edges(s) != edges(c)
    assert s.populations == c.populations
    assert s.label.endswith(":shuffled:3")


def test_shuffled_is_deterministic_per_seed() -> None:
    c = ring()
    assert edges(cd.shuffled(c, seed=1)) == edges(cd.shuffled(c, seed=1))
    assert edges(cd.shuffled(c, seed=1)) != edges(cd.shuffled(c, seed=2))


def test_shuffled_leaves_kept_synapses_in_place() -> None:
    c = ring()
    keep = np.zeros(c.synapses, bool)
    keep[:5] = True
    kept = set(zip(c.pre[:5].tolist(), c.post[:5].tolist(), strict=True))
    assert kept <= set(edges(cd.shuffled(c, seed=0, keep=keep)))
    with pytest.raises(ValueError, match="keep"):
        cd.shuffled(c, seed=0, keep=np.ones(c.synapses - 1, bool))


def test_to_dict_is_json_and_carries_every_row_field() -> None:
    row = cd.Row("r", "touch", "tail", "reduced", ablate=("head",), relative_to="touch")
    protocol = cd.Protocol(stimuli={"touch": ("head",)}, rows=[row], training=[("touch", "head", "active")])
    d = json.loads(json.dumps(protocol.to_dict()))
    assert d["rows"][0] == {
        "id": "r",
        "stimulus": "touch",
        "readout": "tail",
        "predicate": "reduced",
        "reference": "",
        "ablate": ["head"],
        "relative_to": "touch",
        "tier": "experiment",
        "versus": "",
    }
    assert d["training"] == [["touch", "head", "active"]]
    assert d["levels"]["active"] == Levels().active
    assert d["predicates"] == PREDICATES


def test_score_counts_passes_by_tier() -> None:
    connectome = cd.Connectome.from_synapses(2, pre=[], post=[], populations={"a": [0], "b": [1]})
    protocol = cd.Protocol(
        stimuli={"none": ()},
        rows=[
            cd.Row("quiet a", "none", "a", "inactive", tier="core"),
            cd.Row("quiet b", "none", "b", "inactive", tier="extra"),
            cd.Row("loud b", "none", "b", "active", tier="extra"),
        ],
        steps=5,
    )
    result = protocol.score(cd.NeuralGraph(connectome, cd.NeuronModel(dt=1.0)))
    assert result["total"] == 3 and result["passed"] == 2
    assert result["passed_by_tier"] == {"core": 1, "extra": 1}
