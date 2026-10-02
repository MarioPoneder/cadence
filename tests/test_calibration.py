"""Operating points and seams: the specific fact, the calibrated threshold, the naive seam."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.protocol import Levels, shared_code


def two_codes() -> cd.Connectome:
    """Inputs A (0, 1) and B (2, 3); hidden cells 4, 5 answer A, 6, 7 answer B, 8 answers both."""
    pre = [0, 1, 0, 1, 2, 3, 2, 3, 0, 2]
    post = [4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    return cd.Connectome.from_synapses(
        9,
        pre=pre,
        post=post,
        count=[100] * len(pre),
        populations={"a": [0, 1], "b": [2, 3], "hidden": [4, 5, 6, 7, 8]},
    )


def test_shared_code_is_the_intersection_over_the_union() -> None:
    a = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1], dtype=float)
    b = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=float)
    assert shared_code(a, b, [4, 5, 6, 7, 8], 0.5) == pytest.approx(0.2)
    assert shared_code(a, a, [4, 5, 6, 7, 8], 0.5) == pytest.approx(1.0)
    assert shared_code(0 * a, 0 * b, [4, 5], 0.5) == 0.0


def test_specific_needs_two_codes_that_barely_overlap() -> None:
    code = {"mean": 0.5, "fraction": 0.6}
    assert cd.evaluate_predicate("specific", {**code, "shared": 0.2}, code)
    assert not cd.evaluate_predicate("specific", {**code, "shared": 0.4}, code)
    # a silent or near-empty code cannot be specific (a dead control has none), and a reading
    # without a comparison cannot pass
    dead = {"mean": 0.0, "fraction": 0.001}
    assert not cd.evaluate_predicate("specific", {**dead, "shared": 0.0}, code)
    assert not cd.evaluate_predicate("specific", {**code, "shared": 0.0}, dead)
    assert not cd.evaluate_predicate("specific", code, code)
    assert cd.evaluate_predicate(
        "specific", {**code, "shared": 0.4}, code, Levels(specific_max=0.5)
    )


def test_protocol_scores_a_specific_row_and_selects_on_a_specific_fact() -> None:
    connectome = two_codes()
    protocol = cd.Protocol(
        stimuli={"A": ("a",), "B": ("b",)},
        rows=[
            cd.Row("A and B are told apart", "A", "hidden", "specific", versus="B"),
            cd.Row("A is not told from itself", "A", "hidden", "specific", versus="A"),
        ],
        training=[("A", "hidden", "specific", "B")],
        steps=60,
    )
    brain = cd.Brain(connectome, cd.NeuronModel(gain=0.05))
    score = protocol.score(brain)
    told, same = score["rows"]
    assert told["passed"] and told["reading"]["shared"] == pytest.approx(0.2)
    assert not same["passed"] and same["reading"]["shared"] == pytest.approx(1.0)
    assert score["training"][0]["versus"] == "B" and score["training"][0]["passed"]
    # fraction counts the members in the code at code_level, the level specific uses
    graded = cd.Protocol(
        stimuli=protocol.stimuli, rows=protocol.rows, levels=Levels(code_level=0.05)
    )
    assert graded.score(brain)["rows"][0]["reading"]["fraction"] >= told["reading"]["fraction"]
    # a gain at which nothing settles cannot pass the fact; the selection lands on one that can
    gain, table = cd.select_gain(
        lambda g: cd.Brain(connectome, cd.NeuronModel(gain=g)),
        protocol,
        [0.001, 0.05, 0.1],
        sparsity_cap=None,
    )
    assert gain == 0.05
    assert [row["facts_passed"] for row in table] == [0, 1, 1]
    assert "A->hidden vs B" in table[1]["readings"]


def test_calibrate_bias_brings_populations_to_their_targets() -> None:
    connectome = cd.layered(4, 12, 2, density=0.6, seed=1)
    brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0))
    amp = brain.neuron_model.stimulus_amplitude
    drives = np.zeros((2, connectome.n))
    drives[0, [0, 1]] = amp
    drives[1, [2, 3]] = amp
    outputs = list(connectome.populations["output"])
    bias = cd.calibrate_bias(
        brain, drives, {"hidden": 0.3, "output": 0.5}, per_neuron=False, steps=60, tolerance=1e-4
    )
    state = brain.with_parameters(bias=bias).settle_batch(drives, steps=60, tolerance=1e-4)
    assert state.activation[:, connectome.populations["hidden"]].mean() == pytest.approx(
        0.3, abs=0.03
    )
    assert state.activation[:, outputs].mean() == pytest.approx(0.5, abs=0.03)
    # per neuron: each output cell at the target on its own, and the brain untouched
    bias2 = cd.calibrate_bias(brain, drives, {tuple(outputs): 0.5}, per_neuron=True, steps=60)
    state2 = brain.with_parameters(bias=bias2).settle_batch(drives, steps=60, tolerance=1e-4)
    for i in outputs:
        assert state2.activation[:, i].mean() == pytest.approx(0.5, abs=0.03)
    assert np.all(brain.bias == 0.0)
    with pytest.raises(ValueError):
        cd.calibrate_bias(brain, drives, {"output": 1.5})


def test_naive_efficacy_equalises_the_plastic_classes_and_keeps_the_rest() -> None:
    connectome = cd.Connectome.from_synapses(
        4, pre=[0, 1, 0, 2], post=[2, 2, 3, 3], count=[2, 8, 5, 1], sign=[1, 1, -1, 1]
    )
    plastic = np.array([True, True, False, False])
    efficacy = cd.naive_efficacy(connectome, plastic)
    weight = connectome.count * efficacy
    assert weight[0] == pytest.approx(weight[1]) and weight[0] > 0
    assert efficacy[2] == -1.0 and efficacy[3] == 1.0
    with pytest.raises(ValueError):
        cd.naive_efficacy(connectome, np.array([True]))


def test_seam_report_counts_classes_and_coverage() -> None:
    connectome = cd.Connectome.from_synapses(
        6,
        pre=[0, 1, 0, 3],
        post=[4, 4, 5, 5],
        count=[3, 7, 2, 9],
        populations={"pre": [0, 1, 2], "post": [4, 5]},
    )
    report = cd.seam_report(connectome, "pre", "post")
    assert report["classes"] == 3 and report["synapses"] == 12.0
    assert report["coverage_pre"] == pytest.approx(2 / 3)
    assert report["classes_per_post"] == {4: 2, 5: 1}


def test_preflight_names_each_finding_and_its_remedy() -> None:
    # inputs A (0, 1) and B (2, 3); hidden 4, 5 answer A, 6, 7 answer B, 8 both; outputs 9, 10 from every hidden cell
    pre = [0, 1, 0, 1, 2, 3, 2, 3, 0, 2] + [h for h in range(4, 9) for _ in (9, 10)]
    post = [4, 4, 5, 5, 6, 6, 7, 7, 8, 8] + [o for _ in range(4, 9) for o in (9, 10)]
    count = [100] * 10 + [20] * 10
    connectome = cd.Connectome.from_synapses(
        11,
        pre=pre,
        post=post,
        count=count,
        populations={"a": [0, 1], "b": [2, 3], "hidden": [4, 5, 6, 7, 8], "output": [9, 10]},
    )
    outputs = [9, 10]
    hidden = np.zeros(connectome.n, dtype=bool)
    hidden[[4, 5, 6, 7, 8]] = True
    plastic = hidden[connectome.pre]
    brain = cd.Brain(connectome, cd.NeuronModel(gain=0.05))
    amp = brain.neuron_model.stimulus_amplitude
    drives = np.zeros((2, connectome.n))
    drives[0, [0, 1]] = amp
    drives[1, [2, 3]] = amp
    calibrated = brain.with_parameters(
        bias=cd.calibrate_bias(brain, drives, {"output": 0.5}, per_neuron=True, steps=60)
    )
    clean = cd.preflight(calibrated, outputs, plastic, drives, eligibility_min=1, steps=60)
    assert clean["warnings"] == []
    assert clean["shared"][(0, 1)] == pytest.approx(0.2) and clean["eligibility"] == {
        9: [3, 3],
        10: [3, 3],
    }
    # a readout on its rail, the same code under both drives, a seam with too little active input
    on_rail = cd.preflight(brain, outputs, plastic, drives, eligibility_min=1, steps=60)
    assert any("calibrate_bias" in w and "output 9" in w for w in on_rail["warnings"])
    same = cd.preflight(calibrated, outputs, plastic, np.vstack([drives[0], drives[0]]), steps=60)
    assert any("specific fact" in w for w in same["warnings"])
    assert any("same situation" in w for w in same["warnings"]) and same["inputs"][(0, 1)] == 1.0
    assert clean["inputs"][(0, 1)] == 0.0
    thin = cd.preflight(calibrated, outputs, plastic, drives, eligibility_min=100, steps=60)
    assert any("seam is thin" in w for w in thin["warnings"])
    with pytest.raises(ValueError):
        cd.preflight(brain, outputs, np.array([True]), drives)
