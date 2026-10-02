"""A learned temporal model must retain, imagine privately and preserve old skill."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def example():
    path = Path(__file__).resolve().parents[1] / "examples" / "memory_imagination.py"
    spec = importlib.util.spec_from_file_location("memory_imagination_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return example().run()


def test_acquisition_and_protected_plasticity_are_both_measured(report):
    assert report["cue_mse_after"] < report["cue_mse_before"] * 0.01
    assert report["action_mse_after"] < report["action_mse_before"] * 0.001
    assert report["protected_response_max_change"] < 1e-12
    assert report["protected_ranks"] == {"A": 1, "B": 1, "C": 1}
    assert report["learning_updates"] == report["live_parameter_updates"] == 200


def test_same_current_inputs_need_retained_history_and_reset_erases_it(report):
    negative, positive = report["recalled_after_three_neutral_events"]
    assert negative < -0.08 < 0.08 < positive
    assert report["erased_memory_answer"] == 0


def test_private_planning_changes_real_consequences_without_mutating_live_brain(report):
    assert report["private_queries_preserved_model"]
    assert report["private_queries_preserved_protection"]
    assert report["restored_continuation_identical"]
    assert report["predicted_final_half_mse"] < report["predicted_initial_half_mse"] * 1e-5
    assert report["measured_goal_mse"] < 1e-4
    assert report["measured_goal_mse"] < report["no_action_goal_mse"] * 0.01
    inputs = np.asarray(report["planned_inputs"])
    np.testing.assert_array_equal(inputs[:, :, 0], 0)
    assert np.abs(inputs[:, :, 1]).max() <= 1
    # Independent scalar environment replay: the first observation is a cue,
    # after which only proposed actions are applied. No desired output is read.
    state = 0.6 * 0.8
    values = []
    for action in inputs[0, :, 1]:
        state = 0.7 * np.tanh(state) + 0.6 * action
        values.append(0.8 * np.tanh(state))
    np.testing.assert_allclose(np.asarray(report["measured_outputs"])[0, :, 0], values)


def test_a_zero_memory_connection_cannot_pass_the_delayed_cue_check():
    module = example()
    from cadence import TemporalPatchNet

    model = TemporalPatchNet(2, 1, 1)
    model.set_parameters({"A": np.zeros((1, 1)), "B": np.ones((1, 2)), "C": np.ones((1, 1))})
    for cue in (-0.8, 0.8):
        model.reset()
        model.advance(np.array([[[cue, 0.0]]]))
        np.testing.assert_array_equal(model.advance(np.zeros((1, 3, 2))).output, 0)
    # The actual environment still distinguishes these histories.
    _, boundary = module.body(np.array([[[0.8, 0.0]]]))
    measured, _ = module.body(np.zeros((1, 3, 2)), boundary)
    assert measured[0, -1, 0] > 0.08
