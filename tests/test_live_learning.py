"""Behavioral gates require free decisions, not output clamps or stationarity."""

import importlib.util
from pathlib import Path

import pytest


def load_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "live_learning.py"
    spec = importlib.util.spec_from_file_location("live_learning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", (0, 2, 7))
@pytest.mark.parametrize("gate", ("hidden_cue", "retention"))
def test_memory_and_retention(seed, gate):
    result = getattr(load_example(), gate)(seed)
    assert result["passed"], result


def test_delayed_reward_and_reversal():
    # Other fixed seeds are exercised by the runnable experiment; keep CI bounded.
    result = load_example().delayed_reward(0)
    assert result["passed"], result
    assert result["frozen_phases_correct"] == 1
