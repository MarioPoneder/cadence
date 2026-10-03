"""A small continuing System 1 brain, using the published 0.70 API.

Run after installing cadence-net: python examples/continuing_brain.py
Or from this checkout: PYTHONPATH=src python examples/continuing_brain.py

Two sensory cues have observed action labels. The independent environment
rewards the action actually executed; it never rewards an imagined answer.
Brain.compose supplies local state, reciprocal regions, readback, working
trace and consolidating memory. Optional observers can extend the same graph.
Teaching uses the finite rule; this example does not require unreleased APIs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from cadence import Brain


def observation(cue):
    return np.eye(2)[[cue]]


def label(cue):
    """The environment's observed relation: cue 0 -> action 1, cue 1 -> action 0."""
    return np.array([1 - cue])


def execute(action, cue):
    """Execute one action, measure its outcome, then reveal the next cue."""
    reward = (action == label(cue)).astype(float)
    return reward, 1 - cue


def same_checkpoint(first, second):
    with np.load(first, allow_pickle=False) as aa, np.load(second, allow_pickle=False) as bb:
        return set(aa.files) == set(bb.files) and all(
            np.array_equal(aa[key], bb[key]) for key in aa.files
        )


def run():
    brain = Brain.compose(2, 2, modules=(8,), observers=(), backend="cpu", seed=7)
    cue = 0
    # teacher describes this cue. There is no previous action to reward yet.
    action = brain.step(observation(cue), teacher=label(cue))
    transitions = 16
    for _ in range(transitions):
        reward, cue = execute(action, cue)
        action = brain.step(
            observation(cue), reward=reward, done=np.array([True]), teacher=label(cue),
        )

    with tempfile.TemporaryDirectory(prefix="cadence-continuing-") as folder:
        folder = Path(folder)
        # This checkpoint owns an issued action still awaiting its actual outcome.
        saved = brain.save(folder / "pending.npz")
        resumed = Brain.load(saved)
        reward, cue = execute(action, cue)
        next_observation = observation(cue)
        next_action = brain.step(
            next_observation, reward=reward, done=np.array([True]), teacher=label(cue),
        )
        resumed_action = resumed.step(
            next_observation, reward=reward, done=np.array([True]), teacher=label(cue),
        )
        np.testing.assert_array_equal(next_action, resumed_action)
        before = brain.save(folder / "continued.npz")
        assert same_checkpoint(before, resumed.save(folder / "resumed.npz"))

        # Free predictions omit both the working trace and associative memory.
        # The teacher is absent, and these queries do not consume pending feedback.
        recall = brain.predict(np.eye(2))
        assert recall.shape == (2,)
        assert same_checkpoint(before, brain.save(folder / "after-recall.npz"))

    return {
        "executed_transitions": transitions + 1,
        "demonstrated_cues": transitions + 2,
        "free_recall": recall.tolist(),
        "observed_labels": [1, 0],
        "saved_feedback_continuation_identical": True,
        "free_queries_preserved_live_checkpoint": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
