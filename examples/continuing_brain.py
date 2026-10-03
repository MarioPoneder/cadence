"""One System 1 life through bootstrap, unchanged conditions and disruption.

Run after installing cadence-net: python examples/continuing_brain.py
Or from this checkout: PYTHONPATH=src python examples/continuing_brain.py

Two sensory cues have observed action labels. The independent environment
rewards the action actually executed, then changes its rule during the same
life. The brain receives outcomes without a notice that the rule changed.
Brain.compose supplies local state, reciprocal regions, readback, working
trace and consolidating memory. Optional observers can extend the same graph.
Teaching uses the default finite rule; qualified teaching is an explicit choice.
Outcome counts describe this short run, not a promised acquisition/recovery
rate. Unchanged conditions do not mean the brain has learned a stable world
model. See docs/world-model.md for the design and current integration boundaries.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from cadence import Brain


def observation(cue):
    return np.eye(2)[[cue]]


def label(cue, *, changed=False):
    """The independent body's rule, reversed after the declared disturbance."""
    return np.array([cue if changed else 1 - cue])


def execute(action, cue, *, changed=False):
    """Execute one action, measure its outcome, then reveal the next cue."""
    reward = (action == label(cue, changed=changed)).astype(float)
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
    outcomes = []
    stages = []
    for phase, transitions, changed, teach in (
        ("bootstrap", 16, False, True),
        ("unchanged_environment", 8, False, False),
        ("changed_environment", 8, True, False),
    ):
        correct = 0
        for _ in range(transitions):
            # Score the issued action before revealing its measured outcome.
            reward, next_cue = execute(action, cue, changed=changed)
            outcomes.append({
                "phase": phase, "cue": cue, "action": int(action[0]),
                "observed_target": int(label(cue, changed=changed)[0]),
                "reward": float(reward[0]),
            })
            correct += int(reward[0])
            cue = next_cue
            action = brain.step(
                observation(cue), reward=reward, done=np.array([False]),
                teacher=label(cue, changed=changed) if teach else None,
            )
        stages.append({
            "phase": phase, "executed_transitions": transitions,
            "correct_actions": correct, "witnessed_mistakes": transitions - correct,
        })

    # No reset or replacement across stages: both memory pathways remain live.
    assert brain.working_memory is not None
    assert brain.hippocampus is not None
    assert brain.hippocampus.writes == len(outcomes)

    with tempfile.TemporaryDirectory(prefix="cadence-continuing-") as folder:
        folder = Path(folder)
        # This checkpoint owns an issued action still awaiting its actual outcome.
        saved = brain.save(folder / "pending.npz")
        resumed = Brain.load(saved)
        reward, cue = execute(action, cue, changed=True)
        next_observation = observation(cue)
        next_action = brain.step(
            next_observation, reward=reward, done=np.array([False]),
        )
        resumed_action = resumed.step(
            next_observation, reward=reward, done=np.array([False]),
        )
        np.testing.assert_array_equal(next_action, resumed_action)
        before = brain.save(folder / "continued.npz")
        assert same_checkpoint(before, resumed.save(folder / "resumed.npz"))

        # Separate unit control: predict omits both memory pathways. It does
        # not measure the continuing policy or consume its pending feedback.
        recall = brain.predict(np.eye(2))
        assert recall.shape == (2,)
        assert same_checkpoint(before, brain.save(folder / "after-recall.npz"))

    return {
        "stages": stages,
        "outcomes": outcomes,
        "executed_transitions": len(outcomes) + 1,
        "associative_writes": brain.hippocampus.writes,
        "memory_free_graph_control": {
            "predictions": recall.tolist(), "changed_environment_labels": [0, 1],
        },
        "saved_feedback_continuation_identical": True,
        "free_queries_preserved_live_checkpoint": True,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
