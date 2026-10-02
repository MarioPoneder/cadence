"""Learn, remember, imagine and protect responses with one small temporal model.

Run: PYTHONPATH=src python examples/memory_imagination.py

The body below is a declared deterministic test environment, not a second
controller. Its observations teach the model; planning receives only a goal
and a mask of action ports. Protection is explicitly requested and finite.
This does not demonstrate autonomous importance selection or a general brain.
"""

from __future__ import annotations

import json

import numpy as np

from cadence import TemporalMemory, TemporalPatchNet


def body(inputs, state=None):
    """Independent environment: a fading hidden quantity, cue/action, and sensor."""
    state = np.zeros((len(inputs), 1)) if state is None else state.copy()
    outputs = []
    for drive in inputs.transpose(1, 0, 2):
        state = 0.7 * np.tanh(state) + 0.6 * drive.sum(axis=1, keepdims=True)
        outputs.append(0.8 * np.tanh(state))
    return np.stack(outputs, axis=1), state


def same_snapshot(before, after):
    return before.keys() == after.keys() and all(
        np.array_equal(before[key], after[key]) for key in before
    )


def prepare():
    """Two bounded learning stages; retained old responses constrain the second."""
    model = TemporalPatchNet(2, 1, 1, seed=7, tolerance=1e-10)
    # A declared weak initialization, not the environment's learned answer.
    model.set_parameters(
        {"A": np.array([[0.3]]), "B": np.array([[0.05, 0.0]]), "C": np.array([[0.4]])}
    )
    cues = np.zeros((4, 4, 2))
    cues[:, 0, 0] = [-0.8, -0.4, 0.4, 0.8]
    targets, _ = body(cues)
    cold = np.zeros((4, 1))

    def mse(inputs, expected):
        actual = model.imagine(inputs, state=np.zeros((len(inputs), 1))).output
        return float(np.mean(np.square(actual - expected)))

    cue_before = mse(cues, targets)
    for _ in range(120):
        model.reset()
        result = model.observe(cues, targets, beta=0.01, rate=0.5)
        if not result.updated:
            raise RuntimeError(f"Cue learning refused: {result.reason}")
    cue_after = mse(cues, targets)
    protected = model.imagine(cues, state=cold).output
    memory = TemporalMemory()
    protection = memory.protect(model, cues, state=cold)
    actions = np.zeros((4, 1, 2))
    actions[:, 0, 1] = [-0.8, -0.4, 0.4, 0.8]
    outcomes, _ = body(actions)
    action_before = mse(actions, outcomes)
    for _ in range(80):
        model.reset()
        result = memory.observe(model, actions, outcomes, beta=0.01, rate=0.5)
        if not result.updated:
            raise RuntimeError(f"Protected action learning refused: {result.reason}")
    report = {
        "cue_mse_before": cue_before,
        "cue_mse_after": cue_after,
        "action_mse_before": action_before,
        "action_mse_after": mse(actions, outcomes),
        "protected_response_max_change": float(
            np.max(np.abs(model.imagine(cues, state=cold).output - protected))
        ),
        "protected_ranks": protection.ranks,
        "protection_bytes": protection.bytes,
        "learning_updates": model.updates,
        "presented_paths": 4 * (120 + 80),
        "presented_events": 4 * (120 * 4 + 80),
    }
    return model, memory, report


def run():
    model, memory, report = prepare()
    neutral = np.zeros((1, 3, 2))
    answers = []
    for cue in (-0.8, 0.8):
        model.reset()
        model.advance(np.array([[[cue, 0.0]]]))
        answers.append(float(model.advance(neutral).output[0, -1, 0]))
    model.reset()
    reset_answer = float(model.advance(neutral).output[0, -1, 0])
    report.update(
        recalled_after_three_neutral_events=answers,
        erased_memory_answer=reset_answer,
    )

    # Re-establish one actual cue. The environment has its own independent state.
    model.reset()
    cue = np.array([[[0.8, 0.0]]])
    model.advance(cue)
    _, actual_boundary = body(cue)
    before = model.snapshot()
    stored_memory = memory.snapshot()
    private = model.imagine(neutral)
    goal = np.zeros((1, 3, 1))
    plan = model.plan(
        neutral,
        goal=goal,
        controls=np.array([False, True]),
        bounds=(-1.0, 1.0),
        method="bfgs",
        beta=0.001,
        max_steps=20,
        goal_tolerance=1e-7,
        tolerance=1e-8,
    )
    private_unchanged = same_snapshot(before, model.snapshot())
    memory_unchanged = same_snapshot(stored_memory, memory.snapshot())
    if not private.converged or not plan.prediction.converged:
        raise RuntimeError("Private prediction did not converge")

    # Execute the proposed actions in the independent environment. Do not turn
    # the desired goal or imagined outputs into teaching observations.
    measured, _ = body(plan.inputs, actual_boundary)
    no_action, _ = body(neutral, actual_boundary)
    restored = TemporalPatchNet.restore(before)
    continuation = restored.advance(neutral)
    live_continuation = model.advance(neutral)
    report.update(
        private_queries_preserved_model=private_unchanged,
        private_queries_preserved_protection=memory_unchanged,
        restored_continuation_identical=bool(
            np.array_equal(continuation.output, live_continuation.output)
            and same_snapshot(restored.snapshot(), model.snapshot())
        ),
        predicted_initial_half_mse=plan.initial_cost,
        predicted_final_half_mse=plan.cost,
        planning_reason=plan.reason,
        planning_phase_calls=plan.phase_calls,
        planning_energy_evaluations=plan.energy_evaluations,
        planning_peak_message_bytes=plan.peak_message_bytes,
        planned_inputs=plan.inputs.tolist(),
        measured_outputs=measured.tolist(),
        measured_goal_mse=float(np.mean(np.square(measured - goal))),
        no_action_goal_mse=float(np.mean(np.square(no_action - goal))),
        live_parameter_updates=model.updates,
        limits=[
            "One scalar hidden quantity, two declared ports, deterministic synthetic body.",
            "Goal and protection are supplied; no learned importance or automatic goal formation.",
            "Finite protection covers selected boundaries/paths, not every possible cue.",
            "Private model forecasts are not measured outcomes or new learning evidence.",
            "This temporal-model example does not assert Brain world-model integration.",
        ],
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
