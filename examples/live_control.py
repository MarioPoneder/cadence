"""Run a learned one-dimensional body model behind a nonblocking controller.

    python examples/live_control.py

The supplied body has position, a velocity port and measured transition records.
Cadence's observer reads patch state and prediction error in the same equilibrium;
actual body records repair that model through ordinary ``observe``. The action
comparison and distance/effort reward are supplied application rules outside the
brain. This is model-based control, not reinforcement learning or a full pet.
Learning progress is recorded from actual prediction error; it does not select
actions or supply temporal credit. These hand-set rules are controls/candidate
genes, not evidence of evolution or a recursive-depth advantage.

The simulation waits for each command while polling without blocking on a solve.
Reported submission-to-command latency includes that polling, but is a small
machine-dependent observation, not a hard real-time guarantee or a benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cadence import (  # noqa: E402
    Cortex,
    LearningProgress,
    LiveController,
    bootstrap,
    slew,
)

DT = 0.25
ACTIONS = (-0.8, 0.0, 0.8)


class PointBody:
    """Supplied point kinematics: x' = x + dt * commanded velocity."""

    def __init__(self, position):
        self.position = position

    def advance(self, velocity):
        self.position += DT * velocity
        return self.position


def body_reward(before, after, action):
    """Measured reduction in squared distance to zero, minus actuator effort."""
    return before**2 - after**2 - 0.01 * action**2 * DT


def experiences(positions, actions):
    """Reset the simulator to collect real one-step teaching witnesses."""
    return [
        ({"body": [x, u]}, {"next_position": [PointBody(x).advance(u)]})
        for x, u in product(positions, actions)
    ]


def prepare(seed=0):
    """Teach consequences, checking subsequent unclamped predictions."""
    cortex = Cortex(seed=seed)
    senses = cortex.input("body", shape=2)
    base = cortex.column("perception", patches=4, inputs=senses)
    observer = cortex.observer("reflection", patches=2, inputs=senses, observes=base)
    cortex.output("next_position", shape=1, reads=observer)
    brain = cortex.build()
    report = bootstrap(
        brain,
        experiences((-0.5, 0.5), (-0.8, 0.8)),
        checks=experiences((-0.25, 0.25), (-0.4, 0.4)),
        max_error=0.06,
        epochs=30,
        seed=seed,
    )
    return brain, report


class ModelController:
    """One owner predicts, admits measured transitions and compares actions."""

    def __init__(self, brain):
        self.brain, self.progress = brain, LearningProgress(rate=0.25)
        self.transitions = []

    def admit(self, transition):
        before, action, measured = transition
        inputs = {"body": [before, action]}
        predicted = self.brain.predict(inputs)["next_position"][0]
        error = abs(predicted - measured)  # Outcome error, never stationarity.
        progress = self.progress.update(f"velocity:{action:.2f}", error)
        admitted = self.brain.observe(inputs, {"next_position": [measured]})
        self.transitions.append(
            dict(
                before=before,
                action=action,
                measured=measured,
                predicted=predicted,
                prediction_error=error,
                body_reward=body_reward(before, measured, action),
                learning_progress=progress,
                accepted=admitted["accepted"],
            )
        )
        if not admitted["accepted"]:
            raise RuntimeError("Actual transition witness was refused")

    def __call__(self, observation):
        if observation["transition"] is not None:
            self.admit(observation["transition"])
        position, velocity = observation["position"], observation["velocity"]
        candidates = dict.fromkeys(
            slew([velocity], [target], rate=2, dt=DT)[0] for target in ACTIONS
        )
        scores = []
        for action in candidates:
            result = self.brain.settle({"body": [position, action]})
            if not result["qualified"]:
                return {"qualified": False}
            predicted = result["outputs"]["next_position"][0]
            scores.append((body_reward(position, predicted, action), action))
        return {"qualified": True, "command": [max(scores)[1]]}


def run(*, decisions=20, seed=0, deadline=1.0):
    """Return acquisition, body, progress and observed latency diagnostics."""
    if type(decisions) is not int or not 1 <= decisions <= 100:
        raise ValueError("decisions must be an integer in [1, 100]")
    if isinstance(deadline, bool) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("deadline must be positive and finite")
    brain, teaching = prepare(seed)
    if not teaching["passed"]:
        return {"passed": False, "bootstrap": teaching}
    model, body = ModelController(brain), PointBody(0.55)
    controller = LiveController(model, fallback=[0], max_age=deadline)
    velocity, transition, timings, statuses = 0.0, None, [], []
    try:
        for _ in range(decisions):
            started = time.monotonic()
            identity = controller.submit(
                dict(position=body.position, velocity=velocity, transition=transition)
            )
            while controller.inspect()["last_result_id"] != identity:
                if time.monotonic() - started > 10:
                    raise RuntimeError("Controller did not complete within 10 seconds")
                controller.read()  # A renderer can use this fallback/prior command.
                time.sleep(0.001)
            target = controller.read()
            timings.append(time.monotonic() - started)
            statuses.append(controller.inspect()["status"])
            velocity = slew([velocity], target, rate=2, dt=DT)[0]
            before = body.position
            transition = before, velocity, body.advance(velocity)
    finally:
        if not controller.close(timeout=1):
            raise RuntimeError("Worker still owns its brain after bounded close")
    model.admit(transition)  # Final actual witness, after ownership returns.
    final_need = body.position**2
    return {
        "passed": all(s == "qualified" for s in statuses) and final_need < 0.55**2,
        "bootstrap": teaching,
        "initial_need": 0.55**2,
        "final_need": final_need,
        "final_position": body.position,
        "transitions": model.transitions,
        "runtime": controller.inspect(),
        "deadline_seconds": deadline,
        "submission_to_command_seconds": timings,
        "deadline_misses": sum(value > deadline for value in timings),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run(decisions=args.decisions, seed=args.seed)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
