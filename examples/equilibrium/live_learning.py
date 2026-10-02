"""Small, deterministic memory/reward/retention gates; no full-pet claim.

Run from a checkout: python examples/equilibrium/live_learning.py --seeds 0 2 7
The reward task has two actual action ticks per episode. Reward at the second
tick depends on the first action, which is explicitly included in the next
sensory context. No desired action is supplied as a learning target.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cadence.experimental.equilibrium import (  # noqa: E402
    Brain,
    Cortex,
    History,
    Reinforcement,
    bootstrap,
)


def hidden_cue(seed):
    """Recall an occluded cue from explicit history, then remove that history."""
    history = History(1, steps=3)
    layout = Cortex(seed=seed)
    sensor = layout.input("history", shape=history.shape)
    base = layout.column(patches=4, inputs=sensor)
    integration = layout.column("integration", patches=2, inputs=base)
    layout.output("location", shape=1, reads=integration)
    brain = layout.build()

    def case(value):
        history.reset()
        history.push([value])
        history.push([0])
        return {"history": history.push([0])}, {"location": [value]}

    report = bootstrap(
        brain,
        [case(-0.8), case(0.8)],
        checks=[case(-0.4), case(0.4)],
        max_error=0.15,
        epochs=40,
        batch_size=2,
    )
    fresh = [brain.predict(case(x)[0])["location"][0] for x in (-0.3, 0.3)]
    ablated = brain.predict(case(0)[0])["location"][0]
    restored = Brain.from_snapshot(brain.snapshot())
    assert restored.predict(case(0.3)[0]) == brain.predict(case(0.3)[0])
    return dict(
        passed=report["passed"] and fresh[0] < -0.15 and fresh[1] > 0.15 and abs(ablated) < 0.1,
        fresh_predictions=fresh,
        erased_history_prediction=ablated,
        patches=6,
        history_steps=3,
    )


def retention(seed):
    """Acquire a second independent relation while replaying the first."""
    layout = Cortex(seed=seed)
    sensor = layout.input("senses", shape=2)
    base = layout.column(patches=4, inputs=sensor)
    integration = layout.column("integration", patches=2, inputs=base)
    layout.output("answer", shape=1, reads=integration)
    brain = layout.build()

    def cases(axis, amplitude):
        return [
            ({"senses": [x if i == axis else 0 for i in range(2)]}, {"answer": [x]})
            for x in (-amplitude, amplitude)
        ]

    old, new = cases(0, 0.8), cases(1, 0.8)
    checks = cases(0, 0.4) + cases(1, 0.4)
    first = bootstrap(brain, old, checks=checks[:2], max_error=0.15, batch_size=2, epochs=40)
    second = bootstrap(brain, old + new, checks=checks, max_error=0.15, batch_size=4, epochs=40)
    errors = [abs(brain.predict(x)["answer"][0] - y["answer"][0]) for x, y in checks]
    restored = Brain.from_snapshot(brain.snapshot())
    x, y = cases(1, 0.6)[1]
    assert restored.observe(x, y)["accepted"]
    assert brain.observe(x, y)["accepted"]
    return dict(
        passed=first["passed"]
        and second["passed"]
        and max(errors) < 0.15
        and restored.snapshot() == brain.snapshot(),
        old_errors=errors[:2],
        new_errors=errors[2:],
        patches=6,
    )


def reward_brain(seed):
    layout = Cortex(seed=seed, initial_scale=1.5)
    senses = layout.input("senses", shape=3)
    action = layout.input("action", shape=2)
    base = layout.column(patches=6, inputs=(senses, action))
    integration = layout.column("integration", patches=3, inputs=base)
    layout.output("value", shape=(), reads=integration)
    return Reinforcement(
        layout.build(),
        actions=2,
        discount=0.5,
        exploration=0.4,
        batch_size=8,
        capacity=32,
        seed=seed,
    )


def delayed_reward(seed, *, episodes=50):
    """Acquire and reverse a two-step choice under reward-only feedback."""
    learner = reward_brain(seed)
    start = {"senses": [1, 0, 0]}
    frozen = Reinforcement.from_snapshot(learner.snapshot())
    before = frozen.act(start, explore=False)
    frozen.reset()
    outcomes = []
    for preferred in (1, 0):
        for _ in range(episodes):
            first = learner.act(start)
            if not first["accepted"]:
                return dict(passed=False, refusal="first_action")
            action = first["action"]
            later = {"senses": [0, float(action == 0), float(action == 1)]}
            update = learner.feedback(
                0,
                later,
                decision_id=first["decision_id"],
                executed_action=action,
            )
            if not update["accepted"]:
                return dict(passed=False, refusal=update.get("reason"))
            second = learner.act(later)
            if not second["accepted"]:
                return dict(passed=False, refusal="second_action")
            update = learner.feedback(
                1 if action == preferred else -1,
                terminal=True,
                decision_id=second["decision_id"],
                executed_action=second["action"],
            )
            if not update["accepted"]:
                return dict(passed=False, refusal=update.get("reason"))
        decision = learner.act(start, explore=False)
        outcomes.append(
            dict(
                preferred=preferred,
                chosen=decision["action"],
                values=decision["values"],
            )
        )
        learner.reset()
    restored = Reinforcement.from_snapshot(learner.snapshot())
    same = learner.act(start) == restored.act(start)
    return dict(
        passed=all(r["preferred"] == r["chosen"] for r in outcomes) and same,
        phases=outcomes,
        frozen_choice=before["action"],
        frozen_phases_correct=sum(before["action"] == target for target in (1, 0)),
        episodes_per_phase=episodes,
        patches=9,
        transitions=learner.inspect()["transitions"],
        settings=dict(learner.config),
    )


def run(seeds):
    rows = []
    for seed in seeds:
        for name, gate in (
            ("hidden_cue", hidden_cue),
            ("retention", retention),
            ("delayed_reward_and_reversal", delayed_reward),
        ):
            started = time.perf_counter()
            result = dict(
                seed=seed,
                gate=name,
                **gate(seed),
                seconds=time.perf_counter() - started,
            )
            rows.append(result)
            print(json.dumps(result), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 7])
    arguments = parser.parse_args()
    results = run(arguments.seeds)
    raise SystemExit(0 if all(row["passed"] for row in results) else 1)
