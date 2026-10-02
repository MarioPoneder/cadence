"""Bounded temporal-context and delayed-credit qualification on current sources.

Run: python examples/equilibrium/temporal_credit.py --out /tmp/temporal-credit.json
History is supplied external memory. The reward fixture supplies a one-hot
stage/previous-action observation, not a desired action or future reward.
The two fixtures isolate memory from credit; they are not an integrated agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cadence.experimental.equilibrium import (  # noqa: E402
    Brain,
    Cortex,
    History,
    Reinforcement,
    bootstrap,
)


class Work:
    def __init__(self):
        self.totals = Counter()
        self.calls = 0
        self.refusals = 0

    def add(self, result):
        self.calls += 1
        self.totals.update(result.get("work", {}))
        if result.get("reason") != "learning_disabled":
            self.refusals += not result.get(
                "accepted", result.get("qualified", result.get("passed", False))
            )
        return result

    def report(self):
        return dict(calls=self.calls, refusals=self.refusals, solver=dict(self.totals))


def layout(size, seed, architecture, outputs):
    cortex = Cortex(seed=seed, parameter_prior=0.02, tolerance=1e-5)
    senses = cortex.input("senses", shape=size)
    base = cortex.column(patches=2, inputs=senses)
    if architecture == "ordinary":
        values = cortex.column(patches=2, inputs=(senses, base))
    elif architecture == "observer":
        values = cortex.observer(patches=2, inputs=senses, observes=base)
    else:
        raise ValueError("Unknown architecture")
    for index, name in enumerate(outputs):
        cortex.output(name, shape=(), reads=values, indices=(index,))
    return cortex.build()


def cue_sequence(cue, delay, suffix_seed):
    rng = random.Random(suffix_seed)
    # Distractor channel is independent of the cue, including for paired signs.
    return [[cue, 0.0]] + [[0.0, rng.uniform(-0.7, 0.7)] for _ in range(delay)]


def encode(sequence, *, memoryless=False):
    history = History(2, steps=len(sequence))
    for sample in sequence:
        if memoryless:
            history.reset()
        context = history.push(sample)
    return {"senses": context}


def memory_case(seed, delay, architecture):
    work = Work()
    brain = layout(3 * (delay + 1), seed, architecture, ("cue",))
    training = [
        (encode(cue_sequence(cue, delay, suffix)), {"cue": cue})
        for suffix in (10, 11)
        for cue in (-0.8, 0.8)
    ]
    development = [(encode(cue_sequence(cue, delay, 20)), {"cue": cue}) for cue in (-0.4, 0.4)]
    trained = work.add(
        bootstrap(
            brain,
            training,
            checks=development,
            max_error=0.15,
            epochs=40,
            batch_size=4,
        )
    )
    snapshot = brain.snapshot()
    rows = []
    for suffix in (100, 101):
        for cue in (-0.6, -0.3, 0.3, 0.6):
            sequence = cue_sequence(cue, delay, suffix)
            for control in (
                "history_retained_state",
                "history_reset_state",
                "memoryless_retained_state",
                "memoryless_reset_state",
            ):
                learner = Brain.from_snapshot(snapshot)
                history = History(2, steps=delay + 1)
                continuation = None
                identical = True
                for tick, sample in enumerate(sequence):
                    if "memoryless" in control:
                        history.reset()
                    context = {"senses": history.push(sample)}
                    if "reset_state" in control:
                        learner = Brain.from_snapshot(snapshot)
                    result = work.add(learner.step(context))
                    if not result["accepted"]:
                        break
                    if continuation is not None:
                        twin_brain, twin_history = continuation
                        if "memoryless" in control:
                            twin_history.reset()
                        if "reset_state" in control:
                            twin_brain = Brain.from_snapshot(snapshot)
                        twin = work.add(twin_brain.step({"senses": twin_history.push(sample)}))
                        identical &= result == twin
                        identical &= learner.snapshot() == twin_brain.snapshot()
                        identical &= history.snapshot() == twin_history.snapshot()
                        continuation = twin_brain, twin_history
                    if tick == delay // 2:
                        continuation = (
                            Brain.from_snapshot(learner.snapshot()),
                            History.from_snapshot(history.snapshot()),
                        )
                prediction = result["outputs"]["cue"][0] if result["accepted"] else None
                rows.append(
                    dict(
                        cue=cue,
                        suffix=suffix,
                        control=control,
                        prediction=prediction,
                        resumed_equal=identical,
                    )
                )
    measured = [r for r in rows if r["control"].startswith("history_")]
    passed = (
        trained["passed"]
        and work.refusals == 0
        and all(r["resumed_equal"] for r in rows)
        and all(abs(r["prediction"] - r["cue"]) < 0.2 for r in measured)
    )
    return dict(
        kind="memory",
        seed=seed,
        delay=delay,
        architecture=architecture,
        passed=passed,
        bootstrap=trained,
        rows=rows,
        work=work.report(),
        topology=brain.inspect(),
    )


def context(stage, chosen, delay):
    values = [0.0] * (1 + 2 * delay)
    values[0 if stage == 0 else 1 + 2 * (stage - 1) + chosen] = 1.0
    return {"senses": values}


def credit_case(seed, delay, mode, *, episodes=60, preferred=1):
    if type(episodes) is not int or not 1 <= episodes <= 100:
        raise ValueError("episodes must be an integer in [1,100]")
    if delay not in (0, 2, 4, 8) or type(delay) is not int:
        raise ValueError("delay must be 0, 2, 4 or 8")
    if mode not in ("td_replay", "no_bootstrap", "frozen"):
        raise ValueError("Unknown credit mode")
    work = Work()
    discount = 0.0 if mode == "no_bootstrap" else 0.8
    learner = Reinforcement(
        layout(1 + 2 * delay, seed, "ordinary", ("q0", "q1")),
        actions=2,
        action_input=None,
        value_output=("q0", "q1"),
        discount=discount,
        exploration=0.4,
        capacity=256,
        batch_size=8,
        seed=seed,
    )
    if type(preferred) is not int or preferred not in (0, 1):
        raise ValueError("preferred must be 0 or 1")
    outcomes = []
    resumed_equal = True
    for episode in range(episodes):
        chosen = 0
        twin = None
        executed = []
        for stage in range(delay + 1):
            observation = context(stage, chosen, delay)
            decision = work.add(learner.act(observation))
            if not decision["accepted"]:
                return dict(
                    kind="credit",
                    seed=seed,
                    delay=delay,
                    mode=mode,
                    passed=False,
                    outcomes=outcomes,
                    work=work.report(),
                )
            if stage == 0:
                chosen = decision["action"]
            executed.append(decision["action"])
            # A single mid-episode save includes an outstanding executed action.
            if episode == episodes // 2 and stage == delay // 2:
                twin = Reinforcement.from_snapshot(learner.snapshot())
            elif twin is not None:
                resumed_equal &= decision == work.add(twin.act(observation))
            terminal = stage == delay
            reward = (1 if chosen == preferred else -1) if terminal else 0
            following = None if terminal else context(stage + 1, chosen, delay)
            update = work.add(
                learner.feedback(
                    reward,
                    following,
                    terminal=terminal,
                    learn=mode != "frozen",
                    decision_id=decision["decision_id"],
                    executed_action=decision["action"],
                )
            )
            if twin is not None:
                resumed_equal &= update == work.add(
                    twin.feedback(
                        reward,
                        following,
                        terminal=terminal,
                        learn=mode != "frozen",
                        decision_id=decision["decision_id"],
                        executed_action=decision["action"],
                    )
                )
                resumed_equal &= learner.snapshot() == twin.snapshot()
        outcomes.append(dict(actions=executed, reward=reward))
    # Execute complete frozen-evaluation episodes, including ordinary exploration.
    evaluation = []
    values = None
    for _ in range(40):
        chosen = 0
        for stage in range(delay + 1):
            decision = work.add(learner.act(context(stage, chosen, delay)))
            if not decision["accepted"]:
                return dict(
                    kind="credit",
                    seed=seed,
                    delay=delay,
                    mode=mode,
                    passed=False,
                    outcomes=outcomes,
                    work=work.report(),
                )
            if stage == 0:
                chosen = decision["action"]
                values = decision["values"]
            terminal = stage == delay
            reward = (1 if chosen == preferred else -1) if terminal else 0
            work.add(
                learner.feedback(
                    reward,
                    None if terminal else context(stage + 1, chosen, delay),
                    decision_id=decision["decision_id"],
                    executed_action=decision["action"],
                    terminal=terminal,
                    learn=False,
                )
            )
        evaluation.append(dict(chosen=chosen, reward=reward))
    success = sum(r["reward"] > 0 for r in evaluation) / len(evaluation)
    ideal_magnitude = (1 - discount) * 0.9 * discount**delay
    return dict(
        kind="credit",
        seed=seed,
        delay=delay,
        mode=mode,
        passed=work.refusals == 0 and resumed_equal and (mode != "td_replay" or success >= 0.65),
        preferred=preferred,
        outcomes=outcomes,
        evaluation=evaluation,
        success=success,
        values=values,
        ideal_value_magnitude=ideal_magnitude,
        resumed_equal=resumed_equal,
        settings=dict(learner.config),
        experience=learner.inspect(),
        work=work.report(),
    )


def run(seeds):
    started, cpu = time.perf_counter(), time.process_time()
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        *sorted((root / "src/cadence/experimental/equilibrium").glob("*.py")),
    ]
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    rows = []
    for seed in seeds:
        for delay in (2, 4, 8):
            for architecture in ("ordinary", "observer"):
                row = memory_case(seed, delay, architecture)
                rows.append(row)
                print(
                    f"memory seed={seed} delay={delay} {architecture}: {row['passed']}",
                    file=sys.stderr,
                    flush=True,
                )
        for delay in (0, 2, 4, 8):
            for preferred in (0, 1):
                for mode in ("td_replay", "no_bootstrap", "frozen"):
                    row = credit_case(seed, delay, mode, preferred=preferred)
                    rows.append(row)
                    print(
                        f"credit seed={seed} delay={delay} target={preferred} {mode}: "
                        f"{row['passed']} success={row.get('success')}",
                        file=sys.stderr,
                        flush=True,
                    )
    return dict(
        schema="temporal-credit/1",
        seeds=seeds,
        rows=rows,
        passed=all(row["passed"] for row in rows),
        sources=sources,
        sources_unchanged=all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in sources.items()
        ),
        seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu,
        scope="External history and discrete one-step TD replay, not recurrent memory",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 7])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seeds) > 8 or any(seed < 0 for seed in args.seeds):
        parser.error("Supply one to eight nonnegative seeds")
    report = run(args.seeds)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)
