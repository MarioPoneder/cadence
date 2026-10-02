"""Bootstrap independent brains from a tiny simulated body, using CPU processes.

Run from the repository with Cadence installed::

    python examples/equilibrium/parallel_bootstrap.py --lives 4 --workers 2

Each worker owns its simulator and brain. Witnesses are measured positions after
commanded displacements, not rewards or a supplied action policy. Four separate
inputs check readiness without output clamps. The live phase predicts fresh
outcomes before admitting them as experience. This is a small dynamics lesson,
not a control task or evidence of a recursive-depth advantage.

Processes accelerate independent lives or architecture trials. They do not
combine checkpoints or concurrently update one brain. Small lessons may run
slower with workers because process startup costs more than the lesson itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product, repeat
from multiprocessing import get_context
from time import perf_counter

from cadence.experimental.equilibrium import Cortex, bootstrap

MAX_ERROR = 0.12


class PointBody:
    """A unit-mass point with commanded velocity and a fixed 0.25-second step."""

    def __init__(self, position):
        self.position = position

    def advance(self, velocity):
        """Apply the command and return the measured resulting position."""
        self.position += 0.25 * velocity
        return self.position


def experiences(positions, velocities):
    """Collect independent one-step witnesses from reset simulator states."""
    return [
        (
            {"body": [position, velocity]},
            {"next_position": [PointBody(position).advance(velocity)]},
        )
        for position, velocity in product(positions, velocities)
    ]


def run_life(seed, epochs):
    """Own one complete serial life; return readiness, work and fresh outcomes."""
    started = perf_counter()
    cortex = Cortex(seed=seed)
    body = cortex.input("body", shape=2)
    base = cortex.column("perception", patches=4, inputs=body)
    integration = cortex.column("integration", patches=2, inputs=(body, base))
    cortex.output("next_position", shape=1, reads=integration)
    brain = cortex.build()
    report = bootstrap(
        brain,
        experiences((-0.5, 0.5), (-0.6, 0.6)),
        checks=experiences((-0.25, 0.25), (-0.3, 0.3)),
        max_error=MAX_ERROR,
        epochs=epochs,
        seed=seed,
    )
    live = []
    if report["passed"]:
        # Fresh inputs, absent from both bootstrap examples and readiness checks.
        # The outcome reaches the brain only AFTER its unclamped prediction.
        for position, velocity in ((-0.35, 0.1), (0.35, -0.1), (0.05, -0.5)):
            inputs = {"body": [position, velocity]}
            query = brain.step(inputs)
            if not query["accepted"]:
                raise RuntimeError(f"Life seed {seed}: live query refused")
            predicted = query["outputs"]["next_position"][0]
            measured = PointBody(position).advance(velocity)
            admission = brain.observe(inputs, {"next_position": [measured]})
            if not admission["accepted"]:
                raise RuntimeError(f"Life seed {seed}: live witness refused")
            live.append({"predicted": predicted, "measured": measured})
    return {
        "seed": seed,
        "config": dict(brain.config),
        "bootstrap": report,
        "live": live,
        "admissions": brain.inspect()["admissions"],
        "checkpoint_sha256": hashlib.sha256(brain.snapshot().encode()).hexdigest(),
        "process_id": os.getpid(),
        "elapsed_seconds": perf_counter() - started,
    }


def bounded(value, name, low, high):
    """Reject invalid workload sizes before creating a worker pool."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def run_lives(seeds, *, workers=1, epochs=20):
    """Return all lives in seed order; worker exceptions propagate to the caller.

    The spawn context is local to this executor; no process-wide start method
    or numerical-library thread settings are changed.
    """
    seeds = tuple(seeds)
    bounded(len(seeds), "lives", 1, 64)
    bounded(workers, "workers", 1, 64)
    bounded(epochs, "epochs", 0, 100)
    if workers == 1:
        return [run_life(seed, epochs) for seed in seeds]
    with ProcessPoolExecutor(
        max_workers=min(workers, len(seeds)), mp_context=get_context("spawn")
    ) as pool:
        # Consuming map propagates exceptions; a missing life cannot look passed.
        return list(pool.map(run_life, seeds, repeat(epochs)))


def main(argv=None):
    """Print a complete JSON result; unsuccessful readiness exits with status 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lives", type=int, default=4, help="independent brains (1–64)")
    parser.add_argument("--workers", type=int, default=1, help="CPU processes (1–64)")
    parser.add_argument("--epochs", type=int, default=20, help="replay limit (0–100)")
    parser.add_argument("--seed", type=int, default=0, help="first nonnegative seed")
    args = parser.parse_args(argv)
    try:
        bounded(args.lives, "lives", 1, 64)
        if args.seed < 0:
            raise ValueError("seed must be nonnegative")
        started = perf_counter()
        lives = run_lives(
            range(args.seed, args.seed + args.lives),
            workers=args.workers,
            epochs=args.epochs,
        )
    except ValueError as error:
        parser.error(str(error))
    passed = all(life["bootstrap"]["passed"] for life in lives)
    print(
        json.dumps(
            {
                "passed": passed,
                "workers": min(args.workers, args.lives),
                "elapsed_seconds": perf_counter() - started,
                "lives": lives,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
