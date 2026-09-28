"""Compare complete batch bootstrapping on the same small supervised task.

From the repository with Cadence installed::

    python examples/batch_bootstrap.py --devices python cpu mps --batch-size 8

Optional tensor devices require the ``gpu`` extra and working hardware. This
is a small continuous relation, not a game benchmark or a depth comparison.
All backends receive identical examples, starting seeds and readiness criteria.
Startup/warmup, bootstrap and fresh-test times are recorded separately. Changing
batch size changes the learning objective per update; it is not a pure execution
speed comparison. Missing hardware fails explicitly rather than being omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
from time import perf_counter

from cadence import Cortex, bootstrap


def examples(count, seed):
    """Measure two simple relations from independently varied sensor samples."""
    rng = random.Random(seed)
    records = []
    for _ in range(count):
        x = [rng.uniform(-0.8, 0.8) for _ in range(8)]
        y = [0.4 * x[0] + 0.2 * x[1], 0.3 * x[2] - 0.2 * x[3]]
        records.append(({"senses": x}, {"answer": y}))
    return records


def run(device, batch_size, seed, epochs):
    """Own one brain; record complete qualification, learning and timing."""
    started = perf_counter()
    cortex = Cortex(seed=seed, device=device)
    senses = cortex.input("senses", shape=8)
    base = cortex.column("perception", patches=12, inputs=senses)
    observer = cortex.observer("reflection", patches=4, inputs=senses, observes=base)
    cortex.output("answer", shape=2, reads=observer)
    brain = cortex.build()
    teaching, checks, tests = examples(32, 19), examples(8, 20), examples(16, 21)
    warmup = brain.settle(teaching[0][0])
    startup_seconds = perf_counter() - started
    started = perf_counter()
    report = bootstrap(
        brain,
        teaching,
        checks=checks,
        max_error=0.08,
        epochs=epochs,
        seed=31,
        batch_size=batch_size,
    )
    bootstrap_seconds = perf_counter() - started
    started = perf_counter()
    fresh = [(brain.settle(inputs), target["answer"]) for inputs, target in tests]
    test_seconds = perf_counter() - started
    qualified = all(result["qualified"] for result, _ in fresh)
    error = max(
        abs(value - target)
        for result, targets in fresh
        for value, target in zip(result["outputs"]["answer"], targets, strict=True)
    )
    return {
        "seed": seed,
        "device": device,
        "config": dict(brain.config),
        "implementation": brain.inspect()["implementation"],
        "data_sha256": hashlib.sha256(
            json.dumps([teaching, checks, tests], sort_keys=True).encode()
        ).hexdigest(),
        "startup_seconds": startup_seconds,
        "bootstrap_seconds": bootstrap_seconds,
        "test_seconds": test_seconds,
        "warmup_qualified": warmup["qualified"],
        "bootstrap": report,
        "test_qualified": qualified,
        "test_max_error": error,
        "passed": warmup["qualified"]
        and report["passed"]
        and qualified
        and error <= 0.08,
        "execution": warmup.get("execution", {"device": "python", "dtype": "float64"}),
    }


def positive(value):
    number = int(value)
    if not 1 <= number <= 128:
        raise argparse.ArgumentTypeError("Use an integer in [1, 128]")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["python"])
    parser.add_argument("--batch-size", type=positive, default=8)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--epochs", type=positive, default=20)
    parser.add_argument(
        "--threads",
        type=positive,
        default=1,
        help="PyTorch CPU threads per process (set before tensor work)",
    )
    args = parser.parse_args()
    if any(device != "python" for device in args.devices):
        import torch

        torch.set_num_threads(args.threads)
    rows = [
        run(device, args.batch_size, seed, args.epochs)
        for seed in range(args.repeats)
        for device in args.devices
    ]
    medians = {
        device: statistics.median(
            row["bootstrap_seconds"] for row in rows if row["device"] == device
        )
        for device in args.devices
    }
    passed = all(row["passed"] for row in rows)
    print(
        json.dumps(
            {
                "passed": passed,
                "python": platform.python_version(),
                "machine": platform.machine(),
                "platform": platform.platform(),
                "options": vars(args),
                "median_bootstrap_seconds": medians,
                "runs": rows,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
