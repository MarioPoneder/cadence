"""Bounded query-cost probe for equally wide flat, composed and observing layouts.

    python examples/layout_cost.py --out /tmp/layout-cost.json

This is an untrained, numerical cost probe, not a learning or intelligence test.
All arms receive four scalar sensors and expose two settled patch coordinates.
Flat and recursive arms have six patches, 24 edges and 30 parameters. Ordinary
composition has 16 edges; a separate raw-input-skip control restores 24 edges.
Graph geometry, conditioning, output-connected capacity and random coefficients
remain different. One seed does not give different topologies identical weights.

Only pure ``settle`` queries run, each from unchanged zero activity and fixed
parameters. No witness, warm-up query, tuning, learning, GPU or cloud is used.
The input-only arm additionally checks the independent exact optimum
clip(tanh(b + W input) / (1 + state_prior)). That calculation is timed separately;
it does not replace a library call or propose a second learning rule.

The output starts with a source-bound protocol BEFORE measurement and is then
replaced by the final receipt. Existing output files are refused. A 45-second
scheduling limit preserves every completed outcome and every missing case;
one already-running 512-sweep solve can overrun that scheduling limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def source_hashes():
    paths = [Path(__file__).resolve(), *sorted((ROOT / "src/cadence").glob("*.py"))]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


SEEDS = (2, 7, 17)
ARMS = ("flat", "composed", "composed_skip", "recursive")
INPUTS = tuple(
    tuple(round(0.75 * math.sin((i + 1) * (j + 1) * 0.47), 8) for j in range(4))
    for i in range(12)
)
SETTINGS = dict(
    fan_in=None,
    initial_scale=0.3,
    state_prior=0.01,
    parameter_prior=0.1,
    tolerance=1e-6,
    settle_budget=512,
    device="python",
    dtype="float64",
)


def build(Cortex, arm, seed):
    cortex = Cortex(seed=seed, **SETTINGS)
    sensor = cortex.input("sensor", shape=4)
    if arm == "flat":
        output = cortex.column("flat", patches=6, inputs=sensor)
        indices = (4, 5)
    else:
        base = cortex.column("base", patches=2, inputs=sensor)
        if arm == "recursive":
            middle = cortex.observer("middle", patches=2, observes=base)
            output = cortex.observer("output", patches=2, observes=middle)
        else:
            middle = cortex.column("middle", patches=2, inputs=base)
            inputs = (middle, sensor) if arm == "composed_skip" else middle
            output = cortex.column("output", patches=2, inputs=inputs)
        indices = (0, 1)
    cortex.output("answer", shape=2, reads=output, indices=indices)
    return cortex.build()


def exact_flat(brain, sensor):
    """Independent closed form for fixed-parameter, input-only query energy."""
    drive = list(brain.biases)
    for (kind, source, target), weight in zip(
        brain.graph.edges, brain.weights, strict=True
    ):
        if kind != "input":
            raise ValueError("The flat closed form requires input-only connections")
        drive[target] += weight * sensor[source]
    predictions = [math.tanh(value) for value in drive]
    prior, bound = brain.config["state_prior"], brain.config["state_bound"]
    states = [max(-bound, min(bound, p / (1 + prior))) for p in predictions]
    energy = sum(
        0.5 * (x - p) ** 2 + 0.5 * prior * x * x
        for x, p in zip(states, predictions, strict=True)
    )
    return states, energy


def write(path, report, *, create=False):
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if create:
        with path.open("x") as handle:
            handle.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def summarize(rows):
    summary = {}
    for arm in ARMS:
        cases = [row for row in rows if row["arm"] == arm]
        measured = [r for r in cases if "seconds" in r]
        timings = sorted(r["seconds"] for r in measured)
        work = Counter()
        for row in measured:
            work.update(row.get("work", {}))
        summary[arm] = {
            "scheduled": len(SEEDS) * len(INPUTS),
            "returned": len(cases),
            "qualified": sum(r.get("qualified", False) for r in cases),
            "errors": sum("exception" in r for r in cases),
            "median_query_ms": 1000 * statistics.median(timings) if timings else None,
            "p95_query_ms": 1000 * timings[math.ceil(0.95 * len(timings)) - 1]
            if timings
            else None,
            "max_query_ms": 1000 * max(timings) if timings else None,
            "query_seconds": sum(timings),
            "median_sweeps": statistics.median(r["sweeps"] for r in measured)
            if measured
            else None,
            "work": dict(work),
        }
    return summary


def run(output):
    before_import = source_hashes()
    import cadence
    from cadence import Cortex

    sources = source_hashes()
    if sources != before_import:
        raise RuntimeError("Source changed during import; no probe started")
    schedule = [
        {
            "seed": seed,
            "input_index": i,
            "arm": ARMS[(slot + seed_index + i) % len(ARMS)],
        }
        for seed_index, seed in enumerate(SEEDS)
        for i in range(len(INPUTS))
        for slot in range(len(ARMS))
    ]
    protocol = {
        "schema": "cadence-layout-cost-protocol/1",
        "scope": "Untrained fixed-parameter query cost only; no learned task or intelligence comparison",
        "seeds": SEEDS,
        "inputs": INPUTS,
        "settings": SETTINGS,
        "arms": {
            "flat": "6 input-only patches; 24 edges, 30 parameters; last two patches are outputs",
            "composed": "2+2+2 state-reading patches; 16 edges, 22 parameters",
            "composed_skip": "2+2+2 state-reading patches, raw-sensor skip to final population; 24 edges, 30 parameters",
            "recursive": "2+2+2 patches, later populations read preceding live states and exact errors; 24 edges, 30 parameters",
        },
        "schedule": schedule,
        "schedule_rule": "Rotate arm order by model-seed index and input index; no timing-based ordering",
        "wall_scheduling_budget_seconds": 45,
        "timing": "Each complete public settle call only; graph construction, snapshot checks and independent flat calculations are separately timed or outside query timings. Imports excluded. Shared host may have concurrent work.",
        "continuation": "Pure queries from unchanged zero activity, fixed parameters, same input sequence; no learning or retained-state warm start",
        "flat_reference": "All patch coordinates: x=clip(tanh(b+W input)/(1+state_prior)); qualified-state tolerance 2e-6, energy tolerance 1e-10",
        "source_sha256": sources,
        "nonclaims": [
            "No useful-recursion, quality, task accuracy, learning-throughput or physical-energy claim",
            "Matching patch/edge counts does not match graph geometry, conditioning or output-connected capacity",
            "The skip control has a direct sensory route absent from the recursive layout",
            "Random seeds fix construction but do not align coefficients across different topologies",
            "The flat exact formula applies only to fixed-parameter, input-only queries",
        ],
    }
    report = {
        "schema": "cadence-layout-cost/1",
        "status": "protocol_frozen_before_measurement",
        "protocol": protocol,
        "protocol_sha256": digest(protocol),
        "runtime": {
            "cadence_version": cadence.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "models": [],
        "rows": [],
    }
    write(output, report, create=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    models = {}
    for seed in SEEDS:
        for arm in ARMS:
            if time.perf_counter() - started >= 45:
                break
            then = time.perf_counter()
            brain = build(Cortex, arm, seed)
            construction = time.perf_counter() - then
            snapshot = brain.snapshot()
            info = brain.inspect()
            models[(seed, arm)] = brain, snapshot
            report["models"].append(
                {
                    "seed": seed,
                    "arm": arm,
                    "construction_seconds": construction,
                    "patches": info["patches"],
                    "connections": info["connections"],
                    "parameters": len(brain.weights) + len(brain.biases),
                    "output_connected_patches": info["output_connected_patches"],
                    "fingerprint": info["fingerprint"],
                    "implementation": info["implementation"],
                    "config": info["config"],
                    "edges": info["edges"],
                    "weights": brain.weights,
                    "biases": brain.biases,
                    "initial_snapshot_sha256": hashlib.sha256(
                        snapshot.encode()
                    ).hexdigest(),
                }
            )
    for case in schedule:
        if (
            time.perf_counter() - started >= 45
            or (case["seed"], case["arm"]) not in models
        ):
            break
        brain, _ = models[(case["seed"], case["arm"])]
        inputs = INPUTS[case["input_index"]]
        row = dict(case)
        try:
            then = time.perf_counter()
            result = brain.settle({"sensor": inputs})
            row.update(
                seconds=time.perf_counter() - then,
                **{
                    key: result[key]
                    for key in (
                        "qualified",
                        "reason",
                        "sweeps",
                        "stationarity",
                        "energy",
                        "state",
                        "outputs",
                        "work",
                    )
                },
            )
            if case["arm"] == "flat":
                then = time.perf_counter()
                exact, energy = exact_flat(brain, inputs)
                row["exact_flat"] = {
                    "seconds": time.perf_counter() - then,
                    "state": exact,
                    "energy": energy,
                    "max_state_difference": max(
                        abs(x - y) for x, y in zip(exact, result["state"], strict=True)
                    ),
                    "energy_difference": abs(energy - result["energy"]),
                }
        except Exception as exc:
            row["exception"] = f"{type(exc).__name__}: {exc}"
        report["rows"].append(row)
    after = source_hashes()
    unchanged = all(brain.snapshot() == snapshot for brain, snapshot in models.values())
    complete = len(report["rows"]) == len(schedule)
    flat_checks = [r["exact_flat"] for r in report["rows"] if "exact_flat" in r]
    flat_agreement = len(flat_checks) == 36 and all(
        r["max_state_difference"] <= 2e-6 and r["energy_difference"] <= 1e-10
        for r in flat_checks
    )
    report.update(
        status="complete" if complete else "budget_stopped",
        wall_seconds=time.perf_counter() - started,
        cpu_seconds=time.process_time() - cpu_started,
        source_sha256_after=after,
        sources_unchanged=sources == after,
        continuations_unchanged=unchanged,
        all_scheduled_complete=complete,
        missing_schedule=schedule[len(report["rows"]) :],
        flat_exact_agreement=flat_agreement,
        summary=summarize(report["rows"]),
        passed=complete
        and unchanged
        and sources == after
        and flat_agreement
        and all(
            r.get("qualified", False) and "exception" not in r for r in report["rows"]
        ),
    )
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.parent.is_dir():
        parser.error("Output parent directory must already exist")
    if args.out.exists():
        parser.error("Refusing to overwrite an existing receipt")
    report = run(args.out)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "status": report["status"],
                "wall_seconds": report["wall_seconds"],
                "summary": report["summary"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
