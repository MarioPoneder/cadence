"""Teach, query and resume three layouts through the same public interface.

Run: PYTHONPATH=src python examples/layout_learning.py
The default is the smallest brain: four ``features`` patches read the sensor and
one ``response`` patch reads them. Select --layout deep for deeper composition,
or explicitly select --layout recursive/all to study observer wiring. This tiny
relation demonstrates the API, not an architectural advantage.
"""

import argparse
import json

from cadence import Brain, Cortex, bootstrap

LAYOUTS = ("small", "deep", "recursive")


def make_brain(kind, *, seed=2):
    """Change wiring only; every brain exposes signal -> answer."""
    if kind not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    layout = Cortex(seed=seed)
    signal = layout.input("signal", shape=1)
    base = layout.column("representation", patches=4, inputs=signal)
    if kind == "small":
        response = layout.column("response", patches=1, inputs=base)
    elif kind == "deep":
        middle = layout.column("integration", patches=2, inputs=base)
        response = layout.column("response", patches=1, inputs=middle)
    else:
        middle = layout.observer("integration", patches=2, observes=base)
        response = layout.observer("response", patches=1, observes=middle)
    layout.output("answer", shape=1, reads=response)
    return layout.build()


def learn(kind, *, seed=2):
    brain = make_brain(kind, seed=seed)
    # Actual values of this explicitly supplied toy relation.
    examples = [
        ({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.8, -0.4, 0.4, 0.8)
    ]
    checks = [({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.6, 0.6)]
    report = bootstrap(
        brain,
        examples,
        checks=checks,
        epochs=20,
        max_error=0.1,
        batch_size=4,
        seed=2,
    )
    if not report["passed"]:
        raise RuntimeError(f"{kind} did not acquire the relation: {report}")

    # Development checks affect stopping; these four inputs do not.
    saved = brain.snapshot()
    restored = Brain.from_snapshot(saved)
    errors = []
    query_work = dict.fromkeys(report["work"], 0)
    for x in (-0.7, -0.2, 0.2, 0.7):
        inputs = {"signal": [x]}
        result = brain.settle(inputs)
        if not result["qualified"]:
            raise RuntimeError(f"{kind} refused a final query: {result['reason']}")
        assert restored.settle(inputs) == result
        errors.append(abs(result["outputs"]["answer"][0] - 0.6 * x))
        for key, count in result["work"].items():
            query_work[key] += count
    assert brain.snapshot() == restored.snapshot() == saved
    assert max(errors) < 0.1

    # A live step retains activity. Learning remains an explicit witness call.
    activity = brain.step({"signal": [0.3]})
    assert activity["accepted"]
    witness = brain.observe({"signal": [0.3]}, {"answer": [0.18]})
    assert witness["accepted"]
    description = brain.inspect()
    return {
        "layout": kind,
        "seed": seed,
        "patches": description["patches"],
        "connections": description["connections"],
        "accepted_examples": report["accepted"],
        "updates": report["updates"],
        "max_fresh_error": max(errors),
        "bootstrap_work": report["work"],
        "four_query_work": query_work,
        "resume_query_work": query_work.copy(),
        "live_step_work": activity["work"],
        "live_witness_work": witness["work"],
        "resume_exact": True,
        "scope": "API and tiny-relation acquisition; different capacities and costs",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("all", *LAYOUTS), default="small")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args(argv)
    selected = LAYOUTS if args.layout == "all" else (args.layout,)
    print(json.dumps([learn(kind, seed=args.seed) for kind in selected], indent=2))


if __name__ == "__main__":
    main()
