"""Ordinary wiring is explicit; retained queries still qualify the full graph."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cadence import Cortex, _repair

ROOT = Path(__file__).resolve().parents[1]


def ordinary(depth, *, fan_in=None):
    layout = Cortex(seed=7, fan_in=fan_in)
    source = layout.input("senses", shape=2)
    for index in range(depth):
        source = layout.column(f"layer{index}", patches=3, inputs=source)
    layout.output("answer", shape=1, reads=source)
    return layout.build()


@pytest.mark.parametrize("depth", (1, 3))
@pytest.mark.parametrize("fan_in", (None, 1))
def test_ordinary_depth_never_adds_observed_error_contacts(depth, fan_in):
    brain = ordinary(depth, fan_in=fan_in)
    assert brain.graph.n_patches == 3 * depth
    assert all(p["role"] == "processing" for p in brain.inspect()["populations"])
    assert all(not p["observes"] for p in brain.inspect()["populations"])
    assert {kind for kind, _, _ in brain.graph.edges} == (
        {"input"} if depth == 1 else {"input", "state"}
    )
    for kind, source, target in brain.graph.edges:
        if kind == "state":
            assert source // 3 == target // 3 - 1
    if fan_in is None:
        assert len(brain.graph.edges) == 6 + 9 * (depth - 1)


@pytest.mark.parametrize("depth", (1, 3))
def test_unchanged_retained_input_skips_proposals_but_not_full_checks(
    monkeypatch, depth
):
    brain = ordinary(depth)
    inputs = {"senses": [0.6, -0.3]}
    assert brain.observe(inputs, {"answer": [0.35]}, event_id=8)["accepted"]
    assert brain.step(inputs)["accepted"]
    before = brain.snapshot()
    evaluations = []
    evaluate = _repair._evaluate

    def record(*args, **kwargs):
        assert kwargs.get("_query_cache") is None
        result = evaluate(*args, **kwargs)
        evaluations.append(result)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(_repair, "_evaluate", record)
        repeated = brain.step(inputs, budget=0)
    assert repeated["accepted"] and repeated["qualified"]
    assert repeated["sweeps"] == repeated["work"]["proposals"] == 0
    assert repeated["work"]["backtracks"] == 0
    assert repeated["work"]["evaluations"] == len(evaluations) == 2
    assert repeated["work"]["patch_visits"] == 4 * brain.graph.n_patches
    assert repeated["work"]["edge_visits"] == 4 * len(brain.graph.edges)
    assert brain.snapshot() == before
    # Independently project every free state derivative from the fresh check.
    bound = brain.config["state_bound"]
    residual = max(
        abs(x - max(-bound, min(bound, x - g)))
        for x, g in zip(brain.state, evaluations[-1]["gradient_state"], strict=True)
    )
    # The solver uses a cancellation-safe form; the direct projection above
    # can round a sub-ulp displacement to zero.
    assert repeated["stationarity"] == pytest.approx(residual, abs=1e-15, rel=1e-12)
    assert max(repeated["stationarity"], residual) <= brain.config["tolerance"]
    changed = {"senses": [-0.6, 0.3]}
    refused = brain.step(changed, budget=0)
    assert not refused["accepted"] and brain.snapshot() == before
    assert brain.step(changed)["accepted"]
    # Query repair preserves the latest experience and all learned relations.
    after = json.loads(brain.snapshot())
    prior = json.loads(before)
    assert {k: v for k, v in after.items() if k != "state"} == {
        k: v for k, v in prior.items() if k != "state"
    }


def test_default_import_and_ordinary_solve_start_no_implicit_worker():
    script = """
import importlib.abc
import sys
import threading
sys.path.insert(0, sys.argv[1])
blocked = {'cadence._attention', 'cadence._experience', 'cadence._records',
           'cadence._incremental', 'torch'}
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError('implicit optional runtime import: ' + fullname)
sys.meta_path.insert(0, Reject())
def no_worker(*args, **kwargs):
    raise AssertionError('ordinary brain started a worker')
threading.Thread.start = no_worker
from cadence import Cortex
c = Cortex()
x = c.input('sense', shape=1)
h = c.column(patches=3, inputs=x)
y = c.column(patches=1, inputs=h)
c.output('answer', shape=1, reads=y)
b = c.build()
assert b.step({'sense': [.4]})['accepted']
assert b.step({'sense': [.4]}, budget=0)['accepted']
assert not blocked.intersection(sys.modules)
"""
    run = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT / "src")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode == 0, run.stdout + run.stderr


def example(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "examples" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,entry,args",
    (
        ("batch_bootstrap", "run", ("python", 8, 0, 20)),
        ("parallel_bootstrap", "run_life", (0, 20)),
        ("live_control", "prepare", ()),
        ("live_learning", "hidden_cue", (0,)),
        ("live_learning", "retention", (0,)),
        ("live_learning", "reward_brain", (0,)),
    ),
)
def test_generic_examples_construct_only_ordinary_populations(
    monkeypatch, name, entry, args
):
    module = example(name)
    build = Cortex.build

    class Captured(Exception):
        pass

    def inspect_builder(layout):
        brain = build(layout)
        assert all(kind != "residual" for kind, _, _ in brain.graph.edges)
        assert all(not p["observes"] for p in brain.inspect()["populations"])
        raise Captured

    # Stop immediately after genuine compilation; behavior has separate gates.
    monkeypatch.setattr(Cortex, "build", inspect_builder)
    with pytest.raises(Captured):
        getattr(module, entry)(*args)


def test_layout_cli_defaults_to_flat_and_keeps_explicit_research_choices(
    monkeypatch, capsys
):
    module = example("layout_learning")
    selected = []
    monkeypatch.setattr(
        module,
        "learn",
        lambda kind, **kwargs: selected.append(kind) or {"layout": kind},
    )
    module.main([])
    assert selected == ["flat"]
    assert json.loads(capsys.readouterr().out) == [{"layout": "flat"}]
    selected.clear()
    module.main(["--layout", "all"])
    assert selected == list(module.LAYOUTS)
    for kind in module.LAYOUTS:
        brain = module.make_brain(kind)
        assert any(k == "residual" for k, _, _ in brain.graph.edges) == (
            kind == "recursive"
        )
