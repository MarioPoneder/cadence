<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: self-reading cortical columns with local state, readback and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Examples](https://github.com/muellerberndt/cadence-demos) · [Paper](https://philpapers.org/rec/MUECAP-2) · [PyPI](https://pypi.org/project/cadence-net/) · [Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/REFERENCE.md)

[![PyPI](https://img.shields.io/pypi/v/cadence-net)](https://pypi.org/project/cadence-net/)
[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/cadence-net)](https://pypi.org/project/cadence-net/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/muellerberndt/cadence/blob/main/LICENSE)

**Deep Recursive Settlement Networks (DRSNs), built from cortical columns that
continually update their world model from experience.**

Cadence is a library for building persistent, self-reading learning systems.
New experience changes the evidence held by local columns, disturbing their
existing equilibrium. Local feedback repairs the affected message relationships
until their beliefs settle again. The system retains what it has learned for
the next interaction. Its world model is this retained evidence and the settled
beliefs about the quantities an application asks it to model.

The central idea is **recursive observation inside the same equilibrium**.
An observer reads a column's live uncertainty and sends feedback that changes
it. Further observer stages read and regulate that activity in turn.
**The observers become part of the system they observe: all stages remain live
and settle together.** This is the recursive structure at the heart of a DRSN.

The building block is the **cortical column**: a bounded patch with local
memory, typed observation ports, readback and repair. Inspired by mammalian
cortical organization, Cadence's columns are mathematical abstractions, not
biophysical simulations. A `Cortex` combines them into contextual predictors;
an optional action-value interface supports learning from the consequences
of actions. The current models learn scalar quantities and context-dependent
outputs; useful depth and comparative performance are still being evaluated.

The library is pure Python and uses only the standard library. Building on
Cadence with a coding agent: hand it [AGENTS.md](AGENTS.md), the operational
pitfalls file, alongside the docs.

## The roadmap

The goal is continuing intelligence: learning from experience, retaining useful
knowledge, imagining alternatives and solving problems across domains.
The building block stays as simple as possible; every part answers through
local settlement, and evolution across lives is preferred to hand-designed
complexity.

We want to reach a point where we can effortlessly evolve a human-like brain,
teach it first by imitation and then through its own life, and give it an
experience identical to that of a human living in our world. Brains with
capabilities far beyond ours are thinkable on the same path; human-level
competence comes first, as a sensible milestone. The steps from the
brains in this library to that milestone are tracked as issues in this repository,
rung by rung, each with its task, its control and its falsifier.

## Install

From a checkout, with Python 3.11 or later:

```sh
python -m pip install -e .
```

The optional `games` extra adds Gymnasium and ALE; neither is needed below.

## Learn from a sensor

```python
from cadence import CorticalColumn

column = CorticalColumn(height=3, decay=0.9)
for reading in (0.2, 0.4, 0.3):
    admitted = column.add(reading)
    assert admitted["accepted"]

belief = column.query()
assert belief["qualified"]
print(belief["answer"], belief["variance"])
```

`height=3` adds two recursive rate-observer stages above the base
belief–precision loop. All of them participate in the same settlement.
Each `add` proposes new retained evidence and admits it after the solve
qualifies; `query` reads the resulting belief without adding an observation.
Use explicit ordered event IDs with `observe` when delivery may be retried.

## Add context

```python
from cadence import Cortex

model = Cortex.from_dimensions(
    n_inputs=2, n_outputs=1, bounds=(-1.0, 1.0), bins=8, depth=1, height=3,
)
context = (-0.5, 0.25)

before = model.predict(context)[0]
result = model.observe(context, -0.3)
assert result["accepted"]
after = model.predict(context)[0]
print(before, after)

saved = model.snapshot()
restored = Cortex.from_snapshot(saved)
assert restored.predict(context) == model.predict(context)
```

Here two normalized sensor values select a context, and the observed target
teaches one scalar output. Built-in binning supplies the representation;
Cadence learns the evidence within those contexts. `height` controls recursive
observer stages inside each column; `depth` controls the context hierarchy.
They are independent choices, alongside output count and context resolution.

Within a column, observer feedback is reciprocal. Between Cortex levels,
coarser beliefs supply priors to finer ones; this directed hierarchy is a
different structure. Each output's active context chain is solved and qualified
separately. Additional levels or observer stages are choices to test, not a
guaranteed improvement.

| Start here | What it covers |
| --- | --- |
| [Quickstart](https://github.com/muellerberndt/cadence/blob/main/docs/QUICKSTART.md) | Sensors, multiple outputs, retries, checkpoints and optional actions. |
| [Reference](https://github.com/muellerberndt/cadence/blob/main/docs/REFERENCE.md) | Public classes, methods, parameters, defaults and errors. |
| [Element](https://github.com/muellerberndt/cadence/blob/main/docs/ELEMENT.md) | Evidence, executed equations, residuals and qualification. |
| [Variants](https://github.com/muellerberndt/cadence/blob/main/docs/VARIANTS.md) | Context depth, width, observer height and evidence limits. |
| [Specification](https://github.com/muellerberndt/cadence/blob/main/docs/SPECIFICATION.md) | What the API guarantees and what the application supplies. |

MIT license.
