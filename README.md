<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: self-reading cortical columns with local state, readback and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Examples](https://github.com/muellerberndt/cadence-demos) · [Paper](https://philpapers.org/rec/MUECAP-2) · [PyPI](https://pypi.org/project/cadence-net/) · [Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/REFERENCE.md)

[![PyPI](https://img.shields.io/pypi/v/cadence-net)](https://pypi.org/project/cadence-net/)
[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/cadence-net)](https://pypi.org/project/cadence-net/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/muellerberndt/cadence/blob/main/LICENSE)

**Research toward general intelligence through overlap consensus, equilibrium detuning and self-reflection.**

Cadence's goal is a continuing learning system with the flexibility of animal
and human problem solving: acquiring skills from experience, retaining useful
knowledge, imagining alternatives and creating solutions across domains.
The mission is to find the smallest persistent state and local update rule
that can support these abilities. The building block stays as simple as possible,
like in nature; every part of a brain answers with a settled state of that one
rule; and where a choice appears, evolution across lives is preferred to design.
General intelligence is the research goal; the current library establishes
bounded learning, memory and control results.

That building block has a name and a shape: the **cortical column**. Our columns
are mathematical abstractions of the cortical
columns found in the cerebral cortex of mammals - the human brain
included - the repeating vertical motif in which stacked layers of
neurons read and regulate one another's activity. What we take from
biology is the architecture: a bounded unit with its own retained
evidence, an observer stage that reads the unit's live uncertainty, and
feedback that enters the same executed equations. What we do not take
is biophysics: these are not validated models of biological neurons,
and no result in this library rests on a neuroscience claim.

Cadence builds online predictors from these **self-reading columns**: small
stateful models with explicit observation ports, retained evidence and
reciprocal feedback between a belief and an observer of its uncertainty. A
`Cortex` combines columns into context-dependent predictors; an optional
action-value interface adds reinforcement learning.

The library is pure Python and uses only the standard library.

## The roadmap

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

column = CorticalColumn(decay=0.9)
for reading in (0.2, 0.4, 0.3):
    admitted = column.add(reading)
    assert admitted["accepted"]

belief = column.query()
assert belief["qualified"]
print(belief["answer"], belief["variance"])
```

The column retains discounted evidence, then jointly settles its belief and
uncertainty observer. Querying does not add a new observation. Use explicit
ordered event IDs with `observe` when delivery may be retried.

## Add context

```python
from cadence import Cortex

model = Cortex.from_dimensions(
    n_inputs=2, n_outputs=1, bounds=(-1.0, 1.0), bins=8, depth=1,
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
Cadence learns the evidence within those contexts. Choose output count,
context resolution and observer height explicitly, or use the defaults.

Within a column, observer feedback is reciprocal. Between Cortex levels,
coarser beliefs supply priors to finer ones; this directed hierarchy is a
different structure. Additional levels or observer stages are choices to test,
not a guaranteed improvement.

| Start here | What it covers |
| --- | --- |
| [Quickstart](https://github.com/muellerberndt/cadence/blob/main/docs/QUICKSTART.md) | Sensors, multiple outputs, retries, checkpoints and optional actions. |
| [Reference](https://github.com/muellerberndt/cadence/blob/main/docs/REFERENCE.md) | Public classes, methods, parameters, defaults and errors. |
| [Element](https://github.com/muellerberndt/cadence/blob/main/docs/ELEMENT.md) | Evidence, executed equations, residuals and qualification. |
| [Variants](https://github.com/muellerberndt/cadence/blob/main/docs/VARIANTS.md) | Context depth, width, observer height and evidence limits. |
| [Specification](https://github.com/muellerberndt/cadence/blob/main/docs/SPECIFICATION.md) | What the API guarantees and what the application supplies. |

MIT license.
