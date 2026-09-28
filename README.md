<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: recursive settling populations with local state, readback and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Examples](https://github.com/muellerberndt/cadence-demos) · [Paper](https://philpapers.org/rec/MUECAP-2) · [PyPI](https://pypi.org/project/cadence-net/) · [Documentation](docs/index.md)

[![PyPI](https://img.shields.io/pypi/v/cadence-net)](https://pypi.org/project/cadence-net/)
[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/cadence-net)](https://pypi.org/project/cadence-net/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/muellerberndt/cadence/blob/main/LICENSE)

**Build Deep Recursive Settlement Networks: processing populations and their
observers, learning and settling together.**

Cadence explores a simple idea: a brain can be a persistent system of local
relations that repairs its state as experience arrives. An observer reads
what other patches are doing and influences them while remaining part of the
same equilibrium. Another observer can read that enlarged system in turn.

Cadence implements this population architecture in pure
Python, using only the standard library. Every processing and observer patch
uses one local prediction relation. The compiled brain supports joint
settlement, supervised witness learning, diagnostics and continuation.

## Three pillars

- **Minimalism:** one patch rule, a small public API, no runtime dependencies.
  Focused modules keep each responsibility easy to find.
- **User-friendliness:** explicit layouts, sensible defaults and runnable examples.
- **Agent-friendliness:** exact contracts, inspectable state and explicit refusal.

Every core change is reviewed against all three. The
[contributor instructions](AGENTS.md) define the checks.

## Install

Python 3.11 or later; no runtime dependencies:

```sh
python -m pip install --upgrade "cadence-net>=0.45.0"
```

If your package index has not listed a new release yet, its wheel is also
available from [GitHub Releases](https://github.com/muellerberndt/cadence/releases/latest).

## Declare the brain

```python
from cadence import Cortex

cortex = Cortex(seed=7)
eyes = cortex.input("eyes", shape=(8, 8))
ears = cortex.input("ears", shape=(2, 16))
body = cortex.input("sensory_nerves", shape=(8,))
senses = (eyes, ears, body)

c1 = cortex.column("perception", patches=16, inputs=senses)
c2 = cortex.observer(
    "integration", patches=8, inputs=senses, observes=(c1,),
)
c3 = cortex.observer(
    "reflection", patches=8, inputs=senses, observes=(c1, c2),
)
cortex.output("motor_nerves", shape=(8,), reads=c3)
brain = cortex.build()

assert brain.inspect()["patches"] == 32
assert brain.inspect()["sensor_coverage"] == 104
```

`inputs` supplies sensory or represented data. `observes` adds readback of
**live patch states and exact current prediction errors**. Observer constraints
feed back into the observed states through the same energy and repair process.
They are not a separate decision made after the lower system finishes.

Population size counts processing patches. Sensor shape counts input samples;
a larger camera shape does not supply trained visual understanding. The eight
outputs expose eight settled patch values, without a separate policy network.
The guide also shows parallel sensory branches and inspection of actual wiring.
Every patch reads all coordinates of its declared sources by default; explicit
`fan_in` opts into sparse sampling. This is a layout example, not a pretrained
vision/audio model. For a first learning task, start smaller using the
[training guide](docs/TRAINING.md).
Inspection also reports sensor coverage for each output coordinate, so you can
check the decision's connections before training.

## Learn a small relation

This inexpensive example teaches a small brain a signed input/output relation.
It uses the same population primitive, with fewer patches for a quick run.

```python
teacher_layout = Cortex(seed=2)
signal = teacher_layout.input("signal", shape=(1,))
base = teacher_layout.column("base", patches=4, inputs=signal)
reflection = teacher_layout.observer(
    "reflection", patches=2, inputs=signal, observes=base,
)
teacher_layout.output("answer", shape=(1,), reads=reflection)
learner = teacher_layout.build()

for event_id in range(40):
    value = (-0.8, 0.8)[event_id % 2]
    update = learner.observe(
        {"signal": [value]}, {"answer": [value]}, event_id=event_id,
    )
    assert update["accepted"]

# Fresh amplitudes, with no output targets supplied to the brain.
assert learner.predict({"signal": [-0.4]})["answer"][0] < -0.2
assert learner.predict({"signal": [0.4]})["answer"][0] > 0.2
```

Actual witnesses clamp the supplied outputs and allow joint repair of live
state and retained relation parameters. Only qualified proposals are committed.
Queries hold learned parameters fixed; hypothetical clamps never become
experience. Learning is evaluated on later unclamped predictions, as above.
This example establishes a small acquired relation, not a benefit from depth.

## Scope

The reference engine repairs a nonlinear residual energy using analytic
derivatives and a bounded descent procedure. Every participating population is
included in the final constrained-stationarity check. Prediction errors can
remain nonzero at a qualified compromise; different starting states can reach
different stationary points. A solver can refuse when its budget is exhausted.

Performance evaluations and comparisons remain ongoing. Tests cover layouts,
derivatives through recursive error readback, reciprocal
interventions, actual acquisition, refusal and checkpoint custody. The goal is
a reusable learner across perception, memory, reasoning and embodied action.
Reward-driven temporal credit, broader capability and advantages from recursive
depth require further controlled experiments. Follow the
[DRSN completion epic](https://github.com/muellerberndt/cadence/issues/51).

| Documentation | Scope |
| --- | --- |
| [DRSN guide](docs/DRSN.md) | Population layouts, equations, learning, all configuration, diagnostics and checkpoints |
| [Quickstart](docs/QUICKSTART.md) | Inputs, settlement, learning and saved continuation |
| [Training and size](docs/TRAINING.md) | Small starting layouts, data conditioning, teaching and validation |
| [API reference](docs/REFERENCE.md) | Every public class, method and configuration parameter |
| [Mathematical specification](docs/SPECIFICATION.md) | Guarantees, qualification and evidence boundaries |

MIT license.
