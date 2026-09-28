# Choosing a layout

Cadence separates processing capacity, parallel branches and recursive
observation. Every population uses the same patch law.

| Choice | Construction | Meaning |
| --- | --- | --- |
| Width | `column(patches=256, inputs=...)` | 256 processing states and their incoming relations |
| Parallel branches | Multiple columns reading different sensors | Separate sensory representations participating in the same solve |
| Ordinary composition | `column(inputs=(vision, hearing), ...)` | A population reading other populations' states |
| Recursive observation | `observer(observes=(vision, hearing), ...)` | State **and exact live error** readback with feedback in the joint energy |
| Deeper observation | An observer includes an earlier observer in `observes` | Observation of a system that already observes other patches |
| Motor readout | `output(shape=(8,), reads=population)` | Eight selected patch states, with no separate readout network |

An observer can receive ordinary `inputs` too. Extra depth gives additional
coupled constraints and capacity; it does not grant a final veto or guarantee
better reasoning. A population's role is its wiring, not a different neuron
class. Sensory `shape` describes supplied data, not learned interpretation.

Patches in a flat population have separate incoming relations. Selecting one
as an output does not give it access to all the other patches: connect a second
population to the first to create a learned hidden representation. More unused
flat patches are not a substitute for that connection. See
[bootstrapping](BOOTSTRAP.md) for small starting sizes and measured setup
requirements. The bootstrapping and live phases can use the same persistent
layout; live experience can continue to repair its relations through `observe`.

## A parallel system with recursive observation

```python
from cadence import Cortex

cortex = Cortex(seed=7)
eyes = cortex.input("eyes", shape=(4, 4))
ears = cortex.input("ears", shape=(4,))
vision = cortex.column("vision", patches=8, inputs=eyes)
hearing = cortex.column("hearing", patches=8, inputs=ears)
fusion = cortex.column("fusion", patches=8, inputs=(vision, hearing))
reflection = cortex.observer(
    "reflection", patches=8, observes=(vision, hearing, fusion),
)
meta = cortex.observer("meta", patches=4, observes=(fusion, reflection))
cortex.output("move", shape=(2,), reads=meta)
brain = cortex.build()
assert brain.inspect()["patches"] == 36
assert brain.inspect()["sensor_coverage"] == 20
```

## Connectivity and cost

The default `fan_in=None` connects each patch to every coordinate of its declared
sources. Set a positive integer, such as `Cortex(fan_in=8)`, to request sparse
wiring. That count is capped at source width and raised when needed for aggregate
coverage across the destination population. Aggregate coverage does not guarantee
that each selected output can use every sample: unconnected patches cannot relay
what they read through shared, fixed sensors.

Observation has both state and error ports. Sharing a state port through `inputs`
and `observes` does not duplicate that connection. Dense wiring increases cost;
256 patches reading a 240×320 RGB image require 58,982,400 input connections,
exceeding the default one-million limit. Use a smaller declared representation,
explicit sparsity or a larger budget; the builder never silently drops inputs to
fit. Neither wiring choice supplies learned visual features by itself.

`max_inputs`, `max_patches` and `max_connections` bound compilation. The solver's
budget counts accepted repair sweeps; work also includes initial evaluations,
rejected line-search proposals and final qualification. Defaults are starting
points for small models, not a guarantee of interactive speed at arbitrary
width or sensor resolution. This pure-Python implementation does not execute
branches on parallel accelerators.

To test the value of recursion, compare a nested layout with shallow and
ordinary-composition controls, match information and capacity, and account for
all bootstrapping, live learning and settlement work. Measure target-free task
performance, numerical refusal rates and latency. A wider brain or a larger
budget alone is not proof that observation adds value.
