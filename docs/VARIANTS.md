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

## Sparse connectivity and cost

For each declared source, `fan_in` is capped by its available coordinates and
raised when necessary to cover every source coordinate across the destination
population. Observation has both state and error ports. Sharing a state port
through `inputs` and `observes` does not duplicate that connection. Inspection
shows the actual edges; full source coverage does not make every patch see every
sample, and does not establish perceptual sufficiency.

`max_inputs`, `max_patches` and `max_connections` bound compilation. The solver's
budget counts accepted repair sweeps; work also includes initial evaluations,
rejected line-search proposals and final qualification. Defaults are starting
points for small models, not a guarantee of interactive speed at arbitrary
width or sensor resolution. This pure-Python implementation does not execute
branches on parallel accelerators.

To test the value of recursion, compare a nested layout with shallow and
ordinary-composition controls, match information and capacity, and account for
all training and solving work. Measure target-free task performance, numerical
refusal rates and latency. A wider brain or a larger budget alone is not
proof that observation adds value.
