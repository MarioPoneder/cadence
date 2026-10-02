# Advanced population experiments

Start with [the main brain guide](../brain.md) for System 1 and optional
System 2. This section covers `cadence.experimental.equilibrium`, a separate
experimental population model for studying exact prediction-error feedback.

Each bounded patch has state, ports, retained relations and a local prediction.
Connected patches repair their shared state. A population can read live states
with `inputs=` or also read exact current errors with `observes=`. Returning
influence participates in the same energy and qualification check.

This model uses explicit `History` for temporal context. It does not include
Brain's Trace/SynapticMemory or the temporal model's private planner.
Choose it to experiment with its particular state-and-error equations.

```python
from cadence.experimental.equilibrium import Cortex
```

- [Quickstart](QUICKSTART.md): build, teach, query and save a small model.
- [Wiring](VARIANTS.md): population sizes, branches and error-reading observers.
- [Live operation](LIVE.md): actual outcomes and continued learning.
- [Equations](DRSN.md), [API](REFERENCE.md) and [numerical contract](SPECIFICATION.md).
- [Devices](ACCELERATION.md), [cost](PERFORMANCE.md) and [experimental scope](EXPERIMENTAL.md).

The package is experimental. APIs and saved formats may change. Save the source
and complete application state for experiments you need to reproduce; there is
no cross-version compatibility commitment.
