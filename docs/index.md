# Cadence documentation

Cadence is an experimental brain built from local state, ports, plastic
relationships, memory and repair. **System 1** is the default continuing brain.
**System 2** optionally adds observing cortical regions whose recursive feedback
joins the same neural-graph settlement. The base can already be deep and modular.
The goal is a simulated human-like brain; the current mechanisms have bounded,
testable contracts.

Animal and human brains learn from experience and not by backpropagation with
gradient descent. Cadence follows that design: [local free/nudged learning](learning.md),
[reward plasticity](reward.md) and [memory](memory.md) change the brain while it
runs, with no separate training mode. [Concepts](concepts.md#compared-with-backprop-networks)
compares the update mechanisms, and the [README](../README.md#why-cadence) lists
what this gives an embodied system.

Python 3.11+ and NumPy are required.

```sh
python -m pip install cadence-net==0.70.0
```

## Start here

1. [Build a brain](brain.md): compose System 1, add optional observers and save it.
2. [Continuous interaction](continuous.md): observations, actual rewards,
   demonstrations, memory and private imagination.
3. [Quickstarts](quickstart.md): small learning tasks you can run.
4. [Contracts](contracts.md): numerical qualification, learning and refusal.

`Brain.compose(inputs, actions, modules=(64,), observers=())` includes
working and consolidating memory. Add observer widths when you want recursive
feedback. Actions require a qualified full state; more regions do not guarantee
better decisions.

## Choose a deeper guide

| Need | Guide |
| --- | --- |
| Trace and associative-memory rules | [Memory](memory.md), [continued learning](continuous.md) |
| Event records and consolidation | [Record patch](record-patch.md) |
| Learned environmental consequences and private action planning | [Interaction](interaction.md), [temporal model](temporal.md), [planning](planning.md) |
| Finite protection of selected learned responses | [Temporal memory](temporal-memory.md), [runnable example](../examples/memory_imagination.py) |
| Recursive wiring and learning | [Recursive settlement](recursive-settlement.md), [recursive training](recursive-training.md) |
| Custom regions and connections | [Cortices](cortex.md), [genomes](evolution.md), [data shapes](build.md) |
| Device execution and cost | [Backends](backends.md), [scaling](scaling.md) |
| Exact state-and-error population model | [Advanced population solver](equilibrium/index.md) |

<a id="kept-for-existing-experiments"></a>

Other model interfaces include [PatchNet](patchnet.md), [belief patches](belief.md),
[steering](steering.md) and [population execution](population.md). Use their stated
learning and numerical contracts when combining them.

## Evaluate and contribute

[Task design](task-design.md), [common missteps](missteps.md),
[certificates](certificate.md), [protocols](protocols.md) and [receipts](receipts.md)
help separate numerical qualification from useful acquired behavior. Test free
recall, competing experience, actual outcomes and saved continuation.
[Examples](https://github.com/muellerberndt/cadence-demos) show application work.

[API](api.md) · [Architecture](architecture.md) · [Troubleshooting](troubleshooting.md) ·
[Contributing](../CONTRIBUTING.md) · [Changelog](../CHANGELOG.md) ·
[Paper](https://philpapers.org/rec/MUECAP-2)

Conditional proofs and their audit live in the canonical
[Cadence flagship Lean library](https://github.com/FloatingPragma/oph-meta/blob/main/cadence-flagship/lean/README.md).
