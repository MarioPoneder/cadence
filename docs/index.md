# Cadence documentation

New readers start with [the guided entry](README.md): the principle, the test that tells a
Cadence brain from a layered network, what is implemented today, and the guides in reading
order. This page is the catalogue.

Cadence is an experimental brain built from local state, ports, plastic
relationships, memory and repair. **System 1** is the default continuing brain.
**System 2** optionally adds observing cortical regions whose recursive feedback
joins the same neural-graph settlement. The base can already be deep and modular.
The goal is a simulated human-like brain; biological names describe functional
roles. Bootstrap useful reciprocal relations and memory, act in the world,
repair witnessed failures, and continue the same acquired brain.

Start with [one continuing equilibrium brain](world-model.md). It explains how
parameters and memory support a family of equilibria under changing evidence,
and where the implemented policy/memory loop ends and integrated world-model
development begins. [Local learning](learning.md), [reward plasticity](reward.md)
and [memory](memory.md) have distinct tested update rules. Internal consistency
does not establish correct understanding or inexpensive computation.

Python 3.11+ and NumPy are required.
These guides describe Cadence 0.71.1. Install the published package:

```sh
python -m pip install cadence-net==0.71.1
```

## Start here

1. [The continuing world model](world-model.md): lifecycle, design intent and current boundaries.
2. [Quickstart](quickstart.md): run one brain through observations and outcomes.
3. [Build a brain](brain.md): compose System 1, add optional observers and save it.
4. [Continuous interaction](continuous.md): observations, actual rewards,
   demonstrations, memory and private imagination.
5. [Contracts](contracts.md): numerical qualification, learning and refusal.

`Brain.compose(inputs, actions, modules=(64,), observers=())` includes
working and consolidating memory. Add observer widths when you want recursive
feedback. Actions require a qualified full state; more regions do not guarantee
better decisions.

## Choose a deeper guide

| Need | Guide |
| --- | --- |
| Bootstrap, use, disruption and saved continuation in one life | [Continuing brain example](../examples/continuing_brain.py), [experience design](experience.md) |
| Isolated graph learning or calibration controls | [Learning rule](learning.md), [task recipes](tasks.md) |
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
[Application demos](https://github.com/muellerberndt/cadence-demos) show current
applications, including `Brain.compose`. The separate
[research archive](https://github.com/muellerberndt/cadence-examples) retains
examples and viewer tools with their own library pins.

[API](api.md) · [Architecture](architecture.md) · [Troubleshooting](troubleshooting.md) ·
[Contributing](../CONTRIBUTING.md) · [Changelog](../CHANGELOG.md) ·
[Paper](https://philpapers.org/rec/MUECAP-2)

Conditional proofs and their audit live in the canonical
[Cadence flagship Lean library](https://github.com/FloatingPragma/oph-meta/blob/main/cadence-flagship/lean/README.md).
