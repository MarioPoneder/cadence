# Cadence 0.61 documentation

This guide covers **`0.61.0.dev1` on `main`**. Start with the smallest adequate
flat network and add ordinary deep layers when the task needs intermediate
representations. Measure quality and latency before scaling. Both use the same
bounded patch rule, settlement and learning API.

| You need | Start with |
| --- | --- |
| A small direct response | A flat population reading the sensors |
| A more expressive routine | Ordinary deep layers whose live states settle together |
| An explicit internal-error experiment | Experimental recursive observers, compared with capable ordinary controls |

**System 1** means acquired routine competence, which can use ordinary deep
layers. Depth is not a latency guarantee. **System 2** names the intended extra
recursive correction when routine behavior fails; that complete capability is
unproven. Explicit observers participate in every synchronous solve and may slow
every call. They do not enable automatic on-demand attention or independent
population clocks. See the [experimental capability boundary](EXPERIMENTAL.md).

## A short learning path

1. [Quickstart](QUICKSTART.md): construct a brain, bootstrap a relation, test
   fresh predictions without targets, then save and resume.
2. [Layout quickstarts](VARIANTS.md): build flat and ordinary deep layouts;
   observer recipes are a separate experimental option.
3. [Brain design](BRAIN_DESIGN.md): choose observations, temporal context,
   connected capacity and a cost budget for the task you actually need.
4. [Bootstrapping](BOOTSTRAP.md), then [live operation](LIVE.md): acquire a
   behavior, execute it, learn from actual outcomes and check old skills.

Bootstrapping and live operation are lifecycle phases, separate from System 1
and System 2. The same brain can continue learning in both phases. Numerical
settlement and useful behavior need separate checks.

## Guides and runnable examples

| Read | Purpose |
| --- | --- |
| [Runnable examples](../examples/README.md) | Layout learning, a learned body controller, explicit history, reward, batching and cost measurement |
| [Agent recipe](AGENTS.md) | A compact integration workflow and rules for accurate architecture claims |
| [GPU execution and parallel experience](ACCELERATION.md) | Devices, precision, batch repair and independent lives |
| [Architecture guide](DRSN.md) | Patch equations, multimodal branches, ordinary coupling and recursive readback |
| [Depth, latency and useful work](PERFORMANCE.md) | Query and learning costs, comparison controls and versioned demo evidence |
| [Processing patch](ELEMENT.md) | Local state, prediction errors, repair and retained relations |
| [API reference](REFERENCE.md) | Public classes, parameters, methods and diagnostics |
| [Specification](SPECIFICATION.md) | Qualification, refusal, witness admission and saved continuation |
| [Experimental capabilities](EXPERIMENTAL.md) | Observer costs, unfinished System 2 behavior and evidence limits |

[Source](https://github.com/muellerberndt/cadence) ·
[Public demos](https://github.com/muellerberndt/cadence-demos) ·
[Release notes](../CHANGELOG.md)
