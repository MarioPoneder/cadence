# Cadence 0.61 documentation

This guide covers **`0.61.0`**. A Cadence brain is one equilibrium: every
patch is an observer that predicts its own state from the ports it reads and
settles its disagreement with the others, and the settled whole is the brain's
world model. A disturbance is repaired, and learning is the same repair with
the relations made eligible. The builder refuses a population that settles with
no other population, so the smallest brain has two populations. Start there and
add depth when the task needs intermediate representations. Measure quality and
latency before scaling.

| You need | Start with |
| --- | --- |
| A direct response | Two coupled populations: one reading the sensors, one reading the first |
| A more expressive routine | Deeper composition whose live states settle together |
| An explicit internal-error experiment | Experimental recursive observers, compared with state-coupled controls |

**System 1** means acquired routine competence, which can use several coupled
populations. Depth is not a latency guarantee. **System 2** names the intended
extra recursive correction when routine behavior fails; that complete
capability is unproven. Explicit observers participate in every synchronous
solve and may slow every call. They do not enable automatic on-demand attention
or independent population clocks. See the
[experimental capability boundary](EXPERIMENTAL.md).

## A short learning path

1. [Quickstart](QUICKSTART.md): construct a brain, bootstrap a relation, test
   fresh predictions without targets, then save and resume.
2. [Layout quickstarts](VARIANTS.md): build the two-population brain and deeper
   composition; observer recipes are a separate experimental option.
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
| [Architecture guide](DRSN.md) | Patch equations, multimodal branches, coupling and recursive readback |
| [Depth, latency and useful work](PERFORMANCE.md) | Query and learning costs, comparison controls and versioned demo evidence |
| [Processing patch](ELEMENT.md) | Local state, prediction errors, repair and retained relations |
| [API reference](REFERENCE.md) | Public classes, parameters, methods and diagnostics |
| [Specification](SPECIFICATION.md) | Qualification, refusal, witness admission and saved continuation |
| [Experimental capabilities](EXPERIMENTAL.md) | Observer costs, unfinished System 2 behavior and evidence limits |

[Source](https://github.com/muellerberndt/cadence) ·
[Public demos](https://github.com/muellerberndt/cadence-demos) ·
[Release notes](../CHANGELOG.md)
