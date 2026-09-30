# Cadence documentation

Build brains with three supported design patterns: **flat settlement**,
**state-coupled settlement**, and **recursive observer settlement**. All use
the same patch law and qualified repair process; populations can be combined
into Deep Recursive Settlement Networks.
Prepare and check a brain during its **bootstrapping phase**, then use it during
its **live phase**, continuing to admit actual witnesses when available.
Both phases use the same mechanism; neither implies an automatic mode switch.
The current documentation covers Cadence 0.50.0, including `History`,
`Reinforcement`, `LearningProgress`, `LiveController` and `slew`.

For a first project, follow the quickstart through learning and saving. Use
`observe` for measured targets and `Reinforcement` for discrete choices from
reward; add `History` when the latest observation omits relevant recent context.
The live guide covers these choices and the responsibilities of a body adapter.
For exact signatures, result fields and refusal behavior, use the API reference.

| Read | Purpose |
| --- | --- |
| [Quickstart](QUICKSTART.md) | Build, query, learn and save a small brain |
| [Bootstrapping and size](BOOTSTRAP.md) | Starting configurations, individual/batch admissions and task checks |
| [Memory, rewards and live control](LIVE.md) | Temporal history, reward credit, replay, curiosity and responsive bodies |
| [GPU execution and parallel experience](ACCELERATION.md) | CPU/GPU selection, batch repair, numerical checks and independent lives |
| [DRSN guide](DRSN.md) | Multimodal layouts and how recursive observation works |
| [Depth, latency and useful work](PERFORMANCE.md) | When flat learners suffice, what recursive settlement costs and how to compare capability |
| [Three settlement design patterns](VARIANTS.md) | Runnable flat, state-coupled and recursive layouts; width, branches and connection cost |
| [Processing patch](ELEMENT.md) | Local state, prediction errors, repair and retained relations |
| [API reference](REFERENCE.md) | Every public class, parameter, method and diagnostic |
| [Specification](SPECIFICATION.md) | Qualification, refusal, witness admission and continuation |

[Demo](https://github.com/muellerberndt/cadence-demos) ·
[Python examples](../examples/README.md) ·
[Source](https://github.com/muellerberndt/cadence) ·
[Release notes](../CHANGELOG.md)
