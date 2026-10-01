# Cadence documentation

Build brains with three supported design patterns: **flat settlement**,
**state-coupled settlement**, and **recursive observer settlement**. All use
the same patch law and qualified repair process; populations can be combined
into Deep Recursive Settlement Networks.
Prepare and check a brain during its **bootstrapping phase**, then use it during
its **live phase**, continuing to admit actual witnesses when available.
Both phases use the same mechanism; neither implies an automatic mode switch.
Convenience helpers include `History`, `Reinforcement`, `LearningProgress`,
`LiveController` and `slew`. The [0.60 candidate migration notes](MIGRATION_060.md)
explain changed contracts and checkpoint compatibility.

## Start here

1. [Quickstart](QUICKSTART.md): construct one flat brain, teach a relation,
   query without targets and save/resume it.
2. [Three layout quickstarts](VARIANTS.md): swap in deep ordinary layers or
   recursive observers while keeping the same body interface.
3. [Brain design](BRAIN_DESIGN.md): choose sufficient observations, connected
   capacity, acquisition checks and a declared compute budget for your task.
4. [Bootstrapping](BOOTSTRAP.md), then [live operation](LIVE.md): acquire a
   behavior, check it in the environment and continue with actual outcomes.

For agents implementing an integration, use the [API reference](REFERENCE.md)
for exact signatures and the [specification](SPECIFICATION.md) for mutation,
refusal, event identity and checkpoint rules. The public methods are the same
for all layouts; users do not wire an external evaluator to each population.
Ordinary deep layers already settle jointly and can support routine skills.
The intended automatic allocation between routine and deeper corrective work
remains an [unreleased integration goal](MIGRATION_060.md#keep-the-application-boundary-small).

## Guides and contracts

| Read | Purpose |
| --- | --- |
| [Quickstart](QUICKSTART.md) | Build, query, learn and save a small brain |
| [Designing an efficient, capable brain](BRAIN_DESIGN.md) | Choose information, layout, capacity, training checks and compute budgets |
| [0.60 candidate migration notes](MIGRATION_060.md) | Changed contracts, experimental scope and checkpoint compatibility |
| [Bootstrapping and size](BOOTSTRAP.md) | Starting configurations, individual/batch admissions and task checks |
| [Memory, rewards and live control](LIVE.md) | Temporal history, reward credit, replay, curiosity and responsive bodies |
| [GPU execution and parallel experience](ACCELERATION.md) | CPU/GPU selection, batch repair, numerical checks and independent lives |
| [DRSN guide](DRSN.md) | Multimodal layouts and how recursive observation works |
| [Depth, latency and useful work](PERFORMANCE.md) | When flat learners suffice, what recursive settlement costs and how to compare capability |
| [Three settlement design patterns](VARIANTS.md) | Teach, query and resume flat, deep ordinary and recursive layouts; inspect connection cost |
| [Processing patch](ELEMENT.md) | Local state, prediction errors, repair and retained relations |
| [API reference](REFERENCE.md) | Every public class, parameter, method and diagnostic |
| [Specification](SPECIFICATION.md) | Qualification, refusal, witness admission and continuation |

[Demo](https://github.com/muellerberndt/cadence-demos) ·
[Python examples](../examples/README.md) ·
[Source](https://github.com/muellerberndt/cadence) ·
[Release notes](../CHANGELOG.md)
