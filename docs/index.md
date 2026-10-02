# Cadence documentation

Cadence aims to build a continuing human-like brain from simplified biological
mechanisms. Local state, ports, plastic relationships, memory and repair form
its default **System 1** foundation. That animal-like brain can already be deep
and modular. **System 2** optionally adds observing cortical regions whose
recursive feedback joins the same neural-graph settlement. These names describe
their roles; useful learned self-correction is measured on the task.

Cadence 0.70 is an alpha release. It restores working memory,
record, continuous-learning and imagination mechanisms from the pre-reset
library. Python 3.11+ and NumPy are required. Original source-bound experiments
retain their own versions; their behavior is not automatically reproduced by
a new package.

## Start with System 1

- [Quickstarts](quickstart.md): runnable learning examples.
- `GenericBrain.compose(inputs, actions, modules=(64,), observers=())` builds
  the continuing foundation with working and consolidating memory. Optional
  `observers=()` keeps System 1 alone; observer widths enable optional System 2
  feedback in that same graph. Base modules already provide depth.
  Actions require the full equation residual to qualify; an exhausted budget
  produces a refusal, not a partly settled action.
- [Build a brain](brain.md) and [continuous interaction](continuous.md):
  observations, actual action outcomes, working traces, associative memory and
  complete `GenericBrain` continuation.
- [Memory and imagination example](../examples/memory_imagination.py): retained
  cues, finite response protection and private planning with actual toy outcomes.
- [Learn, act and observe](interaction.md): learn a temporal world model,
  privately consider controls and learn from the actual consequence.
- [Build from your data](build.md), [orientation](orientation.md) and
  [troubleshooting](troubleshooting.md): shapes, preprocessing and first checks.

## Memory, imagination and optional observation

| Capability | Guide |
| --- | --- |
| Traces, records and plastic associations | [Memory](memory.md), [continued learning](continuous.md) |
| Event context, individual records and learning during sleep | [Record patch](record-patch.md) |
| Learned temporal paths and private imagination | [Temporal model](temporal.md) |
| Continuous action proposals under the learned model | [Planning](planning.md) |
| Bounded protection of declared learned responses | [Temporal memory](temporal-memory.md) |
| Jointly settled observer and observed populations | [Recursive settlement](recursive-settlement.md), [recursive training](recursive-training.md) |
| Explicit regions, ports and custom wiring | [Cortices](cortex.md), [structured ports](api.md), [genomes](evolution.md) |

These mechanisms have different numerical and learning contracts. Read
[contracts](contracts.md) before combining them. The newer state-and-error
population solver remains a separate
[experimental implementation](equilibrium/index.md); it does not replace the
foundation or prove the older memory capabilities redundant.

<a id="kept-for-existing-experiments"></a>

## Other supported compositions

[Belief patches](belief.md), [steering and life](steering.md),
[record composition](record-patch.md), [PatchNet](patchnet.md) and
[population execution](population.md) remain available for their declared uses.
Their guides state how computation and learning proceed. Sequential steering
or finite repair must not be described as a jointly qualified observer graph
without that guarantee.

## Measure and reproduce

Use [task design](task-design.md), [common missteps](missteps.md),
[certificates](certificate.md), [protocols](protocols.md) and [receipts](receipts.md)
to separate numerical qualification from acquired behavior. Check free recall,
interference, actual outcomes and saved continuation. Use
[backends](backends.md) and [scaling](scaling.md) for declared device/work costs.

The [examples repository](https://github.com/muellerberndt/cadence-examples)
preserves Amen, Connect Four and other applications with their own sources and
receipts. [Atari Arcade](https://github.com/muellerberndt/cadence-demos/tree/main/atari-arcade)
uses a separate population-engine browser port. Keep each application's actual
runtime, learned checkpoint and supplied assistance explicit during recovery.

## Reference

[API](api.md) · [Changelog](../CHANGELOG.md) · [Contributing](../CONTRIBUTING.md) ·
[Architecture](architecture.md) · [World models](equilibrium-world-models.md) ·
[Creativity and self-reflection](creativity.md) · [Paper](https://philpapers.org/rec/MUECAP-2)

Formal theorems and their audit live in the canonical
[Cadence flagship Lean library](https://github.com/FloatingPragma/oph-meta/blob/main/cadence-flagship/lean/README.md).
