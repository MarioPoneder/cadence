> This page documents the preserved experimental engine, imported from
> `cadence.experimental.equilibrium`, inside Cadence 0.70.0. Its source-bound
> engine identity remains `0.62.0`; it is separate from the default Cadence API.

# Agent guide: build one brain, measure its behavior

Follow the repository's [contributor instructions](../../AGENTS.md), including
the principle they open with: a Cadence brain is one equilibrium of patches
settling against each other. For API spelling and numerical semantics, use
[the reference](REFERENCE.md) and [specification](SPECIFICATION.md). The
practical explanation is [brain design](BRAIN_DESIGN.md); start a small
application from [the quickstart](QUICKSTART.md).

The design goal is a simulated human-like brain built from simplified biological
mechanisms. Cortical organization, memory, plasticity and recursive correction
must earn their functional claims through explicit models and measured behavior.
Preserve useful historical mechanisms such as Afterglow when evaluating a new
substrate; consult the current [memory](https://github.com/muellerberndt/cadence/issues/84),
[plasticity](https://github.com/muellerberndt/cadence/issues/85) and
[recursive integration](https://github.com/muellerberndt/cadence/issues/86) goals.

## Explain the supported model first

Use `Cortex` to declare one connected graph, then `build()` to create its
persistent `Brain`. Inputs are supplied samples, populations contain processing
patches, and outputs expose selected patch states. Each patch predicts its own
state from its incoming ports and carries the disagreement as a live error.
All eligible coordinates settle together under one energy and stationarity
check. Stationarity can include nonzero disagreement; it is not task success.

The builder requires at least two populations and rejects disconnected groups.
Population size and connections are separate design choices: widening adds
states and relations, while an intermediate population changes the paths
through the graph. `inputs=` reads sensors or live states; `observes=` adds
exact current error readback. Both return influence through the joint energy.
The solver computes analytic derivatives, including reverse propagation through
error-read dependencies. Learning makes relation parameters eligible within
that same solve; it is not an extra output-loss training pass.

Begin with a small connected graph and measure free behavior before adding
capacity. All populations use the same patch rule. Optional recursive readback
and the intended System 1/System 2 roles have one
[experimental boundary](EXPERIMENTAL.md); do not infer capabilities from a
population's name or graph depth.

The application owns sensor meaning, action decoding and outcome units. `step`
retains qualified activity for the next call; this warm start is not a guarantee
of temporal memory. Learned relations persist and remain plastic through later
`observe` calls. For runnable wiring changes, use [the examples](VARIANTS.md):

```sh
PYTHONPATH=src python examples/equilibrium/layout_learning.py
PYTHONPATH=src python examples/equilibrium/layout_learning.py --layout deep
```

`deep` labels the example with one extra intermediate population; `recursive`
labels the example with error readback. They share the same learning/query/save
interface. Previously declared sources give the public builder acyclic read
dependencies, even though the joint repair returns influence upstream.

## Application recipe

1. **Declare one body boundary.** Define observations, output units, actual
   execution and measured outcomes. Keep preprocessing and action decoding
   explicit. Preserve enough causal context to distinguish required answers;
   future targets, unavailable teacher decisions and privileged environment
   state are not actor inputs. Past executed actions may be part of that context.
   Include measured actuator state when commands alone are ambiguous: a held
   note can be silent after its sample ends, just as a movement command can fail
   against an obstacle. Give a routine branch the signals its job needs; attach
   broader context where a controlled comparison shows it helps.
   A bounded `History` is explicit memory, not learned recurrence.
2. **Acquire a capable routine.** Start with two coupled populations and add
   composition as the task requires. Check output connectivity and target range.
   Use public `observe`/`observe_batch` or `bootstrap`; measured labels are
   witnesses, derived teaching values use `source="estimate"`. A batch has
   private row states and one shared parameter admission, not an implicit
   sequence. When interpreting forecasts as probabilities, preserve the relevant
   event distribution or explicitly account for resampling. Replaying only
   events where an action was available can bias a shared predictor; supply
   selected-action value targets only where that action and its outcome were
   actually recorded.
3. **Test free behavior.** Disconnect the teacher and leave future outputs
   unclamped. Check `qualified` before acting and `accepted` before counting
   learning. Measure the actual body, not just prediction MAE or command flags.
   Refusals and timeouts remain outcomes. Compare independent assessment cases,
   old-skill retention, work and complete command latency.
4. **Attach consequences correctly.** Record the issued forecast before seeing
   its outcome, and preserve the action actually executed. For supported
   discrete reward learning, use the documented `Reinforcement` decision and
   execution acknowledgments. Its estimated Q targets are not automatic
   brain-wide emotion or a planning model. Keep outcome records separate from
   replaceable sensory summaries.
5. **Measure the contribution of added connections.** When investigating error
   readback, compare a competent state-coupled control on the same causal
   information, with disclosed capacity, exposure and work. Establish routine,
   disturb the actual body, measure useful recovery, and recheck retention and
   cost after recovery. Include routine-plus-factual-fit as a control so
   consolidation alone is not called a planning benefit. Adding observers or
   changing a free output does not by itself establish System 2.

Use the operation that matches the intended state change:

| Intent | Operation |
| --- | --- |
| Inspect a free or conditional answer without committing it | `settle`; use `predict` for outputs only with refusal raised as an exception |
| Continue activity with parameters fixed | `step`, optionally with the same `targets`/`interventions` as `settle` |
| Admit one labeled experience and its activity | `observe` |
| Admit labeled examples while preserving current activity | `observe_batch` |

Clamps on `step` and `settle` condition only that call. They do not admit
teaching evidence; a goal-clamped future output is an intention, not a
forecast. A subsequent free call must qualify again. Label measured targets
as witnesses and derived targets as estimates; provenance labels do not
authenticate what the body actually did.

Save the brain with preprocessing, body state, data position, history and RNG.
For reward learning, save the complete learner so pending ownership survives.
Serialize calls to one brain; parallelize independent brains or collection.
`LiveController` keeps a caller responsive while one callback owns the brain.
It does not let an unfinished branch issue a new whole-brain qualified answer.

## Documentation and implementation checks

Keep examples on the existing public API. Prefer a link to one runnable example
over another wrapper or a new application-specific core abstraction. Currently
all populations use the same patch rule. Future specialized mechanisms need
explicit bounded state, ports, readback, learning and repair semantics, with
declared qualification and measured general benefit. Simplicity is a design
requirement, not proof that the current primitive can replace every memory
mechanism. Keep experimental mechanisms distinct from supported public behavior.

For future automatic fast/slow execution, state what invalidates reused work,
how a forecast miss or unmet need recruits correction, and what each
qualification covers. Separately settled or stale states cannot be presented
as one simultaneous global equilibrium. Concurrent action/learning proposals
and separately scheduled state blocks are different runtime claims. Neither
may silently exclude unresolved coordinates from current qualification.
The intended architecture should require no application attention flag or
per-population evaluator.

Keep implemented behavior separate from that target and from experimental
results. Query caching is arithmetic reuse, not learned attention; low
stationarity is not worldly success; a valid snapshot is not task competence.

After documentation changes, run the focused checks from the repository root:

```sh
PYTHONPATH=src python -m pytest -q tests/equilibrium/test_documentation.py tests/equilibrium/test_packaging.py
```

The documentation test executes every Python fence unless it has an explicit
reason to be skipped. Keep signatures, defaults, mutation semantics and refusal
behavior aligned with the reference; report behavioral evidence separately from
test counts. Runtime changes also require the full checks in the root guide.
