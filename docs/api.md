# API reference

Every public name, by module. Start with the [quickstarts](quickstart.md) and
[build your own brain](build.md); each section links the guide that explains its
operations. Optional arguments should be passed by keyword.

The temporal patch: [TemporalPatchNet](#temporalpatchnet-cadencetemporal),
[TemporalPlan](#temporalplan-cadenceplanning), [TemporalMemory](#temporalmemory-cadencetemporal_memory),
[fixed connectivity](#experimental-fixed-connectivity-cadenceexperimental).
The record and belief patches: [RecordPatchNet](#recordpatchnet-cadencerecord_patch),
[Ports](#ports-cadenceports), [BeliefPatch](#beliefpatch-cadencebelief).
The settling brain: [Connectome](#connectome-cadenceconnectome), [Neuron model](#neuron-model-cadenceneuron),
[Brain](#brain-cadencebrain), [Blocks](#blocks-cadenceblocks), [Streams](#streams-cadencestream),
[Records](#records-cadencerecords), [Regions](#regions-cadenceregions), [Generic brain](#generic-brain-cadencegeneric),
[Genome](#genome-cadencegenome), [Learning](#learning-cadencelearning), [The agent and the valence](#the-agent-and-the-valence-cadenceplasticity),
[Certificate](#certificate-cadencecertificate).
Instruments: [the quickstart demos](#the-quickstart-demos-cadencedemo), [Timing](#timing-cadencetiming),
[the reference](#neuron-by-neuron-reference-cadencereference), [Protocols](#protocols-cadenceprotocol),
[Checkpoints](#checkpoints-cadencecheckpoint), [Atlas](#atlas-cadenceatlas), [Receipts](#receipts-cadencereceipts),
[recording](#record-every-settling-step). Kept compositions: [EquilibriumActor](#equilibriumactor-cadenceactor),
[PatchNet](#patchnet-cadencepatch), [task compositions](#optional-task-compositions),
[bounded memory and rehearsal](#bounded-memory-and-rehearsal).

## TemporalPatchNet (`cadence.temporal`)

- `TemporalPatchNet(inputs, hidden, outputs, *, seed=0, output_precision=None, ...)`
  creates the residual temporal model with shared A/B/C maps. Paths have shape
  `(batch, time, ports)`; omitted positive output precision means all ones.
- `observe(inputs, target, *, beta=0.01, rate=0.1, backtrack=False,
  symmetry_tolerance=0.15, max_halvings=8)` repairs observed teaching
  paths and returns `TemporalObservation`. Only valid free activity becomes live;
  detuned activity does not become an observed record. Beta is halved, up to
  `max_halvings` times, until the two detuned paths sit within
  `symmetry_tolerance` of center around the free path; a pair that stays off
  center is refused as `contrast_asymmetric`. See
  [temporal learning](temporal.md#checking-that-the-contrast-is-centered).
  The optional `backtrack` argument:
  accept parameters only after a decreasing causal replay from the original
  boundary. See [temporal learning](temporal.md#checking-a-learning-step).
- `contrast_asymmetry(free, plus, minus)` in `cadence.temporal` is the ratio the
  check reads: `||plus + minus - 2 free|| / ||plus - minus||` over hidden paths.
- `advance(inputs)` carries a free path into live context. `imagine(inputs, *,
  state=None)` predicts privately. `settle(inputs, *, target=None, beta=0.0,
  state=None)` exposes a private phase directly.
- `plan(inputs, *, goal, controls, bounds=None, state=None, beta=0.01,
  rate=1.0, max_steps=32, max_backtracks=16, tolerance=1e-6,
  goal_tolerance=1e-6, symmetry_tolerance=0.15, max_halvings=8,
  method="steepest")` repairs
  selected continuous input ports. Boolean controls
  and bounds broadcast to the input shape. It executes no action and changes
  no live state. Each contrast passes the same symmetry check as `observe`.
  `method="bfgs"` steps along a quasi-Newton direction built from the accepted
  steps, with the same line search and replay.
  See [planning](planning.md) for precise cost and failure semantics.
- `readback()`, `state`, `parameters()` and `snapshot()` expose detached values.
  `reset()` clears activity, not learned parameters. `save(path)`, `load(path)`
  and `restore(snapshot)` preserve continuation state and configuration.
- `set_output_precision(precision)` replaces the supplied loss geometry
  atomically. Free energy and recurrence do not use teaching precision, so
  parameter revisions, free diagnostics and protected constraints stay bound to
  the unchanged maps. `set_parameters(mapping)` validates and replaces all
  learned arrays in one transaction.
- `TemporalPhase` exposes solved hidden/output paths, residuals, curvature and
  work. `TemporalObservation` exposes `updated`, `reason`, the phases, raw
  `delta`, the `beta` of the last contrast and its `contrast_halvings`. With
  backtracking it also reports initial/final loss, accepted rate,
  trial losses and replay count. `TemporalReadback` binds current activity to
  its parameter revision.

## TemporalPlan (`cadence.planning`)

The detached result contains proposed `inputs`, a target-free `prediction`,
`initial_prediction`, fixed `boundary`, `losses`, `step_sizes` and model revision.
`cost`, `initial_cost`, `improved` and `iterations` summarize accepted work.
`converged` concerns the finite-beta projected residual; `predicted_goal_met`
checks modeled cost. Neither certifies an actual outcome. `beta` and
`contrast_halvings` report the detuning of the last contrast and how often it
was halved to center the detuned paths; `method` names the search direction.
Work and failure
fields are described in the [planning guide](planning.md).

## The quickstart demos (`cadence.demo`)

`StreamDemo(day_passes=8, night_passes=240)`, `DecideDemo(steps=80)` and
`BodyDemo(batches=256, decisions=12)` run the three quickstart brains with their seeds and
report as they go: `run()` returns the result; `snapshot(since)` the phase, curves, stats
and the frames of activity and heat since a frame index; `weights_json(what)` the synapses'
absolute weights or their last change; `connectome()` and `atlas_json()` the brain for the
viewer. `serve(demo, port, open_browser)` puts a demo behind a local page and `main` is the
`cadence-demo` command. See [the quickstarts in your browser](demos.md).

## RecordPatchNet (`cadence.record_patch`)

See the [record patch guide](record-patch.md).

- `RecordPatchNet(inputs, hidden, outputs, *, seed=0, output_precision=None,
  cells=4096, active=32, record_rate=0.5, habituation=1e-5, record_bias=0.3,
  slowest=128.0, record_averaging=False, record_homeostasis=0.0, groups=None,
  record_writes="sequential", record_width=None)` creates a
  gated linear context with a `Records` store over the reading
  `[u * sqrt(n) / s, r * h]`, both blocks of unit variance per unit (`s` is the
  running rms norm of witnessed inputs). `record_averaging` and
  `record_homeostasis` pass to `Records` as `averaging` and `homeostasis`. Paths have shape `(batch, time, ports)`.
  `groups=(n1, n2, ...)` makes the ports categorical: one softmax per group,
  cross-entropy for the slow readout, records of `onehot - softmax`.
  `record_writes="batch"` writes a call's moments at once through
  `Records.write_batch(codes, targets)`, each cell moving by the mean of its
  writers' moves. `record_width=w` makes the store hold a fixed random
  `w`-column sign code of the residual instead of one column per output.
- `RecordPatchStack(inputs, hidden, outputs, *, lower=None, seed=0,
  slowest=128.0, groups=None, **upper)` puts a context patch of width `lower`
  below a `RecordPatchNet` that reads `[u, r1 * h1]`; `observe`, `imagine`
  (`state` is the pair of contexts), `advance`, `reset`, `parameters`,
  `snapshot` and `restore` as for one patch. `observe` returns
  `StackObservation`: `updated`, `reason`, the upper patch's `prediction`,
  `delta`, `initial_loss`, `final_loss`, `accepted_rate` and `writes`.
- `JointRecordPatches(cortices, own_inputs, ports, *, rounds=1, damping=1.0,
  cross_adjoint=True)` settles several `RecordPatchNet`s as one equilibrium;
  each `Port(source, target, start, width)` carries a band of the source's scaled
  context into the target's inputs in the same moment, over `rounds` Jacobi
  rounds. `observe(xs, ys, rate=, backtrack=, write=)` returns a
  `JointObservation` (`updated`, `reason`, the `settled` path with `seam` and
  `settle` per moment and round, `delta` per cortex, the losses, `writes`);
  `imagine(xs, states=)`, `advance(xs)`, `reset`, `parameters`, `snapshot`,
  `restore`, `clone`, `save`, `load`; `cut = True` zeroes every port.
  `cadence.record_ports.build(hidden, own_inputs, outputs, ports, *, seed, ...)`
  grows the patches at the widths the ports need.
- `observe(inputs, target, *, rate=1.0, backtrack=False, write=True)`
  predicts with the records at the start of the call, moves the slow
  parameters against the adjoint gradient of the precision-weighted half
  mean squared error of the slow readout `C h + c` (admitted by causal
  replay when `backtrack=True`) and writes the residual `target - C h - c`
  into each reading's records. Returns `RecordObservation`: `updated`,
  `reason`, the `prediction` (`RecordPath` with `hidden`, `output`, `gate`,
  `read`, `loss` of the prediction and `slow_loss` of the slow readout),
  `delta`, the slow readout's admission losses and rates, replay count and
  write count.
- `imagine(inputs, *, state=None)` is private; `advance(inputs)` carries
  context; `reset()` clears context and keeps parameters and records.
- `dream(inputs)` is the store's completion of a cue from rest, as a target (the chosen
  category per group for categorical ports); `sleep(cues, *, passes=1, rate=1.0,
  backtrack=False, dawn_passes=2)` dreams every cue once, teaches the slow weights the
  fixed dreams by `observe(write=False)`, and at dawn writes the dreams back so the store
  holds only what the slow weights did not take. Returns admitted updates, the mean slow
  loss on the dreams before and after, and the dawn writes.
- `detune(inputs, target, *, beta=1e-3, state=None, tolerance=1e-14,
  max_iterations=10000)` solves both detuned equilibria of the quadratic
  energy by conjugate gradients and returns `RecordContrast`: the centered
  `contrast`, both hidden paths, energies, iterations, residuals and
  `converged`. It changes nothing.
- `parameters()` and `set_parameters(mapping)` cover `G`, `g`, `B`, `b`, `C`
  and `c`; `records` is the `Records` store. `readback()` returns
  `RecordReadback`: `state`, `updates`, `writes`, `parameter_revision`,
  `state_parameter_revision` and `record_entries`, read-only and never
  admitted as teaching data by itself. `snapshot()`, `restore(snapshot)`,
  `save(path)` and `load(path)` carry parameters, records, counts and live
  context.

## Ports (`cadence.ports`)

- `MapBlock(start, channels_in, height, width, channels_out, kernel, stride=1)`: a tied local
  kernel over a grid of the inputs; `DenseBlock(start, inputs, outputs)`: a matrix over a slice.
- `StructuredPort(inputs, blocks, broadcast=None, mask=None)`: `apply(u, weights)`,
  `transpose(v, weights)`, `gradient(v, u)`, `initial(rng, scale)`, `weight_shape(block)`,
  `dense_matrix(weights)`, `to_dict()`, `from_dict(d)`. `broadcast=(start, count)` tiles that
  slice into every map block as constant channels. `mask` `(inputs,)` names which inputs the
  port hears; a masked input is zero to every block in all three maps and the mask travels
  with `to_dict`.
- `RecordPatchNet(..., port=StructuredPort)` and `RecordPatchStack(..., lower_port=StructuredPort)`
  read their inputs through the port; `hidden` (or `lower`) equals the port's outputs.

## BeliefPatch (`cadence.belief`)

See the [belief patch guide](belief.md).

- `BeliefPatch(observation: StructuredPort, actions, belief, outputs, *, iterations=2, damping=0.5,
  cells=4096, active=32, record_rate=0.5, record_width=64, habituation=1e-5, record_bias=0.3,
  output_precision=None, seed=0)`. `block_count` is the number of the port's blocks.
- `assimilate(observations, actions, observed=None, *, state=None, gains=None, probe=False,
  keep_live=False) -> BeliefPath`: advance the belief through observed moments; nothing
  learned or written. `imagine(actions, *, state=None, gains=None) -> BeliefPath`: the
  transition alone under declared actions, private. `observe(observations, actions,
  target=None, *, observed=None, rate=1.0, write=True, state=None, gains=None,
  loss_weight=None, output_gradient=None, backtrack=None, probe=False, keep_live=False)
  -> BeliefObservation`: one backward scan and the store's writes. `state` starts the
  moments from a given boundary instead of the live belief; the final belief becomes the
  live state unless `keep_live=True`. `observed` masks moments `(time,)` or rows
  `(batch, time)`; a row that observes nothing keeps its expectation. `gains` `(blocks,)`,
  `(batch, blocks)` or `(batch, time, blocks)` multiplies each block's encoded evidence before
  the repair map and the store read see it. `loss_weight` `(time,)` or `(batch, time)` weighs
  each moment's error, normalized by its sum; a moment of weight zero is neither taught nor
  written. `output_gradient` `(batch, time, outputs)` replaces `target`: the adjoint of an
  external loss on the outputs; nothing is written and no loss is reported. A step on a
  target is admitted (`backtrack` None or True): the largest halving of the start whose
  replay of the chunk from the same boundary, with the store as it stands, lowers the loss
  by the Armijo margin (sixteen halvings at most), the start being twice the last admitted
  step and at most `rate`; `backtrack=False` is the plain step at `rate`, and a step on an
  external gradient is plain. `probe=True` computes the residual-alone probe per block.
- `readback(observations, actions, *, state=None) -> BeliefReadback`: one moment
  `(batch, inputs)`, `(batch, actions)` before its repair, from the live belief or `state`:
  `expectation` `(batch, belief)`, `residual_alone` and `surprise` `(batch, blocks)`,
  `evidence` `(batch, encoded)` before any gain. `encode(observations) -> (..., encoded)`:
  the encoded evidence of any reading before any gain. Both change nothing.
- `step_size`: the last admitted step, None before one; the next admission starts from
  twice it. `reset_step()` drops it; `reset()` keeps it; it travels with the snapshot.
- `macs_per_moment(*, probes=False) -> int`: the multiply-accumulates of one moment.
  `cost`: `{"moments", "macs", "replays"}` counted since `reset_cost()`, every moment
  assimilated, observed, imagined or replayed in an admission.
- `set_implied_reading(implied, units=None)`: declares the map from the outputs
  `(batch, outputs)` to the reading each block should give `(batch, inputs)`, channels left
  `NaN` not compared, and the persistence error of each block's compared channels `(blocks,)`;
  paths then carry `surprise`. `None` withdraws it. Not part of a snapshot.
- `BeliefPath`: `belief`, `expectation`, `residual`, `step` (the last repair move per unit, whose
  norm is `residual`), `output`, `read`, `loss`, `slow_output`, `final_state`; `evidence`
  `(batch, time, encoded)`, the encoded evidence after the gains; `code` `(batch, time, cells)`,
  the store's plain code at the final reading; `gains` `(batch, time, blocks)`, the gains used;
  `residual_alone` `(batch, time, blocks)`, the repair map's move with one block heard and the
  store read at zero (with `probe=True`); `surprise` `(batch, time, blocks)`, each block's
  reading against the reading the previous belief's slow readout implies, in persistence
  units (with an implied reading declared).
- `BeliefObservation`: `updated`, `reason`, `path`, `delta`, `initial_loss`, `writes`,
  `final_loss` (the replayed loss under the admitted parameters), `accepted_rate` (the rate of
  the step taken, None without a step), `replay_calls`, `gain_gradient` `(batch, time, blocks)`.
- `reset()` (the live state; the step size and the counters stay), `state`, `parameters()`,
  `set_parameters()`, `set_output_precision()`, `records`, `snapshot()`, `restore()`,
  `save()`, `load()`.
- `cadence.belief_torch.TorchBelief(port, actions, belief, outputs, *, iterations, damping, record_width)`:
  the slow half on torch; `forward(observations | None, actions, state=None, reads=None,
  gains=None, observed=None)`, `export()`, `load(params)`. A gains tensor that requires grad
  receives the gradient into the gains.

## TemporalMemory (`cadence.temporal_memory`)

- `TemporalMemory(*, relative_tolerance=1e-12)` creates explicit local response
  constraints. `protect(net, inputs, *, state=None)` admits the current free
  response of a caller-selected query, without targets or network mutation,
  and returns `ConstraintReport`: `ranks`, `bytes` and `maximum_residual`.
- `observe(net, inputs, target, *, beta=0.01, rate=0.1,
  readout_damping=None, symmetry_tolerance=0.15, max_halvings=8)` stages
  protected learning atomically. Positive finite
  readout damping selects the local metric and causal acceptance check described
  in the [memory guide](temporal-memory.md); `None` retains ordinary projection.
  The symmetry settings pass through to `TemporalPatchNet.observe`.
- `report()` returns the same `ConstraintReport` for the current bases. `snapshot()` and
  `restore(snapshot)` preserve bases and parameter binding. Save the net and
  its memory together. Lower-level `project(before, proposed)` requires the
  caller's explicit transaction and is documented in the source.

## Experimental fixed connectivity (`cadence.experimental`)

This namespace is not exported at
the top level. `PartitionedTemporalPatchNet(inputs, hidden, outputs, *,
masks=None, **options)` accepts the temporal constructor options and boolean
`A`/`B`/`C` masks; omitted masks allow all entries. It preserves the existing
solver and masks parameter gradients before candidate admission.

`two_group_masks(context_hidden, motor_hidden, inputs, outputs, *,
context_inputs, motor_inputs, cross_coupling=True)` builds one supplied routing
pattern. The model's `masks` property returns copies and
`trainable_parameter_count` counts permitted entries, not allocated storage.
Restore with the subclass's `restore`/`load` to preserve mask enforcement.
**`TemporalMemory.observe` rejects this subclass before mutation.** See the
[experimental guide](partitioned.md) for checkpoint, planning and evidence scope.

## EquilibriumActor (`cadence.actor`)

`BodyModel` and `EquilibriumActor` provide fixed linear-body planning with a
Gaussian compressed past. `admit(position, *, identifier, executed_action=None)`
records an actual reading as an `ObservationRecord` (`identifier`, `position`);
identifiers enforce ordering, not authenticity. `plan(*, horizon=None,
goal=None)` privately proposes an action toward a supplied goal and returns
`ActorPlan`: the boundary, covariance, states, actions, readings, seams, cost
terms, residual, minimum pivot, message and coefficient bytes, block
factorizations, goal, model binding and the observation it starts from.
`readback()` returns `ActorReadback`: the last record, mean, covariance,
residual, minimum pivot, model binding, admitted count, marginalizations and
`numeric_persistent_bytes`.
`numeric_persistent_bytes()` counts the retained array, scalar, identifier and
hash payload, excluding Python objects and serialized archives; the guide states
that accounting and its scope. Their state, covariance, checkpoint and
fixed-model restrictions are distinct from `TemporalPatchNet`: see the complete
[actor guide](actor.md).

## PatchNet (`cadence.patch`)

`PatchNet` composes the existing `Brain` and `Learner` for ongoing continuous
observations. The [guide](patchnet.md) explains the equations, memory boundaries
and a complete runnable example.

- `PatchNet.create(inputs, hidden, outputs, *, seed=0, ...)` creates a reciprocal
  graph with declared input, hidden and output ports. Configure learning with
  `config=LearnerConfig(...)`, phase budgets with `steps` and `tolerance`, and
  optional temporal overlap with `context_strength` and `context_mask`.
- `stimulus(inputs, *, amplitude=1.0)` converts a batch of continuous input
  values into full neural drives. `settle(drive)` updates live free activity;
  `read(phase)` reads the output ports of an `Equilibrium` or `BrainState`.
- `observe(drive, target, *, observed=None, weight=None, source_id=None)` returns
  a `PatchObservation` with the free and nudged phases, `updated` and `reason`.
  Targets only enter the nudged phases. A required phase that misses the
  residual tolerance prevents the learning commit. The observation mask is
  shared across batch rows; nonnegative teaching weights are per row.
- `imagine(drives, *, state=None)` returns consecutive free equilibria on a
  private branch without modifying live activity, parameters or evidence IDs.
- `reset()` clears activity while preserving learned parameters, optimizer
  history and the optional bounded evidence-ID window. `state` and `snapshot()`
  expose detached state for inspection.
- `save(path, *, compressed=True)` and `PatchNet.load(path, ...)` preserve
  continuation parameters, optimizer history, current activity and configuration.
  Checkpoint correctness does not establish retention during new learning.

## Connectome (`cadence.connectome`)

- `Connectome.from_synapses(n, *, pre, post, count=None, sign=None, populations=None, label="connectome", min_count=0.0)`:
  build from synapse lists. Parallel synapses merge (counts add), autapses drop, synapses
  below `min_count` drop, and the arrays are sorted by `(post, pre)`. `count` (synaptic
  contacts per synapse) defaults to 1 and `sign` to +1.
- Fields: `n`, `pre`, `post`, `count`, `sign` (arrays), `populations` (name → tuple of
  neurons), `label`. Property `synapses` (the number of synapses).
- `members(*names)`, `with_populations(**populations)`, `in_degree()`, `out_degree()`,
  `summary()`, `digest()` (SHA-256 of the sorted arrays).

## Neuron model (`cadence.neuron`)

- `NeuronModel(dt=0.2, slope=4.0, threshold=1.5, gain=0.02, stimulus_amplitude=3.0, adaptation=None, leak=0.0)`:
  the graded (rate) neuron. `activation(v)` is the rectified, re-based sigmoid with optional leak;
  `slope_at(v)` its derivative; `rest_emission` the raw sigmoid's value at rest, which is
  subtracted so rest publishes zero. `replace(**changes)`, `to_dict()`.
- `Adaptation(tau_steps=50.0, strength=1.0)`: a slow per-neuron variable that can
  produce rhythm in suitable circuits.
- `learning_neuron_model(gain=1.0, *, slope=1.0, leak=0.1, dt=0.5, stimulus_amplitude=1.0)` (in
  `cadence.learning`): responsive starting settings for local learning, to validate for your task.

## Brain (`cadence.brain`)

- `Brain(connectome, neuron_model, *, backend="cpu" | "torch" | "mlx", efficacy=None, log_gain=None, bias=None, device=None, dense_limit=2048, layout=None, precision=None)`:
  the runnable brain. `efficacy` (the learned synaptic efficacy) defaults to the connectome's
  signs; `log_gain` and `bias` to zero. Attributes include `connectome`, `neuron_model`,
  `efficacy`, `log_gain` and `bias`. When the connectome's dense blocks fit in `dense_limit`
  squared entries the transport is the block transport (see `cadence.blocks`); `layout`
  passes a precomputed cut, as `with_parameters` does. `brain.layout` is the cut in use.
  `precision` (torch only) is `"float32"` or `"float64"`; the default is float64 except on
  MPS. Float32 is the speed of a consumer GPU; compare it with float64 and record the
  measured precision error.
- `settle(stimulus=None, *, steps=60, state=None, mask=None, trajectory=False, nudge=None, tolerance=None) -> BrainState`:
  one stimulus; `stimulus` is a list of neurons at full amplitude, a `{neuron: level}` map, or a
  dense vector. Map values are levels multiplied by `stimulus_amplitude`; dense vectors
  are drives directly. `settle_batch(drive, ...)` takes `(batch, n)` drives.
  `mask` accepts `(n,)`, `(1, n)`, or `(batch, n)`, with finite values in `[0, 1]`; zero suppresses a neuron.
  With a positive `tolerance`, both methods stop once every activation moves less than
  that amount in a step. `None` or zero uses the full step cap. Nonfinite drives and warm
  potentials/adaptation are rejected. Use floating arrays for dense drives and maps for
  selected indices: the legacy integer vector of length `n` containing only 0/1 is a drive.
- `equilibrate(drive, *, budget=512, chunk=32, tolerance=1e-5, state=None, mask=None, nudge=None) -> Equilibrium`:
  seek a joint state whose equation residual is below tolerance, checking after each chunk.
  `budget` caps additional settling steps exactly, including a short final chunk;
  zero checks the starting state. Each check uses one transport; unread float64 Torch
  states stay on their device and return one scalar per row. The returned
  `Equilibrium` has `state`, per-row `residual`, total `steps`, `tolerance`, and a boolean
  per-row `converged` property. `state.steps` is the last chunk's count. Convergence here
  does not prove stability, uniqueness or task quality.
- `residual(drive, state, *, nudge=None, mask=None, on_device=True)`: per-row maximum remaining
  fixed-point equation discrepancy, including adaptation when enabled. One transport
  evaluation, no state change. Unread float64 Torch states use the resident kernel;
  float32 states and other backends use the float64 CPU reference. `on_device=False`
  selects that reference explicitly. It is a diagnostic and does not prove stability or uniqueness.
- `ep_structure(brain, *, fixed_inputs=(), tolerance=1e-12) -> EPStructure`: reports the
  maximum asymmetry of effective free/free weights, incoming mass on excluded source
  neurons, and adaptation. Its `compatible` flag covers structure only: inspect phase
  residuals, unchanged unnudged inputs, smoothness, stability, finite-beta bias, and
  parameter/loss units separately. See [the certificate guide](certificate.md#equilibrium-propagation-scope).
- `stimulus_vector(stimulus)`, `stimulus_levels(levels)` (finite levels times the stimulus
  amplitude, with signed values allowed), `readings(state, names, i=0, level=0.5)`, `with_parameters(*, efficacy=None, log_gain=None, bias=None)`,
  `weights` (effective drive per synapse), `dense()` (the `W[pre, post]` matrix), `to_dict()`.
- `Brain.contrast_on_device(plus, minus)`: the learning rule's per-synapse and per-neuron
  contrast computed on the device when both states carry its handle; `None` otherwise.
- `BrainState`: `v`, `activation`, `adaptation`, `steps`, `trajectory`, `activity_change`
  (the total movement of the activations while settling, per row), `device` (the same
  state on the accelerator that produced it, or `None`). State arrays have shape `(n,)`
  from `settle` or `(batch, n)` from `settle_batch`; trajectories add a leading step axis.
  `row(i)`, `mean(members, i)`, `fraction_active(members, level, i)`, `active(level, i)`,
  `batched`.
- `Nudge(target, mask, beta, softmax_temperature=None, weight=None, groups=None)`: extra drive
  `beta · (target − s)` on the masked neurons, or `beta · (target − softmax(s/T))` over the
  masked group with a temperature; `weight` scales rows. `groups` assigns a separate
  softmax group per neuron (`-1` excludes a neuron). `drive(s)`.
- `available_backends()`: `{"cpu": "numpy float64", "torch": "mps float32" | "cuda float64" | "cpu float64", "mlx": "gpu float32"}`, for what is installed.

## Blocks (`cadence.blocks`)

- `layout(connectome, *, max_pairs=256) -> Layout`: cut the neurons into contiguous ranges at
  the boundaries of the connectome's contiguous populations and pair the ranges that carry
  synapses. A connectome with no contiguous populations, or one that fragments into more
  than `max_pairs` blocks, gets one block, the full matrix.
- `Layout`: `starts`, `pair_pre`, `pair_post`, `offset`, `edge_index`; `ranges`, `pairs`,
  `size`, `bounds(k)`, `sources()` (ranges that receive no synapses), `flat(weights)`,
  `blocks(flat)`, `to_dict()`.
- `BlockTransport(layout, flat)`: `synaptic_input(s)` returns the synaptic input for the
  activations `s`, reusing the product of every source range that did not move since the
  previous call.
- `block_contrast(layout, s_plus, s_minus)`: the learning rule's per-synapse contrast as one
  Gram product per block.

## Streams (`cadence.stream`)

- `stateful(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0, context_init=1.0) -> (Connectome, tie_groups)`:
  `embedded` plus a `context` range of `hidden` neurons that receive no synapses and reach
  every hidden neuron. Populations `input`, `context`, `embedding`, `hidden`, `output`.
- `Trace(connectome, decay=0.5, amplitude=1.0, focus=0.0, source="hidden", target="context")`:
  the memory of the moment before, per stream. It keeps a trace of the `source` range's
  activation and adds it to the next settling as a stimulus on the paired `target` range
  (one neuron per source neuron). `focus` above zero weights each neuron by its movement
  since the last moment (its share of the row's mean movement, to that power), so the trace
  is brightest where the moment changed. `reset(batch, rows=None)`, `stimulate(drive)`,
  `update(state)`, `keep(rows)`, `ringing(floor=0.1)` (each source neuron's share of what is
  still ringing, one elsewhere; a salience for `ActorCritic.salience`), `to_dict()`.
- `Echo(connectome, decay=0.5, amplitude=1.0, ...)`: a `Trace` with the same defaults, used at
  focus 0 into the `context` range (the carried state of the earlier moments).
- `Afterglow(connectome, decay=0.5, amplitude=1.0, focus=1.0, source="hidden", target="afterglow")`:
  the focused `Trace`. With `source="input"` it is an afterimage of the picture itself, the
  memory that reads a cue against a static background (`tests/test_child.py`: 1.00 where the
  Echo reads chance).
- `FastSynapses(pre, post, decay=1.0, rate=1.0, amplitude=1.0, normalize=False, replace=False, rule="hebb", separator=None, writes=0)`:
  one mutable `(pre, post)` matrix per stream; `writes` counts the rows written. `separator`
  is a `PatternSeparator` applied to keys and queries, which makes the matrix
  `(expansion, post)` per stream. `rule="delta"`
  uses unit keys and writes `rate * outer(key, value - prediction)`; it rejects
  `normalize`/`replace` and rates outside `[0, 1]`. The default `rule="hebb"` is additive
  Hebbian memory, with optional count-averaged reads (`normalize`) and one-hot row
  replacement (`replace`).
  `observe(key, value, write=None)` uses `(batch, width)` ports and writes all rows unless
  given a boolean mask; `recall(key)` reads without decay. `update(state, write=None, post=None)`
  reads named neuron activations instead and defaults to decay-only; `read(drive)` reads
  key neurons' drive columns. `stimulate(drive, inplace=False)` adds the read into post columns.
  `reset(batch, rows=None)`, `keep(rows)`, `to_dict()`. See [memory](memory.md) for stream
  identity, representation alignment, key interference, and checkpoint boundaries.
- `cadence.stream.columns(index)`: a slice when the neurons are one contiguous range,
  else the index array. Slices can avoid the copies required by advanced indexing.
- `PatternSeparator(inputs, expansion, winners, seed=0, center=0.0)`: pattern separation of
  the keys of a `FastSynapses` or `SynapticMemory`. `projection` is a fixed
  `(inputs, expansion)` Gaussian matrix from `numpy.random.default_rng(seed)`, divided by
  `sqrt(inputs)`; `mean` is the running key mean. `code(key, learn=False) -> (batch, expansion)`:
  with `center` above zero, `mean` is subtracted first, and `learn=True` first moves `mean`
  by the forgetting factor `center` toward the batch mean of the keys (writes learn, reads
  do not); the code keeps the `winners` largest entries of the rectified `key @ projection`
  and sets the rest to zero. `habituate(keys)` sets `mean` to the mean of a nonempty sample
  of keys. `to_dict()`; attributes `inputs`, `expansion`, `winners`, `seed`, `center`,
  `projection` and `mean`. Raises `ValueError` for nonpositive sizes, `winners` above
  `expansion`, `center` outside `[0, 1)`, and keys whose code overflows.
- `SynapticMemory(pre, post, decay=0.9, rate=1.0, amplitude=1.0, normalize=False, replace=False, rule="delta", separator=None, writes=0, consolidation=0.05)`:
  normalized delta synapses with a shared persistent `consolidated` matrix and per-stream
  effective `strength` matrices. `observe(key, value, write=None, *, salience=None,
  value_mask=None)` consolidates only observed values. Salience is a finite nonnegative
  `(batch,)` vector; the observed-value mask is boolean with the values' shape.
  `reset(batch, rows=None)` clears transient residuals and retains persistent synapses,
  including across batch changes. `clear()` erases both. Inherits `recall`, `read`,
  `stimulate`, `update` and `keep`; its rule is always delta. Reads never learn. A
  `separator` must have `center=0`.
  See [the equations and lifecycle](continuous.md#repetition-and-salience-become-lasting-synaptic-changes).

## Records (`cadence.records`)

- `Records(inputs, fields, *, cells=8000, active=40, rate=0.2, valued=(), valued_rate=1.0, habituation=1e-5, bias=0.3, pathways=(), pathway_rate=0.002, tasks=(), fan_in=0, seed=0, averaging=False, homeostasis=0.0)`:
  the records cortex over readings of `inputs` units. `fields` maps each predicted field to
  its width; of `cells` code cells, `active` stay per reading. `rate` is the write rate of
  the consequence fields (with `averaging`, each cell's rate is the larger of `rate` and one
  over the code mass written into it, so a fresh cell takes its first outcome whole);
  `homeostasis` is the rate at which each cell's activation share is tracked and its offset
  moved toward `active / cells` (0 leaves the offsets fixed); `valued` names the fields that read and write through the valued
  code, at `valued_rate`. `habituation` is the slowest rate of each unit's running mean (0
  subtracts nothing). `pathways` are one-dimensional index arrays into the reading whose
  running norms, moved at `pathway_rate`, equalise their say in the valued code; empty
  pathways are dropped. `tasks` indexes the reading's task units: the cells are divided into
  one group per task unit, and the valued code of a reading draws its winners from the group
  of the unit with the largest value, from every cell when no unit is positive. `fan_in`
  restricts each cell to that many pathways, drawn for the cell from the generator; the
  cell reads those and every input outside the pathways, and its projection column is
  rescaled by the square root of `inputs` over the inputs it reads (0 reads every input).
  `bias` scales the cells' fixed offsets. `seed` starts the `Mulberry32` generator that
  draws the projection, then the offsets, then the division into groups (a Fisher-Yates
  shuffle of the cells from `cells - 1` uniform draws), then the pathways of each cell in
  turn (a shuffle of the pathways from `len(pathways) - 1` draws per cell). Raises
  `ValueError` for a nonpositive `inputs`, `cells`, `active` or field width, `active` above
  `cells`, no fields, a valued name outside `fields`, `rate` or `valued_rate` outside
  `[0, 2]`, `habituation` or `pathway_rate` outside `[0, 1]`, a negative or nonfinite `bias`,
  a pathway or task index outside the reading, and fewer than `active` cells per task
  group. See [records](memory.md#records).
  - `code(readings, *, adapt=False, valued=True) -> ndarray`: the codes of `(batch, inputs)` readings, or
    of one `(inputs,)` reading, as `(2, batch, cells)`: the plain code, then the valued code.
    With `habituation` above zero, `adapt=True` first counts each reading in `seen` and moves
    `mean` toward it at the rate `max(habituation, 1 / seen)`, and every reading is coded
    with `mean` subtracted. A code keeps the `active` cells with the largest drive
    `reading @ projection + offset`, rectifies them and scales the row to unit length (a row
    with no positive drive stays zero). With `pathways`, the valued code divides each
    pathway of the mean-free reading by its running norm plus `1e-3`, and `adapt=True` first
    moves `pathway_norm` toward the norms of the readings' pathways; without pathways the
    valued code is the plain code. Witnessed readings adapt; imagined readings do not.
    Raises `ValueError` for a nonfinite reading or a wrong width.
  - `read(code) -> dict[str, ndarray]`: each field's read, the field's code times its table:
    `(width,)` for a `(2, cells)` code, `(batch, width)` for a `(2, batch, cells)` code.
    Valued fields read the valued code.
  - `write(code, targets, known=None) -> int`: the witnessed outcome of one reading, `code`
    of shape `(2, cells)`. Each field named in `targets`, a finite vector of its width, moves
    by `rate` (`valued_rate` for a valued field) times `outer(code, target - code @ table)`,
    through the active cells only; a field absent from `targets` is not written. `known`
    maps a field to a boolean mask of the observed target entries; the others have zero
    error. Returns the number of fields written and adds it to `writes`.
  - `parameters() -> int`: the record entries, `cells` times each field's width, summed.
  - `to_dict() -> dict`: the configuration (`inputs`, `fields`, `cells`, `active`, `rate`,
    `valued` as a sorted list, `valued_rate`, `habituation`, `bias`, `pathways` as lists,
    `pathway_rate`, `tasks` as a list, `fan_in`, `seed`); `Records(**records.to_dict())` rebuilds the
    same `projection`, `offset` and `task_of_cell`.
  - Attributes: `projection`, `(inputs, cells)` standard normal draws divided by
    `sqrt(inputs)`; `offset`, `(cells,)` standard normal draws times `bias`; `mean`, the
    `(inputs,)` running mean, zero at construction; `seen`, the witnessed readings counted
    into the mean; `pathway_norm`, one running norm per pathway, one at construction;
    `tables`, a dict from field name to its `(cells, width)` records, zero at construction;
    `writes`, the number of field writes; `tasks`, the task units, and `task_of_cell`, the
    `(cells,)` group of every cell (all zero without tasks). `mean`, `seen`, `pathway_norm`,
    `tables` and `writes` are the learned state. The configuration is readable as `inputs`, `fields`,
    `cells`, `active`, `rate`, `valued` (a frozenset), `valued_rate`, `habituation`, `bias`,
    `pathways`, `pathway_rate` and `seed` (reduced to 32 bits).
- `Mulberry32(seed)`: the 32-bit generator `mulberry` of `brain_scan.js`, so a page draws
  the same numbers from the same seed; `state` holds the 32-bit state, the seed reduced to
  32 bits at construction.
  - `random() -> float`: one uniform draw in `[0, 1)`.
  - `batch(n) -> ndarray`: `n` draws at once, equal to `n` calls of `random` and leaving the
    same `state`; raises `ValueError` for a negative `n`.
  - `normals(n) -> ndarray`: `n` standard normal draws by the Box-Muller transform of
    `batch(2 * ceil(n / 2))`: the first half of the uniform draws gives the radii and the
    second half the angles, and the cosine values precede the sine values before the cut
    to `n`.

## Regions (`cadence.regions`)

- `Region(name, size=0, circuit=None, inputs=None, outputs=None)`: a named group of neurons.
  A blank region has a `size` and no synapses of its own. A designed region has a `circuit`
  (a `Connectome`) and takes its size from it; `inputs` and `outputs` name the circuit
  populations that receive and send projections, and default to the whole region.
  `designed`, `neurons(population=None)` (region-local indices), `to_dict()`.
- `visual_cortex(height, width, *, channels=1, features=8, field=3, stride=1, init=1.0, seed=0, name="visual")`:
  population `input` with one neuron per pixel and channel, in the order of an image array
  `(height, width, channels)` flattened row by row, and population `output` with `features`
  feature maps. Each feature neuron receives synapses from one `field` by `field` window of
  the input; windows step by `stride`. Efficacies start random and fan-scaled.
- `cortex(size, *, lateral=0.0, name="association")`: a blank region, or with a negative
  `lateral` a designed one whose neurons inhibit each other pairwise.
- `motor_cortex(actions, *, lateral=0.0, name="motor")`: population `actions` with one neuron
  per action and optional pairwise lateral inhibition.
- `prefrontal_cortex(holds, *, name="prefrontal")`: a blank region with one neuron per neuron
  of `holds` (a `Region` or a size), for a `Trace` from the held region.

See [write a cortex](cortex.md) for regions, projections, ports and learning heads.

## Generic brain (`cadence.generic`)

- `GenericBrain.build(inputs, actions, *, hidden=64, density=1.0, lateral=-0.5, working_memory=False, memory_scale=12.0, episodic=True, features=8, field=3, seed=0, **options)`:
  develops `GenericBrain.genome(...)` and wraps it. `inputs` is a vector length, or an image
  shape `(height, width)` or `(height, width, channels)` for a `visual_cortex`. `options` go
  to the constructor.
- `GenericBrain.genome(inputs, actions, ...) -> Genome`: regions `sensory` (or `visual`),
  `association` (`hidden` neurons), `motor` (`motor_cortex(actions, lateral=lateral)`) and,
  with `working_memory`, `prefrontal`; projections sensory to association (reciprocal for a
  visual cortex), association to motor (reciprocal), and prefrontal to association at
  `memory_scale`.
- `GenericBrain(connectome, *, episodic=True, consolidation=0.05, working_memory_decay=0.2, working_memory_amplitude=3.0, learning=None, reward=None, seed=0, backend="cpu", device=None)`:
  needs populations `sensory` or `visual/input`, `association` and `motor`, and uses
  `prefrontal` for a working memory when present. `learning` defaults to
  `LearnerConfig(beta=0.1, eta=0.5, temperature=0.2, tolerance=3e-3, nudged_steps=12, momentum=0.9)`,
  `reward` to `ActorCriticConfig(gamma=0.9, lam=0.8, eta=1.0, eta_critic=0.3)`.
  Attributes `connectome`, `brain`, `learner`, `basal_ganglia` (`ActorCritic` reading the
  association cortex), `working_memory` (`Trace` or `None`), `hippocampus` (`SynapticMemory`
  from sensory to motor neurons, or `None`; old checkpoints retain `FastSynapses`), `sensory_index`, `association_index`,
  `motor_index`.
  - `stimulus(observations, *, memory=True)`: the drive of a batch; with `memory`, the
    working memory and the hippocampal recall are added.
  - `step(observations, *, reward=None, done=None, teacher=None, salience=None, bootstrap=None)`:
    the ongoing interaction API, returning the next sampled actions. Reward/done concern
    the preceding action; teacher labels concern the current observation. Omitted reward
    means a real zero-reward transition. Wait for the outcome before calling again.
    There is one operating mode; no training/inference toggle is needed.
    The first call cannot receive past-action feedback.
    `last_learning` exposes the previous transition's report and demonstration count.
    Supplied salience controls memory consolidation; by default it is absolute reward.
  - `fit(observations, labels, *, epochs=30, batch=32) -> list[float]` (training accuracy per
    epoch), `predict(observations)`, `accuracy(observations, labels)`: independent samples,
    without memory. These operations do not switch modes. `fit` resets pending stream
    state before its updates; use `step` for a continuing life.
  - `act(observations, *, greedy=False) -> actions`: one row per stream; updates the working
    memory. `learn(reward, done, next_observations, *, bootstrap=None, salience=None) -> report`: the hippocampus records the
    reward of the chosen action for its situation, `done` rows reset their working memory,
    and the basal ganglia learn from dopamine. For truncated episodes `bootstrap` supplies
    the value of the old episode's final observation; `next_observations` holds the reset
    observation for ended rows. Reward and bootstrap are finite batch vectors; done is boolean.
  - `reset()` clears working state, action cache, eligibility and reward centering; hippocampal
    records and slow parameters are kept. `parameters()` counts actor/critic parameters
    and the shared consolidated memory matrix; per-stream state is additional storage.
  - `save(path) -> Path`, `GenericBrain.load(path, *, backend="cpu", device=None, precision=None)`:
    complete composition checkpoints, including both optimizers, critic, random state,
    stream traces, prepared state, both memory timescales and any action awaiting feedback.
    Format 2 also retains pending nudged states; format 1 still loads.
    The archive is replaced atomically. A learner-only checkpoint
    is rejected by `GenericBrain.load`; `Learner.load` can extract a learner from either.
  Observations must be a nonempty finite batch, with image dimensions flattened per row.
  `fit` rejects noninteger labels and mismatched batches before updating. It resets current
  action/working state but keeps episodic records. `brain` always returns the current
  `learner.brain`, including after learning. See [compose a brain](brain.md#genericbrain).

## Genome (`cadence.genome`)

- `Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, reciprocal=True)`:
  synapses from `pre` to a fraction `density` of `post`. An end is a region name, meaning
  the region's `outputs` (for `pre`) or `inputs` (for `post`), or `region/population`.
  `sign` is the mean sign (−1 all inhibitory, +1 all excitatory, 0 mixed); `scale`
  multiplies the fan-scaled magnitudes; `reciprocal` adds the reverse synapses with the
  same weights.
  `Genome(regions, projections, label="genome")`: a connectome before development; it
  checks that region names are unique and that every projection end names a region or one of
  its populations. `region(name)`, `to_dict()`, `Genome.from_dict(d, designed=None)` (a record keeps a
  designed region's circuit label and digest, and `designed` supplies the region by name).
- `develop(genome, seed=0) -> Connectome`: development, deterministic in the seed. Regions
  are laid out in order as contiguous populations named after them; a designed region adds
  its circuit's synapses and its populations as `region/population`; each projection is
  drawn between its ends.
- `cadence.genome.mutate(genome, rng, *, size_step=0.25, fixed=(), tied=())`: one offspring
  (module level; `evolve` uses it). Designed regions and regions in `fixed` keep their size;
  each `(leader, follower)` pair in `tied` keeps the follower the size of the leader.
- `genes(space, *, rate=1.0)` returns a mutation over a dict genome for a declared space of
  `log`, `linear`, `int` and `choice` genes; see [any genome](evolution.md#any-genome).
- `evolve(fitness, genome, *, generations=10, population=8, keep=2, seed=0, mapper=map, report=None, mutate=None, grow=None, **mutation) -> Lineage`:
  selection under `fitness(connectome, seed) -> float`. Generation 0 scores the genome and
  `population - 1` offspring; each later generation scores `population` offspring of the
  `keep` best genomes of the generation before. Each life develops its genome at the seed
  `seed + 1000 * generation + index` and passes that seed to `fitness`; scores must be
  finite. `mapper` runs a generation's lives (a process pool's `map` needs a picklable
  `fitness`); `report` is called with the lineage after every generation; the remaining
  keyword arguments go to `mutate`. See [evolve a brain](evolution.md).
- `cadence.genome.Lineage`: `generations` (one record per generation with `generation`,
  `best_fitness`, `mean_fitness`, and `best` as `Genome.to_dict()`), `best` (the best genome
  over all generations, the earliest on a tie) and `best_fitness`.

## Timing (`cadence.timing`)

- `latency(decide, *, repeats=1000, warmup=20)`: time `decide()` `repeats` times; the median,
  90th and 99th percentiles and maximum in microseconds, the mean, `jitter` (p99 over p50
  minus one), and the voluntary and involuntary context switches during the measurement.
- `environment()`: machine, cores, Python, thread limits, pinned cores (Linux), load average,
  library versions.

## Neuron-by-neuron reference (`cadence.reference`)

- `cadence.reference.settle_neuron_by_neuron(connectome, neuron_model, stimulus, *, steps, log_gain=None, bias=None, efficacy=None) -> (trajectory, Ledger)` (module level; `conformance` uses it):
  one neuron at a time, each reading only its own state and its synaptic input, counting one
  transmission per declared synapse per step.
- `conformance(brain, stimulus, *, steps=60) -> dict`: the brain against the reference on
  the same stimulus; `max_abs_deviation`, the ledger, the backend.
- `cadence.reference.Ledger`: `declared_synapses`, `steps`, `transmissions`, `undeclared`;
  property `clean`, also reported by `to_dict()`.

## Protocols (`cadence.protocol`)

- `Row(id, stimulus, readout, predicate, reference="", ablate=(), relative_to="", tier="experiment")`.
- `Protocol(stimuli, rows, training=(), levels=Levels(), steps=60)`: `score(brain)`,
  `neurons_for(connectome, stimulus)`, `to_dict()`. `stimuli` maps each stimulus name to the
  populations it drives at full amplitude.
- `cadence.protocol.Levels(active=0.5, inactive=0.2, margin=0.15, sparse_min=0.005, sparse_max=0.2, densify_margin=0.05, specific_max=0.25, code_level=0.5)` (module level).
- `Row(id, stimulus, readout, predicate, reference="", ablate=(), relative_to="", tier="experiment", versus="")`: `versus` names the second stimulus a `specific` row holds the readout's code apart from; a training fact with a fourth element does the same.
- `cadence.protocol.shared_code(activation, other, members, level) -> float`: what the two codes (the `members` at or above `level` under each activation) share, as a share of their union.
- `evaluate_predicate(predicate, value, reference, levels=None) -> bool`; `PREDICATES`
  maps each name to its definition.
- `shuffled(connectome, seed, *, keep=None) -> Connectome`: the control.
- `select_gain(make_brain, protocol, grid, *, sparsity_cap=0.05) -> (gain, table)`:
  among admissible candidates, maximize training facts passed and break ties by smallest
  value. `make_brain` builds the brain for a candidate, so the candidate can be the global gain
  or any number a dictionary declares, such as one population's gain through
  `Brain(log_gain=...)`. `sparsity_cap=None` admits every candidate. An empty grid or no
  admissible candidate raises `ValueError`.

## Checkpoints (`cadence.checkpoint`)

- `save(learner, path, *, compressed=True) -> Path` and `load(path, *, backend=None, device=None, config=None, precision=None) -> Learner`,
  also as `Learner.save(path, *, compressed=True)` and `Learner.load(path, ...)`: one `.npz` file holding the
  connectome, every synapse's efficacy, every neuron's gain and bias, the neuron model, the
  configuration, the outputs and slots, the plasticity masks, tie groups, synapse rates,
  momentum and normalisation state, and the update count. `compressed=False` writes an
  uncompressed archive, faster for a very large brain. `backend` and `device` may differ from the
  saved ones; `precision` overrides saved precision and `config` replaces the saved
  configuration. `predict` and `free` read the parameters and update nothing.
  Separate `FastSynapses`, `Trace` and `ActorCritic` objects are not saved by this API.
  Use `GenericBrain.save/load` for the full standard composition. Archive replacement is
  atomic, so a failed write leaves the previous checkpoint intact.

## Learning (`cadence.learning`)

- `LearnerConfig(beta=0.1, eta=0.2, eta_bias=0.02, centered=True, free_steps=100, nudged_steps=50, tolerance=1e-4, nudge="cross_entropy", temperature=0.2, normalize=0.0, normalize_floor=1e-3, momentum=0.0, decay=0.0, scale_cap=8.0)`:
  `scale_cap` is the magnitude a plastic synapse's efficacy may not exceed (every update clips
  to it); a smaller cap keeps a readout neuron out of saturation, where a nudge has no slope
  ([the latch](reward.md#traps-with-their-measurements)).
  `momentum` steps each synapse on a running average of its own contrast; `decay` shrinks every
  plastic synapse's efficacy and every plastic neuron's bias by that fraction on each update
  (a leak on the synapses, for streams).
- `Learner(brain, outputs, config=LearnerConfig(), plastic_synapses=None, plastic_neurons=None, reciprocal=True, tie_groups=None, synapse_rate=None, slots=1, updates=0, contrast_updates=0)`:
  `plastic_synapses` and `plastic_neurons` are bool masks over synapses and neurons; only those
  move and decay, so two learners can share one brain without one's decay eroding the other's
  synapses. With `reciprocal`, each reciprocal synapse pair shares one efficacy.
  `tie_groups` is an int per synapse (−1 for none); synapses in a group share one efficacy and
  move by the mean of their contrasts, which is how an embedding is shared across positions.
  `synapse_rate` is a nonnegative float per synapse that multiplies its step before tying
  (`None`: every synapse at one).
  Overlapping reciprocal/explicit ties form one group; arbitrary group IDs are compacted.
  Ties constrain increments, so initialize tied values equally to keep them equal. Frozen
  members keep their values, including under decay and clipping, and contribute zero to
  the group's mean increment. Reciprocal learning rejects ambiguous parallel pairs;
  merge those with `from_synapses`, or use `reciprocal=False`. Masks remain mutable;
  changing the original boolean array affects later updates. Rebuild the learner to
  change tie topology.
  `slots` splits the outputs into softmax groups (a count of equal groups, or one size per
  group); `updates` counts all applied updates, while `contrast_updates` counts only
  this learner's own optimizer history, excluding external reward/direct updates.
  - `free(drive, warm=None)`, `nudged(drive, free, target, sign=1.0, weight=None)`,
    `targets(labels)`, `nudge_for(target, beta, weight=None)`;
  - `contrast(free, nudged, opposite=None) -> (per_synapse, per_neuron)`,
    `contrast_rows(free, nudged, opposite=None)` (the same differences per batch row),
    `update(free, nudged, opposite=None) -> {"scale_step", "bias_step"}`,
    `apply(delta_scale, delta_bias) -> {"scale_step", "bias_step"}` (a computed step through
    the masks, synapse rates, tying, decay and clipping),
    `step(drive, labels, warm=None, weight=None) -> (LearnedState, report)`;
    labels are integer indices within each output group: `(batch,)` for one group,
    `(batch, slots)` for several; `accuracy` averages all row/slot choices;
  - `calibrate(drive, *, level=0.5, grid=None)`, `predict(drive)`, `accuracy(drive, labels, batch=256)`,
    `parameters()`, `to_dict()`; attributes `brain`, `reverse` (index of each synapse's
    reverse, or −1), `second_moment` (when normalising).
- `calibrate_bias(brain, drives, targets, *, per_neuron=False, rounds=3, span=(-6.0, 6.0), iterations=16, steps=100, tolerance=1e-4) -> np.ndarray`:
  biases that bring populations to declared activity targets under a set of drives (rows as
  `settle_batch` takes them). `targets` maps a population name or neuron indices to the mean
  activation it should have over its members and the drives; one shared bias per population,
  or one per member with `per_neuron=True` (a readout's cells). Bisection, every population in
  turn, `rounds` times; the brain is unchanged and the array is passed to `Brain(bias=...)`.
- `naive_efficacy(connectome, plastic) -> np.ndarray`: efficacies that give every plastic
  synapse class the same weight (sign times mean count over the class's count); the other
  synapses keep their sign. For a lesson that should start naive at a memory site whose counts
  are a specimen's memories.
- `preflight(brain, outputs, plastic, drives, *, level=0.5, saturation=0.02, shared_max=0.25, eligibility_min=10, steps=100, tolerance=1e-4) -> dict`:
  the checks a lesson needs before its first decision, under the drives the brain will decide
  in: a readout with no slope (within `saturation` of 0 or 1 under every drive), the plastic
  senders' code shared between two drives beyond `shared_max`, fewer than `eligibility_min`
  plastic synapses from active senders onto a readout, two drives that drive the same neurons
  beyond `shared_max` (the senses, not the wiring, are to be told apart). Returns the readings and `warnings`, one
  sentence per finding naming the block that repairs it; an empty list is what a receipt shows.
- `seam_report(connectome, pre, post) -> dict`: `classes`, `synapses`, `coverage_pre` (the
  fraction of pre neurons with a class onto the post population), `classes_per_post`,
  `median_count`; what custody left of a plastic seam.
- `layered(inputs, hidden, outputs, *, density=0.3, feedback=1.0, lateral=0.0, seed=0, count=1.0, init=1.0, skip=False, skip_init=None, excitatory_forward=False) -> Connectome`
  with populations `input`, `hidden`, `output`. `skip_init=None` uses `init` for
  direct input-to-output projections. A finite nonnegative value overrides that
  scale; `skip=True, skip_init=0.0` adds trainable zero-efficacy projections while
  preserving the other effective parameters and initial predictions.
- `embedded(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0) -> (Connectome, tie_groups)`:
  a window of one-hot tokens through one embedding table shared across positions, then a
  dense hidden layer and the outputs, feedback synapses tied in pairs; populations `input`,
  `embedding`, `hidden`, `output`.
- `LearnedState(free, nudged, opposite=None)`.

## The agent and the valence (`cadence.plasticity`)

- `ActorCritic(learner, critic, config=None, seed=0, population=None)`: the agent of a stream
  of moments, the composition of the elements (`reward.md`). `learner` is a `Learner` whose
  outputs are the action neurons; `critic` the neurons whose settled activation, read through a
  learned linear readout, is the expectation of the reward to come; `population` a `Bins` for
  a continuous action.
  - `act(drive, greedy=False) -> action`: one free phase, then a draw from the softmax
    over the output neurons (with `Bins`, one draw per dimension), or the most probable.
    Cache reuse requires the same drive and unchanged brain parameters; caller buffers
    are copied. After learning, the next action refreshes its warm state. A greedy action clears
    pending eligibility and cannot be followed by `learn`. Repeated `act` replaces the
    pending decision. `Bins` requires at least two levels per dimension. Without
    `Bins`, `Learner(slots=[2, 3])` returns two categorical action indices per row;
    padding is never sampled. Actor nudges differentiate the softmax policy,
    independently of the learner's imitation loss;
  - `learn(reward, done, next_drive, bootstrap=None, *, observed=None) -> report`: the prediction error
    `reward + gamma * V(next) - V(now)` made into the dopamine by the valence and written
    through every synapse's eligibility, the trace of the last act's contrast decaying by
    `gamma * lam` a moment. The critic uses its own trace and `critic_signal`: raw
    prediction error (`"td"`) or modulated error (`"modulated"`); the default `"auto"`
    is `"td"` whenever the dopamine is centred and `"modulated"` otherwise
    (`ActorCriticConfig.critic_target` is the resolved choice).
    Reports include absolute raw `td_error`, absolute modulated `delta`, signed
    `dopamine`, `saturation` (the fraction of output activations within 0.02 of 0 or 1,
    where a nudge has no slope) and `trace` (the mean absolute eligibility over the plastic
    synapses); a `saturation` near 1 with a `trace` near 0 is the latch of
    [learning from reward](reward.md#traps-with-their-measurements).
    `done` rows start their next life from rest; a truncated row passes `value_of` its last
    observation as `bootstrap`;
    `observed` is a boolean batch vector for real transitions. Padding rows do not
    teach the actor or critic or enter reward statistics; their eligibility resets.
    Updates average over observed rows. At least one row must be observed.
  - `reset()` (cached input/state, eligibility, salience and centering cleared; learned
    parameters and optimizer history retained), `probabilities(state)` (shape `(batch, actions)`
    or `(batch, slots, max_size)` for categorical slots, with exact zero padding;
    `(batch, dims, size)` with `Bins`), `settle(drive)`,
    `value(state)`, `value_of(drive)`, `parameters()`, `to_dict()`; the attributes `valence`,
    `salience`, `delta_mean`, `delta_var`.
- `ActorCriticConfig(gamma=0.99, lam=0.9, eta=0.5, eta_bias=0.05, eta_critic=0.05, normalize=0.0, momentum=0.0, dopamine_cap=1.0, dopamine_center=0.0, dopamine_floor=0.0, center_scale=True, critic_normalize=True, critic_signal="auto")`:
  `gamma` the discount and `lam` the trace's decay; `eta` and `eta_bias` the actor's rates,
  `eta_critic` the critic's; `normalize` and `momentum` the adaptive local step, as the
  learner's; `dopamine_center` the rate at which the reward's running level and scale follow
  it (0 for no centring), `dopamine_floor` the band around the level, in scales, within
  which the dopamine is zero, `dopamine_cap` its cap, `center_scale` whether the surprise is
  measured in scales of the usual (`True`) or in the reward's own units; `critic_normalize`
  divides the critic's step by its trace's energy. `critic_signal="td"` keeps the
  critic target in reward units; `"modulated"` may change its fixed point through
  clipping or centring; `"auto"` (the default) is `"td"` when `dopamine_center > 0` and
  `"modulated"` otherwise, because a critic fed the centred signal chases a moving target
  (measured: a value running to -15 within 300 decisions on the fruit fly's T-maze). See
  [the choice and its measured tradeoff](reward.md).
- `Bins(dims, size=9)`: the population code for `dims` continuous dimensions, each a softmax
  over `size` bins (`centres`, `groups`, `read`, `size`).
- `Valence(level=0.0, floor=0.0, cap=1.0, units=True, per_stream=True, mean=0.0, var=1.0)`: the
  reward less its expectation, made into the dopamine. Called on `delta` (a prediction error,
  or the reward alone), it subtracts the running level (`level` is the forgetting factor; 0 for
  none), kept per stream with `per_stream` or shared across streams otherwise; `mean` and
  `var` hold that running level and variance. The result is in the reward's own units or over
  its running scale (`units`), zero within `floor` scales of the level (quiet while the reward
  is what it usually is), and capped at `cap` (0 for no cap). `reset()`. `ActorCritic.valence`
  is the agent's, built per stream from `dopamine_center`, `dopamine_floor` and `center_scale`
  of its config; the cap is `dopamine_cap`, applied by `learn`.
  Calling `valence(delta, observed=mask)` excludes unobserved rows from the
  running statistics and returns zero for them. An all-false mask leaves it unchanged.
- `ActorCritic.state`: the free phase of the latest moment, the state `act` read or `learn`
  settled; `None` after `reset`.
- `ActorCritic.salience`: `(batch, neurons)`, set before `learn`; each synapse's eligibility is
  weighted by its pre neuron's entry (a `Trace.ringing`), so that what is still ringing is
  what a signal writes through. None by default.

## Certificate (`cadence.certificate`)

- `certificate(brain) -> Certificate`: the settling certificate of the free phase, from
  `row_mass(brain)`, `lipschitz_constant(brain.neuron_model)`, the model's `dt` and whether
  it has adaptation. See [the certificate guide](certificate.md).
- `Certificate(row_mass, lipschitz, dt, adaptation)` (frozen): properties `rate`
  (`1 - dt * (1 - lipschitz * row_mass)`), `certified` (`rate < 1` and no adaptation) and
  `mass_limit` (`1 / lipschitz`). `error_bound(movement)` is the remaining sup-norm distance
  `movement / (1 - rate)` from the last step's potential movement, per row, and
  `apriori_bound(first_movement, steps)` is `first_movement * rate ** steps / (1 - rate)`;
  both return infinity for an uncertified brain. `steps_for(change, tolerance)` returns the
  warm-start steps after a stimulus change of sup-norm size `change` (zero for no change)
  and raises `ValueError` for an uncertified brain. `to_dict()`. Construction requires
  finite values, a nonnegative `row_mass`, a positive `lipschitz` and `dt` in `(0, 1]`.
- `row_mass(brain) -> float`: the largest absolute incoming effective weight sum over all
  neurons.
- `lipschitz_constant(model) -> float`: the supremum of the activation's slope,
  `(slope / 4) * max(1 / (1 - rest), leak / rest)` with `rest` the rest emission.
- `EPStructure` (frozen), returned by `ep_structure`: `fixed_inputs`, `free_neurons`,
  `free_asymmetry` (the largest absolute difference between a free/free effective weight and
  its reverse, parallel synapses summed), `fixed_incoming_mass` (the largest absolute
  incoming effective weight of an excluded input), `adaptation` and `tolerance`; property
  `compatible` (both measures within `tolerance` and no adaptation); `to_dict()`.

## Atlas (`cadence.atlas`)

- `build_atlas(connectome, weights=None, *, regions=None, shapes=None, positions=None, roles=None, seed=0, iterations=24) -> Atlas`:
  one layout of a whole connectome. `weights` are effective synaptic weights in connectome
  order (`brain.weights`); without them the contact counts stand in. `regions` partitions
  the neurons by name, by default the connectome's populations, coarsest first, each neuron
  in one region and the rest in `other`. `shapes` declares sheets `(rows, cols[, channels])`
  per region name and places them on a grid; `positions` supplies `(count, 2)` coordinates
  per region name, fitted into the region's place, or under `"*"` one `(n, 2)` frame for
  every neuron, scaled into the square; `roles` overrides the role a region's name
  suggests; `iterations` counts the neighbour-averaging passes that place the remaining
  neurons by their synapses. Deterministic under `seed`.
- `atlas_of(brain, **options) -> Atlas`: `build_atlas(brain.connectome, brain.weights, **options)`.
- `Atlas`: `n`, `positions` (`(n, 2)`, both axes in `[-1, 1]`), `region_index`, `regions`
  (layout records of `cadence.atlas.Region` with name, role, colour, centre, extent and
  size, distinct from the `cadence.regions.Region` of a genome), `pre`, `post`, `weight`,
  `seed` and `extras`; property `synapses`. `region_of(neuron)`; `summary()` (neurons,
  synapses, and each region's name, role and size); `to_dict()` and `to_json()` (the
  `cadence.atlas/v1` payload with base64 arrays); `subsample_edges(limit, seed=None)` (at
  most `limit` synapses, drawn with probability proportional to absolute weight);
  `frames(activation, potential=None)` (recorded `(steps, n)` settling steps quantised to
  eight bits per neuron and step); `frames_from_record(record, row=0)` (the frames of one
  batch row of a `SettlementRecord`);
  `page(*, frames=None, brain=None, title="Cadence brain scan", note=..., inputs=None, limit=2000) -> str`
  (a self-contained HTML page; `note` is the text under the title, `inputs` the neurons the
  live page's `Detune` drives, and `limit` the largest live brain, whose dense weight matrix
  the page embeds; a live brain with adaptation raises `ValueError`).
- `brain_scan_script() -> str`: the source of the shipped renderer `brain_scan.js`.
- `cadence.atlas.role_of(name, roles=None)`: the role a region name declares, by an explicit
  map or by its wording; `cadence.atlas.PALETTE` maps each role to its colour. See
  [the brain viewer](pages.md).

## Receipts (`cadence.receipts`)

- `Receipt.build(kind, body, sources=()) -> Receipt`; `write(path)`; `Receipt.read(path)`;
  `Receipt.verify(path, *, sources=None, check=None) -> (ok, message)`; `to_dict()`.
- `canonical_json(value)`; `cadence.receipts.canonical_sha256(value)` and `cadence.receipts.source_manifest(files)` at module level.

## Optional task compositions

`from cadence.circuits import assemble, reflex_arc, imagine, Deliberator, ActivityMonitor` imports small
sensorimotor, counterfactual-search and self-reading compositions. See
[defaults and the thinking clock](continuous.md#defaults-and-the-thinking-clock) for budgets
and scheduling. These optional architectural
helpers compose ordinary neuron dynamics with explicit host-side orchestration.

`assemble(regions, synapses=()) -> Connectome` merges an insertion-ordered mapping of
region names to connectomes into one connectome. Each entry of `synapses` is
`(source_region, local_neuron, target_region, local_neuron, weight)`, a directed synapse
with count 1 and sign `weight`. Each region remains addressable as a population, and each
of its populations as `region/population`. Use the result with one `Brain` and concatenate
drives in region insertion order. Unknown regions, neurons outside their region, autapses
and nonfinite weights are rejected. The helper adds topology only; convergence depends on
the combined system.


- `Deliberator(actions, transition, evaluate, terminal, *, depth=6, max_nodes=10000, adversarial=False, prune=False, clone=deepcopy)`:
  resumable iterative deepening over isolated futures. `start(live)` snapshots new input
  and replaces old work; `tick(nodes=128)` visits at most that many new positions and
  returns an isolated copy of the last completed `Deliberation`, or `None`.
  `pending` reports unfinished work; `pause()`/`resume()` preserve it; `cancel()` drops
  the continuation and obsolete result. `nodes` counts total work for this observation,
  `budget_exhausted` reports the hard limit, and `result` is the last completed depth.
  No work starts until the caller supplies input and ticks. Callbacks obey `imagine`'s
  isolation contract and remain fixed for one search; node limits do not preempt callbacks.
  This object stores no learned weights and writes no real-action feedback.
  See [defaults and scheduling](continuous.md#defaults-and-the-thinking-clock).
- `imagine(live, actions, transition, evaluate, terminal, *, depth=2, max_nodes=10000, adversarial=False, prune=False, clone=deepcopy) -> Deliberation`:
  compare copied futures. Scores use the root actor's perspective; adversarial layers
  alternate min/max. Pruning uses alpha-beta bounds with fresh bounds per root action.
  `clone` must isolate mutable branch state; other callbacks must not mutate external
  objects. Invalid budgets/nonfinite scores raise. Exhaustion raises before the next
  transition and returns no partial ranking. `Deliberation.futures` are sorted by score;
  each `Future` holds `action`, `score`, `sequence`, `state`. `nodes` counts visited
  successor states, and `depth` records the requested horizon.
- `ActivityMonitor().read(activity, scores, *, pressure=0) -> Readback`: finite 1D vectors
  and pressure in `[0, 1]`. Returns `activity_change`, `ambiguity`, `pressure`, `uncertainty`,
  `request_more`, `state`. `reset()` clears its previous reading and state. This heuristic
  does not learn by itself and does not establish consciousness.
- `reflex_arc(axes=2)`: sensory error ports and opposing motor pairs; the application
  supplies body dynamics and interprets the motor readout.

## Record every settling step

`record_settlements(callback, *, label="")` captures calls made inside its context,
including calls inside `Learner`, `ActorCritic` and supplied imagination routines.
The callback receives a `SettlementRecord` after each call. It contains the
connectome, neuron model, effective weights, bias, drive, mask, nudge and full
potential, activation and adaptation histories. Histories have shape
`(steps + 1, batch, neurons)`: the first row is the initial state, then one row
for every actual iteration. Differences of successive potentials are the signed
local repairs. Even a zero-step call has its initial row.

```python
records = []
with cd.record_settlements(records.append, label="observe and act"):
    result = brain.settle(stimulus={0: 1.0}, steps=32)
repairs = np.diff(records[0].potential, axis=0)
```

Recording is opt-in. It copies every neuron's state after every step and can be
expensive for large brains. Write bounded chunks from the callback rather than
keeping an entire long task in RAM. Callback diagnostics do not recursively
record themselves; nested recording contexts restore the previous callback on
exit. Callback failures propagate. State and parameter arrays own their storage;
subsequent learning cannot change those saved arrays. Treat the shared connectome
as read-only.

CPU recording uses the inspectable NumPy kernel rather than the fused path, so
round-off and wall time may differ. These are measurements of the recorded run.
Iteration traces are simulated neural activity.

`ActorCritic.learn` reports signed mean `dopamine` alongside the existing mean
absolute `delta`. For a one-stream agent it is that transition's signed,
centered and capped learning signal. It is a global modulation signal; spatial
neurotransmitter diffusion is not part of this model.


## Bounded memory and rehearsal

- `ContentMemory(inputs, outputs, capacity, match=0.75, key_rate=0.1, value_rate=1.0)`:
  `select(cue)` returns slots/scores; `recall(cue)` reads without mutation;
  `observe(cue, value, write=None)` learns from observed values; `clear()` erases
  the shared store. Novel cues allocate or evict a slot. See [content memory](content_memory.md).
- `ReservoirReplay(capacity, inputs, seed=0)`: `sample(count)` returns owned prior
  feature/label rows; `observe(features, labels)` admits actual observations into
  a uniform bounded reservoir. Learning and checkpointing are caller-owned.
  See [rehearsal](replay.md) for information and storage costs.
- `cadence.sequence.SequenceCache(features, values, *, capacity=128, temperature=0.1,
  center_rate=0.02)`: per-stream content readback; call `reset(batch)`, then
  `read(features)` before `observe(features, observed_values)`.
  `SequenceRead` exposes value, entropy, maximum weight and record count.
- `cadence.sequence.BoundedTrace(width, *, decay=0.5, radius=1.0, center=True)`:
  `reset(batch)`, `observe(value)` and non-mutating `read()`. Readback has at most
  the declared L2 radius; this is no guarantee of better sequence prediction.
  See [sequence readback](sequence.md).


## `cadence.population`

- `PopulationPatch(inputs, hidden, outputs, *, instances, streams, seed=0, cells=4096, active=32, record_rate=0.5, habituation=1e-5, record_bias=0.3, slowest=2.0, groups=None, device=None, dtype=torch.float32)`: record patches in lockstep on a device; `from_patch(net, *, instances, streams, device=None, dtype=torch.float32)`; `imagine(inputs) -> {output, slow, read, hidden, code}`; `observe(inputs, target, *, rate=1.0, write=True) -> {loss, slow, hidden, residual, code}`; `code(readings)`, `read(code)`, `clear_records()`, `inherit(parents, *, sigma=0.0, generator=None)`, `parameters()`, `state()`, `memory_bytes()`.
- `device_of(name=None)`: cuda, then mps, then cpu.
