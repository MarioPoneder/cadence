# Changelog

## Unreleased

- Add `PatchNet.recursive(inputs, layers, outputs)` to build a base and chosen
  observer widths in one reciprocal graph. Every observer reads and feeds back
  into the earlier populations; output neurons participate in the same solve.
  The factory reuses existing settlement and learning and supplies no privileged
  executive. Document initialization, full-state qualification and fair
  transformer comparisons in the recursive-settlement guide.

## 0.17.0 — 2026-09-27

Production contract hardening:

- Keep observer and observed neurons in one qualified reciprocal graph with the
  runnable [recursive settlement guide](docs/recursive-settlement.md). This uses
  existing `PatchNet` operations; it does not add a second solver or claim that
  greater recursive depth has demonstrated a capability or efficiency advantage.
- Reject mismatched free/nudged stream shapes and overflowing parameter
  proposals before clipping. A nonfinite graph solve returns unqualified
  diagnostics and cannot poison the next `PatchNet` observation.
- Roll back single-record observations completely when a later record write
  fails, including records, learned state and live activity.
- Keep Life's baseline and surprise readback finite, including perfect predictions
  and very small baselines; validate imagination candidates and costs. Replay windows stop at resets, so learning cannot fabricate transitions
  between episodes. Joint record compositions reject aliased child instances and
  reject invalid observation masks before they can become factual record writes.

- Reject nonfinite device parameters, effective weights and precision-dependent
  neuron constants before installing them. Failed learner and actor/critic
  updates restore optimizer, eligibility and reward statistics for a valid retry.
- Make connectome arrays/populations and exposed Brain parameter arrays
  read-only. Replace parameters through complete validated setters or
  `with_parameters`; change groups with `with_populations`. This prevents stale
  cached transport after in-place edits.
- Check record detuning's negative-phase curvature even at a zero-gradient
  saddle, recompute its actual equation residual, and reject nonfinite or
  malformed solves. Stabilize input normalization and validate sleep options
  before learning.
- Preserve cut-port ablations in joint-record checkpoints and write snapshots
  atomically. Validate joint paths/options before state changes, reject nonfinite
  trial losses, and restore earlier record writes when a later cortex write fails.
  The composition still runs a fixed number of rounds, without a convergence guarantee.
- Zero-rate observations on single/joint record patches retain valid activity and
  record writes but return `no_step`, without claiming a slow-weight update or
  incrementing its counter. Gradients remain available to composition callers.
- Preserve complete learned state on rejected online learning, detach replay
  history from caller arrays, and replay birth windows from their original
  boundary. Steered checkpoints now preserve the live boundary, gaze/heard/age
  channels and ablations; custom callables must be provided when restoring.
- Make joint parameter proposals exception-safe, correct the scaled Armijo
  directional derivative, honor sleeping-cortex configuration, and explicitly
  reject unsupported Steered record writes. Capped governors fall back to habit;
  the threshold governor's cooldown gene now takes effect.
- Count spatial kernel reuse and executed composition readbacks in dense
  forward-work estimates. Document excluded backward work and distinguish
  these estimates from total training FLOPs.
- Report unobserved orienting returns as nullable/censored, retain partial
  habituation bins, and validate traces and frame counts.
- Add [numerical and learning contracts](docs/contracts.md), distinguishing
  qualified equilibria, fixed-budget repair, adjoints and joint parameter
  admission. Update connectome/modeling scope, API migration and runnable guides.
- Repair the minimal-install CI reference to a removed guide, include belief
  and steering smoke checks, and package the contributor guide and linked
  benchmark sources in source distributions.

The composition release: what the rung demos (the shell game, the room, the arm, the eye
and the ear, the dozing cat, the ventriloquist, the lighthouse keeper, the night nursery)
and the nursery sent back. The belief patch first.

- A step on a target is admitted by default: `BeliefPatch.observe` replays the chunk from
  its boundary under the proposed parameters and takes the largest halving that lowers the
  loss by the Armijo margin, starting from twice the last admitted step and at most `rate`,
  so `rate` is a ceiling and the patch finds its step within it. `step_size` holds the last
  admitted step, travels with the snapshot, survives `reset()` and is dropped by
  `reset_step()`. `backtrack=False` is the plain step; a step on an external gradient
  (`output_gradient=`) is plain, since the library can replay only the loss it can see.
  Several tested lanes found divergence or weak learning with fixed steps; the two demos
  that kept a step size restarted from twice the last accepted one. This is evidence for
  an admission option, not a universal failure theorem for fixed-rate learning.
- `BeliefReadback.evidence` and `BeliefPatch.encode(observations)`: the encoded evidence
  before any gain, what each block is reading, for a steering patch that decides a block's
  gain (the night nursery earned rung 4 on it).
- `assimilate` and `observe` take `keep_live=True`, which leaves the live state where it
  was: a life that stepped through a window replays it from the boundary it kept without
  losing its place.
- `macs_per_moment(probes=)` and the `cost` counters (`moments`, `macs`, `replays` since
  `reset_cost()`, every moment assimilated, observed, imagined or replayed in an admission):
  a common estimate of dense forward work, where four demos had written their own.
  Complete training comparisons must also measure backward work and the exclusions
  documented in the contracts guide.
- `StructuredPort(mask=)`: which inputs the port hears, one flag per input, zero to every
  block in the forward map, its transpose and the gradient, and carried by `to_dict`; which
  channels a patch reads is a gene without touching the weights.

- `cadence.steering`: `Steered`, a cortex whose gains a steering patch of the same rule sets
  from the cortex's readback of the moment (the probes, the surprises, the previous residual,
  the previous outputs, the encoded evidence of named blocks, extra channels), through a
  weighing: `Softmax` (gains that sum to the blocks), `Gaze` (a window over a ring of blocks
  whose centre the steering output turns, under a price on the turn) or `Rule` (the
  hand-written control). One joint step is admitted by a replay of the chunk from twice the
  last admitted step; the cortex can sleep while the steering patch learns; a `Boundary`
  carries the life between chunks; ablations, one accounting and custody. Three demos wrote it
  before it was here (the ventriloquist, the lighthouse keeper, the night nursery).
- `cadence.life`: `Life`, the loop in which a governor reads the brain's own signals (a
  readback of seven channels) and names the mode, habit, imagine or learn; the executed
  window replayed from its boundary with rollback; the baseline of the surprise; the cost by
  mode. `PatchGovernor`, a settling brain whose every synapse is a gene, with the dozing cat's
  wiring as the hand-set genome and its space for `genes`; `ThresholdGovernor`, `AlwaysAwake`
  and `NeverWakes` as the controls. Over a belief patch or a steered cortex.
- `cadence.instruments`: `orienting` (capture, latency, return, the habituation curve per
  kind of event) and `dishabituation`, the night nursery's measures.
- `BeliefPatch.observe` leaves the store unwritten unless `write=True`: in every rung demo the
  store paid nothing and was left unwritten. `readback(probe=False)` skips the probes.
- Guides: [a brain that reads itself](docs/steering.md) and [how to build a rung demo](docs/howto-rung.md);
  the belief guide's sections on the admitted step, the online life and the cost; the API
  reference for every new name; the index and the README's table of brains.
- Removed: `EquilibriumActor` with `BodyModel`, `ActorPlan`, `ActorReadback` and
  `ObservationRecord` (`cadence.actor`), `ReservoirReplay`, `ContentMemory` and the sequence
  readback (`cadence.sequence`), with their guides. No file outside the library and its own
  tests imported any of them (the usage inventory of 2026-09-27 over the demos, the examples,
  the lanes and the benchmarks), the guides had called them kept for the experiments that used
  them, and the manifesto had recorded that the actor's fixed linear-model hypotheses do not
  transfer. Their evidence stays in the paper's receipts and in git history. `GenericBrain` is
  kept for its one remaining application and is deprecated in favour of the life below.

The population kernel at the width of a game: what the poker neuroevolution lane sent back.

- `PopulationPatch` reads and writes its record tables through the active cells alone (a
  flat row index into the table viewed as rows, `index_select` for the read, `index_add_`
  for the write), so a settle no longer touches every cell of every stream; at an 830-port
  reading over 4,096 streams on an M4 an imagined reading fell from 74 ms to 15 and an
  observation with its write from 213 ms to 41. The adjoint of the one-moment loss is written
  out instead of taken by autograd (six matmuls; the same numbers, held by the parity tests to
  1e-9), and the dense path uses `matmul` where `einsum` fell back to slow kernels on MPS.
- `PopulationPatch.observe` takes `mask`, an (instances, streams) gate over the streams that
  have a moment to learn from: the slow loss of an instance is the mean over its masked
  streams, only those streams write and move their statistics, and an instance with no masked
  stream does not move. A population whose streams live through episodes of unequal length
  (hands of poker) learns in hindsight through it. `stream_loss` gives the loss per stream.
- `imagine` and `observe` take `stream_of`, (moments,) or (instances, moments), assigning each
  moment of the batch to a stream's store: the five actions a drive imagines per stream settle
  in one launch, and every queued lesson of every stream is taught in one observe (the writes
  of one stream land against the store as it stood, the NumPy patch's `write_batch`).
- `observe` takes `weight` over the outputs of a linear readout: a patch that predicts a wide
  reading of which a few ports matter (two reward ports among 830) counts them by their
  weight; the batched poker body normalizes per block, every block a slot reads counting once.

What the adversarial audit of the Neural Computation submission (2026-09-26) sent back:
four runtime defects, their regressions, a direct measurement of the local rule, and the
corrected reading of three benchmarks.

- `Records.write` and `write_batch` validate every supplied field, its selected code and its
  known-output mask before any table or averaging count changes, and commit all touched rows
  together; an invalid later target, a nonfinite selected code or an update that would
  overflow raises `ValueError` and leaves the store as it was. Known masks must be boolean
  with the target's exact shape; a consequence-only code may still leave its unused valued
  part NaN.
- `Brain.with_parameters(efficacy=...)` and the `efficacy` setter compute the effective
  weights at once and reject a nonfinite product of finite efficacy and contact factors,
  where the optimized cpu path had deferred the product and could install infinity.
- `TemporalMemory.protect` computes its rank threshold from a scaled norm, so a finite input
  of 1e200 keeps its protected direction instead of overflowing to rank zero.
- `contrast_asymmetry` returns infinity when the two detuned paths coincide away from the
  free path, where it returned zero; its docstring says what the ratio is, a diagnostic and
  not an error bound.
- `benchmarks/local_equilibrium` and `benchmarks/local_equilibrium_longer`: the public
  centered contrast rule against an implicit-gradient learner on the same network, frozen
  copies and an affine control, on a nonlinear agreement judgment with supplied roles, with
  a global row-mass projection after each update as the convergence controller. The 800-update
  run misses its predeclared criterion; the separately specified 8,000-update follow-up
  reaches complete held-out accuracy in all five fresh seeds for both learned arms. A
  NumPy-only verifier reconstructs the final checkpoints and rejects five tamperings.
- `PopulationPatch.observe` gets the same contract: the inputs must be finite, the targets
  finite on the unmasked moments (a masked-out moment's target is ignored and may be NaN),
  and the write and the slow step are staged and checked together, so a moment that would
  leave a nonfinite table, statistic, parameter or Adam moment raises `ValueError` with
  nothing changed; a write whose rows could overflow is summed on a copy of those rows
  before it lands. Returned residuals are zero on masked-out moments. `stream_of` and a
  `weight` given on the cpu land on the patch's device (the loss had read the weight where
  it was given).
- Twelve regression cases in `tests/test_submission_audit.py` and twenty-one in
  `tests/test_population_audit.py`.
- The docs read their numbers as the receipts allow: the 390-fold population figure is a
  device and implementation ratio at one patch shape; the grammar's low-data comparison is
  at one eighth of the pairs with unequal parameter counts and one seed, and its slow
  learners were trained through sentence time; the record store's one-pass result is greedy
  validity after a calibration pass; the contrast at a finite nudge is an estimate with a
  `beta^2` bias term and a `1/beta` phase-solve term, and the nudged phases need their own
  contraction bound.

## 0.16.0 (2026-09-25)

What learning a discrimination on the fruit fly's measured wiring sent back: the operating
points a connectome does not carry, and the custody of a plastic seam. Each is generic; the
numbers are the fly's. The release also brings a population of record patches that learn in
lockstep on a device.

- `specific`, a protocol predicate over two stimuli: the readout's codes under the row's
  stimulus and under `versus` share at most `specific_max` of their union, both codes non-empty
  at `code_level` (`Row(..., versus=...)`, a training fact with a fourth element,
  `shared_code`); `code_level` is also the level `fraction` counts at, so `sparse` and
  `specific` read the same code, and `Brain.readings` takes a `level`. It is the fact a
  population's gain is selected on before learning at that population: a code that is the same for every stimulus carries nothing a lesson can attach
  to. `select_gain` takes it, and its docstring says the candidate can be any number a
  dictionary declares, such as one population's gain through `Brain(log_gain=...)`; the fly's
  antennal lobe ignites at the global gain through the mutual excitation of its cholinergic
  local neurons (cosine 0.98 between two odours' Kenyon cell codes at every gain, a third of
  the local neurons still active at a bias of -6) and settles to specific codes (cosine 0.32)
  at a twentieth of their gain.
- `calibrate_bias(brain, drives, targets, per_neuron=...)`: biases that bring populations to
  declared activity targets under the drives they will meet, one per population or one per
  member, found jointly. A readout cell on a rail (the fly's approach cell at 1.00 under every
  odour) has no slope for a nudge; calibrated to one half it learns.
- `naive_efficacy(connectome, plastic)`: every plastic class the same weight, for a lesson that
  should start naive at a memory site whose synapse counts are a specimen's memories (the
  fly's measured counts made one odour aversive and the other attractive before any lesson).
- `preflight(brain, outputs, plastic, drives)`: the checks a lesson needs before its first
  decision, under the drives the brain will decide in, each warning naming the block that
  repairs it (a readout with no slope, a shared code at the plastic seam's senders, a seam too
  thin for eligibility, two drives that are the same situation). Run it first; a receipt should
  show an empty list.
- `seam_report(connectome, pre, post)`: classes, synapses and coverage of a plastic seam. A
  synapse floor that removes noise elsewhere removes a distributed memory: the fly's floor of
  five kept 231 of 1,079 Kenyon cell classes onto one output neuron and 14 of 336 onto the
  other; the example's fixture now keeps the seam at every count.
- Docs: [brains from a connectome](docs/connectomes.md) (the seam in custody, the gain per cell
  class with the selection code, the two operating points), [protocols](docs/protocols.md)
  (`specific`), the traps of [learning from reward](docs/reward.md), troubleshooting, the API.
- `cadence.population.PopulationPatch`: record patches in lockstep on a torch device, a
  population of instances each with many streams, every settle one batched product; the
  slow step of each instance is the adjoint of its own one-moment loss (autograd), the
  records the delta rule at the code, nothing crossing an instance, a stream or a moment;
  `from_patch` broadcasts a NumPy patch and the tests hold the twin to it at 1e-9;
  `inherit` copies parents for selection. Guide: `docs/population.md`.
- `benchmarks/population/throughput.py`: the population kernel against the NumPy patch on
  one core, one moment's work per brain and stream, with a receipt (390 times one core on an
  Apple M4 laptop's graphics processor at 8,192 streams; parity of the reading 2e-16).
- The paper is published as a preprint on PhilPapers (https://philpapers.org/rec/MUECAP-2);
  the README and the guides link it there, and the PDF copy at the repository root is removed.

## 0.15.0 (2026-09-24)

The release of the fruit fly: what building a whole nervous system as one brain, and learning
on it, sent back into the library. The example itself, and the browser engine every settling
page shares, live in [cadence-examples](https://github.com/muellerberndt/cadence-examples).

- `LearnerConfig.scale_cap` is the magnitude a plastic synapse's efficacy may not exceed
  (default 8, the former module constant); it is applied inside the update on both backends,
  so a cap no longer needs a rebuilt brain per decision.
- `ActorCriticConfig.critic_signal` defaults to `"auto"`: the raw prediction error for the
  critic whenever the dopamine is centred (`dopamine_center > 0`), the modulated one otherwise;
  `critic_target` is the resolved choice. A critic fed the centred signal ran away.
- The `learn` report carries `saturation` (the fraction of output activations within 0.02 of 0
  or 1, where a nudge has no slope) and `trace` (the mean absolute eligibility of the plastic
  synapses), the readings of the latch.
- `brain_scan.js` v4.2: the glow colours (`hot`, `cool`), `labelCount` and a `dpr` cap are
  options.
- Docs: [brains from a connectome](docs/connectomes.md) (custody, the gain by protocol, the
  sub-net a page settles, what a rate model cannot carry, one gain and many circuits), the
  traps of learning from reward with their measurements, the troubleshooting entries for both.
- The README no longer lists every example; it links the examples page and shows two.

## 0.14.0 (2026-09-23)

- The brain viewer has a second style, `style: "brain"`, and is the patch-net
  visualizer every page shares: the same net wrapped into the volume of one
  stylised animal brain in three dimensions. A translucent shell with a rim
  light (two lobed hemispheres, a cerebellum, a stem) turns slowly on its own,
  drag rotates it, scroll zooms; every region is assigned a lobe by its role
  and name (sensory at the back, memory deep as the hippocampus, association
  across the cortex, motor in the frontal strip, a governor at the front as the
  prefrontal area) and keeps the atlas's arrangement inside it; neurons are
  glowing somata; every synapse is a curved ribbon in three dimensions, very
  dark at rest and lit in real time by the messages that travel on it, with a
  glow along the path and a pulse with a tail, brightness by message times
  weight, strongest first under the budgets; additive blending, bloom and a
  depth fog. `setStyle` switches between `"scan"` (unchanged, still the
  default) and `"brain"` on the same canvas, with the same API in both. The
  brain style's options: `restAlpha` (0.025) for the resting web, `bloom`
  (0.5) for the bloom's gain, `shell: false` hides the shell, `spin: false`
  stops the self-rotation, `spinRate` (0.12 radians per second) sets its
  speed; `mode`, `particles`, `edges`, `lineBudget` and `particleBudget` mean
  the same as in the scan style, and `snapshot()` reports the style, the
  view's yaw and pitch and the frame time. `lobeOf` and `brainLayout` are
  exported. The guide `docs/pages.md` frames the viewer as the patch-net
  visualizer every page shares, with both styles and their options.
- Documentation for builders. The README opens with the roadmap (an
  effortlessly evolved human-like brain, taught by imitation and then by its
  own life, whose experience is that of a human in our world; human-level
  competence as the first milestone), what Cadence is in machine-learning terms, the install, the three browser demos and one runnable
  brain, a table of which brain to start from, the four examples with what each
  shows and what it is built from, and a map of the guides; the vision sections
  follow. New guides: `docs/orientation.md` (the four brains against familiar
  models, where each learning signal comes from, a comparison with backprop, the
  shapes, a glossary), `docs/build.md` (your own data to a trained, evaluated and
  saved record patch and settling brain, with every block run by the tests and
  the minimal-install job), `docs/troubleshooting.md` and `CONTRIBUTING.md`.
  `docs/index.md` is the map in the order a builder needs. Guides that pointed at
  a quickstart loop that no longer exists point at the settling brain quickstart;
  the settling-brain guides no longer describe themselves as kept for earlier
  experiments; release-number narration and file paths of unpublished
  experiments are gone from the guides; the API reference opens with a contents
  list by brain. The two figures in `docs/assets/` use the neuron and synapse
  vocabulary of the guides.
- `BeliefPatch.assimilate` and `observe` take `state=`, a boundary to start the
  moments from instead of the live belief, as `imagine` already did: a life
  that learns from its executed window replays it from the belief that was
  live at the window's first moment. `BeliefPath.step` is the last repair
  iteration's move per belief unit, whose norm is `residual`: where the
  belief moved under the evidence, for a governor and for a page.
- `JointRecordPatches` and `Port` (`cadence.record_ports`): several record
  patches settled as one equilibrium, moment by moment. A port is a declared
  band of one patch's scaled context read as an input of another in the same
  moment; the joint energy adds a seam term per port, `rounds` Jacobi rounds
  settle the coupled patches, the seam residual per moment and round is the
  fourth instrument, and one backward scan carries the adjoint across every
  seam (`cross_adjoint`), with the library's backtracking on the joint slow
  loss. Records are read at the settled readings and written after the path as
  before; snapshots are per patch plus the topology; `imagine` changes nothing.
  Which patch hears which, the band and the width are a genome for `evolve`.
  On two coupled symbol streams the joint port lifted the dependent stream from
  0.39 to 0.54 at matched parameters against two independent patches, and the
  same-moment settling round carried it; the guide is in `docs/record-patch.md`.
  `RecordPatchNet._slow` takes optional readout parameters for a replay.
- The brain viewer allocates its render targets on its first draw even when the
  canvas was already sized by an earlier viewer: a page that built a second
  `BrainScan` on the same canvas drew its tissue field into an unallocated
  buffer and the view filled with one colour. Pages that swap brains should
  call `setAtlas` on the viewer they have; either way now draws.
- `evolve` selects over any genome: a `Genome` grows by `develop` and mutates as
  before; any other genome, a dict of a governor's thresholds, a port topology,
  a patch's sizes, takes its own `mutate(genome, rng)` and an optional
  `grow(genome, seed)`, and the lineage records it as it is. `cadence.genes(space)`
  supplies a mutation over a declared space of `log`, `linear`, `int` and
  `choice` genes. Where a rule looks designed it becomes a gene, with the
  hand-set value as the control.
- The three quickstart brains run in the browser: `cadence-demo stream`,
  `cadence-demo decide`, `cadence-demo body` (`cadence.demo`). Each command
  trains its brain in front of you from the quickstart's seeds, in seconds to a
  minute, and serves a local page: the whole brain in the shipped viewer with
  every neuron and synapse, the neurons switchable to their distance from
  equilibrium (how far each still moves as the brain settles, how far the
  teaching pull displaces it), the synapses switchable to their last change
  from the detuning contrast, the task, a stats strip with the numbers that
  define the equilibrium and its detuning (settle steps and last movement;
  free residual and energy, the two detuned phases, their asymmetry, the
  update size; the seam defect, the slow loss, the writes), the learning
  curves, and a paragraph for each phase. Nothing is hosted and there is no
  checkpoint. The guide is `docs/demos.md`: what equilibrium means in each
  brain and what detuning buys.
- `BeliefPatch`: a gain per observation-port block inside the repair. `assimilate`, `observe`
  and `imagine` take `gains=` (`(blocks,)`, `(batch, blocks)` or `(batch, time, blocks)`),
  which multiplies each block's encoded evidence before the repair map and the store read
  see it, so the gain acts in every iteration and in the store's code; the adjoint returns
  the gradient into the gains as `BeliefObservation.gain_gradient` and the path carries the
  gains used. The tests check the gain gradient against finite differences (worst relative
  error 1.3e-10 at the test's sizes).
- `BeliefPath` carries the readback of every moment: `evidence` (the encoded evidence the
  repair read, after the gains), `code` (the store's plain code at the final reading),
  `gains`, `residual_alone` (the repair map's move with one block heard and the store read
  at zero, with `probe=True`) and `surprise` (each block's reading against the reading the
  previous belief's slow readout implies, in the block's persistence units, once
  `set_implied_reading(implied, units)` declares the map; channels the map leaves `NaN` are
  not compared). `readback(observations, actions, state=)` gives one moment's expectation,
  probe and surprise before its repair, what a steering patch reads before it sets the
  moment's gains; it changes nothing.
- `BeliefPatch.observe(backtrack=True)`: an admitted step, as the record patch has. The chunk
  is replayed from the same boundary under the proposed parameters with the store as it
  stands, and the largest halving of `rate` that lowers the loss by the Armijo margin is
  taken (sixteen halvings at most); `accepted_rate`, `final_loss` and `replay_calls` are on
  the observation, and the store is written after the admission. In the tests a rate of 100
  that diverges within three chunks without it is admitted with it, and the loss never rises
  on an accepted step. The admission adds no state to the patch.
- `BeliefPatch.observe(loss_weight=)`, a weight per moment `(time,)` or `(batch, time)`,
  normalized by its sum instead of the moment count in the loss and the adjoint; a moment
  of weight zero is neither taught nor written. `observed=` is accepted per row as
  `(batch, time)` as well as `(time,)` in `assimilate`, `observe` and `imagine`'s mask
  handling; a row that observes nothing keeps its expectation while the other rows repair.
  The adjoint under a weight and a per-row mask matches finite differences (worst relative
  error 2.2e-10 on the parameters).
- `BeliefPatch.observe(output_gradient=)` in place of `target`: the adjoint under an external
  gradient on the outputs, so a patch is taught through a seam (a steering patch, from the
  cortex's gain gradient through the caller's map). Nothing is written and no loss is
  reported. With the gradient of the target form the parameter and gain gradients equal the
  target form's exactly, and they match finite differences of the linear functional (worst
  relative error 2.8e-11).
- `TorchBelief.forward` takes `gains=` and `observed=` in the library's shapes, since the
  forward is shared; a gains tensor that requires grad receives the gradient into the gains,
  and the parity test covers the beliefs, the outputs, the parameter gradients and the gain
  gradient under a weighted loss and a per-row mask. The loss weight, the admission and the
  external output gradient are the caller's loss and optimiser on torch; the readback of a
  moment stays on the library's path.
- `cadence.record_ports` ships with its arithmetic bound by finite-difference tests
  and its line length allowed; the module's type annotations follow in a later
  release (a file-level mypy opt-out marks it).

## 0.13.0 (2026-09-22)

Toward a learned world model: a belief that a transition carries under action, evidence
that repairs it by iteration, and maps over grids at the port. The
[belief patch guide](docs/belief.md) states the training rule the world model needs (the
imagination loss: a rollout must predict every input it consumes) and the first Atari
evidence: from pixels, the belief patch's imagination with no observation explains 0.61 of
the changed cells eight decisions ahead on Pong, and planning in that imagination scores
3.7 times random play on Demon Attack.

- `StructuredPort` (`cadence.ports`): a record patch's input port as blocks. A
  `MapBlock` is a tied local kernel over a grid of the inputs (the same kernel at every
  position); a `DenseBlock` a matrix over a slice. `broadcast=(start, count)` tiles a slice
  of the inputs into every map block as constant channels, so an action meets the scene
  before the nonlinearity. `RecordPatchNet(port=...)` and `RecordPatchStack(lower_port=...)`
  route their input drive through the port in the forward scan, the adjoint, the detuning
  check and the checkpoint; a port of dense blocks reproduces the plain patch exactly.
- `BeliefPatch` (`cadence.belief`): a belief carried by a learned transition under the
  executed action, repaired by a few damped iterations of one nonlinear map that reads the
  belief, the encoded observation, the expectation and the record store together, and a
  linear readout the store patches. `imagine(actions)` continues privately under declared
  actions with no observation and changes nothing. `observe` learns by one backward scan
  through every iteration and the transition; the adjoint matches finite differences.
  `belief_torch.TorchBelief` is the same slow half on torch with autograd, exporting in
  the library's packing.
- The belief patch guide's section on the imagination loss: a belief patch trained on the
  one-step read alone learns to lean on the next frame and its imagination decays; the
  open-loop drift from random moments, added to the loss, trains the transition to carry
  the belief. The recipe stays in the application; the library supplies the imagination
  with a state and no observation.
- Removed `cadence.custody` (unused).

A centered contrast is a derivative only while the two detuned paths sit
symmetrically around the free path. On a long path, or a path far from its
goal, one detuned solve can converge on another branch of the energy while
still reporting convergence; the contrast then points somewhere else. On a
learned cart-pole model over sixteen steps the default beta gave an action
contrast 25 times too large and 58 degrees off the true derivative, and the
planner's line search found no decreasing step.

- `cadence.temporal.contrast_asymmetry(free, plus, minus)`: the norm of
  `plus + minus - 2 free` over the norm of `plus - minus` on the hidden paths.
  The ratio grows in proportion to beta and the contrast error with its square.
- `TemporalPatchNet.observe`, `TemporalPatchNet.plan` and
  `TemporalMemory.observe` take `symmetry_tolerance=0.15` and
  `max_halvings=8`. A pair of detuned paths above the tolerance is solved again
  at half the beta; a pair that stays off center is refused with the reason
  `contrast_asymmetric`, changing no parameters, revisions or update counts.
  `TemporalObservation` and `TemporalPlan` report the `beta` of the last
  contrast and its `contrast_halvings`. An infinite tolerance disables the
  check and reproduces the unguarded update.
- `TemporalPatchNet.plan(method="bfgs")`: a quasi-Newton direction over the
  controlled ports, built from the accepted steps, with the same projected
  line search and causal replay. The cost over an action path is badly
  conditioned and the steepest direction zigzags; on a twelve-step toy the same
  stationary cost takes 52 phase calls instead of 121. The estimate is dropped
  whenever a step is clipped, fails the curvature condition or gives no
  decrease. `TemporalPlan.method` names the direction; the default
  `"steepest"` is unchanged.

## 0.12.0 (2026-09-22)

The record patch: a temporal patch with a gated linear context and a record
store inside it. An observation is written once by day; by night the slow
weights learn from the store's own dreams.

- `RecordPatchNet` is a temporal patch with a gated linear context and a
  `Records` store inside it. Its energy is quadratic, so the centered
  detuning contrast equals the adjoint gradient of the same loss; `observe`
  computes it by one backward scan and `detune` solves the two detuned
  equilibria as the acceptance check. Records store what the slow readout got
  wrong at each reading. Complete checkpoints, private branches and causal
  parameter admission are available; planning and protected responses are
  not.
- `dream(inputs)` is the store's completion of a cue from rest, as a target.
  `sleep(cues, *, passes=1, rate=1.0, backtrack=False, dawn_passes=2)`
  splits acquisition into two phases: by day each observation is written once
  into the records with the slow weights at rate zero; by night the slow
  weights learn the store's own completions of the day's cues, with the corpus
  closed; at dawn the dreams are written back so the store holds only what
  the slow weights did not take. On the guide's small grammar the slow weights
  alone go from 0.815 to 1.00 of never-taught combinations after one night.
  Sleep replays the store's errors as well as its recalls; the guide measures
  the recombination and teacher arms and states when to sleep from three
  examples under the day and night regime.
- Categorical ports: `RecordPatchNet(groups=(n1, n2, ...))` ends the slow
  readout in one softmax per group, learned by cross-entropy; a record holds
  the bounded residual `onehot - softmax`.
- `Records.write_batch(codes, targets)` and `RecordPatchNet(record_writes="batch")`
  write a call's readings at once, each cell moving by the mean of the moves its
  writers would have made alone: one writer reproduces `write`, agreeing writers
  do not overshoot. Nets with categorical ports or batch writes save checkpoint
  format `cadence-record-patch/3`; default nets keep format 2.
- `RecordPatchNet(record_width=w)` makes the cells hold a fixed random `w`-column
  sign code of the residual, decoded by the transpose, so a store can be
  narrower than its port. The guide records the measured cost of the code and
  of cells and passes for exact recitation.
- `RecordPatchStack` puts a context patch below a complete `RecordPatchNet`;
  the upper adjoint scan hands its input gradient to the lower scan, so the
  upper gate can tell a subject from a later noun. Gradient parity with finite
  differences holds in both patches, with a linear readout and with categorical
  ports. `observe` returns `StackObservation`.
- The record patch's reading gives its two blocks unit variance per unit:
  the input block is scaled by `sqrt(n) / s` with `s` the running rms norm of
  witnessed inputs. On the composer stream this spreads the code from 2,225
  to 5,812 of 8,192 cells, makes the address follow the event heard, and
  raises held-out accuracy. Checkpoint format `cadence-record-patch/2`
  carries the input norm and the record counts.
- `Records(..., averaging=True)` makes each cell's write rate one over its
  written mass with `rate` as the floor; `Records(..., homeostasis=r)`
  moves each cell's offset toward an equal activation share. Both are off by
  default and exposed on `RecordPatchNet` as `record_averaging` and
  `record_homeostasis`; `Records.state` includes `count`, `usage`, `boost`
  and `drive_scale`.
- `Records.witness`, `Records.state` and `Records.load_state` expose the
  running mean and the learned tables for checkpoints.
- Optional `TemporalPatchNet.observe(..., backtrack=True)` accepts a parameter
  step only after target-free replay of the observed path improves its loss
  from the original hidden boundary. It reports trial costs and accepted rate;
  rejection preserves learned parameters and update counts. Fixed-rate behavior
  is the default. This is a current-observation check, not a retention or
  generalization guarantee.
- `cadence.experimental.PartitionedTemporalPatchNet` adds explicit fixed
  connectivity masks over the existing temporal solver, with masked detuning,
  complete checkpoints and private planning. Supplied routing is an experimental
  control, not learned specialization. `TemporalMemory.observe` rejects this
  subclass before mutation; the default temporal model is unchanged.
- Removed: `cadence.legacy` and the 0.8 names it installed at import, kept as
  deprecated aliases for one release since 0.9.0. Code that used them takes the
  current names from the API reference.
- The README and the documentation index present two primitives, the settling
  patch and the record patch, and one quickstart page with three brains: a
  record patch that learns a stream, remembers in one shot and sleeps; a
  settling brain that decides; a temporal patch that learns a consequence and
  plans. Version narration in the guides is replaced by plain statements; the
  compatibility compositions are listed as kept for the experiments that used
  them. The API reference states the return types of `EquilibriumActor`,
  `TemporalMemory`, `RecordPatchNet.readback` and `RecordPatchStack.observe`.
- The record patch guide covers categorical ports, batch writes, store sizing,
  the two-patch stack and a grammar from records alone. Documentation from the
  composer work: how to diagnose a record store (cells in use, top-cell share,
  code overlap per block, frozen against online scores), what a write rate
  stores, boundary readings, running a trained record patch outside Python
  with a parity test, the scale of a `Records` reading, seven new rows in the
  missteps guide, and a section on recall being by content and not by position
  (code overlap 0.91 where the stream recurs, 0.16 at the same place after a
  phase shift).
- Scaling and task-design guidance separates unique experience, repeated
  training, model width and temporal scope, and distinguishes actual sensory
  feedback from carrying a model's predicted state.
- The paper is a PDF at the repository root (`cadence-paper.pdf`), linked from
  the README. The README and the documentation index link the public
  [examples repository](https://github.com/muellerberndt/cadence-examples); the
  application sentences name its four examples, and the creativity guide states
  where the published composer stands against the standard for original
  creation.
- The wheel-only CI job runs the record patch and fixed-connectivity guides in
  addition to the eight earlier pages; its quickstart check follows the new
  quickstart.

## 0.11.0 (2026-09-20)

- `TemporalPatchNet.plan` repairs caller-selected continuous input ports using
  centered equilibrium contrasts, bound projection and decreasing causal replay.
  It keeps model parameters, live context and actual records unchanged.
- Detached `TemporalPlan` diagnostics distinguish stationarity, predicted goal
  satisfaction, work caps and failed phases. Desired outcomes never become
  observed training records through planning.
- Optional `TemporalMemory.observe(readout_damping=...)` applies a damped local
  readout metric inside the protected subspace. A causal replay rejects harmful
  metric candidates; no replay corpus or persistent optimizer state is added.
- General-intelligence and creativity goals now organize the README, API index
  and documentation. Task-specific examples illustrate measured components.
  Evolving recursive self-reflection remains a research requirement.
- Planning documentation and executable examples cover the action/readback
  boundary; memory documentation distinguishes remaining rank from conditioning.

## 0.10.0 (2026-09-20)

- `TemporalPatchNet`: time-varying observation paths, persistent hidden context,
  centered local equilibrium detuning, curvature/convergence-gated updates,
  isolated imagined paths, detached readback and exact continuation checkpoints.
- `TemporalMemory`: explicitly selected local response constraints, compressed
  into bounded presynaptic subspaces; atomic constrained learning and binding
  checks. Protection is conditional and finite; importance is supplied.
- `EquilibriumActor`: ordered actual observations, fixed-model Gaussian past
  repair, joint future-state/action repair, supplied goals and private plans.
  This two-coordinate linear body is a controlled capacity component.
- A unified architecture guide and executable examples document the feature
  boundaries. Existing graph APIs remain compatible.
- Task-design and common-missteps guides cover observation/action semantics,
  information loss, behavioral checks, and the unproven boundary of scaling.
- The pinned Lean package audits 169 conditional theorems, including causal path
  uniqueness, quadratic boundary elimination and protected-path induction.

Automatic relevance, forgetting, hierarchical self-modeling and general
long-form composition remain experimental requirements. The new APIs do not
silently turn predicted outcomes into observations or claim biological equivalence.

## 0.9.0 (2026-09-20)

`PatchNet` is the starting point for new experiments with local overlap repair
and equilibrium detuning. It combines continuous observations, persistent
neural activity, reciprocal connections, residual-checked free/nudged learning,
isolated imagination branches and complete continuation checkpoints. Optional
temporal overlap uses the preceding free activity as a fixed boundary during
each learning event. Learned importance, reliable lifelong retention and useful
autonomous improvisation remain open experimental questions.

- Added the [PatchNet guide](docs/patchnet.md) and public API reference.
- Bundled a pinned [Lean proof library](lean/README.md): 156 theorems with a
  source-bound build and axiom audit, explicit assumptions and implementation
  limits. Optional linear-record lemmas are separate from the PatchNet core.
- Removed links to the retired example and Patch World pages from current
  library documentation.

Other changes since 0.8.1:

Biological names throughout, brain regions, and a generic brain.

- `Records` (`cadence.records`): the records cortex. A reading of `inputs` units loses each
  unit's running mean (the plain average of the witnessed readings until one over their
  count falls to `habituation`, then a running mean at that rate); a fixed random expansion
  keeps the `active` most driven of `cells` cells at unit length; one delta-rule table per
  predicted field is read through the active cells and written by the error, at `rate` for
  consequence fields and at `valued_rate` for `valued` fields, which read a second code
  whose input `pathways` are divided by their running norms. `known` masks unobserved target
  entries and `to_dict` rebuilds the fixed cells from the seed. `tests/test_records.py`
  covers the codes, the write identity, a correlated walk, habituation, the valued code,
  the masks and the settling mean.
- `Mulberry32`: the 32-bit generator of `brain_scan.js`, with `batch` and Box-Muller
  `normals`. `Records` draws its projection and offsets from it, so a page rebuilds the same
  cells from the seed.
- The evidence trees `experiments/` and `benchmarks/` leave the repository together with the
  tests bound to them, `docs/comparisons.md`, whose rows cited them, and every documentation
  reference to them. `docs/assets/cadence-logo-prompt.md` is removed.
- Documentation: `docs/cortex.md` (regions, the catalogue, projections, ports, the synapses
  a head owns, when to settle and when to record), `docs/brain.md` (development, checked
  settling, one experience step by hand with records beside a policy head, `GenericBrain`,
  checkpoints, the browser page) and `docs/evolution.md` (mutation, selection over short
  lives, parallel lives, equilibrium detuning). Their snippets run in the documentation
  tests, together with those of `docs/concepts.md` and `docs/certificate.md`.
- Documentation: the record principle in the README, the concepts, the records section of
  `docs/memory.md`, and the learning, reward, experience, task, continuous-interaction,
  rehearsal, content-memory and sequence pages.
- Documentation corrections: the certificate example uses a certified brain and states that
  `steps_for` raises for an uncertified one. The API reference adds `Records`,
  `Mulberry32`, `PatternSeparator`, the certificate, the atlas, `Lineage`,
  `FastSynapses(separator=)`, `Learner(synapse_rate=)`, `Learner.apply`,
  `Learner.contrast_rows`, `save(..., compressed=)` and the keyword-only arguments of
  `record_settlements`, `SequenceCache` and `BoundedTrace`. The learning knob table gives
  the defaults of `eta_bias` and `layered(density=)` and adds `synapse_rate`; the viewer
  page gives `particleBudget` as 300,000, `subsample_edges(limit, seed=None)` and
  `page(note=, inputs=)`. The pages drop contrastive and status wording.

- The README and quickstart center one ongoing experience loop without a
  training/inference mode switch. Retired example links and the bundled demo catalog
  are removed; their behavioral checks remain in the test suite. Minimal-install CI
  runs the actual quickstart from the built wheel.
- Action caches refresh after parameter changes, including reward updates and checkpoint
  resume. Memory and trace queries preserve live streams when the query batch differs.
- Complete-brain loading validates continuation arrays, counters and pending actions.
  Overflowing fast-memory writes preserve previous records and separator state.
- `ContentMemory` provides competitive prototype recall; `ReservoirReplay`
  retains a bounded uniform sample of real observations for rehearsal. Both
  expose their memory budgets and keep observation separate from pure reads.
- `cadence.sequence` adds causal content readback and bounded activity traces,
  with fixed coordinate frames per read and stable extreme-value arithmetic.
- Persistent memory now honors expanded separator keys, and `GenericBrain`
  checkpoints preserve the actual separator projection and running mean.
- Float64 Torch residual checks can stay on device; float32 keeps the independent
  host reference. Transport avoids unnecessary accelerator synchronizations.
  `ep_structure` checks effective free/free symmetry under declared fixed inputs.
- Host, compiled and accelerator paths factor small phase contrasts before products
  to reduce cancellation, including per-stream reward traces and restored phases.
- `layered(skip=True, skip_init=0.0)` adds a learnable direct sensory-to-motor
  route with unchanged initial predictions; the existing initializer is the default.
- Discrete reward eligibility now differentiates the sampled softmax policy even
  when imitation uses quadratic nudges. Native categorical motor slots can have
  unequal sizes and never sample padding. Numerical derivative and CPU/Torch
  regression tests cover both contracts.
- `ActorCriticConfig(critic_signal="td")` lets the critic fit return in reward
  units independently of actor clipping/centring. The default `"modulated"`
  preserves the original bounded critic update. Reports distinguish raw TD error
  from the actor's modulation; the reward guide explains calibration and tuning.

- `cadence.certificate(brain)`: the settling certificate. From the largest absolute incoming
  effective weight sum (`row_mass`) and the neuron model's slope bound (`lipschitz_constant`)
  it reports the contraction rate `1 - dt (1 - L rho)`, whether the free phase is certified,
  the a posteriori error bound from the last step's movement, the a priori bound, and the
  warm-start step budget after a stimulus change. Under `learning_neuron_model()` a brain is
  certified below a row mass of 2. Documented in `docs/certificate.md`.
- `PatternSeparator` and `FastSynapses(separator=...)`: keys pass through a fixed random
  expansion and a k-winners-take-all before the record sees them, with an optional running
  mean removed first. Expansion can reduce overlap between correlated keys;
  exact noninterference requires disjoint supports and is not guaranteed by expansion.
  Documented in `docs/memory.md`.

- `cadence.atlas_of(brain)` / `build_atlas(connectome)`: the brain atlas, one integrated
  layout of the whole connectome. Regions (the populations) are placed by a force layout of
  the region graph, neurons inside a region by a whitened spectral embedding of their
  synapses, sheets by declared shapes, or by supplied coordinates. `atlas.frames` quantises
  recorded settlings, `atlas.page` writes a self-contained page, and `brain_scan_script()`
  returns the shipped WebGL2 renderer (`brain_scan.js`): tissue in region colours whose
  brightness is the activation, a hot scan-coloured glow where neurons change, synapses that
  light up as their presynaptic neuron changes, messages travelling along synapses, and an
  EEG-style montage of every region. Documented in `docs/pages.md`.
- `GenericBrain.step` coordinates ongoing perception, previous-action feedback, optional
  current demonstrations and the next action, without a training/inference mode switch.
- `SynapticMemory` adds shared persistent synapses and fading per-stream residuals.
  Repetition and salience consolidate only actually observed value components. Persistent
  weights survive episode/batch resets and can be revised or explicitly cleared. New
  `episodic=True` brains use this rule; older checkpoints keep their original fast memory.
- Complete brain checkpoints now include pending actions and their nudged states, so
  interaction can resume before reward arrives. Generic checkpoint format 2 loads format 1.
  `parameters()` includes consolidated synaptic storage. The continuous-learning guide
  documents the clocks, equations, biological analogy and measured retention checks.
- The imitation optimizer counts only its own contrasts for bias correction; interleaved
  reward/direct updates no longer advance an optimizer whose moments did not change.
  That count is checkpointed; older checkpoints retain their saved count as the fallback.
- Tests pin the default batch row of `fraction_active`, integer stimulus indices, the
  60-step `settle` default, and a host warm state that the float64 torch kernel shares
  memory with and must not write. The NumPy and torch settling loops update fresh arrays
  in place (#1, Jonathan Hill).

- Names: `Connectome` (was `Wiring`), `Brain` (was `Settlement`), `BrainState`,
  `NeuronModel` (was `GradedRule`), `learning_neuron_model`, neurons and synapses (were
  owners, overlaps and seams), `populations` (were `sets`), `synapses` (was `edges`),
  `stimulus` (was `clamp`), `efficacy` (was `edge_scale`), `plastic_synapses` and
  `plastic_neurons` (were `trainable_overlaps` and `trainable_owners`), `reciprocal` (was
  `symmetric`), `activity_change` (was `repair`), `FastSynapses` (was `FastSeams`),
  `Genome` and `develop` (were `Constitution` and `grow`), `cadence.circuits` with
  `assemble` and `reflex_arc` (were `cadence.brains`, `couple` and `sensor_motor`), and the
  modules `cadence.connectome`, `cadence.neuron`, `cadence.brain` and `cadence.genome`.
- `cadence.legacy` keeps every 0.8 name, module path, keyword, attribute and method as a
  deprecated alias for one release; each use warns with its replacement.
- Checkpoints are written in format 2 with the new entry names; format 1 loads.
- Dictionaries use the new names: `Connectome.summary()["populations"]`,
  `Brain.to_dict()["neuron_model"]` and `["efficacy_changed"]`, `Learner.to_dict()["brain"]`,
  and the conformance ledger's `declared_synapses`, `transmissions` and
  `undeclared_transmissions`. Error messages use the new vocabulary.
- `Region`: a blank group of neurons or a designed region with its own circuit, inputs and
  outputs. A `Genome` mixes both; `develop` lays designed circuits into the connectome and
  draws projections between region populations (`region/population`), and `mutate` keeps
  designed regions' sizes. Blank genomes develop to the same connectome as in 0.8.
- `cadence.regions`: `visual_cortex` (retinotopic input and feature maps with local
  receptive fields), `cortex`, `motor_cortex` and `prefrontal_cortex`.
- `GenericBrain`: a sensory region or visual cortex, an association cortex and a motor cortex
  in one connectome, with an external reward helper (`ActorCritic`), optional prefrontal working
  memory (a `Trace` of the association cortex) and an optional hippocampus (`SynapticMemory`)
  for one-trial records. It learns from labels (`fit`) and reward (`act`, `learn`), and its
  genome can be evolved.
- `ActorCritic.state`: the free phase of the latest moment.
- `Brain.equilibrate` checks the whole circuit's equation residual under an exact step
  budget and returns per-row convergence information, including for zero-budget checks.
- `GenericBrain.save/load` preserves the standard composition: critic,
  both optimizers, random state, eligibility, working memory and episodic records.
  Checkpoint replacement is atomic; malformed learner optimizer state is rejected.
- Reward decisions now honor changed observations, own cached inputs, clear stale greedy
  eligibility and reset ended rows before one next-state phase. Batch neighbours no longer
  receive extra settling when another stream ends. Eligibility survives host/device transitions.
- Generic learning validates labels and transitions before mutation, owns observation/action
  buffers, exposes the current learned brain and accepts truncation bootstrap values.
- Fixed renamed device-cache keys, dense transport dropping parallel synapses, frozen
  weights changing under clipping, and overlapping reciprocal/explicit ties. Large tie IDs
  no longer allocate by label magnitude. Complete parameter assignment refreshes each backend.
  Sparse learning uses a bounded-memory host contrast instead of an unsupported device
  block operation or a potentially full dense Gram allocation; bias-only reports stay finite.
- Fused nudges honor fractional quadratic masks and excluded softmax groups. Signed
  traces produce nonnegative bounded salience; population-code probabilities normalize per axis.
- Added validation for reward settings, masks, nudges, neuron indices, region ports and sizes,
  genome constraints and optimizer state. Zero-density layered projections are supported.
- `imagine` accepts optional adversarial pruning and an isolated `clone` callback, supports
  array-valued action sequences, and checks its budget before invoking another transition.
  `ActivityMonitor.reset` starts an independent monitoring episode.
- Documentation now includes a runnable README quickstart, a complete brain-design guide,
  biological-object/function mapping with support boundaries, and tested introductory snippets
  and links. Standard cortex builders remain optional functional analogues.
- Timing tolerates unavailable Unix statistics. Minimal wheel CI also runs on Windows.

## Historical changes before 0.9.0

The entries below describe earlier interfaces and documentation. Use the current
README and quickstart for the ongoing experience API.

- Class-label and output-port validation prevents negative indices, duplicate outputs
  and accidental batch broadcasting. Slotted accuracy averages over every row and slot.
- Single-state softmax nudges and empty nudge masks work consistently. Zero-step
  accelerator trajectories return an empty trajectory, and invalid step counts fail
  with a clear error before execution.
- Wiring validates endpoints before merging edges and preserves tiny signed weights.
  Shuffled controls preserve degree counts even when endpoint swaps overlap.
- Gain selection breaks score ties toward the smallest admissible gain and rejects a
  grid with no admissible candidate. Checkpoints preserve explicit precision, and
  traces retain an owned snapshot when callers reuse state storage.
- Beginner documentation and minimal-dependency installation checks were added.
- Sparse CPU settlement uses optional SciPy CSR transport, avoiding the batch-by-edge
  message array. NumPy-only installations retain segmented sums; local dynamics and
  parameter semantics are unchanged.
- Blocked PyTorch learners keep adaptive optimizer history and updates on the device.
  Host reads, edits and checkpoints remain supported. Parameter rebuilds preserve an
  explicit torch device, and in-place trainability-mask edits invalidate cached masks.
- `FastSeams.observe`/`recall`: direct key/value ports, with opt-in `rule="delta"`
  residual writes and per-row resets. Existing Hebbian modes retain their semantics.
  Independent LMS, interference, legacy, reset and validation tests cover the change.
- `Settlement.residual`: remaining potential/adaptation equation discrepancy, so
  saturation and tiny steps cannot masquerade as equilibrium.
- Fixed fused source freezing on activation saturation, fractional warm-mask handling,
  per-row batch masks, and `GradedRule.replace` losing the `Adaptation` object.
- Warm starts rebuild published activity under the current mask on CPU, PyTorch and
  MLX, including after changing or removing a fractional mask.
- Receipt verification messages distinguish checks actually performed. Documentation
  separates records, transient dynamics and equilibrium, and states gradient scaling.

## 0.8.1 (2026-09-14)

- A checkpoint saved by an earlier release loads. `checkpoint.load` keeps the fields of the
  saved configuration that the running `LearnerConfig` has and drops the rest; 0.8.0 raised
  on the retired `consolidate` and `restore` of a 0.7 checkpoint. The test saves a
  checkpoint with the retired knobs in its metadata and loads it.
- The valence's and the population code's returns are typed as arrays, for the mypy of the
  CI runner.

## 0.8.0 (2026-09-14): the condensation

The library condensed into its elements (`docs/concepts.md`): owners and seams, the
settlement, the contrast, the trace, the valence, and one step. Dropped from the top level,
each with what replaces it. The gates it passed: the core suite with the child's
experiments; the four example rungs' receipts verifying under it; the NES player's
one-frame afterimage brain retrained on it reading the held-out presses at the same
0.862 / 0.325 / 0.579 bits as on 0.7.1.


- `Rehearsal`, `RehearsalConfig` (the clipped rehearsal learner): the trace and the valence
  of `ActorCritic`; E4 of the NES player measured the rehearsal a failure.
- `ValueNet`, `ValueConfig` (a separate value net as the critic): the critic is the
  settlement's own value readout, `ActorCritic(learner, critic=<owners>)`.
- `Population` (the continuous population code with Gaussian exploration): `Bins`.
- `DreamActorCritic`, `DreamConfig`, `DiscreteCode`, `actor_critic_wiring` (`dream.py`) and
  `Seams`, `SleepConfig` (`structure.py`, fast and slow strengths, pruning and sprouting):
  removed; the consolidation of `LearnerConfig` is the slow strength that stays.
- `ActorCriticConfig`: `critic_init` (the critic starts at zero), `lam_critic` (the critic's
  trace decays as the actor's), `normalize_floor` (a constant) and `center_per_stream` (the
  valence is per stream, always: one brain playing several games keeps a level for each)
  are gone; sixteen fields become twelve.
- `LearnerConfig.scale_floor`, `scale_cap`, `target_level`, `off_level`: never set by any
  script in five repositories; the cap is the constant `cadence.learning.SCALE_CAP` (8), the
  nudge's targets one and zero, and a seam may cross zero.
- `LearnerConfig.consolidate` and `restore` (a slow copy of the seams the fast ones are
  pulled back toward): removed; the NES player's E4 measured it no better than without at
  the settings tried, and the trace and the valence carry what it was for.
- `Ledger`, `settle_owner_by_owner`, `canonical_sha256`, `source_manifest`, `mutate`: still in
  `cadence.reference`, `cadence.receipts` and `cadence.constitution`, no longer top level.

## 0.7.2 (not released; shipped in 0.8.0)

- `Trace`: one class for the memory of the moment before, per stream, clamped into the next
  settlement: the trace of a range's activation, weighted by each owner's movement since the
  last moment when `focus` is above zero (`source` and `target` name the ranges). `Echo` is
  the `Trace` at focus 0 into the `context` range and `Afterglow` the focused one into the
  `afterglow` range, both unchanged in use; `ringing` moved to the `Trace`.
- `Valence`: the reward less its expectation as its own element (the running level per
  stream, the reward's own units or its scale, the floor, the cap); `ActorCritic.valence` is
  the agent's, built from its config, and `delta_mean`/`delta_var` read through to it.
- The first two steps of the condensation planned in `docs/concepts.md`; the API doc and the
  README carry the new names.

## 0.7.1 (2026-09-14)

- `Afterglow`: an owned state like `Echo`, a fading picture of the preceding moments that
  is brightest where they changed: the trace of the hidden equilibria weighted by each
  owner's movement since the last moment (its share of the mean movement, to the power
  `focus`), entering as a clamp on the `afterglow` range. What the settlement just had to
  repair stays lit; what stood still fades. With `focus` 0 it is the Echo. `source` names the
  range traced: the hidden owners (the interpretation), or the input owners for an afterimage
  of the picture itself, which is the one that works: predicting the symbol seen one moment
  ago against twelve owners of always-on background, the echo reads chance (0.25), the
  afterglow of the interpretation 0.30, the unfocused afterimage 0.58 and the focused
  afterimage 1.00 (`tests/test_child.py`).
- `ActorCritic.salience` and `Afterglow.ringing`: the eligibility of every seam weighted by
  its pre owner's salience, set before `learn`; the afterglow's ringing (each source owner's
  trace over the row's mean, plus a floor) is one such salience, so that what is still
  ringing is what a signal writes through. Off unless set. In the small experiment (a cue
  paid three moments later against twelve owners of always-on background) the plain trace
  learns the cue too, since the background's eligibility is noise that averages out, and the
  weighting is a faster start on some seeds and no difference on others: a mechanism, not
  yet a gain.
- `tests/test_child.py`: the child's capabilities as small experiments, each from simpler
  components already here: the eligibility trace credits a press paid three moments later
  (hit rate 1.00 with the trace, 0.47 without); the dopamine that is quiet for the usual
  reward and speaks for a missing or a larger one, in proportion; the afterimage above; and
  the settlement as a memory (a capped repair from the previous equilibrium still holds the
  moment before, a full repair forgets it).
- `ActorCriticConfig.center_per_stream`: the centred dopamine keeps one running mean and
  scale per stream instead of one over the batch, for streams on different tasks (one brain
  playing several games, whose rewards differ in size); the batch-wide centre is unchanged
  and stays the default.
- `ActorCriticConfig.dopamine_floor`: the centred dopamine within that many scales of its
  mean is nothing, so the broadcast is quiet while the reward is what it usually is and
  speaks only for a surprise; with the floor at 0 (the default) nothing changes.
- `ActorCriticConfig.center_scale`: whether the centred dopamine is divided by its running
  scale (the default, a unit signal whatever the reward's size) or left in the reward's own
  units, so a stage cleared is ten coins and not one.
- `LearnerConfig.momentum` and `normalize` are the adaptive local step proper: Adam's order
  (the running average of each seam's own contrast, divided by the RMS of its raw contrast)
  with both corrected for their short history, as `ActorCritic` already did. Before, the
  first steps were inflated and the two were composed the other way round, which is why they
  hurt on Pong; the corrected step lived in that rung's script only.
- `Learner(slots=(9, 2, 2))`: slot sizes instead of a count, for a controller whose
  choices differ in size (a move of nine, a grip of two); `slot_sizes`, `slot_offsets`;
  the checkpoint keeps the slots.
- `mutate(..., tied=((leader, follower), ...))`: region pairs whose sizes move together (a context range with one owner per hidden owner); `evolve` passes it through.
- `LearnerConfig(consolidate=, restore=)`: a slow copy of the seams and biases that follows them by `consolidate` per update and toward which every update pulls them back by `restore`, on the host and on the device: what is learned and kept settles into the slow copy, what is learned and then unlearned drifts back to it.
- `ActorCritic`: streams settled on the torch kernel keep their eligibility traces and the step on the device (`contrast_rows`: each stream's own contrast (the contrast through `contrast_tensors`, the step through the learner's device-side apply); the host arrays are never made. A reward-learning decision on a 7.2-million-seam brain: 121 ms on the Mac's GPU against 320 ms on the host.
- `Settlement.with_parameters` on the host engine: a copy whose derived arrays are remade on first use, not a rebuilt engine (60 ms a decision at 7.2 million seams).
- `Constitution.from_dict`: a constitution read back from a lineage's record, the inverse of `to_dict`.
- `evolve(..., report=callback)`: the lineage so far after every generation, so a long run is written out as it goes.
- `evolve(..., mapper=map)`: a generation's lives run through `mapper`; pass a pool's `map` to run them side by side (the fitness must then be a module-level function).
- `FastSeams`: fast Hebbian seams between two ranges of a batch of streams, as owned state
  (an outer-product write per step, a read as a drive, a decay); `columns` (slices for
  contiguous ranges) exported from `cadence.stream`.
- The torch kernel: the block weights scattered on the device through an edge index kept
  there; a settled state fetches its host arrays only when something reads them; no host
  copy of a warm state; the per-overlap arrays uploaded only on the gather-scatter path;
  no reference cycle in the device handle. The seam update addresses the paired and the
  tied overlaps through index arrays made once. A wiring keeps its segment boundaries.
  At 7,300 owners and 10.8 million overlaps, batch 2,048 on an A10G: one learning update
  4.4 s to 0.73 s.
- The learner's update stays on the torch device when both phases rest there: the contrast,
  the masks, the pairing, the tying, the decay and the bounds as device tensors, the
  kernel's blocks updated in place, and `Settlement.edge_scale` and `bias` fetched only
  when something reads them (a checkpoint, a receipt, a life of the seams). A test checks
  it against the host update to 1e-9. The seam update at the shape above: 265 ms to 40 ms.
  `normalize` and `momentum` take the host path.
- `Echo` reads and writes its ranges through slices.

## 0.7.0 (2026-09-11)

The accelerators, in the core.

- `backend="mlx"`: the settlement on Apple silicon through MLX (`pip install
  "cadence-net[apple]"`): the block transport as device matrix products in float32 on the
  unified memory, nudges, adaptation, masks, tolerance, trajectories and `repair` as on the
  other backends; checked against the CPU engine to 1e-4 with every feature.
- A device handle on `SettledState` (`state.device`): a settlement that continues from a
  state the same engine produced starts from the device copy instead of uploading the host
  arrays, and `Settlement.contrast_on_device` gives the learning rule its per-overlap
  contrast as block Gram products on the device, so only one number per overlap comes back.
  `Learner.contrast` uses it when both phases carry the handle; on torch/CUDA float64 the
  learned scales agree with the CPU engine to 2e-16, on MPS and MLX float32 to 2e-7.
- `available_backends()` lists `mlx`; the backends doc has a hardware guide (Apple silicon,
  Intel and AMD CPUs, NVIDIA) with measured updates at language-model shapes.

## 0.6.0 (2026-09-11)

The cost of a settlement step is now the cost of the owners that move.

- `cadence.blocks`: the block transport. The overlap matrix is stored as dense blocks
  between the contiguous ranges the wiring's named sets cut, and the product of a range
  whose activation did not change since the previous step is reused. The clamped inputs of
  a layered net are multiplied once per settlement instead of once per step; the inbox is
  the same sum in a different association order, and the conformance check against the
  owner-by-owner reference still holds to rounding (2e-16). `Settlement(..., layout=)` and
  `Settlement.layout`; `to_dict()` reports the layout. The `dense_limit` now bounds the
  block entries, so a layered net of several thousand owners settles on blocks.
- The fused kernel skips owners whose potential did not move and freezes ranges that hear
  nothing once they are still (exact: the update of such an owner is a fixed function of
  its own state), and no longer recomputes the activation of a state it continues from.
- The learning rule's contrast is one small Gram product per block, and one product
  instead of two for a range that is the same in both phases.
- On the MNIST shape (784 inputs, 256 hidden, batch 256) one learning update went from
  63 ms to 18 ms on one M4 core, and from 39 ms to 7 ms at 32 hidden owners; before, the
  cost of an update barely depended on the hidden width because the input block dominated.
- `cadence.timing`: `latency(decide)` times one decision many times and reports the median,
  the tails, and the scheduler's context switches from `getrusage`; `environment()` records
  threads, pinning (Linux), load and library versions for a receipt.
- `Settlement(precision=)` on the torch backend: float32 on CUDA when speed matters more than
  the receipt (read out on the cpu backend), float64 where the device has it.
- `cadence.stream`: owned state as a clamp. `stateful(...)` builds a windowed net with a range
  of context owners, one per hidden owner, wired densely into the hidden owners; `Echo` keeps
  a leaky trace of the hidden owners' equilibria across a batch of streams and writes it into
  the context clamp, so each state reverberates and fades. A test learns to name the symbol
  seen one input ago from a window of one, which no window can.
- `SettledState.repair`: the total movement of the published activations during a settlement,
  one number per row, on every backend. It is the work the net did to get from where it was
  to rest: the surprise of an input, measured rather than inferred from the step count.
- `Learner(slots=)`: the output owners as equal groups, each its own softmax choice, all nudged
  together; `targets` takes one label per slot and `predict` returns one choice per slot. A
  whole utterance settles at once. Each slot's nudge carries `beta / slots`, so the contrast
  is the gradient of the mean loss over the slots and `eta` means the same at any count
  (a span of four at full beta drove the hidden owners into saturation).
- `cadence.constitution`: where a wiring comes from. `Constitution` (regions and projections as
  a few numbers each), `grow` (development into a wiring with named sets, deterministic in
  the seed), `mutate`, `evolve` (selection over constitutions under a fitness the caller
  supplies). The phase before learning; not a new primitive.
- CI is green again: the 0.5.0 modules are formatted and typed.
- Tests: 58.

## 0.5.0 (2026-09-10)

Learning from reward and a life for the seams, built for the paper "You don't need attention
after all" and measured on its gates.

- `ActorCritic` (`cadence.plasticity`): the three-factor rule. An eligibility trace at every
  seam of the free/nudged contrast for the action taken, a linear critic on named owners
  with its own trace, a dopamine owner broadcasting the temporal-difference error; the
  adaptive local step (`momentum`, `normalize`, bias-corrected); `dopamine_cap`,
  `dopamine_center`, `critic_normalize`; `bootstrap=` for time limits. `Population` for
  continuous actions as a bump code. Cart-pole: 500 on every seed with the threshold at 40k
  to 60k steps, 1,716 parameters, two warm settlement steps per decision at deployment.
- `DreamActorCritic` and `actor_critic_wiring` (`cadence.dream`): an actor and a critic in
  one net with a memory; imagine-and-feel action selection (candidates felt in the critic),
  the critic's dream toward a target read with the slow strengths, the actor imitating the
  action taken. `DiscreteCode` for discrete actions.
- `Seams` and `SleepConfig` (`cadence.structure`): fast and slow strengths, tags, sleep
  (consolidate, downscale, prune, sprout within a budget), conserved incoming strength.
- `cadence.fused`: a compiled dense settlement kernel and a fused three-factor step on the
  CPU backend when numba is installed (`pip install "cadence-net[fast]"`); identical
  arithmetic, checked against the NumPy loop to 2e-16; `CADENCE_FUSED=0` forces the loop.
- `Learner.apply` and `Learner.contrast_rows`; the contrast as a Gram matrix product read
  at the overlaps (forty times faster than the gather on dense wirings).
- Docs: `reward.md`, `life.md`; the capability table carries the gates.
- Tests: 50.

## 0.4.1 (2026-09-09)

- Docs only. The grey parrot rung was withdrawn from the application demonstrations (its imitations did not
  reach the bar); `embodied.md` now works through cart-pole and the sign writer, `tasks.md`
  keeps the several-learners-in-one-net recipe without the parrot, and the examples table
  lists the nine rungs.

## 0.4.0 (2026-09-09)

- `Learner.save` / `Learner.load` (`cadence.save`, `cadence.load`): one-file checkpoints of a
  trained learner, wiring and parameters included, that load on any backend and keep learning.
- `LearnerConfig.decay`: a leak on the seams, so a net that never stops learning stays plastic.
- `LearnerConfig.momentum`: each seam steps on a running average of its own contrast.
- `Learner(trainable_owners=...)` beside `trainable_overlaps`: two learners can share one net,
  each moving and decaying only its own seams and owners; a frozen overlap never moves, not
  even through the seam tying.
- `Learner(tie_groups=...)` and `embedded(...)`: a shared embedding across window positions.
- `py.typed`: the package is typed; `mypy --strict` clean.
- Tests: 42 across every module, 96% line coverage; the torch kernel is checked against the
  CPU engine with nudges, weights, adaptation and trajectories.
- Docs: task recipes (tabular place codes, regression as a pattern, streams, few labels, several
  learners in one net), `embodied.md` for deployment.

## 0.3.0

- The free/nudged learning rule, layered wirings, the torch backend, receipts, the examples ladder.
