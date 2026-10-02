# Connected graphs: capability and cost

Start with the least expensive connected graph that learns the required
behavior. Population sizes, intermediate populations, branches and error
readback change capacity and cost. They all use the same patch law, joint
repair and final stationarity check. More depth does not guarantee better
reasoning; more repair time does not establish more thought.

The [wiring examples](VARIANTS.md) introduce these choices. Optional error
readback and automatic-attention claims have the
[experimental boundary](EXPERIMENTAL.md). This page explains costs and source-bound
evidence. Recorded benchmarks retain their original sources and versions,
including 0.50.0; they are not automatically new-version results.

## What the demo history calls a patch

Several different mechanisms appear in the demo history. Count the computation
actually executed, rather than treating every use of "patch" as the same model.

| Mechanism | Computation for an answer | Relevant distinction |
| --- | --- | --- |
| Legacy gated record patch | An explicit context update, record lookup and readout | May retain temporal state without an observer hierarchy or iterative DRSN settlement |
| Input-only population (refused since 0.61.0) | Independent predictions from fixed sensors, with a separable state objective | No state or error contacts between patches: not a brain |
| Current coupled composition | Populations read other populations' live states | States are coupled even with no error-reading observers |
| Current recursive observation | Observers read live states and exact errors; observers may observe observers | Error dependencies participate in the same coupled objective and returning derivatives |

Patches within one population do not read each other. A population that reads
only sensors and is read by nobody therefore settles nothing between patches;
each of its patches is a bounded nonlinear function of an affine sensory
combination, and the group is a set of independent regressions. Since `0.61.0`
the builder refuses such a layout. Add a population that reads it, or let it
read one, and test whether state-and-error observation adds value beyond that
coupling. See [layout choices](VARIANTS.md).

A loop through an environment, temporal context inside a patch, game-tree
search and recursive observation are also different kinds of recurrence.
A model can have the first three without an observer observing an observer.

## Why an uncoupled population is not a brain

During a query, parameters and sensory inputs are fixed. If a patch reads
only those inputs, its prediction `p_i` is independent of the adjustable
states, and if nothing reads the patch, nothing else depends on its state.
Its term of the energy separates into a scalar quadratic:

```text
p_i = tanh(b_i + sum_j w_ij * input_j)
E_i = 1/2 * (x_i - p_i)^2 + state_prior/2 * x_i^2
x_i* = clip(p_i / (1 + state_prior), -state_bound, state_bound)
```

This is the exact minimum for such a coordinate. A population of such patches
has a closed-form equilibrium, but no disagreement repair between patches.
The builder excludes that uncoupled model, as well as disconnected groups of
populations. Its speed does not measure the cost of connected settlement.
A sensing population that another population reads
keeps the same closed-form prediction `p_i`, but its states now also carry the
reader's constraint, and the joint repair settles them together. The query
cache described below reuses only the invariant predictions; the states, errors
and returning derivatives remain live.

This shortcut does **not** apply to coupled queries, or to learning that makes
weights and biases eligible.

## Where coupled settlement spends time

State contacts make one prediction depend on other adjustable states. Error
contacts add derived signals whose changes must be propagated through the
objective's derivatives. A full answer requires the stationarity test across
all eligible coordinates, with predictions and errors recomputed exactly.

The Python reference traverses patches and edges in forward and reverse order
for each energy/gradient evaluation. That work is roughly linear in the graph
size per evaluation; recursive observation does not inherently require an
exponential traversal. Total cost also depends on how many evaluations are
needed. Conditioning, weights, state initialization, clamps and tolerance can
change accepted sweep counts and rejected line-search proposals substantially.
An ordinary state-coupled layout can already be much slower than an input-only
one. Error-reading depth is one contributor, not the explanation for every gap.

A useful accounting is:

```text
query time = validation/setup
           + objective/derivative evaluations and repair proposals
           + final qualification and output construction
```

A reference solve that attempts no repair proposal uses its fresh initial
full evaluation for qualification, counted once. If it attempts a proposal,
it performs a fresh final evaluation even when that proposal was rejected.
This reuse stays within the same call; every new call checks the current full
graph. A retained activity state can save repair work without being learned
attention or guaranteed temporal memory.

Count `work.evaluations`, `work.proposals`, `work.backtracks`, edge/patch visits,
sweeps and wall time. Sweep budgets do not bound elapsed seconds. Tensor
execution groups independent work within error-dependency levels; more such
levels can add sequential stages and dispatch overhead. CPU/GPU overhead and
final float64 checking matter especially for small graphs. See
[execution and precision](ACCELERATION.md).

## A bounded comparison on Cadence 0.50.0

The [query-cost example](../examples/layout_cost.py) and its
[source-bound receipt](../examples/receipts/layout_cost.json) compare four small
layouts through the public API. All use six patches, four numerical inputs,
two output coordinates, seeds 2/7/17, twelve inputs per seed, Python float64,
state prior `0.01`, tolerance `1e-6` and a 512-sweep budget. They make pure queries
from unchanged initial activity, with no teaching or task-quality score.

| Layout | Edges / parameters | Median query ms | Median accepted sweeps |
| --- | ---: | ---: | ---: |
| Input-only (refused since 0.61.0) | 24 / 30 | 0.180 | 2 |
| Ordinary composition | 16 / 22 | 0.426 | 8 |
| Ordinary composition with sensory skip | 24 / 30 | 0.429 | 8 |
| Two observer levels | 24 / 30 | 0.404 | 6 |

These are small local measurements on macOS 15.3 arm64 with Python 3.13.0,
under shared-host load, not portable latency promises. All 144 calls qualified.
The input-only, skipped composition and recursive layouts match patch, edge and
parameter counts; their wiring and representational capacity still differ. The skipped control adds direct sensory access to the
last population to match connection count, which is itself an architectural
choice. Identical counts do not make the learned functions identical.

The input-only arm is fastest in this screen, as its closed form predicts; it
is a cost reference, not a brain. Recursive is slightly faster than the composed
controls at the median. Observer depth alone does not predict the measured
latency: wiring and convergence matter too. The receipt preserves each call,
refusal/error outcomes, work, timing, runtime and source hashes. It demonstrates cost mechanisms,
not a capability advantage or a real-time guarantee.

Run the example from the repository root. Choose a fresh output path; the
runner refuses to overwrite a receipt and records its protocol before measuring:

```sh
PYTHONPATH=src python examples/layout_cost.py --out /tmp/cadence-layout-cost.json
```

## What the older demonstrations establish

### AMEN: a fast temporal record composer

The classic [AMEN demo](https://github.com/muellerberndt/cadence-demos/tree/main/amen)
ships a brain trained on Cadence 0.11.0 and parity-checked under the examples'
0.12.0 pin. Its browser performs a gated linear context update, sparse record
read and linear output readout. This is temporally stateful computation with
one explicit update per event; it is not the 0.50 joint state-and-error solver.
Its record writes are disabled during page generation.

The page computes an event sequence, renders the audio offline and then plays
it while displaying stored activity. Fast generation is meaningful, but the
playback frame rate does not measure a live sensory-to-action deadline.
The supplied event representation, instrument and playing rules also contribute
to the result. Comparisons with later recursive composers must account for those
interfaces, checkpoint quality, hardware, implementation and learning costs.

### Connect Four: a cheap value query inside a search system

The [shipped Connect Four example](https://github.com/muellerberndt/cadence-examples/tree/main/connect4)
uses a legacy 0.12 record patch as a board evaluator. It resets context for each
board, and the exported record table is empty. Its ordinary value query reduces
to a gated hidden representation and linear readout, without an observer ladder
or an iterative DRSN solve.

The complete player also supplies legal moves, tactical wins/blocks, negamax
search, alpha-beta pruning, transposition caching and iterative deepening.
The [sealed browser receipt](https://github.com/muellerberndt/cadence-examples/blob/main/connect4/web/receipt.json)
contains complete decision timings: opponent-dependent means of about
224–399 ms, p95 values of 604–2,281 ms and maxima of 3.24–8.36 seconds.
Those are search decisions, not single value-network calls. The deployed
search can request thousands of values for one move.

This is substantial capability from offline learning plus search. The player
was prepared from 4.44 million positions and does not learn from browser games.
Its search recursion is not recursive neural observation. The rules-only control
uses a different search implementation and budget, so it does not isolate the
learned evaluator's contribution at equal work. Separate old deep pilots and
unreceipted 0.50 development code do not establish a matched recursive upgrade.

Other older examples include temporal patches, sequential supervisory loops
and recurrent equilibrium networks. “All older demos are non-iterative flat
networks” is not an accurate common description. Identify the deployed version,
checkpoint and actual execution path before comparing latency.

## Does recursive observation buy more capability?

It can represent interactions involving internal states and their errors that
an independent input-only output cannot use. Whether learning makes those
interactions useful is an experimental question. Coupling, useful learned
representation and useful recursive observation must be tested separately.

Two archived task comparisons show why the inexpensive baseline matters. The
[historical evidence excerpt](../examples/receipts/layout_cost_history.json)
preserves the versions, source hashes, task conditions and numerical summaries.

- **CartPole, Cadence 0.43.0:** one flat patch, six flat patches, and four
  processing plus two observer patches each completed all 60 reserved episodes
  at the 500-step ceiling. Mean decision times were 0.038, 0.109 and 6.166 ms
  respectively. The observer used about 98.72 accepted sweeps per action, versus
  three for the flat layouts, and had 40 rather than 24 edges relative to flat
  six. All received 128 supplied-state teaching examples per model; one patch
  could represent the teacher's control relation. The extra observer work did
  not improve this already saturated task. Shared-host observer calls reached
  roughly 160 ms, illustrating why the mean is not a deadline.
- **Changing-body stream, Cadence 0.48.0:** flat/composed/recursive model times
  averaged 0.1783/20.9222/23.9354 seconds per complete stream, including queries,
  teaching and off-policy goal probes. Recursive mean absolute forecast error
  was 0.15120 versus flat's 0.15314; recursive cost was about 134 times flat's.
  The ordinary composed model already incurred most of that extra cost and had
  MAE 0.15062. About 23.72 of the recursive model's 23.94 seconds were learning.
  Its actual-input forecasts took 0.1531 seconds per stream versus flat's
  0.0103 seconds, excluding their other query categories. The 134-fold total
  ratio must not be reported as an inference-speed ratio.
  Information and experience were matched, while parameters and
  computation were not. A tested MLP was faster and better on squared error
  and goal error. These are system-identification results with an external
  probe controller, not autonomous robot-control deadlines.

These examples support fast small learners on some useful tasks. They do not
establish that recursion always helps or never helps. A claim that a recursive
model is “smarter” should name the improved task, the control, the training
budget and the additional inference/learning cost. For Connect Four, hold the
school data, reserved positions, search code and search budget fixed; measure
both prediction quality and complete playing strength.

## Choosing a layout for a real-time application

The current query implementation reuses predictions that depend only on
fixed sensory inputs during one repair. This also applies to sensory populations
inside a recursive graph: their predictions are constant, while their states,
errors and feedback remain live. Initial and final checks still traverse the
complete graph. Reuse changes neither the patch law nor the number of required
repair sweeps. It introduces no caller configuration and keeps no cache between
observations. Measure elapsed time as well as edge visits: cache setup and copies
cost time but are not edge traversals. This numerical optimization is not learned
attention or evidence of improved behavior.

1. Establish the smallest coupled baseline with the actual available sensors:
   a sensing population read by the output population. Measure useful behavior
   after bootstrapping, including refusal and failure cases.
2. Add composition if the direct mapping lacks a useful representation.
   Keep recursive layouts as a separate opt-in experiment, with capable
   state-coupled controls before attributing any gain to error readback.
3. Keep width, edge/parameter counts, history, target access, starting checkpoint,
   tolerance and backend visible. Match relevant capacity and information; also
   compare within the same elapsed-time budget. Include all training selection.
4. Measure complete commands: preprocessing, every candidate query, search,
   replay/learning, final qualification, queue delay and action delivery. Report
   latency distributions and deadline misses, not just an average inference rate.
5. Compare repeated pure `settle` queries with the intended `step` continuation
   policy explicitly. Retaining a useful state can change subsequent settling
   cost; it is not a weight update or evidence of protected long-term memory.

The intended fast/slow design keeps one brain and one body interface: familiar
behavior stays inexpensive, and internal correction receives more work when
needed. This release does not implement that automatic allocation.
Every query must still qualify the complete connected brain; an application
scheduler cannot certify a partially solved branch. `LiveController` keeps
rendering responsive while a serial worker owns the brain; it does not make an
unfinished answer ready sooner. See
[experimental scope](EXPERIMENTAL.md) for the capability boundary and
[live control](LIVE.md) for execution.
