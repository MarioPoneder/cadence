# Population-model cost

Choose the smallest connected graph that learns the required behavior. Population
sizes, state contacts and exact-error readback affect both capacity and cost.
Use the [wiring guide](VARIANTS.md) and measure complete learning and query work.
A successful numerical solve is separate from useful acquired behavior.

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
