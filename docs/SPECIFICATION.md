# Behavioral specification

This page separates the shipped contract from a proposed general intelligence
architecture. The [reference](REFERENCE.md) defines argument types and defaults;
the [element guide](ELEMENT.md) gives the equations.

## Model boundaries

- A scalar `CorticalColumn` retains sufficient statistics and an event cursor.
  Its live reciprocal loop settles a belief variance and observer precision.
- A `Cortex` indexes evidence columns by supplied contexts and output channels.
  Local observer loops are reciprocal; coarse-to-fine priors are directed.
- Exact factor federation operates on supplied binary tables over a certified
  cluster forest. It neither learns those tables nor imports them into scalar
  evidence automatically.
- Custom context maps, sensor normalization, likelihood families, goals, action
  meanings and reward construction are application-supplied choices.

These components share port repair machinery. That does not make their model
families, state formats or scientific guarantees interchangeable.

## Observations and memory

Actual observations enter through explicit admission methods. Predictions,
queries and imagined values are not independently witnessed events. Ordered
identifiers support retry handling; they do not prove that the caller's data
are true or statistically independent.

Admission first forms prospective statistics and checks the required solve and
resource conditions. A failed scalar observation or Cortex native multi-output
observation must not partially replace retained evidence. The latest identical
retry is a duplicate; changed or out-of-order evidence must be handled as an
error. Native `observe` and the optional reinforcement-learning transition
wrapper have separate documented target meanings.

Read operations do not acquire new witnesses. A Cortex may populate caches and
advance work counters while answering, but does not retain new context columns.
Scalar column queries leave their retained checkpoint state unchanged. Neither
form of read implies an external world-model update.

## Qualification

A solve is qualified against its declared message equations and residual
tolerance. A capped solve may be returned as unqualified; high-level prediction
and action selection reject such results. Action selection qualifies all output
beliefs before either random exploration or scoring. Damping and more sweeps
change the numerical execution budget, not the target equations.

Qualification is not an accuracy score, a confidence calibration theorem,
proof of a global minimum or proof of uniqueness for arbitrary port graphs.
The numerical `variance` and `novelty` fields are model-derived quantities;
they are not automatically calibrated confidence or universal measures of
ignorance or safe action risk. Coarse and fine levels can reuse correlated
evidence; their witness weights do not make those observations independent.

For lesion diagnostics, distinguish the altered equations' residual from the
intact model's residual. A lesion can be a valid solution of a different model.
Full nonlinear biological circuit equivalence is not implied by a successful
finite factor calculation or a local perturbation test.

## Bounded operation

Event capacity, statistic-size limits, context-column count, cached answers,
pending transitions and settling work have explicit budgets. Resource refusal
must remain visible. A limit on item count is not a guarantee of a fixed process
RSS, a hard real-time deadline or bounded application-owned sensor data.

Configuration is fixed for a model's lifetime, apart from the documented
learning and prior-cut control switches. Construct a new model for a different
likelihood, wiring or numerical contract. Changing a configuration must not
silently reuse answers cached under the old one.

## Continuation

Checkpoints bind retained model state and its declared configuration. Cortex
continuation also includes stochastic action state and pending transitions;
a checkpoint does not capture the external environment or arbitrary feature-map
code. Built-in feature descriptors support reconstruction. Custom maps require
matching code supplied by the application and a matching declared identity.

Restore validates the complete proposal before changing the live model.
Malformed or incompatible state must not partially replace a valid instance.
A checkpoint establishes declared-state consistency, not authenticity of the
application's original observations.

## Evaluation

An application claim needs task-level evaluation in addition to library checks.
Compare matched information and exposure, preserve failed solves, and charge
calibration, all message sweeps, admission work, storage and optional RL episode
processing. Context depth and observer height are separate interventions.

Library checks do not validate biological column identity, establish general
useful recursion or show generic efficiency over conventional learners. Those
are separate empirical questions, with a declared task and resource budget.
