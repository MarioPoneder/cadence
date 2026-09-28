# The column and its repair law

A `CorticalColumn` is a small probabilistic model with retained evidence and a
live belief–uncertainty feedback loop. It is an observer-like software patch:
it owns state, exposes typed ports and readback, and checks its executed
relationships before admitting new evidence. The name is an architectural
analogy; these distributions are not established biological column equations.

## Retained evidence

An isolated scalar column retains three sufficient statistics:

- `w`: evidence weight, including an initial zero-centered prior.
- `s1`: weighted sum of witnessed values.
- `s2`: weighted sum of squared witnessed values.

For one admitted value `v` and decay `d`, its proposal is

```text
w'  = d*w  + 1
s1' = d*s1 + v
s2' = d*s2 + v*v
```

The public column stores these statistics as exact rational numbers. Its
belief messages are evaluated in floating point. Exact retained arithmetic
does not make the numerical settling step exact.

The retained state also includes the ordered event cursor and latest admitted
value. It does not keep a replay list of every observation. A duplicate latest
event does not add evidence; a conflicting or out-of-order event is rejected.
The proposed statistics replace the old ones only after qualification and
resource checks. A failed admission leaves the column unchanged.

`capacity=None` removes the fixed event-count limit, not all resource limits.
`max_statistic_bits` bounds the exact statistics; an application must handle a
resource refusal rather than assume unlimited continuous learning.

## The live scalar loop

Given the retained statistics, define

```text
m = s1 / w
E = s2 - 2*m*s1 + w*m*m

lower belief:       (m, V),  where V = 1 / (w*tau)
precision observer: tau = (prior_shape + w/2)
                         / (rate + (E + w*V)/2)
```

At height 1, `rate = prior_rate`. The lower belief publishes its current
variance `V`; the observer reads that value and returns precision `tau`.
Changing either message changes the proposal on the other side. Both remain
live during the same solve. The isolated column's mean is the retained
weighted mean; this uncertainty loop does not independently alter it.

Height 2 adds a rate observer above the precision observer. Further heights
repeat this construction. Each additional rate message has the form

```text
rate_out = fixed_anchor + meta_shape / (rate_in + live_value_below)
```

The highest stage uses the supplied `meta_rate` boundary. Fixed positive
anchors are part of this declared model; they avoid the unanchored chain's
zero-rate degeneracy. These stages change the settled uncertainty. Their
existence does not establish useful metacognition or improved learning.

## What settlement checks

`settle` works on named directed ports. Each source implements
`emit(port, inbox)`, which proposes a message from its local state and incoming
messages. A sweep visits ports in order, using the most recently available
messages. Scalar and moment messages may be damped; exact table messages use
undamped updates.

After a solve, the residual is measured by re-proposing every message against
the returned inboxes:

```text
residual = max over ports distance(proposed_message, returned_message)
```

The scalar distance is absolute difference; the moments distance is the
maximum absolute difference of mean and variance; exact tables use equality.
A sweep cap is a work budget, not a convergence assertion. Qualification
requires finite messages, a residual within tolerance and a freshly recomputed
executed sequential sweep whose largest change is also within tolerance. A
nonempty graph must execute at least one sweep. Table messages require exact
equality even if the caller selects a large tolerance. Qualification does not
prove a unique solution, a global energy minimum or successful task behavior.
Arbitrary custom port networks do not inherit a convergence theorem.

The intervention helper separates two checks. `executed_residual` tests the
actually executed equations, including a declared lesion. `full_residual`
tests the intact equations. A cut network may settle under its own altered
model while violating the intact one. The separately reported `stationarity`
measures one further intact sequential sweep; it is not task accuracy.

## A Cortex is a composition

A Cortex stores a bank of evidence columns indexed by context, level and
output. Each level runs a reciprocal local belief–observer loop. A coarser
level also sends a prior to the next finer level. That cross-level connection
is directed: the coarse level does not read the finer level's uncertainty.
The active context chain for one output settles together. Different outputs
have separate evidence and solves; sharing an observation does not introduce
cross-output constraint messages. Multi-output admission coordinates the
commits after those solves qualify.

The finer belief combines its own evidence precision with the coarse prior:

```text
p_own   = w*tau
p_prior = coupling / (V_coarse + scatter_coarse/w_coarse)
m_fine  = (p_own*m_own + p_prior*m_coarse) / (p_own + p_prior)
V_fine  = 1 / (p_own + p_prior)
```

At the coarsest level, or with `coupling=0`, the prior contribution is absent.
The added coarse scatter limits how much a precise estimate of a global
average can dominate a different local context.
The same witness can contribute to both levels, so this precision fusion is a
declared shrinkage rule, not a calibrated Bayesian posterior from independent
datasets. Witness-mass weighting does not establish that independence.

The finer observer reads both its live variance and the displacement of its
fused mean from its own evidence mean:

```text
tau_fine = (prior_shape + w/2)
           / (rate + (E + w*(V_fine + (m_fine - m_own)**2))/2)
```

Thus a prior-induced change in the mean also enters the same local observer
loop. The coarser level still receives no feedback from the finer one.

Context maps and their resolution are supplied representations. When maps
provide context counts, level witness mass is scaled relative to the finest
map. Neither this weighting nor hierarchy construction discovers a task's
useful representation automatically. Cortex's retained statistics use its
own floating-point evidence implementation; they are not the scalar column's
exact rational checkpoint format.

Native `observe` supplies actual output observations. The optional `learn`
interface constructs action-value targets using supplied rewards, a discount,
a novelty bonus and bootstrapping. That temporal-difference wrapper is an
additional learning policy, not a consequence of the column equations alone.

## Exact finite federation

`solve_cluster_forest` applies exact sum-product messages to finite binary
factor tables with integer or `Fraction` potentials. It checks that the cluster
graph is a forest and satisfies running intersection: clusters containing the
same variable must remain connected through separators containing it.

Under that structure, qualified messages yield exact normalized cluster
beliefs. Cyclic cluster graphs, violated running intersection and zero joint
support are rejected. A cluster may contain a cyclic relationship internally;
its complete table must already express that relationship. This is not an
exact solver for arbitrary loopy graphs, and dense cluster tables can grow
exponentially with scope size.

The factors are supplied, not learned by this API. Solving them does not write
a scalar column's retained evidence. The shared machinery is the port repair
law; the scalar evidence model and exact finite factor model remain distinct
families.

## Scope of the model

Library tests check admission, message equations, resource handling, checkpoints
and finite factor inference. Those checks do not establish general intelligence,
biological column identity or a performance advantage from additional stages.
Application learning, retention, useful recursion and efficiency need their own
controls and measured outcomes. Different configurations are different models;
qualify the state returned by each executed solve.
