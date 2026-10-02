# Scaling learned capability

Cadence aims to improve useful behavior through more experience and training,
without redesigning the brain for each application. A bounded observer-like
patch exposes local state, ports, records and readback; learning repairs its
relationships. Scaling this structure is an empirical requirement, not a
consequence of naming it a brain or reaching an equilibrium.

## Separate the causes of improvement

| Variable | Controlled comparison |
| --- | --- |
| Unique experience | Nested sets of distinct observations; equal update count and batch size. |
| Training exposure | The same training set and initialization, evaluated at predeclared update counts. |
| Model capacity | The same observations, ports, objective and evaluation at different hidden widths. |
| Temporal scope | Longer experienced sequences, counting both episodes and scalar teaching values. |
| Internal readback | Aligned feedback versus disconnected, shuffled and equally capable recurrent controls. |

Keep evaluation data separate, use several initializations and report each
paired result. Seeds on one dataset do not establish robustness across data
distributions. More steps with a short temporal objective need not teach a
longer task. Increasing width can worsen optimization or long-path behavior.

Before comparing neural models, establish simple controls: training-set means,
last-observation persistence, an appropriate fitted linear predictor and a
known-body controller where available. A context-sensitive prediction can still
lose to a constant. A copied target, new decoder, additional supplied intention
or expanded action vocabulary is a changed interface, not data scaling.

## Count the full work

Report attempted and committed updates, phase failures, parameter replays,
block-chain attempts, solver tolerances, persistent state and transient work.
Keep every scheduled seed in the report, including unattempted or time-capped
cells. Compare equal checkpoints; never substitute an early result for a
missing late checkpoint or compare unmatched averages as a paired effect.
Wall-clock limits depend on host contention, so include operation counts too.

The NumPy temporal implementation has dense hidden-width messages. For batch
size N, path length T and hidden width H, one detuned chain factorization costs
on the order of N*T*H³ operations. Its explicit message storage is
8*N*T*(H²+H) bytes in float64; Hessian blocks, arrays, clones and solver
workspace cost more. Free causal prediction avoids this factorization but has
a sequential dependency across time. A GPU backend for a different graph API
does not accelerate this temporal kernel automatically.

[Experimental connection masks](partitioned.md) do not make that solver sparse.
Count permitted parameters separately from allocated parameters and mask bytes;
equal hidden width still uses the same dense message dimensions. Zeroing
connections also changes the initial recurrence spectrum, and supplied input
routing changes information access. Neither is evidence of learned specialization.

Longer and wider runs therefore need measured numerical conditioning and a
verified implementation path before hardware scaling. Treat speed and task
quality as separate measurements. No general Cadence scaling law has been
established.

## Keep manual design visible

Record supplied encodings, timing, topology, sample banks, goals, loss weights,
selection rules and optimization constants. Distinguish them from learned
relationships. Prefer a correction that improves the same learning operation
across domains to another named task module. Parameter-step backtracking, for
example, can prevent overshoot without supplying a task's answer, but it does
not determine relevance or protect old experience.

A useful general improvement must eventually combine acquisition, retention,
correction and novel behavior. Establish these at small scale before a long
run. In creative applications, evaluate novelty, purpose, coherent development
and actual execution separately from prediction loss. Musical composition is
one application of this general contract.
