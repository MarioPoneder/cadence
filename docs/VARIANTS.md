# Cortex variants: flat, deep, wide

A variant is a wiring, not a different algorithm: every variant runs
the same element, the same settle law, the same admission custody.

## The menu

| Variant | Wiring | When |
| --- | --- | --- |
| Flat | `depth=0`: the finest level alone | Ablation baseline; small fully-observed tasks. Expect no cross-context generalization. |
| Global-prior | `depth=1`: finest + per-action global prior | The cheapest hierarchy; enough when one action is globally better. |
| Deep (h2) | `depth=2`: finest, 4-bin coarsening, global | Default. Coarse level generalizes early, fine level refines. |
| Deep (h3) | `depth=3`: finest, 16-bin, 4-bin, global | More regional structure; helps when regions differ strongly. |
| Custom depth | `ladder=(16, 8, 4)` etc. | Explicit intermediate bins over the primary controllable byte. |
| Wide | `width=N`: finest level reads top-N controllable + top-N drive bytes | When one byte pair cannot express the task (see Pong below). |

Depth multiplies levels; width multiplies the finest level's context
cells. Witness mass per level is automatic: a level with k-fold fewer
cells than the finest admits each target at 1/k mass, so all levels
share one evidence timescale.

## Measured evidence (development receipts, 2026-09-28)

Freeway (ALE, RAM, frame-skip 4, sticky actions off; 24 episodes per
run; controls 8): random and frozen-memory controls scored 0.00 in all
runs. Last-8-episode mean crossings per two-minute episode:

| Config | Seed 1 | Seed 2 |
| --- | --- | --- |
| flat | 14.6 | 7.5 |
| deep h2 | 16.0 | 19.5 |
| deep h3 | 14.2 | 19.4 |

Where flat struggles (seed 2), the hierarchy roughly doubles it; the
always-UP ceiling is about 21. An earlier protocol without witness-mass
weighting reached 22.0 sustained (peak 26, above the ceiling) on one h3
run but oscillated on h2; both receipts are retained in the
development workspace (cadence-atari lane) with per-episode returns,
counters and source hashes.

Pong, width 1: every variant pinned at -21, worse than random (-19.3).
One controllable byte and one drive byte cannot express "paddle must
meet ball", and sitting still is the learned local optimum. This is
the measured reason `width` exists. A width-2, depth-2 probe at 16
episodes still sat near -20.5: raw extra bytes are not enough at
small budgets, and relational wiring (for example, ball position
relative to the paddle) is an open, undelivered revision. Treat
Pong-class games as unsolved by this wiring until a receipt says
otherwise.

## The four structural rules

These live in code, not in parameters; each was locked in after a
measured failure on the corridor and Freeway fixtures:

1. Mean-grounded bootstraps; novelty bonus enters a target once,
   scaled by (1-discount). Unscaled or bootstrapped bonuses build a
   reward-free self-sustaining value web (observed: values 3.4+ on a
   corridor whose true optimum is ~1.4, then policy collapse).
2. Dispersion-floored priors: a coarse level's transferable precision
   is capped by its between-target scatter, or the global average
   flattens every region (observed: all values pinned near the global
   mean, policy dead).
3. Backward exactly-once episodic admission: one witnessed reward
   welds the whole chain in one flush; nothing is recounted as fresh
   evidence (forward one-step admission stranded rewards).
4. Witness mass per level: without it, one 2,048-step episode washes a
   coarse column's window many times over (observed: alternating 21/0
   episodes on Freeway h2).
