# The Doom campaign report, 2026-09-29

One day of work on a single question: why did a 336-patch settlement
network trained on 20,000 teacher witnesses play Doom as if it had
learned nothing, and what does the answer change about how Cadence
networks should be trained, evaluated, and deployed. The short answer:
the network had learned; the operating point, one hyperparameter, one
architectural read, and the evaluation lens were wrong, and each error
was measurable. Every number below comes from a receipt produced that
day. The reproduction scripts and both application pipelines are public
in the demos repository; the synthetic study is
[issue 70](https://github.com/muellerberndt/cadence/issues/70).

## 1. The run and the symptom

The `ultimate` layout (five input streams, 336 patches, 327,104 edges,
motor read from a 32-patch reflection observer) trained by batched
admission, batch 96, `parameter_prior=0.1`, on 20,000 witnesses from a
privileged scripted teacher. Hourly checkpoints were exported with a
150-row probe. The first trained checkpoint reported per-button
accuracies of 0.83 to 0.99, and in live play the agent either stood
still or rotated.

Base rates on the probe rows explained the paradox. The teacher pressed
forward on 24.7% of frames, so a policy that never presses forward
scores 0.753 on that button; most reported accuracies restated base
rates. Live behavior and probe accuracy measured different things.

## 2. The operating point: scores rest at target means

Motor targets were ±0.6. With forward pressed on a quarter of frames,
the mean target is near −0.3, and the measured settled score mean for
forward was **−0.284**. Every button's score sat below zero, so the
fixed press-above-zero decode emitted nothing; by the third hour even
the 90th-percentile forward score was negative.

Recalibrating the decode, per-button thresholds at the teaching base
rates' score quantiles, on the same checkpoint and rows:

| measure | zero threshold | calibrated |
| --- | --- | --- |
| forward pressed recall | 0.341 | 0.643 |
| back recall | 0.000 | 1.000 |
| closed-loop path, two seeds | 0.0 / 0.0 | 73.6 / 201.0 |

The brain moved the moment it was read correctly. Conclusion one: under
imbalanced ± targets, decode against base-rate thresholds and judge by
ranking and pressed recall. This is now standing guidance in LIVE.md.

## 3. The pass stopped paying, and readiness decayed

Fixed-row probes (identical 150 held-out rows per checkpoint) across
the run:

| checkpoint | forward AUC | calibrated fwd recall | refusals at budget 384 | kill foresight MAE |
| --- | --- | --- | --- | --- |
| b00014 | 0.818 | 0.595 | 0 | 0.021 |
| b00027 | 0.803 | 0.643 | 0 | 0.016 |
| b00039 | 0.792 | · | 0 | 0.023 |
| b00050 | 0.813 | 0.643* | 16 | 0.038 |
| b00063 | 0.792 | 0.613 | 52 | 0.024 |

(*measured at 512 budget in the calibration test.) Ranking was flat,
usable recall was flat, foresight peaked early, and settling cost at
the deployment budget grew from zero refusals to a third of rows. The
run was stopped at batch 65 of 208. Conclusion two: probe on fixed
rows, track ranking and the qualification rate at the deployment budget
across training, and stop when they flatten. Trainer probes that sample
fresh rows each export add noise that hides all of this.

## 4. The linear floor and the false ceiling

Ridge regression on the same witnesses and evaluation rows reached
forward AUC 0.856 on the rows the run had consumed and 0.871 on the
full corpus. The network sat below a bound its own inputs provably
supported. A minimal ladder reproduced the gap in minutes at toy scale,
then dissolved it:

| `parameter_prior` | mean AUC, 8 linear rules at 5% rate |
| --- | --- |
| 0.005 | 0.733 |
| 0.02 | 0.806 |
| 0.1 (the run's value) | 0.922 |
| 0.2 | 0.956 |
| 0.4 | **0.977** |

Ridge scores 0.951 on that task. The curve is monotone through the
tested range; single-witness admission scored 0.796, so batching
helped; widening the column changed no per-epoch AUC to three decimals
although the builds differed, and eight epochs oscillated without a
trend. The settled readout behaves as a fixed point that capacity and
exposure do not move and the prior places. Real data confirmed the
direction with smaller margins: doom frames 0.735 ridge, 0.763 at
prior 0.1, 0.781 at 0.4; Pong 0.569 / 0.768 / 0.775; Breakout 0.545 /
0.596 / 0.602 with the low band explained by input resolution. The
network cleared the linear floor on every real task once configured
sanely, with the largest margins on sub-1% actions where ridge scored
at or below chance. Conclusion three: the run's prior was mis-set, the
core extracts, and the linear baseline on identical rows is the
preflight every supervised run owes itself.

## 5. The read was a bottleneck

The layout grid that followed put the same tasks through the same
dials with five layout families. Reading the output through one small
late observer, the pattern of the Doom brain's motor path, scored near
chance on real Pong pixels (AUC 0.52 to 0.55) while direct reads and a
wide observer over every stage scored 0.78 to 0.82 on the doom rows.
Conclusion four: capacity behind a narrow read does not reach the
output; read outputs from enough of the network. Now in BOOTSTRAP.md.

## 6. What the demonstrator makes visible bounds imitation

The teacher steered from privileged route state, so part of its policy
was invisible in pixels; agreement and imitation AUC cap accordingly.
The Atari arcade made this vivid: replacing a demonstrator habit keyed
to an internal counter with a machine-state policy moved attainable
agreement from about 0.4 to 0.94 with no learner change. Conclusion
five: measure the demonstrator's pixel-visible ceiling before judging
the learner, and gate any takeover on what the ceiling permits.

## 7. Engine facts measured along the way

- A query settle reaches equilibrium in one numerical repair. Repeated
  micro-budget settles showed one state change of 0.291 and then
  exactly zero, at any budget. Settled per-patch errors are near zero
  whenever qualification succeeds, so they are not a skill measure.
  Visualize and instrument the equilibrium tracking its inputs, not a
  relaxation film.
- `observe_batch` thread scaling saturates: identical batches (sweeps
  771/438/751) ran 387/211/336 seconds at 64 threads and 445/248/425
  at 128. Measure before adding cores; a 192-vCPU host was three times
  oversized.
- Forked workers that imported torch deadlock; multiprocessing sweeps
  need the spawn context.

## 8. The life phase works

The arcade ran the full bootstrap-then-life loop on real pixel games
with this library: watch a teacher, admit witness batches, take over at
an agreement gate, then learn from reward through `Reinforcement` with
record-only feedback and budgeted replay pulses, the environment thread
holding the last action while the single brain owner thinks. A brain
born at page load watched 240 witnesses, took over at 0.95 agreement,
reached expert on its first solo episode, and later beat its teacher,
best 23.0 against a 21.2 teacher mean. Two of four games still degrade
in self-play and are tracked with acceptance criteria in the demos
repository. Conclusion six: imitation is the bootstrap, and the
campaign design should include the reward-driven life the library
already supports.

## 9. Adjustments adopted for the second campaign

1. Prior near 0.4, final value from the running dial-and-layout grid.
2. Motor read through a wide policy observer over every stage, with
   the layout carrying an action input and a value output from birth
   so checkpoints support the life phase without retraining.
3. Ridge preflight on the corpus before any training hour is spent.
4. Calibrated decode, fixed-row probes with AUC and pressed recall,
   and the qualification-rate trend as first-class run dashboards.
5. Teacher-mixed correction rounds after the pass, then a reward phase
   on the game's native rewards.
6. An unpressed-target flag for asymmetric motor targets, pending the
   follow-up grid that scores retention and admission economics beside
   accuracy so the chosen configuration is a good settlement network
   and a good scorer.
7. 64 threads, batch 96, a host sized to that ceiling.

## 10. What remains open

The grid's real-data tail, the follow-up pass over epochs, batch size,
`state_prior` and target asymmetry with retention checks, and the
two-seed guarantee harness for the life phase are running as this
report is written; their numbers pin the final configuration. The
growth of settling cost with admitted content at fixed budget remains
the one core-level trend to watch across the second campaign.
