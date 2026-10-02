# Learning from reward

Cadence combines free/nudged contrasts, eligibility traces and a reward prediction
error without a backward computation graph. Each synapse keeps its own eligibility
trace, and one broadcast dopamine signal gates every trace into a weight change.
This page describes those components and the tests that check them.

For application code, [`GenericBrain.step`](continuous.md) receives this moment's
observation and the preceding action's reward, updates plasticity, and returns the next
action. There is no separate training mode. The phase-level API below exposes the same
mechanism for custom architectures and measurement.

## Eligibility traces and dopamine

`ActorCritic` wraps a `Learner`. Acting runs the free phase; the action is a draw from the
softmax over each categorical motor slot (or, with `Bins`, one softmax draw per
continuous dimension). Unequal categorical slots use the existing `Learner(slots=[...])`
layout; every slot participates in the same settlement. Two nudged phases, toward and away from the
action taken, give every synapse its per-row contrast, an estimate of that action's score
under the [equilibrium assumptions](learning.md). The actor always uses a softmax
nudge, even when the learner uses quadratic imitation. For a fixed temperature `T`,
the contrast estimates `T * grad(log policy)`. Its common scale is absorbed in the
actor learning rate; several slots estimate the sum of their log probabilities. Each synapse keeps an eligibility trace
of those contrasts:

```
elig[e]      <- gamma * lam * elig[e] + contrast[e]
delta         = r + gamma * V(s') - V(s)
efficacy[e] += eta * delta * elig[e]
```

This is a three-factor rule. Presynaptic and postsynaptic activity enter through the
contrast and accumulate in the synapse's eligibility trace; the prediction error `delta`,
the dopamine signal, is the same for every synapse and sets the sign and size of the
weight change. `V` is a linear readout of the neurons named as the critic (usually the
hidden population). The actor signal is clipped (`dopamine_cap`) and can be centred
and scaled by its own running statistics (`dopamine_center`). The critic has two
explicit choices:

- `critic_signal="modulated"` preserves the original rule: the critic shares the
  actor signal. This can stabilize an update, but clipping or centring means it need
  not predict mean discounted return in reward units.
- `critic_signal="td"` uses the raw prediction error for the critic, independently
  of actor modulation. Use this when a calibrated return prediction is required.
  It can require retuning the critic rate and reward scale. The default is
  `"modulated"`.

The critic's step can be divided by its trace's energy (`critic_normalize`). Reports
include the absolute raw `td_error`, the absolute modulated `delta`, and signed
`dopamine`. A time limit is not a terminal state: pass `bootstrap=` the value of
the last observation for a truncated row.

The adaptive local step uses `momentum` and `normalize` to keep a running mean and RMS
of each synapse's own steps, with corrections for its short history. Each synapse reads its
own optimizer state; this does not guarantee stable learning for every task or setting.

For custom wiring, build an actor on a declared output population:

```python
import numpy as np
import cadence as cd

connectome = cd.layered(2, 8, 3, density=1.0, seed=0)
learner = cd.Learner(
    cd.Brain(connectome, cd.learning_neuron_model()), connectome.populations["output"]
)
drive = np.zeros((1, connectome.n))
drive[:, list(connectome.populations["input"])] = [[0.5, 0.8]]
ac = cd.ActorCritic(learner, critic=connectome.populations["hidden"],
                    config=cd.ActorCriticConfig(gamma=0.99, lam=0.95, eta=0.001, eta_bias=0.0001,
                                                eta_critic=0.5, normalize=0.999, momentum=0.9))
action = ac.act(drive)                       # settle, draw, keep eligibility
reward, done, next_drive = np.ones(len(drive)), np.zeros(len(drive), bool), drive  # from your environment
ac.learn(reward, done, next_drive)           # settle the next state, dopamine, every synapse moves
```

## Decision and reset contract

Drives are nonempty finite `(batch, neurons)` arrays. Rewards and optional bootstrap
values have shape `(batch,)`; `done` is boolean. Validation precedes trace and parameter
updates. Each batch row keeps its own potential, adaptation, eligibility and reward
centering. Ended rows reset before the next free phase; their neighbours get the usual
single phase. The next free phase estimates the bootstrap value before the parameter
update. The next action settles again from that warm state under the updated parameters.
A cached free phase is reused only when both the drive and the brain parameters are
unchanged. Budget for both free phases and the action’s two nudged phases.

`act(..., greedy=True)` is an evaluation read: it clears eligibility and cannot be
followed by `learn`. Calling another `act` replaces the pending action. Finish the
reward transition before changing learner parameters elsewhere, or call `reset`.
`reset` clears stream state and centering, but preserves the learned critic, actor and
optimizer history. `GenericBrain.save/load` includes this full state; `Learner.save`
does not save a separately constructed actor-critic.

## Reward evidence

Measure actual task outcomes separately from shaped reward. An
undiscounted potential difference does not generally preserve the original
objective under discounting; that guarantee requires the discounted form
`gamma * Phi(next) - Phi(now)` with appropriate terminal handling.

The [experience guide](experience.md) connects eligibility, prediction error
and valence to the rest of the learning life.

## Budget the ongoing loop

Choose the settling budget using changing observations and the task's outcome
metric. Compare cold and warm starts, reset independent episodes, and
measure all decision work. A small step count or a cached response to one unchanged
input does not establish reliable control or lower decision cost.

## What delayed credit requires

A reward that follows the reading it belongs to is a record: written in one exposure at
the fast rate and read back at that reading ([records](memory.md#records)). The
eligibility trace is for the policy where the reward arrives after other decisions.

An eligibility trace remembers which action could have caused a later reward; it
does not remember an arbitrary observation or learn a memory address. At a delay of
`d` transitions its contribution is multiplied by `(gamma * lam) ** d`. A trace
that fades too soon cannot assign useful credit, while a long trace includes more
unrelated actions. State representation, exploration and the critic matter too.

The [delayed-credit test](../tests/test_child.py) pays a choice three moments later,
after blank moments with presses of their own; the learned greedy choice reaches a hit
rate of at least 0.8 with the trace (`lam=0.9`) and at most 0.65 without it (`lam=0`). The
[policy credit tests](../tests/test_policy_credit.py) check that the eligibility
differentiates the sampled policy. Passing these small tests does not establish
performance on any larger task.

For a real task, record raw held-out return, forgetting and every interaction used
by rehearsal or planning. Choose the eligibility horizon from actual action-to-reward
delays, and test imitation, reward practice and rehearsal separately. A biological
analogy supplies a hypothesis; the behavioral test determines whether it works.

## Traps, with their measurements

Each of these cost a day on a real brain (the fruit fly of
[cadence-examples](https://github.com/muellerberndt/cadence-examples), 150,802 neurons, an
actor-critic on the Kenyon-cell-to-MBON synapses; the worm met the second one first). They are
properties of the rule and the readout, and each has a reading in the `learn` report. The last
three are read before any lesson by `preflight(brain, outputs, plastic, drives)`, which names
the remedy for each; run it first.

- **The temperature is relative to the activation range.** The action is a softmax over the
  output neurons' activations divided by `temperature`. Activations lie in [0, 1], so at the
  worm's 0.05 two outputs that differ by 0.3 make a choice with probability 0.998, the nudge's
  push `beta * (target - p)` is nothing, and no synapse moves: the fly sat at 1.00/1.00 for 600
  decisions. Outputs that live near rest (the worm's command neurons) take 0.05; outputs that
  sit at 0.7 to 1.0 under their drive take 0.3. Set it from the naive activations, not from
  another example.
- **The saturation latch.** Once one output saturates and the other is silenced, the activation
  has no slope, the contrast `a_pre * (b_plus - b_minus)` is zero and the rule cannot leave the
  state whatever the reward says. `report["saturation"]` is the fraction of output activations
  within 0.02 of 0 or 1 and `report["trace"]` the mean absolute eligibility of the plastic
  synapses: saturation near 1 with the trace near 0 is the latch, visible at decision 100
  where the reward curve shows it at 800. The remedies are a gain at which the outputs sit in
  the sensitive band, `LearnerConfig(scale_cap=...)` below the default 8 so learning cannot
  drive them out of it, and a readout whose cells are neither silent nor saturated under the
  task's drive before any lesson.
- **The value on a code that cannot see progress.** With a terminal reward and a linear critic
  on a state code that is the same all along the approach, the value stays near the discounted
  mean everywhere: the fly's was 0.25, so sugar surprised by +0.75 and an empty arm by -0.25,
  and the synapses every odour shares drifted toward approach until the latch. Make the
  outcomes symmetric where the assay allows it (the animal's differential conditioning pairs
  sugar with quinine), or give the code the progress (a level that rises with distance).
- **The assay decides whether avoidance can be rewarded.** In an open field one avoidance turn
  leads nowhere, so a fly that avoids a third of the time reaches no source and no reward
  arrives; the rule then learns "approach everything" whatever the wiring. In a T-maze
  avoiding one arm's odour means taking the other, every search ends at an arm, the
  approach-everything policy scores one half and the association scores one. Build the arena
  so that every action the readout can take has an outcome.
- **Centring and the critic.** `dopamine_center > 0` makes the actor's dopamine the surprise
  over its running level; a critic fed the same signal chases a moving target and its value ran
  to -15 within 300 decisions. `critic_signal="auto"` (the default) gives the critic the raw
  error whenever the dopamine is centred.
- **A cap applied by rebuilding the brain.** Clipping efficacies by `brain.with_parameters` after
  every decision re-uploads every weight on the torch backend and cost a factor of ten per
  decision. The cap is a config field, `LearnerConfig(scale_cap=...)`, applied inside the update.
- **A code the lesson cannot attach to.** The actor moves the synapses from the active cells of
  the state code; if that code is the same for every stimulus, every lesson moves every
  stimulus. The fly's Kenyon cell codes for two odours had a cosine of 0.98 (the antennal lobe
  ignited through its cholinergic local neurons at the global gain), and twelve blows at one
  odour drove both approach probabilities from 0.9 to 0.09 together. Measure the code with the
  `specific` fact before the lesson and select the offending population's gain on it
  ([brains from a connectome](connectomes.md)); no rule downstream repairs it.
- **A readout on a rail, and a readout with a past.** A connectome carries no operating point:
  at the global threshold the fly's approach cell sat at 1.00 under every odour and its avoidance
  cell at 0.01, and the nudge had no slope on either. `calibrate_bias` puts each readout cell
  at one half over the situations it will decide in, jointly. And on the measured counts of the
  memory seam the naive readout already avoided one odour (0.17) and approached the other
  (0.83): a specimen's synapse counts at its memory site are its memories, and a smell that is
  never approached is never rewarded. `naive_efficacy` starts the seam with every plastic class
  at the same weight.
- **A seam thinned by custody.** A synapse floor removes a distributed memory along with the
  noise: the fly's floor of five kept 231 of 1,079 Kenyon cell classes onto one output neuron
  and 14 of 336 onto the other, and nine blows moved the approach probability by 0.03. Keep the
  seam a lesson will move at every count and read `seam_report` before designing on it.
- **The critic that learns faster than the actor.** With `eta_critic` at 0.5 the value reached
  the outcome in four trials and the dopamine went to zero while the actor, whose contrast is
  small near a rail, had barely moved; with `eta` at 10 one lesson put an output on its rail.
  The fly runs `eta` 0.25 and `eta_critic` 0.05; read `report["delta"]` across repeated
  outcomes and `report["saturation"]` before raising either.
- **Eligibility mixed along one approach.** A decision every 0.3 s along an approach gave the
  trace approach and avoid nudges from one flight, and the approach nudge had less room because
  its cell sat near saturation; sugar then rewarded whichever nudge had been larger, and three
  rewards taught avoidance. One decision per episode, credited to that decision, as the T-maze
  has it.
