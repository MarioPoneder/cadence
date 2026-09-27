# How to build a rung demo

Every step of the program is accepted by a demo, under one rule: the page carries the whole brain
in the viewer, the world, the instrument strip and the plain-words account; a three-way switch
(the step on; off, the brain below it; the hand-designed control) makes the acceptance visible;
the person is in the world (the mouse as the laser dot, the hand that drags the voice, the click
that makes the baby cry). The step is accepted when the demo works with the switch on and fails
with it off, on the numbers in its report, at matched information and compute. This page is the
order of work, with the night nursery (rung 4, the tiger) as the worked example. The pieces it
uses are in [a brain that reads itself](steering.md) and [the belief patch](belief.md).

This recipe describes the existing sequential composition demos. A task-scoped
acceptance does not establish observer and observed activity settling jointly,
or a depth advantage over matched conventional learners; see the
[numerical contracts](contracts.md).

## 1. Read the rung

The ladder states, for each rung, what the steering patch reads and returns, what it unlocks, the
experiments, the control and the falsifier. Write them at the top of the report before anything
runs. Rung 4: a large surprise below seizes the steering patch's state within one decision;
repetition without consequence habituates; a consequential event keeps capturing. Control: a
hand-set threshold rule, its constants genes if it wins; the steering patch cut. Falsifier: the
threshold rule matches capture and habituation at matched compute.

## 2. Build the world

A world for a rung is a body and senses whose readings are declared functions of its state, a
schedule of events, and a target the brain should predict.

- **Senses as blocks.** One block of the cortex's port per sense, so each sense has a gain. The
  night nursery has an eye (a retina and two coordinates, scaled by the light) and an ear (an
  interaural cue and three timbre channels).
- **Identity linearly reachable.** A steering patch can only learn to treat a sound differently
  if what distinguishes the sound is in its readback. "The clock is at one of these corners"
  was never learned; a timbre axis only the cry has was. Put the distinguishing channel in the
  reading and read it through `evidence=`.
- **Noise where the weighing must rest.** With the ear as clean as the eye the day's weighing
  rested on the ear and there was nothing to capture; at ten times the eye's noise the weighing
  rests on the eye in the quiet and the dark still pulls the ear. A probe of the day alone at
  two noise levels on two seeds is a receipt worth keeping.
- **Persistence units.** The surprise is measured in units of each sense's persistence error,
  the mean squared change of its compared channels from one quiet moment to the next; compute
  them on the training streams and declare them with `set_implied_reading(implied, units)`.
- **The events.** Each event is `(kind, start, length)`; the consequential kind changes the
  world (after the cry the moon jumps to the baby). A kind that arrives late in the night (the
  phone) measures novelty.
- **Seeds.** Five campaign seeds and one design seed for the page, named as such.

## 3. Build the four brains

All four arms are constructions of one class:

```text
the rung        Steered(cortex, steering, Softmax(2, span), evidence=[ear])   the steering patch hears all channels
the brain below Steered(cortex, steering, Softmax(2, span))                   its port's mask hides the evidence
fixed gains     Steered(cortex)                                                the rung-0 brain
the control     Steered(cortex, weighing=Rule(threshold_rule, macs=8))         the hand-designed rule
```

The genome is a dict with the hand-set values (the steering patch's size, damping, span,
port scale, rate scale, retention, and a `reads_*` flag per channel group), and `genes(space)`
gives `evolve` its mutation over it. Keep the cortex identical across arms; count every arm with
`macs_per_moment()` and `cost`. These estimate dense forward work; add backward,
write, callback and search costs and measure elapsed time for training comparisons.

## 4. The day

Raise every arm by day on whole quiet streams with cries: batches of eight streams of 64
moments, one admitted joint step per chunk from a fresh boundary (`run(o, a, y, rate=cap,
state=brain._fresh(n))`), validation on held-out streams every two epochs, the best snapshot
kept. A day of 120 epochs; short days of 24 to 40 epochs leave the weighing on the wrong sense
and mislead every pilot. Save the dawn brains (`snapshot()`), they are what the page loads.

Those settings describe the reported nursery experiments, not universal learning
requirements. Keep training, calibration/validation and final test streams
disjoint. Any governor baseline or threshold is fixed on calibration data before
the online night; test targets never choose it.

## 5. The night

The night is one long stream lived online: chunks of eight moments, a cap of 0.1, the boundary
carried (`run(..., state=None)` continues from the live boundary), the cortex asleep
(`learn_cortex=False`) while the steering patch learns, or gated by a governor (`Life`). A cap
of 4 on 512-moment batches, the day's regime, wrecks online tracking; a steering rate scale
above one pins the ear. The exception needs 48,000 frames; 16,000 forget the clock and do not
keep the cry. Record the gains, the readbacks, the errors and the events.

## 6. Measure

- `orienting(gain, events)` for capture, latency, return and the habituation curve per kind;
  `dishabituation` around the consequential events; the held-out error per condition (quiet,
  dark, each sound, the jump).
- The sign of "capture" depends on the dawn baseline: where the ear rests open, habituation is
  a closing of the ear at known sounds. Report the gain at the consequential event against the
  gain at the habituated one, and the error at the consequence, beside the signed capture.
- Ablations on the trained rung brain at test time: `ablation = "cut"` and `deaf = mask`.
- The night school: the same night in batched epochs with the cortex asleep is the clean
  unlock (the rung against the brain below on the consequential event alone).

## 7. Select

`evolve(fitness, hand_set, mutate=genes(space), grow=grow, generations=, population=, keep=)`
with the hand-set genome as the lineage's first member and a random search over the same space
at the same number of evaluations as the second control, on held-out seeds. The fitness reads
only what the genome cannot reweight, and prices both compute and surprise; a price on compute
alone switches learning off. If the control's constants are genes and the control wins on the
total error, say so; the night nursery's evolved threshold rule won the night's error and lost
the cry, which is the rung's answer.

## 8. The page

The page is a Python server (the brains, the world, `/state`, `/atlas`, `/weights`, `/control`)
and one HTML file: the world on the left with the person's controls in it, the whole brain in
the viewer on the right (`build_atlas`, `brain_scan_script`), the instrument strip, four tiles,
one line to switch the brain, everything else in a collapsed section. The world runs from the
design seed's dawn brains so the page starts at once. Every number the page quotes for the
selected brain comes from the frame on screen, not from the server's latest stat, since the
picture trails the server by the frames still queued. A stat that is not a number is sent as
null: the browser's parser rejects `NaN` and the whole state with it.

## 9. The check

A headless check drives the page through the scenario of its promise (Playwright, software GL):
it waits for the first live frame, selects each arm, drives the sounds and the cries, reads the
instrument tiles, takes a screenshot per arm and at phone width, counts console errors and
horizontal overflow, and writes `receipts/page_check.json`. Run it right after a server start,
so the driven sounds are the night's first; a check begun mid-night gives small captures and
must not replace the receipt. Run it before every deploy.

## 10. The receipts and the report

`receipts/*.json` in canonical JSON (`canonical_json`), one per campaign (the arms, the night
school, the rate sweep, the pilot, the evolution, the page check), each carrying the protocol,
the library's version and commit, and the numbers. The report states the acceptance sentence
(the brain below fails at matched compute and the brain with the rung passes, or not), the
numbers with their receipts, the three-way switch, what did not work, the limits and the
regime, and what the library should carry next. A `PROPOSAL.md` lists what the demo had to write
around the library; two demos asking for the same thing is the library's signal.
