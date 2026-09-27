# RecordPatchNet

`RecordPatchNet` is a temporal patch whose context is linear and gated and
whose memory of particular readings is a record store inside the patch. The
nonlinearity sits at the ports. Pin a release or a commit when reproducing an
experiment.

For inputs `u[t]`, context `h[t]` and outputs `y[t]`:

```text
l[t] = sigmoid(g + G u[t])               per-channel retention in (0, 1)
z[t] = tanh(B u[t] + b)                  the input port
h[t] = l[t] * h[t-1] + (1 - l[t]) * z[t]
m[t] = read(code([u[t] * sqrt(n) / s, r * h[t]]))   the record read, a port value
y[t] = C h[t] + c + m[t]
```

The context is a convex combination, so a bounded initial context stays
bounded. With inputs and parameters fixed, the seam residual
`h[t] - l[t] h[t-1] - (1 - l[t]) z[t]` is linear in the context path. With the
linear readout its energy is quadratic, and the free path is its unique
zero-defect normal form. The two teaching-detuned paths are unique when their
quadratic systems remain positive definite; the solver's convergence result
must be checked. Categorical ports have a different loss, described below.
Channels start at retention timescales log-spaced from two moments to
`slowest` moments. The gate parameters are learned: `slowest` is an
initialization scale, not a hard memory horizon. `r` reads each channel in the
unit of its initial fluctuations.

The path can change at every moment while satisfying its temporal equations.
That is different from repeatedly revising an interpretation of one fixed
observation. The [equilibrium and world-model guide](equilibrium-world-models.md)
distinguishes environment time, inference iterations and learning, and states
which parts of a shared world-model architecture remain unimplemented.

## One observed path

```python
import numpy as np
from cadence import RecordPatchNet

net = RecordPatchNet(inputs=17, hidden=8, outputs=6, seed=2, cells=4096, active=32)
rng = np.random.default_rng(7)
inputs = np.zeros((1, 16, 17))
inputs[0, 0, 0] = 1.0                            # one cue
inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0  # sixteen distinct heard events
identities = rng.integers(6, size=16)
target = np.eye(6)[identities][None]

first = net.observe(inputs, target, rate=1.0, backtrack=True)
assert first.updated and first.writes == 16
recalled = net.imagine(inputs, state=np.zeros((1, 8)))
assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
```

`observe` predicts the path with the records as they stood when the call
began, moves the slow parameters against the adjoint gradient of the
precision-weighted half mean squared error of the slow readout `C h + c`,
and then writes what that readout got wrong at each reading into the
reading's records. The record read never enters the slow gradient: the slow
parameters learn the observation itself, and the record patches their
current error until they have. With `backtrack=True` a parameter step is
admitted only after a target-free replay from the original boundary lowers
the slow readout's loss, trying up to sixteen halved rates. `write=False`
learns without writing. The final free context becomes the live state;
`advance` carries context without learning or writing, `imagine` is
private, and `reset` clears context while keeping parameters and records.

## The reading: two blocks of unit variance per unit

The record's key is the input block and the context block side by side. The
context is read in units of each channel's fluctuation (`r`), which gives its
units unit variance. The input block is scaled by `sqrt(n) / s`, where `n` is
the number of input ports and `s` the running rms norm of witnessed inputs,
so its units have unit variance too and a cell's drive has unit scale, with
the fixed offset a small preference. Before this scaling the context block
outweighed the input block about five to one on the composer stream: the
address hardly moved with the event heard (48-cell code overlap 0.66 when
only the crop changed), a few hundred hub cells took three quarters of all
activations, and the store forgot the training corpus behind its last
writers. With the scaling the same stream uses 5,812 of 8,192 cells instead
of 2,225, the address follows the crop heard (overlap 0.15), and held-out
next-crop and bass-note accuracy both rise. Two further options exist and
are off by default: `record_averaging` makes a cell's write rate one over its
written mass with `record_rate` as the floor (a fresh cell takes its first
outcome whole, a familiar one averages), which on the composer stream
calibrated reads at unfamiliar readings and improved retention but slowed
adaptation to a new track; `record_homeostasis` moves each cell's offset
toward an equal activation share, which spread the code further but cost
adaptation as well.

## Records hold what the slow model does not know

Each write target is `y_obs[t] - C h[t] - c`. A reading the slow parameters
have learned leaves a record near zero, so the store holds only the
residual the slow model has not absorbed. One write corrects its own
reading; readings whose codes overlap move by their overlap, as in the
record theorems. Keys of neighbouring moments share the slowly varying
context, so one write is not exact and the delta rule converges over
repeated observation. Readings that differ only by the bar they belong to,
under a periodic clock, are separated only partly by a leaky context; a
heard stream, in which the previous event is an input, supplies the rest.
Whether carried context alone can return a theme after many bars is an
open measurement, not a property of this class.

Records are read with the tables at the start of the call and written after
the path, so a prediction is a function of the parameters, the records and
the boundary alone, and a private branch reproduces it exactly. To let a
record written at one moment inform the next, observe in shorter paths.

## Diagnosing a record store

A record store can fail silently: prediction with records still beats
prediction without them, because the store always holds something. Four
numbers show whether it holds the right thing. Replay the readings of a
training stream through `records.code(..., valued=False)` and count, per
cell, how often it is active.

- **Cells in use** and **the share of activations taken by the most active
  cells.** With `active` of `cells` winners the ideal share of any 200 cells
  is `200 / cells`. On the composer stream the defective reading used 2,225
  of 8,192 cells with 0.74 of all activations in the top 200; the repaired
  one 5,812 cells with 0.27.
- **Code overlap when only one block of the reading changes.** Hold the
  context, change the input (and the reverse), and compare the two codes.
  An address that does not move with a block cannot store anything about
  it. There the overlap was 0.62 to 0.71 when only the event heard changed
  and 0.92 when the context moved four bars; after the repair 0.15 and 0.88.
- **Held-out scores with records frozen, zeroed, writing online, and writing
  online from empty.** Frozen far below online means the store forgets its
  training stream; online equal to online-from-empty means the trained
  records contribute nothing to a new stream.
- **The read at readings no stream has written,** for instance a rollout
  from silence. A read far from zero there is interference from unrelated
  writers, not memory.

## What a write rate stores

The delta rule at a rate near one half makes a cell hold its last few
writers. That is the right memory for a new stream: its first moments
overwrite the cells they touch, and the read then calibrates the prediction
to that stream (on the composer, four observed bars of an unheard track
moved the change port's read from +0.13 to -0.22 and its rate of predicted
changes from 0.24 to 0.07 per moment). It is the wrong memory for a corpus:
with records frozen the same store recalled 0.03 to 0.11 of the training
basslines. `record_averaging` makes each cell average its writers instead,
with `record_rate` as the floor; retention and calibration at unfamiliar
readings improve, adaptation to a new stream slows in proportion. One table
cannot do both at full strength; choose by which the task reads.

## Recall is by content, not by position

A record is keyed by the reading: what was heard and the context. Where the
stream recurs, the key recurs. On a composer playing its loop from silence,
the code at one place and the code at the same place one loop later share
0.91 of their cells, against 0.04 for two unrelated moments. When the
playing drifts in phase, so that the same place in the bar hears another
event, the overlap at the same place is 0.16 and nothing written there is
read back. This decided an experiment: a composer that wrote its own first
four bars into its records, as it does for a heard track, did not bring
its opening back four bars later (the opening's departures returned at 0.09
to 0.20 of their places, against 0.18 without the writes), because by then
its playing had shifted in phase and it heard other events at those places.
A patch recalls a
moment when it meets the same event in a similar context. If a task needs
recall by position (the same bar of a phrase, the same step of an episode),
position has to be in the reading as an input of its own, with enough ports
to move the code; a clock of eight ports among eighty inputs does not.

## Boundary readings

A reading met once per stream, such as the wake moment with nothing heard,
receives a handful of updates per epoch. The slow parameters barely learn
it and its record cells are shared with common readings, so the first
prediction of a rollout from silence is poorly determined (on the composer
it started the loop out of phase with the clock). Make the boundary a
common reading by convention instead of hoping it is learned: there the
wake hears a count-in, the last event of the loop, and the rule the patch
knows best produces the first event.

## Detuning as the acceptance check

```python
net.reset()
adjoint = net.observe(inputs, target, rate=0.0, write=False).delta
net.reset()
contrast = net.detune(inputs, target, beta=1e-4)
assert contrast.converged
for name, value in adjoint.items():
    assert np.linalg.norm(contrast.contrast[name] - value) < 1e-6 * np.linalg.norm(value)
```

`detune` solves the two detuned equilibria of the quadratic energy by
conjugate gradients from the free path and returns the centered contrast of
the energy's parameter derivatives. The energy describes the slow patch: its
readout residual is `y - C h - c`, and the record read lies outside it. For
this patch the contrast equals the adjoint gradient up to terms of order
`beta^2`; `observe` computes the same learning signal by one backward scan.
Neither route sees the record: the read is a port value added to the
readout.

## Cost and state

One update is one forward scan, one backward scan, one record read per
moment and one write per observed moment. Work per moment is linear in the
context width plus the record projection; state is the context, the
parameters and the record tables, constant in stream length. The dense
hidden-width block messages of `TemporalPatchNet` do not appear.

## Checkpoints

`snapshot`, `restore`, `save` and `load` carry the parameters, the record
tables, the running mean and counts, the output precision and the live
context. `readback` exposes the detached state, update and write counts, the
parameter revision the state was computed under and the record entry count.

These snapshots contain the patch's continuation state, not the application's
observation history or a planner's external state. An application that branches
imagination also owns its random generator, action adapter, observation
normalization and any additional memories. Freeze those with the model during
a decision, and verify that private reads leave the factual snapshot unchanged.
Keep imagined targets distinguishable from observed outcomes; `sleep` consumes
model-generated targets and does not authenticate them as observations.

## Running a trained patch outside Python

The forward pass needs no library. Export the six parameter arrays, the
record table, the running mean of the reading, the input norm and the
record configuration; everything else is regenerated. The record projection
and offsets come from the seed: the `Mulberry32` generator, then
`normals(n)` by Box-Muller over `2 * ceil(n / 2)` draws with the radii from
the first half of the draws and the angles from the second half, cosines
before sines; the projection takes `reading * cells` normals divided by
`sqrt(reading)`, row by row over the reading, and the offsets take the next
`cells` normals times `bias`. A moment is then the gate and the context
update, the reading `[u * sqrt(n) / s, r * h]` minus the mean, the drives,
the `active` largest of them (a heap of that size is enough), their
positive parts normalised to unit length, the table rows weighted by that
code, and the readout. At 208 reading units and 8,192 cells a moment is
about 1.7 million multiply-adds. A JavaScript port built the projection in
0.1 s and ran 128 moments in 0.5 s; with the table shipped as float32 it
reproduced an archived Python rollout in every played event, with outputs
equal to 2e-8. Keep such an archived rollout as the port's parity test.

## Categorical ports

Symbols (words, moves, note names) are taught as distributions, not as levels.
With `groups` the slow readout ends in one softmax per group of ports and
learns by cross-entropy, whose output error is `softmax - target`:

```python
import numpy as np
from cadence import RecordPatchNet

net = RecordPatchNet(inputs=17, hidden=8, outputs=6, seed=2, cells=4096, active=32,
                     record_rate=1.0, groups=(6,))
inputs = np.zeros((1, 16, 17))
inputs[0, 0, 0] = 1.0
inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0
identities = np.random.default_rng(7).integers(6, size=16)
first = net.observe(inputs, np.eye(6)[identities][None], rate=0.0)
assert np.allclose(first.prediction.slow_output.sum(axis=-1), 1.0)
recalled = net.imagine(inputs, state=np.zeros((1, 8)))
assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
```

A record then holds `onehot - softmax(C h + c)`, a residual bounded by one in
every port, so the record algebra is the one above: a reading the slow weights
have learned leaves a record near zero. The prediction `softmax + read` can
contain negative entries and is not generally a distribution. Its largest
port can be used for classification, but sampling, probabilistic planning and
log-loss evaluation require a declared nonnegative, normalized distribution
for each group.

The current loss implementation clips each entry to at least `1e-12` and then
normalizes the group. A residual that pushes the true category below zero can
therefore incur a very large loss. That exposes a problem with the resulting
probabilities even when the winning category is correct on many other rows.
Report mean held-out log loss and calibration for the declared normalization,
alongside accuracy and the slow-only scores. State the clipping floor; a median
loss may supplement these measures but must not replace the mean or conceal
rare confident errors. A learned probabilistic combination of records and
slow logits would be a different implementation and needs its own tests.

Each group's teaching weight is the mean `output_precision` of its ports.
`detune` refuses categorical ports: the quadratic detuning result applies only
to the linear readout. Default nets keep checkpoint format 2; a net with
`groups` or batch writes saves format 3.

## Writing a batch

`record_writes="batch"` writes all moments of a call at once through
`Records.write_batch`: every error is taken against the tables as they stood,
and each cell moves by the mean of the moves its writers would have made alone.
One writer reproduces `Records.write` to rounding; writers that agree move a
shared cell as far as one of them would, so a batch does not overshoot. On a
path of four streams of ten symbols presented twelve times, batch writes left
34, 12, 7, 4, 1 wrong moments after the first five presentations and sequential
writes 34, 22, 18, 10, 6; both end at the one moment that is undecidable (two
streams open with the same symbol and different outcomes). Sequential writes
within a call let a later moment overwrite an earlier one that shares its
cells; the batch average does not.

## Do not record a choice

When a reading has several right outcomes (a message that can be said in four
ways), the slow readout learns their distribution and its error at that moment
is the choice that happened to be taught. A record of it pulls later
predictions toward one arbitrary choice and spends cells. Write records for
what was observed once and must be reproduced (a sentence read, a phrase
heard), and leave choice points to the slow weights.

## Sizing a store for recitation

A store recites a sequence exactly only when it has more cells than
associations to hold, and it needs several presentations, because keys of
neighbouring moments overlap. Closed-loop recitation of random sentences of 8
to 22 words from a 1,500-word vocabulary, keyed by a message and the word just
said, records only (untrained slow weights), 32 active cells, eight passes:

| sentences | associations | cells | words right, teacher-forced | sentences exact, closed loop |
| --- | --- | --- | --- | --- |
| 300 | 4,400 | 8,192 | 0.93 | 0.36 |
| 1,000 | 14,700 | 8,192 | 0.78 | 0.04 |
| 1,000 | 14,700 | 32,768 | 0.97 | 0.66 |

A closed loop multiplies the per-word rate over the sentence, so exact
recitation needs the per-word rate near one: more cells, more passes, and slow
weights that learn most of the material first.

## A store narrower than its port

A port of thousands of categories need not give the store one column per
category. With `record_width=w` the cells hold a fixed random sign code of the
residual, `residual @ R` with `R` of shape `(outputs, w)` and entries
`+-1/sqrt(w)`, and the read is decoded by the transpose, `held @ R.T`. A stored
residual comes back with crosstalk of standard deviation `|residual| /
sqrt(w)` per port, which the largest port survives. The delta rule has to
compare like with like: the coded residual with what the cells hold. Taking
the error after decoding and projecting it again multiplies the step by
`outputs / w` (10.7 at 5,481 words and 512 columns), and the store diverges:
in the language work this recited nothing until it was found. Closed-loop
recitation of 300 random sentences (4,353 associations, records only, a
1,500-word port): one column per word 0.44 exact after eight passes, a
256-column code 0.41, at a sixth of the memory. At write rate one a
16,384-cell store recited 0.39, 0.74, 0.93, 0.987 of the sentences exactly
after 4, 8, 16, 32 passes, and a 32,768-cell store 0.77, 0.957, 0.98, 0.997:
about seven cells per association and thirty passes for exact recitation
when the slow weights know nothing.

## A grammar from records alone

Records generalise by overlap, and that is enough for a small grammar. A speaker whose input is
a message (act, kind, number and agreement features, 124 ports) and the word it has just said,
with the slow weights left at their random initial values, was given one pass of delta-rule
writes over 10,672 (message, sentence) pairs of a grammar of 1,276 messages and 6,173 sentences,
then asked to say sentences for messages it had never met: a quarter of the (subgenre, era,
form) combinations and a tenth of the other pairs were held out. With 65,536 cells, 32 active, a
320-column output code and write rate one, the store alone produced a sentence inside the
grammar for 0.815 of the held-out combinations and 0.885 of the other held-out messages, and
preferred the grammatical member of every minimal pair (a/an, is/are, deal/deals, both/all); a
second pass gave 0.85 and 0.915. The write pass took 88 seconds after a calibration pass over
the inputs that set the context's centering and scale, and the scores are greedy decoding:
sampling the first three words at temperature 0.8 gives 0.03 and 0.05. The store held
65,536 cells of 320 float32 coefficients, 80 MiB, so the comparison is not capacity-matched.
A new message shares most of its features with taught ones, so its reading touches their
records, and the read averages them: the record principle doing agreement and word order
without a gradient. A two-layer patch whose slow weights learned the same pairs by fifty
epochs of cross-entropy reached 1.00 on the same held-out sets, and so did a GRU and a
transformer of the same width; those three slow learners were trained by backpropagation
through sentence time and layers, with unequal parameter counts (9.68M, 3.90M and 4.70M) and
one model seed each, so their run says nothing about the local contrast rule. At one eighth of
the pairs and equal updates the patch had the lowest held-out loss of the three at every width
(0.32 against 0.34 and 0.36 nats per word at width 64, with 2.4M parameters against 0.83M and
0.88M). The grammar is finite and every learner reaches its ceiling with enough data; what the
store shows is what one pass of writes buys, and the held-out messages are new combinations of
taught features, not new vocabulary.

One knob at a time from that base (65,536 cells, 32 active, a 320-wide output code, write
rate 1, one pass), on 200 never-taught combinations: 64 active cells lift the valid share
from 0.815 to 0.900 and 128 keep it there, 16 fall to 0.595; a write rate of 0.5 in a store
written once falls to 0.675, since a record then holds half of its residual; 16,384 cells
fall to 0.655 while 32,768 give 0.870 and 131,072 give 0.860; a second pass of writes gives
0.850, a fourth 0.795; the output code's width (160, 320, 640) changes nothing; doubling the
message's or the context's share of the address falls to 0.670 and 0.680, and halving the
context's share gives 0.825. The rule these numbers give: address with enough active cells
that neighbours overlap (64 of 65,536 here), write at rate 1 when each reading is written
once, and keep the token, the cue and the context in balance in the address.

## Acquisition in two phases: records by day, weights by night

A transformer acquires a world model and a language the same way: a gradient over a
corpus moves every weight a little at every token, for as many passes as it takes. A
record patch acquires in two phases, and the phases are the two learning rules the
patch already has.

**By day, what is observed is written, once.** A reading is the message or cue, the
event just heard and the context; its record takes the outcome in one write and holds
it exactly (Theorem: one-shot memory). The slow weights do not move. This is why the
patch can be told a fact in a conversation and use it in the next sentence, why it can
take a corpus in one pass, and why nothing it learns by day disturbs what its weights
hold. It also has a cost: a store generalises by overlap and no further, and it
disturbs itself when it is shared (writing 24 new sentences into a store that also
held a library moved the library from 1.000 to 0.973 exact; a conversation needs its
own store). The store is linear, so two stores read together are one store: what keeps
a conversation apart from the library is which store a cue reads. Written into an empty
store of their own and read alone, the same 24 sentences left the library at 1.000, and
0.79 of them stayed exact after all 24 writes, as in the shared store: what limits them
is their overlap with one another, and a store of their own does not change that.

**By night, the slow weights take what the store holds, from dreams.** The corpus is
gone; the store is the only copy. `sleep(cues)` dreams every cue of the day once (the
free path with the records: what the patch would answer awake), teaches the slow
weights those fixed dreams by the ordinary contrast with `write=False`, and at dawn
writes the dreams back so the store holds only what the weights did not take. Nothing
outside the patch is consulted. Dreaming is necessary and not decorative: the slow
weights need many presentations of a regularity, and the only place the day's
observations survive is the store, so the presentations have to be the store's own
completions. It is also where generalisation is made: the completion of a cue the
store never met is what the records of its neighbours agree on, and the slow weights
learn that agreement as a rule. Measured on the grammar of the language work: the
store alone gave 0.815 of the never-taught (subgenre, era, form) combinations a valid
sentence at bedtime; after one night of 8,000 dreams from the day's own cues (0.91 of
them inside the grammar) and 1,200 slow updates, the slow weights alone gave 1.00, with
the corpus closed. The awake control that re-read the corpus twice more with the
teacher present reached 0.905. Nothing new entered during the night; what was held was
redistributed into the weights, and the weights, being smooth, finished the pattern.

**What sleep does not do.** Dreams carry the store's errors as faithfully as its
regularities: on the held-out messages whose completions the store got wrong at
bedtime, the night changed nothing (0.885 to 0.89), where the teacher with the corpus
for the same 1,200 updates reached 0.98. Dreams from recombined cues, which land on
combinations never taught, gave 0.83 and 0.875: the store's guesses at unmet cues are
taught as facts, whereas dreams from the cues the store met let the smooth weights extend
the pattern. Sleep restructures what is held and does not correct it; correction needs a
critic, a re-reading of the disputed cue, or another day.

**Why the dreams are fixed and the store rewritten.** A record is the residual against
the slow readout at the time it was written. Dreaming again after every update would
teach the weights their own drift (in a toy, six of twenty-four right after sixty such
steps), and reading the old residuals against new weights would count each outcome
twice. So the night fixes its dreams at bedtime, and dawn re-references the store:
two passes of writes leave it holding only what the weights did not take, and a new
fact written the next day lands beside them. This is the two-timescale account of
memory that the brain suggests, written as two operations on one patch: the store is
fast, exact and local; the weights are slow, smooth and general; the night moves
knowledge from the first to the second.

**When to sleep.** The night pays when the observations are gone and the store is the
only copy of them. Three of the published examples were run under the day and night
regime with this class unchanged. The Connect Four value patch, with its school closed
after one day of 67,399 writes, went from chance to 0.755, 0.768 and 0.772 on the
held-out sign after one night at 4,096, 16,384 and 65,536 cells; the school reached
0.785 from the same 263 updates on the outcomes themselves, and stayed one to three
points ahead at every matched update count. The dawn readout followed what the store
read at bedtime (0.733 to 0.789), and a second night on the same store added nothing:
its dreams were what the weights said. On the full school of 4.4 million positions, taken as
67 days and nights, the sleep-trained readout reached 0.838 against the school's 0.896 at the
same 52,000 updates: the gap grows with the school. The composer, with its corpus closed after one
day, reached in one night of 3,833 updates what three epochs with the corpus reached in
11,499 (held-out next-event accuracy of the slow parameters alone 0.826 to 0.852
against 0.828 to 0.850). In the soft-body world a sleep gene was selected out in every
run, from 0.37 of the founders to 0.01 to 0.03 of the living, with a higher model error
than the beings that learn by day: an awake being has the world at every tick, and a
sleeper trades its actions for second-hand presentations of what its store holds. So:
a learner with the world present learns by day at a slow rate above zero; a learner
whose day is over, or whose store holds what nothing will show it again, sleeps. What
the store reads at bedtime bounds the night; a store that is drowned by the day
(`record_averaging` for a store written hundreds of times per cell) teaches a drowned
rule.

## Maps: a structured input port

A record patch reads its inputs through the port `B u` (and the gate through `G u`).
By default that is one dense row per context channel. `StructuredPort` (in
`cadence.ports`) reads a declared layout instead: a `MapBlock` is a tied local
kernel over a `(channels, height, width)` grid of the inputs, the same kernel at
every position, whose output is a grid of channels laid out as retinotopic maps;
a `DenseBlock` is a plain matrix over a slice of the inputs. The patch's
equations do not change. The port supplies the three linear maps the adjoint
scan needs (apply, transpose, parameter gradient), so learning is the same rule,
and the context width is the port's output count.

```python
import numpy as np
from cadence import RecordPatchNet
from cadence.ports import DenseBlock, MapBlock, StructuredPort

fovea = MapBlock(start=0, channels_in=3, height=24, width=24, channels_out=8, kernel=5, stride=2)
rest = DenseBlock(start=fovea.inputs, inputs=52, outputs=64)
port = StructuredPort(fovea.inputs + 52, [fovea, rest])
net = RecordPatchNet(port.inputs, port.outputs, 4, port=port, cells=4096, active=32)
```

A port of dense blocks covering the inputs is the plain patch (the tests check the
outputs and the updates agree), and a map block answers the same thing wherever it
appears (a thing at one place and the same thing two positions on give the same
channel response two cells on). The blocks are genes: which slice, what grid, how
many channels, what kernel, what stride. `mask=` names which inputs the port hears,
one flag per input: a masked input is zero to every block in the forward map, its
transpose and the gradient, so which channels a patch reads is a gene too, without
touching the weights (a steering patch that reads the surprises and not the probes,
or a brain below a rung that is deaf to the channel the rung reads). A structured
port's checkpoint carries the port's layout and its mask; its `B` and `G` are the
blocks' kernels, packed.

## Two patches in depth

The gate of a record patch sees only the present input. That is enough for a
conjunction of the last two symbols: an arbitrary function of the ordered pair
is learned by the slow weights alone (0.9999 on a 16-symbol probe), because
the gate multiplies the carried context. It is not enough to tell a subject
from a later noun. On subject-verb agreement across prepositional phrases with
nouns of the opposite number, one patch chose the right verb number in 1.00,
1.00 and 0.03 of sentences with zero, one and two distracting nouns (0.51 with
records); a GRU and a two-layer transformer scored 1.00 throughout. A second
patch that reads the first patch's context has a gate that sees what came
before, and scored 1.00, 1.00, 1.00:

```python
import numpy as np
from cadence import RecordPatchStack

stack = RecordPatchStack(inputs=5, hidden=8, outputs=4, seed=5, cells=2048, active=16,
                         groups=(4,), record_rate=1.0)
symbols = np.random.default_rng(4).integers(5, size=(3, 8))
target = np.eye(4)[(symbols + np.roll(symbols, 1, axis=1)) % 4]
seen = stack.observe(np.eye(5)[symbols], target, rate=2.0, backtrack=True)
assert seen.updated and seen.writes == 24
```

`RecordPatchStack` is a context patch below a complete `RecordPatchNet` that
reads `[u, r1 * h1]`; the upper patch owns the readout and the records. The
upper adjoint scan hands its input gradient to the lower scan, and the
gradient matches finite differences in both patches. Work per moment stays
linear in the two widths.

For a single patch the previous-state Jacobian is diagonal when its input is
fixed. Its gates still multiply carried history, so this is not a theorem that
the patch represents only one level of conjunctions. In a stack the upper
nonlinearity mixes the lower context. The agreement result supports that
composition on the measured task; it does not establish a general world model.

Each component's quadratic seam statement holds conditional on its inputs.
Freeing both layers together makes the upper nonlinear port depend on the
lower state, so a jointly coupled energy is not automatically quadratic.
`RecordPatchStack` supplies an adjoint learning scan, not a joint `detune`
solver or a joint equilibrium convergence guarantee.

## Several patches joined by ports

`JointRecordPatches` couples several record patches through a fixed number of repair rounds. A
`Port(source, target, start, width)` declares a seam: the band
`[start, start + width)` of the source patch's scaled context `r * h` (the unit
its records read it in, and the unit the stack hands upward) is an input of the
target patch in the same moment. The target's gate and input port read it
through their ordinary weights, and its records key on what crosses; nothing
else crosses. The joint energy adds `1/2 |p[t] - S r h_source[t]|^2` per port to
the patches' seam energies, and at every moment `rounds` Jacobi rounds from the
previous moment's contexts approximate the coupled equations: round one is the delayed
port, each further round re-reads the other patch's context of this moment. For
a single one-way port from an independent source, two undamped rounds reach the
fixed point; longer port chains need more rounds. Recurrent ports have no general
convergence guarantee here. Inspect the remaining equations independently before
claiming a joint equilibrium: a fixed round count is not a stopping certificate.
The seam residual per moment and round is
the fourth instrument beside the residual, the surprise and the update.

Snapshots retain whether ports were cut, alongside each cortex's learned and
live state. Invalid/nonfinite paths cannot become live contexts. Joint record
writes are transactional: failure on a later cortex restores earlier writes.

```python
import numpy as np
from cadence import JointRecordPatches, Port, RecordPatchNet

eye = RecordPatchNet(inputs=5 + 3, hidden=8, outputs=4, seed=1, cells=256, active=8, groups=(4,))
ear = RecordPatchNet(inputs=5 + 2, hidden=6, outputs=4, seed=2, cells=256, active=8, groups=(4,))
brain = JointRecordPatches([eye, ear], own_inputs=[5, 5],
                           ports=[Port(1, 0, 0, 3), Port(0, 1, 2, 2)], rounds=2)
rng = np.random.default_rng(0)
xs = [np.eye(5)[rng.integers(5, size=(3, 8))] for _ in range(2)]
ys = [np.eye(4)[rng.integers(4, size=(3, 8))] for _ in range(2)]
seen = brain.observe(xs, ys, rate=2.0, backtrack=True)
assert seen.settled.seam.shape == (3, 8, 2, 2)      # batch, time, rounds, ports
```

Learning is one backward scan through the moments and the rounds; with
`cross_adjoint=True` the gradient of a target's loss reaches the source's context
through the port, with `False` the port is a plain input. The step is admitted by
the library's backtracking on the joint slow loss. Records are read at the
settled readings after the scan and written after the path, as in one patch.
Snapshots are the patches' snapshots plus the topology; `imagine` changes
nothing; `cut = True` makes every port carry zeros, the ablation. Which patch
hears which, the band and the width are a genome for `evolve` with `genes`;
the channels are ordered by timescale, so the band decides what crosses: the
fastest channels carry the source's current input, the slowest its carried
state. On two coupled symbol streams with one hidden cause, the joint port
lifted the dependent stream from 0.39 (two independent patches) to 0.54 at
matched parameters, the same-moment settling round carried the gain, and
cutting the port after learning dropped that stream to 0.27. There is no joint
`detune`: the joint energy is not quadratic in the joint path.

## What this class does not do

Planning through action ports, protected responses through `TemporalMemory`
and coupled components beyond the two-patch stack are not yet available for
this class. The stack has no `detune`. The record
rate, the number of active cells and the reading scale are supplied. There
is no automatic importance, forgetting, or learned relevance: a record is
written for every observed moment and moves only when its cells are read
again.
