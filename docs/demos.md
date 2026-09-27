# The quickstarts in your browser

Each of the three quickstart brains runs behind a local page. The page draws the whole brain
with the shipped [brain viewer](pages.md), every neuron a point coloured by its region and
every synapse a line, and animates it as the brain computes; it plots the learning as it is
measured; it shows the task; and it says, in words, what is going on in the phase you are
watching. Nothing is hosted and nothing is trained ahead of time: the command trains the brain
in front of you, from fixed seeds, so every run is the same.

```bash
python -m pip install cadence-net
cadence-demo stream     # a record patch learns a stream, remembers in one shot, and sleeps
cadence-demo decide     # a settling brain decides
cadence-demo body       # a temporal patch learns a consequence and plans
```

`--port` picks the port and `--no-browser` only serves. A run takes seconds to a minute.

## What equilibrium means in each brain, and what detuning buys

In Cadence a brain computes by settling into an equilibrium and learns by detuning it. The
three brains are three forms of that one idea, and each page shows the equilibrium in more
than one way: as the activity of the neurons, as a heat you can switch on that colours every
neuron by its distance from the equilibrium, as the last change of every synapse, and as
numbers in the stats strip.

**The settling brain** (`decide`). The task: five senses, one at a time, and three possible
actions; sense k asks for action k mod 3. The brain answers with its most active motor neuron
and is then shown the action that was wanted, and it has to learn to choose the right action
for all five senses. Five senses, a cortex of twenty-four neurons, three actions. The
equilibrium is the settled state: every neuron's potential moves under its
synapses until no neuron moves, and the state the brain stops in is its answer. On the page
the heat is how far each neuron still moves at each settling step; it fades to black as the
brain settles, and the stats strip counts the steps and the largest last movement. Learning
is the detuning of that equilibrium: the brain settles once more with the chosen action
nudged, and every synapse moves on the contrast of the two settled states, the nudged minus
the free. The heat on the nudged frame is how far the nudge displaced each neuron; switch the
synapses to *last change* to see where the contrast landed. The benefit is that nothing is
propagated backward and no gradient is stored: the same settle that produces the answer
produces, with one nudge, the learning signal, locally at every synapse. The accuracy curve
rises to one and the synapse-change curve falls as the two equilibria come to agree.

**The temporal patch** (`body`). The task: a body on a line with one measured position and
one push between minus one and plus one per step, moving by a rule the brain is never shown.
First learn how pushes move the body from random pushes and the positions they produced,
well enough to predict six steps ahead on unseen paths; then drive the body to the goal
position 0.35 and hold it there, planning six pushes ahead under the learned model,
executing the first and replanning from the measured position, through an unannounced shove
halfway. Here the equilibrium is a path: the causal recurrence over six moments is the zero-defect
solution of the patch's energy, and the stats strip shows the free residual and energy. The
observed positions detune it: the path is settled twice more with the outputs pulled by plus
and minus beta, and the synapses move on the difference of the two detuned paths divided by
two beta. The stats show how many solves each detuned phase took, the asymmetry of the two
around the free path (the check that the contrast is a derivative at all, with beta halved
when it is not), and the size of the update. The heat is how far the plus phase displaced
each unit of the body from its free equilibrium. Planning uses the same contrast on the
action port with the synapses frozen; the page shows the imagined path of every plan, the
number of contrast steps it took, and the measured position against the goal, with a
disturbance halfway that the next plan simply starts from. The benefit is one operation for
learning and for acting: a detuned equilibrium is a learning signal when the weights are
free and an action repair when they are frozen.

**The record patch** (`stream`). The task: three streams of eight symbols, each a number from
0 to 11, play one moment at a time, and at every moment the brain has to say one of five
outcomes, the current symbol plus the previous one modulo 5. It is never told the rule; it is
shown the right outcome after each moment. It has to recall every outcome after one pass by
day, and after a night with the streams closed say every outcome right with its memory
erased. The equilibrium is again a path, the gated context:
at every moment it satisfies its gate equation exactly, so the seam defect is zero (the stats
strip says so) and the reading of input and context is settled before the record store reads
it. Sixteen of the 2,048 record cells light for each reading; by day the outcome of the
moment is written once into those cells and the slow weights do not move, and the records
recall the stream after a single pass. At night the stream is closed, the store completes each
cue from rest, the completions are fixed as dreams, and the slow weights learn them: the
teaching loss pulls on the outcome ports, the heat on the page, and the adjoint scan carries
the pull back along the settled path, which on this quadratic path is the centered detuning
contrast. At dawn the dreams are written back, so the store holds only what the weights did
not take. The benefit is a memory that takes a fact in one write and a rule that moves into
the weights without any teacher outside the patch: the store's own equilibrium completions
are the night's targets. The chart shows the two accuracies, with the records and with the
slow weights alone, crossing over the night.

## The rung demos

Beyond the three quickstarts, every step of the program is accepted by a demo of the same
shape: the whole brain in the viewer, the world with the person in it (the mouse as the
laser dot, the hand that drags the voice, the click that makes the baby cry), the instrument
strip, and a three-way switch (the step on; off, the brain below it; the hand-designed
control) so the acceptance is visible on the page. The dozing cat is rung 1 (a governor that
reads the brain's surprise and decides when to wake) and the fifth canonical example; the
ventriloquist is rung 2 (a steering patch weighs the eye and the ear); the lighthouse keeper
is rung 3 (a beam the brain sweeps under a cost); the night nursery is rung 4 (a clock that is
forgotten and a cry that keeps the ear). What they wrote around the library is the library's
[composition layer](steering.md), and [how to build a rung demo](howto-rung.md) is the order
of work for the next one.

## What the page draws

- **The whole brain.** Neurons as points coloured by region, synapses as lines. Brightness is
  activation; the glow is change; the particles are messages along synapses. *Neurons* can
  show activation or the distance from equilibrium. *Synapses* can show the weights or their
  last change. The strip below traces every region over the last steps.
- **The task.** The stream with the outcome the brain says under each symbol; the five senses
  against the three actions with the asked and the chosen action; the body on its line with
  the goal, the imagined path of the current plan, and the body with no action.
- **What is going on.** One paragraph for the phase you are watching, then the result.
- **The charts.** What the brain knows, and the detuning: the slow loss by night, the synapse
  change per step, the update size per batch.
- **The stats.** The numbers that define the equilibrium and its detuning in that brain.

The demos are the quickstart snippets with the same seeds and the same calls; `cadence.demo`
holds them (`StreamDemo`, `DecideDemo`, `BodyDemo`, `serve`, `main`), and each exposes its
brain as a `Connectome` for the viewer. The `decide` demo records its settlements through
`record_settlements`, which uses the inspectable NumPy path, so its arithmetic can differ
from the fused kernel at round-off.
