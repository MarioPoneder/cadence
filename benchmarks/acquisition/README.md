# Acquisition and retention microscope

This bounded CPU protocol teaches a System 1 brain actual sensory/action rows
from the recorded no-damage Castlevania movie. The brain consists of bounded
observer-like regions with local state, sensory/action ports, readback, records
and plastic repair. Every issued answer requires the original full equations at
residual tolerance `0.003`; a teaching target is never an answer clamp.

From the library checkout:

```sh
.venv/bin/python benchmarks/acquisition/run.py \
  --out /tmp/cadence-acquisition-run \
  --recipe both --updates 128 --check-every 8 \
  --free-steps 1024 --nudged-steps 1024 --seconds 120
```

Use a new output directory for each attempt. The finite control always keeps the
0.70 configuration: `650` inputs, `36` actions, modules `(32,16)`, no observers,
seed `0`, `1024/12/12` maximum phase sweeps, beta `0.1`, synaptic rate `0.5`, bias
rate `0.02`, cross-entropy temperature `0.2` and momentum `0.9`. `--seed` changes
both founders explicitly. Rate, phase budgets, nudge and damping arguments alter
the qualified candidate. They are candidate genes; the historical finite
configuration remains the control. Both use the raw sensor boundary and
float64, without calibration or an auxiliary classifier.

The explicit candidate `--gene fixed-lateral-local-rms` starts sensory biases at
`0.6`, freezes the existing motor-to-motor lateral synapses, and uses synaptic
rate `0.005` with local RMS normalization `0.99`. Other synapses, every bias,
reciprocal tying and the System 1 mechanisms remain present. This candidate does
not change the default library. Use `--tolerance 1e-6 --free-steps 4096
--nudged-steps 4096` to check it with the original `dt=1` model and bounded
numerical damping.

`protocol.json` freezes arguments, fixture/runtime/source hashes, gates and
resource model before numerical work. The microscope reproduces the finite
free/positive/negative phases, checks the original residual through independent
edge scatter, reads one/two further undamped steps, and checks contrast
arithmetic with independent array products. It records the 24-row diagnostic
without teaching it. Training starts with two selected relations, then four,
then 24; each stage starts from a fresh identical founder. Expansion requires
perfect qualified free recall on the preceding stage. The 24-row screen requires
at least `18/24` correct and zero refusals. A screen is acquisition evidence on
selection rows.

Every attempted training phase has raw potential, activation and adaptation
arrays, original-equation residuals, motor readback and work reports. Initial and
final checkpoints, refused phase states, unknown failures and capped attempts
remain in their own directories. Costs include accepted row presentations,
phase/query sweeps, independent checks, reported residual transports and saved
continuation replay. Transport estimates do not measure hardware operations.
The native `run.py` default limit is 60 seconds/64 MiB; it permits at most 300
seconds/80 MiB. The other instruments freeze their own limits before work.
These are bounded local runs and launch no cloud resource.

After a passed 24-row screen the frozen model reads independent TRAIN and
development panels once, then checks save/load predictions and one additional
teaching continuation. Same-set replay does not establish retention. A separate
family-disjoint interference test and five fresh-seed held-out confirmation are
required before a promoted acquisition/generalization claim.

The small fixture preserves the exact 24 historical selected rows and their
native frame/action/input hashes. Independent TRAIN supplies only 18 of these
families: six TRAIN classes occur once. Development supplies 19 selected
families. Fourteen families have disjoint TRAIN/development/test rows; their
balanced confirmation panels are frozen separately. The runner never reads
held-out arrays. This single movie supplies no fresh gameplay or emulator-parity
claim. The parent source and native receipt stay in the owning application.

Regenerate into a new folder from the local workspace parent fixture:

```sh
.venv/bin/python benchmarks/acquisition/extract.py --out /tmp/cadence-school
.venv/bin/python -m pytest -q benchmarks/acquisition/test_protocol.py
```

The extractor checks the parent witness/native/selection hashes and each selected
row's position, frame, label and little-endian float64 sensory hash. It copies
only the small panels, not the parent corpus. Provenance records the original
selection and source boundary; new results retain their own source identity.

Runs retain source bytes under `source/library/cadence`, plus the harness and
fixture. Replay an artifact using its archived source:

```sh
PYTHONPATH=/absolute/run/source/library .venv/bin/python \
  benchmarks/acquisition/verify.py /absolute/run
```

The verifier checks hashes, the complete scheduled case census, independent
phase residuals, replayed parameters and optimizer/counters, work sums and final
free recall. Its receipt names checks it does not replay. `--allow-source-drift`
permits diagnostic comparison and produces no source-bound pass. Earlier recall,
microscope trajectories, development and retention need their own independent
checks. Ten adversarial mutations test false acceptance, exposure/work counts,
altered phases/checkpoints, missing cases and changed summary/protocol identity.

## Identifiable relations and durable memory

Movie action labels are future-window summaries, not invariant semantic cue
families. Keep the native results above, including weak independent recall.
`relations.py` supplies a separate association instrument: 24 disjoint cue
families, a fixed shuffled mapping to 24 of the 36 actions, and independently
drawn nuisance variants. Four TRAIN instances, two development instances and two
sealed held-out instances belong to each family. A family credit requires every
instance in the queried panel to be correct. This measures recall of learned
families under nuisance changes; it does not measure native gameplay or inference
of unseen facts. Nearest-example and centroid controls establish that the declared
relation is identifiable; their answers are never Cadence answers.

```sh
.venv/bin/python benchmarks/acquisition/relations.py --out /tmp/cadence-relations
.venv/bin/python benchmarks/acquisition/relation_development.py \
  --out /tmp/cadence-relation-development --genes canonical \
  --updates 512 --seconds-per-gene 600
.venv/bin/python benchmarks/acquisition/continual.py \
  --out /tmp/cadence-memory-controls --relations --rounds 128
.venv/bin/python benchmarks/acquisition/continual.py \
  --out /tmp/cadence-memory-confirmation --relations --confirm-memory \
  --rounds 128 --seconds-per-arm 300
.venv/bin/python benchmarks/acquisition/graph_confirmation.py \
  --out /tmp/cadence-graph-confirmation --prepare-only
.venv/bin/python -m pytest -q benchmarks/acquisition/test_relations.py
```

The graph confirmation command above freezes the complete five-founder
protocol for review without teaching. Its full run has a 1,200-second
admission bound per founder and a separately recorded mandatory readback tail;
it is longer than the native microscope. Use a new output directory for an
executing run without `--prepare-only`. The declared source census retains all
five founders, including failures, and keeps their held-out panels sealed
unless each matching development gate passes.

The `canonical` graph control uses the parameter defaults of `LearnerConfig`
with explicit qualified phases, 4,096-sweep phase budgets, tolerance `.003` and
at most three numerical halvings. It is distinct from the implicit teaching
configuration of `Brain.compose`. Graph development uses only local free/±nudge
contrasts, with no associative read or write. Accepted phases retain per-row
equation checks and exact vector/parameter hashes for source-bound replay;
refused phases retain raw states. Cold queries start with fresh graph activity.

The `continual.py` memory arms explicitly observe actual cue/teacher-label pairs
through the existing consolidating store. They perform no graph contrast updates
and do not reinterpret a teacher label as a reward. Every answer still comes from
the original graph equilibrium with recall supplying drive at its motor ports.
Queries clear live activity, working trace and the fast associative
residual, retaining consolidated weights. `Brain.act(..., greedy=True)` is checked
separately in confirmation. These arms test durable associative storage, not the
graph learner's acquisition contract.

Retention starts from an actually acquired four-family checkpoint. Twenty new
families are then taught for a frozen schedule, with a separate rehearsal arm
that pays for old-example presentations. Old and new families are queried one
cue at a time after clearing temporary state. This does not require 24 facts to
fit simultaneously in working memory. The gates are at least three old and
15 new family credits, all scheduled rounds completed, and zero refusals.
Save/load checks include the next actual write or teaching update. Five-founder
confirmation retains every founder and opens held-out data only for a matching
recipe that passed development. A memory-control pass does not discharge the
separate local-contrast gate; neither pass proves lifelong retention or an
efficiency advantage.
