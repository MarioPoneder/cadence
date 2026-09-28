# GPU execution and parallel experience

Cadence can accelerate repair with CPU or GPU tensors and run independent
simulated lives in separate processes. The brain's interface stays the same:
`settle` queries, `step` retains qualified activity, and `observe` admits an
actual witness through joint state/parameter repair. `bootstrap` replays those
witnesses and checks unclamped predictions.

More width or recursive depth increases capacity and cost. Whether it improves
a task must be tested; adding layers does not guarantee successful learning.
Acceleration changes execution, not the information supplied, the learning
objective or the need for a useful teaching stream.

## Select execution explicitly

The default `Cortex()` uses Python float64 and has no runtime dependencies.
Install the optional tensor backend when needed:

```sh
python -m pip install "cadence-net[gpu]"
```

| `device` | Proposal arithmetic | Default `dtype` |
| --- | --- | --- |
| `"python"` | Standard-library reference engine | `"float64"` only |
| `"cpu"` | PyTorch tensors on CPU | `"float64"`; also supports `"float32"` |
| `"mps"` | PyTorch tensors on an available Apple GPU | `"float32"` only |
| `"cuda"` or `"cuda:N"` | PyTorch tensors on an available NVIDIA GPU | `"float64"`; also supports `"float32"` |

`"cuda"` resolves to `"cuda:0"`. Set `dtype` explicitly to override the selected
device's default where supported. PyTorch supplies tensor arithmetic; Cadence
computes its own analytic derivatives and repair steps. There is no PyTorch
neural-network module, autograd optimizer or second learning algorithm here.

This small reference example is runnable with the base installation:

```python
from cadence import Brain, Cortex, bootstrap

layout = Cortex(seed=2, device="python")
signal = layout.input("signal", shape=1)
base = layout.column("perception", patches=4, inputs=signal)
observer = layout.observer("reflection", patches=2, inputs=signal, observes=base)
layout.output("answer", shape=1, reads=observer)
brain = layout.build()

examples = [({"signal": [x]}, {"answer": [x]}) for x in (-0.8, 0.8)]
checks = [({"signal": [x]}, {"answer": [x]}) for x in (-0.4, 0.4)]
report = bootstrap(brain, examples, checks=checks, max_error=0.2)
assert report["passed"], report
saved = brain.snapshot()
```

Set `device="mps"` or `device="cuda"` on the same constructor to bootstrap on
the corresponding GPU. Use `device="cpu"` to test tensor execution without a
GPU. The hardware and tensor library must support the selected configuration.
Construction and checkpoint loading do not import PyTorch or reserve a device;
the first solve does. Missing PyTorch raises `ImportError`, and unavailable
hardware raises `ValueError`. Cadence does not silently choose another device.

CPU float64 and Apple MPS float32 have been exercised through bootstrapping,
held-out predictions and live continuation. The CUDA path is implemented, but
this release has not been validated on NVIDIA hardware.

## Keep admission precise

All tensor proposals are finally checked by the Python float64 reference engine
against the **original** inputs, exact clamps, original parameter anchors and,
for queries, original frozen weights/biases. Float32 proposal arithmetic never
loosens the configured stationarity tolerance. Reference refinement can use
the remaining accepted-sweep allowance. If the candidate raises the original
float64 energy beyond the permitted rounding allowance, reference repair
restarts from the original coordinates within that remaining allowance.
For float32, the device phase accepts at most `max(1, budget // 2)` sweeps
when the budget is positive, reserving at least half of any budget of two or
more for reference refinement. Float64 may use the full device allowance.
The device stopping hint is `max(tolerance, 64 * dtype_epsilon)`; it does not
change the final admission tolerance. The complete solve still obeys `budget`;
a refused proposal is never committed.

The two precisions need not take identical steps or reach the same stationary
point. Inputs and parameters must fit the selected proposal dtype. Float32
may need more reference refinement at a tight tolerance, and unsuitable
numeric magnitudes can raise `ValueError` without changing continuation.

For tensor solves, `result["execution"]` records `device`, `dtype`, `torch`,
`tensor_sweeps`, `reference_sweeps`, `reference_evaluations` and
`reference_restart`. The ordinary `work` and `sweeps` totals include device
and reference work. `energy`, `errors`, `stationarity` and `qualified` are the
final reference diagnostics. `energy_history` combines the approximate device
trajectory and any reference refinement; it is not a float64 certificate of
monotonic decrease at every intermediate device step.

## Move a continuation between devices

Use an explicit override when loading a compatible checkpoint:

```python
reference = Brain.from_snapshot(saved, device="python")
assert reference.state == brain.state
assert reference.weights == brain.weights
assert reference.biases == brain.biases
assert reference.inspect()["admissions"] == brain.inspect()["admissions"]
```

For example, move the same saved brain to an Apple GPU:

<!-- not-run: Requires the optional PyTorch installation and available MPS hardware. -->
```python
gpu = Brain.from_snapshot(saved, device="mps")
result = gpu.settle({"signal": [0.4]})
assert result["qualified"], result
print(result["outputs"], result["execution"])
```

The entire original checkpoint is validated before applying execution overrides.
This retains its arrays and witness cursor exactly, then changes configuration
and fingerprint. A device override without `dtype` selects that device's
default; a dtype-only override keeps the saved device. With both omitted,
saved settings are retained. `restore` still requires the exact complete
configuration. Source compatibility remains strict: transfer between devices
is not permission to load a checkpoint from different engine sources.

A snapshot preserves arrays and implementation identity exactly, but tensor
arithmetic can vary with hardware and PyTorch version. In particular, PyTorch
documents that [CUDA `index_add_` can be nondeterministic](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.index_add_.html).
A seed alone therefore does not guarantee bitwise GPU replay. Record hardware,
library versions and execution settings alongside experiment receipts. Cadence
does not change global deterministic flags; the default Python engine does
not use these tensor reductions.

## Parallel simulated lives

From the repository with Cadence installed, run:

```sh
python examples/parallel_bootstrap.py --lives 4 --workers 2
```

The [complete example](../examples/parallel_bootstrap.py) creates independent
point-body simulators and brains in spawned CPU processes. Each brain
bootstraps a small relation between position, velocity and measured next
position; live queries predict fresh outcomes before those outcomes become
new witnesses. Its JSON report includes readiness, counted work, admissions,
fresh predictions, wall time and checkpoint digests. This is a dynamics lesson,
not a learned walking policy or proof of a depth advantage.

Each worker owns its brain. Experiences within a brain stay ordered because
one witness changes the anchors used by the next. Sharing a mutable brain
between workers or averaging checkpoints does not preserve this rule.
Independent environment collectors can instead send an ordered witness stream
to one owner; the application must preserve event identity and provenance.
Do not start one GPU process per CPU core: many small jobs can contend for the
same accelerator. This example deliberately uses the dependency-free engine.

## Measure the complete improvement

Compare the same starting arrays, inputs, witness sequence, stopping criteria
and task checks. Warm up each device separately and disclose startup cost.
Measure wall time around complete calls, including tensor transfers, device
synchronization and final reference checks. Record device/dtype, hardware,
library versions, qualification/refusal counts, sweep/work totals, acquisition
and retention alongside speed. Count repeated bootstrap presentations
separately from new simulator experience.

Small graphs may run faster in Python because dispatch and synchronization
cost more than their arithmetic. Wider graphs can provide more parallel work;
additional error-observation levels also add dependencies. Benchmark the
actual intended layout before choosing hardware or increasing worker count.
Changing `tolerance` changes what gets admitted and must not be hidden inside
a speed comparison.

## Measured starting points

On an Apple M4 with Python 3.13 and PyTorch 2.14, a bounded comparison used
64 inputs, four output coordinates, three seeds, two repetitions and tolerance
`1e-6`. Each repeat queried twice, admitted two supplied witnesses, then queried
twice more. All 432 public calls qualified; final checks and refinement are
included in the timings below.

| Layout | CPU tensor / Python speedup | MPS / Python speedup |
| --- | ---: | ---: |
| 8 patches | 2.47× | 0.25× |
| 64 patches | 9.16× | 1.12× |
| 256 patches | 19.55× | 2.31× |
| 128 patches + 32 observers | 15.34× | 2.74× |

These are medians of paired **witness-update** time ratios, not complete skill
acquisition or query-only speedups. A ratio below one is a slowdown. Over the
complete six-call sequence, the 256-patch ratios were 16.25× for CPU tensors and
2.17× for MPS; the recursive layout ratios were 13.63× and 2.40×. Initialization
was measured separately. Devices did not produce bit-identical trajectories;
maximum paired output difference after the witness updates was `8.20e-7`.

For these sizes, try `device="cpu"` first. GPU support is useful without being
the fastest option for every layout. These measurements establish execution
improvements on the stated workload, not an architectural advantage from depth.

The independent-life example also ran 16 lives with one, two and four workers,
three repetitions each. Median complete process times, including startup and
JSON output, were 0.795, 0.496 and 0.340 seconds: 1.60× and 2.34× speedups.
All 144 life outcomes matched serial execution apart from timing and process
IDs. Small workloads can still lose to process overhead on other machines.
