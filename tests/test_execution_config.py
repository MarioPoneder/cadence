"""Execution configuration and optional dependencies require no tensor runtime."""

import builtins
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cadence import Brain, Cortex


def make_brain(device="python", dtype=None):
    cortex = Cortex(seed=4, device=device, dtype=dtype)
    sensor = cortex.input("sensor", shape=1)
    features = cortex.column("features", patches=2, inputs=sensor)
    patch = cortex.column("patch", patches=1, inputs=features)
    cortex.output("answer", shape=1, reads=patch)
    return cortex.build()


@pytest.fixture
def forbid_torch(monkeypatch):
    original = builtins.__import__
    attempts = []

    def unavailable(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            attempts.append(name)
            raise ImportError("Tensor runtime intentionally unavailable in this test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    return attempts


@pytest.mark.parametrize(
    ("device", "dtype", "resolved_device", "resolved_dtype"),
    (
        ("python", None, "python", "float64"),
        ("python", "float64", "python", "float64"),
        ("cpu", None, "cpu", "float64"),
        ("cpu", "float64", "cpu", "float64"),
        ("cpu", "float32", "cpu", "float32"),
        ("mps", None, "mps", "float32"),
        ("mps", "float32", "mps", "float32"),
        ("cuda", None, "cuda:0", "float64"),
        ("cuda", "float32", "cuda:0", "float32"),
        ("cuda:0", "float64", "cuda:0", "float64"),
        ("cuda:2", None, "cuda:2", "float64"),
        ("cuda:2", "float32", "cuda:2", "float32"),
    ),
)
def test_supported_execution_settings_resolve_without_loading_torch(
    forbid_torch,
    device,
    dtype,
    resolved_device,
    resolved_dtype,
):
    brain = make_brain(device, dtype)
    assert brain.config["device"] == resolved_device
    assert brain.config["dtype"] == resolved_dtype
    data = json.loads(brain.snapshot())
    assert data["config"]["device"] == resolved_device
    assert data["config"]["dtype"] == resolved_dtype
    assert Brain.from_snapshot(brain.snapshot()).snapshot() == brain.snapshot()
    assert forbid_torch == []


@pytest.mark.parametrize(
    "device",
    (
        None,
        True,
        0,
        "",
        "CPU",
        "auto",
        "cpu:0",
        "cuda:-1",
        "cuda:x",
        "cuda:1.5",
        "mps:0",
        "cuda:0 ",
    ),
)
def test_invalid_device_selectors_are_rejected_before_runtime_import(
    forbid_torch, device
):
    with pytest.raises(ValueError, match="device must be"):
        Cortex(device=device)
    assert forbid_torch == []


@pytest.mark.parametrize(
    "dtype", (True, 32, "", "double", "float16", "bfloat16", "torch.float32")
)
def test_invalid_precision_is_rejected_before_runtime_import(forbid_torch, dtype):
    with pytest.raises(ValueError, match="dtype must be"):
        Cortex(device="cpu", dtype=dtype)
    assert forbid_torch == []


@pytest.mark.parametrize(
    ("device", "dtype", "message"),
    (
        ("python", "float32", "reference engine uses float64"),
        ("mps", "float64", "requires float32"),
    ),
)
def test_unsupported_device_precision_pairs_fail_early(
    forbid_torch, device, dtype, message
):
    with pytest.raises(ValueError, match=message):
        Cortex(device=device, dtype=dtype)
    assert forbid_torch == []


@pytest.mark.parametrize(
    ("device", "dtype"), (("cpu", "float32"), ("mps", None), ("cuda", None))
)
def test_checkpoint_transfer_needs_no_tensor_runtime(forbid_torch, device, dtype):
    original = make_brain()
    assert original.observe({"sensor": [0.3]}, {"answer": [0.1]}, event_id=3)[
        "accepted"
    ]
    before = original.snapshot()
    transferred = Brain.from_snapshot(before, device=device, dtype=dtype)
    assert transferred.state == original.state
    assert transferred.weights == original.weights
    assert transferred.biases == original.biases
    assert transferred.inspect()["last_event_id"] == 3
    assert transferred.inspect()["admissions"] == 1
    assert transferred.inspect()["fingerprint"] != original.inspect()["fingerprint"]
    assert original.snapshot() == before
    assert transferred.observe({"sensor": [0.3]}, {"answer": [0.1]}, event_id=3)[
        "duplicate"
    ]
    round_trip = Brain.from_snapshot(transferred.snapshot(), device="python")
    assert round_trip.snapshot() == before
    assert forbid_torch == []


def test_dtype_only_checkpoint_override_preserves_selected_device(forbid_torch):
    brain = make_brain("cpu", "float32")
    transferred = Brain.from_snapshot(brain.snapshot(), dtype="float64")
    assert transferred.config["device"] == "cpu"
    assert transferred.config["dtype"] == "float64"
    assert transferred.weights == brain.weights
    with pytest.raises(ValueError, match="reference engine uses float64"):
        Brain.from_snapshot(make_brain().snapshot(), dtype="float32")
    assert forbid_torch == []


@pytest.mark.parametrize("tamper", ("source", "config", "fingerprint"))
def test_transfer_validates_original_checkpoint_before_overrides(forbid_torch, tamper):
    data = json.loads(make_brain().snapshot())
    if tamper == "source":
        data["implementation"]["_tensor.py"] = "0" * 64
    elif tamper == "config":
        data["config"]["device"] = "cpu"
    else:
        data["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="mismatch"):
        Brain.from_snapshot(json.dumps(data), device="mps")
    assert forbid_torch == []


@pytest.mark.parametrize("operation", ("settle", "step", "observe"))
def test_missing_optional_dependency_is_clear_and_keeps_continuation(
    forbid_torch, operation
):
    brain = make_brain("cpu")
    before = brain.snapshot()
    args = (
        ({"sensor": [0.3]}, {"answer": [0.1]})
        if operation == "observe"
        else ({"sensor": [0.3]},)
    )
    with pytest.raises(ImportError, match=r'pip install "cadence-net\[gpu\]"'):
        getattr(brain, operation)(*args)
    assert brain.snapshot() == before
    assert brain.inspect()["last_event_id"] == -1
    assert forbid_torch == ["torch"]


def test_unavailable_device_does_not_install_partial_continuation(monkeypatch):
    requests = []

    def unavailable_empty(size, *, device, dtype):
        requests.append((size, device, dtype))
        raise RuntimeError("simulated unavailable device")

    # An allocation stub checks Cadence's error/rollback path on base-only CI.
    # It is deliberately not evidence of successful CUDA execution.
    fake_dtype = object()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float64=fake_dtype, empty=unavailable_empty),
    )
    brain = make_brain("cuda:7")
    before = brain.snapshot()
    with pytest.raises(ValueError, match="cuda:7.*unavailable"):
        brain.observe({"sensor": [0.3]}, {"answer": [0.1]}, event_id=8)
    assert brain.snapshot() == before
    assert requests == [(0, "cuda:7", fake_dtype)]


def test_default_path_never_imports_torch_in_a_clean_process():
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import builtins
import sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('default execution imported torch')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked
from cadence import Brain, Cortex
cortex = Cortex(seed=4)
sensor = cortex.input('sensor', shape=1)
features = cortex.column(patches=2, inputs=sensor)
patch = cortex.column(patches=1, inputs=features)
cortex.output('answer', shape=1, reads=patch)
brain = cortex.build()
assert brain.observe({'sensor': [0.3]}, {'answer': [0.1]})['accepted']
assert brain.step({'sensor': [0.3]})['accepted']
assert brain.predict({'sensor': [0.3]})
assert Brain.from_snapshot(brain.snapshot()).snapshot() == brain.snapshot()
assert 'torch' not in sys.modules
print('dependency-free execution passed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert result.stdout.strip() == "dependency-free execution passed"
