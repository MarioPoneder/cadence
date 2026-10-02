"""A separator's actual coordinate system survives full-brain checkpoints."""

import json

import numpy as np
import pytest

import cadence as cd


def separated_brain(kind="consolidating", backend="cpu"):
    brain = cd.GenericBrain.build(
        3,
        3,
        hidden=5,
        episodic=True,
        working_memory=False,
        seed=9,
        backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    separator = cd.PatternSeparator(3, 16, 3, seed=7, center=0 if kind == "consolidating" else 0.3)
    # The seed cannot recover a customized projection or the observed running mean.
    separator.projection = np.roll(separator.projection, 4, axis=1) * 1.2
    separator.mean = np.array([0.2, -0.3, 0.4])
    memory_class = cd.SynapticMemory if kind == "consolidating" else cd.FastSynapses
    options = {"consolidation": 0.25} if kind == "consolidating" else {}
    brain.hippocampus = memory_class(
        brain.sensory_index,
        brain.motor_index,
        separator=separator,
        decay=0.85,
        rate=0.7,
        amplitude=1.3,
        rule="delta",
        **options,
    )
    brain.hippocampus.observe(np.eye(3)[:2], np.array([[1.0, -0.2, 0.5], [-0.3, 0.6, 1.2]]))
    brain.step(np.eye(3)[:2])  # The next reward must finish this saved pending action.
    return brain


def assert_memory_equal(left, right):
    assert left.to_dict() == right.to_dict()
    for name in ("pre", "post", "strength", "mass"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    if isinstance(left, cd.SynapticMemory):
        np.testing.assert_array_equal(left.consolidated, right.consolidated)
    if left.separator is not None:
        for name in ("projection", "mean"):
            a, b = getattr(left.separator, name), getattr(right.separator, name)
            np.testing.assert_array_equal(a, b)
            assert not np.shares_memory(a, b)


@pytest.mark.parametrize("kind", ["consolidating", "fast"])
@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_expanded_memory_resumes_pending_action_and_future_learning(tmp_path, kind, backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    original = separated_brain(kind, backend)
    restored = cd.GenericBrain.load(
        original.save(tmp_path / "separated.npz"),
        backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    assert_memory_equal(original.hippocampus, restored.hippocampus)
    assert restored.hippocampus.strength.shape == (2, 16, 3)
    for i in range(5):
        x = np.eye(3)[[i % 3, (i + 1) % 3]]
        reward, done = np.array([0.7, -0.4]), np.array([i == 2, i == 3])
        np.testing.assert_array_equal(
            original.step(x, reward=reward, done=done),
            restored.step(x, reward=reward, done=done),
        )
        np.testing.assert_array_equal(original.brain.efficacy, restored.brain.efficacy)
        np.testing.assert_array_equal(
            original.basal_ganglia.w_critic, restored.basal_ganglia.w_critic
        )
        assert_memory_equal(original.hippocampus, restored.hippocampus)
    # A new stream must also reconstruct the persistent matrix in expanded coordinates.
    original.reset()
    restored.reset()
    np.testing.assert_array_equal(original.step(np.eye(3)), restored.step(np.eye(3)))
    assert_memory_equal(original.hippocampus, restored.hippocampus)


def rewrite_checkpoint(path, mutate):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved}
    metadata = json.loads(str(arrays["generic"]))
    mutate(metadata, arrays)
    arrays["generic"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("kind", ["consolidating", "fast", "none"])
def test_legacy_unseparated_memory_keeps_its_original_rule(tmp_path, version, kind):
    original = cd.GenericBrain.build(3, 3, hidden=5, working_memory=False)
    if kind == "fast":
        original.hippocampus = cd.FastSynapses(original.sensory_index, original.motor_index)
    elif kind == "none":
        original.hippocampus = None
    original.step(np.eye(3)[:2])
    original.step(np.eye(3)[:2], reward=np.array([0.7, -0.2]))
    path = original.save(tmp_path / "legacy.npz")

    def make_legacy(metadata, arrays):
        metadata["format"] = f"cadence-generic/{version}"
        if metadata["hippocampus"] is not None:
            metadata["hippocampus"].pop("separator")

    rewrite_checkpoint(path, make_legacy)
    restored = cd.GenericBrain.load(path)
    np.testing.assert_array_equal(
        original.step(np.eye(3)[:2], reward=np.array([0.1, 0.5])),
        restored.step(np.eye(3)[:2], reward=np.array([0.1, 0.5])),
    )
    np.testing.assert_array_equal(original.brain.efficacy, restored.brain.efficacy)
    if kind == "none":
        assert restored.hippocampus is None
    else:
        assert_memory_equal(original.hippocampus, restored.hippocampus)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_projection",
        "missing_mean",
        "projection_shape",
        "projection_nan",
        "mean_shape",
        "mean_inf",
        "missing_separator",
        "missing_hippocampus",
        "separator_inputs",
        "separator_winners",
        "separator_seed",
        "separator_center",
        "separator_unknown",
        "strength_shape",
        "mass_shape",
        "mass_negative",
        "consolidated_shape",
        "consolidated_nan",
        "kind",
        "pre_bounds",
        "pre_fraction",
        "writes",
        "normalize",
        "missing_rate",
        "persistent_parameters",
    ],
)
def test_malformed_memory_rejected_before_resumed_composition_is_constructed(
    tmp_path, monkeypatch, defect
):
    original = separated_brain()
    path = original.save(tmp_path / "malformed.npz")

    def corrupt(metadata, arrays):
        memory = metadata["hippocampus"]
        if defect.startswith("missing_") and defect in ("missing_projection", "missing_mean"):
            del arrays["episodic/separator/" + defect.removeprefix("missing_")]
        elif defect == "projection_shape":
            arrays["episodic/separator/projection"] = np.zeros((3, 15))
        elif defect == "projection_nan":
            arrays["episodic/separator/projection"][0, 0] = np.nan
        elif defect == "mean_shape":
            arrays["episodic/separator/mean"] = np.zeros(4)
        elif defect == "mean_inf":
            arrays["episodic/separator/mean"][0] = np.inf
        elif defect == "missing_separator":
            del memory["separator"]
        elif defect == "missing_hippocampus":
            metadata["hippocampus"] = None
        elif defect.startswith("separator_"):
            name = defect.removeprefix("separator_")
            memory["separator"][name] = {
                "inputs": 4,
                "winners": 17,
                "seed": -1,
                "center": 0.2,
                "unknown": 0,
            }[name]
        elif defect == "strength_shape":
            arrays["episodic/strength"] = np.zeros((2, 3, 3))
        elif defect == "mass_shape":
            arrays["episodic/mass"] = np.zeros(3)
        elif defect == "mass_negative":
            arrays["episodic/mass"][0] = -1
        elif defect == "consolidated_shape":
            arrays["episodic/consolidated"] = np.zeros((3, 3))
        elif defect == "consolidated_nan":
            arrays["episodic/consolidated"][0, 0] = np.nan
        elif defect == "kind":
            memory["kind"] = "unknown"
        elif defect == "pre_bounds":
            arrays["episodic/pre"][0] = original.connectome.n
        elif defect == "pre_fraction":
            arrays["episodic/pre"] = arrays["episodic/pre"].astype(float) + 0.5
        elif defect == "writes":
            memory["writes"] = -1
        elif defect == "normalize":
            memory["normalize"] = "false"
        elif defect == "missing_rate":
            del memory["rate"]
        elif defect == "persistent_parameters":
            memory["persistent_parameters"] = 1

    rewrite_checkpoint(path, corrupt)
    before = path.read_bytes()

    def must_not_construct(*args, **kwargs):
        pytest.fail("malformed memory must be rejected before constructing the resumed brain")

    monkeypatch.setattr(cd.GenericBrain, "__init__", must_not_construct)
    with pytest.raises(ValueError):
        cd.GenericBrain.load(path)
    assert path.read_bytes() == before
