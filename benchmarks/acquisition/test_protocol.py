"""Independent accounting/fixture guards for the native acquisition protocol."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadence import Brain, BrainState, LearnerConfig

ROOT = Path(__file__).parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extract = load_script("extract")
run = load_script("run")
verifier = load_script("verify")


def test_fixture_panels_are_hash_bound_disjoint_balanced_and_sealed():
    folder = ROOT / "fixture"
    provenance = json.loads((folder / "provenance.json").read_text())
    assert extract.sha256(folder / "school.npz") == provenance["fixture_sha256"]
    with np.load(folder / "school.npz", allow_pickle=False) as data:
        for name, identities in provenance["panels"].items():
            inputs, labels = data[name + "_inputs"], data[name + "_labels"]
            assert inputs.dtype == np.dtype("float64")
            assert inputs.shape == (len(identities), 650)
            assert np.isfinite(inputs).all()
            for row, identity in enumerate(identities):
                assert extract.row_hash(inputs[row]) == identity["input_sha256"]
                assert labels[row] == identity["label"]
        school = {r["parent_row"] for r in provenance["panels"]["school"]}
        independent = {r["parent_row"] for r in provenance["panels"]["independent_train"]}
        assert not school & independent
        assert len(independent) == 18
        assert len(provenance["panels"]["development"]) == 19
        families = provenance["confirmation_families"]
        assert len(families) == 14
        for name in ("confirmation_train", "confirmation_development", "heldout"):
            assert sorted(data[name + "_labels"].tolist()) == families


def test_independent_residual_matches_original_nudge_and_detects_false_cache():
    brain = Brain.compose(
        inputs=2,
        actions=2,
        modules=(3,),
        seed=4,
        learning=LearnerConfig(free_steps=12, nudged_steps=12),
    )
    inputs, labels = np.eye(2), np.array([0, 1])
    drive = brain.stimulus(inputs, memory=False)
    free = brain.learner.free(drive)
    for sign in (0, 1, -1):
        target = brain.learner.targets(labels)
        state = free if sign == 0 else brain.learner.nudged(drive, free, target, sign=sign)
        nudge = None if sign == 0 else brain.learner.nudge_for(target, sign * 0.1)
        residual, cache = run.independent_residual(brain, drive, state, labels, sign)
        np.testing.assert_allclose(
            residual, brain.brain.residual(drive, state, nudge=nudge), atol=2e-14, rtol=2e-14
        )
        np.testing.assert_allclose(cache, 0, atol=2e-14)
    corrupted = BrainState(free.v, free.activation + 0.5, free.adaptation, free.steps)
    _, cache = run.independent_residual(brain, drive, corrupted)
    assert np.all(cache > 0.49)


def test_candidate_arguments_cannot_silently_change_finite_baseline():
    args = SimpleNamespace(rate=0.01, free_steps=17, nudged_steps=19, nudge="quadratic", damping=2)
    finite = run.make_brain("finite", 0, args)
    cfg = finite.learner.config
    assert (cfg.free_steps, cfg.nudged_steps, cfg.eta, cfg.nudge) == (
        1024,
        12,
        0.5,
        "cross_entropy",
    )
    candidate = run.make_brain("qualified", 0, args)
    cfg = candidate.learner.config
    assert (cfg.free_steps, cfg.nudged_steps, cfg.eta, cfg.nudge, cfg.qualified) == (
        17,
        19,
        0.01,
        "quadratic",
        True,
    )


def test_corrupted_source_receipt_refuses_extraction_before_writing(tmp_path):
    witness = tmp_path / "witness.json"
    verification = tmp_path / "verification.json"
    selection = tmp_path / "selection.json"
    witness.write_text(json.dumps({"source": {"inputs": 650, "window": 12}}))
    verification.write_text(json.dumps({"synced": True, "deaths": 0}))
    selection.write_text(json.dumps({"dataset_sha256": "false", "verification_sha256": "false"}))
    with pytest.raises(ValueError, match="parent fixture hashes"):
        extract.extract(witness, verification, selection, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.fixture(scope="module")
def refused_artifact(tmp_path_factory):
    folder = tmp_path_factory.mktemp("native-school") / "refused"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--out",
            str(folder),
            "--recipe",
            "qualified",
            "--updates",
            "1",
            "--check-every",
            "1",
            "--free-steps",
            "1",
            "--nudged-steps",
            "1",
            "--seconds",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return folder


def test_refused_artifact_replays_with_zero_accepted_exposure(refused_artifact):
    result = verifier.verify(refused_artifact)
    assert result["passed"] and result["source_bound"]
    assert result["cases"][0]["accepted"] == 0
    assert result["cases"][0]["refused"] == 1


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("accepted_phase", "accepted qualified phase"),
        ("exposure", "accepted exposure"),
        ("phase_hash", "raw phase hash"),
        ("checkpoint_hash", "final checkpoint hash"),
        ("removed_case", "completeness"),
        ("summary", "summary outcome"),
        ("protocol_hash", "summary protocol hash"),
        ("work", "work accounting"),
        ("missing_work", "mandatory work field"),
    ],
)
def test_verifier_rejects_false_green_mutations(refused_artifact, tmp_path, mutation, reason):
    folder = tmp_path / "mutated"
    shutil.copytree(refused_artifact, folder)
    case = folder / "qualified-2"
    receipt = json.loads((case / "receipt.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if mutation == "accepted_phase":
        receipt["updates"][0]["report"]["accepted"] = 1
    elif mutation == "exposure":
        receipt["accepted_row_exposures"] += 1
    elif mutation == "work":
        receipt["work"]["phase_row_sweeps"] += 1
    elif mutation == "missing_work":
        del receipt["work"]["phase_row_sweeps"]
    elif mutation in ("phase_hash", "checkpoint_hash"):
        path = case / ("phases-0001.npz" if mutation == "phase_hash" else "final.npz")
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
    elif mutation == "summary":
        summary["outcomes"][0]["stages"][0]["correct"] += 1
    elif mutation == "protocol_hash":
        summary["protocol_sha256"] = "false"
    if mutation == "removed_case":
        shutil.rmtree(case)
    else:
        extract.write_json(case / "receipt.json", receipt)
    extract.write_json(folder / "summary.json", summary)
    with pytest.raises(ValueError, match=reason):
        verifier.verify(folder)


def test_verifier_requires_all_centered_phases_for_an_accepted_lesson(tmp_path):
    folder = tmp_path / "finite"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--out",
            str(folder),
            "--recipe",
            "finite",
            "--updates",
            "1",
            "--check-every",
            "1",
            "--seconds",
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    receipt_path = folder / "finite-2/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    del receipt["updates"][0]["phases"]["opposite"]
    extract.write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="accepted lesson phase census"):
        verifier.verify(folder)
