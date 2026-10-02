"""Replay-order controls preserve evidence and per-record exposure."""

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[2] / "examples" / "equilibrium" / "credit_exposure.py"
SPEC = importlib.util.spec_from_file_location("credit_exposure", PATH)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def collection():
    owner = experiment.diagnostic.make(2, 1)
    records, outcomes, work, status = experiment.diagnostic.collect(owner, 1, 0)
    assert status == "complete"
    return dict(
        protocol=dict(
            seed=2,
            delay=1,
            preferred=0,
            episodes=24,
            sources={
                "examples/equilibrium/credit_diagnostic.py": experiment.diagnostic.digest(
                    Path(experiment.diagnostic.__file__)
                ),
            },
        ),
        records=records,
        outcomes=outcomes,
        collection_work=work,
    )


def test_reverse_order_matches_every_record_multiplicity_at_block_boundaries():
    data = collection()
    rows = experiment.diagnostic.schedule(data["records"], 99, updates=128)
    reversed_rows = experiment.reverse_blocks(rows, data["records"])
    assert rows != reversed_rows
    assert all(experiment.matched_prefixes(rows, reversed_rows, [32, 64, 96, 128]).values())
    for start in range(0, 128, 32):
        flat = [index for batch in reversed_rows[start : start + 32] for index in batch]
        stages = [data["records"][index]["stage"] for index in flat]
        assert stages == sorted(stages, reverse=True)
        assert Counter(flat) == Counter(
            index for batch in rows[start : start + 32] for index in batch
        )
    assert not experiment.matched_prefixes(rows[:31], reversed_rows, [32])["32"]


def test_partial_or_unequal_batches_are_rejected():
    with pytest.raises(ValueError):
        experiment.reverse_blocks([[0] * 16] * 31, [{"stage": 0}])
    with pytest.raises(ValueError):
        experiment.reverse_blocks([[0] * 15] * 32, [{"stage": 0}])


@pytest.mark.parametrize("mutation", ("action", "reward", "outcome"))
def test_reconstruction_rejects_changed_execution_or_reward(mutation):
    data = collection()
    if mutation == "outcome":
        data["outcomes"][0]["reward"] *= -1
    elif mutation == "action":
        data["records"][0]["action"] = 1 - data["records"][0]["action"]
    else:
        data["records"][-1]["reward"] *= -1
    with pytest.raises(ValueError, match="differ"):
        experiment.reconstruct(data)


def test_return_calibration_is_pure_and_never_teaches_the_diagnostic_targets():
    data = collection()
    owner, _ = experiment.reconstruct(data)
    before = owner.snapshot()
    result = experiment.calibration(owner, data["records"])
    assert result["qualified"] and result["query_pure"]
    assert all(
        row["absolute_error"] == abs(row["value"] - row["observed_return"])
        for row in result["rows"]
    )
    assert owner.snapshot() == before


def test_bounded_runner_preserves_matched_exposure_and_only_one_step_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "UPDATES", 4)
    monkeypatch.setattr(experiment, "BLOCK", 2)
    monkeypatch.setattr(experiment, "CHECKPOINTS", (0, 2, 4))
    source = tmp_path / "collection.json"
    source.write_text(json.dumps(collection()))
    destination = tmp_path / "run"
    report = experiment.run(SimpleNamespace(collection=source, out=destination))
    assert report["status"] == "complete", report.get("error")
    assert report["collection_verified"] and report["sources_unchanged"]
    assert all(report["actual_matched_prefixes"].values())
    assert report["protocol"]["collection_sha256"] == experiment.diagnostic.digest(source)
    for arm in report["arms"]:
        assert [check["update"] for check in arm["checkpoints"]] == [0, 2, 4]
        for row in arm["updates"]:
            assert row["source"] == "estimate" and row["accepted"]
            assert all(horizon == 1 for horizon in row["horizons"])
        assert arm["total_work"]["evaluations"] > arm["training_work"]["evaluations"]


def test_changed_source_is_recorded_as_failure_before_training(tmp_path):
    data = collection()
    data["protocol"]["sources"]["examples/equilibrium/credit_diagnostic.py"] = "changed"
    source = tmp_path / "collection.json"
    source.write_text(json.dumps(data))
    report = experiment.run(SimpleNamespace(collection=source, out=tmp_path / "run"))
    assert report["status"] == "error" and report["arms"] == []
    assert "identity is incompatible" in report["error"]
