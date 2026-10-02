"""Keep the preserved engine identity separate from the default API family."""

import hashlib
from importlib.resources import files
from pathlib import Path

import cadence.experimental.equilibrium as equilibrium
from cadence.experimental.equilibrium.brain import IMPLEMENTATION


def test_preserved_engine_version_and_resource_identity():
    assert equilibrium.__version__ == "0.62.0"
    assert equilibrium.Brain.__module__ == "cadence.experimental.equilibrium.brain"
    assert IMPLEMENTATION == {
        name: hashlib.sha256(files(equilibrium).joinpath(name).read_bytes()).hexdigest()
        for name in IMPLEMENTATION
    }


def test_documented_namespace_and_engine_identity():
    docs = Path(__file__).resolve().parents[2] / "docs" / "equilibrium"
    for name in ("README.md", "QUICKSTART.md", "EXPERIMENTAL.md"):
        text = (docs / name).read_text()
        assert f"`{equilibrium.__version__}`" in text, name
    assert "cadence.experimental.equilibrium" in (docs / "README.md").read_text()
