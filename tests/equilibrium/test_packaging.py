"""Check installed population-model identity and source-bound continuation."""

import hashlib
from importlib.resources import files

import cadence
import cadence.experimental.equilibrium as equilibrium
from cadence.experimental.equilibrium.brain import IMPLEMENTATION


def test_package_version_and_resource_identity():
    assert equilibrium.__version__ == cadence.__version__
    assert equilibrium.Brain.__module__ == "cadence.experimental.equilibrium.brain"
    assert IMPLEMENTATION == {
        name: hashlib.sha256(files(equilibrium).joinpath(name).read_bytes()).hexdigest()
        for name in IMPLEMENTATION
    }
