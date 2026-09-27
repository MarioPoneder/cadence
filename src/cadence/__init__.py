"""Cadence: research toward general intelligence through local repair.

Bounded observer-like patches carry state, expose ports and readback, and repair
shared relationships. ``TemporalPatchNet`` learns observed paths by centered
equilibrium detuning, carries context, imagines privately and repairs continuous
action proposals under the same learned model. ``TemporalMemory`` adds explicit
conditional response protection. Actual outcomes remain distinct from desired
or imagined outcomes.

The research goal is reusable learning, retention, creativity and evolving
functional self-reflection across applications. Current APIs expose tested
operations; they do not establish general intelligence or an automatically
learned recursive hierarchy. See the architecture and interaction guides.

``PatchNet``, ``Brain``, ``Learner``, ``GenericBrain``, ``Records`` and graph
construction/evolution helpers remain supported as distinct compatibility
compositions. Their optional mechanisms are not required by the temporal core.
"""

from __future__ import annotations

from . import regions
from .actor import ActorPlan, ActorReadback, BodyModel, EquilibriumActor, ObservationRecord
from .atlas import Atlas, atlas_of, brain_scan_script, build_atlas
from .belief import BeliefObservation, BeliefPatch, BeliefPath, BeliefReadback
from .brain import Brain, BrainState, Equilibrium, Nudge, available_backends
from .certificate import (
    Certificate,
    EPStructure,
    certificate,
    ep_structure,
    lipschitz_constant,
    row_mass,
)
from .checkpoint import load, save
from .connectome import Connectome
from .content_memory import ContentMemory
from .generic import GenericBrain
from .genome import Genome, Projection, develop, evolve, genes
from .life import (
    AlwaysAwake,
    Decision,
    Governor,
    Life,
    LifeConfig,
    NeverWakes,
    PatchGovernor,
    Signals,
    ThresholdGovernor,
)
from .instruments import dishabituation, orienting
from .learning import (
    Learner,
    LearnerConfig,
    calibrate_bias,
    embedded,
    layered,
    learning_neuron_model,
    naive_efficacy,
    preflight,
    seam_report,
)
from .memory import SynapticMemory
from .neuron import Adaptation, NeuronModel
from .patch import PatchNet, PatchObservation
from .planning import TemporalPlan
from .plasticity import (
    ActorCritic,
    ActorCriticConfig,
    Bins,
    Valence,
)
from .ports import DenseBlock, MapBlock, StructuredPort
from .protocol import Protocol, Row, evaluate_predicate, select_gain, shuffled
from .receipts import Receipt, canonical_json
from .record_patch import (
    RecordContrast,
    RecordObservation,
    RecordPatchNet,
    RecordPath,
    RecordReadback,
)
from .record_ports import JointObservation, JointRecordPatches, Port
from .record_stack import RecordPatchStack, StackObservation
from .recording import SettlementRecord, record_settlements
from .records import Mulberry32, Records
from .reference import conformance
from .regions import Region
from .replay import ReservoirReplay
from .steering import Boundary, Gaze, Rule, Softmax, Steered, SteeredPath
from .stream import Afterglow, Echo, FastSynapses, PatternSeparator, Trace, stateful
from .temporal import TemporalObservation, TemporalPatchNet, TemporalPhase, TemporalReadback
from .temporal_memory import ConstraintReport, TemporalMemory

__all__ = [
    "BeliefObservation",
    "BeliefPatch",
    "BeliefPath",
    "BeliefReadback",
    "Steered",
    "SteeredPath",
    "Life",
    "LifeConfig",
    "Decision",
    "Signals",
    "Governor",
    "PatchGovernor",
    "ThresholdGovernor",
    "AlwaysAwake",
    "NeverWakes",
    "orienting",
    "dishabituation",
    "Boundary",
    "Softmax",
    "Gaze",
    "Rule",
    "DenseBlock",
    "MapBlock",
    "StructuredPort",
    "ActorPlan",
    "ActorReadback",
    "BodyModel",
    "EquilibriumActor",
    "ObservationRecord",
    "ConstraintReport",
    "TemporalMemory",
    "TemporalPlan",
    "RecordPatchNet",
    "RecordPatchStack",
    "StackObservation",
    "RecordPath",
    "RecordObservation",
    "RecordContrast",
    "RecordReadback",
    "TemporalPatchNet",
    "TemporalPhase",
    "TemporalObservation",
    "TemporalReadback",
    "PatchNet",
    "PatchObservation",
    "ContentMemory",
    "ReservoirReplay",
    "Atlas",
    "atlas_of",
    "build_atlas",
    "brain_scan_script",
    "Certificate",
    "EPStructure",
    "certificate",
    "ep_structure",
    "lipschitz_constant",
    "row_mass",
    "PatternSeparator",
    "Records",
    "Mulberry32",
    "ActorCritic",
    "ActorCriticConfig",
    "Valence",
    "Bins",
    "Adaptation",
    "Region",
    "GenericBrain",
    "regions",
    "Projection",
    "Genome",
    "Echo",
    "Afterglow",
    "Trace",
    "FastSynapses",
    "SynapticMemory",
    "NeuronModel",
    "Learner",
    "LearnerConfig",
    "Nudge",
    "Protocol",
    "Receipt",
    "SettlementRecord",
    "record_settlements",
    "JointRecordPatches",
    "JointObservation",
    "Port",
    "Row",
    "Brain",
    "BrainState",
    "Equilibrium",
    "Connectome",
    "available_backends",
    "canonical_json",
    "conformance",
    "embedded",
    "develop",
    "evolve",
    "genes",
    "evaluate_predicate",
    "layered",
    "learning_neuron_model",
    "calibrate_bias",
    "naive_efficacy",
    "preflight",
    "seam_report",
    "load",
    "save",
    "select_gain",
    "shuffled",
    "stateful",
]

__version__ = "0.16.0"
