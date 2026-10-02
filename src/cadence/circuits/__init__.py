"""Circuits: a reflex arc, bounded futures and a self-reading monitor.

These use the graded neuron model. They are engineered patterns, not
claims of biological universality, autonomous world-model learning or consciousness.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from ..brain import Brain, BrainState
from ..connectome import Connectome
from ..neuron import NeuronModel
from .assembly import assemble
from .deliberation import Deliberation, Deliberator, Future, imagine


def reflex_arc(axes: int = 2) -> Connectome:
    """Sensory error ports followed by opposing motor pairs for each axis.

    A body reads ``max(0, positive) - max(0, negative)``. Body physics and sensor
    encodings are supplied by the application. Continue state to retain dynamics.
    """
    if isinstance(axes, bool) or not isinstance(axes, int) or axes < 1:
        raise ValueError("axes must be a positive integer")
    return Connectome.from_synapses(
        3 * axes,
        pre=[i for i in range(axes) for _ in range(2)],
        post=[axes + i for i in range(2 * axes)],
        sign=[v for _ in range(axes) for v in (2.0, -2.0)],
        populations={
            "sensory": range(axes),
            "motor": range(axes, 3 * axes),
            "positive": range(axes, 3 * axes, 2),
            "negative": range(axes + 1, 3 * axes, 2),
        },
        label="sensory-motor",
    )


@dataclass(frozen=True)
class Readback:
    activity_change: float
    ambiguity: float
    pressure: float
    uncertainty: float
    request_more: bool
    state: BrainState


class ActivityMonitor:
    """Read the controller's activity and alternatives, then gate more deliberation.

    This six-neuron monitor observes normalized activity change, option ambiguity,
    budget pressure and their integration. Its readout can request more work;
    an application must connect that output to its actual compute budget. The
    signal is a heuristic, not a calibrated probability or a consciousness test.
    """

    def __init__(self) -> None:
        self.brain = Brain(
            Connectome.from_synapses(
                6,
                pre=[0, 1, 2, 3, 4, 2],
                post=[4, 4, 4, 4, 5, 5],
                sign=[0.6, 0.8, -0.4, 0.4, 1.0, -1.0],
                populations={"readback": range(4), "integration": [4], "request": [5]},
                label="activity-monitor",
            ),
            NeuronModel(gain=1, slope=2, threshold=0, leak=1, dt=0.25, stimulus_amplitude=1),
        )
        self.state: BrainState | None = None
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        """Start monitoring an independent episode without a previous activity comparison."""
        self.state, self.previous = None, None

    def read(self, activity: np.ndarray, scores: np.ndarray, *, pressure: float = 0) -> Readback:
        activity, scores = np.asarray(activity, float), np.asarray(scores, float)
        if (
            activity.ndim != 1
            or not activity.size
            or scores.ndim != 1
            or not scores.size
            or not np.isfinite(activity).all()
            or not np.isfinite(scores).all()
            or not isfinite(pressure)
            or not 0 <= pressure <= 1
        ):
            raise ValueError("finite vectors and pressure in [0, 1] required")
        activity_change = (
            0.0
            if self.previous is None or self.previous.shape != activity.shape
            else float(
                np.max(np.abs(activity - self.previous)) / max(1.0, float(np.max(np.abs(activity))))
            )
        )
        ordered = np.sort(scores)
        gap = float(ordered[-1] - ordered[-2]) if scores.size > 1 else float("inf")
        ambiguity = 1.0 / (1.0 + gap)
        drive = np.array(
            [min(1, activity_change), ambiguity, pressure, float(scores.size > 1), 0, 0]
        )
        self.state = self.brain.settle(drive, state=self.state, steps=40, tolerance=0)
        self.previous = activity.copy()
        uncertainty = float(self.state.activation[4])
        return Readback(
            activity_change,
            ambiguity,
            pressure,
            uncertainty,
            bool(self.state.activation[5] > 0.35 and pressure < 1),
            self.state,
        )


__all__ = [
    "ActivityMonitor",
    "Deliberation",
    "Deliberator",
    "Future",
    "Readback",
    "assemble",
    "imagine",
    "reflex_arc",
]
