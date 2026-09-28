"""Cadence: brains built from one element, the cortical column.

``CorticalColumn(height=H)`` is the element with a vertical dimension -
H observer stages on one belief; height 1 is the qualified candidate
exactly (``cadence.element`` holds that frozen source). ``Cortex`` is a
trainable hierarchy of column banks wired by calibration; its structure
is depth x width x height. Quickstart in the README; API and parameters
in docs/REFERENCE.md; variants in docs/VARIANTS.md.
"""
from .column import CorticalColumn
from .cortex import Cortex
from .element import settle, solve_cluster_forest
from .wiring import calibrate, wire

__version__ = '0.20.0'
__all__ = ['CorticalColumn', 'Cortex', 'calibrate', 'wire',
           'settle', 'solve_cluster_forest', '__version__']
