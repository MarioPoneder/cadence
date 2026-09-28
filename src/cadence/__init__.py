"""Cadence: brains built from one element, the cortical column.

The element (``CorticalColumn``) is a bounded self-reading settling
patch with one repair law; a ``Cortex`` is a trainable hierarchy of its
value columns, wired by calibration. Quickstart in the README; API and
parameters in docs/REFERENCE.md; variants in docs/VARIANTS.md.
"""
from .element import CorticalColumn, settle, solve_cluster_forest
from .cortex import Cortex
from .wiring import calibrate, wire

__version__ = '0.20.0'
__all__ = ['CorticalColumn', 'Cortex', 'calibrate', 'wire',
           'settle', 'solve_cluster_forest', '__version__']
