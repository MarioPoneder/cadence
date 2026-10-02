"""Opt-in research interfaces; no experimental feature changes core defaults."""

from .partitioned import PartitionedTemporalPatchNet, two_group_masks

__all__ = ["PartitionedTemporalPatchNet", "two_group_masks"]
