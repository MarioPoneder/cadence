"""Generic bounded feature wiring and environment calibration contracts."""

from __future__ import annotations

import math

import pytest

from cadence import Cortex
from cadence.wiring import BinnedFeatures, calibrate, grid, wire


def test_real_features_boundaries_clipping_and_declared_cells():
    mapping = BinnedFeatures(((-1, 1), (0, 10)), bins=(4, 2))
    assert mapping((-1, 0)) == (0, 0)
    assert mapping((0, 5)) == (2, 1)
    assert mapping((1, 10)) == (3, 1)
    assert mapping((-4, 20)) == (0, 1)
    assert mapping.cells == 8


def test_selected_indices_preserve_order_and_ignore_unselected_values():
    mapping = BinnedFeatures(((-1, 1), (0, 1)), bins=(2, 4), indices=(2, 0))
    assert mapping((0.75, 3.0, -0.75)) == (0, 3)
    assert mapping((0.75, 9.0, -0.75)) == (0, 3)
    assert mapping.cells == 8


@pytest.mark.parametrize(
    "observation", [(), (0,), (math.nan, 1), (0, math.inf), ("0", 1), (True, 1)]
)
def test_features_reject_malformed_nonfinite_observations(observation):
    mapping = BinnedFeatures(((0, 1), (0, 1)))
    with pytest.raises((TypeError, ValueError)):
        mapping(observation)


def test_out_of_domain_can_be_rejected_instead_of_clipped():
    mapping = BinnedFeatures(((0, 1),), bins=4, clip=False)
    with pytest.raises(ValueError):
        mapping((1.01,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": ()},
        {"bounds": ((1, 0),)},
        {"bounds": ((math.nan, 1),)},
        {"bounds": ((0, math.inf),)},
        {"bounds": ((False, 1),)},
        {"bounds": ((0, 1, 2),)},
        {"bounds": ((0, 1),), "bins": True},
        {"bounds": ((0, 1),), "bins": 0},
        {"bounds": ((0, 1),), "bins": 2.5},
        {"bounds": ((0, 1),), "bins": 10**400},
        {"bounds": ((0, 1),), "bins": (2, 3)},
        {"bounds": ((0, 1),), "indices": (-1,)},
        {"bounds": ((0, 1),), "indices": (True,)},
        {"bounds": ((0, 1),), "indices": (0, 0)},
        {"bounds": ((0, 1),), "clip": "false"},
    ],
)
def test_features_validate_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        BinnedFeatures(**kwargs)


def test_feature_configuration_is_detached_from_mutable_arguments():
    bounds = [[0, 1], [-1, 1]]
    bins, indices = [2, 4], [0, 1]
    mapping = BinnedFeatures(bounds, bins=bins, indices=indices)
    expected = mapping((0.25, 0.25))
    bounds[0][1], bins[0], indices[0] = 100, 100, 1
    assert mapping((0.25, 0.25)) == expected
    assert mapping.cells == 8


@pytest.mark.parametrize("field", ["bounds", "bins", "indices", "clip", "cells"])
def test_builtin_feature_configuration_is_immutable(field):
    mapping = BinnedFeatures(((0, 1),), bins=4)
    with pytest.raises(AttributeError):
        setattr(mapping, field, getattr(mapping, field))


def test_grid_is_serializable_generic_fine_to_coarse_wiring():
    maps = grid(((-1, 1), (-1, 1)), bins=4, depth=2)
    assert maps[0]((-0.75, 0.75)) != maps[0]((0.75, -0.75))
    assert maps[-1]((-0.75, 0.75)) == maps[-1]((0.75, -0.75)) == ()
    assert maps[0].cells > maps[-1].cells == 1
    net = Cortex(n_outputs=2, feature_maps=maps)
    net.observe((0.25, -0.25), [1.0, -1.0])
    twin = Cortex.from_snapshot(net.snapshot())
    assert twin.predict((0.25, -0.25)) == net.predict((0.25, -0.25))


@pytest.mark.parametrize("depth", [True, -1, 1.5])
def test_grid_rejects_noninteger_or_negative_depth(depth):
    with pytest.raises((TypeError, ValueError)):
        grid(((0, 1),), depth=depth)


class VectorEnv:
    class ActionSpace:
        n = 2

    action_space = ActionSpace()

    def __init__(self, *, failure=None, seed_only=False):
        self.closed = False
        self.failure = failure
        self.seed_only = seed_only
        self.steps = 0
        self.seed = 0

    def reset(self, seed=0):
        self.seed = seed
        self.steps = 0
        if self.failure == "reset":
            raise RuntimeError("reset failure")
        return (float(seed), -0.5), {}

    def step(self, action):
        self.steps += 1
        if self.failure == "step":
            raise RuntimeError("step failure")
        if self.failure == "shape":
            return (0.0,), 0.0, False, False, {}
        if self.failure == "nonfinite":
            return (math.nan, 0.0), 0.0, False, False, {}
        x = float(self.seed) if self.seed_only else 0.125 * (action + 1) * self.steps
        return (x, -0.5 + 0.125 * (self.steps % 3)), 0.0, False, False, {}

    def close(self):
        self.closed = True


def factory_recording(**kwargs):
    envs = []

    def factory():
        env = VectorEnv(**kwargs)
        envs.append(env)
        return env

    return factory, envs


def test_low_dimensional_continuous_calibration_wires_and_closes():
    factory, envs = factory_recording()
    calibration = calibrate(factory, 2, steps=4, seed=0)
    maps = wire(calibration)
    assert maps[0]((0.125, -0.5)) != maps[0]((1.0, -0.25))
    assert all(env.closed for env in envs)
    net = Cortex.for_environment(factory, calibration_steps=4)
    assert len(net.predict((0.5, -0.25))) == 2
    assert all(env.closed for env in envs)


def test_calibration_pairs_initial_seed_across_forced_actions():
    factory, envs = factory_recording(seed_only=True)
    calibration = calibrate(factory, 2, steps=4, seed=37)
    assert [env.seed for env in envs] == [37, 37]
    assert all(score == 0.0 for _, score in calibration["controllability_rank"])


@pytest.mark.parametrize("failure", ["reset", "step", "shape", "nonfinite"])
def test_calibration_closes_environments_on_invalid_data_and_exceptions(failure):
    factory, envs = factory_recording(failure=failure)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        calibrate(factory, 2, steps=2)
    assert envs and all(env.closed for env in envs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_actions": True},
        {"n_actions": 0},
        {"n_actions": 1.5},
        {"steps": True},
        {"steps": 0},
        {"steps": -1},
        {"steps": 1.5},
        {"seed": True},
    ],
)
def test_calibration_validates_before_creating_environment(kwargs):
    factory, envs = factory_recording()
    params = {"n_actions": 2, "steps": 2}
    params.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        calibrate(factory, **params)
    assert envs == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth": True},
        {"depth": 2.0},
        {"depth": -1},
        {"width": True},
        {"width": 0},
        {"width": 1.5},
    ],
)
def test_wire_rejects_coercive_structure_parameters(kwargs):
    calibration = calibrate(VectorEnv, 2, steps=2)
    with pytest.raises((TypeError, ValueError)):
        wire(calibration, **kwargs)
