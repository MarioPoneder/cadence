"""Cortex contracts: constructor, wiring, custody, and that it learns."""

import math
import random

import pytest

from cadence import Cortex, wire


class Corridor:
    """A 12-cell corridor: reach the top for +1; a distractor byte wiggles."""

    N, ACTIONS = 12, 3

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.position = self.distractor = 0

    def reset(self):
        self.position, self.distractor = 0, self.rng.randrange(8)
        return (self.position, self.distractor)

    def step(self, action):
        if action == 1:
            self.position = min(self.N - 1, self.position + 1)
        elif action == 2:
            self.position = max(0, self.position - 1)
        self.distractor = self.rng.randrange(8)
        reward = 0.0
        if self.position == self.N - 1:
            reward, self.position = 1.0, 0
        return (self.position, self.distractor), reward


def corridor_maps():
    return (lambda o: (o[0], o[1]), lambda o: (o[0] // 3,), lambda o: ())


def run_corridor(cortex, episodes, steps=48, seed=0):
    world = Corridor(seed)
    returns = []
    for _ in range(episodes):
        observation, total = world.reset(), 0.0
        for _ in range(steps):
            action = cortex.act(observation)
            next_observation, reward = world.step(action)
            cortex.learn(observation, action, reward, next_observation, False)
            observation, total = next_observation, total + reward
        cortex.flush()
        returns.append(total)
    return returns


class FakeRamEnv:
    """Minimal reset/step RAM environment: byte 7 follows the action."""

    class ActionSpace:
        n = 3

    action_space = ActionSpace()

    def __init__(self):
        self.state, self.rng = [0] * 128, random.Random(0)

    def reset(self, seed=0):
        self.rng = random.Random(seed)
        self.state = [0] * 128
        return list(self.state), {}

    def step(self, action):
        self.state[7] = min(255, max(0, self.state[7] + (10 if action == 1 else -10)))
        self.state[30] = self.rng.randrange(256)
        return list(self.state), 0.0, False, False, {}

    def close(self):
        pass


def test_constructor_validates():
    with pytest.raises(ValueError):
        Cortex(0, corridor_maps())
    with pytest.raises(ValueError):
        Cortex(3, ())
    with pytest.raises(ValueError):
        Cortex(3, corridor_maps(), wiring_id="corridor-v1", discount=1.0)


def test_wiring_depth_and_width():
    cortex = Cortex.for_environment(FakeRamEnv, depth=2, width=1, calibration_steps=60)
    assert cortex.n_outputs == 3 and cortex.levels == 3
    assert cortex.calibration["controllability_rank"][0][0] == 7
    assert cortex.level_weights[0] == 1.0 and cortex.level_weights[-1] < 0.01
    wide = wire(cortex.calibration, depth=3, width=2)
    assert len(wide) == 4 and wide[0].cells == 16 * 8 * 8 * 4
    with pytest.raises(ValueError):
        wire(cortex.calibration, depth=33)
    with pytest.raises(ValueError):
        wire(cortex.calibration, depth=2, width=0)


def test_interface_and_custody():
    cortex = Cortex(3, corridor_maps(), wiring_id="corridor-v1", seed=2)
    run_corridor(cortex, episodes=3, seed=2)
    observation = (4, 2)
    belief = cortex.value(observation, 1)
    assert belief["qualified"] is True and math.isfinite(belief["mean"])
    assert belief["novelty"] > 0 and belief["residual"] <= 1e-11
    saved = cortex.snapshot()
    twin = Cortex(3, corridor_maps(), wiring_id="corridor-v1", seed=2)
    twin.restore(saved)
    assert twin.snapshot() == saved
    for action in range(3):
        assert twin.value(observation, action) == cortex.value(observation, action)
    with pytest.raises(ValueError):
        twin.restore(saved.replace("cortex-state/1", "other/9"))


def test_corridor_learning_beats_controls():
    learner = Cortex(3, corridor_maps(), wiring_id="corridor-v1", seed=3)
    frozen = Cortex(
        3, corridor_maps(), wiring_id="corridor-v1", seed=3, learning_enabled=False
    )
    uniform = Cortex(
        3,
        corridor_maps(),
        wiring_id="corridor-v1",
        seed=3,
        learning_enabled=False,
        epsilon=1.0,
    )
    learned = run_corridor(learner, episodes=60, seed=3)
    frozen_returns = run_corridor(frozen, episodes=10, seed=3)
    uniform_returns = run_corridor(uniform, episodes=10, seed=3)
    late = sum(learned[-10:]) / 10
    assert late >= 2.0, f"learned late-mean {late}"
    assert late >= 4 * max(sum(frozen_returns) / 10, 0.25)
    assert late >= 4 * max(sum(uniform_returns) / 10, 0.25)
    assert learner.counters["rejected_updates"] == 0


def test_height_two_cortex_runs_and_binds_custody():
    # Mechanism and custody only: learning above height 1 is not yet a
    # contract (corridor: 1 of 3 seeds learned at the declared constants;
    # docs/VARIANTS.md records that as an open result, not a promise).
    tall = Cortex(3, corridor_maps(), wiring_id="corridor-v1", seed=7, height=2)
    run_corridor(tall, episodes=8, seed=7)
    belief = tall.value((4, 2), 1)
    assert belief["qualified"] is True and belief["novelty"] > 0
    assert tall.counters["rejected_updates"] == 0
    saved = tall.snapshot()
    with pytest.raises(ValueError):
        Cortex(3, corridor_maps(), wiring_id="corridor-v1", seed=7, height=1).restore(
            saved
        )
    twin = Cortex(3, corridor_maps(), wiring_id="corridor-v1", seed=7, height=2)
    twin.restore(saved)
    assert twin.snapshot() == saved
    for action in range(3):
        assert twin.value((4, 2), action) == tall.value((4, 2), action)


def test_hierarchy_helps_early_generalization():
    deep = Cortex(3, corridor_maps(), wiring_id="corridor-v1", seed=4)
    flat = Cortex(3, corridor_maps()[:1], wiring_id="corridor-flat-v1", seed=4)
    deep_returns = run_corridor(deep, episodes=25, seed=4)
    flat_returns = run_corridor(flat, episodes=25, seed=4)
    assert sum(deep_returns) >= sum(flat_returns)
