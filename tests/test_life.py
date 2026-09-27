"""The life: the governors' modes on the brain's signals, the loop on a small world, the learn
call from the window's boundary with its rollback, the cost, and a steered cortex as the brain."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import BeliefPatch, DenseBlock, Softmax, Steered, StructuredPort
from cadence.genome import genes
from cadence.life import (
    MODES,
    READBACK,
    AlwaysAwake,
    Life,
    LifeConfig,
    NeverWakes,
    PatchGovernor,
    Signals,
    ThresholdGovernor,
)


def _signals(fast=0.0, slow=0.0, residual=1.0, mode="habit", surprise=0.0, baseline=1.0, recent=(), can_learn=True):
    one_hot = np.eye(3)[MODES.index(mode)]
    readback = np.array([fast, slow, residual, *one_hot, 1.0])
    return Signals(readback, surprise, baseline, residual, slow, mode, list(recent), can_learn)


def test_the_patch_governor_settles_to_the_mode_the_hand_set_wiring_says():
    governor = PatchGovernor()
    assert governor.n == len(READBACK) + 4 + 3 and governor.synapses > 0
    quiet, steps = governor.settle(_signals())
    assert quiet == "habit" and steps > 0
    spike, _ = governor.settle(_signals(fast=np.log1p(8.0)))  # a spike of eight baselines
    assert spike == "imagine"
    persistent, _ = governor.settle(_signals(slow=1.4))  # a changed law's slow average
    assert persistent == "learn"
    space = PatchGovernor.space()
    assert set(space) == set(PatchGovernor.hand_set()) and "rm_0_1" in space
    mutate = genes(space, rate=0.3)
    child = mutate(PatchGovernor.hand_set(), np.random.default_rng(0))
    assert set(child) == set(space)
    PatchGovernor(child).settle(_signals())  # every genome grows into a brain that settles
    warm = PatchGovernor({"warm": True})
    warm.settle(_signals(fast=2.0))
    assert warm.state is not None
    warm.reset()
    assert warm.state is None


def test_the_threshold_governor_and_the_two_ends_of_the_switch():
    g = ThresholdGovernor({"k_imagine": 4.0, "imagine_budget": 3, "k_learn": 2.0, "persist": 4, "persist_share": 0.5})
    assert g.settle(_signals())[0] == "habit"
    g.after(_signals(surprise=5.0, baseline=1.0))  # a spike above four baselines
    assert [g.settle(_signals())[0] for _ in range(4)] == ["imagine", "imagine", "imagine", "habit"]
    hot = _signals(recent=[3.0, 3.0, 0.1, 3.0], baseline=1.0)
    assert g.settle(hot)[0] == "learn"
    assert g.settle(_signals(recent=[3.0, 3.0, 0.1, 3.0], can_learn=False))[0] == "habit"
    assert AlwaysAwake({"persist": 4, "k_learn": 2.0}).settle(_signals())[0] == "imagine"
    assert AlwaysAwake({"persist": 4, "k_learn": 2.0, "persist_share": 0.5}).settle(hot)[0] == "learn"
    assert NeverWakes().settle(hot) == ("habit", 0)


class _World:
    """A dot that drifts and a paw the action moves; the law flips at ``change``."""

    def __init__(self, seed=0, change=10**9):
        self.rng = np.random.default_rng(seed)
        self.x, self.p, self.t, self.change = 0.5, 0.5, 0, change

    def reading(self):
        return np.array([self.x, self.p])

    def step(self, a):
        self.t += 1
        drift = 0.01 if self.t < self.change else -0.03
        self.x = float(np.clip(self.x + drift + self.rng.normal(0, 0.001), 0, 1))
        self.p = float(np.clip(self.p + 0.1 * float(a[0]), 0, 1))
        return self.reading()


def _patch(seed=0):
    return BeliefPatch(StructuredPort(2, [DenseBlock(0, 2, 6)]), actions=1, belief=8, outputs=2, cells=32, active=4, record_width=4, seed=seed)


def _life(patch, governor, **cfg):
    config = LifeConfig(**{"window": 16, "min_window": 8, "min_cooldown": 4, "passes": 3, "learn_rate": 2.0, "horizon": 3, **cfg})
    return Life(
        patch,
        governor,
        habit=lambda r: np.array([0.5 * (0.5 - r[1])]),
        propose=lambda r: np.array([[-1.0], [0.0], [1.0]]),
        advance=lambda r, y: r + y,
        cost=lambda R, A: (R[:, 0] - R[:, 1]) ** 2 + 0.01 * A[:, 0] ** 2,
        target=lambda r, r2: r2 - r,
        baseline=1e-4,
        residual0=0.1,
        config=config,
    )


def test_the_loop_learns_from_the_window_and_counts_its_cost():
    world = _World(1, change=40)
    life = _life(_patch(1), ThresholdGovernor({"k_imagine": 3.0, "imagine_budget": 2, "k_learn": 1.5, "persist": 6, "persist_share": 0.5, "cooldown": 0}))
    r = world.reading()
    for _ in range(90):
        r, d = life.step(r, world.step)
    modes = life.totals["decisions"]
    assert sum(modes.values()) == 90 and modes["imagine"] > 0 and modes["learn"] > 0
    assert life.learns and all(e["window"] <= 16 for e in life.learns)
    kept = [e for e in life.learns if e["kept"]]
    assert kept and all(e["loss_after"] < 0.9 * e["loss_before"] for e in kept)
    c = life.compute()
    assert c["learn_calls"] == len(life.learns) == c["kept"] + c["undone"]
    assert c["moments"] >= 90 and c["moments_per_decision"] > 1.0
    assert life.records[-1].t == 89 and life.records[-1].surprise is not None
    assert len(life.o) == len(life.a) == len(life.y) == len(life.boundaries) == 90


def test_an_invalid_learn_call_is_undone_and_habituates_the_baseline():
    world = _World(2, change=1)
    patch = _patch(2)
    life = _life(patch, ThresholdGovernor({"k_learn": 0.0, "persist": 8, "persist_share": 0.0, "cooldown": 0}), validity=0.0)  # no window can pass a validity of zero
    birth = patch.parameters()
    r = world.reading()
    for _ in range(60):
        r, d = life.step(r, world.step)
    assert life.totals["learn_calls"] >= 1
    assert life.totals["undone"] == life.totals["learn_calls"] and life.totals["kept"] == 0
    assert all(not e["valid"] and not e["kept"] and e["loss_before"] is not None for e in life.learns)
    assert all(np.array_equal(birth[k], patch.parameters()[k]) for k in birth)  # every step rolled back
    assert life.baseline >= life.floor > 0
    with pytest.raises(RuntimeError, match="outcome"):
        life.decide(r)
        life.decide(r)


def test_a_steered_cortex_lives_and_learns_under_a_governor():
    cortex = BeliefPatch(StructuredPort(2, [DenseBlock(0, 1, 4), DenseBlock(1, 1, 4)]), actions=1, belief=8, outputs=2, cells=32, active=4, record_width=4, seed=3)
    steering = BeliefPatch(StructuredPort(5, [DenseBlock(0, 5, 4)]), actions=1, belief=4, outputs=2, cells=16, active=2, record_width=4, seed=4)
    brain = Steered(cortex, steering, Softmax(2, span=1.5))
    world = _World(3, change=30)
    life = _life(brain, AlwaysAwake({"k_learn": 1.0, "persist": 6, "persist_share": 0.5, "cooldown": 0}))
    r = world.reading()
    for _ in range(60):
        r, d = life.step(r, world.step)
    assert life.totals["decisions"]["imagine"] + life.totals["decisions"]["learn"] == 60
    assert life.learns and brain.boundary() is not None and brain.boundary().moments > 0
    assert life.compute()["moments"] > 60  # the steering patch's moments count beside the cortex's
    with pytest.raises(ValueError, match="BeliefPatch or a Steered"):
        Life(object(), NeverWakes(), habit=lambda r: r, propose=lambda r: r, advance=lambda r, y: r, cost=lambda R, A: R[:, 0], target=lambda r, r2: r2)
