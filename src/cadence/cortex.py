"""Cortex: a trainable hierarchy of cortical columns.

A Cortex holds one value column per (level, context, action). Levels are
ordered fine to coarse; a coarser level's settled belief is the prior of
the finer one, every level runs the element's observer loop on its own
uncertainty, and every readout and every admission goes through the
element's one repair law (``element.settle``). Configuration mirrors a
small torch model: structure comes from the wiring (``depth``, ``width``
or explicit feature maps), dynamics from constructor keywords.

Quickstart::

    from cadence import Cortex

    cortex = Cortex.for_environment(env_factory, depth=2, width=1)
    action = cortex.act(observation)
    cortex.learn(observation, action, reward, next_observation, terminated)
    cortex.end_episode()   # at every episode end

Four structural rules are code, not tuning knobs; each was locked in
after a measured failure (docs/VARIANTS.md has the receipts story):

1. Bootstraps are grounded in means only; the novelty bonus enters a
   target once, scaled by (1 - discount). A persistent unscaled bonus
   has bootstrap fixed point bonus/(1-discount): a reward-free,
   self-sustaining value web.
2. A coarse prior's transferable precision is floored by its between-
   target dispersion (scatter/weight); otherwise the global column's
   certainty about the global average flattens every region onto it.
3. Episodes are admitted backward at episode end, each transition
   exactly once, so one witnessed reward welds the whole value chain
   without recounting any event as fresh evidence.
4. A level sharing one column across k finest contexts admits each
   target at witness mass 1/k and tempers by decay**(1/k), so every
   level integrates the same evidence mass per unit of experience.
"""
from __future__ import annotations

import json
import math
import random
from typing import Any, Callable, Sequence

from . import element as el


class LevelBelief:
    """One level's belief; the coarser level's settled belief is its prior.

    Without a prior port this is exactly the element's ``LowerBelief``
    formula. With one, the level fuses its own evidence with the prior
    by precision: evidence precision is weight times the level's live
    observer precision; prior precision is the coupling over the prior's
    settled variance floored by its between-target dispersion. A fresh
    level inherits the coarser belief whole; a well-evidenced level
    overrides it.
    """

    def __init__(self, weight: float, mean: float, level: int, coupling: float,
                 prior_dispersion: float = 0.0):
        self.weight, self.mean, self.level, self.coupling = weight, mean, level, coupling
        self.prior_dispersion = prior_dispersion

    def emit(self, port, inbox):
        tau = inbox[f'feedback{self.level}']
        own_precision = self.weight * tau
        prior = inbox.get(f'prior{self.level}')
        if prior is None or self.coupling == 0.0:
            return (self.mean, 1.0 / own_precision)
        prior_mean, prior_variance = prior
        prior_precision = self.coupling / max(prior_variance + self.prior_dispersion, 1e-9)
        total = own_precision + prior_precision
        mean = (own_precision * self.mean + prior_precision * prior_mean) / total
        return (mean, 1.0 / total)


class LevelObserver(el.PrecisionObserver):
    """The element's precision observer, addressed by level port names."""

    def __init__(self, weight: float, scatter: float, level: int):
        super().__init__(weight, scatter)
        self.level = level

    def emit(self, port, inbox):
        return super().emit(port, {'readback': inbox[f'readback{self.level}']})


class ValueColumn:
    """Retained evidence for one (level, context, action); float statistics."""

    __slots__ = ('weight', 'linear', 'square', 'version')

    def __init__(self):
        self.weight, self.linear, self.square = float(el.PRIOR_WEIGHT), 0.0, 0.0
        self.version = 0

    def stats(self):
        return (self.weight, self.linear, self.square)

    def admit(self, target: float, decay: float, witness_weight: float = 1.0):
        """Tempered admission at declared witness mass (structural rule 4)."""
        return (el.EvidenceFactor(self.weight, self.linear, self.square)
                .temper(decay ** witness_weight)
                .combine(el.EvidenceFactor(witness_weight,
                                           witness_weight * target,
                                           witness_weight * target * target)))

    def commit(self, prospective):
        self.weight, self.linear, self.square = (prospective.weight,
                                                 prospective.linear,
                                                 prospective.square)
        self.version += 1


class Cortex:
    """Hierarchical value learner built from the cortical-column element.

    Two arguments are required; every other parameter is a documented
    default (docs/REFERENCE.md). ``for_environment`` removes even the two.

    Required:

    - ``n_actions``: how many discrete actions the body offers.
    - ``feature_maps``: context functions, finest first; each maps an
      observation to a hashable tuple, the last usually the constant
      context ``lambda o: ()`` (a per-action global prior). Build them
      with ``cadence.wire(calibration, depth, width)`` for RAM
      observations, or write your own. A map may declare ``cells`` (its
      context count) to enable witness-mass weighting; ``wire`` does.

    Dynamics (declared defaults; change with a reason):

    - ``decay`` (0.99): column leak per admitted unit witness mass; the
      effective evidence window is 1/(1-decay) targets. 1.0 never
      forgets and stops tracking context switches.
    - ``discount`` (0.97): TD horizon, 1/(1-discount) steps.
    - ``optimism`` (0.5): novelty weight. Acting scores
      mean + optimism*sqrt(novelty); targets add the same bonus scaled
      by (1-discount). Too high masks value gradients with novelty
      noise; 0 never seeks unrewarded regions.
    - ``epsilon`` (0.02): residual random-action rate; novelty does the
      directed exploring.
    - ``coupling`` (1.0): coarse-as-prior strength in pseudo-evidence
      units; 0 severs the hierarchy.
    - ``target_bound`` (8.0): TD targets clip here; the element's
      declared value scale, not a knob.
    - ``settle_budget`` (64): sweeps per settle; unsettled states are
      rejected, never used.
    - ``seed`` (0): the only randomness (tie-breaks, epsilon).
    - ``learning_enabled`` (True): False is the frozen-memory control.
    """

    def __init__(self, n_actions: int, feature_maps: Sequence[Callable[[Any], tuple]], *,
                 decay: float = 0.99, discount: float = 0.97, optimism: float = 0.5,
                 epsilon: float = 0.02, coupling: float = 1.0, target_bound: float = 8.0,
                 settle_budget: int = 64, seed: int = 0, learning_enabled: bool = True):
        if isinstance(n_actions, bool) or not isinstance(n_actions, int) or n_actions < 2:
            raise ValueError('n_actions must be an integer of at least 2')
        if not feature_maps or not all(callable(f) for f in feature_maps):
            raise ValueError('feature_maps must be a nonempty sequence of callables, fine to coarse')
        if not 0.0 < decay <= 1.0 or not 0.0 <= discount < 1.0:
            raise ValueError('decay must be in (0, 1] and discount in [0, 1)')
        if target_bound <= 0 or settle_budget < 8:
            raise ValueError('target_bound must be positive and settle_budget at least 8')
        self.n_actions = n_actions
        self.feature_maps = tuple(feature_maps)
        self.levels = len(self.feature_maps)
        self.decay, self.discount = float(decay), float(discount)
        self.optimism, self.epsilon = float(optimism), float(epsilon)
        self.coupling, self.target_bound = float(coupling), float(target_bound)
        self.settle_budget = int(settle_budget)
        self.learning_enabled = bool(learning_enabled)
        self.prior_ports_cut = False  # readout lesion switch for mechanism checks
        cells = [getattr(f, 'cells', None) for f in self.feature_maps]
        if all(isinstance(c, int) and c > 0 for c in cells):
            self.level_weights = tuple(min(1.0, c / cells[0]) for c in cells)
        else:
            self.level_weights = tuple(1.0 for _ in self.feature_maps)
        self._rng = random.Random(seed)
        self._columns: list[dict] = [{} for _ in range(self.levels)]
        self._cache: dict = {}
        self._episode: list = []
        self.calibration = None
        self.counters = {'updates': 0, 'rejected_updates': 0, 'settles': 0,
                         'cache_hits': 0, 'greedy_actions': 0, 'random_actions': 0}

    @classmethod
    def for_environment(cls, env_factory, n_actions: int | None = None, *,
                        depth: int = 2, width: int = 1, calibration_steps: int = 400,
                        seed: int = 0, **overrides) -> 'Cortex':
        """Build a Cortex for a reset/step environment with no manual wiring.

        Probes ``env.action_space.n`` when ``n_actions`` is omitted, runs
        the calibration probe, wires ``depth``/``width`` feature maps and
        passes ``overrides`` to the constructor. To compare several
        cortices on identical wiring, run ``cadence.calibrate`` once and
        pass ``cadence.wire(calibration, depth, width)`` yourself.
        """
        from .wiring import calibrate, wire
        if n_actions is None:
            probe = env_factory()
            space = getattr(probe, 'action_space', None)
            n_actions = getattr(space, 'n', None)
            close = getattr(probe, 'close', None)
            if callable(close):
                close()
            if n_actions is None:
                raise ValueError('Pass n_actions; the environment exposes no action_space.n')
        calibration = calibrate(env_factory, int(n_actions),
                                steps=calibration_steps, seed=seed)
        cortex = cls(int(n_actions), wire(calibration, depth=depth, width=width),
                     seed=seed, **overrides)
        cortex.calibration = calibration
        return cortex

    # -- columns ------------------------------------------------------------
    def _column(self, level: int, context, action: int) -> ValueColumn:
        table = self._columns[level]
        key = (context, action)
        column = table.get(key)
        if column is None:
            column = table[key] = ValueColumn()
        return column

    def _chain(self, observation, action: int) -> list[ValueColumn]:
        return [self._column(level, self.feature_maps[level](observation), action)
                for level in range(self.levels)]

    # -- the settled readout (the element's law over the level chain) ------
    def _chain_ports(self, stats_list):
        levels = len(stats_list)
        beliefs, ports = [None] * levels, []
        for level in range(levels):
            wf, mean, scatter = el._stats(*stats_list[level])
            if level == levels - 1:
                belief = LevelBelief(wf, mean, level, 0.0)
            else:
                up_wf, _, up_scatter = el._stats(*stats_list[level + 1])
                belief = LevelBelief(wf, mean, level, self.coupling,
                                     prior_dispersion=up_scatter / up_wf)
            beliefs[level] = (belief, LevelObserver(wf, scatter, level), wf, mean)
        for level in reversed(range(levels)):
            belief, observer, wf, mean = beliefs[level]
            if level < levels - 1:
                upper_belief, _, upper_wf, upper_mean = beliefs[level + 1]
                ports.append(el.Port(f'prior{level}', upper_belief, belief, 'moments',
                                     (upper_mean, 1.0 / (upper_wf * el.PRIOR_PRECISION))))
            ports.append(el.Port(f'readback{level}', belief, observer, 'moments',
                                 (mean, 1.0 / (wf * el.PRIOR_PRECISION))))
            ports.append(el.Port(f'feedback{level}', observer, belief, 'scalar',
                                 el.PRIOR_PRECISION))
        return ports

    def value(self, observation, action: int) -> dict:
        """Settle the level chain for one action; cached by column versions."""
        chain = self._chain(observation, action)
        if self.prior_ports_cut:
            chain = chain[:1]
        key = (action, self.prior_ports_cut,
               tuple(id(c) for c in chain), tuple(c.version for c in chain))
        cached = self._cache.get(key)
        if cached is not None:
            self.counters['cache_hits'] += 1
            return cached
        stats_list = [c.stats() for c in chain]
        ports = self._chain_ports(stats_list)
        result = el.settle(ports, budget=self.settle_budget)
        self.counters['settles'] += 1
        mean, variance = result['messages']['readback0']
        novelty = sum(1.0 / (stats_list[level][0] * result['messages'][f'feedback{level}'])
                      for level in range(len(stats_list)))
        outcome = {'mean': mean, 'variance': variance, 'novelty': novelty,
                   'qualified': result['converged'], 'sweeps': result['sweeps'],
                   'residual': result['full_residual']}
        if len(self._cache) > 100000:
            self._cache.clear()
        self._cache[key] = outcome
        return outcome

    # -- acting ---------------------------------------------------------------
    def _score(self, observation, action: int) -> float:
        belief = self.value(observation, action)
        return belief['mean'] + self.optimism * math.sqrt(max(belief['novelty'], 0.0))

    def act(self, observation) -> int:
        if self._rng.random() < self.epsilon:
            self.counters['random_actions'] += 1
            return self._rng.randrange(self.n_actions)
        self.counters['greedy_actions'] += 1
        best_action, best_score = 0, -math.inf
        order = list(range(self.n_actions))
        self._rng.shuffle(order)  # seeded tie-breaking
        for action in order:
            score = self._score(observation, action)
            if score > best_score:
                best_action, best_score = action, score
        return best_action

    # -- learning ---------------------------------------------------------------
    def learn(self, observation, action: int, reward: float,
              next_observation, terminal: bool) -> dict:
        """Buffer one witnessed transition; admission happens at episode end."""
        if not self.learning_enabled:
            return {'buffered': 0, 'admitted': 0}
        self._episode.append((observation, action, float(reward),
                              next_observation, bool(terminal)))
        admitted = self.end_episode() if terminal else 0
        return {'buffered': len(self._episode), 'admitted': admitted}

    def end_episode(self) -> int:
        """Admit the buffered episode backward through every level, once each."""
        transitions, self._episode = self._episode, []
        if not self.learning_enabled:
            return 0
        admitted = 0
        for observation, action, reward, next_observation, terminal in reversed(transitions):
            bootstrap = 0.0
            if not terminal:
                bootstrap = max(self.value(next_observation, a)['mean']
                                for a in range(self.n_actions))
            novelty = self.value(observation, action)['novelty']
            bonus = self.optimism * (1.0 - self.discount) * math.sqrt(max(novelty, 0.0))
            target = reward + bonus + self.discount * bootstrap
            target = max(-self.target_bound, min(self.target_bound, target))
            for level, column in enumerate(self._chain(observation, action)):
                prospective = column.admit(target, self.decay, self.level_weights[level])
                ports = self._chain_ports([(prospective.weight, prospective.linear,
                                            prospective.square)])
                result = el.settle(ports, budget=self.settle_budget)
                self.counters['settles'] += 1
                if result['converged']:
                    column.commit(prospective)
                    admitted += 1
                    self.counters['updates'] += 1
                else:
                    self.counters['rejected_updates'] += 1
        return admitted

    # -- continuation ---------------------------------------------------------------
    def snapshot(self) -> str:
        rows = []
        for level, table in enumerate(self._columns):
            for (context, action), column in table.items():
                rows.append([level, list(context), action,
                             column.weight, column.linear, column.square])
        return json.dumps({'schema': 'cortex-state/1', 'levels': self.levels,
                           'n_actions': self.n_actions, 'rows': rows},
                          sort_keys=True, separators=(',', ':'))

    def restore(self, text: str) -> None:
        state = json.loads(text)
        if state.get('schema') != 'cortex-state/1':
            raise ValueError('Unknown cortex checkpoint schema')
        if state.get('levels') != self.levels or state.get('n_actions') != self.n_actions:
            raise ValueError('Checkpoint shape does not match this cortex')
        columns: list[dict] = [{} for _ in range(self.levels)]
        for level, context, action, weight, linear, square in state['rows']:
            if not 0 <= level < self.levels or not 0 <= action < self.n_actions:
                raise ValueError('Checkpoint row outside the declared shape')
            if not (weight > 0 and math.isfinite(weight)
                    and math.isfinite(linear) and math.isfinite(square) and square >= 0):
                raise ValueError('Checkpoint statistics are not admissible evidence')
            column = ValueColumn()
            column.weight, column.linear, column.square = weight, linear, square
            columns[level][(tuple(context), action)] = column
        self._columns = columns
        self._cache.clear()

    def stats(self) -> dict:
        return {**self.counters,
                'columns_per_level': [len(t) for t in self._columns],
                'level_weights': list(self.level_weights),
                'levels': self.levels}
