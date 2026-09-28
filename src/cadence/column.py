"""The public CorticalColumn: the element with a vertical dimension.

``height`` is the column's internal microcircuit depth - how many
observer stages sit on the belief, the analogue of a layer's hidden
size in other frameworks:

- height 1: belief + precision observer. This is the qualified element
  exactly; every number it produces is identical to
  ``cadence.element.CorticalColumn`` (tested).
- height 2: adds a rate hyper-observer that reads the precision
  observer's live proposal and feeds its rate back - the demonstrated
  recursive-readback stack.
- height H: further rate observers, each reading the stage below and
  feeding its rate down, all settled by the unchanged repair law. No
  new solver, no new schedule: stacking is composition.

Admission custody (ordered events, duplicates, budgets, capacity,
checkpoint validation) mirrors the element verbatim; the height-1
identity test pins the mirror. ``META_SHAPE``/``META_RATE`` are the
declared hyper-observer constants; their prior expected rate is 1, the
element's ``PRIOR_RATE``, so growing height leaves priors consistent.
"""
from __future__ import annotations

import json

from . import element as el

META_SHAPE = 4.0
META_RATE = 4.0
SCHEMA = 'cortical-column-state/2'


class RateObserver:
    """A rate hyper-observer: reads the stage below, proposes its total rate.

    The proposal is ``anchor + META_SHAPE / (rate_in + below)``: the
    stage below keeps its fixed prior rate as an anchor and the
    hyper-observer adds a live, restrainable surplus. Without the
    anchor the chain is degenerate at zero dispersion - the rate
    message drains toward zero, licensing unbounded precision, and the
    stack has no finite fixed point (observed as tall columns refusing
    to qualify heavy zero-scatter evidence). ``read_key``/``up_key``
    are injected so one class serves both the standalone column stack
    and per-level stacks inside a Cortex.
    """

    def __init__(self, anchor: float, read_key: str, up_key: str | None = None):
        self.anchor, self.read_key, self.up_key = anchor, read_key, up_key

    def emit(self, port, inbox):
        below = inbox[self.read_key]
        rate_in = inbox.get(self.up_key, META_RATE) if self.up_key else META_RATE
        return self.anchor + META_SHAPE / (rate_in + below)


def stage_ports(observer, height: int, tau_init: float, *, prefix: str = ''):
    """Hyper-observer ports for stages 2..height above ``observer``.

    With a ``prefix`` (a Cortex level tag) every port name is scoped so
    several stacks can settle in one graph. Stage 2 uses the element's
    own ``meta_feedback`` key so the unchanged ``PrecisionObserver``
    reads its rate; higher stages chain rate observers.
    """
    ports = []
    below, read_key = observer, prefix + 'meta_readback'
    for stage in range(2, height + 1):
        down_key = (prefix + 'meta_feedback' if stage == 2
                    else '%shyper_feedback%d' % (prefix, stage))
        up_key = ('%shyper_feedback%d' % (prefix, stage + 1)
                  if stage < height else None)
        rate = RateObserver(el.PRIOR_RATE if stage == 2 else META_RATE,
                            read_key, up_key)
        ports.append(el.Port(read_key, below, rate, 'scalar',
                             tau_init if stage == 2 else META_SHAPE / META_RATE))
        ports.append(el.Port(down_key, rate, below, 'scalar',
                             (el.PRIOR_RATE if stage == 2 else META_RATE)
                             + META_SHAPE / (META_RATE + 1.0)))
        below, read_key = rate, '%shyper_readback%d' % (prefix, stage + 1)
    return ports


def stack_ports(w, s1, s2, height: int, start=None):
    ports = el.scalar_ports(w, s1, s2, start)
    if height > 1:
        ports += stage_ports(ports[0].target, height, ports[1].message)
    return ports


def settle_stack(w, s1, s2, height: int, *, budget=el.MAX_SWEEPS, start=None, lesion=None):
    ports = stack_ports(w, s1, s2, height, start)
    result = el.settle(ports, budget=budget, lesion=lesion)
    mean, variance = result['messages']['readback']
    stages = {stage: result['messages']['meta_feedback' if stage == 2
                                        else 'hyper_feedback%d' % stage]
              for stage in range(2, height + 1)}
    return {'mean': mean, 'variance': variance,
            'precision': result['messages']['feedback'], 'stages': stages,
            'sweeps': result['sweeps'], 'converged': result['converged'],
            'executed_residual': result['executed_residual'],
            'full_residual': result['full_residual'],
            'stationarity': result['stationarity']}


class CorticalColumn:
    """One column, ``height`` observer stages tall; height 1 is the element."""

    def __init__(self, height: int = 1):
        if isinstance(height, bool) or not isinstance(height, int) or height < 1:
            raise ValueError('height must be an integer of at least 1')
        self.height = height
        self._evidence = el.EvidenceFactor(el.PRIOR_WEIGHT, el.Fraction(0), el.Fraction(0))
        self._cursor = 0
        self._last = None

    # -- transactional witness admission (mirrors the element verbatim) -----
    def observe(self, event, value, *, kind='witness', budget=el.MAX_SWEEPS):
        if kind != 'witness':
            raise ValueError('Only witnessed events are admissible; %r is not owned experience' % (kind,))
        if isinstance(event, bool) or not isinstance(event, int):
            raise ValueError('Event identifier must be an integer')
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('Witness value must be an integer')
        if abs(value) > el.VALUE_BOUND:
            raise ValueError('Witness value outside the declared integer domain')
        if event == self._cursor and self._cursor > 0:
            if value == self._last:
                return {'accepted': False, 'qualified': False, 'duplicate': True, 'sweeps': 0}
            raise ValueError('Conflicting retry of the latest committed event')
        if event != self._cursor + 1:
            raise ValueError('Out-of-order event identifier; expected %d' % (self._cursor + 1))
        if self._cursor >= el.CAPACITY:
            raise ValueError('Declared witness capacity (%d) exhausted' % el.CAPACITY)
        prospective = self._evidence.temper(el.DECAY).combine(el.witness_factor(value))
        result = settle_stack(prospective.weight, prospective.linear,
                              prospective.square, self.height, budget=budget)
        if not result['converged']:
            return {'accepted': False, 'qualified': False, 'duplicate': False,
                    'sweeps': result['sweeps']}
        self._evidence = prospective
        self._cursor, self._last = event, value
        return {'accepted': True, 'qualified': True, 'duplicate': False,
                'sweeps': result['sweeps']}

    # -- read-only query -----------------------------------------------------
    def query(self):
        result = settle_stack(self._evidence.weight, self._evidence.linear,
                              self._evidence.square, self.height)
        return {'answer': result['mean'], 'variance': result['variance'],
                'precision': result['precision'], 'stages': result['stages'],
                'qualified': result['converged'], 'residual': result['full_residual']}

    # -- persistent continuation ----------------------------------------------
    def snapshot(self):
        return json.dumps({'schema': SCHEMA, 'height': self.height,
                           'cursor': self._cursor, 'last': self._last,
                           'w': str(self._evidence.weight),
                           's1': str(self._evidence.linear),
                           's2': str(self._evidence.square)},
                          sort_keys=True, separators=(',', ':'))

    def restore(self, text):
        try:
            state = json.loads(text)
        except (TypeError, ValueError) as error:
            raise ValueError('Malformed checkpoint: %s' % error)
        if not isinstance(state, dict):
            raise ValueError('Malformed checkpoint')
        if state.get('schema') == el.SCHEMA and self.height == 1:
            state = {**state, 'schema': SCHEMA, 'height': 1}  # element continuity
        if state.get('schema') != SCHEMA:
            raise ValueError('Unknown checkpoint schema')
        if state.get('height') != self.height:
            raise ValueError('Checkpoint height does not match this column')
        if set(state) != {'schema', 'height', 'cursor', 'last', 'w', 's1', 's2'}:
            raise ValueError('Checkpoint fields do not match the declared contract')
        cursor, last = state['cursor'], state['last']
        if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor <= el.CAPACITY:
            raise ValueError('Checkpoint cursor outside the declared horizon')
        if last is not None and (isinstance(last, bool) or not isinstance(last, int)
                                 or abs(last) > el.VALUE_BOUND):
            raise ValueError('Checkpoint latest value outside the declared domain')
        if (cursor == 0) != (last is None):
            raise ValueError('Checkpoint cursor and latest value disagree')
        weight = el._parse_fraction(state['w'])
        linear = el._parse_fraction(state['s1'])
        square = el._parse_fraction(state['s2'])
        if weight <= 0 or square < 0:
            raise ValueError('Checkpoint statistics are not admissible evidence')
        self._evidence = el.EvidenceFactor(weight, linear, square)
        self._cursor, self._last = cursor, last

    # -- actual uncertainty readback and feedback ------------------------------
    def observer_intervention(self):
        stats = (self._evidence.weight, self._evidence.linear, self._evidence.square)
        base = settle_stack(*stats, self.height)
        if not base['converged']:
            raise ValueError('Cannot intervene on an unqualified settled state')
        wf, mean, scatter = el._stats(*stats)
        lower = el.LowerBelief(wf, mean)
        observer = el.PrecisionObserver(wf, scatter)
        readback = el.Port('readback', lower, observer, 'moments', (mean, base['variance']))
        feedback = el.Port('feedback', observer, lower, 'scalar', base['precision'])
        variance, tau = base['variance'], base['precision']
        rate_in = base['stages'].get(2, el.PRIOR_RATE)
        recovered = settle_stack(*stats, self.height, start=(variance + 0.05, tau))
        lesions = {}
        for name in el.LESIONS:
            result = settle_stack(*stats, self.height, lesion=name)
            lesions[name] = {'converged': result['converged'],
                             'full_residual': result['full_residual'],
                             'stationarity': result['stationarity']}
        inbox = {'readback': (mean, variance), 'meta_feedback': rate_in}
        after = {'readback': (mean, variance + 0.05), 'meta_feedback': rate_in}
        return {'observer_before': observer.emit(feedback, inbox),
                'observer_after': observer.emit(feedback, after),
                'lower_before': lower.emit(readback, {'feedback': tau})[1],
                'lower_after': lower.emit(readback, {'feedback': tau + 0.1})[1],
                'recovered_qualified': recovered['converged'],
                'recovered_residual': recovered['full_residual'],
                'recovered_stationarity': recovered['stationarity'],
                'height': self.height, 'stages': base['stages'], 'lesions': lesions}

    # -- exact separator-port federation ---------------------------------------
    def solve_factors(self, factors, edges):
        return el.solve_cluster_forest(factors, edges)
