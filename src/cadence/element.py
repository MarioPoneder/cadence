"""The cortical-column element: one bounded settling patch, one repair law.

Provenance: this module is the qualified candidate source
``cadence-candidates/cortical-column/unified.py`` (development workspace,
sha256 44654b3d6053645438b52b0accbead0e367c7746305ee40dd5aa82e9e96e3f07),
vendored with identical executed code; only this docstring differs.
Qualification receipts (7/7 implemented bounded probes and 3/3 candidate
challenge fixtures of the cadence candidate battery, plus the depth-2
recursive readback demonstration) bind to that origin source. Bounded
probe passes are bounded slices, not full capability claims.

The element is a bounded observer-like self-reading patch:

- owned local state: discounted conjugate evidence with leaky
  integration, an ordered-source cursor and the latest committed witness;
- typed boundary ports: witness admission, read-only query, an observer
  loop that reads the live lower-belief uncertainty and feeds precision
  back, and exact separator ports for finite cluster federation;
- one repair law, ``settle``: repropose every port message from its
  source patch's local factor and current inbox, projected to the port's
  declared family, until stationary within a declared residual, or
  reject;
- records: transactional commits; a failed or capped admission cannot
  change committed memory; a retry of the latest identical witness is a
  duplicate, not fresh evidence.

Scalar model (the column's own belief):
    m   = s1 / w
    V   = 1 / (w * tau)
    tau = (a0 + w/2) / (b0 + (E + w*V)/2),  E = s2 - 2*m*s1 + w*m*m
with discounted statistics (w, s1, s2), decay DECAY per admitted witness
and the prior pseudo-evidence folded in. Cutting either direction of the
observer loop is a lesion: the lesioned system settles, but the intact
equations stay violated.

Discrete model (a federation of column ports): exact sum-product
settling over supplied binary factor clusters in Fractions, certified
only on forests satisfying the running-intersection property; cyclic,
non-certifiable or zero-support problems are rejected.

Constants below are the element's declared diagnostic choices, frozen
with the qualified candidate. Applications parameterize their model
around the law (see ``cadence.cortex``), never the law itself.
"""
from fractions import Fraction
from itertools import product
import json

SCHEMA = 'cortical-column-state/1'
DECAY = Fraction(1, 2)      # leaky integration per admitted witness
PRIOR_WEIGHT = Fraction(1)  # pseudo-evidence at value zero
PRIOR_SHAPE = 1.0           # a0: observer shape
PRIOR_RATE = 1.0            # b0: observer rate
PRIOR_PRECISION = PRIOR_SHAPE / PRIOR_RATE
CAPACITY = 24               # declared finite witness capacity
VALUE_BOUND = 8             # declared integer witness domain |v| <= 8
MAX_SWEEPS = 256
FACTOR_SWEEPS = 64
SETTLE_TOL = 1e-14
RESIDUAL_TOL = 1e-11
LESIONS = ('variance_blind', 'feedback_cut')


# --------------------------------------------------------------------------
# Families: declared payload contracts for port messages.
def _moments_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _scalar_distance(a, b):
    return abs(a - b)


def _table_distance(a, b):
    return 0.0 if a == b else 1.0


FAMILIES = {'moments': _moments_distance, 'scalar': _scalar_distance,
            'table': _table_distance}


class Port:
    """One directed boundary between two patches, carrying one family."""
    __slots__ = ('name', 'source', 'target', 'family', 'message', 'meta')

    def __init__(self, name, source, target, family, message, meta=None):
        if family not in FAMILIES:
            raise ValueError('Undeclared port family: %r' % (family,))
        self.name, self.source, self.target = name, source, target
        self.family, self.message, self.meta = family, message, meta or {}


def _apply_lesion(inbox, lesion):
    """Declared port cuts; a lesion severs one read direction, nothing else."""
    if lesion is None:
        return inbox
    if lesion == 'variance_blind' and 'readback' in inbox:
        mean, _ = inbox['readback']
        inbox = {**inbox, 'readback': (mean, 0.0)}
    elif lesion == 'feedback_cut' and 'feedback' in inbox:
        inbox = {**inbox, 'feedback': PRIOR_PRECISION}
    return inbox


def settle(ports, *, budget=MAX_SWEEPS, lesion=None):
    """The one repair law: repropose every port message until stationary.

    ``full_residual`` and ``stationarity`` are always measured against the
    intact equations; ``executed_residual`` against the equations actually
    run (lesioned or intact). Qualification uses the executed equations.
    """
    if lesion is not None and lesion not in LESIONS:
        raise ValueError('Unknown lesion: %r' % (lesion,))
    messages = {port.name: port.message for port in ports}

    def inbox(patch, law, current):
        box = {p.name: current[p.name] for p in ports if p.target is patch}
        return _apply_lesion(box, law)

    sweeps = 0
    for _ in range(max(0, int(budget))):
        delta = 0.0
        for port in ports:
            proposal = port.source.emit(port, inbox(port.source, lesion, messages))
            delta = max(delta, FAMILIES[port.family](proposal, messages[port.name]))
            messages[port.name] = proposal
        sweeps += 1
        if delta <= SETTLE_TOL:
            break

    def residual(law):
        worst = 0.0
        for port in ports:
            proposal = port.source.emit(port, inbox(port.source, law, messages))
            worst = max(worst, FAMILIES[port.family](proposal, messages[port.name]))
        return worst

    executed_residual = residual(lesion)
    full_residual = residual(None)
    trial, stationarity = dict(messages), 0.0
    for port in ports:
        proposal = port.source.emit(port, inbox(port.source, None, trial))
        stationarity = max(stationarity, FAMILIES[port.family](proposal, trial[port.name]))
        trial[port.name] = proposal
    return {'messages': messages, 'sweeps': sweeps,
            'converged': bool(executed_residual <= RESIDUAL_TOL),
            'executed_residual': executed_residual,
            'full_residual': full_residual, 'stationarity': stationarity}


# --------------------------------------------------------------------------
# Scalar column patches.
def _stats(w, s1, s2):
    wf, s1f, s2f = float(w), float(s1), float(s2)
    mean = s1f / wf
    scatter = s2f - 2.0 * mean * s1f + wf * mean * mean
    return wf, mean, max(scatter, 0.0)


class LowerBelief:
    """Superficial belief over the latent center; the evidence weight lives here."""

    def __init__(self, weight, mean):
        self.weight, self.mean = weight, mean

    def emit(self, port, inbox):
        tau = inbox['feedback']
        return (self.mean, 1.0 / (self.weight * tau))


class PrecisionObserver:
    """Deep observer reading the live lower belief, proposing precision."""

    def __init__(self, weight, scatter):
        self.weight, self.scatter = weight, scatter

    def emit(self, port, inbox):
        _, variance = inbox['readback']
        return ((PRIOR_SHAPE + self.weight / 2.0)
                / (inbox.get('meta_feedback', PRIOR_RATE)
                   + (self.scatter + self.weight * variance) / 2.0))


def scalar_ports(w, s1, s2, start=None):
    wf, mean, scatter = _stats(w, s1, s2)
    lower = LowerBelief(wf, mean)
    observer = PrecisionObserver(wf, scatter)
    tau = start[1] if start is not None else PRIOR_PRECISION
    variance = start[0] if start is not None else 1.0 / (wf * tau)
    return [Port('readback', lower, observer, 'moments', (mean, variance)),
            Port('feedback', observer, lower, 'scalar', tau)]


def settle_scalar(w, s1, s2, *, budget=MAX_SWEEPS, start=None, lesion=None):
    ports = scalar_ports(w, s1, s2, start)
    result = settle(ports, budget=budget, lesion=lesion)
    mean, variance = result['messages']['readback']
    return {'mean': mean, 'variance': variance,
            'precision': result['messages']['feedback'],
            'sweeps': result['sweeps'], 'converged': result['converged'],
            'executed_residual': result['executed_residual'],
            'full_residual': result['full_residual'],
            'stationarity': result['stationarity']}


# --------------------------------------------------------------------------
# Retained evidence as a factor with declared combine/temper rules.
class EvidenceFactor:
    """Natural parameters of the column's retained Gaussian evidence."""
    __slots__ = ('weight', 'linear', 'square')

    def __init__(self, weight, linear, square):
        self.weight, self.linear, self.square = weight, linear, square

    def temper(self, decay):
        return EvidenceFactor(decay * self.weight, decay * self.linear,
                              decay * self.square)

    def combine(self, other):
        return EvidenceFactor(self.weight + other.weight,
                              self.linear + other.linear,
                              self.square + other.square)


def witness_factor(value):
    return EvidenceFactor(Fraction(1), Fraction(value), Fraction(value * value))


def _parse_fraction(text):
    if not isinstance(text, str):
        raise ValueError('Serialized statistic must be a string')
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError('Malformed serialized statistic: %s' % error)


class CorticalColumn:
    """One cortical-column candidate instance, settled by the one repair law."""

    def __init__(self):
        self._evidence = EvidenceFactor(PRIOR_WEIGHT, Fraction(0), Fraction(0))
        self._cursor = 0
        self._last = None

    # -- transactional witness admission ------------------------------------
    def observe(self, event, value, *, kind='witness', budget=MAX_SWEEPS):
        if kind != 'witness':
            raise ValueError('Only witnessed events are admissible; %r is not owned experience' % (kind,))
        if isinstance(event, bool) or not isinstance(event, int):
            raise ValueError('Event identifier must be an integer')
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('Witness value must be an integer')
        if abs(value) > VALUE_BOUND:
            raise ValueError('Witness value outside the declared integer domain')
        if event == self._cursor and self._cursor > 0:
            if value == self._last:
                return {'accepted': False, 'qualified': False, 'duplicate': True, 'sweeps': 0}
            raise ValueError('Conflicting retry of the latest committed event')
        if event != self._cursor + 1:
            raise ValueError('Out-of-order event identifier; expected %d' % (self._cursor + 1))
        if self._cursor >= CAPACITY:
            raise ValueError('Declared witness capacity (%d) exhausted' % CAPACITY)
        prospective = self._evidence.temper(DECAY).combine(witness_factor(value))
        result = settle_scalar(prospective.weight, prospective.linear,
                               prospective.square, budget=budget)
        if not result['converged']:
            return {'accepted': False, 'qualified': False, 'duplicate': False,
                    'sweeps': result['sweeps']}
        self._evidence = prospective
        self._cursor, self._last = event, value
        return {'accepted': True, 'qualified': True, 'duplicate': False,
                'sweeps': result['sweeps']}

    # -- read-only query -----------------------------------------------------
    def query(self):
        result = settle_scalar(self._evidence.weight, self._evidence.linear,
                               self._evidence.square)
        return {'answer': result['mean'], 'variance': result['variance'],
                'precision': result['precision'], 'qualified': result['converged'],
                'residual': result['full_residual']}

    # -- persistent continuation ----------------------------------------------
    def snapshot(self):
        return json.dumps({'schema': SCHEMA, 'cursor': self._cursor, 'last': self._last,
                           'w': str(self._evidence.weight),
                           's1': str(self._evidence.linear),
                           's2': str(self._evidence.square)},
                          sort_keys=True, separators=(',', ':'))

    def restore(self, text):
        try:
            state = json.loads(text)
        except (TypeError, ValueError) as error:
            raise ValueError('Malformed checkpoint: %s' % error)
        if not isinstance(state, dict) or state.get('schema') != SCHEMA:
            raise ValueError('Unknown checkpoint schema')
        if set(state) != {'schema', 'cursor', 'last', 'w', 's1', 's2'}:
            raise ValueError('Checkpoint fields do not match the declared contract')
        cursor, last = state['cursor'], state['last']
        if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor <= CAPACITY:
            raise ValueError('Checkpoint cursor outside the declared horizon')
        if last is not None and (isinstance(last, bool) or not isinstance(last, int)
                                 or abs(last) > VALUE_BOUND):
            raise ValueError('Checkpoint latest value outside the declared domain')
        if (cursor == 0) != (last is None):
            raise ValueError('Checkpoint cursor and latest value disagree')
        weight = _parse_fraction(state['w'])
        linear = _parse_fraction(state['s1'])
        square = _parse_fraction(state['s2'])
        if weight <= 0 or square < 0:
            raise ValueError('Checkpoint statistics are not admissible evidence')
        self._evidence = EvidenceFactor(weight, linear, square)
        self._cursor, self._last = cursor, last

    # -- actual uncertainty readback and feedback ------------------------------
    def observer_intervention(self):
        stats = (self._evidence.weight, self._evidence.linear, self._evidence.square)
        base = settle_scalar(*stats)
        if not base['converged']:
            raise ValueError('Cannot intervene on an unqualified settled state')
        wf, mean, scatter = _stats(*stats)
        lower = LowerBelief(wf, mean)
        observer = PrecisionObserver(wf, scatter)
        readback = Port('readback', lower, observer, 'moments', (mean, base['variance']))
        feedback = Port('feedback', observer, lower, 'scalar', base['precision'])
        variance, tau = base['variance'], base['precision']
        recovered = settle_scalar(*stats, start=(variance + 0.05, tau))
        lesions = {}
        for name in LESIONS:
            result = settle_scalar(*stats, lesion=name)
            lesions[name] = {'converged': result['converged'],
                             'full_residual': result['full_residual'],
                             'stationarity': result['stationarity']}
        return {'observer_before': observer.emit(feedback, {'readback': (mean, variance)}),
                'observer_after': observer.emit(feedback, {'readback': (mean, variance + 0.05)}),
                'lower_before': lower.emit(readback, {'feedback': tau})[1],
                'lower_after': lower.emit(readback, {'feedback': tau + 0.1})[1],
                'recovered_qualified': recovered['converged'],
                'recovered_residual': recovered['full_residual'],
                'recovered_stationarity': recovered['stationarity'],
                'lesions': lesions}

    # -- exact separator-port federation ---------------------------------------
    def solve_factors(self, factors, edges):
        return solve_cluster_forest(factors, edges)


def _validate_clusters(factors):
    if not isinstance(factors, (list, tuple)) or not factors:
        raise ValueError('A nonempty factor list is required')
    clusters = []
    for spec in factors:
        if not isinstance(spec, dict):
            raise ValueError('Each factor must be a scope/values mapping')
        scope = tuple(spec.get('scope', ()))
        if not scope or len(set(scope)) != len(scope):
            raise ValueError('Factor scope must list distinct variables')
        values = spec.get('values')
        expected = set(product((0, 1), repeat=len(scope)))
        if not isinstance(values, dict) or {tuple(k) for k in values} != expected or len(values) != len(expected):
            raise ValueError('Factor table must cover every binary assignment exactly once')
        table = {}
        for key, raw in values.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, Fraction)):
                raise ValueError('Potentials must be exact integers or fractions')
            weight = Fraction(raw)
            if weight < 0:
                raise ValueError('Potentials must be nonnegative')
            table[tuple(key)] = weight
        clusters.append((scope, table))
    return clusters


def _validate_forest(edges, count):
    pairs, seen = [], set()
    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for edge in edges:
        pair = tuple(edge)
        if len(pair) != 2:
            raise ValueError('Edges must join exactly two clusters')
        i, j = pair
        for index in (i, j):
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
                raise ValueError('Edge endpoint outside the cluster list')
        if i == j or frozenset((i, j)) in seen:
            raise ValueError('Degenerate or duplicate separator edge')
        if find(i) == find(j):
            raise ValueError('Cyclic cluster graph; exactness is only certified on forests')
        parent[find(i)] = find(j)
        seen.add(frozenset((i, j)))
        pairs.append((i, j))
    return pairs


def _check_running_intersection(clusters, pairs, separators):
    for variable in set().union(*(set(scope) for scope, _ in clusters)):
        holders = [k for k, (scope, _) in enumerate(clusters) if variable in scope]
        parent = {k: k for k in holders}

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for i, j in pairs:
            if variable in separators[(i, j)]:
                parent[find(i)] = find(j)
        if len({find(k) for k in holders}) > 1:
            raise ValueError('Running intersection violated for %r; exactness not certified'
                             % (variable,))


class ClusterPatch:
    """One finite column cluster: local exact table plus separator ports."""

    def __init__(self, scope, table):
        self.scope, self.table = scope, table
        self.positions = {v: i for i, v in enumerate(scope)}
        self.incoming = {}  # port name -> separator variables

    def emit(self, port, inbox):
        sep, exclude = port.meta['sep'], port.meta['exclude']
        active = [(vars_, inbox[name]) for name, vars_ in self.incoming.items()
                  if name != exclude and name in inbox]
        out = {a: Fraction(0) for a in product((0, 1), repeat=len(sep))}
        for assignment, weight in self.table.items():
            for vars_, message in active:
                weight = weight * message[tuple(assignment[self.positions[v]] for v in vars_)]
            out[tuple(assignment[self.positions[v]] for v in sep)] += weight
        return out


def solve_cluster_forest(factors, edges, budget=FACTOR_SWEEPS):
    """Settle exact separator messages on a certified cluster forest."""
    clusters = _validate_clusters(factors)
    pairs = _validate_forest(edges, len(clusters))
    separators = {}
    for i, j in pairs:
        shared = tuple(sorted(set(clusters[i][0]) & set(clusters[j][0]), key=str))
        separators[(i, j)] = shared
        separators[(j, i)] = shared
    _check_running_intersection(clusters, pairs, separators)
    patches = [ClusterPatch(scope, table) for scope, table in clusters]
    ports = []
    for i, j in [(i, j) for i, j in pairs] + [(j, i) for i, j in pairs]:
        name, reverse = '%d->%d' % (i, j), '%d->%d' % (j, i)
        sep = separators[(i, j)]
        ports.append(Port(name, patches[i], patches[j], 'table',
                          {a: Fraction(1) for a in product((0, 1), repeat=len(sep))},
                          meta={'sep': sep, 'exclude': reverse}))
        patches[j].incoming[name] = sep
    result = settle(ports, budget=budget)
    if not result['converged']:
        raise ValueError('Separator messages did not settle within the declared budget')
    messages = result['messages']
    beliefs = []
    for index, patch in enumerate(patches):
        values = {}
        for assignment, weight in patch.table.items():
            for name, vars_ in patch.incoming.items():
                weight = weight * messages[name][tuple(assignment[patch.positions[v]] for v in vars_)]
            values[assignment] = weight
        mass = sum(values.values(), Fraction(0))
        if mass == 0:
            raise ValueError('Zero joint support; the supplied constraints are contradictory')
        beliefs.append({'scope': patch.scope,
                        'values': {a: w / mass for a, w in values.items()}})
    return beliefs
