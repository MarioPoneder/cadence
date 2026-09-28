"""The cortical-column element: one bounded settling patch, one repair law.

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

"""

import json
import math
import re
from decimal import Decimal
from fractions import Fraction
from itertools import product
from numbers import Real

SCHEMA = "element-state/1"
DECAY = Fraction(1, 2)  # leaky integration per admitted witness
PRIOR_WEIGHT = Fraction(1)  # pseudo-evidence at value zero
PRIOR_SHAPE = 1.0  # a0: observer shape
PRIOR_RATE = 1.0  # b0: observer rate
PRIOR_PRECISION = PRIOR_SHAPE / PRIOR_RATE
CAPACITY = 24  # declared finite witness capacity
VALUE_BOUND = 8  # declared integer witness domain |v| <= 8
MAX_SWEEPS = 256
FACTOR_SWEEPS = 64
SETTLE_TOL = 1e-14
RESIDUAL_TOL = 1e-11
LESIONS = ("variance_blind", "feedback_cut")
MAX_STATISTIC_BITS = 16384


def _finite(value, name, *, positive=False):
    """Validate a real numeric scalar without accepting booleans or strings."""
    # Settlement produces builtin floats. Avoid numeric ABC dispatch for them
    # while keeping the full conversion contract for external numeric types.
    if type(value) is float:
        number = value
    else:
        if type(value) is not int and (
            isinstance(value, bool) or not isinstance(value, (Real, Decimal))
        ):
            raise ValueError(f"{name} must be a finite real number")
        try:
            number = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(
            "{} must be {}finite".format(name, "positive and " if positive else "")
        )
    return number


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _rational(value, name, max_bits=MAX_STATISTIC_BITS):
    _finite(value, name)
    # Decimal representations give reproducible decimal-valued observations;
    # Fraction inputs retain their exact supplied rational value.
    if isinstance(value, Fraction):
        result = value
    else:
        decimal = Decimal(str(value))
        digits = decimal.as_tuple()
        if len(digits.digits) > max_bits or abs(digits.exponent) > max_bits:
            raise ValueError(f"{name} exceeds the exact arithmetic budget")
        result = Fraction(decimal)
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > max_bits:
        raise ValueError(f"{name} exceeds the exact arithmetic budget")
    return result


def _payload(family, value):
    if family == "scalar":
        return _finite(value, "scalar message")
    if family == "moments":
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("A moments message must contain mean and variance")
        mean, variance = (_finite(v, "moment") for v in value)
        if variance < 0:
            raise ValueError("Message variance must be nonnegative")
        return (mean, variance)
    if family == "table":
        if not isinstance(value, dict) or not value:
            raise ValueError("A table message must be a nonempty exact mapping")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, Fraction)) or v < 0
            for v in value.values()
        ):
            raise ValueError("Table messages require nonnegative exact weights")
        return dict(value)
    raise ValueError(f"Undeclared port family: {family!r}")


# --------------------------------------------------------------------------
# Families: declared payload contracts for port messages.
def _moments_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _scalar_distance(a, b):
    return abs(a - b)


def _table_distance(a, b):
    return 0.0 if a == b else 1.0


FAMILIES = {
    "moments": _moments_distance,
    "scalar": _scalar_distance,
    "table": _table_distance,
}


class Port:
    """One directed boundary between two patches, carrying one family."""

    __slots__ = ("name", "source", "target", "family", "message", "meta")

    def __init__(self, name, source, target, family, message, meta=None):
        if not isinstance(name, str) or not name:
            raise ValueError("A port requires a nonempty string name")
        if not callable(getattr(source, "emit", None)) or target is None:
            raise ValueError("A port needs an emitting source and a target patch")
        if not isinstance(family, str) or family not in FAMILIES:
            raise ValueError(f"Undeclared port family: {family!r}")
        if meta is not None and not isinstance(meta, dict):
            raise ValueError("Port metadata must be a dictionary")
        self.name, self.source, self.target = name, source, target
        self.family, self.message, self.meta = (
            family,
            _payload(family, message),
            dict(meta or {}),
        )


def _apply_lesion(inbox, lesion, precision=PRIOR_PRECISION):
    """Declared port cuts; a lesion severs one read direction, nothing else."""
    if lesion is None:
        return inbox
    if lesion == "variance_blind" and "readback" in inbox:
        mean, _ = inbox["readback"]
        inbox = {**inbox, "readback": (mean, 0.0)}
    elif lesion == "feedback_cut" and "feedback" in inbox:
        inbox = {**inbox, "feedback": precision}
    return inbox


def settle(
    ports,
    *,
    budget=MAX_SWEEPS,
    lesion=None,
    tolerance=RESIDUAL_TOL,
    damping=1.0,
    lesion_precision=PRIOR_PRECISION,
):
    """The one repair law: repropose every port message until stationary.

    ``full_residual`` and ``stationarity`` are always measured against the
    intact equations; ``executed_residual`` against the equations actually
    run (lesioned or intact). Qualification checks the undamped executed
    equations and a fresh sweep. ``damping`` in (0, 1] mixes scalar/moment
    proposals; exact table messages require damping=1. An exhausted budget
    returns an unqualified result; a zero budget never qualifies nonempty
    ports. Emitters must be deterministic, pure functions of their inbox.
    """
    _integer(budget, "budget")
    tolerance = _finite(tolerance, "tolerance", positive=True)
    damping = _finite(damping, "damping", positive=True)
    if damping > 1:
        raise ValueError("damping must be in (0, 1]")
    lesion_precision = _finite(lesion_precision, "lesion precision", positive=True)
    if lesion is not None and lesion not in LESIONS:
        raise ValueError(f"Unknown lesion: {lesion!r}")
    try:
        ports = tuple(ports)
    except TypeError as error:
        raise ValueError("settle requires an iterable of ports") from error
    if any(not isinstance(port, Port) for port in ports):
        raise ValueError("settle requires Port instances")
    if len({port.name for port in ports}) != len(ports):
        raise ValueError("Port names must be unique within a settlement")
    if damping != 1 and any(p.family == "table" for p in ports):
        raise ValueError("Exact table messages do not support damping")
    equation_tolerance = (
        min(tolerance, 0.5) if any(p.family == "table" for p in ports) else tolerance
    )
    messages = {port.name: _payload(port.family, port.message) for port in ports}
    incoming = {}
    for port in ports:
        incoming.setdefault(id(port.target), []).append(port)

    def inbox(patch, law, current):
        # Initial, emitted and damped messages are already canonical. Scalars
        # and moment tuples are immutable; table dictionaries need a local copy.
        box = {
            p.name: current[p.name].copy() if p.family == "table" else current[p.name]
            for p in incoming.get(id(patch), ())
        }
        return _apply_lesion(box, law, lesion_precision)

    def propose(port, law, current):
        try:
            value = _payload(
                port.family, port.source.emit(port, inbox(port.source, law, current))
            )
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError(
                f"Port {port.name!r} produced invalid arithmetic"
            ) from error
        if port.family == "table" and value.keys() != messages[port.name].keys():
            raise ValueError(f"Port {port.name!r} changed its table domain")
        return value

    def residual(law):
        return max(
            (
                FAMILIES[p.family](propose(p, law, messages), messages[p.name])
                for p in ports
            ),
            default=0.0,
        )

    def sweep_residual(law):
        trial, worst = dict(messages), 0.0
        for port in ports:
            proposal = propose(port, law, trial)
            worst = max(worst, FAMILIES[port.family](proposal, trial[port.name]))
            trial[port.name] = proposal
        return worst

    sweeps = 0
    for _ in range(budget):
        delta = 0.0
        for port in ports:
            proposal = propose(port, lesion, messages)
            if damping != 1:
                old = messages[port.name]
                if port.family == "moments":
                    proposal = tuple(
                        (1 - damping) * a + damping * b
                        for a, b in zip(old, proposal, strict=True)
                    )
                else:
                    proposal = (1 - damping) * old + damping * proposal
                proposal = _payload(port.family, proposal)
            delta = max(delta, FAMILIES[port.family](proposal, messages[port.name]))
            messages[port.name] = proposal
        sweeps += 1
        if (
            delta <= min(SETTLE_TOL, equation_tolerance)
            and residual(lesion) <= equation_tolerance
            and sweep_residual(lesion) <= equation_tolerance
        ):
            break

    executed_residual = residual(lesion)
    full_residual = residual(None)

    stationarity = sweep_residual(None)
    executed_stationarity = sweep_residual(lesion)
    return {
        "messages": messages,
        "sweeps": sweeps,
        "converged": bool(
            (sweeps > 0 or not ports)
            and max(executed_residual, executed_stationarity) <= equation_tolerance
        ),
        "executed_residual": executed_residual,
        "full_residual": full_residual,
        "stationarity": stationarity,
        "executed_stationarity": executed_stationarity,
    }


# --------------------------------------------------------------------------
# Scalar column patches.
def _stats(w, s1, s2):
    """Validate moments and derive finite weight, center and centered scatter."""
    wf = _finite(w, "evidence weight", positive=True)
    s1f, s2f = _finite(s1, "linear statistic"), _finite(s2, "square statistic")
    if s2f < 0:
        raise ValueError("Square statistic must be nonnegative")
    exact = all(isinstance(v, (int, Fraction)) for v in (w, s1, s2))
    if exact:
        if w * s2 < s1 * s1:
            raise ValueError("Evidence violates weight*square >= linear**2")
    else:
        lhs, rhs = s2f, (s1f / wf) * s1f
        if not math.isfinite(rhs) or lhs + 64 * math.ulp(max(abs(lhs), abs(rhs))) < rhs:
            raise ValueError("Evidence violates weight*square >= linear**2")
    mean = s1f / wf
    scatter = float(Fraction(w * s2 - s1 * s1, w)) if exact else s2f - mean * s1f
    if not math.isfinite(mean) or not math.isfinite(scatter):
        raise ValueError("Evidence moments overflow the numeric solver")
    return wf, mean, max(scatter, 0.0)


class LowerBelief:
    """Superficial belief over the latent center; the evidence weight lives here."""

    def __init__(self, weight, mean):
        self.weight = _finite(weight, "weight", positive=True)
        self.mean = _finite(mean, "mean")

    def emit(self, port, inbox):
        """Propose the evidence center and variance under incoming precision."""
        tau = _finite(inbox["feedback"], "precision", positive=True)
        precision = _finite(self.weight * tau, "total precision", positive=True)
        return (self.mean, 1.0 / precision)


class PrecisionObserver:
    """Deep observer reading the live lower belief, proposing precision."""

    def __init__(
        self, weight, scatter, *, prior_shape=PRIOR_SHAPE, prior_rate=PRIOR_RATE
    ):
        self.weight = _finite(weight, "weight", positive=True)
        self.scatter = _finite(scatter, "scatter")
        if self.scatter < 0:
            raise ValueError("Scatter must be nonnegative")
        self.prior_shape = _finite(prior_shape, "prior_shape", positive=True)
        self.prior_rate = _finite(prior_rate, "prior_rate", positive=True)

    def emit(self, port, inbox):
        """Propose precision using the lower belief's live variance."""
        _, variance = _payload("moments", inbox["readback"])
        rate = _finite(
            inbox.get("meta_feedback", self.prior_rate), "observer rate", positive=True
        )
        return (self.prior_shape + self.weight / 2.0) / (
            rate + (self.scatter + self.weight * variance) / 2.0
        )


def scalar_ports(
    w, s1, s2, start=None, *, prior_shape=PRIOR_SHAPE, prior_rate=PRIOR_RATE
):
    """Build a belief/precision loop from valid retained moments."""
    wf, mean, scatter = _stats(w, s1, s2)
    lower = LowerBelief(wf, mean)
    observer = PrecisionObserver(
        wf, scatter, prior_shape=prior_shape, prior_rate=prior_rate
    )
    if start is not None and (not isinstance(start, (tuple, list)) or len(start) != 2):
        raise ValueError("start must be (variance, precision)")
    tau = start[1] if start is not None else observer.prior_shape / observer.prior_rate
    tau = _finite(tau, "initial precision", positive=True)
    variance = start[0] if start is not None else 1.0 / (wf * tau)
    return [
        Port("readback", lower, observer, "moments", (mean, variance)),
        Port("feedback", observer, lower, "scalar", tau),
    ]


def settle_scalar(
    w,
    s1,
    s2,
    *,
    budget=MAX_SWEEPS,
    start=None,
    lesion=None,
    prior_shape=PRIOR_SHAPE,
    prior_rate=PRIOR_RATE,
    tolerance=RESIDUAL_TOL,
    damping=1.0,
):
    """Settle one scalar belief; qualification always measures actual equations."""
    ports = scalar_ports(
        w, s1, s2, start, prior_shape=prior_shape, prior_rate=prior_rate
    )
    result = settle(
        ports,
        budget=budget,
        lesion=lesion,
        tolerance=tolerance,
        damping=damping,
        lesion_precision=prior_shape / prior_rate,
    )
    mean, variance = result["messages"]["readback"]
    return {
        "mean": mean,
        "variance": variance,
        "precision": result["messages"]["feedback"],
        "sweeps": result["sweeps"],
        "converged": result["converged"],
        "executed_residual": result["executed_residual"],
        "full_residual": result["full_residual"],
        "stationarity": result["stationarity"],
    }


# --------------------------------------------------------------------------
# Retained evidence as a factor with declared combine/temper rules.
class EvidenceFactor:
    """Natural parameters of the column's retained Gaussian evidence."""

    __slots__ = ("weight", "linear", "square")

    def __init__(self, weight, linear, square):
        _stats(weight, linear, square)
        self.weight, self.linear, self.square = weight, linear, square

    def temper(self, decay):
        """Return a new factor with all sufficient statistics scaled together."""
        decay = (
            _finite(decay, "decay", positive=True)
            if not isinstance(decay, Fraction)
            else decay
        )
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        return EvidenceFactor(
            decay * self.weight, decay * self.linear, decay * self.square
        )

    def combine(self, other):
        """Return the product factor by adding valid natural parameters."""
        return EvidenceFactor(
            self.weight + other.weight,
            self.linear + other.linear,
            self.square + other.square,
        )


def witness_factor(value):
    """The unit-mass Gaussian evidence factor for one exact real observation."""
    value = _rational(value, "witness", 65536)
    return EvidenceFactor(Fraction(1), Fraction(value), Fraction(value * value))


def _fraction_text(value):
    """Canonical decimal rational text, without changing Python's global limits."""

    def integer_text(number):
        if number < 0:
            return "-" + integer_text(-number)
        chunks = []
        while number >= 10**9:
            number, chunk = divmod(number, 10**9)
            chunks.append(chunk)
        return str(number) + "".join(f"{chunk:09d}" for chunk in reversed(chunks))

    numerator = integer_text(value.numerator)
    return (
        numerator
        if value.denominator == 1
        else numerator + "/" + integer_text(value.denominator)
    )


def _parse_fraction(text, max_bits=MAX_STATISTIC_BITS):
    """Parse only canonical bounded integers or n/d; never decimal exponents."""
    _integer(max_bits, "max_statistic_bits", minimum=64)
    limit = max_bits * 30103 // 100000 + 2
    if not isinstance(text, str) or len(text) > 2 * limit + 2:
        raise ValueError("Serialized statistic exceeds its arithmetic budget")
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", text):
        raise ValueError("Statistic must be a canonical integer or rational n/d")

    def parse_integer(part):
        negative = part.startswith("-")
        part = part[negative:]
        if len(part) > limit:
            raise ValueError("Serialized integer exceeds its arithmetic budget")
        value = 0
        for offset in range(0, len(part), 9):
            chunk = part[offset : offset + 9]
            value = value * 10 ** len(chunk) + int(chunk)
        if value.bit_length() > max_bits:
            raise ValueError("Serialized integer exceeds its arithmetic budget")
        return -value if negative else value

    parts = text.split("/")
    value = Fraction(
        parse_integer(parts[0]), parse_integer(parts[1]) if len(parts) == 2 else 1
    )
    if _fraction_text(value) != text:
        raise ValueError("Serialized statistic is not in canonical lowest terms")
    return value


def _read_checkpoint(text, max_bits=MAX_STATISTIC_BITS):
    if not isinstance(text, str) or len(text) > max(8192, 12 * max_bits):
        raise ValueError("Checkpoint exceeds its declared text budget")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate checkpoint field")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("Nonfinite checkpoint value")

    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"Malformed checkpoint: {error}") from error


class CorticalColumn:
    """A scalar belief and precision observer with transactional evidence.

    ``decay`` tempers retained evidence per admitted witness, in (0, 1].
    ``capacity=None`` permits any event count within ``max_statistic_bits``;
    A positive integer capacity limits the number of committed events.
    ``value_bound`` is the maximum absolute real observation. ``settle_budget``
    bounds local sweeps, ``tolerance`` bounds actual equation residuals, and
    ``damping`` mixes proposals in (0, 1]. Positive ``prior_weight`` represents
    initial zero-valued pseudo-evidence; positive ``prior_shape``/``prior_rate``
    configure the precision observer. Exact rational moments are retained;
    hitting their bit budget rejects admission atomically, without rounding.

    This is a model of a scalar stream, not an arbitrary neural circuit.
    Queries never admit evidence. ``add`` allocates the next local event ID;
    use ``observe`` when the caller owns IDs and needs retry protection.
    """

    _schema = SCHEMA

    def __init__(
        self,
        *,
        decay=DECAY,
        capacity=None,
        value_bound=VALUE_BOUND,
        settle_budget=MAX_SWEEPS,
        tolerance=RESIDUAL_TOL,
        damping=1.0,
        prior_weight=PRIOR_WEIGHT,
        prior_shape=PRIOR_SHAPE,
        prior_rate=PRIOR_RATE,
        max_statistic_bits=MAX_STATISTIC_BITS,
    ):
        from types import MappingProxyType

        _integer(max_statistic_bits, "max_statistic_bits", minimum=64)
        if max_statistic_bits > 65536:
            raise ValueError("max_statistic_bits must not exceed 65536")
        decay = _rational(decay, "decay", max_statistic_bits)
        value_bound = _rational(value_bound, "value_bound", max_statistic_bits)
        prior_weight = _rational(prior_weight, "prior_weight", max_statistic_bits)
        if not 0 < decay <= 1 or value_bound <= 0 or prior_weight <= 0:
            raise ValueError(
                "decay must be in (0, 1]; bounds and prior weight must be positive"
            )
        if capacity is not None:
            _integer(capacity, "capacity", minimum=1)
        _integer(settle_budget, "settle_budget", minimum=1)
        tolerance = _finite(tolerance, "tolerance", positive=True)
        damping = _finite(damping, "damping", positive=True)
        if damping > 1:
            raise ValueError("damping must be in (0, 1]")
        prior_shape = _finite(prior_shape, "prior_shape", positive=True)
        prior_rate = _finite(prior_rate, "prior_rate", positive=True)
        if prior_shape + (float(prior_weight) - 1) / 2 <= 0:
            raise ValueError("prior_shape + (prior_weight - 1)/2 must be positive")
        tau = _finite(prior_shape / prior_rate, "prior precision", positive=True)
        total = _finite(
            float(prior_weight) * tau, "total prior precision", positive=True
        )
        _finite(1.0 / total, "initial variance", positive=True)
        stationary_tau = _finite(
            (prior_shape + (float(prior_weight) - 1) / 2) / prior_rate,
            "stationary prior precision",
            positive=True,
        )
        total = _finite(
            float(prior_weight) * stationary_tau,
            "stationary total prior precision",
            positive=True,
        )
        _finite(1.0 / total, "stationary prior variance", positive=True)
        self._config = MappingProxyType(
            dict(
                decay=decay,
                capacity=capacity,
                value_bound=value_bound,
                settle_budget=settle_budget,
                tolerance=tolerance,
                damping=damping,
                prior_weight=prior_weight,
                prior_shape=prior_shape,
                prior_rate=prior_rate,
                max_statistic_bits=max_statistic_bits,
            )
        )
        self._evidence = EvidenceFactor(prior_weight, Fraction(0), Fraction(0))
        self._cursor, self._last = 0, None

    @property
    def config(self):
        """Read-only numeric configuration; checkpoint identity includes every key."""
        return self._config

    @property
    def count(self):
        """Number of committed, distinct witnesses."""
        return self._cursor

    def _settle(self, evidence, *, budget=None, start=None, lesion=None):
        return settle_scalar(
            evidence.weight,
            evidence.linear,
            evidence.square,
            budget=self.config["settle_budget"] if budget is None else budget,
            start=start,
            lesion=lesion,
            prior_shape=self.config["prior_shape"],
            prior_rate=self.config["prior_rate"],
            tolerance=self.config["tolerance"],
            damping=self.config["damping"],
        )

    def _bounded_evidence(self, evidence):
        limit = self.config["max_statistic_bits"]
        for value in (evidence.weight, evidence.linear, evidence.square):
            if (
                max(value.numerator.bit_length(), value.denominator.bit_length())
                > limit
            ):
                raise ValueError(
                    "Evidence exceeds max_statistic_bits; admission was not committed"
                )
        _stats(evidence.weight, evidence.linear, evidence.square)

    def add(self, value, *, budget=None):
        """Admit a real observation using the next local ordered event ID.

        A capped solve returns ``accepted=False`` and does not consume an ID.
        Calling ``add`` twice is two events, even if the values are identical.
        """
        return self.observe(self._cursor + 1, value, budget=budget)

    def observe(self, event, value, *, kind="witness", budget=None):
        """Condition on one owned witness; identical latest retries are no-ops.

        ``value`` accepts finite real scalars, Decimal and Fraction. Floats
        use their shortest decimal representation as exact retained evidence.
        Wrong IDs, invalid inputs and resource overflow raise ValueError;
        nonconvergence returns an unqualified result. Both leave memory intact.
        """
        if kind != "witness":
            raise ValueError("Only witnessed events are admissible")
        _integer(event, "event", minimum=1)
        actual_budget = (
            self.config["settle_budget"]
            if budget is None
            else _integer(budget, "budget")
        )
        value = _rational(value, "witness", self.config["max_statistic_bits"])
        if abs(value) > self.config["value_bound"]:
            raise ValueError("Witness value outside value_bound")
        if event == self._cursor and value == self._last:
            return {
                "accepted": False,
                "qualified": False,
                "duplicate": True,
                "sweeps": 0,
            }
        if event != self._cursor + 1:
            raise ValueError(
                f"Conflicting or out-of-order event; expected {self._cursor + 1}"
            )
        capacity = self.config["capacity"]
        if capacity is not None and self._cursor >= capacity:
            raise ValueError(f"Declared witness capacity ({capacity}) exhausted")
        prospective = self._evidence.temper(self.config["decay"]).combine(
            witness_factor(value)
        )
        self._bounded_evidence(prospective)
        result = self._settle(prospective, budget=actual_budget)
        if result["converged"]:
            self._evidence, self._cursor, self._last = prospective, event, value
        return {
            "accepted": result["converged"],
            "qualified": result["converged"],
            "duplicate": False,
            "sweeps": result["sweeps"],
        }

    def query(self):
        """Settle and measure the current belief without changing retained evidence."""
        result = self._settle(self._evidence)
        return {
            "answer": result["mean"],
            "variance": result["variance"],
            "precision": result["precision"],
            "qualified": result["converged"],
            "residual": result["full_residual"],
        }

    def _checkpoint_config(self):
        return {
            key: _fraction_text(value) if isinstance(value, Fraction) else value
            for key, value in self.config.items()
        }

    def snapshot(self):
        """Return a bounded JSON continuation with complete configuration identity."""
        return json.dumps(
            {
                "schema": self._schema,
                "config": self._checkpoint_config(),
                "cursor": self._cursor,
                "last": None if self._last is None else _fraction_text(self._last),
                "w": _fraction_text(self._evidence.weight),
                "s1": _fraction_text(self._evidence.linear),
                "s2": _fraction_text(self._evidence.square),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def restore(self, text):
        """Atomically restore a compatible checkpoint, rejecting impossible moments.

        Validation establishes a possible bounded evidence state, not authentic
        provenance of the original sensor history. Treat checkpoints as caller-
        supplied state; they are not signed event logs.
        """
        state = _read_checkpoint(text, self.config["max_statistic_bits"])
        if not isinstance(state, dict):
            raise ValueError("Checkpoint must be an object")
        expected = {"schema", "config", "cursor", "last", "w", "s1", "s2"}
        if state.get("schema") != self._schema:
            raise ValueError("Unknown checkpoint schema")
        if json.dumps(
            state.get("config"), sort_keys=True, allow_nan=False
        ) != json.dumps(self._checkpoint_config(), sort_keys=True):
            raise ValueError("Checkpoint configuration does not match this column")
        if set(state) != expected:
            raise ValueError("Checkpoint fields do not match the declared contract")
        cursor, last = state["cursor"], state["last"]
        _integer(cursor, "checkpoint cursor")
        if self.config["capacity"] is not None and cursor > self.config["capacity"]:
            raise ValueError("Checkpoint cursor exceeds capacity")
        if (cursor == 0) != (last is None):
            raise ValueError("Checkpoint cursor and latest witness disagree")
        limit = self.config["max_statistic_bits"]
        if last is not None:
            last = _parse_fraction(last, limit)
            if abs(last) > self.config["value_bound"]:
                raise ValueError("Checkpoint latest witness exceeds value_bound")
        weight, linear, square = (
            _parse_fraction(state[key], limit) for key in ("w", "s1", "s2")
        )
        evidence = EvidenceFactor(weight, linear, square)
        self._bounded_evidence(evidence)
        decay, initial = self.config["decay"], self.config["prior_weight"]
        if cursor == 0:
            if (weight, linear, square) != (initial, 0, 0):
                raise ValueError("An empty checkpoint must contain the declared prior")
        else:
            # The weight recurrence is known independently of observations.
            if decay == 1:
                expected_weight, witness_mass = initial + cursor, Fraction(cursor)
            else:
                fixed = 1 / (1 - decay)
                if initial == fixed:
                    expected_weight = fixed
                else:
                    coefficient = initial - fixed
                    growth = cursor * (decay.denominator.bit_length() - 1)
                    cancellation_budget = (
                        abs(coefficient.numerator).bit_length()
                        + fixed.denominator.bit_length()
                    )
                    if growth > limit + cancellation_budget:
                        raise ValueError(
                            "Checkpoint count exceeds exact reconstruction budget"
                        )
                    expected_weight = fixed + coefficient * decay**cursor
                # Bounds using total mass are conservative when the initial
                # prior is at zero; exact history authentication is not claimed.
                witness_mass = expected_weight
            if weight != expected_weight:
                raise ValueError("Checkpoint weight does not match its event count")
            bound = self.config["value_bound"]
            if (
                abs(linear) > witness_mass * bound
                or square > witness_mass * bound * bound
            ):
                raise ValueError("Checkpoint evidence exceeds the witness domain")
            remaining = weight - 1
            if (
                remaining <= 0
                or square < last * last
                or (linear - last) ** 2 > remaining * (square - last * last)
            ):
                raise ValueError("Latest witness is inconsistent with retained moments")
            if cursor == 1 and (linear != last or square != last * last):
                raise ValueError("First witness does not match its retained moments")
        self._evidence, self._cursor, self._last = evidence, cursor, last

    @classmethod
    def from_snapshot(cls, text):
        """Construct from a current-format checkpoint, then apply full validation."""
        state = _read_checkpoint(text, 65536)
        if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
            raise ValueError(
                "from_snapshot requires a current configuration-bound checkpoint"
            )
        config = dict(state["config"])
        limit = config.get("max_statistic_bits")
        _integer(limit, "max_statistic_bits", minimum=64)
        if limit > 65536:
            raise ValueError("max_statistic_bits exceeds the supported bound")
        for key in ("decay", "value_bound", "prior_weight"):
            config[key] = _parse_fraction(config.get(key), limit)
        try:
            column = cls(**config)
        except TypeError as error:
            raise ValueError("Invalid checkpoint configuration") from error
        column.restore(text)
        return column

    def observer_intervention(self):
        """Measure both readback directions and diagnostic lesions without learning."""
        base = self._settle(self._evidence)
        if not base["converged"]:
            raise ValueError("Cannot intervene on an unqualified settled state")
        wf, mean, scatter = _stats(
            self._evidence.weight, self._evidence.linear, self._evidence.square
        )
        lower = LowerBelief(wf, mean)
        observer = PrecisionObserver(
            wf,
            scatter,
            prior_shape=self.config["prior_shape"],
            prior_rate=self.config["prior_rate"],
        )
        readback = Port(
            "readback", lower, observer, "moments", (mean, base["variance"])
        )
        feedback = Port("feedback", observer, lower, "scalar", base["precision"])
        variance, tau = base["variance"], base["precision"]
        recovered = self._settle(self._evidence, start=(variance + 0.05, tau))
        lesions = {}
        for name in LESIONS:
            result = self._settle(self._evidence, lesion=name)
            lesions[name] = {
                key: result[key]
                for key in ("converged", "full_residual", "stationarity")
            }
        rate = base.get("stages", {}).get(2, self.config["prior_rate"])
        return {
            "observer_before": observer.emit(
                feedback, {"readback": (mean, variance), "meta_feedback": rate}
            ),
            "observer_after": observer.emit(
                feedback, {"readback": (mean, variance + 0.05), "meta_feedback": rate}
            ),
            "lower_before": lower.emit(readback, {"feedback": tau})[1],
            "lower_after": lower.emit(readback, {"feedback": tau + 0.1})[1],
            "recovered_qualified": recovered["converged"],
            "recovered_residual": recovered["full_residual"],
            "recovered_stationarity": recovered["stationarity"],
            "lesions": lesions,
        }

    def solve_factors(self, factors, edges):
        """Settle supplied exact binary cluster factors on a certified forest."""
        return solve_cluster_forest(factors, edges)


def _validate_clusters(factors):
    if not isinstance(factors, (list, tuple)) or not factors:
        raise ValueError("A nonempty factor list is required")
    clusters = []
    for spec in factors:
        if not isinstance(spec, dict):
            raise ValueError("Each factor must be a scope/values mapping")
        raw_scope = spec.get("scope", ())
        if not isinstance(raw_scope, (tuple, list)):
            raise ValueError("Factor scope must be a sequence of variables")
        scope = tuple(raw_scope)
        try:
            valid_scope = bool(scope) and len(set(scope)) == len(scope)
        except TypeError as error:
            raise ValueError("Factor variables must be hashable") from error
        if not valid_scope:
            raise ValueError("Factor scope must list distinct variables")
        values = spec.get("values")
        if (
            not isinstance(values, dict)
            or len(values) != 1 << len(scope)
            or any(
                not isinstance(k, tuple)
                or len(k) != len(scope)
                or any(type(bit) is not int or bit not in (0, 1) for bit in k)
                for k in values
            )
        ):
            raise ValueError(
                "Factor table must cover every binary assignment exactly once"
            )
        table = {}
        for key, raw in values.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, Fraction)):
                raise ValueError("Potentials must be exact integers or fractions")
            weight = Fraction(raw)
            if weight < 0:
                raise ValueError("Potentials must be nonnegative")
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

    if not isinstance(edges, (tuple, list)):
        raise ValueError("Edges must be a sequence of cluster pairs")
    for edge in edges:
        if not isinstance(edge, (tuple, list)):
            raise ValueError("Edges must join exactly two clusters")
        pair = tuple(edge)
        if len(pair) != 2:
            raise ValueError("Edges must join exactly two clusters")
        i, j = pair
        for index in (i, j):
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < count
            ):
                raise ValueError("Edge endpoint outside the cluster list")
        if i == j or frozenset((i, j)) in seen:
            raise ValueError("Degenerate or duplicate separator edge")
        if find(i) == find(j):
            raise ValueError(
                "Cyclic cluster graph; exactness is only certified on forests"
            )
        parent[find(i)] = find(j)
        seen.add(frozenset((i, j)))
        pairs.append((i, j))
    return pairs


def _check_running_intersection(clusters, pairs, separators):
    for variable in set().union(*(set(scope) for scope, _ in clusters)):
        holders = [k for k, (scope, _) in enumerate(clusters) if variable in scope]
        parent = {k: k for k in holders}

        def find(node, parent=parent):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for i, j in pairs:
            if variable in separators[(i, j)]:
                parent[find(i)] = find(j)
        if len({find(k) for k in holders}) > 1:
            raise ValueError(
                f"Running intersection violated for {variable!r}; exactness not certified"
            )


class ClusterPatch:
    """One finite column cluster: local exact table plus separator ports."""

    def __init__(self, scope, table):
        self.scope, self.table = scope, table
        self.positions = {v: i for i, v in enumerate(scope)}
        self.incoming = {}  # port name -> separator variables

    def emit(self, port, inbox):
        sep, exclude = port.meta["sep"], port.meta["exclude"]
        active = [
            (vars_, inbox[name])
            for name, vars_ in self.incoming.items()
            if name != exclude and name in inbox
        ]
        out = {a: Fraction(0) for a in product((0, 1), repeat=len(sep))}
        for assignment, weight in self.table.items():
            for vars_, message in active:
                weight = (
                    weight
                    * message[tuple(assignment[self.positions[v]] for v in vars_)]
                )
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
        name, reverse = f"{i}->{j}", f"{j}->{i}"
        sep = separators[(i, j)]
        ports.append(
            Port(
                name,
                patches[i],
                patches[j],
                "table",
                {a: Fraction(1) for a in product((0, 1), repeat=len(sep))},
                meta={"sep": sep, "exclude": reverse},
            )
        )
        patches[j].incoming[name] = sep
    result = settle(ports, budget=budget)
    if not result["converged"]:
        raise ValueError("Separator messages did not settle within the declared budget")
    messages = result["messages"]
    beliefs = []
    for patch in patches:
        values = {}
        for assignment, weight in patch.table.items():
            for name, vars_ in patch.incoming.items():
                weight = (
                    weight
                    * messages[name][
                        tuple(assignment[patch.positions[v]] for v in vars_)
                    ]
                )
            values[assignment] = weight
        mass = sum(values.values(), Fraction(0))
        if mass == 0:
            raise ValueError(
                "Zero joint support; the supplied constraints are contradictory"
            )
        beliefs.append(
            {"scope": patch.scope, "values": {a: w / mass for a, w in values.items()}}
        )
    return beliefs
