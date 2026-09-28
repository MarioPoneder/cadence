"""A configurable scalar column with recursively coupled precision observers.

Height one is the element's belief/precision loop. Extra stages read the live
proposal below and supply its rate: all stages settle in the same port graph.
A taller stack changes the model; depth alone promises neither better accuracy
nor biological fidelity. Admission and checkpoint custody are shared with the
base element rather than independently reimplemented.
"""

from __future__ import annotations

from types import MappingProxyType

from . import element as el

META_SHAPE = 4.0
META_RATE = 4.0
SCHEMA = "column-state/1"


def _rate_parameters(shape, rate):
    """Validate both supplied hyperparameters and their derived initial ratio."""
    shape = el._finite(shape, "meta_shape", positive=True)
    rate = el._finite(rate, "meta_rate", positive=True)
    ratio = el._finite(shape / rate, "initial meta precision", positive=True)
    return shape, rate, ratio


def _initial_rate(anchor, shape, rate):
    """Require a representable initial feedback message before building ports."""
    return el._finite(
        anchor + shape / (rate + 1.0), "initial meta feedback", positive=True
    )


class RateObserver:
    """Read a lower precision/rate and return anchor + shape/(rate_in+below).

    The positive anchor prevents the unanchored zero-dispersion degeneracy.
    ``shape`` and ``rate`` are supplied model parameters, not learned biology.
    """

    def __init__(
        self, anchor, read_key, up_key=None, *, shape=META_SHAPE, rate=META_RATE
    ):
        self.anchor = el._finite(anchor, "anchor", positive=True)
        self.shape, self.rate, _ = _rate_parameters(shape, rate)
        _initial_rate(self.anchor, self.shape, self.rate)
        if (
            not isinstance(read_key, str)
            or not read_key
            or (up_key is not None and (not isinstance(up_key, str) or not up_key))
        ):
            raise ValueError("Observer port keys must be nonempty strings")
        self.read_key, self.up_key = read_key, up_key

    def emit(self, port, inbox):
        below = el._finite(inbox[self.read_key], "lower proposal", positive=True)
        rate_in = inbox.get(self.up_key, self.rate) if self.up_key else self.rate
        rate_in = el._finite(rate_in, "incoming rate", positive=True)
        denominator = el._finite(rate_in + below, "total incoming rate", positive=True)
        return el._finite(
            self.anchor + self.shape / denominator, "outgoing rate", positive=True
        )


def stage_ports(
    observer,
    height,
    tau_init,
    *,
    prefix="",
    prior_rate=el.PRIOR_RATE,
    meta_shape=META_SHAPE,
    meta_rate=META_RATE,
):
    """Build stages 2..height; a prefix scopes names in a larger shared graph."""
    el._integer(height, "height", minimum=1)
    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    prior_rate = el._finite(prior_rate, "prior_rate", positive=True)
    meta_shape, meta_rate, meta_ratio = _rate_parameters(meta_shape, meta_rate)
    tau_init = el._finite(tau_init, "initial precision", positive=True)
    ports = []
    below, read_key = observer, prefix + "meta_readback"
    for stage in range(2, height + 1):
        down_key = (
            prefix + "meta_feedback" if stage == 2 else f"{prefix}hyper_feedback{stage}"
        )
        up_key = f"{prefix}hyper_feedback{stage + 1}" if stage < height else None
        anchor = prior_rate if stage == 2 else meta_rate
        rate = RateObserver(anchor, read_key, up_key, shape=meta_shape, rate=meta_rate)
        ports.append(
            el.Port(
                read_key,
                below,
                rate,
                "scalar",
                tau_init if stage == 2 else meta_ratio,
            )
        )
        ports.append(
            el.Port(
                down_key,
                rate,
                below,
                "scalar",
                _initial_rate(anchor, meta_shape, meta_rate),
            )
        )
        below, read_key = rate, f"{prefix}hyper_readback{stage + 1}"
    return ports


def stack_ports(
    w,
    s1,
    s2,
    height,
    start=None,
    *,
    prior_shape=el.PRIOR_SHAPE,
    prior_rate=el.PRIOR_RATE,
    meta_shape=META_SHAPE,
    meta_rate=META_RATE,
):
    """Compose the belief and every observer into one port graph."""
    el._integer(height, "height", minimum=1)
    ports = el.scalar_ports(
        w, s1, s2, start, prior_shape=prior_shape, prior_rate=prior_rate
    )
    if height > 1:
        ports += stage_ports(
            ports[0].target,
            height,
            ports[1].message,
            prior_rate=prior_rate,
            meta_shape=meta_shape,
            meta_rate=meta_rate,
        )
    return ports


def settle_stack(
    w,
    s1,
    s2,
    height,
    *,
    budget=el.MAX_SWEEPS,
    start=None,
    lesion=None,
    prior_shape=el.PRIOR_SHAPE,
    prior_rate=el.PRIOR_RATE,
    meta_shape=META_SHAPE,
    meta_rate=META_RATE,
    tolerance=el.RESIDUAL_TOL,
    damping=1.0,
):
    """Run the generic repair law over a composed column; report all stages."""
    ports = stack_ports(
        w,
        s1,
        s2,
        height,
        start,
        prior_shape=prior_shape,
        prior_rate=prior_rate,
        meta_shape=meta_shape,
        meta_rate=meta_rate,
    )
    result = el.settle(
        ports,
        budget=budget,
        lesion=lesion,
        tolerance=tolerance,
        damping=damping,
        lesion_precision=prior_shape / prior_rate,
    )
    mean, variance = result["messages"]["readback"]
    stages = {
        stage: result["messages"][
            "meta_feedback" if stage == 2 else f"hyper_feedback{stage}"
        ]
        for stage in range(2, height + 1)
    }
    return {
        "mean": mean,
        "variance": variance,
        "precision": result["messages"]["feedback"],
        "stages": stages,
        **{
            key: result[key]
            for key in (
                "sweeps",
                "converged",
                "executed_residual",
                "full_residual",
                "stationarity",
            )
        },
    }


class CorticalColumn(el.CorticalColumn):
    """A scalar column with ``height`` jointly settling observer stages.

    Parameters inherit :class:`cadence.element.CorticalColumn`: decay, capacity,
    value_bound, settle_budget, tolerance, damping, prior_weight, prior_shape,
    prior_rate and max_statistic_bits. Height defaults to 1; meta_shape=4 and
    meta_rate=4 configure additional rate observers at heights 2 and above.
    ``config`` is read-only and fully bound into checkpoints.

    Example::

        column = CorticalColumn(height=2, decay=0.95, value_bound=100)
        column.add(12.5)
        result = column.query()
        if result['qualified']:
            print(result['answer'], result['variance'])
    """

    _schema = SCHEMA

    def __init__(
        self,
        height=1,
        *,
        decay=el.DECAY,
        capacity=None,
        value_bound=el.VALUE_BOUND,
        settle_budget=el.MAX_SWEEPS,
        tolerance=el.RESIDUAL_TOL,
        damping=1.0,
        prior_weight=el.PRIOR_WEIGHT,
        prior_shape=el.PRIOR_SHAPE,
        prior_rate=el.PRIOR_RATE,
        meta_shape=META_SHAPE,
        meta_rate=META_RATE,
        max_statistic_bits=el.MAX_STATISTIC_BITS,
    ):
        el._integer(height, "height", minimum=1)
        super().__init__(
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
        meta_shape, meta_rate, _ = _rate_parameters(meta_shape, meta_rate)
        if height > 1:
            _initial_rate(self.config["prior_rate"], meta_shape, meta_rate)
        if height > 2:
            _initial_rate(meta_rate, meta_shape, meta_rate)
        self._config = MappingProxyType(
            {
                **self.config,
                "height": height,
                "meta_shape": meta_shape,
                "meta_rate": meta_rate,
            }
        )

    @property
    def height(self):
        """Number of coupled observer stages (read-only)."""
        return self.config["height"]

    def _settle(self, evidence, *, budget=None, start=None, lesion=None):
        return settle_stack(
            evidence.weight,
            evidence.linear,
            evidence.square,
            self.height,
            budget=self.config["settle_budget"] if budget is None else budget,
            start=start,
            lesion=lesion,
            prior_shape=self.config["prior_shape"],
            prior_rate=self.config["prior_rate"],
            meta_shape=self.config["meta_shape"],
            meta_rate=self.config["meta_rate"],
            tolerance=self.config["tolerance"],
            damping=self.config["damping"],
        )

    def query(self):
        """Return the settled scalar belief and stage rates without admitting data."""
        result = self._settle(self._evidence)
        return {
            "answer": result["mean"],
            "variance": result["variance"],
            "precision": result["precision"],
            "stages": result["stages"],
            "qualified": result["converged"],
            "residual": result["full_residual"],
        }

    def observer_intervention(self):
        """Run readback, feedback and recovery diagnostics without memory changes."""
        result = super().observer_intervention()
        base = self._settle(self._evidence)
        return {**result, "height": self.height, "stages": base["stages"]}
