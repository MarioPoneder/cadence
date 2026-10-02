"""Assemble named regions into one connectome for one ``Brain``."""

from collections.abc import Mapping, Sequence

from ..connectome import Connectome


def assemble(
    regions: Mapping[str, Connectome],
    synapses: Sequence[tuple[str, int, str, int, float]] = (),
) -> Connectome:
    """Merge regions plus directed synapses between them, ``(source, i, target, j, weight)``.

    Region names identify complete neuron populations; child populations become
    ``region/population``. Pass the returned connectome to a single ``Brain``, with one
    concatenated drive and state. Regions do not run independently. Convergence/optimality requires
    properties of the combined system, not merely each isolated region.
    """
    if not regions or any(not name or "/" in name for name in regions):
        raise ValueError("nonempty region names without '/' required")
    offsets: dict[str, int] = {}
    populations: dict[str, tuple[int, ...]] = {}
    pre: list[int] = []
    post: list[int] = []
    count: list[float] = []
    sign: list[float] = []
    n = 0
    for name, connectome in regions.items():
        offsets[name] = n
        populations[name] = tuple(range(n, n + connectome.n))
        populations.update(
            {
                f"{name}/{key}": tuple(n + i for i in ids)
                for key, ids in connectome.populations.items()
            }
        )
        pre.extend(n + int(i) for i in connectome.pre)
        post.extend(n + int(i) for i in connectome.post)
        count.extend(float(v) for v in connectome.count)
        sign.extend(float(v) for v in connectome.sign)
        n += connectome.n
    for a, i, b, j, weight in synapses:
        if a not in regions or b not in regions:
            raise ValueError("a synapse names an unknown region")
        if (
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < regions[a].n
            or isinstance(j, bool)
            or not isinstance(j, int)
            or not 0 <= j < regions[b].n
        ):
            raise ValueError("a synapse names a neuron outside its region")
        if offsets[a] + i == offsets[b] + j:
            raise ValueError("autapses are not supported")
        pre.append(offsets[a] + i)
        post.append(offsets[b] + j)
        count.append(1.0)
        sign.append(weight)
    return Connectome.from_synapses(
        n,
        pre=pre,
        post=post,
        count=count,
        sign=sign,
        populations=populations,
        label="assembled-brain",
    )
