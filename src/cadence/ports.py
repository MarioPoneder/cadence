"""Input ports of a record patch: how the inputs reach the context channels.

The dense port is the record patch as published: every context channel reads every input
through its own row. A structured port reads a declared layout of the inputs through
blocks: a *map* block is a tied local kernel over a grid of the input (a convolution, the
same kernel at every position, so a thing is the same thing wherever it appears), whose
output is a grid of channels laid out as retinotopic maps; a *dense* block is a plain
matrix over a slice of the inputs. The patch's equations do not change: the port is the
linear map ``u -> B u`` (and ``u -> G u`` for the gate), and the adjoint scan needs its
transpose and its parameter gradient, which every block supplies. Blocks are the genes a
genome sets: which slice, what grid, how many channels, what kernel, what stride.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _dimension(name: str, value: Any, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class MapBlock:
    """A tied local kernel over a ``(channels_in, height, width)`` grid of the inputs."""

    start: int  # first input of the grid (channel-major, row-major)
    channels_in: int
    height: int
    width: int
    channels_out: int
    kernel: int
    stride: int = 1

    def __post_init__(self) -> None:
        _dimension("start", self.start, 0)
        for name in ("channels_in", "height", "width", "channels_out", "kernel", "stride"):
            _dimension(name, getattr(self, name))
        if self.kernel > min(self.height, self.width):
            raise ValueError("kernel must fit inside the input grid")

    @property
    def inputs(self) -> int:
        return self.channels_in * self.height * self.width

    @property
    def out_height(self) -> int:
        return (self.height - self.kernel) // self.stride + 1

    @property
    def out_width(self) -> int:
        return (self.width - self.kernel) // self.stride + 1

    @property
    def outputs(self) -> int:
        return self.channels_out * self.out_height * self.out_width

    @property
    def weights(self) -> tuple[int, ...]:
        return (self.channels_out, self.channels_in, self.kernel, self.kernel)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "map",
            "start": self.start,
            "channels_in": self.channels_in,
            "height": self.height,
            "width": self.width,
            "channels_out": self.channels_out,
            "kernel": self.kernel,
            "stride": self.stride,
        }


@dataclass(frozen=True)
class DenseBlock:
    """A plain matrix over a slice of the inputs."""

    start: int
    inputs: int
    outputs: int

    def __post_init__(self) -> None:
        _dimension("start", self.start, 0)
        _dimension("inputs", self.inputs)
        _dimension("outputs", self.outputs)

    @property
    def weights(self) -> tuple[int, ...]:
        return (self.outputs, self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "dense",
            "start": self.start,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


Block = MapBlock | DenseBlock


def block_from_dict(d: dict[str, Any]) -> Block:
    d = dict(d)
    kind = d.pop("kind")
    if kind not in ("map", "dense"):
        raise ValueError(f"unknown port block kind {kind!r}")
    return MapBlock(**d) if kind == "map" else DenseBlock(**d)


class StructuredPort:
    """``B u`` as a sum of blocks, each reading its slice of ``u`` into its slice of the output.

    ``weights`` is a list of arrays, one per block, in the block order. The port is linear
    in ``u`` and in each block's weights; ``apply``, ``transpose`` and ``gradient`` are the
    three operations the adjoint scan needs. ``broadcast`` names a slice ``(start, count)``
    of the inputs that every map block also reads as constant channels tiled over its grid,
    so what a body did and where it is reach every position before the nonlinearity: the
    effect of an action can then depend on what is where. A map block's kernel then has
    ``channels_in + count`` input channels. ``mask`` names which inputs the port hears, one
    flag per input: a masked input is zero to every block, in the forward map, its
    transpose and the gradient, so which channels a patch reads can be a gene without
    touching the weights (a steering patch that reads the surprises and not the probes)."""

    def __init__(
        self,
        inputs: int,
        blocks: list[Block],
        broadcast: tuple[int, int] | None = None,
        mask: Any = None,
    ) -> None:
        self.inputs = _dimension("inputs", inputs)
        self.blocks = list(blocks)
        if not self.blocks or not all(isinstance(b, (DenseBlock, MapBlock)) for b in self.blocks):
            raise ValueError("a port needs at least one dense or map block")
        for b in self.blocks:
            if b.start < 0 or b.start + b.inputs > self.inputs:
                raise ValueError("a block reads outside the inputs")
        if broadcast is not None and len(broadcast) != 2:
            raise ValueError("broadcast must be a (start, count) pair")
        self.broadcast = (
            None
            if broadcast is None
            else (
                _dimension("broadcast start", broadcast[0], 0),
                _dimension("broadcast count", broadcast[1]),
            )
        )
        if self.broadcast is not None:
            s0, count = self.broadcast
            if count < 1 or s0 < 0 or s0 + count > self.inputs:
                raise ValueError("the broadcast slice must lie inside the inputs")
        self.mask: np.ndarray | None = None
        if mask is not None:
            flags = np.asarray(mask)
            if flags.shape != (self.inputs,):
                raise ValueError("mask must have one flag per input")
            if not np.isin(flags, [False, True]).all():
                raise ValueError("mask must contain only boolean or 0/1 flags")
            self.mask = flags.astype(bool)
        self.outputs = int(sum(b.outputs for b in self.blocks))
        self._offsets = np.cumsum([0] + [b.outputs for b in self.blocks])

    def _heard(self, u: np.ndarray) -> np.ndarray:
        return u if self.mask is None else u * self.mask

    def weight_shape(self, b: Block) -> tuple[int, ...]:
        """A block's kernel shape, with the broadcast channels added for a map block."""
        if isinstance(b, DenseBlock) or self.broadcast is None:
            return b.weights
        return (b.channels_out, b.channels_in + self.broadcast[1], b.kernel, b.kernel)

    def initial(self, rng: np.random.Generator, scale: float = 1.0) -> list[np.ndarray]:
        """Fan-in scaled draws per block (zero with ``scale`` 0, for a gate port)."""
        out = []
        for b in self.blocks:
            shape = self.weight_shape(b)
            fan = b.inputs if isinstance(b, DenseBlock) else shape[1] * b.kernel * b.kernel
            out.append(
                rng.normal(size=shape) * (scale / np.sqrt(fan)) if scale else np.zeros(shape)
            )
        return out

    def _grid(self, u: np.ndarray, b: MapBlock) -> np.ndarray:
        """A map block's input grid ``(..., cin [+ count], H, W)`` with the broadcast tiled in."""
        lead = u.shape[:-1]
        grid = u[..., b.start : b.start + b.inputs].reshape(*lead, b.channels_in, b.height, b.width)
        if self.broadcast is None:
            return grid
        s0, count = self.broadcast
        tiled = np.broadcast_to(
            u[..., s0 : s0 + count][..., :, None, None], (*lead, count, b.height, b.width)
        )
        return np.concatenate([grid, tiled], axis=-3)

    # ------------------------------------------------------------------ the three maps
    def apply(self, u: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
        """``(..., inputs) -> (..., outputs)``."""
        u = self._heard(u)
        lead = u.shape[:-1]
        parts = []
        for b, w in zip(self.blocks, weights, strict=True):
            if isinstance(b, DenseBlock):
                parts.append(u[..., b.start : b.start + b.inputs] @ w.T)
            else:
                parts.append(self._conv(self._grid(u, b), b, w).reshape(*lead, -1))
        return np.concatenate(parts, axis=-1)

    def transpose(self, v: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
        """``(..., outputs) -> (..., inputs)``: the gradient with respect to the inputs."""
        lead = v.shape[:-1]
        out = np.zeros((*lead, self.inputs))
        for k, (b, w) in enumerate(zip(self.blocks, weights, strict=True)):
            y = v[..., self._offsets[k] : self._offsets[k + 1]]
            if isinstance(b, DenseBlock):
                out[..., b.start : b.start + b.inputs] += y @ w
            else:
                back = self._conv_transpose(
                    y.reshape(*lead, b.channels_out, b.out_height, b.out_width), b, w
                )
                out[..., b.start : b.start + b.inputs] += back[..., : b.channels_in, :, :].reshape(
                    *lead, -1
                )
                if self.broadcast is not None:
                    s0, count = self.broadcast
                    out[..., s0 : s0 + count] += back[..., b.channels_in :, :, :].sum(axis=(-2, -1))
        return self._heard(out)

    def gradient(self, v: np.ndarray, u: np.ndarray) -> list[np.ndarray]:
        """The gradient of ``sum(v * (B u))`` with respect to each block's weights, summed over
        every leading axis (batch and time)."""
        n = int(np.prod(u.shape[:-1])) if u.ndim > 1 else 1
        uf, vf = self._heard(u).reshape(n, -1), v.reshape(n, -1)
        out = []
        for k, b in enumerate(self.blocks):
            y = vf[:, self._offsets[k] : self._offsets[k + 1]]
            if isinstance(b, DenseBlock):
                out.append(y.T @ uf[:, b.start : b.start + b.inputs])
            else:
                patches = self._patches(self._grid(uf, b), b)  # (n, oh, ow, cin [+ count], k, k)
                yy = y.reshape(n, b.channels_out, b.out_height, b.out_width)
                out.append(np.einsum("nohw,nhwikl->oikl", yy, patches))
        return out

    # ------------------------------------------------------------------ convolution
    @staticmethod
    def _patches(x: np.ndarray, b: MapBlock) -> np.ndarray:
        """``(..., cin, H, W) -> (..., oh, ow, cin, k, k)`` windows at the stride."""
        windows = sliding_window_view(
            x, (b.kernel, b.kernel), axis=(-2, -1)
        )  # (..., cin, H-k+1, W-k+1, k, k)
        windows = windows[..., :: b.stride, :: b.stride, :, :]
        return np.moveaxis(windows, -5, -3)  # (..., oh, ow, cin, k, k)

    def _conv(self, x: np.ndarray, b: MapBlock, w: np.ndarray) -> np.ndarray:
        patches = self._patches(x, b)
        return np.asarray(np.einsum("...hwikl,oikl->...ohw", patches, w))

    @staticmethod
    def _conv_transpose(y: np.ndarray, b: MapBlock, w: np.ndarray) -> np.ndarray:
        """Scatter each output's kernel back onto the input grid (the correlation's adjoint);
        the grid has the kernel's input channels, broadcast channels included."""
        lead = y.shape[:-3]
        out = np.zeros((*lead, w.shape[1], b.height, b.width))
        contributions = np.einsum("...ohw,oikl->...hwikl", y, w)  # (..., oh, ow, cin, k, k)
        for i in range(b.kernel):
            for j in range(b.kernel):
                rows = slice(i, i + b.stride * b.out_height, b.stride)
                cols = slice(j, j + b.stride * b.out_width, b.stride)
                out[..., :, rows, cols] += np.moveaxis(contributions[..., i, j], -1, -3)
        return out

    # ------------------------------------------------------------------ custody
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"inputs": self.inputs, "blocks": [b.to_dict() for b in self.blocks]}
        if self.broadcast is not None:
            out["broadcast"] = list(self.broadcast)
        if self.mask is not None:
            out["mask"] = [int(v) for v in self.mask]
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructuredPort:
        cast = d.get("broadcast")
        return cls(
            d["inputs"],
            [block_from_dict(b) for b in d["blocks"]],
            None if cast is None else tuple(cast),
            d.get("mask"),
        )

    def dense_matrix(self, weights: list[np.ndarray]) -> np.ndarray:
        """The equivalent ``(outputs, inputs)`` matrix, for tests and small ports."""
        eye = np.eye(self.inputs)
        return self.apply(eye, weights).T
