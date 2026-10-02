"""The belief patch on torch: the same slow half for learning at scale, with export to the
library's patch, which keeps the records, the night and the custody.

``TorchBelief`` holds the encoder port, the transition, the repair map and the readout of a
``BeliefPatch`` as a module and computes the same forward pass with autograd for the
gradient (the NumPy adjoint is the reference; a parity test checks both). The record read
is supplied as given (zero, or a read taken from a NumPy patch), never learned through.
The forward takes the library's ``gains`` per observation-port block and the per-row
``observed`` mask; a weighted loss, an admitted step and an external output gradient are
the caller's loss and optimiser here, and the readback of a moment stays on the library's
path. Batches of streams train with an ordinary optimiser; ``export`` returns the
six-plus-four arrays in the library's packing and ``load`` takes them.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .ports import DenseBlock, StructuredPort


class TorchPort(torch.nn.Module):
    """A structured port's ``apply`` in torch, broadcast channels included."""

    def __init__(self, port: StructuredPort) -> None:
        super().__init__()
        self.port = port
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(port.weight_shape(b))) for b in port.blocks]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # (N, inputs) -> (N, outputs)
        parts = []
        for b, w in zip(self.port.blocks, self.weights, strict=True):
            if isinstance(b, DenseBlock):
                parts.append(u[:, b.start : b.start + b.inputs] @ w.T)
                continue
            grid = u[:, b.start : b.start + b.inputs].reshape(-1, b.channels_in, b.height, b.width)
            if self.port.broadcast is not None:
                s0, count = self.port.broadcast
                tiled = u[:, s0 : s0 + count][:, :, None, None].expand(-1, -1, b.height, b.width)
                grid = torch.cat([grid, tiled], dim=1)
            parts.append(F.conv2d(grid, w, stride=b.stride).flatten(1))
        return torch.cat(parts, dim=1)

    def packed(self) -> np.ndarray:
        return np.concatenate([w.detach().cpu().double().numpy().ravel() for w in self.weights])

    def load_packed(self, flat: np.ndarray) -> None:
        at = 0
        for b, w in zip(self.port.blocks, self.weights, strict=True):
            shape = self.port.weight_shape(b)
            size = int(np.prod(shape))
            w.data.copy_(torch.as_tensor(flat[at : at + size].reshape(shape), dtype=w.dtype))
            at += size


class TorchBelief(torch.nn.Module):
    unit_block: torch.Tensor

    def __init__(
        self,
        port: StructuredPort,
        actions: int,
        belief: int,
        outputs: int,
        *,
        iterations: int = 2,
        damping: float = 0.5,
        record_width: int = 64,
    ) -> None:
        super().__init__()
        self.port = port
        self.encoded, self.actions, self.belief, self.outputs = (
            port.outputs,
            int(actions),
            int(belief),
            int(outputs),
        )
        self.iterations, self.damping, self.record_width = (
            int(iterations),
            float(damping),
            int(record_width),
        )
        self.block_count = len(port.blocks)
        za = self.belief + self.actions
        fi = self.belief + self.encoded + self.belief + self.record_width + 1
        dt = torch.get_default_dtype()
        self.E = TorchPort(port)
        self.e_b = torch.nn.Parameter(torch.zeros(self.encoded, dtype=dt))
        self.T = torch.nn.Parameter(torch.zeros(self.belief, za, dtype=dt))
        self.t_b = torch.nn.Parameter(torch.zeros(self.belief, dtype=dt))
        self.G = torch.nn.Parameter(torch.zeros(self.belief, za, dtype=dt))
        self.g_b = torch.nn.Parameter(torch.ones(self.belief, dtype=dt))
        self.F = torch.nn.Parameter(torch.zeros(self.belief, fi, dtype=dt))
        self.f_b = torch.nn.Parameter(torch.zeros(self.belief, dtype=dt))
        self.C = torch.nn.Parameter(torch.zeros(self.outputs, self.belief, dtype=dt))
        self.c = torch.nn.Parameter(torch.zeros(self.outputs, dtype=dt))
        self.register_buffer(
            "unit_block",
            torch.as_tensor(
                np.concatenate([np.full(b.outputs, i) for i, b in enumerate(port.blocks)]),
                dtype=torch.long,
            ),
        )

    def expect(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, a], dim=-1)
        g = torch.sigmoid(x @ self.G.T + self.g_b)
        return g * z + (1.0 - g) * torch.tanh(x @ self.T.T + self.t_b)

    def repair(
        self, e: torch.Tensor, p: torch.Tensor, read: torch.Tensor, flag: torch.Tensor | None = None
    ) -> torch.Tensor:
        z = p
        if flag is None:
            flag = torch.ones(p.shape[0], 1, dtype=p.dtype, device=p.device)
        for _ in range(self.iterations):
            u = torch.cat([z, e, p, read, flag], dim=-1)
            z = (1.0 - self.damping) * z + self.damping * torch.tanh(u @ self.F.T + self.f_b)
        return z

    def _gains(self, gains: torch.Tensor, n: int, t: int) -> torch.Tensor:
        b = self.block_count
        g = torch.as_tensor(gains)
        if g.shape == (b,):
            return g[None, None, :].expand(n, t, b)
        if g.shape == (n, b):
            return g[:, None, :].expand(n, t, b)
        if g.shape != (n, t, b):
            raise ValueError(f"gains must have shape ({b},), (batch, {b}) or (batch, time, {b})")
        return g

    def forward(
        self,
        observations: torch.Tensor | None,
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
        reads: torch.Tensor | None = None,
        gains: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, T, inputs)`` observations (None to imagine) and ``(N, T, actions)`` give the
        beliefs and the slow outputs.

        ``reads`` supplies the store's coded read per moment ``(N, T, record_width)``; absent,
        the read is zero (a store that holds nothing yet). ``gains`` weighs each block's
        encoded evidence inside the repair, ``(blocks,)``, ``(N, blocks)`` or
        ``(N, T, blocks)``; a tensor that requires grad receives the gradient into the gains.
        ``observed`` ``(N, T)`` masks rows: a row that observes nothing keeps the expectation."""
        n, t = actions.shape[:2]
        z = (
            torch.zeros(n, self.belief, dtype=actions.dtype, device=actions.device)
            if state is None
            else state
        )
        g = None if gains is None else self._gains(gains, n, t)
        rows = None
        if observed is not None:
            rows = torch.as_tensor(observed, dtype=torch.bool, device=actions.device)
            if rows.shape != (n, t):
                raise ValueError("observed must be a (batch, time) mask")
        zs, ys = [], []
        for k in range(t):
            p = self.expect(z, actions[:, k])
            if observations is None:
                z = p
            else:
                e = torch.tanh(self.E(observations[:, k]) + self.e_b)
                flag = None
                if rows is not None:
                    row = rows[:, k][:, None]
                    flag = row.to(e.dtype)
                    e = e * flag
                if g is not None:
                    e = e * g[:, k][:, self.unit_block]
                read = (
                    torch.zeros(n, self.record_width, dtype=z.dtype, device=z.device)
                    if reads is None
                    else reads[:, k]
                )
                repaired = self.repair(e, p, read, flag)
                z = repaired if rows is None else torch.where(row, repaired, p)
            zs.append(z)
            ys.append(z @ self.C.T + self.c)
        return torch.stack(zs, dim=1), torch.stack(ys, dim=1)

    def export(self) -> dict[str, np.ndarray]:
        out = {
            k: getattr(self, k).detach().cpu().double().numpy()
            for k in ("e_b", "T", "t_b", "G", "g_b", "F", "f_b", "C", "c")
        }
        out["E"] = self.E.packed()
        return out

    def load(self, params: dict[str, np.ndarray]) -> None:
        self.E.load_packed(params["E"])
        with torch.no_grad():
            for k in ("e_b", "T", "t_b", "G", "g_b", "F", "f_b", "C", "c"):
                getattr(self, k).copy_(torch.as_tensor(params[k], dtype=getattr(self, k).dtype))
