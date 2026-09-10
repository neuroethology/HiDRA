"""PyTorch equivalents of the hand-rolled JAX layers in `hidra.solution`.

Each class mirrors one JAX class, parameter for parameter and operation for operation, so
a converted checkpoint drops straight in and the outputs can be compared numerically. The
naming of parameters and buffers matches the JAX variable names (`w`, `s`, `b`, `m1`, `m2`,
`n`, `mean`, `std`, `c`) so the checkpoint mapping is the identity.

Two things here are unusual enough to call out, because they are why stock `torch.nn`
layers cannot be substituted:

`Linear` is three operations, not one.
    1. A frozen input standardization: `x = (x - mean) / std`, from running moments
       accumulated during a data-dependent-init pass and then held fixed. Not a LayerNorm
       -- the statistics are per-input-feature and global, not per-token.
    2. Weight normalization: the stored `w` is L2-normalized over the input axis and
       rescaled by a learned per-output gain `s`. The effective weight is
       `w / ||w||_in * s`, recomputed from the parameters rather than stored.
    3. The matmul itself, as `einsum("...i,...io->...o")` so that `batch_dims` give
       independent weights per bodypart (or per bodypart pair) rather than one shared
       projection.

`LSTM` has learned initial states (`h0` passed through tanh, `c0` raw), gates ordered
`i, f, c, o` as contiguous quarters of the input projection's output, and the input
projection carries the standardization statistics above.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# `context["stage"]` in the JAX code. "eval" uses the frozen statistics; "init" is the
# data-dependent-init pass that accumulates them.
STAGE_EVAL = "eval"
STAGE_INIT = "init"
STAGE_TRAIN = "train"


class HidraModule(nn.Module):
    """Base class carrying the `stage` context the JAX `set_context` mechanism provided."""

    def __init__(self):
        super().__init__()
        self.stage = STAGE_EVAL
        self.init_decay = 0.99

    def set_stage(self, stage, init_decay=None):
        """Set `stage` on this module and every HidraModule beneath it."""
        for module in self.modules():
            if isinstance(module, HidraModule):
                module.stage = stage
                if init_decay is not None:
                    module.init_decay = init_decay
        return self


class Linear(HidraModule):
    """`solution.Linear`: input standardization + weight normalization + batched matmul."""

    def __init__(self, input_dim, output_dim, use_bias=True, batch_dims=(), init_bias=0.0,
                 normalize_input=True, dtype=torch.float32, device=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.batch_dims = tuple(int(d) for d in batch_dims)
        self.use_bias = bool(use_bias)
        self.normalize_input = bool(normalize_input)
        self.compute_dtype = dtype
        fkw = dict(dtype=torch.float32, device=device)

        # Initialization matches solution.Linear.create_variables: w ~ N(0,1) scaled down by
        # sqrt(max(in, out))/sqrt(64), s = 1, b = init_bias.
        dim = max(self.input_dim, self.output_dim)
        default_lrmul = math.sqrt(dim) / math.sqrt(64)
        w = torch.randn(*self.batch_dims, self.input_dim, self.output_dim, **fkw) / default_lrmul
        self.w = nn.Parameter(w)
        self.s = nn.Parameter(torch.ones(*self.batch_dims, self.output_dim, **fkw))
        if self.use_bias:
            bias = torch.as_tensor(init_bias, dtype=torch.float32, device=device)
            self.b = nn.Parameter(bias.expand(*self.batch_dims, self.output_dim).clone())
        else:
            self.register_parameter("b", None)

        if self.normalize_input:
            stat_shape = (*self.batch_dims, self.input_dim)
            self.register_buffer("m1", torch.zeros(stat_shape, **fkw))
            self.register_buffer("m2", torch.zeros(stat_shape, **fkw))
            self.register_buffer("n", torch.zeros((), **fkw))
            self.register_buffer("mean", torch.zeros(stat_shape, **fkw))
            self.register_buffer("std", torch.ones(stat_shape, **fkw))

    def effective_weight(self):
        """`w / ||w||_input * s`, i.e. `solution.Linear.get_weights`'s "w".

        The JAX code computes this once when variables are bound (`ParameterizedLayer.
        __init__`). Recomputing it per call gives the identical value -- it is a
        deterministic function of the parameters -- and keeps the gradient path intact for
        training.
        """
        norm = torch.linalg.vector_norm(self.w, dim=-2, keepdim=True)
        return ((self.w / norm) * self.s.unsqueeze(-2)).to(self.compute_dtype)

    def _update_input_stats(self, x):
        """The `stage == "init"` branch: accumulate decayed moments, then derive mean/std."""
        decay = self.init_decay
        leading = tuple(range(x.dim() - len(self.batch_dims) - 1))
        m1 = x.mean(dim=leading)
        m2 = (x ** 2).mean(dim=leading)
        self.m1 += (m1 - self.m1) * (1 - decay)
        self.m2 += (m2 - self.m2) * (1 - decay)
        self.n += 1
        # Bias-correct the decayed averages the same way solution.Linear.apply does.
        norm = 1.0 / (1 - decay ** self.n)
        m1c = self.m1 * norm
        m2c = self.m2 * norm
        self.std.copy_(torch.sqrt(m2c - m1c ** 2) + 1e-4)
        self.mean.copy_(m1c)

    def forward(self, x):
        if self.normalize_input:
            if self.stage == STAGE_INIT:
                with torch.no_grad():
                    self._update_input_stats(x.detach().float())
            x = (x - self.mean) / self.std
            x = x.to(self.compute_dtype)
        y = torch.einsum("...i,...io->...o", x, self.effective_weight())
        if self.use_bias:
            y = y + self.b.to(y.dtype)
        return y

    def extra_repr(self):
        return (f"{self.input_dim}->{self.output_dim}, batch_dims={self.batch_dims}, "
                f"bias={self.use_bias}, normalize_input={self.normalize_input}")


class Constant(HidraModule):
    """`solution.Constant`: a learned tensor with no inputs."""

    def __init__(self, shape, batch_dims=(), dtype=torch.float32, device=None):
        super().__init__()
        self.shape = tuple(int(d) for d in shape)
        self.batch_dims = tuple(int(d) for d in batch_dims)
        self.compute_dtype = dtype
        self.c = nn.Parameter(torch.randn(*self.batch_dims, *self.shape,
                                          dtype=torch.float32, device=device))

    def forward(self):
        return self.c.to(self.compute_dtype)

    def extra_repr(self):
        return f"shape={self.batch_dims + self.shape}"


class Embedding(HidraModule):
    """`solution.Embedding`: a lookup table, indexed directly (no padding_idx, no norms)."""

    def __init__(self, input_dim, cardinality, dtype=torch.float32, device=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.cardinality = int(cardinality)
        self.compute_dtype = dtype
        self.w = nn.Parameter(torch.randn(self.cardinality, self.input_dim,
                                          dtype=torch.float32, device=device))

    def forward(self, indices):
        return self.w.to(self.compute_dtype)[indices.long()]

    def extra_repr(self):
        return f"cardinality={self.cardinality}, dim={self.input_dim}"


class LSTM(HidraModule):
    """`solution.LSTM`: learned initial state, gates ordered i/f/c/o, optional batch_dims.

    Not interchangeable with `torch.nn.LSTM`, which uses i/f/g/o with two bias vectors,
    zero initial state, and no per-bodypart weight sets.
    """

    def __init__(self, input_dim, hidden_dim, forget_bias=0.0, batch_dims=(),
                 dtype=torch.float32, device=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.batch_dims = tuple(int(d) for d in batch_dims)
        self.compute_dtype = dtype

        # Bias only the forget gate, the second quarter of the gate vector.
        init_bias = 0.0
        if forget_bias:
            init_bias = torch.zeros(4 * self.hidden_dim, dtype=torch.float32, device=device)
            init_bias[self.hidden_dim:2 * self.hidden_dim] = forget_bias

        self.is_linear = Linear(self.input_dim, 4 * self.hidden_dim, use_bias=True,
                                batch_dims=self.batch_dims, init_bias=init_bias,
                                normalize_input=True, dtype=dtype, device=device)
        self.ss_linear = Linear(self.hidden_dim, 4 * self.hidden_dim, use_bias=False,
                                batch_dims=self.batch_dims, normalize_input=False,
                                dtype=dtype, device=device)
        self.h0 = Constant((self.hidden_dim,), batch_dims=self.batch_dims, dtype=dtype, device=device)
        self.c0 = Constant((self.hidden_dim,), batch_dims=self.batch_dims, dtype=dtype, device=device)

    def forward(self, x):
        """x: (tsteps, batch, *batch_dims, input_dim) -> (tsteps, batch, *batch_dims, hidden)."""
        tsteps, bs = x.shape[0], x.shape[1]

        h = torch.tanh(self.h0())
        c = self.c0()
        # tile over the batch axis, as solution.LSTM.apply does
        reps = (bs,) + (1,) * (len(self.batch_dims) + 1)
        h = h.unsqueeze(0).repeat(*reps)
        c = c.unsqueeze(0).repeat(*reps)

        xs = self.is_linear(x)
        hs = []
        for t in range(tsteps):
            gates = xs[t] + self.ss_linear(h)
            i, f, g, o = torch.split(gates, self.hidden_dim, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            hs.append(h)
        return torch.stack(hs, dim=0)

    def extra_repr(self):
        return f"{self.input_dim}->{self.hidden_dim}, batch_dims={self.batch_dims}"


class BidirectionalLSTM(HidraModule):
    """`solution.BidirectionalLSTM`: two independent LSTMs, outputs concatenated."""

    def __init__(self, input_dim, hidden_dim, forget_bias=0.0, batch_dims=(),
                 dtype=torch.float32, device=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        kw = dict(forget_bias=forget_bias, batch_dims=batch_dims, dtype=dtype, device=device)
        # Attribute names must match the JAX layer keys ("lstm-fw"/"lstm-bw") after the
        # '-'->'_' substitution the checkpoint mapping applies.
        self.lstm_fw = LSTM(input_dim, hidden_dim, **kw)
        self.lstm_bw = LSTM(input_dim, hidden_dim, **kw)

    def forward(self, x):
        h_fw = self.lstm_fw(x)
        h_bw = self.lstm_bw(torch.flip(x, dims=(0,)))
        h_bw = torch.flip(h_bw, dims=(0,))
        return torch.cat([h_fw, h_bw], dim=-1)


def silu(x):
    """`jax.nn.silu` == x * sigmoid(x). Matches torch's, but named for symmetry with the
    reference so the ported forward passes read line-for-line against it."""
    return F.silu(x)
