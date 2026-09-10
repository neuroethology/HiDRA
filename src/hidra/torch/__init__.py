"""PyTorch backend for HiDRA.

A faithful re-implementation of the JAX model in `hidra.solution` /
`hidra.train_perlab_heads`, plus the tooling to move the published checkpoints across.

The JAX code is a hand-rolled framework, so nothing here maps onto stock `torch.nn`
layers: `Linear` carries both weight normalization and a frozen running input
standardization, and `LSTM` has learned initial states and its own gate ordering. The port
reproduces those exactly rather than substituting `torch.nn.Linear`/`torch.nn.LSTM`, which
is what makes bit-level comparison against the reference possible.

    from hidra.torch import load_unsupervised, load_perlab

    trunk = load_unsupervised("15fps_5bp", device="cuda")
    head = load_perlab("15fps_5bp", trunk, device="cuda")
    probs = head.predict(batch)          # (B, seq_len, len(ACTIONS))
"""
from .checkpoint import load_jax_checkpoint, read_state, save_state
from .layers import LSTM, BidirectionalLSTM, Constant, Embedding, Linear
from .models import MultiTaskPerLabModel, UnsupervisedModel, load_perlab, load_unsupervised

__all__ = [
    "BidirectionalLSTM",
    "Constant",
    "Embedding",
    "Linear",
    "LSTM",
    "MultiTaskPerLabModel",
    "UnsupervisedModel",
    "load_jax_checkpoint",
    "load_perlab",
    "load_unsupervised",
    "read_state",
    "save_state",
]
