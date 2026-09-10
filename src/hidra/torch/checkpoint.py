"""Torch-flavoured checkpoint IO.

The format handling itself is backend-neutral and lives in `hidra.checkpoints`; this module
adds the two things that need torch -- reading straight into tensors on a device, and
writing from tensors -- and re-exports the rest so `hidra.torch.checkpoint` remains a
complete entry point.
"""
from pathlib import Path

import numpy as np

from ..checkpoints import (  # noqa: F401
    flatten_tree,
    load_checkpoint,
    load_flat_checkpoint,
    load_pickle_checkpoint,
    read_metadata,
    resolve_checkpoint,
    save_safetensors,
    unflatten_tree,
)

# The historical name for the pickle reader, kept so existing callers and docs still work.
load_jax_checkpoint = load_pickle_checkpoint


def save_state(path, tree, metadata=None):
    """Write a weight tree -- numpy arrays or torch tensors -- to safetensors."""
    import torch

    flat = flatten_tree(tree) if any(isinstance(v, dict) for v in tree.values()) else dict(tree)
    arrays = {}
    for key, value in flat.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value)
    return save_safetensors(path, arrays, metadata=metadata)


def read_state(path, device="cpu"):
    """Read a safetensors checkpoint into a flat {"a/b/c": tensor} dict on `device`."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(Path(path)), framework="pt", device=str(device)) as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out
