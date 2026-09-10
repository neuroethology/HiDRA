"""Reading the published JAX checkpoints, and writing torch-native ones.

The checkpoints are pickles of a nested dict of `jaxlib` device arrays. Pickle references
exactly one JAX symbol -- `jax._src.array._reconstruct_array` -- and that function's only
job is to rebuild a numpy array and then `device_put` it. Intercepting it lets the weights
be read with numpy alone, so converting a checkpoint (including a user's own fine-tuned
one) never requires JAX to be installed.

    load_jax_checkpoint("models/15fps_5bp_unsupervised.pkl")   # -> nested dict of ndarrays

`save_state`/`read_state` handle the torch-native side. State is stored flat, with "/" as
the path separator, in safetensors: a plain tensor container with no code execution on
load, unlike pickle.
"""
import pickle
from pathlib import Path

import numpy as np

_JAX_RECONSTRUCT = ("jax._src.array", "_reconstruct_array")


def _reconstruct_array_as_numpy(fun, args, arr_state, aval_state):
    """Stand-in for jax._src.array._reconstruct_array that stops before device_put.

    Upstream does:
        np_value = fun(*args); np_value.__setstate__(arr_state)
        jnp_value = api.device_put(np_value); jnp_value.aval = ...; return jnp_value
    We keep the numpy value. `aval_state` only carries JAX's weak-typing flag, which has no
    meaning outside JAX's type-promotion rules and no effect on the stored numbers.
    """
    value = fun(*args)
    value.__setstate__(arr_state)
    return value


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that resolves JAX arrays to numpy arrays."""

    def find_class(self, module, name):
        if (module, name) == _JAX_RECONSTRUCT:
            return _reconstruct_array_as_numpy
        # Anything else must come from numpy or be a builtin container. Refusing the rest
        # keeps an untrusted checkpoint from importing arbitrary modules.
        if module.split(".")[0] not in ("numpy", "builtins", "collections", "_codecs"):
            raise pickle.UnpicklingError(
                f"refusing to load {module}.{name} from a checkpoint; expected only numpy "
                f"arrays and plain containers"
            )
        return super().find_class(module, name)


def load_jax_checkpoint(path):
    """Read a HiDRA JAX checkpoint into a nested dict of numpy arrays, without JAX."""
    with open(path, "rb") as f:
        obj = _NumpyOnlyUnpickler(f).load()
    return _as_numpy_tree(obj)


def _as_numpy_tree(obj):
    if isinstance(obj, dict):
        return {k: _as_numpy_tree(v) for k, v in obj.items()}
    arr = np.asarray(obj)
    # Checkpoints are float32 throughout; keep whatever came in but drop any 0-d weirdness.
    return arr


def flatten_tree(tree, prefix=""):
    """Nested dict -> {"a/b/c": ndarray}."""
    out = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_tree(value, prefix=f"{path}/"))
        else:
            out[path] = value
    return out


def unflatten_tree(flat):
    """{"a/b/c": ndarray} -> nested dict."""
    out = {}
    for path, value in flat.items():
        node = out
        parts = path.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def save_state(path, tree, metadata=None):
    """Write a (possibly nested) weight tree to safetensors."""
    import torch
    from safetensors.torch import save_file

    flat = flatten_tree(tree) if any(isinstance(v, dict) for v in tree.values()) else dict(tree)
    tensors = {}
    for key, value in flat.items():
        t = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
        # safetensors rejects shared storage; contiguous+clone keeps each entry independent.
        tensors[key] = t.contiguous().clone()
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)
    return path


def read_state(path, device="cpu"):
    """Read a safetensors checkpoint into a flat {"a/b/c": tensor} dict."""
    from safetensors import safe_open

    out = {}
    with safe_open(str(path), framework="pt", device=str(device)) as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out


def read_metadata(path):
    """The metadata header of a safetensors checkpoint, as a dict."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return dict(f.metadata() or {})
