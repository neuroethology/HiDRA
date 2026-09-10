"""Reading and writing model checkpoints, in either format, without a backend.

HiDRA's weights ship in two containers:

`.safetensors` -- the default. A flat `{"layer/variable": tensor}` map with a JSON header.
Loading it executes no code and needs neither JAX nor PyTorch.

`.pkl` -- the original format, a pickle of a nested dict of JAX device arrays. It is still
read here, and still without JAX: the pickle references exactly one JAX symbol,
`jax._src.array._reconstruct_array`, whose only job is to rebuild a numpy array and
`device_put` it. Intercepting it yields the numpy array directly.

Everything is numpy in and numpy out, so both backends share this module and a user's own
fine-tuned checkpoint can be converted with neither framework installed.
"""
import json
import pickle
from pathlib import Path

import numpy as np

# The only non-numpy global a HiDRA checkpoint pickle is allowed to name.
_JAX_RECONSTRUCT = ("jax._src.array", "_reconstruct_array")

# Modules whose globals a checkpoint pickle may legitimately reference.
_ALLOWED_MODULE_ROOTS = frozenset({"numpy", "builtins", "collections", "_codecs"})

CHECKPOINT_SUFFIXES = (".safetensors", ".pkl")


# ------------------------------------------------------------------ tree helpers

def flatten_tree(tree, prefix=""):
    """Nested dict -> {"a/b/c": array}."""
    out = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_tree(value, prefix=f"{path}/"))
        else:
            out[path] = value
    return out


def unflatten_tree(flat):
    """{"a/b/c": array} -> nested dict."""
    out = {}
    for path, value in flat.items():
        node = out
        parts = path.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


# ------------------------------------------------------------------ the pickle format

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
    """Unpickler that resolves JAX arrays to numpy arrays and refuses anything else.

    A checkpoint is data a user may have downloaded. This reader exists so JAX is not needed
    to open one; it should not become a general-purpose code loader in the process.
    """

    def find_class(self, module, name):
        if (module, name) == _JAX_RECONSTRUCT:
            return _reconstruct_array_as_numpy
        if module.split(".")[0] not in _ALLOWED_MODULE_ROOTS:
            raise pickle.UnpicklingError(
                f"refusing to load {module}.{name} from a checkpoint; expected only numpy "
                f"arrays and plain containers"
            )
        return super().find_class(module, name)


def load_pickle_checkpoint(path):
    """Read a `.pkl` checkpoint into a nested dict of numpy arrays, without JAX."""
    with open(path, "rb") as f:
        return _as_numpy_tree(_NumpyOnlyUnpickler(f).load())


def _as_numpy_tree(obj):
    if isinstance(obj, dict):
        return {k: _as_numpy_tree(v) for k, v in obj.items()}
    return np.asarray(obj)


# ------------------------------------------------------------------ safetensors

def load_safetensors(path):
    """Read a `.safetensors` checkpoint into a flat {"a/b/c": ndarray} dict."""
    from safetensors.numpy import load_file

    return load_file(str(path))


def save_safetensors(path, tree, metadata=None):
    """Write a (possibly nested) weight tree to safetensors."""
    from safetensors.numpy import save_file

    flat = flatten_tree(tree) if any(isinstance(v, dict) for v in tree.values()) else dict(tree)
    tensors = {k: _as_contiguous(v) for k, v in flat.items()}
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)
    return Path(path)


def _as_contiguous(value):
    """A C-contiguous copy, without changing the shape.

    `np.ascontiguousarray` alone will not do: it has `ndmin=1` semantics and silently turns
    a 0-d array into shape (1,). Several checkpoint variables *are* 0-d -- notably `n`, the
    data-dependent-init counter that feeds `1 / (1 - decay**n)` -- and a stray leading axis
    there would broadcast through the running statistics.
    """
    arr = np.asarray(value)
    if arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr).reshape(arr.shape)


def read_metadata(path):
    """The metadata header of a `.safetensors` checkpoint, as a dict."""
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as f:
        return dict(f.metadata() or {})


# ------------------------------------------------------------------ format-agnostic API

def load_checkpoint(path):
    """Read a checkpoint in either format -> nested dict of numpy arrays."""
    path = Path(path)
    if path.suffix == ".safetensors":
        return unflatten_tree(load_safetensors(path))
    return load_pickle_checkpoint(path)


def load_flat_checkpoint(path):
    """Read a checkpoint in either format -> flat {"a/b/c": ndarray} dict."""
    path = Path(path)
    if path.suffix == ".safetensors":
        return load_safetensors(path)
    return flatten_tree(load_pickle_checkpoint(path))


def resolve_checkpoint(models_dir, stem, prefer=".safetensors"):
    """Find `<models_dir>/<stem>` in whichever format is present.

    Prefers safetensors, because it needs no pickle and no framework to read. Falls back to
    the original `.pkl`, so an install that predates the conversion keeps working.
    """
    models_dir = Path(models_dir)
    order = [prefer] + [s for s in CHECKPOINT_SUFFIXES if s != prefer]
    for suffix in order:
        candidate = models_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no checkpoint for {stem!r} in {models_dir} "
        f"(looked for {', '.join(stem + s for s in order)}); "
        f"fetch the weights with `hidra-download-models`")


# ------------------------------------------------------------------ thresholds

def load_thresholds(path):
    """The ensemble thresholds as {(lab, action): threshold}.

    Accepts `thresholds.json` -- a list of `[lab, action, threshold]` rows, which needs no
    pickle -- or the original `thresholds.pkl`, whose keys are already tuples.
    """
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as f:
            rows = json.load(f)
        return {(str(lab), str(action)): value for lab, action, value in rows}
    with open(path, "rb") as f:
        return pickle.load(f)


def save_thresholds(path, thresholds):
    """Write {(lab, action): threshold} as sorted JSON rows."""
    rows = [[lab, action, thresholds[(lab, action)]] for lab, action in sorted(thresholds)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    return Path(path)


def resolve_thresholds(models_dir):
    """`thresholds.json` if present, else `thresholds.pkl`."""
    models_dir = Path(models_dir)
    for name in ("thresholds.json", "thresholds.pkl"):
        candidate = models_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no thresholds file in {models_dir}; fetch the weights with `hidra-download-models`")
