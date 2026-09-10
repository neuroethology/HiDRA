"""The checkpoint conversion must be a pure re-container, and must not need JAX.

Those are the two properties that let the JAX dependency be dropped: safetensors can be
read with numpy alone, and the numbers inside are bit-identical to the published pickles,
so nothing about the model changes when the container does.
"""
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import requires_cuda, requires_weights

pytestmark = [requires_weights]


def test_reader_needs_no_jax():
    """`load_jax_checkpoint` must work in an interpreter where importing jax fails.

    Run in a subprocess with `jax` poisoned in sys.modules, since the parent process has
    JAX installed and a plain import check would pass for the wrong reason.
    """
    from hidra import paths

    script = f'''
import sys

class Blocker:
    def find_module(self, name, path=None):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("jax is blocked for this test")
        return None
    def find_spec(self, name, path=None, target=None):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("jax is blocked for this test")
        return None

sys.meta_path.insert(0, Blocker())
try:
    import jax
except ImportError:
    pass
else:
    raise AssertionError("blocker failed; jax was importable")

from hidra.torch.checkpoint import flatten_tree, load_jax_checkpoint
flat = flatten_tree(load_jax_checkpoint(r"{paths.models_dir()}/15fps_5bp_supervised_perlab_sniffall.pkl"))
assert "jax" not in sys.modules, "reading a checkpoint pulled in jax"
assert len(flat) == 185, len(flat)
assert flat["out-proj-perlab/w"].shape == (256, 82)
assert all(type(v).__name__ == "ndarray" for v in flat.values())
print("OK", len(flat))
'''
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, f"stdout={out.stdout}\nstderr={out.stderr}"
    assert "OK 185" in out.stdout


def test_reader_refuses_unexpected_globals(tmp_path):
    """The unpickler must not resolve arbitrary modules.

    A checkpoint is data a user may have downloaded; the reader exists precisely so JAX is
    not needed to open it, and it should not become a general-purpose code loader.
    """
    import pickle

    from hidra.torch.checkpoint import load_jax_checkpoint

    bad = tmp_path / "evil.pkl"
    bad.write_bytes(pickle.dumps({"cmd": subprocess.run}))
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        load_jax_checkpoint(bad)


def test_flatten_unflatten_roundtrip():
    from hidra.torch.checkpoint import flatten_tree, unflatten_tree

    tree = {"a": {"b": np.zeros(2), "c": {"d": np.ones(3)}}, "e": np.arange(4)}
    flat = flatten_tree(tree)
    assert sorted(flat) == ["a/b", "a/c/d", "e"]
    back = unflatten_tree(flat)
    assert sorted(back) == ["a", "e"]
    np.testing.assert_array_equal(back["a"]["c"]["d"], tree["a"]["c"]["d"])


@pytest.mark.parametrize("kind", ["unsupervised", "supervised_perlab_sniffall"])
def test_safetensors_is_bit_identical_to_pkl(kind, config_name):
    """Every tensor, byte for byte -- including NaN and -0.0 patterns."""
    from hidra import paths
    from hidra.torch.checkpoint import flatten_tree, load_jax_checkpoint, read_metadata, read_state

    base = paths.models_dir()
    st = base / f"{config_name}_{kind}.safetensors"
    if not st.is_file():
        pytest.skip("checkpoints not converted; run hidra-convert-weights")

    want = flatten_tree(load_jax_checkpoint(base / f"{config_name}_{kind}.pkl"))
    got = read_state(st)
    assert set(got) == set(want)
    for key in want:
        a = np.ascontiguousarray(got[key].cpu().numpy())
        b = np.ascontiguousarray(np.asarray(want[key]))
        assert a.shape == b.shape and a.dtype == b.dtype, key
        assert a.tobytes() == b.tobytes(), f"{key} differs byte-wise"

    meta = read_metadata(st)
    assert meta.get("format") == "hidra-jax-port-v1"
    assert meta.get("source") == f"{config_name}_{kind}.pkl"


@requires_cuda
def test_inference_identical_from_either_container(config_name, batch, device):
    """A model loaded from safetensors must predict *bit-identically* to one loaded from
    the pickle. Same weights, same device, same order of operations -- so unlike a
    cross-backend comparison, this one really should be exact."""
    from hidra import paths, schema
    from hidra.torch import load_perlab, load_unsupervised
    from hidra.torch.infer import to_torch_batch

    base = paths.models_dir()
    st = base / f"{config_name}_unsupervised.safetensors"
    if not st.is_file():
        pytest.skip("checkpoints not converted; run hidra-convert-weights")

    config = schema.get_configs()[config_name]
    tb = to_torch_batch(batch, device)

    outs = []
    for suffix in (".safetensors", ".pkl"):
        trunk = load_unsupervised(config_name, path=base / f"{config_name}_unsupervised{suffix}",
                                  device=device, config=config)
        head = load_perlab(config_name, trunk, device=device, config=config,
                           path=base / f"{config_name}_supervised_perlab_sniffall{suffix}")
        with torch.no_grad():
            outs.append(head.predict(tb).float().cpu().numpy())

    np.testing.assert_array_equal(outs[0], outs[1])


def test_converter_check_mode_reports_status():
    out = subprocess.run([sys.executable, "-m", "hidra.torch.convert", "--check"],
                         capture_output=True, text=True, timeout=300)
    assert "checkpoints converted" in out.stdout, out.stderr


def test_convert_one_verifies(tmp_path, config_name):
    """--one is the path a user's fine-tuned checkpoint takes; it must verify too."""
    from hidra import paths
    from hidra.torch.convert import convert_one

    src = paths.models_dir() / f"{config_name}_supervised_perlab_sniffall.pkl"
    out, n = convert_one(src, tmp_path / "ft.safetensors", verify=True)
    assert out.is_file() and n == 185
