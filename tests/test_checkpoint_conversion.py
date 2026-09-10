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


def test_zero_dim_tensors_survive_a_roundtrip(tmp_path):
    """0-d variables must keep their shape.

    `np.ascontiguousarray` has ndmin=1 semantics and silently promotes a 0-d array to shape
    (1,). Several checkpoint variables are 0-d -- `n` above all, the data-dependent-init
    counter feeding `1 / (1 - decay**n)` -- so a stray leading axis there would broadcast
    through the running statistics rather than failing loudly.
    """
    from hidra.checkpoints import load_flat_checkpoint, save_safetensors

    tree = {
        "layer": {
            "n": np.float32(255.989),                 # 0-d, the shape that used to break
            "w": np.arange(6, dtype=np.float32).reshape(2, 3),
            "s": np.ones(3, dtype=np.float32),
            "t": np.asarray(7.0, dtype=np.float32),   # 0-d ndarray, same case
        }
    }
    path = save_safetensors(tmp_path / "zero_dim.safetensors", tree)
    back = load_flat_checkpoint(path)
    assert back["layer/n"].shape == (), f"0-d became {back['layer/n'].shape}"
    assert back["layer/t"].shape == ()
    assert back["layer/w"].shape == (2, 3)
    assert back["layer/s"].shape == (3,)
    assert float(back["layer/n"]) == pytest.approx(255.989, abs=1e-3)


def test_non_contiguous_arrays_are_saved_correctly(tmp_path):
    """A transposed view must be stored by value, not reinterpreted."""
    from hidra.checkpoints import load_flat_checkpoint, save_safetensors

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    view = base.T                                      # non-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    path = save_safetensors(tmp_path / "view.safetensors", {"v": view})
    np.testing.assert_array_equal(load_flat_checkpoint(path)["v"], view)


def test_thresholds_json_roundtrip(tmp_path, config_name):
    """thresholds.json must reproduce the pickle exactly -- it is the last pickle in the
    load path, and the values pick the decision thresholds."""
    from hidra import paths
    from hidra.checkpoints import load_thresholds, save_thresholds

    original = load_thresholds(paths.models_dir() / "thresholds.pkl")
    path = save_thresholds(tmp_path / "thresholds.json", original)
    back = load_thresholds(path)
    assert back == original, "thresholds changed through JSON"
    assert set(back) == set(original) and len(back) == 80


# ------------------------------------------------------------------ publishing

def test_publish_plan_is_complete_and_safetensors_only(monkeypatch):
    """`hidra-publish-weights` must offer a *complete* weight set and delete the pickles.

    A partial publication is the dangerous case: inference averages all five configs and
    loads a checkpoint for each, so a set missing one config fails at run time rather than
    at publish time.
    """
    from hidra import paths, publish

    models = paths.models_dir()
    if not (models / "thresholds.json").is_file():
        pytest.skip("weights not converted; run hidra-convert-weights")

    fake_siblings = [type("S", (), {"rfilename": n})()
                     for n in ["11fps_4bp_unsupervised.pkl", "thresholds.pkl", ".gitattributes"]]
    fake_info = type("I", (), {"siblings": fake_siblings, "sha": "deadbeef"})()

    class FakeApi:
        def model_info(self, repo, revision=None):
            return fake_info

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: FakeApi())
    adds, deletes, sha = publish.plan()

    names = [n for n, _ in adds]
    assert len(names) == 12, names                      # 10 checkpoints + thresholds + card
    assert names.count("README.md") == 1, "the model card must be published"
    assert "thresholds.json" in names
    assert all(n.endswith((".safetensors", ".json", ".md")) for n in names), names
    for config in ["11fps_4bp", "15fps_5bp", "19fps_6bp", "23fps_7bp", "27fps_6bp"]:
        assert f"{config}_unsupervised.safetensors" in names
        assert f"{config}_supervised_perlab_sniffall.safetensors" in names
    assert all(p.is_file() for _, p in adds)
    assert deletes == ["11fps_4bp_unsupervised.pkl", "thresholds.pkl"], deletes
    assert sha == "deadbeef"


def test_publish_plan_can_keep_the_pickles(monkeypatch):
    from hidra import paths, publish

    if not (paths.models_dir() / "thresholds.json").is_file():
        pytest.skip("weights not converted")
    fake = type("I", (), {"siblings": [type("S", (), {"rfilename": "a.pkl"})()], "sha": "x"})()
    monkeypatch.setattr("huggingface_hub.HfApi",
                        lambda *a, **k: type("A", (), {"model_info": lambda s, r, revision=None: fake})())
    _, deletes, _ = publish.plan(drop_pickles=False)
    assert deletes == []


def test_model_card_ships_with_the_package():
    from hidra import paths

    card = paths.ASSETS_DIR / "model_card.md"
    assert card.is_file(), "the model card must be packaged, not left in the repo root"
    text = card.read_text()
    assert text.startswith("---"), "needs YAML front matter for the Hub to render metadata"
    assert "license:" in text and "safetensors" in text


@pytest.mark.parametrize("name,other", [
    ("15fps_5bp_unsupervised.safetensors", "15fps_5bp_unsupervised.pkl"),
    ("15fps_5bp_unsupervised.pkl", "15fps_5bp_unsupervised.safetensors"),
    ("thresholds.json", "thresholds.pkl"),
    ("thresholds.pkl", "thresholds.json"),
    ("README.md", None),
])
def test_download_format_fallback_mapping(name, other):
    """The downloader must not assume which containers a given revision carries."""
    from hidra.download_models import _other_format_name

    assert _other_format_name(name) == other


def test_download_fallback_only_triggers_on_404():
    """A network or permissions failure must propagate, not be silently retried as the
    other format -- otherwise a broken download looks like a missing file."""
    from huggingface_hub.errors import EntryNotFoundError

    from hidra.download_models import _is_missing_file

    assert _is_missing_file(EntryNotFoundError("gone"))
    assert not _is_missing_file(OSError("network down"))

    class Resp:
        status_code = 403

    err = OSError("forbidden")
    err.response = Resp()
    assert not _is_missing_file(err)


def test_missing_accepts_either_container(tmp_path):
    """An existing models/ full of .pkl must not be re-downloaded as safetensors."""
    from hidra.download_models import CONFIGS, missing

    for config in CONFIGS:
        (tmp_path / f"{config}_unsupervised.pkl").touch()
        (tmp_path / f"{config}_supervised_perlab_sniffall.pkl").touch()
    (tmp_path / "thresholds.pkl").touch()
    assert missing(str(tmp_path)) == [], "a complete pkl-only dir should count as present"
    assert missing(str(tmp_path), fmt="safetensors"), "explicit --format must still be strict"

    (tmp_path / "thresholds.pkl").unlink()
    assert missing(str(tmp_path)) == ["thresholds.json"]
