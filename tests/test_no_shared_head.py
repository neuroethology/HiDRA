"""The canonical per-lab model has no shared 38-way head (the 2026-10 weights).

Up to the 2026-09 release `MultiTaskPerLabModel` was multi-task: a shared `out-proj` head was
trained alongside the per-lab head and its loss taken over every lab. The 2026-10 weights were
trained without it, so both backends now build the model without that layer by default,
load checkpoints that carry one (every earlier release, and the inert copy spliced into the
2026-10 pickles for older HiDRA versions) as well as checkpoints that do not, and the
fine-tuning code paths never touch it. `shared_head=True` / `SHARED_HEAD=1` still build the
legacy architecture.

Everything here runs on random weights and a synthetic batch, so it needs neither the
published weights nor a GPU; the JAX cases need the `jax` extra.
"""
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest
from conftest import requires_jax

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = "15fps_5bp"
SEQ_LEN, PADDING, BATCH = 64, 32, 2


def _config():
    from hidra import schema
    return schema.get_configs()[CONFIG]


@pytest.fixture(scope="module")
def trunk():
    """A randomly initialised trunk, frozen and in eval stage like a loaded one."""
    from hidra.torch.layers import STAGE_EVAL
    from hidra.torch.models import build_unsupervised

    torch.manual_seed(0)
    model = build_unsupervised(_config()).set_stage(STAGE_EVAL).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _flat_state(module):
    """A module's tensors as the flat {"jax/path": array} dict the loader reads."""
    from hidra.torch.models import _jax_to_torch_path

    tensors = dict(module.named_parameters())
    tensors.update({k: v for k, v in module.named_buffers() if k != "P"})
    return {_jax_to_torch_path(k): v.detach().cpu().numpy().copy() for k, v in tensors.items()}


def _build(trunk, shared_head, seed=1):
    from hidra import schema
    from hidra.torch.models import build_perlab

    torch.manual_seed(seed)
    table = schema.lab_action_table()
    head = build_perlab(_config(), trunk, n_heads=len(table), n_actions=len(schema.ACTIONS),
                        n_labs=len(schema.LABS), shared_head=shared_head)
    head.lab_action = table
    head.set_P(schema.build_P(table))
    return head


def _batch(seed=0):
    """One synthetic batch with every field the model and the loss read, numpy as `data.batch`
    yields it: coordinates in mm (no NaN), one lab per sample, labels for a few actions."""
    from hidra import schema

    rng = np.random.default_rng(seed)
    n_bp = _config()["num_bodyparts"]
    in_len = SEQ_LEN + 2 * PADDING
    n_actions = len(schema.ACTIONS)
    walk = np.cumsum(rng.normal(0, 2, size=(BATCH, in_len, n_bp, 2)), axis=1)
    agent = (walk + rng.normal(0, 30, size=(BATCH, 1, n_bp, 2))).astype("float32")
    target = (walk[:, ::-1] + rng.normal(0, 30, size=(BATCH, 1, n_bp, 2))).astype("float32")
    aug = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], "float32"), (BATCH, 1))
    lab_id = np.array([schema.LABS.value_to_idx["GroovyShrew"],
                       schema.LABS.value_to_idx["LyricalHare"]], "int32")[:BATCH]
    self_labels = rng.integers(0, 2, size=(BATCH, SEQ_LEN)).astype("int16")
    self_labels *= schema.ACTIONS.value_to_idx["rear"]
    cross_labels = rng.integers(0, 2, size=(BATCH, SEQ_LEN)).astype("int16")
    cross_labels *= schema.ACTIONS.value_to_idx["sniff"]
    self_mask = np.zeros((BATCH, n_actions), "int8")
    self_mask[:, schema.ACTIONS.value_to_idx["rear"]] = 1
    cross_mask = np.zeros((BATCH, n_actions), "int8")
    cross_mask[:, schema.ACTIONS.value_to_idx["sniff"]] = 1
    return {"agent": agent, "target": target, "augmentation_params": aug, "lab_id": lab_id,
            "self_labels": self_labels, "cross_labels": cross_labels,
            "self_label_mask": self_mask, "cross_label_mask": cross_mask,
            "batch_mask": np.ones(BATCH, "int8")}


def _torch_batch(batch):
    from hidra.torch.infer import to_torch_batch

    tb = to_torch_batch(batch, "cpu")
    for k in ("self_labels", "cross_labels", "self_label_mask", "cross_label_mask", "batch_mask"):
        tb[k] = torch.as_tensor(batch[k])
    return tb


# ------------------------------------------------------------------ the torch model

def test_canonical_head_has_no_shared_layer(trunk):
    head = _build(trunk, shared_head=False)
    assert "out-proj" not in head.layers and not head.shared_head
    assert "out-proj-perlab" in head.layers
    assert not any(k.startswith("out-proj/") for k in _flat_state(head))


def test_legacy_shared_head_is_opt_in(trunk):
    from hidra import schema

    head = _build(trunk, shared_head=True)
    assert "out-proj" in head.layers and head.shared_head
    assert head.layers["out-proj"].output_dim == len(schema.ACTIONS)


def test_loader_ignores_an_inert_shared_head(trunk):
    """A checkpoint carrying `out-proj/*` must load into the canonical model, report those
    tensors as unexpected, and predict exactly as the same checkpoint with them removed."""
    from hidra.torch.models import load_perlab

    legacy_state = _flat_state(_build(trunk, shared_head=True))
    shared = sorted(k for k in legacy_state if k.startswith("out-proj/"))
    assert len(shared) == 8
    stripped = {k: v for k, v in legacy_state.items() if k not in shared}

    a = load_perlab(CONFIG, trunk, state=legacy_state, path="legacy.pkl", device="cpu")
    b = load_perlab(CONFIG, trunk, state=stripped, path="stripped.pkl", device="cpu")
    assert a.load_report["missing"] == [] and a.load_report["unexpected"] == shared
    assert b.load_report["missing"] == [] and b.load_report["unexpected"] == []
    assert "out-proj" not in a.layers and "out-proj" not in b.layers

    tb = _torch_batch(_batch())
    with torch.no_grad():
        pa, pb = a.predict(tb), b.predict(tb)
    assert pa.shape[-1] == len(a.layers["out-proj-perlab"].s) or pa.shape[-1] > 0
    np.testing.assert_array_equal(pa.numpy(), pb.numpy())
    assert float(pa.min()) >= 0.0 and float(pa.max()) <= 1.0


def test_legacy_model_requires_the_shared_head(trunk):
    from hidra.torch.models import load_perlab

    stripped = _flat_state(_build(trunk, shared_head=False))
    with pytest.raises(ValueError, match="absent from the checkpoint"):
        load_perlab(CONFIG, trunk, state=stripped, path="stripped.pkl", device="cpu",
                    shared_head=True)
    legacy = load_perlab(CONFIG, trunk, state=_flat_state(_build(trunk, shared_head=True)),
                         path="legacy.pkl", device="cpu", shared_head=True)
    assert "out-proj" in legacy.layers and legacy.load_report["unexpected"] == []


def test_ddi_and_loss_run_without_the_shared_head(trunk):
    from hidra.torch import train as T

    batch = _torch_batch(_batch())
    head = _build(trunk, shared_head=False)
    head.set_stage("init", init_decay=0.99)
    with torch.no_grad():
        T.ddi_forward(head, batch)
    head.set_stage("train")
    assert float(head.layers["out-proj-perlab"].n) == 1.0, "the per-lab head was not calibrated"

    columns = T.head_column_selector(head.lab_action, "GroovyShrew", ["rear"])
    loss, metrics = T.perlab_loss(head, batch, head_columns=columns)
    assert torch.isfinite(loss) and metrics["nll_perlab"][1] > 0
    assert "nll_shared" not in metrics
    with pytest.raises(ValueError, match="shared_head=True"):
        T.perlab_loss(head, batch, head_columns=columns, include_shared=True)

    legacy = _build(trunk, shared_head=True)
    legacy.set_stage("init", init_decay=0.99)
    with torch.no_grad():
        T.ddi_forward(legacy, batch)
    legacy.set_stage("train")
    assert float(legacy.layers["out-proj"].n) == 1.0, "the legacy shared head must still be visited"
    _, metrics = T.perlab_loss(legacy, batch, head_columns=columns, include_shared=True)
    assert "nll_shared" in metrics


def test_finetune_checkpoint_layout_has_no_shared_head(trunk):
    """What the torch fine-tuner saves must round-trip through the loader, with no `out-proj`."""
    from hidra.checkpoints import flatten_tree
    from hidra.torch.models import load_perlab
    from hidra.torch.train_perlab import to_jax_layout

    head = _build(trunk, shared_head=False)
    state = dict(head.named_parameters())
    state.update(dict(head.named_buffers()))
    tree = to_jax_layout(state)
    assert "out-proj" not in tree and "out-proj-perlab" in tree and "P" not in tree
    flat = flatten_tree(tree)
    assert len(flat) == 177
    back = load_perlab(CONFIG, trunk, state=flat, path="ft.pkl", device="cpu")
    assert back.load_report["missing"] == [] and back.load_report["unexpected"] == []


def test_head_mode_skips_the_statistics_pass_by_default(monkeypatch):
    from hidra.torch import train_perlab as TP

    monkeypatch.setenv("LABTAIL", "GroovyShrew")
    for var in ("LABTAIL_DDI_STEPS", "LABTAIL_HEAD_ONLY", "LABTAIL_EMB_ONLY"):
        monkeypatch.delenv(var, raising=False)
    assert TP.env_settings(CONFIG)["ddi_steps"] == 256            # tail mode: the original's
    monkeypatch.setenv("LABTAIL_HEAD_ONLY", "1")
    assert TP.env_settings(CONFIG)["ddi_steps"] == 0              # head mode: keep the statistics
    monkeypatch.setenv("LABTAIL_DDI_STEPS", "32")
    assert TP.env_settings(CONFIG)["ddi_steps"] == 32             # explicit wins


# ------------------------------------------------------------------ the JAX model

def _jax_trunk_checkpoint(path):
    """A random JAX trunk written as the pickle `SupervisedModel.__init__` loads."""
    import jax

    from hidra import solution

    cfg = _config()
    um = solution.UnsupervisedModel(
        d_res=192, d_lstm=192, d_ff=384, d_edge=96, n_layers=4, n_bp=cfg["num_bodyparts"],
        sample_rate=cfg["sample_rate"], aggregation_radius=cfg["aggregation_radius"], dtype="float32")
    values = jax.tree.map(lambda v: np.asarray(v.value), um.create_variables(jax.random.key(0)),
                          is_leaf=solution.is_variable)
    with open(path, "wb") as f:
        pickle.dump(values, f)
    return um


@requires_jax
def test_jax_model_has_no_shared_head_and_binds_either_checkpoint(tmp_path):
    import jax

    from hidra import solution
    from hidra import train_perlab_heads as tph

    assert not tph.SHARED_HEAD
    um = _jax_trunk_checkpoint(tmp_path / "trunk.pkl")
    cfg = _config()
    model = tph.MultiTaskPerLabModel(d_res=256, d_ff=768, d_lstm=256, n_layers=3,
                                     n_bp=cfg["num_bodyparts"], padding=32, dtype="float32",
                                     unsupervised_model=(um, str(tmp_path / "trunk.pkl")))
    assert "out-proj" not in model.layers and "out-proj-perlab" in model.layers

    variables = model.create_variables(jax.random.key(1))
    assert "out-proj" not in variables
    values = jax.tree.map(lambda v: np.asarray(v.value), variables, is_leaf=solution.is_variable)
    # A legacy / spliced checkpoint has one more layer: dropped, not bound.
    legacy = dict(values, **{"out-proj": {"w": np.zeros((256, 38), "float32")}})
    bound = tph.model_layer_weights(legacy, model, source="legacy")
    assert set(bound) == set(model.layers)
    # A layer the model needs but the checkpoint lacks is a clear error.
    with pytest.raises(KeyError, match="out-proj-perlab"):
        tph.model_layer_weights({k: v for k, v in values.items() if k != "out-proj-perlab"}, model)

    # The restore used by LABTAIL_KEEP_STATS touches only the running statistics.
    other = model.create_variables(jax.random.key(2))
    count = [0]
    restored = tph.restore_input_stats(other, values, count)
    assert count[0] > 0
    np.testing.assert_array_equal(np.asarray(restored["out-proj-perlab"]["mean"].value),
                                  values["out-proj-perlab"]["mean"])
    np.testing.assert_array_equal(np.asarray(restored["out-proj-perlab"]["w"].value),
                                  np.asarray(other["out-proj-perlab"]["w"].value))


@requires_jax
def test_shared_head_env_restores_the_legacy_architecture(tmp_path):
    """`SHARED_HEAD=1` is read at import, so it is checked in a subprocess."""
    script = f'''
import os, pickle, numpy as np, jax
os.environ["JAX_PLATFORMS"] = "cpu"
from hidra import solution
from hidra import train_perlab_heads as tph
assert tph.SHARED_HEAD
cfg = solution.get_configs()["{CONFIG}"]
um = solution.UnsupervisedModel(d_res=192, d_lstm=192, d_ff=384, d_edge=96, n_layers=4,
    n_bp=cfg["num_bodyparts"], sample_rate=cfg["sample_rate"],
    aggregation_radius=cfg["aggregation_radius"], dtype="float32")
values = jax.tree.map(lambda v: np.asarray(v.value), um.create_variables(jax.random.key(0)),
                      is_leaf=solution.is_variable)
pickle.dump(values, open(r"{tmp_path / 'trunk.pkl'}", "wb"))
m = tph.MultiTaskPerLabModel(d_res=256, d_ff=768, d_lstm=256, n_layers=3, n_bp=cfg["num_bodyparts"],
                             padding=32, dtype="float32", unsupervised_model=(um, r"{tmp_path / 'trunk.pkl'}"))
assert "out-proj" in m.layers, sorted(m.layers)
assert m.layers["out-proj"].output_dim == len(solution.ACTIONS)
print("LEGACY OK")
'''
    env = {**os.environ, "SNIFFALL": "1", "SHARED_HEAD": "1", "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=600, env=env, cwd=REPO)
    assert out.returncode == 0, f"{out.stdout}\n{out.stderr[-3000:]}"
    assert "LEGACY OK" in out.stdout


@requires_jax
def test_jax_and_torch_agree_without_the_shared_head(tmp_path, trunk):
    """The same random weights on both backends give the same per-lab logits (CPU, float32):
    the no-shared-head model is the same function on either side."""
    import jax
    import jax.numpy as jnp

    from hidra import solution
    from hidra import train_perlab_heads as tph
    from hidra.torch.models import load_perlab, load_state_into

    um = _jax_trunk_checkpoint(tmp_path / "trunk.pkl")
    cfg = _config()
    jmodel = tph.MultiTaskPerLabModel(d_res=256, d_ff=768, d_lstm=256, n_layers=3,
                                      n_bp=cfg["num_bodyparts"], padding=32, dtype="float32",
                                      unsupervised_model=(um, str(tmp_path / "trunk.pkl")))
    jmodel.set_context({"stage": "eval"})
    values = jax.tree.map(lambda v: np.asarray(v.value), jmodel.create_variables(jax.random.key(3)),
                          is_leaf=solution.is_variable)
    # Make the random head read non-trivial statistics so the standardization path is exercised.
    for layer in values.values():
        if isinstance(layer, dict) and "mean" in layer:
            rng = np.random.default_rng(len(layer["mean"]))
            layer["mean"] = rng.normal(0, 0.5, layer["mean"].shape).astype("float32")
            layer["std"] = rng.uniform(0.5, 2.0, layer["std"].shape).astype("float32")
    jhead = jmodel.set_variables(jax.tree.map(jnp.asarray, values))

    from hidra.checkpoints import flatten_tree, load_flat_checkpoint
    load_state_into(trunk, load_flat_checkpoint(tmp_path / "trunk.pkl"))
    thead = load_perlab(CONFIG, trunk, state=flatten_tree(values), path="random.pkl", device="cpu")
    assert thead.load_report["missing"] == [] and thead.load_report["unexpected"] == []

    batch = _batch()
    with jax.default_matmul_precision("float32"):
        jx = jhead._trunk_feats({k: jnp.asarray(v) for k, v in batch.items()}, jax.random.key(0))
        jlogits = np.asarray(jhead.layers["out-proj-perlab"].apply(jx))
    with torch.no_grad():
        tlogits = thead.logits_perlab(_torch_batch(batch)).numpy()
    assert jlogits.shape == tlogits.shape == (BATCH, SEQ_LEN, len(thead.lab_action))
    err = np.abs(jlogits - tlogits).max() / max(np.abs(jlogits).max(), 1e-30)
    assert err < 1e-4, f"backends disagree on the per-lab logits: {err:.2e} relative"
