"""Adding a (lab, action) column to the per-lab head: the remap, the persisted head table,
and the loaders that read it.

`finetune.py train --new-head` widens `out-proj-perlab` by one column. What has to hold:
the 82 published columns land in their new positions bit-identically, the new column sits
where the extended table says, the table travels with the checkpoint (sidecar for a .pkl,
metadata for a .safetensors), and both containers load on the torch backend at the right
width without touching thresholds.json. The end-to-end run is in `test_new_head_e2e.py`.
"""
import json
import pickle

import numpy as np
import pytest
from conftest import requires_weights

from hidra import head_table as H
from hidra import schema

W, S, B = ("out-proj-perlab/w", "out-proj-perlab/s", "out-proj-perlab/b")
NEW = ("GroovyShrew", "attack")      # GroovyShrew has no attack column...
SEED_FROM = ("LyricalHare", "attack")  # ...LyricalHare does


def _fake_flat(table, seed=0):
    """A head-shaped flat state with recognisable random values (no weights needed)."""
    rng = np.random.default_rng(seed)
    n = len(table)
    return {
        W: rng.standard_normal((256, n)).astype("float32"),
        S: rng.uniform(0.5, 2.0, n).astype("float32"),
        B: rng.standard_normal(n).astype("float32"),
        "out-proj-perlab/mean": rng.standard_normal(256).astype("float32"),
        "out-proj-perlab/std": rng.uniform(0.5, 2.0, 256).astype("float32"),
        "out-proj-perlab/n": np.asarray(255.0, dtype="float32"),
        "lab-embedding/w": rng.standard_normal((21, 256)).astype("float32"),
    }


# ------------------------------------------------------------------ the table

def test_published_table_is_the_82_sorted_columns():
    table = schema.lab_action_table()
    assert len(table) == 82 and table == sorted(table)
    assert NEW not in table and SEED_FROM in table


def test_extend_table_sorts_and_refuses_duplicates():
    old = schema.lab_action_table()
    new = H.extend_table(old, [NEW])
    assert len(new) == 83 and new == sorted(new) and set(new) == set(old) | {NEW}
    with pytest.raises(ValueError, match="already have a head column"):
        H.extend_table(old, [SEED_FROM])
    # Lists from JSON and tuples from schema are the same table.
    assert H.extend_table([list(p) for p in old], [list(NEW)]) == new


def test_parse_head_spec():
    assert H.parse_head_spec(" GroovyShrew , attack ") == NEW
    for bad in ("GroovyShrew", "a,b,c", ",attack", ""):
        with pytest.raises(ValueError):
            H.parse_head_spec(bad)


def test_validate_new_head_explains_each_refusal():
    table = schema.lab_action_table()
    assert H.validate_new_head(*NEW, table, seed_from=SEED_FROM) == NEW
    with pytest.raises(ValueError, match="unknown lab"):
        H.validate_new_head("NoSuchLab", "attack", table)
    with pytest.raises(ValueError, match="no published head"):
        H.validate_new_head("CRIM13", "attack", table)          # a lab, but without heads
    with pytest.raises(ValueError, match="not a behaviour in the vocabulary"):
        H.validate_new_head("GroovyShrew", "tailrattle", table)
    with pytest.raises(ValueError, match="not a behaviour in the vocabulary"):
        H.validate_new_head("GroovyShrew", "none", table)
    with pytest.raises(ValueError, match="already has a head column"):
        H.validate_new_head("GroovyShrew", "rear", table)
    with pytest.raises(ValueError, match="not an existing head.*LyricalHare"):
        H.validate_new_head(*NEW, table, seed_from=("GroovyShrew", "attack"))


# ------------------------------------------------------------------ the remap

def test_remap_keeps_old_columns_bit_identical_and_places_the_new_one():
    old = schema.lab_action_table()
    new = H.extend_table(old, [NEW])
    flat = _fake_flat(old)
    out, new_columns = H.remap_head_columns(flat, old, new, seed_from=SEED_FROM, seed=1)

    j = new.index(NEW)
    assert new_columns == {NEW: j}
    assert out[W].shape == (256, 83) and out[S].shape == (83,) and out[B].shape == (83,)
    assert out[W].dtype == np.float32

    # Every published column, found by name, byte for byte.
    for i, pair in enumerate(old):
        k = new.index(pair)
        assert out[W][:, k].tobytes() == flat[W][:, i].tobytes(), pair
        assert out[S][k] == flat[S][i] and out[B][k] == flat[B][i], pair
    # The new column is the seed column.
    i = old.index(SEED_FROM)
    assert out[W][:, j].tobytes() == flat[W][:, i].tobytes()
    assert out[S][j] == flat[S][i] and out[B][j] == flat[B][i]
    # The input statistics are per input feature: untouched. Nothing else changes either,
    # and the input dict is not mutated.
    for key in ("out-proj-perlab/mean", "out-proj-perlab/std", "out-proj-perlab/n", "lab-embedding/w"):
        assert out[key] is flat[key]
    assert flat[W].shape == (256, 82)


def test_remap_is_order_independent():
    """Appending instead of sorting must give the same columns, just elsewhere."""
    old = schema.lab_action_table()
    flat = _fake_flat(old)
    appended = list(old) + [NEW]
    out, cols = H.remap_head_columns(flat, old, appended, seed_from=SEED_FROM)
    assert cols == {NEW: 82}
    assert out[W][:, :82].tobytes() == flat[W].tobytes()
    assert out[W][:, 82].tobytes() == flat[W][:, old.index(SEED_FROM)].tobytes()


def test_remap_fresh_init_follows_the_layer_init_and_is_seeded():
    old = schema.lab_action_table()
    new = H.extend_table(old, [NEW])
    flat = _fake_flat(old)
    out, cols = H.remap_head_columns(flat, old, new, seed=7)
    j = cols[NEW]
    assert out[S][j] == 1.0 and out[B][j] == 0.0
    # Linear.create_variables: N(0, 1) / (sqrt(max(256, 83)) / sqrt(64)) = N(0, 0.5).
    col = out[W][:, j]
    assert np.isfinite(col).all() and 0.35 < col.std() < 0.65 and abs(col.mean()) < 0.15
    again, _ = H.remap_head_columns(flat, old, new, seed=7)
    other, _ = H.remap_head_columns(flat, old, new, seed=8)
    assert again[W][:, j].tobytes() == col.tobytes()
    assert other[W][:, j].tobytes() != col.tobytes()


def test_remap_refuses_a_mismatched_or_shrinking_table():
    old = schema.lab_action_table()
    flat = _fake_flat(old)
    with pytest.raises(ValueError, match="columns but old_table"):
        H.remap_head_columns(flat, old[:-1], old)
    with pytest.raises(ValueError, match="drops existing columns"):
        H.remap_head_columns(flat, old, old[:-1] + [NEW])
    with pytest.raises(ValueError, match="not a column of old_table"):
        H.remap_head_columns(flat, old, H.extend_table(old, [NEW]), seed_from=NEW)


# ------------------------------------------------------------------ persistence

def test_sidecar_roundtrip(tmp_path):
    table = H.extend_table(schema.lab_action_table(), [NEW])
    ckpt = tmp_path / "15fps_5bp__mytag.pkl"
    side = H.save_head_table(ckpt, table)
    assert side == tmp_path / "15fps_5bp__mytag.heads.json"
    doc = json.loads(side.read_text())
    assert doc["n_heads"] == 83 and doc["checkpoint"] == ckpt.name
    assert H.load_head_table(ckpt) == table
    assert H.head_table_for(ckpt) == table
    # A checkpoint without a sidecar is described by the published table.
    plain = tmp_path / "15fps_5bp__plain.pkl"
    assert H.load_head_table(plain) is None
    assert H.head_table_for(plain) == schema.lab_action_table()
    # Malformed sidecars are refused rather than read as an empty table.
    (tmp_path / "bad.heads.json").write_text(json.dumps({"lab_action": [["GroovyShrew"]]}))
    with pytest.raises(ValueError, match="malformed"):
        H.load_head_table(tmp_path / "bad.pkl")


def test_head_table_for_template_requires_agreement(tmp_path):
    table = H.extend_table(schema.lab_action_table(), [NEW])
    template = str(tmp_path / "{config}__tag.pkl")
    configs = ["11fps_4bp", "15fps_5bp"]
    for c in configs:
        (tmp_path / f"{c}__tag.pkl").write_bytes(b"")
    assert H.head_table_for_template(template, configs) is None       # no tables -> published
    for c in configs:
        H.save_head_table(template.format(config=c), table)
    assert H.head_table_for_template(template, configs) == table
    assert H.head_table_for_template(template, configs + ["19fps_6bp"]) == table  # absent file: skipped
    H.sidecar_path(template.format(config="11fps_4bp")).unlink()
    with pytest.raises(ValueError, match="do not"):
        H.head_table_for_template(template, configs)
    H.save_head_table(template.format(config="11fps_4bp"), schema.lab_action_table())
    with pytest.raises(ValueError, match="disagree"):
        H.head_table_for_template(template, configs)


def test_check_head_width_names_the_fix():
    old = schema.lab_action_table()
    flat = _fake_flat(old)
    H.check_head_width(flat, old)
    with pytest.raises(ValueError, match=r"83 head columns.*82 entries.*heads\.json"):
        H.check_head_width(H.remap_head_columns(flat, old, H.extend_table(old, [NEW]))[0], old)


# ------------------------------------------------------------------ loading on the torch backend

@pytest.fixture
def widened_checkpoint(tmp_path, config_name):
    """The published head for one config, widened by (GroovyShrew, attack), as a .pkl in the
    JAX layout plus its sidecar -- exactly what the trainer writes."""
    from hidra.checkpoints import load_flat_checkpoint, unflatten_tree
    from hidra.torch.models import perlab_checkpoint_path

    src = perlab_checkpoint_path(config_name)
    old = schema.lab_action_table()
    new = H.extend_table(old, [NEW])
    flat = load_flat_checkpoint(src)
    widened, cols = H.remap_head_columns(flat, old, new, seed_from=SEED_FROM)
    ckpt = tmp_path / f"{config_name}__newhead.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(unflatten_tree(widened), f)
    H.save_head_table(ckpt, new)
    return dict(path=ckpt, table=new, published=flat, column=cols[NEW])


def _check_loaded_head(head, ck, config_name):
    import torch

    from hidra.torch.train import head_column_selector

    j, table = ck["column"], ck["table"]
    assert head.n_heads == 83 and head.lab_action == table
    w = head.layers["out-proj-perlab"].w.detach().cpu().numpy()
    assert w.shape == (256, 83)
    for i, pair in enumerate(schema.lab_action_table()):
        assert w[:, table.index(pair)].tobytes() == ck["published"][W][:, i].tobytes(), pair
    assert w[:, j].tobytes() == ck["published"][W][:, schema.lab_action_table().index(SEED_FROM)].tobytes()
    # P routes the new column to (GroovyShrew, attack) and nothing else.
    P = head.P.cpu()
    assert float(P.sum()) == 83
    assert P[schema.LABS.value_to_idx["GroovyShrew"], schema.ACTIONS.value_to_idx["attack"], j] == 1
    # The supervision mask finds it by name.
    cols = head_column_selector(head.lab_action, "GroovyShrew", ["attack"])
    assert int(cols.argmax()) == j and float(cols.sum()) == 1
    assert torch.is_tensor(head.layers["out-proj-perlab"].mean)


@requires_weights
def test_widened_pkl_loads_on_torch_with_its_sidecar(widened_checkpoint, config_name):
    from hidra.torch.models import build_unsupervised, load_perlab

    config = schema.get_configs()[config_name]
    trunk = build_unsupervised(config)             # random trunk: only the head is under test
    head = load_perlab(config_name, trunk, path=widened_checkpoint["path"], config=config)
    _check_loaded_head(head, widened_checkpoint, config_name)

    # Separated from its sidecar the checkpoint must refuse to load -- and say why.
    H.sidecar_path(widened_checkpoint["path"]).unlink()
    with pytest.raises(ValueError, match=r"83 head columns.*heads\.json"):
        load_perlab(config_name, trunk, path=widened_checkpoint["path"], config=config)


@requires_weights
def test_convert_one_carries_the_table_into_safetensors(widened_checkpoint, config_name, tmp_path):
    from hidra.checkpoints import read_metadata
    from hidra.torch.convert import convert_one
    from hidra.torch.models import build_unsupervised, load_perlab

    out, n = convert_one(widened_checkpoint["path"], tmp_path / "widened.safetensors", verify=True)
    assert out.is_file() and n == 185
    meta = read_metadata(out)
    assert json.loads(meta["lab_action"]) == [list(p) for p in widened_checkpoint["table"]]
    assert meta["n_heads"] == "83"
    assert H.load_head_table(out) == widened_checkpoint["table"]

    # The safetensors file describes itself: no sidecar needed next to it.
    assert not H.sidecar_path(out).exists()
    config = schema.get_configs()[config_name]
    head = load_perlab(config_name, build_unsupervised(config), path=out, config=config)
    _check_loaded_head(head, widened_checkpoint, config_name)


@requires_weights
def test_published_checkpoint_still_loads_at_82(config_name):
    """No sidecar, no metadata: the published table applies, as before."""
    from hidra.torch.models import build_unsupervised, load_perlab, perlab_checkpoint_path

    config = schema.get_configs()[config_name]
    path = perlab_checkpoint_path(config_name)
    assert H.load_head_table(path) is None
    head = load_perlab(config_name, build_unsupervised(config), path=path, config=config)
    assert head.n_heads == 82 and head.lab_action == schema.lab_action_table()
