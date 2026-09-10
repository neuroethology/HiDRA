"""Whole-model equivalence: trunk features, head logits, and probabilities.

The layer tests prove each operation was ported correctly in isolation. These prove the
*composition* is right -- that the residual stream, the agent/target batch-axis packing,
the cross-mouse half-swap, the bodypart pooling and the padding strip are all wired the
same way. A layer-level suite passes happily while, say, the two mice are swapped.

Comparisons run on real published weights and a real augmented batch from
`solution.Dataset`, which contains NaN keypoints -- the missing-data paths are exercised,
not bypassed.
"""
import exactness as X
import numpy as np
import pytest
import torch
from conftest import requires_cuda, requires_jax, requires_weights

pytestmark = [pytest.mark.jax, requires_jax, requires_weights, requires_cuda]

# Whole-model float32 round-off, accumulated over 4 trunk blocks (each with an LSTM over
# 128 steps) and 3 BiLSTM head blocks. Looser than a single layer, far tighter than TF32.
TOL_SAME_MATH = 1e-4


@pytest.fixture(scope="module")
def jax_batch(batch):
    import jax.numpy as jnp
    return {k: jnp.asarray(v) for k, v in batch.items()}


@pytest.fixture(scope="module")
def torch_batch(batch, device):
    from hidra.torch.infer import to_torch_batch
    return to_torch_batch(batch, device)


@pytest.fixture(scope="module")
def jax_key():
    import jax
    return jax.random.key(0)


def test_batch_is_realistic(batch):
    """Guard the inputs: a batch of all-NaN or all-finite clips would make the comparison
    vacuous, and a wrong lab id would silently select the wrong head columns."""
    agent = batch["agent"]
    nan_frac = float(np.isnan(agent).mean())
    assert 0.0 < nan_frac < 0.5, f"expected some missing keypoints, got {nan_frac:.3f}"
    assert agent.shape[1] == 128, "input clip should be seq_len 64 + 2*32 padding"
    assert set(np.unique(batch["lab_id"][batch["batch_mask"] == 1])) == {9}, \
        "every row should carry the forced embedding lab (GroovyShrew = 9)"


def test_trunk_features_match(torch_trunk, jax_head, torch_batch, jax_batch, jax_key):
    with torch.no_grad():
        got = [f.float().cpu().numpy() for f in torch_trunk.extract_features(torch_batch)]

    jax_trunk = jax_head.module.unsupervised_model
    with X.jax_matmul_precision("float32"):
        want_hi = [np.asarray(f) for f in jax_trunk.extract_features(jax_batch, jax_key)]
    with X.jax_matmul_precision("default"):
        want_def = [np.asarray(f) for f in jax_trunk.extract_features(jax_batch, jax_key)]

    assert len(got) == len(want_hi) == 4, "trunk should emit one tensor per layer"
    rows = []
    for i, (g, wh, wd) in enumerate(zip(got, want_hi, want_def, strict=True)):
        assert g.shape == wh.shape, f"layer {i}: {g.shape} vs {wh.shape}"
        assert np.isfinite(g).all(), f"layer {i} produced non-finite features"
        m = X.compare(f"trunk layer {i}", g, wh, wd, wh.astype(np.float64))
        rows.append(m)
        assert m["torch_vs_jax_hi"] < TOL_SAME_MATH, "\n" + X.format_table(rows)
    print("\n" + X.format_table(rows))


def test_head_logits_match(torch_head, jax_head, torch_batch, jax_batch, jax_key):
    """Logits, not probabilities: most heads sit far into the sigmoid tail on any given
    clip, so probabilities compress real disagreement into 1e-4 differences. The logits
    span roughly -16..+5 here, where a discrepancy has nowhere to hide."""
    with torch.no_grad():
        got = torch_head.logits_perlab(torch_batch).float().cpu().numpy()

    with X.jax_matmul_precision("float32"):
        want_hi = np.asarray(
            jax_head.layers["out-proj-perlab"].apply(jax_head._trunk_feats(jax_batch, jax_key)))
    with X.jax_matmul_precision("default"):
        want_def = np.asarray(
            jax_head.layers["out-proj-perlab"].apply(jax_head._trunk_feats(jax_batch, jax_key)))

    assert got.shape == want_hi.shape == (4, 64, 82)
    assert got.min() < -5 and got.max() > 0, \
        f"logit range {got.min():.1f}..{got.max():.1f} looks degenerate"
    m = X.compare("head logits", got, want_hi, want_def, want_hi.astype(np.float64))
    print("\n" + X.format_table([m]))
    assert m["torch_vs_jax_hi"] < TOL_SAME_MATH, m
    assert m["torch_vs_f64"] <= m["jax_default_vs_f64"] + 1e-12, m


def test_head_probs_match(torch_head, jax_head, torch_batch, jax_batch, jax_key):
    with torch.no_grad():
        got = torch_head.predict(torch_batch).float().cpu().numpy()
    with X.jax_matmul_precision("float32"):
        want_hi = np.asarray(jax_head.predict(jax_batch, jax_key))
    with X.jax_matmul_precision("default"):
        want_def = np.asarray(jax_head.predict(jax_batch, jax_key))

    m = X.compare("head probs", got, want_hi, want_def, want_hi.astype(np.float64))
    print("\n" + X.format_table([m]))
    assert m["torch_vs_jax_hi"] < TOL_SAME_MATH, m
    assert 0.0 <= got.min() and got.max() <= 1.0, \
        f"probabilities out of range: {got.min()}..{got.max()}"


def test_P_routing_selects_one_head_per_action(torch_head):
    """P must *select* a head, never sum several.

    If P were built through the monkeypatched `LABS.encode` the inference driver installs,
    every lab's head would land in one row and this einsum would add all 82 together --
    producing probabilities above 1. That bug is invisible in the logits and only shows up
    here, so it gets its own assertion.
    """
    P = torch_head.P.cpu().numpy()
    assert P.shape == (21, 38, torch_head.n_heads)
    assert set(np.unique(P)) <= {0.0, 1.0}
    assert P.sum(axis=2).max() == 1.0, \
        "some (lab, action) pair maps to more than one head; predict() would sum them"
    assert P.sum() == torch_head.n_heads, "every head column should be routed exactly once"


def test_unused_action_columns_are_exactly_zero(torch_head, torch_batch):
    """Actions the selected lab has no head for must come out exactly 0.0.

    `run_allbehaviors_perlab` filters its saved tracks by head name and relies on this;
    a small nonzero leak there would silently create bouts for behaviours the lab never
    annotated.
    """
    from hidra import schema

    with torch.no_grad():
        probs = torch_head.predict(torch_batch).float().cpu().numpy()
    lab = schema.LABS.decode(9)
    has_head = {a for l, a in torch_head.lab_action if l == lab}
    for aid in range(probs.shape[-1]):
        action = schema.ACTIONS.decode(aid)
        column = probs[..., aid]
        if action in has_head:
            assert np.abs(column).max() > 0, f"{lab}/{action} has a head but is all zero"
        else:
            assert np.array_equal(column, np.zeros_like(column)), \
                f"{lab}/{action} has no head but is nonzero (max {column.max():.2e})"


@pytest.mark.slow
@pytest.mark.parametrize("cfg", ["11fps_4bp", "19fps_6bp", "23fps_7bp", "27fps_6bp"])
def test_other_configs_match(cfg, videos, device):
    """The remaining four ensemble configs.

    Each has a different bodypart count (4/6/7), which changes every `batch_dims` in the
    trunk, so a shape assumption baked into the port would only surface here.
    """
    import jax
    import jax.numpy as jnp

    jax_head, config = X.jax_models(cfg)
    _, torch_head, _ = X.torch_models(cfg, device=device)
    batch = X.make_batch(config, videos, batch_size=2)

    from hidra.torch.infer import to_torch_batch
    with torch.no_grad():
        got = torch_head.logits_perlab(to_torch_batch(batch, device)).float().cpu().numpy()
    jb = {k: jnp.asarray(v) for k, v in batch.items()}
    key = jax.random.key(0)
    with X.jax_matmul_precision("float32"):
        want = np.asarray(
            jax_head.layers["out-proj-perlab"].apply(jax_head._trunk_feats(jb, key)))

    assert got.shape[-1] == 82
    assert batch["agent"].shape[2] == config["num_bodyparts"]
    err = X.rel_err(got, want, want)
    print(f"\n{cfg} (n_bp={config['num_bodyparts']}): torch~jax(f32) rel err {err:.2e}")
    assert err < TOL_SAME_MATH, f"{cfg}: relative error {err:.2e}"


def test_padded_batch_does_not_corrupt_real_rows(config, videos, torch_head, device):
    """A short final batch must run, and its garbage padding must stay contained.

    `solution.batch` fills a short batch out with `np.empty_like` -- uninitialized memory,
    flagged `batch_mask=0`. Those rows carry nonsense coordinates and an out-of-range
    `lab_id`. Two things have to hold: the pass must not crash (it did, with a CUDA
    device-side assert, before the gather was made to clamp like JAX's), and the real rows
    must produce the same numbers they do without the padding present.
    """
    from hidra.torch.infer import iter_batches, make_dataset, to_torch_batch

    dataset = make_dataset(config, videos, num_epochs=1)
    batches = list(iter_batches(dataset, 64))
    padded = batches[-1]
    n_real = int((padded["batch_mask"] == 1).sum())
    assert (padded["batch_mask"] == 0).any(), "expected a short final batch"
    assert not ((padded["lab_id"] >= 0) & (padded["lab_id"] < 21)).all(), \
        "padding rows should carry an out-of-range lab_id (uninitialized memory)"

    with torch.no_grad():
        full = torch_head.predict(to_torch_batch(padded, device)).float().cpu().numpy()

    # Re-run just the real rows, with no padding at all.
    trimmed = {k: v[:n_real] for k, v in padded.items()}
    with torch.no_grad():
        clean = torch_head.predict(to_torch_batch(trimmed, device)).float().cpu().numpy()

    # Not bit-identical, and that is expected: batch height changes which GEMM kernel
    # cuBLAS selects, and with it the reduction order. What matters is that the difference
    # stays at float32 round-off -- garbage rows leaking in would be order-1, not 1e-7.
    diff = np.abs(full[:n_real].astype(np.float64) - clean.astype(np.float64)).max()
    assert diff < 1e-5, f"padding rows perturbed the real rows by {diff:.2e}"
    assert np.isfinite(full[:n_real]).all()
