"""Layer-by-layer equivalence between the JAX layers and the PyTorch port.

Each test drives the JAX layer and the ported layer with the *same published weights* and
the *same input*, and checks three things:

  1. the port matches JAX pinned to full float32 precision, to float32 round-off;
  2. the port is at least as close to a float64 reference as JAX is;
  3. the port's own error against float64 is at float32 round-off.

(2) is the one that matters when the numbers disagree: on Ampere, JAX's default float32
matmul precision is TF32, so the *reference* implementation is the imprecise one. A test
that only diffed the two backends would flag the port as broken.

Tolerances are on the *relative* error, scaled by the magnitude of the tensor.
"""
import exactness as X
import numpy as np
import pytest
import torch
from conftest import requires_cuda, requires_jax, requires_weights

pytestmark = [pytest.mark.jax, requires_jax, requires_weights]

# Full-float32 agreement between two independent implementations of the same arithmetic.
# Reductions happen in a different order, so exact bit equality is not expected; 1e-5
# relative is ~100x tighter than the TF32 error the JAX path carries by default.
TOL_SAME_MATH = 1e-5
# A single float32 layer's own round-off.
TOL_FLOAT32 = 1e-5


def _torch_linear_from(v, input_dim, output_dim, batch_dims, use_bias, normalize_input, device):
    from hidra.torch.layers import Linear

    layer = Linear(input_dim, output_dim, use_bias=use_bias, batch_dims=batch_dims,
                   normalize_input=normalize_input).to(device)
    with torch.no_grad():
        for name in ("w", "s"):
            getattr(layer, name).copy_(torch.as_tensor(np.asarray(v[name])))
        if use_bias:
            layer.b.copy_(torch.as_tensor(np.asarray(v["b"])))
        if normalize_input:
            for name in ("m1", "m2", "n", "mean", "std"):
                getattr(layer, name).copy_(torch.as_tensor(np.asarray(v[name])))
    return layer.eval()


def _jax_linear_from(v, input_dim, output_dim, batch_dims, use_bias, normalize_input):
    from hidra import solution

    layer = solution.Linear(input_dim, output_dim, "float32", use_bias=use_bias,
                            batch_dims=list(batch_dims), normalize_input=normalize_input)
    layer.set_context({"stage": "eval"})
    return layer.set_variables(v)


# (checkpoint, layer key, input_dim, output_dim, batch_dims) covering every Linear shape
# the model actually uses: plain, per-bodypart, and per-bodypart-pair.
LINEAR_CASES = [
    ("head", "feat-flat-proj", 192, 256, ()),
    ("head", "out-proj-perlab", 256, 82, ()),
    ("head", "ff-0-in", 256, 768, ()),
    ("head", "out-proj-0", 512, 256, ()),
    ("head", "ff-merge-in", 1536, 192, (5,)),
    ("trunk", "aug-proj", 6, 192, ()),
    ("trunk", "dt-proj-1", 2, 192, (5,)),
    ("trunk", "merge-0", 384, 192, (5,)),
    ("trunk", "self-in-0", 384, 96, (5, 5)),
    ("trunk", "self-dx-0", 2, 96, (5, 5)),
    ("trunk", "cross-out-3", 96, 192, (5, 5)),
    ("trunk", "out-proj", 192, 256, (5,)),
]


@requires_cuda
@pytest.mark.parametrize("which,key,din,dout,batch_dims", LINEAR_CASES,
                         ids=[f"{w}:{k}" for w, k, *_ in LINEAR_CASES])
def test_linear_matches(which, key, din, dout, batch_dims, trunk_checkpoint,
                        head_checkpoint, device):
    import jax.numpy as jnp

    v = (trunk_checkpoint if which == "trunk" else head_checkpoint)[key]
    assert tuple(v["w"].shape) == batch_dims + (din, dout), \
        f"checkpoint shape {v['w'].shape} disagrees with the case table"

    rng = np.random.default_rng(0)
    x = rng.standard_normal((24, 3) + batch_dims + (din,)).astype("float32")

    tout = _torch_linear_from(v, din, dout, batch_dims, True, True, device)
    with torch.no_grad():
        got = tout(torch.as_tensor(x, device=device)).float().cpu().numpy()

    jl = _jax_linear_from(v, din, dout, batch_dims, True, True)
    with X.jax_matmul_precision("float32"):
        jax_hi = np.asarray(jl.apply(jnp.asarray(x)))
    with X.jax_matmul_precision("default"):
        jax_def = np.asarray(jl.apply(jnp.asarray(x)))

    ref = X.ref_linear(v, x)
    m = X.compare(f"{which}/{key}", got, jax_hi, jax_def, ref)

    assert m["torch_vs_jax_hi"] < TOL_SAME_MATH, f"port disagrees with JAX float32: {m}"
    assert m["torch_vs_f64"] < TOL_FLOAT32, f"port is not float32-accurate: {m}"
    assert m["torch_vs_f64"] <= m["jax_default_vs_f64"] + 1e-12, \
        f"port is less accurate than default-precision JAX: {m}"


@requires_cuda
def test_linear_without_bias_or_normalization(trunk_checkpoint, device):
    """The LSTM's state-to-state projection: no bias, no input standardization."""
    import jax.numpy as jnp

    v = trunk_checkpoint["lstm-0"]["ss_linear"]
    din, dout, bd = 192, 768, (5,)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 2) + bd + (din,)).astype("float32")

    layer = _torch_linear_from(v, din, dout, bd, False, False, device)
    with torch.no_grad():
        got = layer(torch.as_tensor(x, device=device)).float().cpu().numpy()
    assert layer.b is None and not hasattr(layer, "mean")

    jl = _jax_linear_from(v, din, dout, bd, False, False)
    with X.jax_matmul_precision("float32"):
        jax_hi = np.asarray(jl.apply(jnp.asarray(x)))
    ref = X.ref_linear(v, x, use_bias=False, normalize_input=False)
    assert X.rel_err(got, jax_hi, ref) < TOL_SAME_MATH
    assert X.rel_err(got, ref, ref) < TOL_FLOAT32


@requires_cuda
def test_lstm_matches(trunk_checkpoint, device):
    """The trunk's per-bodypart LSTM: learned h0/c0, i/f/c/o gates, 24 recurrent steps.

    A recurrent layer is the strictest test in the suite -- an error in the gate ordering
    or the initial state compounds over timesteps instead of averaging out.
    """
    import jax.numpy as jnp

    from hidra import solution
    from hidra.torch.layers import LSTM

    v = trunk_checkpoint["lstm-0"]
    d_res, d_lstm, n_bp = 192, 192, 5
    rng = np.random.default_rng(2)
    x = rng.standard_normal((24, 2, n_bp, d_res)).astype("float32")

    tl = LSTM(d_res, d_lstm, batch_dims=(n_bp,)).to(device)
    from hidra.torch.models import load_state_into
    load_state_into(tl, {k: val for k, val in _flat(v).items()}, strict=False)
    tl.eval()
    with torch.no_grad():
        got = tl(torch.as_tensor(x, device=device)).float().cpu().numpy()

    jl = solution.LSTM(d_res, d_lstm, "float32", batch_dims=[n_bp])
    jl.set_context({"stage": "eval"})
    jm = jl.set_variables(v)
    with X.jax_matmul_precision("float32"):
        jax_hi = np.asarray(jm.apply(jnp.asarray(x)))
    with X.jax_matmul_precision("default"):
        jax_def = np.asarray(jm.apply(jnp.asarray(x)))

    ref = X.ref_lstm(v, x, d_lstm, batch_dims=(n_bp,))
    m = X.compare("lstm-0", got, jax_hi, jax_def, ref)
    assert m["torch_vs_jax_hi"] < TOL_SAME_MATH, f"LSTM disagrees with JAX float32: {m}"
    assert m["torch_vs_f64"] < TOL_FLOAT32, f"LSTM is not float32-accurate: {m}"
    assert m["torch_vs_f64"] <= m["jax_default_vs_f64"] + 1e-12, m


@requires_cuda
def test_bidirectional_lstm_matches(head_checkpoint, device):
    """The head's BiLSTM: the backward pass must see a time-reversed sequence and have its
    output reversed back, and the two directions must not share weights."""
    import jax.numpy as jnp

    from hidra import solution
    from hidra.torch.layers import BidirectionalLSTM
    from hidra.torch.models import load_state_into

    v = head_checkpoint["lstm-0"]
    d_res = d_lstm = 256
    rng = np.random.default_rng(3)
    x = rng.standard_normal((24, 3, d_res)).astype("float32")

    tl = BidirectionalLSTM(d_res, d_lstm).to(device)
    load_state_into(tl, _flat(v), strict=False)
    tl.eval()
    with torch.no_grad():
        got = tl(torch.as_tensor(x, device=device)).float().cpu().numpy()

    jl = solution.BidirectionalLSTM(d_res, d_lstm, "float32")
    jl.set_context({"stage": "eval"})
    with X.jax_matmul_precision("float32"):
        jax_hi = np.asarray(jl.set_variables(v).apply(jnp.asarray(x)))
    with X.jax_matmul_precision("default"):
        jax_def = np.asarray(jl.set_variables(v).apply(jnp.asarray(x)))

    ref = X.ref_bidirectional_lstm(v, x, d_lstm)
    m = X.compare("bilstm-0", got, jax_hi, jax_def, ref)
    assert m["torch_vs_jax_hi"] < TOL_SAME_MATH, m
    assert m["torch_vs_f64"] < TOL_FLOAT32, m
    # The halves must differ: identical halves would mean the backward LSTM got the
    # forward weights, which every aggregate metric above would happily accept.
    fw, bw = np.split(got, 2, axis=-1)
    assert np.abs(fw - bw).max() > 1e-3, "forward and backward outputs are identical"


@requires_cuda
def test_constant_and_embedding_match(trunk_checkpoint, head_checkpoint, device):
    from hidra import solution
    from hidra.torch.layers import Constant, Embedding

    v = trunk_checkpoint["x-emb"]
    jc = solution.Constant([5, 192], "float32")
    jc.set_context({"stage": "eval"})
    tc = Constant((5, 192)).to(device)
    with torch.no_grad():
        tc.c.copy_(torch.as_tensor(np.asarray(v["c"])))
    with torch.no_grad():
        np.testing.assert_array_equal(np.asarray(jc.set_variables(v).apply()),
                                      tc().float().cpu().numpy())

    ve = head_checkpoint["lab-embedding"]
    je = solution.Embedding(256, 21, "float32")
    je.set_context({"stage": "eval"})
    te = Embedding(256, 21).to(device)
    with torch.no_grad():
        te.w.copy_(torch.as_tensor(np.asarray(ve["w"])))
    idx = np.array([0, 9, 20, 3], dtype="int32")
    import jax.numpy as jnp
    with torch.no_grad():
        np.testing.assert_array_equal(
            np.asarray(je.set_variables(ve).apply(jnp.asarray(idx))),
            te(torch.as_tensor(idx, device=device)).float().cpu().numpy())


@requires_cuda
def test_gather_clamps_like_jax(head_checkpoint, device):
    """Out-of-range indices must clamp, not crash.

    `solution.batch` pads a short final batch with `np.empty_like`, so `batch["lab_id"]`
    holds uninitialized memory in the padding rows. JAX's gather clamps those into range;
    an unguarded torch index_select raises a device-side assert and kills the run. The
    values must also *agree*, so full-tensor comparisons stay clean over padding rows.
    """
    import jax.numpy as jnp

    from hidra import solution
    from hidra.torch.layers import Embedding, jax_gather_index

    ve = head_checkpoint["lab-embedding"]
    te = Embedding(256, 21).to(device)
    with torch.no_grad():
        te.w.copy_(torch.as_tensor(np.asarray(ve["w"])))
    je = solution.Embedding(256, 21, "float32")
    je.set_context({"stage": "eval"})

    # 1072902963 is a value actually observed in a padded lab_id row.
    idx = np.array([9, 1072902963, -5, -1, 20, 21], dtype="int32")

    # The clamping belongs to XLA's gather, so the JAX side must be traced to exhibit it.
    # That is how the real pipeline runs it -- run_test_probs_perlab wraps predict() in
    # @jax.jit -- whereas eager JAX here would hit plain numpy indexing and raise.
    #
    # The table has to be a JAX array for the traced gather to work at all. In production
    # it always is: `pickle.load` on a published checkpoint yields jaxlib arrays, because
    # jax's own _reconstruct_array device_puts them. Our numpy-only reader (the whole point
    # of which is not needing JAX) hands back ndarrays, so convert here.
    import jax
    jemb = je.set_variables({"w": jnp.asarray(np.asarray(ve["w"]))})
    want = np.asarray(jax.jit(jemb.apply)(jnp.asarray(idx)))

    with torch.no_grad():
        got = te(torch.as_tensor(idx, device=device)).float().cpu().numpy()
    np.testing.assert_array_equal(got, want)

    normalized = jax_gather_index(torch.as_tensor(idx.copy()), 21).cpu().numpy()
    np.testing.assert_array_equal(normalized, [9, 20, 16, 20, 20, 20])


def _flat(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}/"))
        else:
            out[f"{prefix}{k}"] = v
    return out
