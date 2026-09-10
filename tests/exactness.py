"""Shared machinery for the JAX-vs-PyTorch exactness tests.

The comparison is deliberately three-way rather than two-way. On an Ampere GPU, JAX's
default float32 matmul precision is TF32 (~10 mantissa bits), so the *original*
implementation carries ~1e-4 relative error while the PyTorch port runs true float32 at
~1e-7. Diffing the two backends alone would therefore measure JAX's TF32 error and tell
you nothing about whether the port is correct.

So every comparison reports:

  torch(fp32)  vs  jax(precision="float32")   -- did we port the same math?  (tight)
  torch(fp32)  vs  float64 reference          -- how accurate is the port?
  jax(default) vs  float64 reference          -- how accurate was the original?
  jax(default) vs  torch(fp32)                -- the drift a user actually sees  (loose)

`jax_matmul_precision` is the switch that separates "the port is wrong" from "the GPU used
fewer mantissa bits".
"""
import contextlib
import os
import shutil

import numpy as np
import pandas as pd

# XLA's Triton autotuner logs fusion configs at ERROR level, which buries test output.
os.environ.setdefault("SNIFFALL", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

LAB_ID = "userdata"
DEFAULT_CONFIG = "15fps_5bp"
DEFAULT_LAB = "GroovyShrew"


# ------------------------------------------------------------------ precision control

@contextlib.contextmanager
def jax_matmul_precision(precision):
    """Force JAX's float32 matmul precision inside the block.

    "float32" gives true single precision (what torch does by default); "default" is
    whatever the backend picks, which is TF32 on Ampere and later.
    """
    import jax

    # jax exposes a context manager for this flag; config.read() refuses flags that have
    # one, so go through the context manager rather than saving/restoring by hand.
    with jax.default_matmul_precision(precision):
        yield


# ------------------------------------------------------------------ metrics

def rel_err(a, b, scale=None):
    """max|a-b| normalized by the scale of the reference, computed in float64."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    denom = np.abs(np.asarray(scale if scale is not None else b, np.float64)).max()
    return float(np.abs(a - b).max() / max(denom, 1e-30))


def compare(name, torch_out, jax_hi, jax_default, ref64):
    """The four numbers described in the module docstring, as a dict."""
    return {
        "name": name,
        "torch_vs_jax_hi": rel_err(torch_out, jax_hi, ref64),
        "torch_vs_f64": rel_err(torch_out, ref64, ref64),
        "jax_default_vs_f64": rel_err(jax_default, ref64, ref64),
        "jax_default_vs_torch": rel_err(jax_default, torch_out, ref64),
    }


def format_table(rows):
    head = (f"{'tensor':<26}{'torch~jax(f32)':>16}{'torch~f64':>12}"
            f"{'jax(def)~f64':>15}{'jax(def)~torch':>16}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['name']:<26}{r['torch_vs_jax_hi']:>16.2e}{r['torch_vs_f64']:>12.2e}"
                     f"{r['jax_default_vs_f64']:>15.2e}{r['jax_default_vs_torch']:>16.2e}")
    return "\n".join(lines)


# ------------------------------------------------------------------ float64 references

def ref_linear(v, x, use_bias=True, normalize_input=True):
    """`solution.Linear.apply` at eval stage, in float64."""
    w = np.asarray(v["w"], np.float64)
    s = np.asarray(v["s"], np.float64)
    weff = (w / np.linalg.norm(w, axis=-2, keepdims=True)) * s[..., None, :]
    x = np.asarray(x, np.float64)
    if normalize_input:
        x = (x - np.asarray(v["mean"], np.float64)) / np.asarray(v["std"], np.float64)
    y = np.einsum("...i,...io->...o", x, weff)
    if use_bias:
        y = y + np.asarray(v["b"], np.float64)
    return y


def ref_lstm(v, x, hidden_dim, batch_dims=()):
    """`solution.LSTM.apply` in float64: learned initial state, gates ordered i/f/c/o."""
    x = np.asarray(x, np.float64)
    tsteps, bs = x.shape[0], x.shape[1]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))

    h = np.tanh(np.asarray(v["h0"]["c"], np.float64))
    c = np.asarray(v["c0"]["c"], np.float64)
    reps = (bs,) + (1,) * (len(batch_dims) + 1)
    h = np.tile(h[None], reps)
    c = np.tile(c[None], reps)

    xs = ref_linear(v["is_linear"], x, use_bias=True, normalize_input=True)
    out = np.empty(xs.shape[:-1] + (hidden_dim,), np.float64)
    for t in range(tsteps):
        gates = xs[t] + ref_linear(v["ss_linear"], h, use_bias=False, normalize_input=False)
        i, f, g, o = np.split(gates, 4, axis=-1)
        c = sig(f) * c + sig(i) * np.tanh(g)
        h = sig(o) * np.tanh(c)
        out[t] = h
    return out


def ref_bidirectional_lstm(v, x, hidden_dim):
    fw = ref_lstm(v["lstm-fw"], x, hidden_dim)
    bw = ref_lstm(v["lstm-bw"], np.asarray(x, np.float64)[::-1], hidden_dim)[::-1]
    return np.concatenate([fw, bw], axis=-1)


# ------------------------------------------------------------------ model construction

def load_checkpoints(config_name=DEFAULT_CONFIG, models_dir=None):
    """The trunk and head checkpoints for one config, as nested numpy dicts."""
    from hidra import paths
    from hidra.torch.checkpoint import load_jax_checkpoint

    base = paths.models_dir() if models_dir is None else models_dir
    trunk = load_jax_checkpoint(f"{base}/{config_name}_unsupervised.pkl")
    head = load_jax_checkpoint(f"{base}/{config_name}_supervised_perlab_sniffall.pkl")
    return trunk, head


def jax_models(config_name=DEFAULT_CONFIG, dtype="float32", models_dir=None):
    """The JAX trunk and per-lab head, with published weights bound."""
    import pickle

    from hidra import paths, solution
    from hidra.train_perlab_heads import MultiTaskPerLabModel

    base = paths.models_dir() if models_dir is None else models_dir
    config = solution.get_configs()[config_name]
    trunk = solution.UnsupervisedModel(
        d_res=192, d_lstm=192, d_ff=384, d_edge=96, n_layers=4,
        n_bp=config["num_bodyparts"], sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"], dtype=dtype)
    upath = f"{base}/{config_name}_unsupervised.pkl"
    head = MultiTaskPerLabModel(d_res=256, d_ff=768, d_lstm=256, n_layers=3,
                                n_bp=config["num_bodyparts"], padding=32, dtype=dtype,
                                unsupervised_model=(trunk, upath))
    head.set_context({"stage": "eval"})
    with open(f"{base}/{config_name}_supervised_perlab_sniffall.pkl", "rb") as f:
        weights = pickle.load(f)
    return head.set_variables(weights), config


def torch_models(config_name=DEFAULT_CONFIG, device="cuda", models_dir=None):
    """The ported trunk and per-lab head, with the same published weights."""
    from hidra import schema
    from hidra.torch import load_perlab, load_unsupervised

    config = schema.get_configs()[config_name]
    kw = {} if models_dir is None else {}
    trunk = load_unsupervised(config_name, device=device, config=config, **kw)
    head = load_perlab(config_name, trunk, device=device, config=config)
    return trunk, head, config


# ------------------------------------------------------------------ test data

def stage_videos(track_dir, work_dir, lab=DEFAULT_LAB, pix_per_cm=16.0, fps=30.0):
    """Stage synthetic parquets into the dataset layout `solution.create_video` expects.

    This is the same staging `cli.run` performs before shelling out to the inference
    driver: copy each parquet to `<work>/custom_tracking/userdata/<video_id>.parquet` and
    write a manifest row carrying the pixel scale and frame rate.
    """
    import glob

    from hidra import cli, solution
    from hidra.run_allbehaviors_perlab import AllBehaviorLabels

    work_dir = str(work_dir)
    tdir = os.path.join(work_dir, "custom_tracking", LAB_ID)
    os.makedirs(tdir, exist_ok=True)

    parquets = sorted(glob.glob(os.path.join(str(track_dir), "*.parquet")))
    assert parquets, f"no parquets in {track_dir}"
    rows = []
    for pq in parquets:
        vid = cli.vid_of(pq)
        shutil.copyfile(pq, os.path.join(tdir, f"{vid}.parquet"))
        rows.append(dict(lab_id=LAB_ID, video_id=vid, frames_per_second=fps,
                         pix_per_cm_approx=pix_per_cm, behaviors_labeled="[]"))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(work_dir, "manifest.csv"), index=False)

    # Point the (module-global) dataset paths at the staged tree, as the driver does.
    solution.dataset_dir = work_dir
    solution.working_dir = os.path.join(work_dir, "working")
    os.makedirs(solution.working_dir, exist_ok=True)

    # Force every video's lab to `lab`, so the trunk uses that lab's embedding and the
    # head's P projection selects that lab's columns -- exactly what --embedding-lab does.
    emb_idx = int(solution.LABS.encode(lab))
    solution.LABS.encode = lambda v, _i=emb_idx: _i

    manifest = manifest.copy()
    manifest["mode"] = "custom"
    videos = [solution.create_video(i, row) for i, row in manifest.iterrows()]
    for v in videos:
        v.labels = AllBehaviorLabels(num_frames=v.num_frames,
                                     mouse_ids=v.tracking_data.mouse_index)
    return videos


def make_batch(config, videos, batch_size=4, num_epochs=1):
    """One deterministic batch of augmented clips, straight from `solution.Dataset`."""
    from hidra.torch.infer import iter_batches, make_dataset

    dataset = make_dataset(config, videos, num_epochs=num_epochs)
    return next(iter_batches(dataset, batch_size))
