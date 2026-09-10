"""End-to-end: the whole `predict.py` pipeline on both backends, compared.

This runs the real CLI -- staging, the per-lab subprocess, the 5-config ensemble average,
thresholding, bout extraction -- twice, and compares what a user would actually get:
per-frame probabilities, the binary calls, and the bout table.

What is asserted, and why the thresholds are what they are:

* **Calls must match exactly.** The binary call is the output people act on. A backend
  swap that moved even one frame across a threshold would be a real behavioural change.
* **Probabilities to ~2e-3.** Two effects are folded in here and neither is the port's
  arithmetic: the JAX side runs its matmuls in TF32 (~1e-3 relative on this GPU), and the
  saved probability tracks are cast to float16, whose spacing near 1.0 is ~1e-3.
* **Bout boundaries must match exactly**, since they are derived from the calls.
"""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import requires_cuda, requires_jax, requires_weights

pytestmark = [pytest.mark.slow, requires_weights, requires_cuda]

CONFIG = "15fps_5bp"
LAB = "GroovyShrew"

# Folded TF32 (~1e-3) + the float16 quantization of the stored tracks (~1e-3 near 1.0).
TOL_PROB = 2e-3
# Whole-track agreement: if the two backends really compute the same function, the tracks
# should be near-perfectly correlated, not merely close in max-norm.
MIN_CORR = 0.9999


REPO = pathlib.Path(__file__).resolve().parent.parent


def _run_cli(track_dir, out_dir, backend, configs=CONFIG, extra=()):
    repo = str(REPO)
    cmd = [sys.executable, f"{repo}/predict.py", str(track_dir), "--out", str(out_dir),
           "--labs", LAB, "--configs", configs, "--gpu", "0", "--backend", backend, *extra]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=repo)
    assert res.returncode == 0, f"{backend} run failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    assert "WARNING" not in res.stdout or "partial ensemble" in res.stdout, res.stdout
    assert "inference exited" not in res.stdout, f"subprocess failed:\n{res.stdout[-3000:]}"
    return res


@pytest.fixture(scope="module")
def both_backends(track_dir, tmp_path_factory):
    """Run the CLI once per backend and return the two output directories."""
    outs = {}
    for backend in ("jax", "torch"):
        out = tmp_path_factory.mktemp(f"out_{backend}")
        _run_cli(track_dir, out, backend)
        outs[backend] = out
    return outs


@pytest.fixture(scope="module")
def stems(track_dir):
    import glob
    import os
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(f"{track_dir}/*.parquet"))


@requires_jax
def test_frames_agree(both_backends, stems):
    key = ["subject", "target", "lab", "action", "frame"]
    worst = 0.0
    for stem in stems:
        a = pd.read_parquet(both_backends["jax"] / f"{stem}.frames.parquet")
        b = pd.read_parquet(both_backends["torch"] / f"{stem}.frames.parquet")
        assert len(a) == len(b) and len(a) > 0, f"{stem}: {len(a)} vs {len(b)} rows"
        a = a.sort_values(key, ignore_index=True)
        b = b.sort_values(key, ignore_index=True)
        assert (a[key] == b[key]).all().all(), f"{stem}: the two runs cover different tracks"

        pa = a.prob.to_numpy(np.float64)
        pb = b.prob.to_numpy(np.float64)
        diff = np.abs(pa - pb)
        worst = max(worst, diff.max())
        assert diff.max() < TOL_PROB, (
            f"{stem}: max prob diff {diff.max():.3e} exceeds {TOL_PROB:.0e} "
            f"(mean {diff.mean():.2e})")
        corr = np.corrcoef(pa, pb)[0, 1]
        assert corr > MIN_CORR, f"{stem}: track correlation only {corr:.8f}"
    print(f"\nworst per-frame probability difference across {len(stems)} videos: {worst:.3e}")


@requires_jax
def test_calls_agree_exactly(both_backends, stems):
    """The thresholded calls are the deliverable; they must be identical."""
    key = ["subject", "target", "lab", "action", "frame"]
    total = 0
    for stem in stems:
        a = pd.read_parquet(both_backends["jax"] / f"{stem}.frames.parquet").sort_values(key, ignore_index=True)
        b = pd.read_parquet(both_backends["torch"] / f"{stem}.frames.parquet").sort_values(key, ignore_index=True)
        differing = int((a.call.to_numpy() != b.call.to_numpy()).sum())
        assert differing == 0, f"{stem}: {differing}/{len(a)} calls differ between backends"
        total += len(a)
    print(f"\nall {total} binary calls identical across backends")


@requires_jax
def test_bouts_agree_exactly(both_backends, stems):
    cols = ["subject", "target", "lab", "action", "start_frame", "stop_frame", "n_frames"]
    for stem in stems:
        a = pd.read_csv(both_backends["jax"] / f"{stem}.bouts.csv")
        b = pd.read_csv(both_backends["torch"] / f"{stem}.bouts.csv")
        assert len(a) == len(b), f"{stem}: {len(a)} jax bouts vs {len(b)} torch bouts"
        if not len(a):
            continue
        a = a.sort_values(cols, ignore_index=True)
        b = b.sort_values(cols, ignore_index=True)
        pd.testing.assert_frame_equal(a[cols], b[cols])
        mp = np.abs(a.mean_prob - b.mean_prob).max()
        assert mp < TOL_PROB, f"{stem}: mean_prob differs by {mp:.3e}"
        print(f"\n{stem}: {len(a)} bouts, boundaries identical, "
              f"max|mean_prob diff| {mp:.2e}")


def test_torch_backend_output_is_wellformed(both_backends, stems):
    """Sanity on the torch output on its own terms, independent of the comparison."""
    for stem in stems:
        f = pd.read_parquet(both_backends["torch"] / f"{stem}.frames.parquet")
        assert set(f.columns) == {"frame", "subject", "target", "lab", "action", "prob", "call"}
        assert f.prob.between(0, 1).all(), "probabilities out of [0, 1]"
        assert f.prob.notna().all(), "NaN in the probability track"
        assert set(f.call.unique()) <= {0, 1}
        assert (f.lab == LAB).all()
        # Frames must cover the whole video with no gaps, per track.
        n = f.frame.max() + 1
        for _, g in f.groupby(["subject", "target", "action"]):
            assert np.array_equal(np.sort(g.frame.to_numpy()), np.arange(n)), \
                "a track has missing or duplicated frames"


def test_model_loading_needs_no_jax(config_name):
    """Building the ported models and loading their weights must not touch JAX.

    This part of the promise is already kept: `hidra.torch` reaches `solution` only from
    inside functions, and the checkpoint reader is numpy-only. So the *model* side is
    JAX-free today; it is the data pipeline that is not (see the xfail below).
    """
    script = f"""
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("jax is blocked for this test")
        return None

sys.meta_path.insert(0, Blocker())
import torch
from hidra.torch import load_perlab, load_unsupervised

device = "cuda" if torch.cuda.is_available() else "cpu"
trunk = load_unsupervised({config_name!r}, device=device)
head = load_perlab({config_name!r}, trunk, device=device)

# A synthetic batch, so no data pipeline is involved: this isolates the model path.
n_bp = trunk.n_bp
batch = {{
    "agent": torch.randn(2, 128, n_bp, 2, device=device),
    "target": torch.randn(2, 128, n_bp, 2, device=device),
    "augmentation_params": torch.zeros(2, 6, device=device),
    "lab_id": torch.full((2,), 9, dtype=torch.long, device=device),
}}
with torch.no_grad():
    probs = head.predict(batch)
assert probs.shape == (2, 64, 38), probs.shape
assert torch.isfinite(probs).all()
assert "jax" not in sys.modules, sorted(m for m in sys.modules if "jax" in m)
print("OK", tuple(probs.shape))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=600, cwd=str(REPO))
    assert out.returncode == 0, f"stdout={out.stdout}\nstderr={out.stderr[-3000:]}"
    assert "OK" in out.stdout


@pytest.mark.xfail(strict=True, reason="solution.py imports jax at module scope, so the "
                                       "shared numpy data pipeline still drags in the "
                                       "runtime; splitting it out is the remaining work")
def test_full_torch_inference_needs_no_jax(track_dir):
    """The *whole* torch path -- data pipeline included -- with JAX unavailable.

    `hidra.torch.infer` reuses `solution.Dataset` and `solution.Predictions` on purpose:
    they are pure numpy and deterministically seeded, so a backend comparison isolates the
    model arithmetic and nothing else. The cost is that `solution` imports JAX at module
    scope, so this fails today. xfail(strict=True) means that when the data pipeline is
    split out, this XPASSes, the suite goes red, and the marker has to be deleted -- the
    promise cannot quietly rot.
    """
    script = f"""
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("jax is blocked for this test")
        return None

sys.meta_path.insert(0, Blocker())
sys.argv = ["predict.py"]

import hidra.cli as cli
cli.run({str(track_dir)!r}, out="/tmp/hidra_nojax_out", labs=["GroovyShrew"],
        actions=["rear"], pix_per_cm=16, fps=30, configs=["15fps_5bp"], backend="torch")
assert "jax" not in sys.modules, sorted(m for m in sys.modules if "jax" in m)
print("OK")
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=1200, cwd=str(REPO))
    assert out.returncode == 0, f"stdout={out.stdout[-2000:]}\nstderr={out.stderr[-3000:]}"


@pytest.mark.jax
@requires_jax
def test_full_ensemble_agrees(track_dir, tmp_path_factory, stems):
    """All five configs, the way a user actually runs it.

    The single-config tests above isolate one model; this exercises the real deliverable --
    five checkpoints averaged through `Predictions`, then thresholded into bouts. It is also
    the only test that covers the 4-, 6- and 7-bodypart configs end to end, where a shape
    assumption in the port would surface.
    """
    outs = {}
    for backend in ("jax", "torch"):
        out = tmp_path_factory.mktemp(f"full_{backend}")
        _run_cli(track_dir, out, backend, configs=",".join(
            ["11fps_4bp", "15fps_5bp", "19fps_6bp", "23fps_7bp", "27fps_6bp"]))
        outs[backend] = out

    key = ["subject", "target", "lab", "action", "frame"]
    cols = ["subject", "target", "lab", "action", "start_frame", "stop_frame", "n_frames"]
    total, worst = 0, 0.0
    for stem in stems:
        a = pd.read_parquet(outs["jax"] / f"{stem}.frames.parquet").sort_values(key, ignore_index=True)
        b = pd.read_parquet(outs["torch"] / f"{stem}.frames.parquet").sort_values(key, ignore_index=True)
        assert len(a) == len(b) > 0
        pa, pb = a.prob.to_numpy(np.float64), b.prob.to_numpy(np.float64)
        worst = max(worst, float(np.abs(pa - pb).max()))
        total += len(a)
        assert (a.call.to_numpy() == b.call.to_numpy()).all(), f"{stem}: calls differ"
        assert np.corrcoef(pa, pb)[0, 1] > MIN_CORR

        ba = pd.read_csv(outs["jax"] / f"{stem}.bouts.csv")
        bb = pd.read_csv(outs["torch"] / f"{stem}.bouts.csv")
        assert len(ba) == len(bb), f"{stem}: {len(ba)} vs {len(bb)} bouts"
        if len(ba):
            pd.testing.assert_frame_equal(ba.sort_values(cols, ignore_index=True)[cols],
                                          bb.sort_values(cols, ignore_index=True)[cols])
    assert worst < TOL_PROB
    print(f"\nfull 5-config ensemble: {total} frame-rows, all calls and bouts identical, "
          f"worst probability difference {worst:.2e}")
