"""Training on cached frozen features must be the same training.

`finetune.py train --cache-features` computes everything frozen once per augmented window
(`hidra.torch.train.FeatureCache`) instead of every step. That is only a speed-up if the
numbers are unchanged, so these compare the cached path against the live one on the same
windows: the features themselves, the loss, and the probabilities an evaluation reads.

The one thing that *is* different by design is the augmentation: the cache holds a fixed
set of draws rather than a fresh one per step. That is a training-set choice, not an
arithmetic difference, and is checked separately (`test_passes_add_fresh_draws_of_the_same_windows`).
"""
import os

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import requires_cuda, requires_weights

from hidra import data
from hidra.torch import train as T

pytestmark = [requires_weights, requires_cuda]

LAB, ACTIONS = "GroovyShrew", ["rear", "sniffall"]
BATCH = 4


def _dataset(config, videos, num_epochs=1):
    """The trainer's augmented window stream, single-worker so two builds agree element for element."""
    return data.Dataset(
        videos=videos, seq_len=64, sample_rate=config["sample_rate"], padding=32,
        num_bodyparts=config["num_bodyparts"], num_epochs=num_epochs, unsupervised=False,
        max_scale=config["max_scale"], max_time_dilation=config["max_time_dilation"],
        rotate=True, flip=True, noise_scale=config["noise_scale"], num_workers=1,
        seed=[0, 12345])


def _numpy_batches(config, videos, num_epochs=1):
    return data.batch(_dataset(config, videos, num_epochs).element_iterator(), BATCH)


@pytest.fixture(scope="module")
def columns(torch_head, device):
    return T.head_column_selector(torch_head.lab_action, LAB, ACTIONS, device=device)


@pytest.mark.parametrize("level,forward", [("feats", "trunk_feats"), ("x0", "merge_feats")])
def test_the_cache_stores_the_live_forward_bit_for_bit(level, forward, config, videos,
                                                       torch_head, device):
    """Window for window, the cache holds exactly what the frozen layers computed."""
    from hidra.torch.infer import to_torch_batch

    cache = T.FeatureCache(level).build(torch_head, _numpy_batches(config, videos), device)
    assert cache.n > 0

    at = 0
    for batch in _numpy_batches(config, videos):
        keep = np.flatnonzero(np.asarray(batch["batch_mask"]) == 1)
        if not len(keep):
            continue
        with torch.no_grad():
            live = getattr(torch_head, forward)(to_torch_batch(batch, device))
        if level == "x0":
            live = live.permute(1, 0, 2)          # (tsteps, B, d) -> (B, tsteps, d)
        stored = cache.feats[at:at + len(keep)].to(device)
        torch.testing.assert_close(stored, live[keep], rtol=0, atol=0,
                                   msg=f"{level}: cached features differ at window {at}")
        at += len(keep)
    assert at == cache.n, "the cache and a fresh stream disagree on how many windows there are"


def test_tail_on_cached_x0_matches_the_live_tail(config, videos, torch_head, device):
    """At the `x0` level the tail still runs live, so it must reproduce `trunk_feats`.

    Only to float32 round-off, and not because of the cache: the tail's matmuls reduce in a
    different order when the batch holds a different number of rows, so running it on the
    windows a short final batch actually carried already differs from running it on the
    padded batch by ~1e-5 on features of magnitude ~40. Re-shuffled cached batches are a
    different composition by design.
    """
    from hidra.torch.infer import to_torch_batch

    cache = T.FeatureCache("x0").build(torch_head, _numpy_batches(config, videos), device)
    at = 0
    for batch in _numpy_batches(config, videos):
        keep = np.flatnonzero(np.asarray(batch["batch_mask"]) == 1)
        if not len(keep):
            continue
        with torch.no_grad():
            live = torch_head.trunk_feats(to_torch_batch(batch, device))[keep]
            cached = cache.features(torch_head, cache._batch(
                torch.arange(at, at + len(keep)), device))
        torch.testing.assert_close(cached, live, rtol=1e-5, atol=1e-4)
        at += len(keep)


@pytest.mark.parametrize("level", ["feats", "x0"])
def test_cached_loss_matches_the_live_loss(level, config, videos, torch_head, device, columns):
    """The objective, not just the features: labels and masks survive the round trip."""
    from hidra.torch.infer import to_torch_batch

    cache = T.FeatureCache(level).build(torch_head, _numpy_batches(config, videos), device)
    loss_fn = cache.loss_fn(columns)

    at = 0
    compared = 0
    for batch in _numpy_batches(config, videos):
        keep = np.flatnonzero(np.asarray(batch["batch_mask"]) == 1)
        if not len(keep):
            continue
        tb = to_torch_batch(batch, device)
        for key in T.FeatureCache.LABEL_KEYS[:-1] + ("batch_mask",):
            tb[key] = torch.as_tensor(np.asarray(batch[key]), device=device)
        live, _ = T.perlab_loss(torch_head, tb, head_columns=columns)
        cached, _ = loss_fn(torch_head, cache._batch(torch.arange(at, at + len(keep)), device))
        if not torch.isnan(live):
            # Round-off only; at the x0 level the tail reruns on a different batch size.
            torch.testing.assert_close(cached, live, rtol=1e-5, atol=1e-6)
            compared += 1
        at += len(keep)
    assert compared, "no batch carried supervision for these columns"


def test_dropped_padding_leaves_only_real_windows(config, videos, torch_head, device):
    """`data.batch` pads a short final batch with uninitialized memory flagged
    `batch_mask=0`. Those rows must never enter the cache, or they would train on garbage
    that the live path's mask discards."""
    cache = T.FeatureCache("feats").build(torch_head, _numpy_batches(config, videos), device)
    n_real = sum(int((np.asarray(b["batch_mask"]) == 1).sum())
                 for b in _numpy_batches(config, videos))
    assert cache.n == n_real < cache.n_batches * BATCH
    assert torch.isfinite(cache.feats).all()
    one = cache._batch(torch.arange(min(BATCH, cache.n)), device)
    assert int(one["batch_mask"].sum()) == min(BATCH, cache.n)


def test_passes_add_fresh_draws_of_the_same_windows(config, videos, torch_head, device):
    """More `--cache-passes` means more augmentation draws, the knob that trades memory for
    augmentation variety. Each pass is one epoch of the dataset, and an epoch's window
    count is itself random -- the time-dilation draw decides how many chunks a video is cut
    into -- so a second pass adds a comparable number of windows, not exactly the same."""
    one = T.FeatureCache("feats").build(torch_head, _numpy_batches(config, videos, 1), device)
    two = T.FeatureCache("feats").build(torch_head, _numpy_batches(config, videos, 2), device)
    assert one.n < two.n < 3 * one.n
    assert two.nbytes > one.nbytes
    # The first pass is the same draw in both; the rest is new.
    assert torch.equal(two.feats[:one.n], one.feats)
    assert not torch.equal(two.feats[:one.n], two.feats[one.n:2 * one.n])


def test_train_batches_cycle_every_window(config, videos, torch_head, device):
    """An epoch of cached batches must cover the cache, so no window is quietly never seen."""
    cache = T.FeatureCache("feats").build(torch_head, _numpy_batches(config, videos), device)
    stream = cache.train_batches(BATCH, device, seed=0)
    seen = set()
    for _ in range(3 * max(cache.n // BATCH, 1)):
        b = next(stream)
        assert b["feats"].shape[0] == BATCH
        for row in b["feats"].to("cpu"):
            seen.add(row.numpy().tobytes())
    assert len(seen) >= cache.n - BATCH, f"only {len(seen)} of {cache.n} windows were sampled"


def test_evaluate_scores_like_the_live_predictions(config, videos, torch_head, device):
    """`FeatureCache.evaluate` must agree with running `Predictions` over the live model --
    it is the metric checkpoint selection and early stopping read."""
    from hidra.torch.infer import to_torch_batch, unbatch_numpy

    cache = T.FeatureCache("feats").build(torch_head, _numpy_batches(config, videos), device,
                                          keep_elements=True)
    cached = cache.evaluate(torch_head, videos, BATCH, device)

    predictions = data.Predictions(videos)
    with torch.no_grad():
        for batch in _numpy_batches(config, videos):
            probs = torch_head.predict(to_torch_batch(batch, device))
            probs = probs.detach().to("cpu", torch.float32).numpy()
            for i in range(probs.shape[0]):
                predictions.update(unbatch_numpy(batch, i), probs[i])
    live, _ = predictions.score()

    for key in ("f1", "nll"):
        assert np.isclose(cached[key], float(live[key]), rtol=1e-6, atol=1e-6), key


# ------------------------------------------------------------------ through the CLI

@pytest.mark.slow
def test_cache_features_trains_only_the_head_through_the_cli(tmp_path, track_dir):
    """`finetune.py train --cache-features` end to end: a checkpoint in which only the
    trained layer moved. The cache must not leak gradient into a frozen layer, and it must
    not quietly skip training either."""
    import pickle
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bouts = tmp_path / "bouts.csv"
    rows = []
    rng = np.random.default_rng(5)
    for pq in sorted(track_dir.glob("*.parquet")):
        n = int(pd.read_parquet(pq, columns=["video_frame"]).video_frame.max() + 1)
        for mouse in ("mouse1", "mouse2"):
            t = int(rng.integers(30, 90))
            while t < n - 60:
                dur = int(rng.integers(15, 45))
                rows.append(dict(file=pq.name, agent=mouse, target=mouse, action="rear",
                                 start_frame=t, stop_frame=t + dur))
                t += dur + int(rng.integers(70, 180))
    pd.DataFrame(rows).to_csv(bouts, index=False)

    staged, models = tmp_path / "ft_data", tmp_path / "ft_models"
    for args in (
        ["prepare", "--tracking", str(track_dir), "--annotations", str(bouts),
         "--lab", LAB, "--out", str(staged), "--pix-per-cm", "16", "--fps", "30"],
        ["train", "--data", str(staged), "--lab", LAB, "--actions", "rear",
         "--out", str(models), "--tag", "cached", "--mode", "head", "--steps", "60",
         "--configs", "15fps_5bp", "--eval-interval", "1000", "--cache-features",
         "--cache-passes", "2", "--ddi-steps", "0", "--workdir", str(tmp_path / "work")],
    ):
        res = subprocess.run([sys.executable, os.path.join(repo, "finetune.py"), *args],
                             capture_output=True, text=True, timeout=1800, cwd=repo)
        assert res.returncode == 0, f"{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    assert "[CACHE]" in res.stdout, res.stdout[-2000:]
    losses = [float(l.split("nll_perlab:")[1].split()[0])
              for l in res.stdout.splitlines() if "nll_perlab:" in l]
    assert len(losses) >= 2 and losses[-1] < losses[0], f"loss did not fall: {losses}"

    from hidra import paths
    from hidra.checkpoints import flatten_tree, load_flat_checkpoint

    published = load_flat_checkpoint(
        paths.models_dir() / "15fps_5bp_supervised_perlab_sniffall.safetensors")
    with open(models / "15fps_5bp__cached.pkl", "rb") as f:
        trained = flatten_tree(pickle.load(f))

    def moved(key):
        a = np.asarray(published[key], np.float64)
        b = np.asarray(trained[key], np.float64)
        return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))

    weights = [k for k in trained if k.rsplit("/", 1)[-1] in ("w", "s", "b", "c")]
    head = max(moved(k) for k in weights if k.startswith("out-proj-perlab/"))
    frozen = max(moved(k) for k in weights if not k.startswith("out-proj-perlab/"))
    assert head > 1e-3, f"the trained head barely moved ({head:.2e})"
    assert frozen < 1e-5, f"a frozen layer moved by {frozen:.2e} -- gradient leaked"
