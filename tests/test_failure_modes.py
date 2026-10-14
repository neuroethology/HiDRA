"""Failures that used to pass silently.

Each of these once ended with exit code 0 and output that looked like a result: an empty
ethogram after a crashed inference run, a folder wiped out by one single-mouse recording,
the previous run's probabilities reported as this one's, a job moved off the GPU its
scheduler assigned, and a header-less bouts.csv that the Python API could not read.
"""
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import requires_cuda, requires_jax, requires_weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = "15fps_5bp"


def gpu(fn):
    """Real inference or training: needs the weights and a CUDA device."""
    for mark in (pytest.mark.slow, pytest.mark.gpu, requires_weights, requires_cuda):
        fn = mark(fn)
    return fn


def _cli(script, *args):
    """One CLI invocation, as a user would type it: (returncode, stdout + stderr)."""
    res = subprocess.run([sys.executable, os.path.join(REPO, script), *map(str, args)],
                         capture_output=True, text=True, cwd=REPO, timeout=1800)
    return res.returncode, res.stdout + res.stderr


@pytest.fixture(scope="module")
def two_mice(tmp_path_factory):
    from synth import write_dataset

    out = tmp_path_factory.mktemp("two_mice")
    write_dataset(str(out), n_videos=1, n_frames=600, seed=3)
    return out


# ------------------------------------------------------------------ failures exit non-zero

@pytest.mark.slow
@requires_weights
def test_predict_exits_nonzero_when_inference_fails(two_mice, tmp_path):
    code, out = _cli("predict.py", two_mice, "--out", tmp_path / "out", "--labs", "GroovyShrew",
                     "--actions", "rear", "--configs", CONFIG,
                     "--weights", tmp_path / "missing" / "{config}.pkl")
    assert code != 0, out[-2500:]
    assert "inference failed" in out
    assert not list((tmp_path / "out").glob("*.bouts.csv")), "an empty ethogram was written"


@pytest.mark.slow
@requires_weights
def test_finetune_train_exits_nonzero_when_a_config_fails(two_mice, tmp_path):
    pd.DataFrame([dict(file="synth00.parquet", agent="mouse1", target="mouse2", action="approach",
                       start_frame=100, stop_frame=200)]).to_csv(tmp_path / "bouts.csv", index=False)
    code, out = _cli("finetune.py", "prepare", "--tracking", two_mice, "--annotations",
                     tmp_path / "bouts.csv", "--lab", "GroovyShrew", "--out", tmp_path / "ft")
    assert code == 0, out[-2500:]
    (tmp_path / f"{CONFIG}__corrupt.pkl").write_bytes(b"not a pickle")
    code, out = _cli("finetune.py", "train", "--data", tmp_path / "ft", "--lab", "GroovyShrew",
                     "--actions", "approach", "--out", tmp_path / "models", "--configs", CONFIG,
                     "--steps", "5", "--from-weights", tmp_path / "{config}__corrupt.pkl")
    assert "trained 0/1" in out, out[-2500:]
    assert code != 0


# ------------------------------------------------------------------ single-mouse recordings

def _supervised_keys(pose, labelled, tmp_path, monkeypatch):
    """The training/inference windows data.Dataset builds for one video, and the dataset."""
    from hidra import data, schema

    monkeypatch.setattr(data, "working_dir", str(tmp_path))
    row = pd.Series(dict(lab_id="GroovyShrew", video_id=1, mode="custom", frames_per_second=30.0,
                         pix_per_cm_approx=16.0, behaviors_labeled=json.dumps(labelled)))
    tracking = data.create_tracking_data(row, pose)
    row["tracked_mice"], row["num_frames"] = tracking.mouse_index, tracking.num_frames
    bouts = pd.DataFrame([dict(agent_id="mouse1", target_id="self", action="rear",
                               start_frame=50, stop_frame=120)])
    video = data.Video(1, "GroovyShrew", tracking, data.create_labels(row, bouts), fps=30.0)
    cfg = schema.get_configs()[CONFIG]
    ds = data.Dataset(videos=[video], seq_len=64, sample_rate=cfg["sample_rate"], padding=32,
                      num_bodyparts=cfg["num_bodyparts"], num_epochs=1, unsupervised=False,
                      max_scale=cfg["max_scale"], max_time_dilation=cfg["max_time_dilation"],
                      rotate=True, flip=True, noise_scale=cfg["noise_scale"], num_workers=1, seed=[0])
    return next(ds.epochs_supervised()), ds


def test_a_lone_mouse_is_paired_with_an_unobserved_partner(tmp_path, monkeypatch):
    from synth import synth_video

    keys, ds = _supervised_keys(synth_video(n_frames=600, n_mice=1, seed=1),
                                ["mouse1,self,rear"], tmp_path, monkeypatch)
    assert len(keys) and (keys[:, 1] == keys[:, 2]).all()
    el = ds.get_element(keys[0], dict(element_idx=0, epoch_idx=0))
    assert np.isnan(el["target"]).all(), "the missing partner must read as unobserved"
    assert np.isfinite(el["agent"]).any()


def test_a_mouse_annotated_alone_is_paired_with_another_tracked_mouse(tmp_path, monkeypatch):
    from synth import synth_video

    keys, ds = _supervised_keys(synth_video(n_frames=600, n_mice=2, seed=1),
                                ["mouse1,self,rear"], tmp_path, monkeypatch)
    assert len(keys) and (keys[:, 1] != keys[:, 2]).all()
    el = ds.get_element(keys[0], dict(element_idx=0, epoch_idx=0))
    assert np.isfinite(el["target"]).any()


@gpu
def test_single_mouse_recording_is_predicted_next_to_a_pair(tmp_path):
    from synth import synth_video

    track = tmp_path / "track"
    track.mkdir()
    synth_video(n_frames=600, n_mice=1, seed=1).to_parquet(track / "solo.parquet", index=False)
    synth_video(n_frames=600, n_mice=2, seed=2).to_parquet(track / "pair.parquet", index=False)
    code, out = _cli("predict.py", track, "--out", tmp_path / "out", "--labs", "GroovyShrew",
                     "--actions", "rear,approach", "--configs", CONFIG, "--pix-per-cm", "16",
                     "--fps", "30")
    assert code == 0, out[-2500:]
    pair = pd.read_parquet(tmp_path / "out" / "pair.frames.parquet")
    solo = pd.read_parquet(tmp_path / "out" / "solo.frames.parquet")
    assert set(pair.action) == {"rear", "approach"}
    assert set(zip(solo.subject, solo.target, solo.action, strict=True)) == {("mouse1", "self", "rear")}
    assert solo.prob.between(0, 1).all() and len(solo) == 600


# ------------------------------------------------------------------ stale work dir

@gpu
def test_predict_recomputes_instead_of_reusing_a_stale_work_dir(two_mice, tmp_path):
    """A leftover `_work` (from --keep-work, or an interrupted run) must not be reported as
    this run's result."""
    import hidra
    from hidra import schema
    from hidra.paths import vid_of

    pq = next(two_mice.glob("*.parquet"))
    store = tmp_path / "out" / "_work" / "perlab_allbeh"
    store.mkdir(parents=True)
    with open(store / "GroovyShrew.pkl", "wb") as f:     # a leftover: constant 0.99
        pickle.dump({(vid_of(str(pq)), 0, 0, schema.ACTIONS.value_to_idx["rear"]):
                     np.full(600, 0.99, np.float16)}, f)
    code, out = _cli("predict.py", two_mice, "--out", tmp_path / "out", "--labs", "GroovyShrew",
                     "--actions", "rear", "--configs", CONFIG)
    assert code == 0, out[-2500:]
    assert not np.allclose(hidra.frames(str(tmp_path / "out")).prob, 0.99, atol=1e-3)


# ------------------------------------------------------------------ GPU assignment

@pytest.mark.parametrize("visible, gpu, expected", [
    (None, None, "0"),           # nothing set: GPU 0, as before
    ("3", None, "3"),            # a scheduler's assignment is kept
    ("", None, ""),              # CPU only
    ("3,5", "1", "5"),           # --gpu indexes the visible devices
    (None, "2", "2"),
])
def test_visible_gpu(monkeypatch, visible, gpu, expected):
    from hidra.cli import visible_gpu

    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert visible_gpu(gpu) == expected


def test_visible_gpu_rejects_an_index_past_the_visible_devices(monkeypatch):
    from hidra.cli import visible_gpu

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(SystemExit, match="only 1 GPU"):
        visible_gpu("1")


@requires_weights
def test_predict_hands_the_subprocess_the_schedulers_gpu(two_mice, tmp_path, monkeypatch):
    from hidra import cli

    seen = []

    def fake_run(cmd, env=None, **kw):
        seen.append(env)
        raise SystemExit("stop before inference")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        cli.run(str(two_mice), labs=["GroovyShrew"], actions=["rear"], out=str(tmp_path / "out"),
                configs=[CONFIG])
    assert seen[0]["CUDA_VISIBLE_DEVICES"] == "3"


# ------------------------------------------------------------------ zero-bout videos

@gpu
def test_a_video_with_no_bouts_reads_back_through_the_api(two_mice, tmp_path):
    import hidra

    code, out = _cli("predict.py", two_mice, "--out", tmp_path / "out", "--labs", "GroovyShrew",
                     "--actions", "rear", "--configs", CONFIG, "--threshold", "1.01")
    assert code == 0, out[-2500:]
    b = hidra.bouts(str(tmp_path / "out"))
    assert len(b) == 0 and {"subject", "action", "start_frame", "stop_frame"} <= set(b.columns)


# ------------------------------------------------------------------ small GPUs

@requires_jax
def test_jax_trainer_eval_batch_can_be_lowered(tmp_path, monkeypatch):
    from hidra import solution

    monkeypatch.chdir(tmp_path)

    def trainer():
        return solution.Trainer("t", model=None, optimizer=None, train_dataset=None,
                                train_batch_size=128)

    assert trainer().eval_batch_size == 512
    monkeypatch.setenv("HIDRA_EVAL_BATCH", "64")
    assert trainer().eval_batch_size == 64
