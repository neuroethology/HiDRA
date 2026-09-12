"""End to end: give GroovyShrew an `attack` column with `finetune.py train --new-head`, then run
everything downstream on it -- `predict.py --weights`, `--list-heads`, `calibrate`,
`hidra-convert-weights --one`, and the JAX backend.

One short training run (one config, a few dozen steps) is shared by the tests in this module;
each then checks one downstream consumer. Synthetic pose from `tests/synth.py`, synthetic bouts.
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

from hidra import head_table as H
from hidra import schema

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB, NEW_ACTION = "GroovyShrew", "attack"
SEED_FROM = "LyricalHare,attack"
CONFIG = "15fps_5bp"

pytestmark = [pytest.mark.slow, pytest.mark.gpu, requires_weights, requires_cuda]


def _run(args, env=None, timeout=3600):
    res = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=REPO,
                         env={**os.environ, **(env or {})}, timeout=timeout)
    assert res.returncode == 0, f"{' '.join(map(str, args))}\n{res.stdout[-4000:]}\n{res.stderr[-4000:]}"
    return res.stdout


def _write_bouts(track_dir, path, seed=11):
    """`attack` on both directed pairs and `rear` on both mice, so the new column (social) and
    an existing one (self-directed) both get supervision and every mouse is annotated."""
    rng = np.random.default_rng(seed)
    rows = []
    for pq in sorted(track_dir.glob("*.parquet")):
        n = int(pd.read_parquet(pq, columns=["video_frame"]).video_frame.max() + 1)
        for agent, target, action in (("mouse1", "mouse2", NEW_ACTION), ("mouse2", "mouse1", NEW_ACTION),
                                      ("mouse1", "mouse1", "rear"), ("mouse2", "mouse2", "rear")):
            t = int(rng.integers(20, 80))
            while t < n - 60:
                dur = int(rng.integers(15, 45))
                rows.append(dict(file=pq.name, agent=agent, target=target, action=action,
                                 start_frame=t, stop_frame=t + dur))
                t += dur + int(rng.integers(60, 150))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def new_head_run(track_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("newhead")
    bouts = _write_bouts(track_dir, out / "bouts.csv")
    staged, models = out / "ft_data", out / "ft_models"

    stdout = _run([os.path.join(REPO, "finetune.py"), "prepare", "--tracking", str(track_dir),
                   "--annotations", str(bouts), "--lab", LAB, "--new-head", f"{LAB},{NEW_ACTION}",
                   "--out", str(staged), "--pix-per-cm", "16", "--fps", "30"])
    assert "NEW column" in stdout, stdout[-2000:]

    stdout = _run([os.path.join(REPO, "finetune.py"), "train", "--data", str(staged), "--lab", LAB,
                   "--new-head", f"{LAB},{NEW_ACTION}", "--seed-from", SEED_FROM,
                   "--out", str(models), "--tag", "newhead", "--mode", "head", "--steps", "60",
                   "--configs", CONFIG, "--eval-interval", "1000", "--backend", "torch",
                   "--workdir", str(out / "work")],
                  env={"LABTAIL_DDI_STEPS": "8"})
    assert f"[NEW HEAD] ({LAB}, {NEW_ACTION}) -> column" in stdout, stdout[-3000:]
    assert "trained 1/1 config" in stdout, stdout[-2000:]
    return dict(out=out, bouts=bouts, models=models, ckpt=models / f"{CONFIG}__newhead.pkl",
                template=str(models / "{config}__newhead.pkl"), train_stdout=stdout)


@pytest.fixture(scope="module")
def torch_pred(new_head_run, track_dir, tmp_path_factory):
    """`predict.py --weights` on the torch backend, asking for the new head by name."""
    dest = tmp_path_factory.mktemp("pred_torch")
    stdout = _run([os.path.join(REPO, "predict.py"), str(track_dir), "--out", str(dest),
                   "--labs", LAB, "--actions", f"{NEW_ACTION},rear", "--configs", CONFIG,
                   "--weights", new_head_run["template"], "--pix-per-cm", "16", "--fps", "30"])
    assert "inference exited" not in stdout, stdout[-3000:]
    return dest


def _frames(dest):
    fs = sorted(dest.glob("*.frames.parquet"))
    assert fs, f"no probability tracks in {dest}"
    return pd.concat([pd.read_parquet(f).assign(video=f.name) for f in fs], ignore_index=True)


# ------------------------------------------------------------------ the checkpoint

def test_checkpoint_is_one_column_wider_and_carries_its_table(new_head_run):
    from hidra.checkpoints import flatten_tree, load_flat_checkpoint
    from hidra.torch.models import build_unsupervised, load_perlab, perlab_checkpoint_path

    ckpt = new_head_run["ckpt"]
    assert ckpt.is_file()
    published_table = schema.lab_action_table()
    table = H.extend_table(published_table, [(LAB, NEW_ACTION)])
    assert H.load_head_table(ckpt) == table, "the sidecar must list the extended table"
    j = table.index((LAB, NEW_ACTION))

    with open(ckpt, "rb") as f:
        trained = flatten_tree(pickle.load(f))
    published = load_flat_checkpoint(perlab_checkpoint_path(CONFIG))
    assert set(trained) == set(published)
    assert trained["out-proj-perlab/w"].shape == (256, 83)

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))

    # The published columns sit at their new positions, moved only by the EMA round trip
    # (`sums / (1 - decay**t)` in float32), never by a gradient -- the mask was one column.
    for key in ("w", "s", "b"):
        old, new = published[f"out-proj-perlab/{key}"], trained[f"out-proj-perlab/{key}"]
        for i, pair in enumerate(published_table):
            assert rel(new[..., table.index(pair)], old[..., i]) < 1e-5, (key, pair)
        # ... and the new column has moved away from its seed.
        seeded = old[..., published_table.index(tuple(SEED_FROM.split(",")))]
        if key == "w":
            assert rel(new[..., j], seeded) > 1e-3, "the new column did not train"
    # Everything outside the head is untouched (mode=head).
    frozen = [k for k in trained if not k.startswith("out-proj-perlab/")
              and k.rsplit("/", 1)[-1] in ("w", "s", "b", "c")]
    assert max(rel(trained[k], published[k]) for k in frozen) < 1e-5

    # The loss came down over the run.
    losses = [float(line.split("nll_perlab:")[1].split()[0])
              for line in new_head_run["train_stdout"].splitlines() if "nll_perlab:" in line]
    assert len(losses) >= 2 and losses[-1] < losses[0], losses

    head = load_perlab(CONFIG, build_unsupervised(schema.get_configs()[CONFIG]), path=ckpt)
    assert head.n_heads == 83 and head.lab_action == table
    assert head.P[schema.LABS.value_to_idx[LAB], schema.ACTIONS.value_to_idx[NEW_ACTION], j] == 1


# ------------------------------------------------------------------ inference and thresholds

def test_predict_runs_and_lists_the_new_head(new_head_run, torch_pred):
    fr = _frames(torch_pred)
    attack = fr[(fr.lab == LAB) & (fr.action == NEW_ACTION)]
    assert len(attack) > 0, "the new head's probability track was filtered out"
    assert set(zip(attack.subject, attack.target, strict=True)) == {("mouse1", "mouse2"), ("mouse2", "mouse1")}
    assert attack.prob.between(0, 1).all()
    assert (fr[(fr.lab == LAB) & (fr.action == "rear")].target == "self").all()
    assert all((d / f"{v[: -len('.frames.parquet')]}.bouts.csv").is_file()
               for d, v in ((torch_pred, v) for v in fr.video.unique()))

    stdout = _run([os.path.join(REPO, "predict.py"), "--list-heads", "--weights", new_head_run["template"],
                   "--configs", CONFIG])
    assert "83 classifier heads" in stdout
    line = next(l for l in stdout.splitlines() if l.strip().startswith(LAB))
    assert NEW_ACTION in line.split()
    assert f"{LAB}/{NEW_ACTION}" in stdout and "declared by --weights" in stdout


def test_calibrate_writes_the_new_heads_row(new_head_run, torch_pred, tmp_path):
    thr = tmp_path / "ft_thresholds.csv"
    stdout = _run([os.path.join(REPO, "finetune.py"), "calibrate", "--frames", str(torch_pred),
                   "--annotations", str(new_head_run["bouts"]), "--out", str(thr),
                   "--min-positives", "1"])
    assert f"{LAB}__{NEW_ACTION}" in stdout
    d = pd.read_csv(thr)
    rows = dict(zip(d[d.columns[0]], d[d.columns[1]], strict=True))
    assert f"{LAB}__{NEW_ACTION}" in rows and 0 < rows[f"{LAB}__{NEW_ACTION}"] < 1
    assert len(rows) > 82, "the bundled rows must be merged in"

    # With its row in the thresholds CSV the head is listed with no --weights at all.
    stdout = _run([os.path.join(REPO, "predict.py"), "--list-heads", "--thresholds", str(thr)])
    line = next(l for l in stdout.splitlines() if l.strip().startswith(LAB))
    assert NEW_ACTION in line.split()


# ------------------------------------------------------------------ the other container, the other backend

def test_converted_safetensors_predicts_identically(new_head_run, torch_pred, track_dir, tmp_path):
    from hidra.checkpoints import read_metadata

    stdout = _run(["-m", "hidra.torch.convert", "--one", str(new_head_run["ckpt"])])
    assert "83-column head table in metadata" in stdout
    st = new_head_run["ckpt"].with_suffix(".safetensors")
    assert json.loads(read_metadata(st)["lab_action"]) == [list(p) for p in H.load_head_table(new_head_run["ckpt"])]

    dest = tmp_path / "pred_st"
    _run([os.path.join(REPO, "predict.py"), str(track_dir), "--out", str(dest),
          "--labs", LAB, "--actions", NEW_ACTION, "--configs", CONFIG,
          "--weights", str(new_head_run["models"] / "{config}__newhead.safetensors"),
          "--pix-per-cm", "16", "--fps", "30"])
    a = _frames(torch_pred)
    a = a[a.action == NEW_ACTION].sort_values(["video", "subject", "target", "frame"], ignore_index=True)
    b = _frames(dest).sort_values(["video", "subject", "target", "frame"], ignore_index=True)
    assert len(a) == len(b) > 0
    np.testing.assert_array_equal(a.prob.to_numpy(), b.prob.to_numpy())


@pytest.mark.jax
@requires_jax
def test_jax_backend_loads_the_widened_checkpoint(new_head_run, torch_pred, track_dir, tmp_path):
    """The JAX loader must build the head at the checkpoint's width from the same sidecar."""
    dest = tmp_path / "pred_jax"
    stdout = _run([os.path.join(REPO, "predict.py"), str(track_dir), "--out", str(dest),
                   "--labs", LAB, "--actions", NEW_ACTION, "--configs", CONFIG, "--backend", "jax",
                   "--weights", new_head_run["template"], "--pix-per-cm", "16", "--fps", "30"])
    assert "inference exited" not in stdout, stdout[-3000:]
    a = _frames(torch_pred)
    a = a[a.action == NEW_ACTION].sort_values(["video", "subject", "target", "frame"], ignore_index=True)
    b = _frames(dest).sort_values(["video", "subject", "target", "frame"], ignore_index=True)
    assert len(a) == len(b) > 0
    diff = np.abs(a.prob.to_numpy(np.float64) - b.prob.to_numpy(np.float64)).max()
    assert diff < 2e-3, f"backends disagree on the new head: {diff:.3e}"
