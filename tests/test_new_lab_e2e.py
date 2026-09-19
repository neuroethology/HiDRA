"""End to end: claim a head-free lab slot as your own lab.

This is the workflow `HiDRA_finetune.zip` documents and does not implement — its trainer
never widens the head, so the slot's loss mask is empty and the checkpoint comes out at the
published 82 columns. Here the whole chain is exercised: stage annotations under a slot,
widen the head by two columns seeded from a donor lab, retrain the tail, and run
`predict.py` against the result.

One short training run (one config, a few dozen steps, cached features) is shared by the
tests below. Synthetic pose from `tests/synth.py`, synthetic bouts.
"""
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import requires_cuda, requires_weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOT = "CRIM13"                      # a lab-embedding row with no published head
DONOR = "GroovyShrew"                # ... whose rear and sniffall heads seed it
ACTIONS = ["rear", "sniffall"]
CONFIG = "15fps_5bp"

pytestmark = [pytest.mark.slow, pytest.mark.gpu, requires_weights, requires_cuda]


def _run(args, timeout=3600):
    res = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=REPO,
                         timeout=timeout)
    assert res.returncode == 0, f"{' '.join(map(str, args))}\n{res.stdout[-4000:]}\n{res.stderr[-4000:]}"
    return res.stdout


@pytest.fixture(scope="module")
def new_lab_run(track_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("newlab")
    bouts = out / "bouts.csv"
    rng = np.random.default_rng(23)
    rows = []
    for pq in sorted(track_dir.glob("*.parquet")):
        n = int(pd.read_parquet(pq, columns=["video_frame"]).video_frame.max() + 1)
        for agent, target, action in (("mouse1", "mouse1", "rear"), ("mouse2", "mouse2", "rear"),
                                      ("mouse1", "mouse2", "sniffall"),
                                      ("mouse2", "mouse1", "sniffall")):
            t = int(rng.integers(20, 80))
            while t < n - 60:
                dur = int(rng.integers(15, 45))
                rows.append(dict(file=pq.name, agent=agent, target=target, action=action,
                                 start_frame=t, stop_frame=t + dur))
                t += dur + int(rng.integers(60, 150))
    pd.DataFrame(rows).to_csv(bouts, index=False)

    staged, models = out / "ft_data", out / "ft_models"
    new_head = [a for action in ACTIONS for a in ("--new-head", f"{SLOT},{action}")]
    stdout = _run([os.path.join(REPO, "finetune.py"), "prepare", "--tracking", str(track_dir),
                   "--annotations", str(bouts), "--lab", SLOT, *new_head,
                   "--out", str(staged), "--pix-per-cm", "16", "--fps", "30"])
    assert "NEW columns" in stdout, stdout[-2000:]

    stdout = _run([os.path.join(REPO, "finetune.py"), "train", "--data", str(staged),
                   "--lab", SLOT, *new_head, "--seed-from", DONOR, "--mode", "tail",
                   "--cache-features", "--cache-passes", "2", "--ddi-steps", "0",
                   "--out", str(models), "--tag", "newlab", "--steps", "40",
                   "--configs", CONFIG, "--eval-interval", "1000",
                   "--workdir", str(out / "work")])
    for action in ACTIONS:
        assert f"[NEW HEAD] ({SLOT}, {action})" in stdout, stdout[-3000:]
    assert f"lab-embedding row of {SLOT} seeded from {DONOR}" in stdout, stdout[-3000:]
    assert "trained 1/1 config" in stdout, stdout[-2000:]
    return dict(models=models, ckpt=models / f"{CONFIG}__newlab.pkl",
                template=str(models / "{config}__newlab.pkl"), bouts=bouts)


def test_the_slot_gets_its_own_columns_seeded_from_the_donor(new_lab_run):
    """Two columns wider than the published head, and every published column untouched."""
    from hidra import head_table as H
    from hidra import paths, schema
    from hidra.checkpoints import flatten_tree, load_flat_checkpoint

    published = load_flat_checkpoint(
        paths.models_dir() / f"{CONFIG}_supervised_perlab_sniffall.safetensors")
    with open(new_lab_run["ckpt"], "rb") as f:
        trained = flatten_tree(pickle.load(f))

    old_table = schema.lab_action_table()
    new_table = H.head_table_for(new_lab_run["ckpt"])
    assert len(new_table) == len(old_table) + 2
    assert [p for p in new_table if p[0] == SLOT] == sorted((SLOT, a) for a in ACTIONS)

    w_old = np.asarray(published["out-proj-perlab/w"])
    w_new = np.asarray(trained["out-proj-perlab/w"])
    assert w_new.shape == (w_old.shape[0], len(new_table))
    # `tail` mode trains the tail, so only the head's *published* columns are checked here:
    # they carry no gradient of their own (the loss is masked to the slot's two columns).
    for i, pair in enumerate(old_table):
        j = new_table.index(pair)
        moved = np.abs(w_new[:, j] - w_old[:, i]).max() / max(np.abs(w_old[:, i]).max(), 1e-30)
        assert moved < 1e-5, f"published column {pair} moved by {moved:.2e}"
    # The new columns did move -- they started at the donor's and were trained.
    for action in ACTIONS:
        j = new_table.index((SLOT, action))
        i = old_table.index((DONOR, action))
        assert np.abs(w_new[:, j] - w_old[:, i]).max() > 1e-4, action

    sidecar = new_lab_run["ckpt"].with_suffix(".heads.json")
    assert sidecar.is_file()
    with open(sidecar) as f:
        assert json.load(f)["n_heads"] == len(new_table)


def test_predict_runs_the_slot_as_a_lab(new_lab_run, track_dir, tmp_path):
    """`predict.py --labs <slot> --weights ...` produces tracks; without the weights it refuses."""
    dest = tmp_path / "preds"
    stdout = _run([os.path.join(REPO, "predict.py"), str(track_dir), "--out", str(dest),
                   "--labs", SLOT, "--actions", ",".join(ACTIONS), "--configs", CONFIG,
                   "--weights", new_lab_run["template"], "--pix-per-cm", "16", "--fps", "30"])
    assert "inference exited" not in stdout, stdout[-3000:]

    frames = pd.concat([pd.read_parquet(f) for f in sorted(dest.glob("*.frames.parquet"))],
                       ignore_index=True)
    assert set(frames["lab"]) == {SLOT}
    assert set(frames["action"]) == set(ACTIONS)
    assert frames["prob"].to_numpy().std() > 0, "an all-constant track means nothing was read"

    res = subprocess.run(
        [sys.executable, os.path.join(REPO, "predict.py"), str(track_dir),
         "--out", str(tmp_path / "nope"), "--labs", SLOT, "--actions", "rear",
         "--pix-per-cm", "16", "--fps", "30"],
        capture_output=True, text=True, cwd=REPO, timeout=600)
    assert res.returncode != 0 and SLOT in res.stdout + res.stderr, \
        "a head-free slot must be refused when no weights give it a head"


def test_calibrate_writes_the_slots_rows(new_lab_run, track_dir, tmp_path):
    """The slot's heads reach the thresholds CSV like any other, so later runs need no --weights
    to know about them."""
    dest = tmp_path / "preds"
    _run([os.path.join(REPO, "predict.py"), str(track_dir), "--out", str(dest),
          "--labs", SLOT, "--actions", ",".join(ACTIONS), "--configs", CONFIG,
          "--weights", new_lab_run["template"], "--pix-per-cm", "16", "--fps", "30"])
    thresholds = tmp_path / "ft_thresholds.csv"
    _run([os.path.join(REPO, "finetune.py"), "calibrate", "--frames", str(dest),
          "--annotations", str(new_lab_run["bouts"]), "--out", str(thresholds),
          "--min-positives", "50"])
    rows = pd.read_csv(thresholds)
    heads = set(rows[rows.columns[0]].astype(str))
    assert any(h.startswith(f"{SLOT}__") for h in heads), sorted(h for h in heads if SLOT in h)
    assert "GroovyShrew__rear" in heads, "uncalibrated heads must keep their published row"
