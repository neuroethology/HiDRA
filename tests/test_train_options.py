"""`finetune.py train`'s planning-time options: `--dry-run`, `--from-weights`, `--lr-schedule`,
`--cosine-steps`, `--patience`.

All of these are decided before a GPU is touched, so they can be checked by running the
command and reading the plan it prints. `--dry-run` is also the thing that makes that
possible, so it is both the subject and the instrument here.
"""
import os
import subprocess
import sys

import pandas as pd
import pytest
from conftest import requires_weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOT, DONOR = "CRIM13", "GroovyShrew"

pytestmark = [requires_weights]


def _run(args, expect=0):
    res = subprocess.run([sys.executable, os.path.join(REPO, "finetune.py"), *args],
                         capture_output=True, text=True, cwd=REPO, timeout=900)
    assert res.returncode == expect, f"{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    return res.stdout + res.stderr


@pytest.fixture(scope="module")
def staged(track_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("trainopts")
    bouts = out / "bouts.csv"
    rows = []
    for pq in sorted(track_dir.glob("*.parquet")):
        for mouse in ("mouse1", "mouse2"):
            rows.append(dict(file=pq.name, agent=mouse, target="self", action="rear",
                             start_frame=100, stop_frame=160))
        rows.append(dict(file=pq.name, agent="mouse1", target="mouse2", action="sniff",
                         start_frame=300, stop_frame=360))
    pd.DataFrame(rows).to_csv(bouts, index=False)
    d = out / "ft_data"
    _run(["prepare", "--tracking", str(track_dir), "--annotations", str(bouts), "--lab", SLOT,
          "--new-head", f"{SLOT},rear", "--new-head", f"{SLOT},sniffall",
          "--out", str(d), "--pix-per-cm", "16", "--fps", "30"])
    return dict(data=d, out=out)


def _plan(staged, *extra, expect=0):
    return _run(["train", "--data", str(staged["data"]), "--lab", SLOT,
                 "--new-head", f"{SLOT},rear", "--new-head", f"{SLOT},sniffall",
                 "--out", str(staged["out"] / "models"), "--tag", "plan",
                 "--configs", "15fps_5bp", "--dry-run", *extra], expect=expect)


def test_dry_run_prints_the_plan_and_writes_nothing(staged):
    out = _plan(staged, "--seed-from", DONOR, "--mode", "tail", "--cache-features")
    assert "--dry-run: nothing was trained" in out
    assert "LABTAIL_NEW_HEAD=CRIM13,rear;CRIM13,sniffall" in out
    assert f"LABTAIL_SEED_FROM={DONOR}" in out
    assert "LABTAIL_CACHE=1" in out
    assert "hidra.torch.train_perlab --config 15fps_5bp" in out
    assert not (staged["out"] / "models").exists(), "a dry run must not create the output dir"


def test_dry_run_still_refuses_what_a_real_run_would(staged):
    out = _plan(staged, "--seed-from", "NoSuchLab", expect=1)
    assert "not a lab with published heads" in out
    out = _plan(staged, "--new-head", f"{SLOT},tailrattle", expect=1)
    assert "not a behaviour in the vocabulary" in out


def test_lr_schedule_cosine_sets_the_bundles_horizon(staged):
    assert "LR_COSINE_T" not in _plan(staged, "--seed-from", DONOR)
    # max(steps, 15000), so a short run decays barely at all and the default 8000 ends near 45%.
    assert "LR_COSINE_T=15000" in _plan(staged, "--seed-from", DONOR, "--lr-schedule", "cosine")
    assert "LR_COSINE_T=20000" in _plan(staged, "--seed-from", DONOR, "--lr-schedule", "cosine",
                                        "--steps", "20000")


def test_cosine_steps_moves_the_horizon(staged):
    # the same number as --steps is a schedule that decays fully over the run
    assert "LR_COSINE_T=1000" in _plan(staged, "--seed-from", DONOR, "--lr-schedule", "cosine",
                                       "--steps", "1000", "--cosine-steps", "1000")
    out = _plan(staged, "--seed-from", DONOR, "--cosine-steps", "1000", expect=1)
    assert "--cosine-steps only applies with --lr-schedule cosine" in out


def test_patience_reaches_the_trainer(staged):
    assert "LABTAIL_PATIENCE" not in _plan(staged, "--seed-from", DONOR)   # the trainer's 10000
    assert "LABTAIL_PATIENCE=0" in _plan(staged, "--seed-from", DONOR, "--patience", "0")
    out = _plan(staged, "--seed-from", DONOR, "--patience", "-1", expect=1)
    assert "--patience must be >= 0" in out


def test_from_weights_is_checked_before_anything_runs(staged, tmp_path):
    out = _plan(staged, "--seed-from", DONOR, "--from-weights", "nope.pkl", expect=1)
    assert "must contain a '{config}' placeholder" in out
    out = _plan(staged, "--seed-from", DONOR,
                "--from-weights", str(tmp_path / "{config}__absent.pkl"), expect=1)
    assert "has no checkpoint for config(s) ['15fps_5bp']" in out

    real = tmp_path / "15fps_5bp__there.pkl"
    real.write_bytes(b"not a real checkpoint, but it exists")
    out = _plan(staged, "--seed-from", DONOR, "--from-weights", str(tmp_path / "{config}__there.pkl"))
    assert "warm-starting from" in out
    assert f"HIDRA_PERLAB_CKPT={tmp_path}/{{config}}__there.pkl" in out
