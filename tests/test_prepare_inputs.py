"""`finetune.py prepare` on the two annotation layouts, and on behaviours scored but unseen.

`HiDRA_finetune.zip` took a folder of per-video annotation parquets and had an
`--also-scored` flag; the repo took one bout CSV and had neither. Both are accepted now, so
a dataset already in the bundle's layout needs no conversion and "we watched for this and
saw none" can still be said. Staging is pure pandas: no weights, no GPU.
"""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = "CRIM13"                       # a head-free slot, so --new-head carries the behaviours
QUIET = "synth01"                    # the recording given no social bouts below


def _prepare(*args, expect=0):
    res = subprocess.run([sys.executable, os.path.join(REPO, "finetune.py"), "prepare", *args],
                         capture_output=True, text=True, cwd=REPO, timeout=900)
    assert res.returncode == expect, f"{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    return res.stdout + res.stderr


def _manifest(staged):
    m = pd.read_csv(os.path.join(staged, "TRAIN.csv"))
    return {r["file"]: sorted(json.loads(r["behaviors_labeled"])) for _, r in m.iterrows()}


@pytest.fixture(scope="module")
def annotations(track_dir, tmp_path_factory):
    """The same bouts written both ways: one CSV, and one parquet per recording.

    `rear` on both mice everywhere; `sniff` on both directed pairs except in QUIET, which is
    where `--also-scored` has something to say.
    """
    from hidra.cli import vid_of

    out = tmp_path_factory.mktemp("annot")
    adir = out / "dir"
    adir.mkdir()
    rows = []
    for pq in sorted(track_dir.glob("*.parquet")):
        stem = pq.stem
        for mouse in ("mouse1", "mouse2"):
            rows.append(dict(file=pq.name, agent=mouse, target="self", action="rear",
                             start_frame=100, stop_frame=160))
        if stem != QUIET:
            for a, b in (("mouse1", "mouse2"), ("mouse2", "mouse1")):
                rows.append(dict(file=pq.name, agent=a, target=b, action="sniff",
                                 start_frame=300, stop_frame=360))
    csv = out / "bouts.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    for pq in sorted(track_dir.glob("*.parquet")):
        a = [r for r in rows if r["file"] == pq.name]
        # named by video id, as the bundle names both sides
        pd.DataFrame(dict(agent_id=[r["agent"] for r in a], target_id=[r["target"] for r in a],
                          action=[r["action"] for r in a],
                          start_frame=[r["start_frame"] for r in a],
                          stop_frame=[r["stop_frame"] for r in a])
                     ).to_parquet(adir / f"{vid_of(str(pq))}.parquet")
    return dict(csv=str(csv), dir=str(adir), out=out)


NEW_HEADS = ["--new-head", f"{LAB},rear", "--new-head", f"{LAB},sniff"]


def test_a_folder_of_annotation_parquets_stages_like_the_csv(track_dir, annotations):
    """The bundle's input layout and the repo's must produce the same staged dataset."""
    from_csv = annotations["out"] / "from_csv"
    from_dir = annotations["out"] / "from_dir"
    for src, dest in ((annotations["csv"], from_csv), (annotations["dir"], from_dir)):
        _prepare("--tracking", str(track_dir), "--annotations", src, "--lab", LAB, *NEW_HEADS,
                 "--out", str(dest), "--pix-per-cm", "16", "--fps", "30")
    assert _manifest(from_csv) == _manifest(from_dir)
    a = pd.read_csv(from_csv / "TRAIN.csv").sort_values("video_id", ignore_index=True)
    b = pd.read_csv(from_dir / "TRAIN.csv").sort_values("video_id", ignore_index=True)
    pd.testing.assert_frame_equal(a, b)


def test_also_scored_adds_combinations_and_stages_bout_free_videos(track_dir, annotations):
    """A recording with no social bouts contributes nothing for them by default, and becomes
    all-negative once `--also-scored` says it was watched."""
    without = annotations["out"] / "without"
    with_ = annotations["out"] / "with"
    _prepare("--tracking", str(track_dir), "--annotations", annotations["csv"], "--lab", LAB,
             *NEW_HEADS, "--out", str(without), "--pix-per-cm", "16", "--fps", "30")
    stdout = _prepare("--tracking", str(track_dir), "--annotations", annotations["csv"],
                      "--lab", LAB, *NEW_HEADS,
                      "--also-scored", "mouse1,mouse2,sniff;mouse2,mouse1,sniff",
                      "--out", str(with_), "--pix-per-cm", "16", "--fps", "30")
    assert "--also-scored: 2 combination(s)" in stdout, stdout[-2000:]

    quiet = f"{QUIET}.parquet"
    assert _manifest(without)[quiet] == ["mouse1,self,rear", "mouse2,self,rear"]
    assert _manifest(with_)[quiet] == ["mouse1,mouse2,sniff", "mouse1,self,rear",
                                       "mouse2,mouse1,sniff", "mouse2,self,rear"]
    # Videos that already had the combinations are unchanged.
    other = next(k for k in _manifest(without) if k != quiet)
    assert _manifest(without)[other] == _manifest(with_)[other]


def test_also_scored_sniffall_is_staged_as_sniff(track_dir, annotations):
    """The trainer derives the merged sniffall channel from the sniff family, so a combination
    recorded literally as `sniffall` would carry zero mask -- the same trap bout rows have."""
    dest = annotations["out"] / "sniffall"
    stdout = _prepare("--tracking", str(track_dir), "--annotations", annotations["csv"],
                      "--lab", "GroovyShrew",           # its sniff head IS the merged sniffall
                      "--also-scored", "mouse1,mouse2,sniffall",
                      "--out", str(dest), "--pix-per-cm", "16", "--fps", "30")
    assert "--also-scored 'sniffall' combination(s) as 'sniff'" in stdout, stdout[-2000:]
    assert not any("sniffall" in c for combos in _manifest(dest).values() for c in combos)
    assert any("mouse1,mouse2,sniff" in combos for combos in _manifest(dest).values())


def test_also_scored_refuses_what_the_lab_cannot_train(track_dir, annotations):
    out = _prepare("--tracking", str(track_dir), "--annotations", annotations["csv"],
                   "--lab", LAB, *NEW_HEADS, "--also-scored", "mouse1,mouse2,tailrattle",
                   "--out", str(annotations["out"] / "bad"), "--pix-per-cm", "16", "--fps", "30",
                   expect=1)
    assert "not one this lab can train" in out, out[-2000:]
