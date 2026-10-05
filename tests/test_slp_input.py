"""SLEAP `.slp` files as input, next to the parquets.

The claim under test is that an SLP is read as exactly the tracking parquet, annotation
parquet and manifest row it stands for: so the training arrays built from an SLP are
byte-identical to those built from the parquet layout, `finetune.py prepare` stages the same
dataset from SLPs alone as from parquets plus a bout CSV, and `predict.py` gives the same
probabilities. The reading rules themselves -- user over predicted instances, untracked
instances skipped, track names, inclusive events -- are pinned on hand-built files.

Everything but the last test is CPU-only and needs no weights.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio
from conftest import requires_cuda, requires_weights
from synth import synth_video, write_slp

from hidra import cli, data, slp
from hidra import finetune as ft

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIX, FPS = 16.0, 30.0
SNIFF = (("mouse1", "mouse2"), ("mouse2", "mouse1"))


def _bouts(stem, with_sniff):
    """`rear` on both mice; `sniff` on both directed pairs where `with_sniff`."""
    rows = [dict(agent=m, target="self", action="rear", start_frame=100, stop_frame=160)
            for m in ("mouse1", "mouse2")]
    if with_sniff:
        rows += [dict(agent=a, target=b, action="sniff", start_frame=300, stop_frame=360)
                 for a, b in SNIFF]
    return [dict(r, file=stem) for r in rows]


def _sniff_scored():
    return [f"{a},{b},sniff" for a, b in SNIFF]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Two recordings written both ways, with identical pose and bouts.

    `slp/` holds self-contained SLPs: provenance carries lab, id, split, scale, rate and what
    was scored -- including, in v1, `sniff` that was scored but never seen. `pq/` holds the
    parquet layout of the same two videos (`train.csv` + train_tracking + train_annotation),
    and `flat/` the same parquets as a predict.py/prepare input folder with a bout CSV.
    """
    root = tmp_path_factory.mktemp("slp_corpus")
    lab = "GroovyShrew"
    slp_dir, pq, flat = root / "slp" / lab, root / "pq", root / "flat"
    for d in (slp_dir, pq / "train_tracking" / lab, pq / "train_annotation" / lab, flat):
        d.mkdir(parents=True)
    rows, csv_rows = [], []
    for i, vid in enumerate((1001, 1002)):
        pose = synth_video(n_frames=900, seed=i)
        bouts = _bouts(f"v{i}", with_sniff=(i == 0))
        scored = sorted({f"{b['agent']},{b['target']},{b['action']}" for b in bouts}
                        | set(_sniff_scored()))
        write_slp(slp_dir / f"v{i}.slp", pose, bouts, fps=FPS, provenance=dict(
            lab_id=lab, video_id=vid, split="train", frames_per_second=FPS,
            pix_per_cm_approx=PIX, behaviors_labeled=scored))
        pose.to_parquet(pq / "train_tracking" / lab / f"{vid}.parquet", index=False)
        pd.DataFrame(dict(agent_id=[b["agent"] for b in bouts], target_id=[b["target"] for b in bouts],
                          action=[b["action"] for b in bouts],
                          start_frame=[b["start_frame"] for b in bouts],
                          stop_frame=[b["stop_frame"] for b in bouts])
                     ).to_parquet(pq / "train_annotation" / lab / f"{vid}.parquet", index=False)
        rows.append(dict(lab_id=lab, video_id=vid, frames_per_second=FPS, pix_per_cm_approx=PIX,
                         behaviors_labeled=json.dumps(scored)))
        pose.to_parquet(flat / f"v{i}.parquet", index=False)
        csv_rows += bouts
    pd.DataFrame(rows).to_csv(pq / "train.csv", index=False)
    pd.DataFrame(csv_rows)[ft.ANNOT_COLS].to_csv(root / "bouts.csv", index=False)
    # One test-split video, which the train manifest must leave out.
    write_slp(slp_dir / "held_out.slp", synth_video(n_frames=300, seed=9), [], fps=FPS,
              provenance=dict(lab_id=lab, video_id=1999, split="test", frames_per_second=FPS,
                              pix_per_cm_approx=PIX, behaviors_labeled=[]))
    return dict(root=root, slp=root / "slp", slp_lab=slp_dir, pq=pq, flat=flat,
                csv=root / "bouts.csv", lab=lab)


# ------------------------------------------------------------------ reading one file

def _labels(tracks=("mouse1", "mouse2"), n_videos=1):
    skel = sio.Skeleton(nodes=["nose", "tail_base"])
    videos = [sio.Video(filename=f"v{i}.mp4", backend=None) for i in range(n_videos)]
    return sio.Labels(videos=videos, skeletons=[skel], tracks=[sio.Track(name=t) for t in tracks]), skel


def test_pose_is_the_parquet_it_stands_for(corpus):
    rec = slp.read(corpus["slp_lab"] / "v0.slp")
    expect = synth_video(n_frames=900, seed=0)
    # A NaN keypoint is a missing row, which is what the parquet reader makes of a NaN row.
    expect = expect.dropna(subset=["x", "y"], how="all").reset_index(drop=True)
    pd.testing.assert_frame_equal(rec.pose, expect, check_dtype=False)
    assert rec.pose["x"].dtype == np.float64 and rec.pose["video_frame"].dtype == np.int64
    assert rec.notes == []


def test_user_instance_beats_prediction_and_untracked_is_skipped(tmp_path):
    labels, skel = _labels()
    t1, t2 = labels.tracks
    v = labels.videos[0]
    pred = sio.PredictedInstance.from_numpy(np.array([[1.0, 1.0], [2.0, 2.0]]), skeleton=skel,
                                            point_scores=np.ones(2), score=1.0, track=t1)
    user = sio.Instance.from_numpy(np.array([[5.0, 5.0], [np.nan, np.nan]]), skeleton=skel, track=t1)
    other = sio.PredictedInstance.from_numpy(np.array([[7.0, 7.0], [8.0, 8.0]]), skeleton=skel,
                                             point_scores=np.ones(2), score=1.0, track=t2)
    loose = sio.PredictedInstance.from_numpy(np.array([[9.0, 9.0], [9.0, 9.0]]), skeleton=skel,
                                             point_scores=np.ones(2), score=1.0)
    labels.append(sio.LabeledFrame(video=v, frame_idx=3, instances=[pred, user, other, loose]))
    labels.save(str(tmp_path / "a.slp"))

    rec = slp.read(tmp_path / "a.slp")
    got = {(r.mouse_id, r.bodypart): (r.x, r.y) for r in rec.pose.itertuples()}
    # mouse1 is the user's instance, whose missing tail_base is no row -- not the prediction's
    assert got == {("mouse1", "nose"): (5.0, 5.0), ("mouse2", "nose"): (7.0, 7.0),
                   ("mouse2", "tail_base"): (8.0, 8.0)}
    assert set(rec.pose["video_frame"]) == {3}
    assert any("1 untracked instance" in n for n in rec.notes)


def test_two_instances_of_one_kind_on_a_track_is_an_error(tmp_path):
    labels, skel = _labels()
    t1 = labels.tracks[0]
    a, b = (sio.Instance.from_numpy(np.zeros((2, 2)) + k, skeleton=skel, track=t1) for k in (1, 2))
    labels.append(sio.LabeledFrame(video=labels.videos[0], frame_idx=0, instances=[a, b]))
    labels.save(str(tmp_path / "a.slp"))
    with pytest.raises(ValueError, match="two user instances on track 'mouse1'"):
        slp.read(tmp_path / "a.slp")


def test_events_become_exclusive_bouts(tmp_path):
    labels, skel = _labels()
    t1, t2 = labels.tracks
    v = labels.videos[0]
    labels.append(sio.LabeledFrame(video=v, frame_idx=0, instances=[
        sio.Instance.from_numpy(np.zeros((2, 2)), skeleton=skel, track=t) for t in (t1, t2)]))
    labels.events.append(sio.UserEvent(type="rear", video=v, start_frame=10, end_frame=19, subject=t1))
    labels.events.append(sio.UserEvent(type="sniff", video=v, start_frame=40, subject=t2, target=t1))
    labels.events.append(sio.PredictedEvent(type="attack", video=v, start_frame=0, end_frame=5,
                                            subject=t1, target=t2, score=0.9))
    labels.events.append(sio.UserEvent(type="rear", video=v, start_frame=50, end_frame=60))
    labels.save(str(tmp_path / "a.slp"))

    for pose in (True, False):                      # the lazy read gives the same bouts
        rec = slp.read(tmp_path / "a.slp", pose=pose)
        assert rec.bouts.values.tolist() == [["mouse1", "self", "rear", 10, 20],
                                             ["mouse2", "mouse1", "sniff", 40, 41]]
        assert any("1 predicted event" in n for n in rec.notes)
        assert any("1 event(s) with no subject" in n for n in rec.notes)


def test_tracks_not_named_mouse_n_are_numbered_in_file_order(tmp_path):
    labels, skel = _labels(tracks=("track_0", "track_1"))
    t0, t1 = labels.tracks
    v = labels.videos[0]
    labels.append(sio.LabeledFrame(video=v, frame_idx=0, instances=[
        sio.Instance.from_numpy(np.full((2, 2), k), skeleton=skel, track=t)
        for k, t in ((1.0, t0), (2.0, t1))]))
    labels.events.append(sio.UserEvent(type="sniff", video=v, start_frame=0, subject=t1, target=t0))
    labels.provenance["behaviors_labeled"] = ["track_1,track_0,sniff", "track_0,self,rear"]
    labels.save(str(tmp_path / "a.slp"))

    rec = slp.read(tmp_path / "a.slp")
    assert rec.mice == {"track_0": "mouse1", "track_1": "mouse2"}
    assert rec.pose.groupby("mouse_id")["x"].first().to_dict() == {"mouse1": 1.0, "mouse2": 2.0}
    assert rec.bouts.values.tolist() == [["mouse2", "mouse1", "sniff", 0, 1]]
    assert rec.meta["behaviors_labeled"] == ["mouse2,mouse1,sniff", "mouse1,self,rear"]
    assert any("track_0 -> mouse1" in n for n in rec.notes)


def test_what_cannot_be_read_as_one_recording_is_an_error(tmp_path):
    labels, _ = _labels(tracks=[f"t{i}" for i in range(5)])
    labels.save(str(tmp_path / "five.slp"))
    with pytest.raises(ValueError, match="at most 4 mice"):
        slp.read(tmp_path / "five.slp")
    labels, _ = _labels(n_videos=2)
    labels.save(str(tmp_path / "two.slp"))
    for reader in (slp.read, slp.read_meta):
        with pytest.raises(ValueError, match="one video per SLP"):
            reader(tmp_path / "two.slp")


def test_meta_comes_from_provenance_and_the_video(tmp_path):
    labels, _ = _labels()
    labels.videos[0].fps = 25.0
    labels.provenance.update(lab_id="CRIM13", video_id=7, pix_per_cm=12.5,
                             behaviors_labeled=json.dumps(["mouse1,self,rear", "mouse1,self,rear"]))
    labels.save(str(tmp_path / "a.slp"))
    want = dict(lab_id="CRIM13", video_id=7, split=None, frames_per_second=25.0,
                pix_per_cm_approx=12.5, behaviors_labeled=["mouse1,self,rear"])
    assert slp.read_meta(tmp_path / "a.slp") == want
    assert slp.read(tmp_path / "a.slp", pose=False).meta == want


# ------------------------------------------------------------------ training

def _load(monkeypatch, dataset, work):
    monkeypatch.setattr(data, "dataset_dir", str(dataset))
    monkeypatch.setattr(data, "working_dir", str(work))
    return {v.video_id: v for v in data.load_videos("train", use_cached=False)}


def test_training_reads_an_slp_tree_as_the_parquet_layout(corpus, monkeypatch, tmp_path):
    from_pq = _load(monkeypatch, corpus["pq"], tmp_path / "w_pq")
    from_slp = _load(monkeypatch, corpus["slp"], tmp_path / "w_slp")
    assert sorted(from_slp) == sorted(from_pq) == [1001, 1002]     # the test-split file is out
    for vid, a in from_pq.items():
        b = from_slp[vid]
        assert b.slp and b.slp.endswith(".slp") and a.slp is None
        assert (a.num_frames, a.fps, a.lab_name) == (b.num_frames, b.fps, b.lab_name)
        assert list(a.tracking_data.mouse_index) == list(b.tracking_data.mouse_index)
        assert list(a.tracking_data.bodypart_index) == list(b.tracking_data.bodypart_index)
        assert open(a.tracking_data.path, "rb").read() == open(b.tracking_data.path, "rb").read()
        assert open(a.labels.path, "rb").read() == open(b.labels.path, "rb").read()
        assert sorted(a.labels.labeled_behaviors) == sorted(b.labels.labeled_behaviors)
        assert list(a.labels.label_index) == list(b.labels.label_index)


def test_a_manifest_row_can_point_at_an_slp(corpus, monkeypatch, tmp_path):
    """A train.csv with an `slp` column; with `behaviors_labeled` left blank the file's own
    scored list applies."""
    ds = tmp_path / "ds"
    ds.mkdir()
    pd.DataFrame([dict(lab_id=corpus["lab"], video_id=1001, frames_per_second=FPS,
                       pix_per_cm_approx=PIX, behaviors_labeled=None,
                       slp=str(corpus["slp_lab"] / "v1.slp"))]).to_csv(ds / "train.csv", index=False)
    (v,) = _load(monkeypatch, ds, tmp_path / "w").values()
    scored = set(json.loads(pd.read_csv(corpus["pq"] / "train.csv").behaviors_labeled[1]))
    from hidra.schema import ACTIONS, MOUSE_IDS
    got = {f"mouse{MOUSE_IDS.decode(a)},"
           f"{'self' if a == t else f'mouse{MOUSE_IDS.decode(t)}'},{ACTIONS.decode(x)}"
           for a, t, x in v.labels.labeled_behaviors}
    assert got == scored


def test_slp_manifest_needs_what_training_needs(tmp_path):
    labels, _ = _labels()
    labels.provenance.update(frames_per_second=30.0, pix_per_cm_approx=16.0)
    labels.save(str(tmp_path / "nolab.slp"))
    with pytest.raises(ValueError, match=r"nolab\.slp: \['lab_id'\]"):
        slp.manifest([tmp_path / "nolab.slp"])
    for name in ("a", "b"):
        labels.provenance.update(lab_id="CRIM13", video_id=5)
        labels.save(str(tmp_path / f"{name}.slp"))
    with pytest.raises(ValueError, match=r"video_id\(s\) \[5\]"):
        slp.manifest([tmp_path / "a.slp", tmp_path / "b.slp"])


# ------------------------------------------------------------------ finetune.py / predict.py

LAB = "CRIM13"                       # a head-free slot, so --new-head carries the behaviours
NEW_HEADS = ["--new-head", f"{LAB},rear", "--new-head", f"{LAB},sniff"]


def _prepare(*args, expect=0):
    res = subprocess.run([sys.executable, os.path.join(REPO, "finetune.py"), "prepare", *args],
                         capture_output=True, text=True, cwd=REPO, timeout=900)
    assert res.returncode == expect, f"{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    return res.stdout + res.stderr


def _staged(out):
    m = pd.read_csv(out / "TRAIN.csv")
    m["stem"] = [os.path.splitext(f)[0] for f in m["file"]]
    return m.set_index("stem")


def test_prepare_stages_slps_like_parquets_plus_a_csv(corpus, tmp_path):
    """SLPs alone vs the same pose as parquets + a bout CSV + --also-scored for what the
    SLPs' provenance says was scored: the same bouts, scored lists, scale and rate."""
    from_slp, from_csv = tmp_path / "from_slp", tmp_path / "from_csv"
    out = _prepare("--tracking", str(corpus["slp_lab"]), "--lab", LAB, *NEW_HEADS,
                   "--out", str(from_slp))
    assert "annotations: the events of 3 .slp file(s)" in out, out
    _prepare("--tracking", str(corpus["flat"]), "--annotations", str(corpus["csv"]), "--lab", LAB,
             *NEW_HEADS, "--also-scored", ";".join(_sniff_scored()), "--out", str(from_csv),
             "--pix-per-cm", str(PIX), "--fps", str(FPS))
    a, b = _staged(from_slp), _staged(from_csv)
    # held_out scored nothing, so it teaches nothing and is not staged; --also-scored stages
    # every file, but `flat/` has no held_out.
    assert sorted(a.index) == sorted(b.index) == ["v0", "v1"]
    for stem in a.index:
        assert json.loads(a.behaviors_labeled[stem]) == json.loads(b.behaviors_labeled[stem])
        for col in ("frames_per_second", "pix_per_cm_approx", "num_frames"):
            assert a[col][stem] == b[col][stem]

        def staged(root, kind, stem=stem):
            vid = _staged(root).video_id[stem]
            return pd.read_parquet(root / f"train_{kind}" / LAB / f"{vid}.parquet")
        bouts_a, bouts_b = staged(from_slp, "annotation"), staged(from_csv, "annotation")
        key = ["agent_id", "target_id", "action", "start_frame"]
        pd.testing.assert_frame_equal(bouts_a.sort_values(key, ignore_index=True),
                                      bouts_b.sort_values(key, ignore_index=True))
        pose_b = staged(from_csv, "tracking").dropna(subset=["x", "y"], how="all")
        pd.testing.assert_frame_equal(staged(from_slp, "tracking"), pose_b.reset_index(drop=True),
                                      check_dtype=False)


def test_prepare_without_annotations_needs_slps(corpus, tmp_path):
    out = _prepare("--tracking", str(corpus["flat"]), "--lab", LAB, *NEW_HEADS,
                   "--out", str(tmp_path / "x"), "--pix-per-cm", "16", "--fps", "30", expect=1)
    assert "--annotations is required" in out


def test_calibrate_reads_slp_events_like_the_csv(corpus):
    a = ft.read_annotations(str(corpus["slp_lab"]))
    b = ft.read_annotations(str(corpus["csv"]))
    cols = ["stem", "agent", "target", "action", "start_frame", "stop_frame"]
    pd.testing.assert_frame_equal(a[cols].sort_values(cols, ignore_index=True),
                                  b[cols].sort_values(cols, ignore_index=True), check_dtype=False)
    with pytest.raises(SystemExit, match="--stop-inclusive"):
        ft.read_annotations(str(corpus["slp_lab"] / "v0.slp"), stop_inclusive=True)


def test_discover_and_metadata(corpus, tmp_path, capsys):
    folder = tmp_path / "mixed"
    folder.mkdir()
    for f in ("v0.slp", "v1.slp"):
        os.symlink(corpus["slp_lab"] / f, folder / f)
    os.symlink(corpus["flat"] / "v0.parquet", folder / "other.parquet")
    files = cli.discover(str(folder))
    assert [os.path.basename(f) for f in files] == ["other.parquet", "v0.slp", "v1.slp"]

    slps = [f for f in files if f.endswith(".slp")]
    assert set(cli.load_metadata(str(folder), slps, None, None).values()) == {(PIX, FPS)}
    # a flag beats the file, and says so; a per-file metadata.csv row beats both
    meta = cli.load_metadata(str(folder), slps, 20.0, None)
    assert set(meta.values()) == {(20.0, FPS)}
    assert "the file itself says 16" in capsys.readouterr().out
    pd.DataFrame([dict(file="v1.slp", pix_per_cm=9.0)]).to_csv(folder / "metadata.csv", index=False)
    meta = cli.load_metadata(str(folder), slps, 20.0, None)
    assert meta[slps[1]] == (9.0, FPS) and meta[slps[0]] == (20.0, FPS)
    with pytest.raises(SystemExit, match="no pix_per_cm for other.parquet"):
        cli.load_metadata(str(folder), files[:1], None, 30.0)

    os.symlink(corpus["flat"] / "v1.parquet", folder / "v1.parquet")
    with pytest.raises(SystemExit, match="more than one tracking file"):
        cli.discover(str(folder))


@pytest.mark.slow
@pytest.mark.gpu
@requires_weights
@requires_cuda
def test_predict_on_slps_matches_the_parquets(corpus, tmp_path):
    """predict.py over the SLPs and over the same pose as parquets: identical probabilities.

    Same two recordings on both sides, and nothing else: inference draws its windows from one
    random stream over the whole folder, so an extra recording would shift the others'
    probabilities slightly whatever its format."""
    slps = tmp_path / "slps"
    slps.mkdir()
    for f in ("v0.slp", "v1.slp"):
        os.symlink(corpus["slp_lab"] / f, slps / f)
    outs = {}
    for name, folder, extra in (("slp", slps, []),
                                ("pq", corpus["flat"], ["--pix-per-cm", str(PIX), "--fps", str(FPS)])):
        outs[name] = tmp_path / name
        res = subprocess.run([sys.executable, os.path.join(REPO, "predict.py"), str(folder),
                              "--out", str(outs[name]), "--labs", "GroovyShrew", "--actions", "rear",
                              "--configs", "15fps_5bp", *extra],
                             capture_output=True, text=True, cwd=REPO, timeout=1800)
        assert res.returncode == 0 and "inference exited" not in res.stdout, res.stdout[-3000:]
    for stem in ("v0", "v1"):
        a = pd.read_parquet(outs["slp"] / f"{stem}.frames.parquet")
        b = pd.read_parquet(outs["pq"] / f"{stem}.frames.parquet")
        assert len(a) and a.equals(b)


def test_slp_annotations_for_parquet_tracking_carry_their_scored_list(corpus, tmp_path):
    """`--annotations <.slp files>` over parquet tracking stages what the SLPs alone do."""
    from_slp, mixed = tmp_path / "from_slp", tmp_path / "mixed"
    _prepare("--tracking", str(corpus["slp_lab"]), "--lab", LAB, *NEW_HEADS, "--out", str(from_slp))
    _prepare("--tracking", str(corpus["flat"]), "--annotations", str(corpus["slp_lab"]), "--lab", LAB,
             *NEW_HEADS, "--out", str(mixed), "--pix-per-cm", str(PIX), "--fps", str(FPS))
    a, b = _staged(from_slp), _staged(mixed)
    assert sorted(a.index) == sorted(b.index) == ["v0", "v1"]
    for stem in a.index:
        assert json.loads(a.behaviors_labeled[stem]) == json.loads(b.behaviors_labeled[stem])


def test_slp_manifest_is_in_lab_then_numeric_video_id_order(tmp_path):
    """The competition manifests' order, which the seeded split depends on -- not the file
    paths' string order, which would put 1001 before 999."""
    labels, _ = _labels()
    labels.provenance.update(frames_per_second=30.0, pix_per_cm_approx=16.0)
    for lab, vid in (("LyricalHare", 5), ("GroovyShrew", 1001), ("GroovyShrew", 999)):
        (tmp_path / lab).mkdir(exist_ok=True)
        labels.provenance.update(lab_id=lab, video_id=vid)
        labels.save(str(tmp_path / lab / f"{vid}.slp"))
    m = slp.manifest(slp.find(tmp_path))
    assert list(zip(m.lab_id, m.video_id, strict=True)) == [("GroovyShrew", 999), ("GroovyShrew", 1001),
                                               ("LyricalHare", 5)]
