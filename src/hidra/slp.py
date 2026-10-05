"""SLEAP `.slp` files as HiDRA input, alongside the long-format parquets.

A self-contained SLP holds one video's pose tracks, its human-scored behaviour bouts as
`UserEvent`s, and the video's metadata in `labels.provenance` -- everything the parquet
layout spreads over a tracking parquet, an annotation parquet and a manifest row. The
MABe-2025 release (one `<lab_id>/<video_id>.slp` per video) is that format, and any SLP with
tracked instances works as pose input.

This module turns one SLP into exactly those three pieces, so everything downstream of the
parquet reader runs unchanged:

    pose   video_frame, mouse_id, bodypart, x, y                  (the tracking parquet)
    bouts  agent_id, target_id, action, start_frame, stop_frame   (the annotation parquet;
                                                                    stop_frame exclusive)
    meta   lab_id, video_id, split, frames_per_second, pix_per_cm_approx, behaviors_labeled

How each is read:

- **pose**: the tracked instances of the file's one video, in float64 pixels. Where a frame
  has both a user-labelled and a predicted instance on one track, the user one wins. A point
  that is NaN or not visible is a missing keypoint, i.e. no row, as in the parquets.
  Untracked instances cannot be assigned to a mouse and are skipped (`Recording.notes` says
  how many).
- **mice**: tracks named `mouse1`..`mouse4` keep their names. Otherwise the file's tracks
  become `mouse1`, `mouse2`, ... in the order the file lists them; more than four is an
  error, since the model has four mouse slots.
- **bouts**: `UserEvent`s only -- a `PredictedEvent` is a model proposal, not an
  annotation. The subject is the agent, `target=None` is `self`, and `stop_frame =
  end_frame + 1` because SLP events are inclusive at both ends.
- **meta**: provenance keys of those names (`pix_per_cm` is accepted too). The frame rate
  falls back to the video's. Absent keys are None, and callers decide what that means.
"""
import glob
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sleap_io as sio

from . import paths

POSE_COLUMNS = ["video_frame", "mouse_id", "bodypart", "x", "y"]
BOUT_COLUMNS = ["agent_id", "target_id", "action", "start_frame", "stop_frame"]
MOUSE_NAME = re.compile(r"mouse[1-4]")
MAX_MICE = 4


def is_slp(path):
    return str(path).lower().endswith(".slp")


def find(folder):
    """Every .slp file under `folder`, recursively, sorted."""
    return sorted(glob.glob(os.path.join(str(folder), "**", "*.slp"), recursive=True))


@dataclass
class Recording:
    """One SLP, as the parquet layout's three pieces. `pose` is None when read with
    `pose=False`."""
    path: str
    pose: "pd.DataFrame | None"
    bouts: pd.DataFrame
    meta: dict
    mice: dict                     # SLP track name -> HiDRA mouse id
    notes: list = field(default_factory=list)

    @property
    def stem(self):
        return os.path.splitext(os.path.basename(self.path))[0]


def mouse_ids(track_names):
    """{track name: HiDRA mouse id}. See the module docstring for the rule."""
    names = list(track_names)
    if all(MOUSE_NAME.fullmatch(n) for n in names):
        return {n: n for n in names}
    if len(names) > MAX_MICE:
        raise ValueError(f"{len(names)} tracks ({names}); HiDRA models at most {MAX_MICE} mice. "
                         f"Merge or delete tracks so each animal has one, or name them "
                         f"mouse1..mouse{MAX_MICE}.")
    return {n: f"mouse{i + 1}" for i, n in enumerate(names)}


def _one_video(videos, path):
    if len(videos) != 1:
        raise ValueError(f"{path}: HiDRA reads one video per SLP, and this one has "
                         f"{len(videos)}. Split it into one file per video first.")
    return videos[0]


def _clean(value):
    """A provenance value as a plain scalar, or None for missing."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return None
    return value


def _meta(provenance, video, mice):
    prov = {k: _clean(v) for k, v in dict(provenance).items()}
    pix = prov.get("pix_per_cm_approx")
    if pix is None:
        pix = prov.get("pix_per_cm")
    fps = prov.get("frames_per_second")
    if fps is None:
        fps = _clean(getattr(video, "fps", None))
    labeled = prov.get("behaviors_labeled")
    if labeled is not None:
        if isinstance(labeled, str):
            labeled = json.loads(labeled.replace("'", ""))
        out = []
        for triplet in labeled:
            agent, target, action = (p.strip() for p in str(triplet).replace("'", "").split(","))
            agent = mice.get(agent, agent)
            target = target if target == "self" else mice.get(target, target)
            out.append(f"{agent},{target},{action}")
        labeled = list(dict.fromkeys(out))
    return dict(
        lab_id=None if prov.get("lab_id") is None else str(prov["lab_id"]),
        video_id=None if prov.get("video_id") is None else int(prov["video_id"]),
        split=None if prov.get("split") is None else str(prov["split"]),
        frames_per_second=None if fps is None else float(fps),
        pix_per_cm_approx=None if pix is None else float(pix),
        behaviors_labeled=labeled,
    )


def read_meta(path):
    """Just the `meta` of one SLP, from its header -- milliseconds, no pose or events read."""
    from sleap_io.io import slp as slp_io  # ~0.6 s to import; only when reading

    path = str(path)
    md = slp_io.read_metadata(path)
    video = _one_video(slp_io.read_videos(path, open_backend=False), path)
    mice = mouse_ids(t.name for t in slp_io.read_tracks(path))
    return _meta(slp_io.read_provenance(path, md), video, mice)


def _pose(labels, video, mice, notes):
    chosen = {}                       # (frame_idx, track name) -> instance
    untracked = 0
    for lf in labels.labeled_frames:
        if lf.video is not video:
            continue
        for inst in lf.instances:
            if inst.track is None:
                untracked += 1
                continue
            key = (int(lf.frame_idx), inst.track.name)
            prev = chosen.get(key)
            if prev is None:
                chosen[key] = inst
                continue
            prev_pred = isinstance(prev, sio.PredictedInstance)
            inst_pred = isinstance(inst, sio.PredictedInstance)
            if prev_pred and not inst_pred:
                chosen[key] = inst            # the user's correction beats the prediction
            elif prev_pred == inst_pred:
                raise ValueError(f"frame {key[0]} has two {'predicted' if inst_pred else 'user'} "
                                 f"instances on track {key[1]!r}; a track must be one animal")
    if untracked:
        notes.append(f"skipped {untracked} untracked instance(s): without a track they cannot "
                     f"be assigned to a mouse")
    if not chosen:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
            POSE_COLUMNS, ["int64", "object", "object", "float64", "float64"], strict=True)})

    keys, insts = zip(*chosen.items(), strict=True)
    counts = np.array([len(inst.points) for inst in insts])
    node_names = {}                   # one array per skeleton, not per instance
    parts = np.concatenate([
        node_names.setdefault(id(inst.skeleton), np.asarray(inst.skeleton.node_names, dtype=object))
        for inst in insts])
    coords = np.concatenate([inst.points["xy"] for inst in insts]).astype(np.float64)
    coords[~np.concatenate([inst.points["visible"] for inst in insts])] = np.nan
    keep = ~np.isnan(coords).all(axis=1)
    pose = pd.DataFrame({
        "video_frame": np.repeat(np.array([k[0] for k in keys], dtype=np.int64), counts)[keep],
        "mouse_id": np.repeat(np.array([mice[k[1]] for k in keys], dtype=object), counts)[keep],
        "bodypart": parts[keep],
        "x": coords[keep, 0],
        "y": coords[keep, 1],
    })
    return pose.sort_values(["video_frame", "mouse_id", "bodypart"], kind="stable",
                            ignore_index=True)


def _participant(p, mice, path):
    if p.name in mice:
        return mice[p.name]
    raise ValueError(f"{path}: an event names {p.name!r}, which is not one of the file's "
                     f"tracks {sorted(mice)}")


def _bouts(labels, video, mice, notes, path):
    rows, predicted, no_subject = [], 0, 0
    for ev in labels.events:
        if ev.video is not None and ev.video is not video:
            continue
        if not isinstance(ev, sio.UserEvent):
            predicted += 1
            continue
        if ev.subject is None:
            no_subject += 1
            continue
        target = "self" if ev.target is None else _participant(ev.target, mice, path)
        rows.append((_participant(ev.subject, mice, path), target, ev.type.name,
                     int(ev.start_frame), int(ev.end_frame) + 1))
    if predicted:
        notes.append(f"ignored {predicted} predicted event(s): only user events are annotations")
    if no_subject:
        notes.append(f"skipped {no_subject} event(s) with no subject: a bout needs an acting mouse")
    bouts = pd.DataFrame(rows, columns=BOUT_COLUMNS)
    return bouts.astype({"start_frame": np.int64, "stop_frame": np.int64})


def read(path, pose=True):
    """One SLP as a `Recording`. With `pose=False` the instances are not materialized, which
    is much faster when only the bouts and metadata are wanted."""
    path = str(path)
    labels = sio.load_slp(path, open_videos=False, lazy=not pose)
    video = _one_video(labels.videos, path)
    mice = mouse_ids(t.name for t in labels.tracks)
    notes = []
    renamed = {t: m for t, m in mice.items() if t != m}
    if renamed:
        notes.append("tracks read as " + ", ".join(f"{t} -> {m}" for t, m in renamed.items()))
    return Recording(
        path=path,
        pose=_pose(labels, video, mice, notes) if pose else None,
        bouts=_bouts(labels, video, mice, notes, path),
        meta=_meta(labels.provenance, video, mice),
        mice=mice,
        notes=notes,
    )


def manifest(files, split=None):
    """Manifest rows (the `train.csv` columns plus `slp`) for self-contained SLPs, from their
    provenance. With `split`, only the files whose provenance `split` is that one; a file
    without a `split` counts as `train`. Rows are in (lab_id, video_id) order, as the
    competition's manifests are: the seeded 85/15 split (`data.split_videos`) depends on it.

    Training needs a lab, a pixel scale and a frame rate for every video, so a file missing
    any of them is an error naming it. `behaviors_labeled` may be absent: the row then takes
    it from the file's own bouts when the video is loaded (see `data.create_video`).
    """
    rows, missing = [], []
    for p in files:
        meta = read_meta(p)
        if split is not None and (meta["split"] or "train") != split:
            continue
        need = [k for k in ("lab_id", "frames_per_second", "pix_per_cm_approx") if meta[k] is None]
        if need:
            missing.append(f"{p}: {need}")
            continue
        labeled = meta["behaviors_labeled"]
        rows.append(dict(
            lab_id=meta["lab_id"],
            video_id=meta["video_id"] if meta["video_id"] is not None else paths.vid_of(p),
            frames_per_second=meta["frames_per_second"],
            pix_per_cm_approx=meta["pix_per_cm_approx"],
            behaviors_labeled=None if labeled is None else json.dumps(labeled),
            slp=os.path.abspath(p),
        ))
    if missing:
        raise ValueError("SLP(s) without the provenance training needs (add it to "
                         "`labels.provenance`, or list the files in a train.csv with those "
                         "columns and an `slp` column):\n  " + "\n  ".join(missing))
    df = pd.DataFrame(rows, columns=["lab_id", "video_id", "frames_per_second",
                                     "pix_per_cm_approx", "behaviors_labeled", "slp"])
    dup = df["video_id"][df["video_id"].duplicated()]
    if len(dup):
        raise ValueError(f"video_id(s) {sorted(set(dup))} appear in more than one SLP; "
                         f"each video needs its own id")
    return df.sort_values(["lab_id", "video_id"], kind="stable", ignore_index=True)


def labeled_triplets(bouts):
    """`behaviors_labeled` for a video whose bouts are all that says what was scored: every
    (agent, target, action) with at least one bout."""
    return sorted({f"{r.agent_id},{r.target_id},{r.action}" for r in bouts.itertuples()})
