#!/usr/bin/env python3
"""
HiDRA fine-tuning -- adapt one lab's classifier head(s) to YOUR videos and YOUR annotations.

Zero-shot (predict.py) applies a lab's classifier as-is. Fine-tuning warm-starts from that same
classifier and continues training it on your labelled data, so it learns your arena, your pose
rig, and your annotation style. The self-supervised backbone and the shared feature merge stay
frozen; only the per-lab tail and/or the per-lab head are updated (see --mode).

WHAT YOU CAN FINE-TUNE
  The behaviour vocabulary is fixed (37 names plus the merged sniffall), but the head's
  columns are not. Either
    * ADOPT an existing (lab, action) head whose behaviour matches yours and continue
      training it -- the cheapest and safest route:
          python predict.py --list-heads          # every (lab, action) pair
      Pick the lab whose zero-shot output looked closest (docs/zero-shot.md); or
    * give a lab a NEW column with `train --new-head Lab,action`, including under one of
      the head-free lab slots, which turns that slot into YOUR lab (docs/new-behaviours.md).

MAKING IT FAST
  `--mode` decides how much of the network moves: `head` trains one 256-weight column per
  behaviour, `embedding` adds the lab's embedding row, `tail` retrains the whole per-lab
  tail. Everything before that is frozen, so with `--cache-features` it is computed once per
  window instead of every step -- the difference between hours and minutes per config. See
  docs/fine-tuning.md "Making it fast".

THREE STEPS
  # 1. stage your data into the layout the trainer expects
  python finetune.py prepare --tracking /path/to/parquets --annotations bouts.csv \
      --lab GroovyShrew --out ft_data/ --pix-per-cm 16 --fps 30
  # 2. train (5 configs = 5 checkpoints; one GPU, a few hours for the whole ensemble)
  python finetune.py train --data ft_data/ --lab GroovyShrew --actions rear \
      --out ft_models/ --tag myrig
  # 3. predict with the fine-tuned weights, then re-calibrate the thresholds on held-out data
  python predict.py /path/to/held_out --labs GroovyShrew --actions rear --out ft_preds/ \
      --weights 'ft_models/{config}__myrig.pkl' --pix-per-cm 16 --fps 30
  python finetune.py calibrate --frames ft_preds/ --annotations heldout_bouts.csv \
      --out ft_thresholds.csv
  python predict.py /path/to/new --labs GroovyShrew --actions rear --out results/ \
      --weights 'ft_models/{config}__myrig.pkl' --thresholds ft_thresholds.csv ...

ANNOTATION CSV (for `prepare` and `calibrate`)
  file,agent,target,action,start_frame,stop_frame
  mouseA_day1.parquet,mouse1,mouse2,rear,120,181
  - one row per bout; `file` is the tracking parquet's filename (or its stem).
  - `target` is a mouse id or `self` for self-directed behaviours (rear, dig, selfgroom, ...).
  - `stop_frame` is EXCLUSIVE by default (frames start..stop-1 are positive), matching the
    trainer. Pass --stop-inclusive if your stop_frame is the last positive frame -- which is
    what predict.py's bouts.csv writes.
  - Every frame of a video that is NOT inside a bout counts as a NEGATIVE for the (agent,
    target, action) combinations that video annotates. So annotate each behaviour
    exhaustively within a video, and leave out videos you only skimmed.
  - Only one action can be positive per (agent, target) per frame: overlapping bouts of
    different behaviours on the same pair overwrite each other (last one wins).
  - Label behaviours with the head names `predict.py --list-heads` prints. For the five labs
    whose sniff head is the merged `sniffall`, label sniffing `sniffall` (or a subtype the lab
    has a head for); `prepare` stages it as `sniff`, because the trainer derives the sniffall
    label from the sniff-family labels and a row literally named `sniffall` supervises nothing.

ADDING HEAD COLUMNS (--new-head)
  A lab's head has columns only for the behaviours it annotated. `train --new-head Lab,action`
  (repeatable) gives it more: the published columns are copied into a wider head, each new
  column starts from `--seed-from` or a fresh init, and only the new columns are supervised.
  The action must be a vocabulary name (predict.py --list-heads shows who has which); the
  checkpoint then carries its own column list ({config}__{tag}.heads.json), so predict.py
  --weights knows the head. PyTorch backend.
      python finetune.py prepare --tracking parquets/ --annotations bouts.csv --lab GroovyShrew \
          --new-head GroovyShrew,attack --out ft_data/ --pix-per-cm 16 --fps 30
      python finetune.py train --data ft_data/ --lab GroovyShrew --new-head GroovyShrew,attack \
          --seed-from LyricalHare,attack --out ft_models/ --tag attack

YOUR OWN LAB (--new-head on a head-free slot)
  Five of the 21 lab-embedding rows carry no published head and can be claimed -- print them
  with `python -c "from hidra.head_table import head_free_slots; print(head_free_slots())"`.
  Naming one as --lab and giving it columns makes it your lab: nothing published is
  overwritten, and with --mode tail the lab's whole tail adapts to your rig.
  `--seed-from <DonorLab>` starts each column, and the lab's embedding row, from the
  published lab whose recordings look most like yours.
      python finetune.py prepare --tracking parquets/ --annotations bouts.csv --lab CRIM13 \
          --new-head CRIM13,rear --new-head CRIM13,attack --out ft_data/ --pix-per-cm 16 --fps 30
      python finetune.py train --data ft_data/ --lab CRIM13 \
          --new-head CRIM13,rear --new-head CRIM13,attack --seed-from GroovyShrew \
          --mode tail --cache-features --out ft_models/ --tag mylab
  See docs/new-behaviours.md.

See docs/fine-tuning.md for what each mode trains, how much data helps, and the caveats.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from . import cli as hidra_predict   # vid_of / enum_heads / load_metadata / discover
from . import head_table             # --new-head: the per-lab head's column list, and its sidecar
from .schema import SNIFF_FAMILY     # numpy-only; the labels the merged sniffall head is built from

CONFIGS = ["11fps_4bp", "15fps_5bp", "19fps_6bp", "23fps_7bp", "27fps_6bp"]
ANNOT_COLS = ["file", "agent", "target", "action", "start_frame", "stop_frame"]


# ------------------------------------------------------------------ shared helpers
def heads_by_lab(thr_csv=None):
    """{lab: {action, ...}} for every trained classifier head."""
    out = {}
    for lab, act in hidra_predict.enum_heads(thr_csv or hidra_predict.THR_CSV):
        out.setdefault(lab, set()).add(act)
    return out


def annotatable_actions(lab, hb):
    """Action names a bout CSV may carry for `lab`: its heads plus, when it has the merged
    `sniffall` head, the sniff-family labels that head is derived from. A lab with no
    published head (a head-free slot adopted as your own lab) contributes none, and
    everything it may carry comes from `--new-head`."""
    acts = set(hb.get(lab, ()))
    if "sniffall" in acts:
        acts |= set(SNIFF_FAMILY)
    return acts


def trainable_heads(lab, annotated, hb):
    """The head columns of `lab` that a set of annotated action names can supervise.

    `sniffall` has no label of its own: the trainer synthesizes it as the OR of the
    sniff-family channels. It is included when plain `sniff` is annotated -- which is what
    `prepare` stages a `sniffall` row as -- so that a subtype-only annotation set does not
    silently redefine the merged head as that subtype.
    """
    heads = set(hb.get(lab, ()))
    out = {a for a in annotated if a in heads}
    if "sniffall" in heads and "sniff" in annotated:
        out.add("sniffall")
    return sorted(out)


def supervisable(action, annotated):
    """Does this set of annotated labels give `action`'s head column any supervision?"""
    if action == "sniffall":
        return any(a in annotated for a in SNIFF_FAMILY)
    return action in annotated


def suggested_actions(lab, annotated, hb):
    """(head columns these labels train, columns to offer as an opt-in) for `prepare`'s next step.

    The opt-in is `sniffall` for a subtype-only annotation set: the trainer ORs the sniff family,
    so those labels do supervise the merged head -- just narrowly, as "sniffing is exactly this
    subtype", which is rarely what is meant. Offering it rather than assuming it also keeps the
    suggested command runnable: `trainable_heads` alone can be empty for a set `check_actions`
    accepted, and `--actions ` with no value is an argparse error.
    """
    heads = trainable_heads(lab, annotated, hb)
    opt_in = (["sniffall"] if "sniffall" not in heads and "sniffall" in hb.get(lab, ())
              and supervisable("sniffall", annotated) else [])
    return heads, opt_in


def resolve_new_heads(specs, lab, hb, seed_from=None):
    """Validate the `--new-head LAB,ACTION` flags (and `--seed-from`) for `lab`; returns
    `([(lab, action), ...], [(donor, action|None), ...])`. Exits with the reason on any problem.

    The column list new heads extend is the published one (`schema.lab_action_table()`,
    from models/thresholds.json); the thresholds CSV's heads are checked too, since that is
    what `--list-heads` shows. `seed_from` is one entry or several: `LAB,ACTION` names a
    donor column, a bare `LAB` a donor lab (each new column from its column of the same
    behaviour, and the lab embedding row from it too). `hidra.head_table.seed_sources` has
    the rules for combining them.
    """
    if not specs:
        if seed_from:
            sys.exit("ERROR: --seed-from only makes sense together with --new-head")
        return [], []
    from . import schema
    try:
        pairs = head_table.parse_head_specs(specs)
        seeds = head_table.parse_seed_specs(seed_from)
        table = schema.lab_action_table()
        for pair in pairs:
            if pair[0] != lab:
                raise ValueError(f"--new-head names lab {pair[0]!r} but --lab is {lab!r}; columns "
                                 f"are added under the lab whose data is staged")
            head_table.validate_new_head(pair[0], pair[1], table)
            if pair[1] in hb.get(lab, set()):
                raise ValueError(f"({lab}, {pair[1]}) is already a head in the thresholds CSV; "
                                 f"fine-tune it with --actions {pair[1]} instead of --new-head")
        # Resolving here as well as in the trainer catches a bad donor before a GPU is claimed.
        head_table.seed_sources(pairs, seeds, table, lab=lab)
    except (ValueError, FileNotFoundError) as e:   # FileNotFoundError: no models/thresholds.json yet
        sys.exit(f"ERROR: {e}")
    return pairs, seeds


def norm_mouse(m):
    """'1' / 1 / 'mouse1' -> 'mouse1'; 'self' passes through."""
    m = str(m).strip()
    if m in ("self", "*"):
        return m
    if m.startswith("mouse"):
        return m
    if m.isdigit():
        return f"mouse{m}"
    sys.exit(f"ERROR: bad mouse id {m!r}; use mouse1..mouse4 or 'self'")


# The per-video annotation parquet the trainer itself reads, which is also what
# `HiDRA_finetune.zip`'s `prepare_dataset.py` takes as input. Accepting it here means a
# dataset already in that layout needs no conversion.
BUNDLE_ANNOT_COLS = ["agent_id", "target_id", "action", "start_frame", "stop_frame"]


def read_annotation_dir(path, by_stem=None):
    """A directory of per-video annotation parquets -> the same frame `read_annotations`
    returns. Each file is `<name>.parquet` with columns `agent_id, target_id, action,
    start_frame, stop_frame` and `stop_frame` exclusive.

    The file name identifies the recording. It matches a tracking file by stem, or by the
    `video_id` that stem hashes to -- which is how the bundle names both sides, and the
    only reason `by_stem` is needed here.
    """
    files = sorted(f for f in os.listdir(path) if f.endswith(".parquet"))
    if not files:
        sys.exit(f"ERROR: no .parquet annotation files in {path}")
    by_vid = {str(hidra_predict.vid_of(p)): stem for stem, p in (by_stem or {}).items()}
    frames, unmatched = [], []
    for f in files:
        name = os.path.splitext(f)[0]
        stem = name if by_stem is None or name in by_stem else by_vid.get(name)
        if stem is None:
            unmatched.append(f)
            continue
        d = pd.read_parquet(os.path.join(path, f))
        missing = [c for c in BUNDLE_ANNOT_COLS if c not in d.columns]
        if missing:
            sys.exit(f"ERROR: {os.path.join(path, f)} is missing column(s) {missing}. "
                     f"A per-video annotation parquet needs {BUNDLE_ANNOT_COLS}.")
        frames.append(pd.DataFrame(dict(
            file=stem, stem=stem,
            agent=[norm_mouse(m) for m in d["agent_id"]],
            target=[norm_mouse(m) for m in d["target_id"]],
            action=[str(a).strip() for a in d["action"]],
            start_frame=d["start_frame"].astype(int).to_numpy(),
            stop_frame=d["stop_frame"].astype(int).to_numpy())))
    if unmatched:
        sys.exit(f"ERROR: annotation file(s) {unmatched} match no tracking file by name or by "
                 f"video id. Tracking stems: {sorted(by_stem or ())}")
    return pd.concat(frames, ignore_index=True)[ANNOT_COLS + ["stem"]]


def read_annotations(path, stop_inclusive=False, by_stem=None):
    """Load the annotations -> DataFrame[stem, agent, target, action, start_frame, stop_frame],
    with stop_frame EXCLUSIVE and `stem` the tracking filename without its extension.

    `path` is the bout CSV, or a directory of per-video annotation parquets
    (`read_annotation_dir`).
    """
    if os.path.isdir(path):
        d = read_annotation_dir(path, by_stem)
        if stop_inclusive:
            d["stop_frame"] = d["stop_frame"] + 1
        bad = d[d["stop_frame"] <= d["start_frame"]]
        if len(bad):
            sys.exit(f"ERROR: {len(bad)} bout(s) in {path} have stop_frame <= start_frame "
                     f"(first: {bad.iloc[0].to_dict()}).")
        return d
    d = pd.read_csv(path)
    low = {c.lower().strip(): c for c in d.columns}
    missing = [c for c in ANNOT_COLS if c not in low]
    if missing:
        sys.exit(f"ERROR: {path} is missing column(s) {missing}. Required: {ANNOT_COLS}")
    d = d.rename(columns={low[c]: c for c in ANNOT_COLS})[ANNOT_COLS].copy()
    d["stem"] = [os.path.splitext(os.path.basename(str(f)))[0] for f in d["file"]]
    d["agent"] = [norm_mouse(m) for m in d["agent"]]
    d["target"] = [norm_mouse(m) for m in d["target"]]
    d["action"] = [str(a).strip() for a in d["action"]]
    d["start_frame"] = d["start_frame"].astype(int)
    d["stop_frame"] = d["stop_frame"].astype(int) + (1 if stop_inclusive else 0)
    bad = d[d["stop_frame"] <= d["start_frame"]]
    if len(bad):
        sys.exit(f"ERROR: {len(bad)} bout(s) in {path} have stop_frame <= start_frame "
                 f"(first: {bad.iloc[0].to_dict()}). Pass --stop-inclusive if stop_frame is "
                 "the last positive frame.")
    return d


def check_actions(annot, lab, thr_csv=None, drop_unsupported=False, extra_actions=()):
    """Every annotated action must be a head of `lab`, else there is nothing to fine-tune.

    `extra_actions` admits actions that are not heads yet -- the column `--new-head` adds."""
    hb = heads_by_lab(thr_csv)
    if lab not in hb and not extra_actions:
        sys.exit(f"ERROR: no classifier heads for lab {lab!r}, and no --new-head to give it "
                 f"one. Labs with heads: {sorted(hb)}; head-free slots you can adopt as your "
                 f"own lab with --new-head: {head_table.head_free_slots()}")
    mine = annotatable_actions(lab, hb) | set(extra_actions)
    if "sniffall" in mine:
        # Whether the merged head is published or being added by --new-head, the labels that
        # feed it are the sniff family -- `annotatable_actions` only knows about the
        # published case, so a new sniffall column needs the same admission here.
        mine |= set(SNIFF_FAMILY)
    unsupported = sorted(set(annot["action"]) - mine)
    if unsupported:
        elsewhere = {a: sorted(l for l, acts in hb.items() if a in acts) for a in unsupported}
        msg = "\n".join(f"    {a}: heads exist for {elsewhere[a] or 'NO lab'}" for a in unsupported)
        if not drop_unsupported:
            sys.exit(f"ERROR: {lab} has no head for {unsupported}, so those annotations cannot "
                     f"train anything.\n  {lab} heads: {sorted(mine)}\n{msg}\n"
                     "  Adopt a lab that has the behaviour, run one fine-tune per lab, or pass "
                     "--drop-unsupported to ignore those rows.")
        print(f"  dropping {len(annot[annot['action'].isin(unsupported)])} row(s) for "
              f"unsupported action(s) {unsupported}")
        annot = annot[~annot["action"].isin(unsupported)].copy()
        if annot.empty:
            sys.exit("ERROR: no annotations left after --drop-unsupported")
    n_sniffall = int((annot["action"] == "sniffall").sum())
    if n_sniffall:
        # The model has no `sniffall` label channel: MultiTaskPerLabModel._labels37 (and its
        # torch port) OVERWRITES that channel with the OR of the sniff-family channels, label
        # and mask alike. A bout labelled `sniffall` would therefore supervise nothing -- the
        # run logs a zero-weight loss -- so stage it under the plain family member.
        print(f"  note: staging {n_sniffall} 'sniffall' row(s) as 'sniff' -- the trainer derives the "
              f"merged sniffall label from the sniff-family labels ({', '.join(SNIFF_FAMILY)})")
        annot = annot.copy()
        annot.loc[annot["action"] == "sniffall", "action"] = "sniff"
    return annot


# ------------------------------------------------------------------ prepare
def parse_also_scored(spec, allowed_actions):
    """`--also-scored 'mouse1,mouse2,attack;mouse2,mouse1,attack'` -> [(agent, target, action)].

    These are combinations every staged video *scored* even where it has no bout: without
    them a video with zero bouts of a behaviour contributes nothing for it, and with them
    every one of its frames is a negative. It is the bundle's flag of the same name, and it
    is how "we watched for this and saw none" is expressed.
    """
    out, translated = [], 0
    for tok in (t.strip() for t in str(spec or "").split(";") if t.strip()):
        parts = [p.strip() for p in tok.split(",")]
        if len(parts) != 3 or not all(parts):
            sys.exit(f"ERROR: --also-scored {tok!r}: expected 'agent,target,action'")
        agent, target, action = norm_mouse(parts[0]), norm_mouse(parts[1]), parts[2]
        if action not in allowed_actions:
            sys.exit(f"ERROR: --also-scored names action {action!r}, which is not one this lab "
                     f"can train: {sorted(allowed_actions)}")
        # Same translation `check_actions` applies to bout rows, and for the same reason: the
        # trainer synthesizes the sniffall channel from the sniff family, so a combination
        # recorded literally as `sniffall` would carry zero mask and score nothing.
        if action == "sniffall":
            action, translated = "sniff", translated + 1
        out.append((agent, target, action))
    if translated:
        print(f"  note: staging {translated} --also-scored 'sniffall' combination(s) as 'sniff'")
    return out


def cmd_prepare(args):
    """Stage tracking parquets + bouts into {out}/TRAIN.csv + {out}/train_{tracking,annotation}/{lab}/."""
    parquets = hidra_predict.discover(args.tracking)
    if not parquets:
        sys.exit(f"no .parquet/.pkt files in {args.tracking}")
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in parquets}

    hb = heads_by_lab(args.thresholds)
    annot = read_annotations(args.annotations, args.stop_inclusive, by_stem)
    new_heads, _ = resolve_new_heads(args.new_head, args.lab, hb)
    new_actions = [a for _, a in new_heads]
    annot = check_actions(annot, args.lab, args.thresholds, args.drop_unsupported,
                          extra_actions=new_actions)
    also_scored = parse_also_scored(args.also_scored,
                                    annotatable_actions(args.lab, hb) | set(new_actions))

    unknown = sorted(set(annot["stem"]) - set(by_stem))
    if unknown:
        sys.exit(f"ERROR: annotated file(s) not found in {args.tracking}: {unknown}\n"
                 f"  tracking files present: {sorted(by_stem)}")
    annotated_stems = set(annot["stem"])
    # A video with no bouts is worth staging only when --also-scored says it was watched:
    # then its frames are negatives. Otherwise it teaches the model nothing.
    staged_stems = set(by_stem) if also_scored else annotated_stems
    used = [by_stem[s] for s in sorted(staged_stems)]
    skipped = sorted(set(by_stem) - staged_stems)
    if skipped:
        print(f"  note: {len(skipped)} tracking file(s) have no annotations and are NOT staged "
              f"(an unannotated video teaches the model nothing): {skipped}")
    if also_scored:
        empty = sorted(staged_stems - annotated_stems)
        print(f"  --also-scored: {len(also_scored)} combination(s) marked scored in every staged "
              f"video" + (f", including {len(empty)} with no bouts at all ({empty})" if empty else ""))
    meta = hidra_predict.load_metadata(args.tracking, used, args.pix_per_cm, args.fps)

    # bodypart-name validation needs the model schema (imports jax); skip it politely if the
    # training deps aren't installed on this machine -- predict.py/train re-check it anyway.
    try:
        allbp, canon7 = hidra_predict.bodypart_schema()
    except ImportError as e:
        allbp, canon7 = None, None
        print(f"  note: skipping bodypart-name check ({e}); it runs again at train time")

    tdir = os.path.join(args.out, "train_tracking", args.lab)
    adir = os.path.join(args.out, "train_annotation", args.lab)
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(adir, exist_ok=True)

    rows = []
    for pq in used:
        stem = os.path.splitext(os.path.basename(pq))[0]
        vid = hidra_predict.vid_of(pq)               # same id predict.py uses for this filename
        df = pd.read_parquet(pq)
        need = {"video_frame", "mouse_id", "bodypart", "x", "y"}
        if not need <= set(df.columns):
            sys.exit(f"ERROR: {stem}: tracking parquet needs columns {sorted(need)}, has {list(df.columns)}")
        if allbp is not None:
            bad = set(df["bodypart"].unique()) - allbp
            if bad:
                sys.exit(f"ERROR: {stem} has bodypart names outside the model schema: {sorted(bad)}\n"
                         f"  Rename them to schema names (the 7 the model uses: {canon7}).")
        n_frames = int(df["video_frame"].max()) + 1
        mice = {norm_mouse(m) for m in df["mouse_id"].unique()}
        shutil.copyfile(pq, os.path.join(tdir, f"{vid}.parquet"))

        a = annot[annot["stem"] == stem].copy()
        for col, name in (("agent", "agent"), ("target", "target")):
            missing = {m for m in a[col] if m not in mice and m != "self"}
            if missing:
                sys.exit(f"ERROR: {stem}: annotated {name} {sorted(missing)} not tracked in the "
                         f"parquet (tracked: {sorted(mice)})")
        over = a[a["stop_frame"] > n_frames]
        if len(over):
            print(f"  note: {stem}: clipping {len(over)} bout(s) that end past frame {n_frames}")
            a.loc[a["stop_frame"] > n_frames, "stop_frame"] = n_frames
            a = a[a["stop_frame"] > a["start_frame"]]
        # the trainer reads agent_id/target_id/action/start_frame/stop_frame (stop exclusive)
        out_a = pd.DataFrame(dict(agent_id=a["agent"].to_numpy(), target_id=a["target"].to_numpy(),
                                  action=a["action"].to_numpy(),
                                  start_frame=a["start_frame"].to_numpy(np.int64),
                                  stop_frame=a["stop_frame"].to_numpy(np.int64)))
        out_a.to_parquet(os.path.join(adir, f"{vid}.parquet"))
        # behaviors_labeled must list EVERY (agent, target, action) present in the annotations:
        # it is what marks a behaviour as supervised (and its non-bout frames as negatives).
        labeled = {f"{r.agent_id},{r.target_id},{r.action}" for r in out_a.itertuples()}
        # --also-scored adds combinations this video watched for but recorded no bout of, so
        # every frame of them becomes a negative. Skipped where a named mouse is not tracked.
        for agent, target, action in also_scored:
            if agent in mice and (target in mice or target == "self"):
                labeled.add(f"{agent},{target},{action}")
            else:
                print(f"  note: {stem}: --also-scored {agent},{target},{action} skipped "
                      f"(mouse not tracked; tracked: {sorted(mice)})")
        labeled = sorted(labeled)
        pix, fps = meta[pq]
        rows.append(dict(lab_id=args.lab, video_id=vid, frames_per_second=fps,
                         pix_per_cm_approx=pix, num_frames=n_frames, file=os.path.basename(pq),
                         behaviors_labeled=json.dumps(labeled)))
        pos = int(sum(r.stop_frame - r.start_frame for r in out_a.itertuples()))
        print(f"  {stem} -> video_id {vid}: {n_frames} frames, {len(out_a)} bouts, "
              f"{pos} positive frames ({100 * pos / max(n_frames, 1):.1f}%), {len(labeled)} labelled combos")

    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(args.out, "TRAIN.csv"), index=False)
    # train.csv is what the trainers read; they copy TRAIN.csv over if it is absent,
    # but writing both keeps a re-prepared dataset from being shadowed by a stale train.csv.
    man.to_csv(os.path.join(args.out, "train.csv"), index=False)
    staged = set(annot["action"]) | {a for _, _, a in also_scored}
    # One place decides what `prepare` suggests; --new-head adds its column to that list
    # rather than deriving a second one. Unioning before the `heads + opt_in` concatenation
    # keeps the opt-in last, and makes the suggestion non-empty whenever --new-head is given.
    heads, opt_in = suggested_actions(args.lab, staged, hb)
    extra = ""
    for lab_, action_ in new_heads:
        if not supervisable(action_, staged):
            sys.exit(f"ERROR: --new-head {lab_},{action_} but no bout is labelled "
                     f"{action_!r}; the new column needs annotations to learn from")
        heads = sorted(set(heads) | {action_})
        extra += f" --new-head {lab_},{action_}"
    print(f"\nstaged {len(man)} video(s) for lab {args.lab} -> {args.out}/")
    print(f"  labels staged: {sorted(staged)}")
    if heads:
        new_names = sorted(a for _, a in new_heads)
        print(f"  head columns these can train: {heads}"
              + (f" ({', '.join(new_names)} {'is the NEW column' if len(new_names) == 1 else 'are the NEW columns'})"
                 if new_heads else ""))
    if opt_in:
        print(f"  head column 'sniffall' is trainable too, but only from the subtype(s) you "
              f"annotated ({sorted(staged & set(SNIFF_FAMILY))}): every other sniffing frame becomes "
              f"a negative for the merged head. Drop it from --actions if that is not what you mean.")
    print(f"next: python finetune.py train --data {args.out} --lab {args.lab} "
          f"--actions {','.join(heads + opt_in)}{extra} --out ft_models/ --tag mytag")


# ------------------------------------------------------------------ train
def cmd_train(args):
    """Run train_perlab_heads.py once per config with the LABTAIL (new-lab adaptation) env set."""
    man_path = os.path.join(args.data, "TRAIN.csv")
    if not os.path.isfile(man_path):
        sys.exit(f"ERROR: {man_path} not found -- run `finetune.py prepare` first")
    man = pd.read_csv(man_path)
    labs = sorted(set(man["lab_id"].astype(str)))
    if args.lab not in labs:
        sys.exit(f"ERROR: {args.data} was staged for lab(s) {labs}, not {args.lab}")
    hb = heads_by_lab(args.thresholds)
    annotated = sorted({b.split(",")[-1] for bl in man["behaviors_labeled"] for b in json.loads(bl)})

    from .download_models import require_weights
    require_weights()                      # warm-starting needs the base checkpoints
    new_heads, seeds = resolve_new_heads(args.new_head, args.lab, hb, args.seed_from)
    new_actions = [a for _, a in new_heads]
    if new_heads:
        if getattr(args, "backend", "torch") != "torch":
            sys.exit("ERROR: --new-head is a PyTorch-backend workflow; drop --backend jax")
        # `supervisable`, not membership: a new `sniffall` column is fed by the sniff-family
        # labels, which is what `prepare` stages a `sniffall` row as.
        unannotated = [a for a in new_actions if not supervisable(a, annotated)]
        if unannotated:
            sys.exit(f"ERROR: --new-head declares {unannotated} but they are not annotated in "
                     f"{man_path} (annotated: {annotated}); re-run `prepare --new-head` with bouts for them")

    if args.actions:
        actions = [a.strip() for a in args.actions.split(",") if a.strip()]
        left_out = [a for a in new_actions if a not in actions]
        if left_out:
            sys.exit(f"ERROR: --actions {args.actions} leaves out the new column(s) {left_out}; "
                     f"include them, or drop --actions to train the new columns alone")
    else:
        actions = new_actions or trainable_heads(args.lab, annotated, hb)
        if not actions:
            sys.exit(f"ERROR: none of the staged labels {annotated} can train a {args.lab} head "
                     f"(its heads: {sorted(hb.get(args.lab, ()))})")
    bad = sorted(set(actions) - hb.get(args.lab, set()) - set(new_actions))
    if bad:
        hint = ("; label sniffing 'sniffall' in the CSV, or pass --actions sniffall"
                if set(bad) & set(SNIFF_FAMILY) and "sniffall" in hb.get(args.lab, set()) else "")
        sys.exit(f"ERROR: {args.lab} has no head for {bad}; its heads are "
                 f"{sorted(hb.get(args.lab, ()))}{hint}")
    missing = sorted(a for a in actions if not supervisable(a, annotated))
    if missing:
        detail = (f" ('sniffall' is derived from the sniff-family labels {SNIFF_FAMILY}, none of which "
                  f"is annotated)" if "sniffall" in missing else "")
        sys.exit(f"ERROR: {missing} are not annotated in {man_path} (annotated: {annotated}){detail}")
    if "sniffall" in actions and "sniff" not in annotated:
        print("  note: 'sniffall' will be trained on the annotated subtypes only "
              f"({sorted(set(annotated) & set(SNIFF_FAMILY))}); every other sniffing frame becomes a "
              "negative for the merged head")
    tag = args.tag or f"{args.lab}_{args.mode}"
    configs = [c.strip() for c in args.configs.split(",") if c.strip()] if args.configs else CONFIGS
    bad_cfg = [c for c in configs if c not in CONFIGS]
    if bad_cfg:
        sys.exit(f"ERROR: unknown config(s) {bad_cfg}; available: {CONFIGS}")

    out = os.path.abspath(args.out)
    workdir = args.workdir or f"/dev/shm/hidra_finetune_{args.lab}_{tag}"
    if not args.dry_run:                  # a dry run leaves the filesystem exactly as it was
        os.makedirs(out, exist_ok=True)
        if not args.reuse_cache:
            # the tracking/label cache under workdir is keyed by video_id only, so a
            # re-`prepare`d dataset would silently train on the OLD frames; rebuild it
            # unless told otherwise.
            shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)

    # train_perlab_heads.py is driven entirely by env vars and asserts that the experiment
    # switches are mutually exclusive -- drop any left over in the caller's shell.
    env = {k: v for k, v in os.environ.items() if k not in {
        "FROZEN_TRUNK", "SNIFFALL_SCRATCH", "REFIT", "REFIT_VIDS", "SCALE", "SCALE_VIDS",
        "SCALE_TAG", "LOLO_EXCLUDE", "DISENTANGLE", "DISENTANGLE_AE", "LABTAIL_FRESH",
        "LABTAIL_HEAD_ONLY", "LABTAIL_EMB_ONLY", "LABTAIL_TUNE_MERGE", "LABTAIL_TRIM",
        "LABTAIL_VIDS", "LABTAIL_SRC", "CACHED_X0_DIR", "CACHED_PREMERGE_DIR", "SKIP_PATH",
        "FILM", "CORAL", "TRAJ_KEEP", "DONOR_LAB", "CURATE_ACTION", "CURATE_VIDS", "FND_TAG",
        "LABTAIL_NEW_HEAD", "LABTAIL_SEED_FROM", "LABTAIL_CACHE", "LABTAIL_CACHE_PASSES",
        "LR_COSINE_T", "LR_FLOOR", "HIDRA_PERLAB_CKPT"}}
    env = dict(env,
               SNIFFALL="1",                          # canonical 82-column head layout (adds sniffall)
               HIDRA_DATA_DIR=os.path.abspath(args.data),
               PERLAB_WORKDIR=workdir,
               LABTAIL=args.lab,                      # new-lab adaptation: freeze SSL + feature merge
               LABTAIL_ACTIONS=",".join(actions),     # supervise only these head columns
               LABTAIL_TAG=tag,
               LABTAIL_OUTDIR=out,
               FT_STEPS=str(args.steps),
               FT_LR=str(args.lr),
               CUDA_VISIBLE_DEVICES=str(args.gpu),
               # Deterministic ops on the JAX path, as the LOLO bundle set them, so a
               # `--backend jax` fine-tune is reproducible. Ignored by the torch backend.
               XLA_FLAGS="--xla_gpu_autotune_level=0"
               + (" --xla_gpu_deterministic_ops=true"
                  if getattr(args, "backend", "torch") == "jax" else ""),
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if args.mode == "head":                           # train ONLY out-proj-perlab (frozen tail)
        env["LABTAIL_HEAD_ONLY"] = "1"
    elif args.mode == "embedding":                    # train the lab embedding + head
        env["LABTAIL_EMB_ONLY"] = "1"
    n_train = len(man)
    if args.videos:                                   # restrict the training set to these files
        vids = _resolve_videos(args.videos, man)
        env["LABTAIL_VIDS"] = ",".join(str(v) for v in vids)
        n_train = len(vids)
        print(f"training on {len(vids)} of {len(man)} staged video(s): {sorted(vids)}")
    if args.seed is not None:
        env["LABTAIL_SEED"] = str(args.seed)
    if args.eval_interval:
        env["LABTAIL_EVAL_INTERVAL"] = str(args.eval_interval)
    if args.ddi_steps is not None:
        env["LABTAIL_DDI_STEPS"] = str(args.ddi_steps)
    if args.cache_features:                           # train on cached frozen features
        env["LABTAIL_CACHE"] = "1"
        env["LABTAIL_CACHE_PASSES"] = str(args.cache_passes)
        if getattr(args, "backend", "torch") != "torch":
            sys.exit("ERROR: --cache-features is a PyTorch-backend option; drop --backend jax")
    if args.lr_schedule == "cosine":
        # The bundle's schedule: a cosine from --lr that would reach its floor at
        # max(steps, 15000), so a default 8000-step run ends around 45% of the peak rate.
        env["LR_COSINE_T"] = str(max(args.steps, 15000))
    src_tpl = None
    if args.from_weights:                             # warm-start from something else
        if "{config}" not in args.from_weights:
            sys.exit("ERROR: --from-weights must contain a '{config}' placeholder, e.g. "
                     "lolo_models/{config}__foundation.pkl")
        src_tpl = os.path.abspath(args.from_weights)
        absent = [c for c in configs if not os.path.isfile(src_tpl.format(config=c))]
        if absent:
            sys.exit(f"ERROR: --from-weights has no checkpoint for config(s) {absent} "
                     f"({src_tpl})")
        env["HIDRA_PERLAB_CKPT"] = src_tpl            # the torch trainer's warm-start source
        print(f"  warm-starting from {args.from_weights} instead of the published checkpoints")
    if new_heads:                                     # widen the head (hidra.torch.train_perlab)
        env["LABTAIL_NEW_HEAD"] = ";".join(f"{l},{a}" for l, a in new_heads)
        if seeds:
            env["LABTAIL_SEED_FROM"] = ";".join(
                d if a is None else f"{d},{a}" for d, a in seeds)

    print(f"fine-tuning {args.lab} {actions} (mode={args.mode}, steps={args.steps}, lr={args.lr}, "
          f"backend={getattr(args, 'backend', 'torch')}"
          f"{f', cached {args.cache_passes} pass(es)' if args.cache_features else ''}) "
          f"on {n_train} video(s)\n  -> {out}/{{config}}__{tag}.pkl")
    if new_heads:
        how = ("fresh init" if not seeds else "seeded from "
               + ", ".join(d if a is None else f"{d} {a}" for d, a in seeds))
        print(f"  adding {len(new_heads)} NEW head column(s) "
              f"{[f'{l},{a}' for l, a in new_heads]}, {how}"
              f"; the checkpoint's column list is written to {out}/{{config}}__{tag}.heads.json")
        if args.lab not in hb:
            print(f"  {args.lab} is a head-free lab slot: this checkpoint makes it YOUR lab, and "
                  f"{new_actions} are the only heads it has. Predict with --labs {args.lab}.")
    if args.mode == "tail":
        print("  note: mode=tail retrains the per-lab tail, which is SHARED across labs -- in the "
              f"resulting checkpoint only {args.lab} is meaningful, so always predict with "
              f"--labs {args.lab}.")
    if args.dry_run:
        # Everything above has validated the staged data, the heads, the donors and the
        # checkpoints; this prints what would run and stops before claiming a GPU.
        shown = ["HIDRA_DATA_DIR", "LABTAIL", "LABTAIL_ACTIONS", "LABTAIL_NEW_HEAD",
                 "LABTAIL_SEED_FROM", "LABTAIL_HEAD_ONLY", "LABTAIL_EMB_ONLY", "LABTAIL_VIDS",
                 "LABTAIL_CACHE", "LABTAIL_CACHE_PASSES", "LABTAIL_DDI_STEPS", "FT_STEPS",
                 "FT_LR", "LR_COSINE_T", "HIDRA_PERLAB_CKPT", "LABTAIL_TAG", "LABTAIL_OUTDIR"]
        module = ("hidra.torch.train_perlab" if getattr(args, "backend", "torch") == "torch"
                  else "hidra.train_perlab_heads")
        print("\n--dry-run: nothing was trained. Per config, this would run")
        print("  " + " ".join(f"{k}={env[k]}" for k in shown if k in env))
        for cfg in configs:
            extra = f" LABTAIL_SRC={src_tpl.format(config=cfg)}" if src_tpl else ""
            print(f"  [{cfg}]{extra} {sys.executable} -m {module} --config {cfg}"
                  f"   -> {out}/{cfg}__{tag}.pkl")
        return

    done, failed = [], []
    for i, cfg in enumerate(configs, 1):
        dest = os.path.join(out, f"{cfg}__{tag}.pkl")
        if os.path.isfile(dest) and not args.overwrite:
            print(f"[{i}/{len(configs)}] {cfg}: {dest} exists, skipping (--overwrite to redo)")
            done.append(cfg); continue
        print(f"[{i}/{len(configs)}] training {cfg} ...", flush=True)
        module = ("hidra.torch.train_perlab" if getattr(args, "backend", "torch") == "torch"
                  else "hidra.train_perlab_heads")
        cmd = [sys.executable, "-m", module, "--config", cfg]
        if args.smoke:
            cmd.append("--smoke")
        # LABTAIL_SRC is the JAX trainer's warm-start source and takes a concrete path, so
        # the template is resolved per config here rather than once above.
        env_cfg = dict(env, LABTAIL_SRC=src_tpl.format(config=cfg)) if src_tpl else env
        r = subprocess.run(cmd, env=env_cfg)
        if r.returncode != 0 or (not args.smoke and not os.path.isfile(dest)):
            print(f"  WARNING: {cfg} exited {r.returncode} and left no checkpoint")
            failed.append(cfg)
        else:
            done.append(cfg)

    print(f"\ntrained {len(done)}/{len(configs)} config(s){' -- failed: ' + str(failed) if failed else ''}")
    if args.smoke:
        print("(--smoke writes no checkpoints; it only proves the data + env are wired up)")
        return
    if not done:
        print("nothing was written; fix the error above (the trainer's own output says why) and rerun.")
        return
    if len(done) < len(CONFIGS):
        print(f"NOTE: inference averages all 5 configs and loads each one's checkpoint, so predict.py "
              f"will fail until {sorted(set(CONFIGS) - set(done))} are trained too.")
    print("predict with the fine-tuned weights:\n"
          f"  python predict.py /path/to/parquets --labs {args.lab} --actions {','.join(actions)} \\\n"
          f"      --out ft_preds/ --weights '{out}/{{config}}__{tag}.pkl' --pix-per-cm N --fps N\n"
          "then re-calibrate thresholds on held-out annotated videos:\n"
          "  python finetune.py calibrate --frames ft_preds/ --annotations heldout_bouts.csv "
          "--out ft_thresholds.csv")
    if new_heads:
        print(f"the new head(s) {[f'{l},{a}' for l, a in new_heads]} are known to predict.py through "
              f"--weights (their {{config}}__{tag}.heads.json sidecar); `calibrate` gives each a "
              f"threshold row, after which `--thresholds ft_thresholds.csv` lists them without "
              f"--weights too.")


def _resolve_videos(spec, man):
    """--videos accepts filenames, stems, or raw video_ids; returns video_ids."""
    by_stem = {os.path.splitext(str(f))[0]: int(v) for f, v in zip(man["file"], man["video_id"])}
    ids = set(int(v) for v in man["video_id"])
    out = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        stem = os.path.splitext(os.path.basename(tok))[0]
        if stem in by_stem:
            out.append(by_stem[stem])
        elif tok.isdigit() and int(tok) in ids:
            out.append(int(tok))
        else:
            sys.exit(f"ERROR: --videos {tok!r} is not a staged video (staged: {sorted(by_stem)})")
    return sorted(set(out))


# ------------------------------------------------------------------ calibrate
def _frames_files(spec):
    fs = []
    for tok in spec if isinstance(spec, list) else [spec]:
        if os.path.isdir(tok):
            fs += sorted(glob.glob(os.path.join(tok, "*.frames.parquet")))
        else:
            fs += sorted(glob.glob(tok))
    if not fs:
        sys.exit(f"ERROR: no *.frames.parquet found in {spec}. Run predict.py with "
                 "--output both|probs first.")
    return fs


def cmd_calibrate(args):
    """Sweep the decision threshold of every (lab, action) against your annotations, pick the
    best-F1 value per head, and write a thresholds CSV predict.py can read with --thresholds."""
    annot = read_annotations(args.annotations, args.stop_inclusive)
    files = _frames_files(args.frames)
    grid = np.round(np.arange(args.min_threshold, args.max_threshold + 1e-9, args.step), 4)

    # gather per-head (score, label) over every scored (video, subject, target) track
    acc = {}
    predicted_actions = set()
    for f in files:
        stem = os.path.basename(f)[: -len(".frames.parquet")]
        fr = pd.read_parquet(f, columns=["frame", "subject", "target", "lab", "action", "prob"])
        predicted_actions |= set(fr["action"].unique())
        va = annot[annot["stem"] == stem]
        if va.empty:
            print(f"  note: no annotations for {stem}; skipped")
            continue
        annotated_actions = set(va["action"])
        # a (subject, target, action) track is scorable only where the video actually annotates
        # that combination -- elsewhere "no bout" means "not looked at", not "negative".
        scored = {(r.agent, r.target, r.action) for r in va.itertuples()}
        for (subj, tgt, lab, act), g in fr.groupby(["subject", "target", "lab", "action"], sort=False):
            if act not in annotated_actions:
                continue
            if (subj, tgt, act) not in scored and not args.all_pairs:
                continue
            n = int(g["frame"].max()) + 1
            y = np.zeros(n, bool)
            for r in va[(va["agent"] == subj) & (va["target"] == tgt) & (va["action"] == act)].itertuples():
                y[r.start_frame: min(r.stop_frame, n)] = True
            p = np.zeros(n, np.float32)
            p[g["frame"].to_numpy()] = g["prob"].to_numpy(np.float32)
            a = acc.setdefault((lab, act), {"p": [], "y": [], "tracks": 0, "videos": set()})
            a["p"].append(p); a["y"].append(y); a["tracks"] += 1; a["videos"].add(stem)
    unpredicted = sorted(set(annot["action"]) - predicted_actions)
    if unpredicted:
        print(f"  note: annotated action(s) {unpredicted} have no prediction track in these files and "
              "are not scored -- either predict.py was not asked for them (--actions/--labs), or the "
              "CSV does not use the head names `predict.py --list-heads` prints (the sniff-splitting "
              "labs' sniff head is 'sniffall').")
    if not acc:
        sys.exit("ERROR: nothing to calibrate -- no (video, subject, target, action) track in the "
                 "predictions matched an annotated combination. Check that the annotation `file` "
                 "names match the prediction filenames, and that agent/target match the mouse ids.")

    rows, report = [], []
    for (lab, act), a in sorted(acc.items()):
        p = np.concatenate(a["p"]); y = np.concatenate(a["y"])
        npos = int(y.sum())
        if npos < args.min_positives:
            print(f"  skipping {lab}__{act}: only {npos} positive frame(s) "
                  f"(< --min-positives {args.min_positives})")
            continue
        # one pass per candidate threshold; frames are cheap and the grid is small
        best = None
        for t in grid:
            c = p >= t
            tp = int(np.count_nonzero(c & y)); fp = int(np.count_nonzero(c & ~y)); fn = npos - tp
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if best is None or f1 > best[1]:
                best = (float(t), f1, tp, fp, fn)
        t, f1, tp, fp, fn = best
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append((f"{lab}__{act}", t))
        report.append(dict(head=f"{lab}__{act}", threshold=t, f1=round(f1, 4),
                           precision=round(prec, 4), recall=round(rec, 4), pos_frames=npos,
                           frames=int(y.size), tracks=a["tracks"], videos=len(a["videos"])))
    if not rows:
        sys.exit("ERROR: no head had enough positive frames to calibrate; lower --min-positives "
                 "or annotate more data.")
    rep = pd.DataFrame(report)
    print("\n" + rep.to_string(index=False))
    out_rows = dict(rows)
    if not args.no_merge:
        # start from the bundled train-calibrated thresholds so heads you did NOT calibrate keep
        # their published value instead of falling back to predict.py's 0.30 default.
        base = pd.read_csv(hidra_predict.THR_CSV)
        merged = {str(k): float(v) for k, v in zip(base[base.columns[0]], base[base.columns[1]])}
        merged.update(out_rows)
        out_rows = merged
    pd.DataFrame(sorted(out_rows.items()), columns=["", "threshold"]).to_csv(args.out, index=False)
    print(f"\nwrote {len(rows)} calibrated threshold(s)"
          f"{f' merged into {len(out_rows)} rows' if not args.no_merge else ''} -> {args.out}")
    if args.report:
        rep.to_csv(args.report, index=False)
        print(f"wrote per-head scores -> {args.report}")
    print(f"use it: python predict.py ... --thresholds {args.out}")
    print("NOTE: thresholds calibrated on the same videos you trained on are optimistic -- "
          "hold videos out for an honest number.")


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(prog="finetune.py", description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="stage tracking parquets + a bout CSV for training",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--tracking", required=True, help="folder of pose parquets (as for predict.py)")
    p.add_argument("--annotations", required=True,
                   help="bout CSV (file,agent,target,action,start_frame,stop_frame), or a folder "
                        "of per-video annotation parquets (agent_id,target_id,action,start_frame,"
                        "stop_frame) named by tracking stem or video id")
    p.add_argument("--also-scored", metavar="A,T,ACTION;...",
                   help="(agent,target,action) combinations every staged video scored, even where "
                        "it has no bout of them -- their frames become negatives. Staging then "
                        "also includes tracking files with no annotations at all")
    p.add_argument("--lab", required=True, help="lab slot to adopt (see: python predict.py --list-heads)")
    p.add_argument("--out", default="ft_data", help="staging dir (default: ft_data)")
    p.add_argument("--pix-per-cm", type=float, help="pixels-per-cm for all files (metadata.csv rows override)")
    p.add_argument("--fps", type=float, help="frame rate for all files")
    p.add_argument("--stop-inclusive", action="store_true",
                   help="stop_frame is the LAST positive frame (predict.py bouts.csv convention)")
    p.add_argument("--drop-unsupported", action="store_true",
                   help="drop annotated actions the adopted lab has no head for, instead of erroring")
    p.add_argument("--thresholds", help="thresholds CSV defining the available heads (default: bundled)")
    p.add_argument("--new-head", metavar="LAB,ACTION", action="append",
                   help="also accept bouts labelled ACTION, a vocabulary behaviour the lab has no head "
                        "for yet, to train a NEW head column with `train --new-head` (repeatable; see "
                        "docs/new-behaviours.md)")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="fine-tune the adopted lab's head(s) on the staged data",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--data", required=True, help="staging dir written by `prepare`")
    p.add_argument("--lab", required=True, help="the adopted lab")
    p.add_argument("--actions", help="comma-separated behaviours to fine-tune (default: all annotated)")
    p.add_argument("--out", default="ft_models", help="checkpoint dir (default: ft_models)")
    p.add_argument("--tag", help="checkpoint tag -> {out}/{config}__{tag}.pkl (default: {lab}_{mode})")
    p.add_argument("--mode", choices=["head", "tail", "embedding"], default="head",
                   help="head: retrain only the linear head (default; leaves every other lab's head "
                        "untouched). tail: also retrain the per-lab LSTM/FF tail + embedding "
                        "(stronger, needs more data, and the other labs' heads in the resulting "
                        "checkpoint become unusable). embedding: lab embedding + head.")
    p.add_argument("--steps", type=int, default=8000, help="training steps per config (default 8000)")
    p.add_argument("--lr", type=float, default=0.004, help="peak Adam LR (default 0.004)")
    p.add_argument("--videos", help="comma-separated filenames/stems/video_ids to train on (default: all staged)")
    p.add_argument("--configs", help=f"comma-separated configs to train (default all 5: {','.join(CONFIGS)})")
    p.add_argument("--gpu", default="0", help="CUDA device index (default 0)")
    p.add_argument("--seed", type=int, help="training seed (data order + augmentation)")
    p.add_argument("--eval-interval", type=int, help="steps between validation passes")
    p.add_argument("--workdir", help="scratch dir for the tracking cache (default: /dev/shm/hidra_finetune_*)")
    p.add_argument("--reuse-cache", action="store_true",
                   help="keep the scratch tracking cache from a previous run (faster; only safe if "
                        "the staged data has not changed)")
    p.add_argument("--overwrite", action="store_true", help="retrain configs whose checkpoint already exists")
    p.add_argument("--smoke", action="store_true", help="600-step dry run, writes no checkpoint")
    p.add_argument("--dry-run", action="store_true",
                   help="validate the staged data, heads, donors and checkpoints, print the "
                        "commands that would run, and stop without touching a GPU")
    p.add_argument("--thresholds", help="thresholds CSV defining the available heads (default: bundled)")
    p.add_argument("--backend", default="torch", choices=["torch", "jax"],
                   help="training backend: torch (default) or jax (the original). Both write the "
                        "checkpoint in the same layout, so predict.py --weights loads either one")
    p.add_argument("--new-head", metavar="LAB,ACTION", action="append",
                   help="add a head column for this (lab, action) -- LAB must be --lab, ACTION a "
                        "vocabulary behaviour the lab has no head for -- and train it (torch backend; "
                        "repeatable). The published columns are copied over unchanged; the checkpoint "
                        "gets a {config}__{tag}.heads.json sidecar naming its columns. LAB may be a "
                        "head-free slot, which makes it your own lab (see docs/new-behaviours.md)")
    p.add_argument("--seed-from", metavar="LAB[,ACTION]", action="append",
                   help="with --new-head: where the new column(s) start instead of a fresh init. "
                        "A bare 'LAB' is a donor lab -- each new column starts from that lab's column "
                        "of the same behaviour, and the lab-embedding row from the donor too, which is "
                        "the new-lab recipe. 'LAB,ACTION' names one donor column: given once it seeds "
                        "every new column, and repeated it seeds the column of its own behaviour, so "
                        "'--seed-from A --seed-from B,attack' means A for everything except attack")
    p.add_argument("--from-weights", metavar="TEMPLATE",
                   help="warm-start from these per-config checkpoints instead of the published ones "
                        "(a '{config}' template, as predict.py --weights takes): a leave-one-lab-out "
                        "foundation, or an earlier fine-tune you want to add behaviours to")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="constant --lr (default), or a cosine decay toward zero that would reach it "
                        "at max(--steps, 15000) steps -- the schedule the LOLO bundle used")
    p.add_argument("--cache-features", action="store_true",
                   help="compute the frozen part of the model once per training window and fit the "
                        "trainable part on the cache. Same objective, far fewer seconds per step; the "
                        "augmentation is then a fixed set of --cache-passes draws (torch backend)")
    p.add_argument("--cache-passes", type=int, default=4, metavar="N",
                   help="with --cache-features: augmentation draws cached per window (default 4). "
                        "More = more augmentation variety and more memory")
    p.add_argument("--ddi-steps", type=int, metavar="N",
                   help="batches of input-statistics recalibration before training (default 256, the "
                        "original's). The dominant cost of a cached run -- see docs/fine-tuning.md")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("calibrate", help="pick best-F1 thresholds from predictions + annotations",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--frames", required=True, nargs="+",
                   help="predict.py output dir, or *.frames.parquet file(s)/glob(s)")
    p.add_argument("--annotations", required=True, help="bout CSV for those videos (held-out, ideally)")
    p.add_argument("--out", default="ft_thresholds.csv", help="thresholds CSV to write (default: ft_thresholds.csv)")
    p.add_argument("--report", help="also write the per-head F1/precision/recall table here")
    p.add_argument("--stop-inclusive", action="store_true", help="stop_frame is the LAST positive frame")
    p.add_argument("--all-pairs", action="store_true",
                   help="score every predicted mouse pair in a video that annotates the action, not "
                        "just the annotated pairs (treats un-annotated pairs as all-negative)")
    p.add_argument("--min-positives", type=int, default=100,
                   help="skip heads with fewer positive frames (default 100)")
    p.add_argument("--min-threshold", type=float, default=0.05)
    p.add_argument("--max-threshold", type=float, default=0.95)
    p.add_argument("--step", type=float, default=0.01, help="threshold grid step (default 0.01)")
    p.add_argument("--no-merge", action="store_true",
                   help="write ONLY the calibrated heads (default merges them into the bundled "
                        "thresholds, so uncalibrated heads keep their published value)")
    p.set_defaults(func=cmd_calibrate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
