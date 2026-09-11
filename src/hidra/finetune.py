#!/usr/bin/env python3
"""
HiDRA fine-tuning -- adapt one lab's classifier head(s) to YOUR videos and YOUR annotations.

Zero-shot (predict.py) applies a lab's classifier as-is. Fine-tuning warm-starts from that same
classifier and continues training it on your labelled data, so it learns your arena, your pose
rig, and your annotation style. The self-supervised backbone and the shared feature merge stay
frozen; only the per-lab tail and/or the per-lab head are updated (see --mode).

WHAT YOU CAN FINE-TUNE
  Head columns exist only for (lab, action) pairs the model was trained on, and the lab
  embedding table is fixed -- so you cannot invent a new lab or a new behaviour name. You
  ADOPT an existing (lab, action) head whose behaviour matches yours:
      python predict.py --list-heads          # every (lab, action) pair
  Pick the lab whose zero-shot output looked closest on your data (docs/zero-shot.md).

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
    `sniffall` head, the sniff-family labels that head is derived from."""
    acts = set(hb[lab])
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


def read_annotations(path, stop_inclusive=False):
    """Load the bout CSV -> DataFrame[stem, agent, target, action, start_frame, stop_frame],
    with stop_frame EXCLUSIVE and `stem` the tracking filename without its extension."""
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


def check_actions(annot, lab, thr_csv=None, drop_unsupported=False):
    """Every annotated action must be a head of `lab`, else there is nothing to fine-tune."""
    hb = heads_by_lab(thr_csv)
    if lab not in hb:
        sys.exit(f"ERROR: no classifier heads for lab {lab!r}. Available labs: {sorted(hb)}")
    mine = annotatable_actions(lab, hb)
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
def cmd_prepare(args):
    """Stage tracking parquets + bouts into {out}/TRAIN.csv + {out}/train_{tracking,annotation}/{lab}/."""
    annot = read_annotations(args.annotations, args.stop_inclusive)
    annot = check_actions(annot, args.lab, args.thresholds, args.drop_unsupported)

    parquets = hidra_predict.discover(args.tracking)
    if not parquets:
        sys.exit(f"no .parquet/.pkt files in {args.tracking}")
    by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in parquets}
    unknown = sorted(set(annot["stem"]) - set(by_stem))
    if unknown:
        sys.exit(f"ERROR: annotated file(s) not found in {args.tracking}: {unknown}\n"
                 f"  tracking files present: {sorted(by_stem)}")
    used = [by_stem[s] for s in sorted(set(annot["stem"]))]
    skipped = sorted(set(by_stem) - set(annot["stem"]))
    if skipped:
        print(f"  note: {len(skipped)} tracking file(s) have no annotations and are NOT staged "
              f"(an unannotated video teaches the model nothing): {skipped}")
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
        labeled = sorted({f"{r.agent_id},{r.target_id},{r.action}" for r in out_a.itertuples()})
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
    heads = trainable_heads(args.lab, set(annot["action"]), heads_by_lab(args.thresholds))
    print(f"\nstaged {len(man)} video(s) for lab {args.lab} -> {args.out}/")
    print(f"  labels staged: {sorted(set(annot['action']))}")
    print(f"  head columns these can train: {heads}")
    print(f"next: python finetune.py train --data {args.out} --lab {args.lab} "
          f"--actions {','.join(heads)} --out ft_models/ --tag mytag")


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
    if args.actions:
        actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    else:
        actions = trainable_heads(args.lab, annotated, hb)
        if not actions:
            sys.exit(f"ERROR: none of the staged labels {annotated} can train a {args.lab} head "
                     f"(its heads: {sorted(hb.get(args.lab, ()))})")
    bad = sorted(set(actions) - hb.get(args.lab, set()))
    if bad:
        hint = ("; label sniffing 'sniffall' in the CSV, or pass --actions sniffall"
                if set(bad) & set(SNIFF_FAMILY) and "sniffall" in hb.get(args.lab, set()) else "")
        sys.exit(f"ERROR: {args.lab} has no head for {bad}; its heads are {sorted(hb[args.lab])}{hint}")
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

    from .download_models import require_weights
    require_weights()                      # warm-starting needs the base checkpoints

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    workdir = args.workdir or f"/dev/shm/hidra_finetune_{args.lab}_{tag}"
    if not args.reuse_cache:
        # the tracking/label cache under workdir is keyed by video_id only, so a re-`prepare`d
        # dataset would silently train on the OLD frames; rebuild it unless told otherwise.
        shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)

    # train_perlab_heads.py is driven entirely by env vars and asserts that the experiment
    # switches are mutually exclusive -- drop any left over in the caller's shell.
    env = {k: v for k, v in os.environ.items() if k not in {
        "FROZEN_TRUNK", "SNIFFALL_SCRATCH", "REFIT", "REFIT_VIDS", "SCALE", "SCALE_VIDS",
        "SCALE_TAG", "LOLO_EXCLUDE", "DISENTANGLE", "DISENTANGLE_AE", "LABTAIL_FRESH",
        "LABTAIL_HEAD_ONLY", "LABTAIL_EMB_ONLY", "LABTAIL_TUNE_MERGE", "LABTAIL_TRIM",
        "LABTAIL_VIDS", "LABTAIL_SRC", "CACHED_X0_DIR", "CACHED_PREMERGE_DIR", "SKIP_PATH",
        "FILM", "CORAL", "TRAJ_KEEP", "DONOR_LAB", "CURATE_ACTION", "CURATE_VIDS", "FND_TAG"}}
    env = dict(env,
               SNIFFALL="1",                          # canonical 90-head layout (adds sniffall)
               HIDRA_DATA_DIR=os.path.abspath(args.data),
               PERLAB_WORKDIR=workdir,
               LABTAIL=args.lab,                      # new-lab adaptation: freeze SSL + feature merge
               LABTAIL_ACTIONS=",".join(actions),     # supervise only these head columns
               LABTAIL_TAG=tag,
               LABTAIL_OUTDIR=out,
               FT_STEPS=str(args.steps),
               FT_LR=str(args.lr),
               CUDA_VISIBLE_DEVICES=str(args.gpu),
               XLA_FLAGS="--xla_gpu_autotune_level=0",
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

    print(f"fine-tuning {args.lab} {actions} (mode={args.mode}, steps={args.steps}, lr={args.lr}, "
          f"backend={getattr(args, 'backend', 'torch')}) on {n_train} video(s)"
          f"\n  -> {out}/{{config}}__{tag}.pkl")
    if args.mode == "tail":
        print("  note: mode=tail retrains the per-lab tail, which is SHARED across labs -- in the "
              f"resulting checkpoint only {args.lab} is meaningful, so always predict with "
              f"--labs {args.lab}.")
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
        r = subprocess.run(cmd, env=env)
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
              "are not scored. Label with the head names `predict.py --list-heads` prints -- the "
              "sniff-splitting labs' sniff head is 'sniffall'.")
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
    p.add_argument("--annotations", required=True, help="bout CSV: file,agent,target,action,start_frame,stop_frame")
    p.add_argument("--lab", required=True, help="lab slot to adopt (see: python predict.py --list-heads)")
    p.add_argument("--out", default="ft_data", help="staging dir (default: ft_data)")
    p.add_argument("--pix-per-cm", type=float, help="pixels-per-cm for all files (metadata.csv rows override)")
    p.add_argument("--fps", type=float, help="frame rate for all files")
    p.add_argument("--stop-inclusive", action="store_true",
                   help="stop_frame is the LAST positive frame (predict.py bouts.csv convention)")
    p.add_argument("--drop-unsupported", action="store_true",
                   help="drop annotated actions the adopted lab has no head for, instead of erroring")
    p.add_argument("--thresholds", help="thresholds CSV defining the available heads (default: bundled)")
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
    p.add_argument("--thresholds", help="thresholds CSV defining the available heads (default: bundled)")
    p.add_argument("--backend", default="torch", choices=["torch", "jax"],
                   help="training backend: torch (default) or jax (the original). Both write the "
                        "checkpoint in the same layout, so predict.py --weights loads either one")
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
