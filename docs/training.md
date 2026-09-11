# Training HiDRA

The published weights are two stages per ensemble config:

| stage | file | what it is | trained on |
|---|---|---|---|
| 1 | `{config}_unsupervised` | the **self-supervised trunk**: a keypoint-trajectory forecaster (`solution.UnsupervisedModel`), no labels involved | every video, all 21 labs |
| 2 | `{config}_supervised_perlab_sniffall` | the **supervised tail**: feature merge → lab embedding → 3 × (BiLSTM + FFN) → an 82-column per-lab head, on top of the frozen stage-1 trunk | the annotated videos of the 15 competition labs |

Inference loads both; [fine-tuning](fine-tuning.md) re-trains part of stage 2 on one lab's data.
This document is for rebuilding either stage: reproducing the published weights, training the
trunk on a new species or rig, or adding head columns ([new-behaviours.md](new-behaviours.md)).

Both stages run on the **JAX backend only** — the PyTorch port covers inference and the
fine-tuning path, not pretraining or foundation training — so install both extras:

```bash
uv sync --extra jax --extra torch
```

The scripts are the research code that produced the published checkpoints, driven by environment
variables rather than a polished CLI. Expect to read their header comments.

## 0. Environment and where things go

```bash
export HIDRA_DATASET_DIR=/data/hidra_train        # the dataset tree of §1
export HIDRA_MODELS_DIR=/data/hidra_models_new    # OUTPUT weights -- a fresh directory, see below
export HIDRA_WORKDIR=/dev/shm/hidra-train         # memory-mapped tracking cache (RAM-backed is best)
export XLA_PYTHON_CLIENT_PREALLOCATE=false        # do not grab the whole GPU up front
export XLA_FLAGS=--xla_gpu_autotune_level=0       # the 6-bodypart configs hang without it
export CUDA_VISIBLE_DEVICES=0                     # one GPU per process
```

- **Use a fresh `HIDRA_MODELS_DIR`.** Both stages write their result into it —
  `{config}_unsupervised.pkl`, `{config}_supervised_perlab_sniffall.pkl` — and stage 2 reads the
  stage-1 trunk from it. Pointing it at the directory holding the downloaded weights would
  overwrite nothing (the loader prefers `.safetensors`, so a freshly written `.pkl` next to the
  published `.safetensors` is silently ignored) and confuse everything. Copy `thresholds.json` in
  (stage 2 needs it, §3) and nothing else.
- **`HIDRA_WORKDIR` caches derived data keyed by video id** — `tmp/train/videos.pkl` and the
  memory-mapped tracking/label arrays. It is not invalidated when the manifest changes: delete it
  after editing the dataset.
- Checkpoints during training go to `experiments/{config}/pretrain/checkpoints/` under the
  **current working directory** (stage 1) and to
  `/dev/shm/doom_exp/{config}_train_perlab_sniffall/checkpoints/` (stage 2); the best three are
  kept and `best.pkl` symlinks to the best. Only the final copy in `HIDRA_MODELS_DIR` is needed
  afterwards.
- `SNIFFALL=1` must be set for stage 2 and for anything that reads the published head. It appends
  the merged `sniffall` action to the vocabulary, which is what the 82-column layout was trained
  with; without it you get a 77-column head under a different file name.

## 1. The dataset layout

Both stages read the same tree, which is also what `finetune.py prepare` writes for one lab:

```
$HIDRA_DATASET_DIR/
  train.csv                                   # the manifest (TRAIN.csv is accepted too)
  train_tracking/<lab_id>/<video_id>.parquet  # long-format pose: video_frame, mouse_id, bodypart, x, y
  train_annotation/<lab_id>/<video_id>.parquet  # agent_id, target_id, action, start_frame, stop_frame (exclusive)
```

`train.csv` columns: `lab_id, video_id, frames_per_second, pix_per_cm_approx, behaviors_labeled`.
`behaviors_labeled` is a JSON list of `"agent,target,action"` strings naming every
(agent, target, action) the video annotates — it is what marks those combinations as supervised,
and their non-bout frames as negatives. A video with `behaviors_labeled = "[]"` and no annotation
parquet is fine for stage 1 and contributes nothing to stage 2.

Rules the pipeline enforces, whether or not you use `prepare`:

- `lab_id` must be one of the 21 names in `schema.LABS`, even for stage 1 (the batch encodes it;
  the trunk itself never reads it). Stage 2 trains the lab's embedding row and head columns.
- every video needs **at least two tracked mice**: stage-1 windows are built on mouse pairs, and
  stage 2 forms an (agent, target) pair per annotated agent.
- bodypart names from the model schema; a pixel scale per video (everything is computed in cm).
- `video_id` is any integer, unique across the manifest. `finetune.py prepare` and `predict.py`
  derive it from the filename (`hidra.cli.vid_of`), which keeps one recording at one id across
  prediction, fine-tuning and training.

To stage a multi-lab or label-free set, run `prepare` once per lab into separate directories and
merge the trees and manifests (it rewrites `TRAIN.csv`), or write the manifest yourself:

```python
import glob, shutil, pandas as pd
from hidra.cli import vid_of

rows = []
for f in sorted(glob.glob("parquets/*.parquet")):
    vid = vid_of(f)
    shutil.copy(f, f"{DATASET}/train_tracking/GroovyShrew/{vid}.parquet")
    rows.append(dict(lab_id="GroovyShrew", video_id=vid, frames_per_second=30.0,
                     pix_per_cm_approx=16.0, behaviors_labeled="[]"))       # unlabelled: stage 1 only
pd.DataFrame(rows).to_csv(f"{DATASET}/train.csv", index=False)
```

## 2. Stage 1: the self-supervised trunk

```bash
python -c "from hidra import solution as s; s.pretrain(s.get_configs()['15fps_5bp'])"
```

Once per config (`11fps_4bp, 15fps_5bp, 19fps_6bp, 23fps_7bp, 27fps_6bp`). What `pretrain` does:

- splits videos 85/15 by duration within each lab (seeded by the config, so the split is the one
  the published model used), and builds unsupervised windows: 64 output frames at the config's
  sample rate, 32 frames of padding either side, one random mouse pair per video per epoch, with
  the config's scale / time-dilation / rotation / flip / keypoint-noise augmentation;
- trains `UnsupervisedModel(d_res=192, d_lstm=192, d_ff=384, d_edge=96, n_layers=4)` in bfloat16
  with Adam at 0.02, batch 128, for up to 125,000 steps, evaluating validation NLL every 2,000
  steps and stopping after 15,000 steps without improvement; an EMA (decay 0.9993) of the weights
  is what gets evaluated and saved;
- copies the best EMA checkpoint to `$HIDRA_MODELS_DIR/{config}_unsupervised.pkl`.

Dry-run the data and environment before committing a GPU for days:

```bash
python -c "from hidra import solution as s; s.pretrain(s.get_configs()['15fps_5bp'], max_training_steps=30)"
```

That is the only knob `pretrain` exposes; everything else is fixed to what produced the published
trunks. Measured on one RTX A6000 the step rate is about 1.4 s per batch of 128, so the full
budget is on the order of two days per config (the published trunks were trained on a TPU v5e-8
slice). Early stopping usually ends it sooner.

## 3. Stage 2: the supervised per-lab tail and heads

```bash
cp models/thresholds.json $HIDRA_MODELS_DIR/          # the head table -- see below
SNIFFALL=1 python -m hidra.train_perlab_heads --config 15fps_5bp --smoke    # 600 steps, writes nothing
SNIFFALL=1 python -m hidra.train_perlab_heads --config 15fps_5bp
```

Once per config, after that config's stage-1 trunk is in `HIDRA_MODELS_DIR` (published or your
own). With no experiment flags set, `train_perlab_heads.py` trains the **whole supervised model
from scratch** on top of the frozen trunk — feature merge, lab embedding, the three BiLSTM/FFN
blocks, and both heads:

- the shared 38-way head `out-proj`, whose loss is taken over every video of every lab so the
  tail sees all the data, including the six train-only labs that have no scored head;
- the per-lab head `out-proj-perlab`, one column per (lab, behaviour), whose loss is routed so each
  video supervises only its own lab's columns. Inference reads only this head.

Adam at 0.004, batch 128, up to 50,000 steps, validation F1 (per-lab heads, competition labs)
every 2,000 steps with 10,000 steps of patience, EMA weights. The result is copied to
`$HIDRA_MODELS_DIR/{config}_supervised_perlab_sniffall.pkl`. Measured on one A6000: roughly
0.3–0.7 s per step, so 4–10 hours per config at the full budget.

**The head table.** The columns of `out-proj-perlab` — which (lab, behaviour) pairs get a
classifier, in which order — are the keys of `thresholds.json` in `HIDRA_MODELS_DIR`, minus
PleasantMeerkat's attack/chase/escape (`pm_rule.py`), plus a `sniffall` column for each of the
five sniff-splitting labs when `SNIFFALL=1`. The published table has 80 keys and yields the 82
columns. Its threshold *values* are not read by HiDRA's inference (`derived_thresholds_train.csv`
supplies those), so a row is `[lab, action, 0.3]` for all the trainer cares. This is the one place
a **new (lab, behaviour) column** is declared — [new-behaviours.md §4](new-behaviours.md#4-the-name-is-in-the-vocabulary-but-no-lab-has-a-head-for-it--a-new-column).
The same `thresholds.json` must sit next to the resulting checkpoints at inference time, because
that is where `schema.lab_action_table()` rebuilds the column list from.

**Which data.** Stage 2 uses the videos whose `behaviors_labeled` is non-empty, split with the same
seeded 85/15 rule as stage 1; validation drops the six train-only labs. `HIDRA_DATA_DIR` overrides
the dataset root for this script alone (it is how `finetune.py` points it at a staged folder), and
`PERLAB_WORKDIR` gives one run a private tracking cache so several configs can train concurrently
without racing on `HIDRA_WORKDIR`.

## 4. Thresholds and using the result

```bash
for f in /data/hidra_models_new/*.pkl; do hidra-convert-weights --one "$f"; done   # optional: .pkl -> .safetensors, verified bit-equal
HIDRA_MODELS_DIR=/data/hidra_models_new hidra-predict --list-heads
HIDRA_MODELS_DIR=/data/hidra_models_new hidra-predict held_out/ --out preds/ --labs GroovyShrew --pix-per-cm 16 --fps 30
hidra-finetune calibrate --frames preds/ --annotations held_out_bouts.csv --out my_thresholds.csv
HIDRA_MODELS_DIR=/data/hidra_models_new HIDRA_THRESHOLDS=my_thresholds.csv hidra-predict videos/ --out results/ ...
```

(`hidra-convert-weights` without `--one` expects all ten published pickles, which a partial rebuild
does not have.) `predict.py` resolves both stages from `HIDRA_MODELS_DIR` (either container) and reads the list of
heads and their decision thresholds from a thresholds CSV — the bundled
`derived_thresholds_train.csv` by default, or `--thresholds` / `$HIDRA_THRESHOLDS`. Freshly
trained heads are on their own probability scale, so calibrate them on held-out annotated
recordings with `finetune.py calibrate` exactly as after a fine-tune
([fine-tuning.md §4](fine-tuning.md#4-calibrate-the-thresholds)). The bundled table's own
derivation (leakage-free, calibrated on the consortium training data) is not part of this
repository; `calibrate` is the supported tool. If your head table has columns the bundled CSV does
not list, add `Lab__action,threshold` rows so `--list-heads` and `--actions` know about them.

To publish a weight set to the Hub: `hidra-publish-weights --dry-run`, then without `--dry-run`
([models/README.md](../models/README.md)).

## 5. Tail training on one lab's data

Re-training only the per-lab tail on a single lab — warm-started from the published stage-2
checkpoint, with the trunk and feature merge frozen — is the `LABTAIL` path of the same script,
and it is what `finetune.py train` drives with `--mode tail` (and, more narrowly, `--mode
embedding` / `--mode head`). It is ported to PyTorch, needs no JAX, and is documented in
[fine-tuning.md](fine-tuning.md). Reach for it before either stage above: it is the cheap way to
make the published model speak your lab's dialect.

## Research levers

`train_perlab_heads.py` carries the experiments the paper was built from, all gated on environment
variables and all **JAX-only**; `finetune.py` strips them from the environment before it runs the
trainer, so they never fire by accident. The header comments in the file are the reference; this
is the map:

| variable | what it does |
|---|---|
| `LABTAIL=<lab>` (+ `LABTAIL_ACTIONS`, `LABTAIL_VIDS`, `LABTAIL_HEAD_ONLY`, `LABTAIL_EMB_ONLY`, `LABTAIL_TAG`, `LABTAIL_OUTDIR`, `LABTAIL_SEED`) | new-lab adaptation — what `finetune.py train` sets |
| `LABTAIL_FRESH=1` | re-initialise the tail + head instead of warm-starting (cold start) |
| `LABTAIL_SRC=<file>` | warm-start from a different checkpoint, e.g. a LOLO foundation |
| `LABTAIL_TUNE_MERGE=1` | also fine-tune the feature merge (needs the live SSL path) |
| `DONOR_LAB=<lab>` | seed the adopted lab's embedding row and head columns from a pose-similar lab |
| `CURATE_ACTION`, `CURATE_VIDS` | supervise one action only on the listed videos |
| `LOLO_EXCLUDE=<lab>` | leave-one-lab-out foundation: full stage 2 with that lab removed |
| `FROZEN_TRUNK=1` | warm-start everything, train only new head columns on frozen features — how `sniffall` was added |
| `REFIT=<lab>,<action>` (+ `REFIT_VIDS`) | refit one existing column in place on a video subset |
| `SCALE=<lab>,<action>` (+ `SCALE_VIDS`, `SCALE_TAG`) | data-size sweep: re-initialise one column and fit it on N videos |
| `SKIP_PATH`, `FILM`, `CORAL`, `DISENTANGLE` | architectural experiments on the merge / lab-invariance |
| `CACHED_X0_DIR`, `CACHED_PREMERGE_DIR` | train the tail on precomputed features, skipping the trunk |
| `FT_STEPS`, `FT_LR`, `LR_COSINE_T`, `LR_FLOOR`, `WEIGHT_DECAY`, `GRAD_CLIP` | optimizer schedule (defaults reproduce the published runs) |
| `TRAJ_KEEP=N` | keep every evaluation checkpoint, disable early stopping |

The original competition entry point, `solution.train()` + `solution.predict()` +
`solution.compute_ensemble_thresholds()`, trains the earlier *shared* 37-way model and derives a
`thresholds.pkl` from its validation sweep. That is how the head table's keys were first
produced; HiDRA itself ships the per-lab model above, so those functions are kept as reference
rather than as a workflow.

## Measured on one RTX A6000

| run | backend | rate | note |
|---|---|---|---|
| stage 1 `pretrain`, 15fps_5bp | JAX | ~1.4 s/step, batch 128 | 125k-step budget ≈ 2 days/config |
| stage 2 `train_perlab_heads`, 15fps_5bp, from scratch | JAX | 0.3–0.7 s/step | 50k-step budget ≈ 4–10 h/config |
| `finetune.py train --mode head`, 15fps_5bp | PyTorch | ~0.55 s/step | + ~2 min input-statistics pass |
| `finetune.py train --mode head`, 23fps_7bp | PyTorch | ~1.05 s/step | |
| `finetune.py train --mode head --backend jax`, 15fps_5bp | JAX | ~0.4 s/step | + ~1 min XLA compile |
| `predict.py`, one lab, three 1-minute videos, 5 configs | PyTorch | 8 s | |

Throughput on the small synthetic dataset used here is bound by the numpy data pipeline (forward-
only initialization batches take as long as training steps); larger videos and more CPU cores
shift that.
