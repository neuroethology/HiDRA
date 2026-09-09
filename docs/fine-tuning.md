# Fine-tuning

Fine-tuning takes one lab's trained classifier head and continues training it on **your** videos
and **your** annotations. The self-supervised backbone and the shared feature merge stay frozen;
only the per-lab parts are updated. That is what makes this cheap — you are adapting a classifier
that already knows mouse social behaviour, not training one from scratch — and it is why a few
annotated videos can be enough.

Do this when zero-shot output ([zero-shot.md](zero-shot.md)) is *close but consistently off*: the
model finds the behaviour but calls it too eagerly, cuts bouts short, or misses the variant your
arena produces. If it is only the decision threshold that is wrong, go straight to
[§4 Calibrate](#4-calibrate-the-thresholds) — no training needed.

## Limits (read this before you plan an experiment)

- **You cannot add a behaviour.** The head layer has one column per (lab, action) pair that was
  trained, of which 82 are published, and the action vocabulary is fixed at the 37 names in
  `solution.ACTIONS`. You **adopt** one of those published heads — the one whose behaviour matches
  yours (`python predict.py --list-heads`).
- **You cannot add a lab.** The lab-embedding table is fixed at 21 rows, so your data trains
  under an existing lab's slot. Everything downstream — inference, thresholds — then refers to
  your data by that lab name. Pick the lab whose zero-shot calls were closest.
- **One behaviour per (subject, target) per frame.** Labels are stored as one action id per pair
  per frame, so overlapping bouts of different behaviours on the same pair overwrite each other.
- **Un-annotated frames are negatives.** For each (agent, target, action) a video annotates,
  every frame outside a bout is a negative example for that combination. Annotate each behaviour
  exhaustively within a video, and leave out videos you only skimmed. Combinations a video does
  *not* annotate are ignored entirely, not treated as absent.
- **The ensemble is five configs.** Inference averages the 11/15/19/23/27-fps models and loads a
  checkpoint for each, so a fine-tune is only usable once all five are trained. While iterating
  you can train one (`--configs 15fps_5bp`) and predict with just that one
  (`predict.py --configs 15fps_5bp`) — faster, lower quality, not a result.

## 0. Set up

Weights (`python download_models.py`), the `requirements.txt` environment, and one GPU. Training
writes its scratch cache to `/dev/shm`, so this is a Linux-with-a-GPU workflow — unlike inference,
it is not something to try on a laptop.

## 1. Annotate

One CSV of bouts across all your videos:

```csv
file,agent,target,action,start_frame,stop_frame
mouseA_day1.parquet,mouse1,self,rear,100,160
mouseA_day1.parquet,mouse1,mouse2,sniffgenital,412,455
mouseB_day1.parquet,mouse2,self,rear,88,140
```

- `file` — the tracking parquet's filename (or its stem).
- `agent` / `target` — `mouse1`…`mouse4`, or `self` for a self-directed behaviour
  (`rear, dig, climb, selfgroom, rest, run, freeze, huddle, biteobject, exploreobject`).
- `action` — must be a behaviour the lab you adopt has a head for.
- `start_frame`/`stop_frame` — **`stop_frame` is exclusive** (frames `start … stop-1` are
  positive), matching the trainer. If your `stop_frame` is the last positive frame — which is
  what `predict.py`'s `bouts.csv` writes — pass `--stop-inclusive` and it is converted for you.

How much is enough? There is no single answer — it depends on how far your setup is from the
adopted lab's. Measure it rather than guess: hold out at least one recording, then run
[`calibrate`](#4-calibrate-the-thresholds) twice, once on your zero-shot predictions for those
held-out videos and once on the fine-tuned ones, and compare the F1 columns. Two or three
annotated recordings per behaviour is a reasonable place to start.

## 2. Stage the data

```bash
python finetune.py prepare \
    --tracking /path/to/parquets \
    --annotations bouts.csv \
    --lab GroovyShrew \
    --out ft_data/ \
    --pix-per-cm 16 --fps 30
```

This writes the layout the trainer reads — `ft_data/TRAIN.csv` plus
`ft_data/train_{tracking,annotation}/GroovyShrew/<video_id>.parquet` — and validates as it goes:
bodypart names against the model schema, annotated mice against the tracked mice, bouts against
the video length, and every annotated action against the adopted lab's heads (it tells you which
labs *do* have a behaviour when the one you picked does not, which is the fastest way to choose a
slot). Videos with no annotations are skipped, and `metadata.csv` in the tracking folder
overrides `--pix-per-cm`/`--fps` per file, exactly as for `predict.py`.

Video ids are derived from the filename with the same hash `predict.py` uses, so a given
recording has one id across prediction and training.

It prints the positive-frame percentage per video — worth a look, because a behaviour that
occupies well under 1% of frames gives the head very few positive examples to learn from and
makes the threshold sweep in §4 noisy. Annotate more of it before drawing conclusions.

## 3. Train

```bash
python finetune.py train \
    --data ft_data/ --lab GroovyShrew --actions rear \
    --out ft_models/ --tag myrig
```

One run per config, warm-started from the canonical checkpoint, writing
`ft_models/{config}__myrig.pkl`. Supervision is masked to the head columns you name in
`--actions`, so nothing else gets gradient.

**`--mode` decides what is trainable — the main choice you have to make:**

| mode | trains | other labs' heads | use when |
|---|---|---|---|
| `head` (default) | the linear head only (`out-proj-perlab`) | weights untouched — no gradient reaches them | your setup resembles the adopted lab's; you have little data; you want the cheapest, safest adaptation |
| `tail` | the per-lab LSTM/FF tail + lab embedding + head | **invalidated** — the tail is shared | your arena, pose rig or annotation style differs substantially, and you have several annotated recordings |
| `embedding` | the lab embedding + head | weights untouched | middle ground: re-place your data in lab space without retraining the shared tail |

`tail` is the strongest adaptation and the one the underlying research code was built around, but
the tail is shared across labs: after training it, only the lab you adopted is meaningful in that
checkpoint. `head` cannot pull the trunk around at all, trains in a fraction of the time, and is
the right first attempt.

Either way, **treat a fine-tuned checkpoint as belonging to the lab you adopted** and always pass
`--labs <that lab>` when predicting with it. Even in `head` mode the other columns are not
strictly frozen output: training re-estimates the input-normalization statistics (which are
shared across all 90 head columns) on your data during initialization, so other labs' probability
scales can drift a little.

Other knobs: `--steps` (default 8000 per config), `--lr` (default 0.004), `--videos` to restrict
the training set (filenames, stems or ids — useful for a data-size sweep), `--configs` to train
one config while iterating (pair it with `predict.py --configs <same>`), `--seed`, `--gpu`, and
`--smoke` for a 600-step wiring check that writes no checkpoint. Re-running skips configs whose
checkpoint exists unless you pass `--overwrite`.

Validation is carved out of your staged videos automatically (~15% by duration). With only one
or two staged videos there is nothing to hold out, so the validation set overlaps training and
its F1 is not a generalization estimate — use the held-out `calibrate` run in §4 for that.
The reported val-F1 covers only the behaviours your annotations score.

## 4. Calibrate the thresholds

A fine-tuned head's probabilities are no longer on the scale the bundled thresholds were
calibrated for, so **re-calibrating is part of fine-tuning, not an optional extra.** Predict on
held-out annotated videos, then sweep:

```bash
python predict.py /path/to/held_out --labs GroovyShrew --actions rear \
    --out ft_preds/ --weights 'ft_models/{config}__myrig.pkl' --pix-per-cm 16 --fps 30

python finetune.py calibrate --frames ft_preds/ --annotations heldout_bouts.csv \
    --out ft_thresholds.csv --report ft_scores.csv
```

```
             head  threshold     f1  precision  recall  pos_frames  frames  tracks  videos
GroovyShrew__rear       0.52 0.9485      0.955  0.9422         225    2400       3       2
```

The sweep pools every scored (video, subject, target) track per head, takes the best-F1
threshold on a 0.05–0.95 grid, and merges the result into the bundled table so heads you did not
calibrate keep their published value (`--no-merge` writes only yours). By default only
(video, subject, target, action) combinations your CSV actually annotates are scored;
`--all-pairs` also scores un-annotated pairs in those videos as all-negative. `--min-positives`
(default 100) skips heads with too few positive frames to say anything.

Run the same command against your zero-shot predictions to get the before/after F1 that tells
you whether fine-tuning actually helped. Calibrating on the videos you trained on is optimistic —
hold data out.

## 5. Predict with the result

```bash
python predict.py /path/to/new_videos \
    --labs GroovyShrew --actions rear \
    --weights 'ft_models/{config}__myrig.pkl' \
    --thresholds ft_thresholds.csv \
    --out results/ --pix-per-cm 16 --fps 30
```

Quote the `--weights` template so the shell leaves `{config}` alone; HiDRA fills it in with the
name of each config it runs. Everything else about the output is identical to zero-shot — same
`bouts.csv`, same `frames.parquet`, same [`hidra`](../hidra.py) helpers.

## What is actually happening

Per config, the model is: frozen SSL forecaster → frozen feature merge → `x0` → `+0.1 ×`
lab-embedding → 3 × (BiLSTM + FFN) tail → a 90-column linear head, one column per (lab, action).
Inference reads the columns belonging to the lab you request.

Fine-tuning (the `LABTAIL` path in [`train_perlab_heads.py`](../train_perlab_heads.py)) warm-starts
every layer from the canonical checkpoint, freezes the SSL backbone and the merge, and trains the
subset `--mode` selects. The loss is masked to the head columns you named, so the gradient is not
diluted across the other 89. `finetune.py` is a wrapper that sets that environment up, runs the
five configs, and collects the checkpoints — the research script underneath has many more levers
(cold-start, donor-lab seeding, leave-one-lab-out foundations, merge fine-tuning, cached-feature
fast paths); read its header comments if you need them.

## Troubleshooting

| symptom | likely cause |
|---|---|
| `no classifier head for <action>` | the adopted lab never annotated it — `--list-heads`, adopt a different lab, or run one fine-tune per lab |
| `annotated file(s) not found` | `file` in the CSV must match the tracking filename (extension optional) |
| `annotated agent/target not tracked` | mouse ids in the CSV must exist in that video's `mouse_id` column |
| `stop_frame <= start_frame` | your CSV is inclusive-ended — pass `--stop-inclusive` |
| val-F1 flat at ~0 | the behaviour you are training is not among the behaviours your videos annotate, or has almost no positive frames |
| worse than zero-shot after fine-tuning | thresholds not re-calibrated (§4); or `tail` mode on very little data — try `--mode head` |
| trained fine, predictions look wrong for *other* behaviours | a `tail`-mode checkpoint only speaks for the adopted lab; restrict with `--labs`, or use `--mode head` |
| stale results after re-annotating | the scratch cache is keyed by video id; `train` clears it by default — do not pass `--reuse-cache` after re-running `prepare` |
