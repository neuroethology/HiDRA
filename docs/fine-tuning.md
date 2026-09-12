# Fine-tuning

Fine-tuning takes one lab's trained classifier head and continues training it on **your** videos
and **your** annotations. The self-supervised backbone and the shared feature merge stay frozen;
only the per-lab parts are updated. That is what makes this cheap — you are adapting a classifier
that already knows mouse social behaviour, not training one from scratch — and it is why a few
annotated videos can be enough.

Do this when zero-shot output ([zero-shot.md](zero-shot.md)) is *close but consistently off*: the
model finds the behaviour but calls it too eagerly, cuts bouts short, or misses the variant your
arena produces. If it is only the decision threshold that is wrong, go straight to
[§4 Calibrate](#4-calibrate-the-thresholds) — no training needed. If the behaviour you want has no
head at all, read [new-behaviours.md](new-behaviours.md) first.

`python finetune.py` (from a checkout) and `hidra-finetune` (from an install) are the same program;
the examples use the checkout form. The loop is:

```
annotate  →  prepare  →  train  →  predict --weights  →  calibrate  →  predict --weights --thresholds
```

## Limits (read this before you plan an experiment)

- **You adopt an existing head.** The per-lab head has 82 columns, one per published
  (lab, behaviour) pair — `python predict.py --list-heads` — and `finetune.py` can only train
  those. The behaviour vocabulary (37 names plus the merged `sniffall`) and the 21-row lab
  embedding are fixed, so there is no new behaviour name and no new lab to be had here; what *is*
  possible, and what it costs, is in [new-behaviours.md](new-behaviours.md).
- **Everything downstream refers to your data by the adopted lab's name** — `--labs`, thresholds,
  the `lab` column of the outputs. Pick the lab whose zero-shot calls were closest.
- **One behaviour per (subject, target) per frame.** Labels are stored as one action id per pair
  per frame, so overlapping bouts of different behaviours on the same pair overwrite each other.
- **Annotate at least two mice.** The trainer forms an (agent, target) pair for every annotated
  agent, and for an agent with no cross-directed labels it picks a target at random from the
  *other* mice present. A video that annotates only one mouse leaves nothing to pick from and the
  run dies inside numpy (`a cannot be empty unless no samples are taken`).
- **Un-annotated frames are negatives.** For each (agent, target, action) a video annotates, every
  frame outside a bout is a negative example for that combination. Annotate each behaviour
  exhaustively within a video, and leave out videos you only skimmed. Combinations a video does
  *not* annotate are ignored entirely, not treated as absent.
- **Label sniffing by the head's name.** Five labs — CautiousGiraffe, GroovyShrew,
  InvincibleJellyfish, NiftyGoldfinch, TranquilPanther — have the merged **`sniffall`** head and
  no plain `sniff`; the other sniff labs have plain **`sniff`**. Label bouts with whichever name
  `--list-heads` shows for your lab. The model has no `sniffall` *label* of its own: the trainer
  derives it as the union of the sniff-family labels (`sniff, sniffface, sniffbody, sniffgenital,
  reciprocalsniff`), so `prepare` stages a `sniffall` row as plain `sniff` and tells you so. (A
  row that reached the trainer literally named `sniffall` would carry zero loss weight — the
  symptom is `nll_perlab: nan` on every logged step.)
- **The ensemble is five configs.** Inference averages the 11/15/19/23/27-fps models and loads a
  checkpoint for each, so a fine-tune is only usable once all five are trained. While iterating
  you can train one (`--configs 15fps_5bp`) and predict with just that one
  (`predict.py --configs 15fps_5bp`) — faster, lower quality, not a result.

## 0. Set up

Weights (`python download_models.py`), an environment from `uv sync --extra torch`, and one GPU.
Training writes its scratch cache to `/dev/shm`, so this is a Linux-with-a-GPU workflow — unlike
inference, it is not something to try on a laptop.

The default backend is PyTorch and needs no JAX. `train --backend jax` runs the original trainer
instead (`uv sync --extra jax --extra torch`); both write the checkpoint in the same layout, and
`predict.py --weights` loads it on either backend.

## 1. Annotate

One CSV of bouts across all your videos:

```csv
file,agent,target,action,start_frame,stop_frame
mouseA_day1.parquet,mouse1,self,rear,100,160
mouseA_day1.parquet,mouse1,mouse2,sniffall,412,455
mouseB_day1.parquet,mouse2,self,rear,88,140
```

- `file` — the tracking parquet's filename (or its stem).
- `agent` / `target` — `mouse1`…`mouse4`, or `self` for a self-directed behaviour
  (`rear, dig, climb, selfgroom, rest, run, freeze, huddle, biteobject, exploreobject`).
- `action` — a behaviour the lab you adopt has a head for, spelled as `--list-heads` spells it
  (so `sniffall` for the five splitting labs, `sniff` for the others).
- `start_frame`/`stop_frame` — **`stop_frame` is exclusive** (frames `start … stop-1` are
  positive), matching the trainer. If your `stop_frame` is the last positive frame — which is
  what `predict.py`'s `bouts.csv` writes — pass `--stop-inclusive` and it is converted for you.

How much is enough? There is no single answer — it depends on how far your setup is from the
adopted lab's. Measure it rather than guess: hold out at least one recording, then run
[`calibrate`](#4-calibrate-the-thresholds) twice, once on your zero-shot predictions for the
held-out videos and once on the fine-tuned ones, and compare the F1 columns. Two or three annotated
recordings per behaviour is a reasonable place to start. Annotate the held-out recording in the
same CSV; `train --videos` (§3) decides which ones are trained on.

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
the video length, and every annotated action against the adopted lab's heads (when the lab you
picked lacks a behaviour it tells you which labs *do* have it, which is the fastest way to choose
a slot). Videos with no annotations are skipped, and a `metadata.csv` in the tracking folder
overrides `--pix-per-cm`/`--fps` per file, exactly as for `predict.py`.

Video ids are derived from the filename with the same hash `predict.py` uses, so a given recording
has one id across prediction and training.

It prints the positive-frame percentage per video — worth a look, because a behaviour that
occupies well under 1% of frames gives the head very few positive examples to learn from and makes
the threshold sweep in §4 noisy. Annotate more of it before drawing conclusions. It ends with the
head columns your labels can train and the `train` command to run next:

```
  labels staged: ['rear', 'sniff']
  head columns these can train: ['rear', 'sniffall']
next: python finetune.py train --data ft_data/ --lab GroovyShrew --actions rear,sniffall --out ft_models/ --tag mytag
```

## 3. Train

```bash
python finetune.py train \
    --data ft_data/ --lab GroovyShrew --actions rear,sniffall \
    --videos mouseA_day1,mouseB_day1 \
    --out ft_models/ --tag myrig
```

One run per config, warm-started from the published checkpoint, writing
`ft_models/{config}__myrig.pkl`. Supervision is masked to the head columns you name in
`--actions` (default: every column your staged labels can train), so nothing else gets gradient.
`--videos` (filenames, stems or ids) restricts the training set — here the third recording stays
out for §4. Re-running skips configs whose checkpoint exists unless you pass `--overwrite`.

**`--mode` decides what is trainable — the main choice you have to make:**

| mode | trains | other labs' heads | use when |
|---|---|---|---|
| `head` (default) | the linear head only (`out-proj-perlab`) | weights untouched — no gradient reaches them | your setup resembles the adopted lab's; you have little data; you want the cheapest, safest adaptation |
| `tail` | the per-lab LSTM/FF tail + lab embedding + head | **invalidated** — the tail is shared | your arena, pose rig or annotation style differs substantially, and you have several annotated recordings |
| `embedding` | the lab embedding + head | weights untouched | middle ground: re-place your data in lab space without retraining the shared tail |

`tail` is the strongest adaptation and the one the underlying research code was built around (it
is the "tail training" of [training.md](training.md) restricted to one lab), but the tail is shared
across labs: after training it, only the lab you adopted is meaningful in that checkpoint. `head`
cannot pull the trunk around at all, trains in a fraction of the time, and is the right first
attempt.

Either way, **treat a fine-tuned checkpoint as belonging to the lab you adopted** and always pass
`--labs <that lab>` when predicting with it. Even in `head` mode the other columns are not strictly
frozen output: training re-estimates the input-normalization statistics (which are shared across
all 82 head columns) on your data during initialization, so other labs' probability scales can
drift a little.

Other knobs: `--steps` (default 8000 per config), `--lr` (default 0.004), `--configs` to train one
config while iterating (pair it with `predict.py --configs <same>`), `--seed`, `--gpu`,
`--backend`, and `--smoke` for a short wiring check that writes no checkpoint. Run `--smoke` first
on a new dataset: it takes about a minute and catches every staging mistake.

**Time.** Measured on one RTX A6000 in `head` mode, PyTorch backend: about 0.5 s per step for the
15 fps config and 1 s per step for the 23 fps one, plus a few minutes per config for the
input-statistics pass — so the default 8000 steps is one to two hours per config, and most of a
day for the ensemble. Throughput is bound by the numpy data pipeline (eight worker processes), not
the GPU: forward-only initialization batches take as long as training steps. On a small dataset
the `head`-mode loss plateaus within a few hundred steps; try `--steps 1000` and let §4 decide
whether more helps. `tail` mode is the same speed per step but needs more of them.

Validation is carved out of your staged videos automatically (~15% by duration). With only one or
two staged videos there is nothing to hold out, so the validation set overlaps training and its F1
is not a generalization estimate — use the held-out `calibrate` run in §4 for that. The reported
val-F1 covers only the behaviours your annotations score.

## 4. Calibrate the thresholds

A fine-tuned head's probabilities are no longer on the scale the bundled thresholds were
calibrated for, so **re-calibrating is part of fine-tuning, not an optional extra.** Predict on the
held-out annotated videos, then sweep:

```bash
python predict.py /path/to/held_out --labs GroovyShrew --actions rear,sniffall \
    --out ft_preds/ --weights 'ft_models/{config}__myrig.pkl' --pix-per-cm 16 --fps 30

python finetune.py calibrate --frames ft_preds/ --annotations bouts.csv \
    --out ft_thresholds.csv --report ft_scores.csv
```

```
                 head  threshold     f1  precision  recall  pos_frames  frames  tracks  videos
    GroovyShrew__rear       0.52 0.9485      0.955  0.9422         225    2400       2       1
GroovyShrew__sniffall       0.44 0.8110      0.790  0.8330         310    2400       2       1
```

(Illustrative numbers. `tracks` counts scored (subject, target) tracks, `videos` the held-out
recordings they came from.)

The sweep pools every scored (video, subject, target) track per head, takes the best-F1 threshold
on a 0.05–0.95 grid, and merges the result into the bundled table so heads you did not calibrate
keep their published value (`--no-merge` writes only yours). Only videos that appear in both the
predictions and the CSV are scored, so the same `bouts.csv` serves training and calibration. By
default only (video, subject, target, action) combinations your CSV actually annotates are scored;
`--all-pairs` also scores un-annotated pairs in those videos as all-negative. `--min-positives`
(default 100) skips heads with too few positive frames to say anything, and an annotated action
that has no prediction track at all (a misspelled or wrong-lab name) is reported rather than
silently dropped.

Run the same command against your zero-shot predictions to get the before/after F1 that tells you
whether fine-tuning actually helped. Calibrating on the videos you trained on is optimistic — hold
data out.

## 5. Predict with the result

```bash
python predict.py /path/to/new_videos \
    --labs GroovyShrew --actions rear,sniffall \
    --weights 'ft_models/{config}__myrig.pkl' \
    --thresholds ft_thresholds.csv \
    --out results/ --pix-per-cm 16 --fps 30
```

Quote the `--weights` template so the shell leaves `{config}` alone; HiDRA fills it in with the
name of each config it runs. Everything else about the output is identical to zero-shot — same
`bouts.csv`, same `frames.parquet`, same [`hidra`](../src/hidra/__init__.py) helpers
(`hidra.predict(..., weights=..., thresholds=...)`).

The trainer writes pickles. To share a fine-tune, or to load it where nothing may unpickle,
re-container it as safetensors and point the template at that instead — the values are identical
and `--weights` accepts either suffix on either backend:

```bash
for c in 11fps_4bp 15fps_5bp 19fps_6bp 23fps_7bp 27fps_6bp; do
    hidra-convert-weights --one ft_models/${c}__myrig.pkl        # -> ft_models/${c}__myrig.safetensors
done
python predict.py ... --weights 'ft_models/{config}__myrig.safetensors' --thresholds ft_thresholds.csv
```

Keep `ft_thresholds.csv` next to the checkpoints: a fine-tuned head without its thresholds is
back to the bundled values, which no longer fit it.

Across backends the two implementations agree to about 1e-3 in probability
([pytorch-port.md](pytorch-port.md)), so a frame whose probability lies within that of the
threshold can be called by one backend and not the other. On the published weights and
thresholds no such frame occurs; on a freshly calibrated head a handful per video is normal
(three of 3,600 in the run used for this document, all within 2e-4 of the threshold). Pick one
backend for a study and stay with it.

## What is actually happening

Per config, the model is: frozen SSL forecaster → frozen feature merge → `x0` → `+0.1 ×`
lab-embedding → 3 × (BiLSTM + FFN) tail → an 82-column linear head, one column per published
(lab, action). Inference reads the columns belonging to the lab you request.

Fine-tuning (the `LABTAIL` path in
[`hidra/torch/train_perlab.py`](../src/hidra/torch/train_perlab.py), or
[`train_perlab_heads.py`](../src/hidra/train_perlab_heads.py) with `--backend jax`) warm-starts
every layer from the published checkpoint, freezes the SSL backbone and the merge, and trains the
subset `--mode` selects. The loss is masked to the head columns you named, so the gradient is not
diluted across the other 81. `finetune.py` is a wrapper that sets that environment up, runs the
five configs, and collects the checkpoints — the research script underneath has many more levers
(cold-start, donor-lab seeding, leave-one-lab-out foundations, merge fine-tuning, cached-feature
fast paths); [training.md](training.md) lists them and its header comments explain each.

## Troubleshooting

| symptom | likely cause |
|---|---|
| `no head for <action>` | the adopted lab never annotated it — `--list-heads`, adopt a different lab, or run one fine-tune per lab; see [new-behaviours.md](new-behaviours.md) |
| `no head for ['sniff']` for a sniff-splitting lab | that lab's head is `sniffall`; label the CSV `sniffall` (or pass `--actions sniffall`) |
| `['sniffall'] are not annotated … derived from the sniff-family labels` | the staged manifest carries a literal `sniffall` label (staged before this check existed) — re-run `prepare` |
| `nll_perlab: nan` on every logged step | zero loss weight: no annotated (pair, action) reaches the columns you named — check `--actions` against the staged labels |
| `annotated file(s) not found` | `file` in the CSV must match the tracking filename (extension optional) |
| `annotated agent/target not tracked` | mouse ids in the CSV must exist in that video's `mouse_id` column |
| `stop_frame <= start_frame` | your CSV is inclusive-ended — pass `--stop-inclusive` |
| val-F1 flat at ~0 | the behaviour you are training is not among the behaviours your videos annotate, or has almost no positive frames |
| worse than zero-shot after fine-tuning | thresholds not re-calibrated (§4); or `tail` mode on very little data — try `--mode head` |
| trained fine, predictions look wrong for *other* behaviours | a `tail`-mode checkpoint only speaks for the adopted lab; restrict with `--labs`, or use `--mode head` |
| stale results after re-annotating | the scratch cache is keyed by video id; `train` clears it by default — do not pass `--reuse-cache` after re-running `prepare` |
| `FileNotFoundError: …_unsupervised.pkl` with `--backend jax` | an older HiDRA that resolved the pickles by name; the JAX trainer now reads either container — upgrade |
