# Zero-shot inference

**Zero-shot** here means: you have pose tracking and no behaviour labels, and you want an
ethogram anyway. You pick one of the 82 trained classifier heads — a (lab, behaviour) pair — and
apply it to your videos as-is. Nothing is trained, nothing is fit to your data; the model has
never seen your arena, your camera, or your annotator.

This is what [`predict.py`](../predict.py) does. If zero-shot output is close but systematically
off (your annotators call shorter bouts, your arena is a different size), fine-tune the head you
picked instead: [fine-tuning.md](fine-tuning.md).

## 1. What you need

- Pose parquets in long format (`video_frame, mouse_id, bodypart, x, y`), bodypart names from the
  model schema — the 7 the model uses are `nose, ear_left, ear_right, neck, body_center,
  tail_base, tail_tip`. See [the README](../README.md#input) for the full name list and the rules
  on extra/missing/unrecognized parts.
- `pix_per_cm` and `fps` per recording. **A missing pixel scale silently zeroes every
  prediction**, so HiDRA refuses to run without it. Either a `metadata.csv` in the folder or
  `--pix-per-cm N --fps N` on the command line.
- The weights: `python download_models.py` (~660 MB from
  [talmolab/HiDRA](https://huggingface.co/talmolab/HiDRA)).
- A GPU. CPU works and is much slower.

## 2. Pick a classifier

Each head is one lab's definition of one behaviour. The 15 labs annotated overlapping but
genuinely different behaviour sets, with different conventions — that is the whole point of
keeping them separate rather than averaging them into one "attack" classifier.

```bash
python predict.py --list-heads
```

```
82 classifier heads across 15 labs:

  AdaptableSnail        approach attack avoid chase chaseattack rear submit
  BoisterousParrot      shepherd
  CautiousGiraffe       chase escape reciprocalsniff sniffall sniffbody sniffgenital
  ...
```

Three things to know when choosing:

- **`sniff` vs `sniffall`.** Five labs (CautiousGiraffe, GroovyShrew, InvincibleJellyfish,
  NiftyGoldfinch, TranquilPanther) split sniffing into subtypes, so their sniff-family head is
  the merged **`sniffall`** (sniff ∪ sniffface ∪ sniffbody ∪ sniffgenital ∪ reciprocalsniff). The
  other labs use plain **`sniff`**. Asking for `sniff` from a splitting lab is an error that
  tells you which labs do have it.
- **Self-directed vs social.** `selfgroom, dig, climb, rear, rest, run, biteobject,
  exploreobject, freeze, huddle` are scored on `(subject == target)` — they come out with
  `target == self`. Everything else is scored per ordered pair.
- **Several labs usually have "your" behaviour.** They will not agree. Running all of them on a
  pilot recording and comparing is the cheapest way to find the one whose calls match what you
  would have annotated.

Only `PleasantMeerkat`'s `follow` head is published — its attack/chase/escape annotations were
single-bin artifacts and those heads were dropped.

## 3. Run it

The default with no selection is **every head × every behaviour × every mouse pair**, which is
15 forward passes of the 5-config ensemble — thorough, and slow. Three ways to narrow it:

```bash
# one lab, two behaviours (the quick path)
python predict.py tracking/ --out results/ --pix-per-cm 16 --fps 30 \
    --labs GroovyShrew --actions rear,sniffall

# several labs' version of the same behaviour, to compare them
python predict.py tracking/ --out results/ --pix-per-cm 16 --fps 30 \
    --labs AdaptableSnail,LyricalHare,NiftyGoldfinch --actions attack

# one directed pair only
python predict.py tracking/ --out results/ --pix-per-cm 16 --fps 30 \
    --labs LyricalHare --actions attack --subject mouse1 --target mouse2
```

For per-head control over pairs, use the job sheet: `--dump-jobs jobs.csv` writes one row per
head (`run,lab,action,subject,target`), you delete or zero rows, and pass it back with
`--jobs jobs.csv`. Only labs with an enabled row are run, so the sheet is also the speed dial.

Runtime is dominated by the number of **labs**, not behaviours: each lab is its own trunk
embedding, so one full 5-config pass per lab (a few minutes per folder per lab on a modern GPU).
Asking one lab for ten behaviours costs the same as asking it for one.

## 4. Read the output

Per input parquet, in `--out`:

- **`<stem>.bouts.csv`** — the ethogram: `subject,target,lab,action,start_frame,stop_frame,
  n_frames,mean_prob,threshold`. `stop_frame` is the **last** frame of the bout (inclusive).
- **`<stem>.frames.parquet`** — per-frame `prob` and binary `call` for every kept
  (subject, target, lab, action).

`--output calls` or `--output probs` writes just one of them.

```python
import hidra
bouts = hidra.bouts("results/")                       # all videos, with a `video` column
frames = hidra.frames("results/")                     # per-frame probs
rear = hidra.ethogram("results/", action="rear")      # filtered + sorted
```

## 5. Thresholds

A call is `prob >= threshold`. The bundled thresholds in
[`derived_thresholds_train.csv`](../src/hidra/assets/derived_thresholds_train.csv) are per-(lab, action) and were
calibrated on training data (leakage-free), falling back to a pooled per-action value and then to
0.30 for anything missing. They are a reasonable default on the labs' own footage — they are not
tuned to yours.

Two adjustments, cheapest first:

```bash
# sweep one global threshold by hand: probs are written regardless of the threshold, so you can
# re-threshold offline from frames.parquet without re-running the model
python predict.py tracking/ --out results/ --labs GroovyShrew --actions rear --threshold 0.5 ...

# or calibrate against annotations you do have, on as little as a few labelled videos
python finetune.py calibrate --frames results/ --annotations my_bouts.csv --out my_thresholds.csv
python predict.py tracking/ --thresholds my_thresholds.csv ...
```

`calibrate` picks the best-F1 threshold per head and merges it into the bundled table, so heads
you did not calibrate keep their published value. It needs no GPU and no training — this is
often the whole difference between "the calls look wrong" and "the calls look right", and it is
worth doing before concluding you need to fine-tune. See
[fine-tuning.md § Calibrate](fine-tuning.md#4-calibrate-the-thresholds) for the annotation CSV
format.

## 6. When zero-shot is the wrong tool

- **Your behaviour is not in the vocabulary.** The action names are fixed: 33 of the 37 have a
  trained head, plus the merged `sniffall` (`none`, `dominancemount`, `disengage` and
  `genitalgroom` have none). There is no head to borrow for a behaviour nobody in the consortium
  annotated, and fine-tuning cannot add one either (see
  [fine-tuning.md § Limits](fine-tuning.md#limits-read-this-before-you-plan-an-experiment)).
- **Pose is poor.** The model reads keypoints, not pixels. Swapped identities and dropped
  keypoints propagate straight into the calls.
- **The pixel scale is wrong.** Everything is computed in cm. A `pix_per_cm` that is off by 2×
  makes the model see mice of the wrong size, and the calls degrade quietly rather than failing.
- **The other labs' thresholds are the problem.** Try §5 before anything heavier.
