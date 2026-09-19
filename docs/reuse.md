# Ways to reuse HiDRA

HiDRA is a frozen self-supervised trunk, a shared feature merge, a per-lab tail, and an
82-column linear head. Reusing it on your data means choosing **how far down that stack to
push your annotations** — and the answer is usually "less far than you think". This page is
the map: every route, what it costs, what it needs, and how to choose. The step-by-step
instructions live in [zero-shot.md](zero-shot.md), [fine-tuning.md](fine-tuning.md),
[new-behaviours.md](new-behaviours.md) and [training.md](training.md).

```
pose ─▶ [ SSL trunk ]─▶[ feature merge ]─▶ x0 ─▶ + lab embedding ─▶[ 3× BiLSTM+FFN tail ]─▶ 82-col head ─▶ probs ─▶ threshold ─▶ bouts
        22.7M params      1.7M params                  256/lab          4.8M params           258/column
        └──────────── frozen in every fine-tuning mode ────────────┘   └── `tail` ──┘        └── `head` ──┘
```

## The routes, cheapest first

| # | route | you need | what moves | cost per config | where |
|---|---|---|---|---|---|
| 1 | **Zero-shot** | nothing | nothing | seconds of inference | [zero-shot.md](zero-shot.md) |
| 2 | **Pick a different lab's head** | a pilot recording to eyeball | nothing | one inference pass per lab | [zero-shot.md §2](zero-shot.md#2-pick-a-classifier) |
| 3 | **Re-threshold** | the probabilities you already wrote | nothing | none — `frames.parquet` is re-read | [zero-shot.md §5](zero-shot.md#5-thresholds) |
| 4 | **Calibrate thresholds** | a few annotated recordings | one number per head | seconds, no GPU | [fine-tuning.md §4](fine-tuning.md#4-calibrate-the-thresholds) |
| 5 | **Fine-tune `--mode head`** | ~2–3 annotated recordings | 258 numbers per behaviour | ~15 s cached, ~1 h live | [fine-tuning.md](fine-tuning.md) |
| 6 | **Add a head column** (`--new-head`) | annotations for the new behaviour | +258 numbers per column | as (5) | [new-behaviours.md §4](new-behaviours.md) |
| 7 | **Adopt a head-free slot as your lab** | as (6) | as (6), plus its embedding row | as (5) | [new-behaviours.md §4b](new-behaviours.md) |
| 8 | **Fine-tune `--mode embedding`** | several recordings | 26.5k numbers | as (5) | [fine-tuning.md §3](fine-tuning.md#3-train) |
| 9 | **Fine-tune `--mode tail`** | several recordings, ideally your own lab slot | 4.8M numbers | ~1–6 h | [fine-tuning.md §3](fine-tuning.md#3-train) |
| 10 | **Retrain stage 2** | a corpus | merge + tail + both heads | 4–10 h × 5 configs, JAX | [training.md §3](training.md#3-stage-2-the-supervised-per-lab-tail-and-heads) |
| 11 | **Retrain stage 1** | a corpus, no labels needed | the trunk | ~1 day × 5 configs, JAX | [training.md §2](training.md#2-stage-1-the-self-supervised-trunk) |

Routes 1–4 are free of training entirely, and between them they fix most of what "the calls
look wrong" turns out to mean. **Do them before reaching for a GPU.** In particular, route 4
takes a handful of labelled videos and no training at all, and is often the whole difference.

## How to choose

Work down this list and stop at the first "yes".

1. **Is the behaviour in the 37-name vocabulary, and does some lab have a head for it?**
   `python predict.py --list-heads`. If not, see "What you cannot do" below.
2. **Do the zero-shot calls find the behaviour, roughly?** Run two or three labs' versions on
   one pilot recording and look. If one of them is right but over- or under-calls, you need a
   *threshold*, not training → route 4.
3. **Are they systematically off in shape — bouts cut short, the wrong variant, a different
   annotation convention?** That is what `--mode head` fixes: the same features, read
   differently → route 5.
4. **Is the behaviour missing for the lab whose other heads you like?** Add the column →
   route 6.
5. **Is your rig far from every lab's — different arena size, different camera geometry,
   pose from a different rig?** Then the *features* reaching the head are off-distribution
   and no linear readout will rescue them. Give yourself a lab slot and retrain the tail →
   route 7 + `--mode tail`.
6. **Is the species or the recording modality different?** Now you are outside what the
   published trunk saw → routes 10–11, GPU-days.

A useful sanity check at every level: run `calibrate` on held-out annotations before and
after, and compare the F1 column. If the number did not move, the extra capacity did not
buy anything.

## The human-in-the-loop loop

Route 5 and 6 are cheap enough to sit inside an annotation session, which is the point:

```
annotate a few bouts
   ↓
finetune.py prepare              (seconds, pandas only)
   ↓
finetune.py train --cache-features --ddi-steps 0 --steps 300 --configs 15fps_5bp
   ↓                             (~15 s for one config on one A6000)
predict.py --weights ... --configs 15fps_5bp
   ↓                             (seconds)
look at the new calls, annotate where they are wrong
   ↓
repeat
```

Two things make that possible, and both are worth understanding before relying on it:

- **`--cache-features`** computes the frozen part of the network once per training window
  instead of once per step. In `head` mode the trainable part is 258 numbers per behaviour
  read off a 256-dimensional feature vector, so once those vectors exist a step costs
  milliseconds. Measured: 560 ms per step live, 31 ms cached.
- **`--ddi-steps 0`** skips the input-statistics recalibration, which is otherwise the
  dominant remaining cost (256 batches, ~2.5 min). Skipping it keeps the published
  statistics, which also leaves every *other* lab's head bit-identical in the result.

Both change what the run does, not just how long it takes — the cache fixes the set of
augmentation draws, and skipping recalibration reads your data through the consortium's
statistics. Iterate with them on, and when you have converged on what to annotate, do one
final run at the defaults (`--steps 8000`, all five configs, full recalibration) and
calibrate that on held-out data. That is the number to report.

**Publish with the full ensemble.** Everything above is one config so that a round trip is
seconds. The thresholds were calibrated on the five-config average, and a single config is a
different, worse model.

## Measured

One RTX A6000, `15fps_5bp`, batch 128, three 60-second synthetic recordings
(`python -m tests.synth`), `head` mode, two behaviours. These are *timings*; the synthetic
data says nothing about accuracy.

Per training step:

| component | ms |
|---|---|
| data pipeline, one batch (8 numpy workers) | 101 |
| frozen trunk + feature merge, forward | 274 |
| per-lab tail, forward | 180 |
| `head` mode step, live (forward + backward) | 445 |
| `head` mode step, on cached features | 1.7 |
| `tail` mode step, live | 940 |
| `tail` mode step, on cached `x0` | 672 |

End to end, 300 steps of one config:

| variant | input statistics | feature pass | 300 steps | wall clock |
|---|---|---|---|---|
| live (the default before this) | 142 s | — | 168 s | 5 min 16 s |
| `--cache-features` | 191 s * | 3 s | 9 s | 3 min 27 s |
| `--cache-features --ddi-steps 32` | 28 s | 3 s | 7 s | 42 s |
| `--cache-features --ddi-steps 0` | — | 3 s | 7 s | **14 s** |

\* that run shared the machine with a second training job; the identical pass took 142 s run
alone.

The final training loss was 0.512, 0.511, 0.506 and 0.515 respectively — the same
optimisation, not a cheaper approximation of it.

Extrapolated to the shipped defaults (8000 steps, five configs), `head` mode:

| | one config | five configs |
|---|---|---|
| live | ~77 min | ~6–12 h |
| `--cache-features` | ~7 min | ~35 min |
| `--cache-features --ddi-steps 0` | ~4 min | ~21 min |

`tail` mode is a different story: its 4.8M-parameter BiLSTM stack runs every step whatever
you cache, so caching `x0` saves about 30% and no more. Reach for it when a linear readout
genuinely is not enough, and budget hours.

## What you cannot do

- **A behaviour name outside the 37-word vocabulary.** The label space is baked into every
  checkpoint. The practical substitute is to repurpose a column you do not otherwise need —
  [new-behaviours.md §5](new-behaviours.md#5-the-name-is-not-in-the-vocabulary-at-all).
- **A 22nd lab.** The embedding has 21 rows. Five of them carry no published head and can be
  adopted as yours (route 7); beyond that it is a stage-2 retrain.
- **Fine-tune the shared feature merge.** It maps trunk features into the 256-d stream every
  lab's head reads, so moving it on one user's videos invalidates all 15 labs at once. The
  research code has the lever (`LABTAIL_TUNE_MERGE`, JAX only,
  [training.md](training.md#research-levers)); `finetune.py` deliberately does not expose it.
- **Train on the CPU in any reasonable time**, or on Windows: the trainer memory-maps its
  window cache through `/dev/shm`. Inference is fine anywhere.
