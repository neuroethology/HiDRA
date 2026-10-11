---
license: bsd-3-clause
library_name: hidra
tags:
  - animal-behavior
  - pose-estimation
  - neuroscience
  - mouse
  - ethology
  - safetensors
pipeline_tag: video-classification
---

# HiDRA — the High-Dimensional Rodent Annotator

Model weights for [HiDRA](https://github.com/neuroethology/HiDRA), a per-lab-head mouse social
behaviour classifier ensemble. Given pose tracking for two or more mice, it produces
per-frame probabilities for **82 (lab, behaviour) classifier heads** across **15 labs** —
each head being one lab's own definition of one behaviour, kept separate rather than
averaged into a single consensus classifier.

**This is the October 2026 weight set**: retrained self-supervised trunks and a per-lab
classifier trained *without* the shared cross-lab head the previous model carried. The 82
heads and their names are unchanged; the decision thresholds are re-calibrated for this model.
See [Changes from the previous release](#changes-from-the-previous-release) below, and
[CHANGELOG.md](https://github.com/neuroethology/HiDRA/blob/main/CHANGELOG.md) in the repository.

## Usage

```bash
pip install 'hidra[torch] @ git+https://github.com/neuroethology/HiDRA'
hidra-download-models                        # fetches this repo into models/
hidra-predict --list-heads                   # the 82 (lab, behaviour) heads

hidra-predict tracking/ --out results/ --pix-per-cm 16 --fps 30 \
    --labs GroovyShrew --actions rear,sniffall
```

Input is a folder of long-format pose parquets (`video_frame, mouse_id, bodypart, x, y`)
plus a pixel scale and frame rate per recording. Output is a per-frame probability track and
a thresholded ethogram. See the [README](https://github.com/neuroethology/HiDRA) for the input
schema and [docs/zero-shot.md](https://github.com/neuroethology/HiDRA/blob/main/docs/zero-shot.md)
for choosing a head.

## Files

The ensemble is **five configurations** — `11fps_4bp`, `15fps_5bp`, `19fps_6bp`,
`23fps_7bp`, `27fps_6bp` — differing in temporal resolution and how many keypoints they
consume. Inference averages all five, so all five checkpoints are needed.

| file | count | what it is |
|---|---|---|
| `{config}_unsupervised.safetensors` | 5 | self-supervised trunk (frozen forecaster), 421 tensors |
| `{config}_supervised_perlab_sniffall.safetensors` | 5 | per-lab classifier: feature merge, lab embedding, 3 × (BiLSTM + FFN) tail, one 82-column linear head; 177 tensors, no shared head |
| `thresholds.json` | 1 | the head table: which (lab, action) pairs have a column, in which order (its threshold values are not used; the package bundles the calibrated ones) |

Each config's trunk and head are loaded together; the head consumes the trunk's features.
`{config}` is one of `11fps_4bp`, `15fps_5bp`, `19fps_6bp`, `23fps_7bp`, `27fps_6bp`.

### Model

Per config: frozen self-supervised trunk (a keypoint-trajectory forecaster, `d_res` 192, 4
message-passing + LSTM layers) → feature merge → `+ 0.1 ×` lab embedding → 3 × (BiLSTM + FFN),
`d_res` 256 → one `Linear(256 → 82)` per-lab head. Inference reads the head columns that belong
to the lab you request and averages the five configs. There is no shared cross-lab head.

### Training

- **Stage 1, trunks**: `hidra.solution.pretrain`, 125 000 steps of keypoint forecasting, no
  labels, on 9 008 videos from 21 labs (all DooM videos of the 15 competition labs, CalMS21,
  CRIM13, MABe22). Trained 2026-10-06/07.
- **Stage 2, classifier**: `hidra.train_perlab_heads` (default settings, `SNIFFALL=1`), trunk
  frozen, on the 368 annotated DooM *training* videos of the 15 labs (plus CalMS21 and CRIM13 in
  the tree, which contribute nothing without a shared head); 15 % held out for validation. Adam
  0.004, batch 128, EMA 0.9993, early stopping on validation F1, max 50 000 steps; the saved
  weights are the EMA at the best validation step. Trained 2026-10-07/08.
- **Thresholds**: one per head, the best-F1 threshold of the five-config ensemble's output on
  the DooM training videos (bundled in the package as `derived_thresholds_train.csv`).

### Evaluation

Held-out DooM test videos (215), mean frame-level F1 over the 82 heads with the train-calibrated
thresholds: **0.613**. The same protocol on the two alternatives trained on the same trunks:
0.618 with the shared head and the external labs (the previous architecture), 0.616 with the
shared head but without CalMS21. Per-head numbers are in the repository's CHANGELOG.

### Container

`.safetensors` is what `hidra-download-models` fetches. It executes no code on load and is read
with numpy alone — neither PyTorch nor JAX is required to open it. The values are the trainer's
float32 output, verified byte-for-byte on conversion. Pickled JAX checkpoints of earlier releases
remain available at pinned revisions (`hidra-download-models --format pkl --revision <sha>`).

## Backends

The models were written in JAX and have been ported to PyTorch, which is now the default.
The two agree to float32 round-off. On the previous weight set, measured on a GPU, every
thresholded call and every bout boundary of a full five-config ensemble run was identical,
with a worst per-frame probability difference of 7.3e-4; on this set the per-lab logits of
the two backends agree to ~1e-6 relative (CPU, 15fps_5bp and 11fps_4bp), and the GPU end-to-end
comparison is pending.

One thing worth knowing if you compare backends yourself: on Ampere and later, JAX computes
float32 matmuls in **TF32** by default, so the original path carries ~1e-3 relative error
while the PyTorch path runs true float32 at ~1e-7. Measured against a float64 reference, the
PyTorch backend is the more accurate of the two. The full analysis is in
[docs/pytorch-port.md](https://github.com/neuroethology/HiDRA/blob/main/docs/pytorch-port.md).

## Fine-tuning

You can adapt one lab's head to your own arena, pose rig and annotation style, then
re-calibrate its threshold. The trunk and the shared feature merge stay frozen. You
**adopt** an existing (lab, behaviour) head, or add a column for a vocabulary behaviour the lab
has none for (`finetune.py train --new-head`), including under one of the lab slots that has
no published head — new behaviour *names* are not possible without a retrain. See
[docs/fine-tuning.md](https://github.com/neuroethology/HiDRA/blob/main/docs/fine-tuning.md), and
[docs/new-behaviours.md](https://github.com/neuroethology/HiDRA/blob/main/docs/new-behaviours.md)
for what to do when no head matches your behaviour. Two things specific to this weight set:
head-only fine-tuning keeps the checkpoint's input-normalisation statistics by default (the
previous behaviour, re-estimating them on your videos, pinned head-only fine-tunes of a new lab
below the zero-shot head it started from), and the embedding rows of the head-free lab slots are
untrained, so adopting one wants `--seed-from <DonorLab>`.

## Changes from the previous release

The previous weights (up to 2026-09) were a **multi-task** model: next to the 82-column per-lab
head, a shared 38-way head was trained on every video of every lab, including the six
"train-only" labs (CRIM13, CalMS21, MABe22) that have no scored head. A three-way comparison on
the retrained trunks (held-out DooM test, 2026-10-08) found that the shared head and the external
labs buy nothing measurable:

| model | mean held-out F1 (82 heads) |
|---|---|
| shared head + external labs (the previous architecture) | 0.618 |
| shared head, no CalMS21 | 0.616 |
| **no shared head (this release)** | **0.613** |

so the simpler model is now canonical. What that means for you:

- **All 11 weight files changed**: the trunks are retrained, the classifier is new. The 82 heads,
  their names and their column order are unchanged.
- **The decision thresholds changed** (bundled with the package; `hidra-predict` uses them
  automatically). Do not use the previous release's thresholds CSV with these weights or vice
  versa.
- **Probabilities are not comparable across releases**: a study should stay on one set. To keep
  the previous weights, pin their revision:

  ```bash
  hidra-download-models --revision 6b9770c99f9e6e20a89f7c2b68169a72b14631e5   # the last pre-2026-10 upload (safetensors)
  hidra-download-models --format pkl --revision 4147d4fce1c46f7d56fdbf5dcc3e1e3b2888ba00   # their pickles
  ```

  and pair them with that release's thresholds (`--thresholds`). This set is revision
  `<sha of the 2026-10 upload>`  <!-- TODO: fill in at publish time -->.
- **Checkpoints no longer carry a shared head** (`out-proj`). HiDRA 0.2.0 and later build the
  model without it and still read older checkpoints that have one. HiDRA versions older than
  0.2.0 cannot load these safetensors (they expect that subtree): upgrade, or pin the revision
  above.

## Limitations

- **A pixel scale is mandatory.** A missing `pix_per_cm` silently zeroes every prediction,
  so the tool refuses to run without it.
- **Bodypart names must come from the model's schema.** It recognizes 29 names and uses 7:
  `tail_base, ear_right, ear_left, nose, neck, body_center, tail_tip`. An unrecognized name
  is a hard error; a missing one of the 7 is treated as unobserved.
- **Heads encode a lab's annotation conventions**, not a universal definition. Two labs'
  `attack` heads genuinely disagree; that separation is the point. Pick the head whose
  convention matches yours, and re-calibrate the threshold on your own annotations if the
  calls are systematically off.
- **The bundled thresholds were calibrated on all five configs.** Running a subset is
  faster but lower quality, and the thresholds no longer apply.
- `PleasantMeerkat` contributes only its `follow` head; its attack/chase/escape annotations
  are single-bin scoring artifacts rather than real bout structure.
