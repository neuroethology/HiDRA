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

Model weights for [HiDRA](https://github.com/talmolab/HiDRA), a per-lab-head mouse social
behaviour classifier ensemble. Given pose tracking for two or more mice, it produces
per-frame probabilities for **82 (lab, behaviour) classifier heads** across **15 labs** —
each head being one lab's own definition of one behaviour, kept separate rather than
averaged into a single consensus classifier.

## Usage

```bash
pip install 'hidra[torch] @ git+https://github.com/talmolab/HiDRA'
hidra-download-models                        # fetches this repo into models/
hidra-predict --list-heads                   # the 82 (lab, behaviour) heads

hidra-predict tracking/ --out results/ --pix-per-cm 16 --fps 30 \
    --labs GroovyShrew --actions rear,sniffall
```

Input is a folder of long-format pose parquets (`video_frame, mouse_id, bodypart, x, y`)
plus a pixel scale and frame rate per recording. Output is a per-frame probability track and
a thresholded ethogram. See the [README](https://github.com/talmolab/HiDRA) for the input
schema and [docs/zero-shot.md](https://github.com/talmolab/HiDRA/blob/main/docs/zero-shot.md)
for choosing a head.

## Files

The ensemble is **five configurations** — `11fps_4bp`, `15fps_5bp`, `19fps_6bp`,
`23fps_7bp`, `27fps_6bp` — differing in temporal resolution and how many keypoints they
consume. Inference averages all five, so all five checkpoints are needed.

| file | count | what it is |
|---|---|---|
| `{config}_unsupervised.safetensors` | 5 | self-supervised trunk (frozen forecaster) |
| `{config}_supervised_perlab_sniffall.safetensors` | 5 | per-lab classifier head, 82 columns |
| `thresholds.json` | 1 | per-(lab, action) decision thresholds |

Each config's trunk and head are loaded together; the head consumes the trunk's features.

### Two containers, identical weights

`.safetensors` is the default and what `hidra-download-models` fetches. It executes no code
on load and is read with numpy alone — neither PyTorch nor JAX is required to open it.

`.pkl` is the original format: a pickle of JAX device arrays, produced by the training code.
It is kept here so installs predating the conversion keep working, and is still available
via `hidra-download-models --format pkl`.

The two hold the same float32 values, verified byte-for-byte. Running the JAX backend from
safetensors reproduces a pickle-based run **bit-identically**.

## Backends

The models were written in JAX and have been ported to PyTorch, which is now the default.
The two agree to float32 round-off: across a full five-config ensemble run, every
thresholded call and every bout boundary is identical, with a worst per-frame probability
difference of 7.3e-4.

One thing worth knowing if you compare backends yourself: on Ampere and later, JAX computes
float32 matmuls in **TF32** by default, so the original path carries ~1e-3 relative error
while the PyTorch path runs true float32 at ~1e-7. Measured against a float64 reference, the
PyTorch backend is the more accurate of the two. The full analysis is in
[docs/pytorch-port.md](https://github.com/talmolab/HiDRA/blob/main/docs/pytorch-port.md).

## Fine-tuning

You can adapt one lab's head to your own arena, pose rig and annotation style, then
re-calibrate its threshold. The trunk and the shared feature merge stay frozen. You
**adopt** an existing (lab, behaviour) head — new behaviour names and new labs are not
possible through fine-tuning, since the head layer and lab-embedding table are fixed. See
[docs/fine-tuning.md](https://github.com/talmolab/HiDRA/blob/main/docs/fine-tuning.md), and
[docs/new-behaviours.md](https://github.com/talmolab/HiDRA/blob/main/docs/new-behaviours.md)
for what to do when no head matches your behaviour.

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
