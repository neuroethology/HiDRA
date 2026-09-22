# HiDRA — the High-Dimensional Rodent Annotator

Apply the trained **per-lab-head behaviour-classifier ensemble** (5-config: 11fps_4bp, 15fps_5bp,
19fps_6bp, 23fps_7bp, 27fps_6bp) to your own pose-tracking parquets. By default HiDRA runs **every
lab's classifiers, for every behaviour, on every mouse pair**; you can narrow this with `--labs` /
`--actions` or a job sheet.

Four things you can do with it, cheapest first:

- **[Zero-shot inference](docs/zero-shot.md)** — no labels of your own. Pick one of the 82 trained
  (lab, behaviour) classifier heads and run it on your videos: `predict.py`.
- **[Fine-tuning](docs/fine-tuning.md)** — you have some annotations. Adapt the head you picked to
  your arena, pose rig and annotation style, then re-calibrate its threshold: `finetune.py`.
  `--cache-features` makes a single-config round trip take seconds rather than an hour, which is
  what makes an annotate-train-look loop practical.
- **[Adding a behaviour](docs/new-behaviours.md)** — the behaviour you want has no head, or not
  under the lab you want. Adopt another lab's head, give your lab a new head column with
  `finetune.py train --new-head Lab,action`, or claim one of the head-free lab slots and make it
  your own lab.
- **[Training from scratch](docs/training.md)** — rebuild the self-supervised trunk and/or the
  supervised per-lab tail with the research code that produced the published weights.

**[docs/reuse.md](docs/reuse.md)** puts all of these side by side — what each one costs, how much
annotation it needs, and how to pick.

There is also a small Python API in [`src/hidra/__init__.py`](src/hidra/__init__.py)
for notebook use: `import hidra`.

## Setup

Python 3.12+ and an NVIDIA GPU (driver ≥ 525). HiDRA is a normal Python package; install it
with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra torch          # in a checkout: creates .venv
```

or into an existing environment:

```bash
pip install 'hidra[torch] @ git+https://github.com/neuroethology/HiDRA'
```

That gives you the `hidra-predict`, `hidra-finetune`, `hidra-download-models` and
`hidra-convert-weights` commands, an importable `hidra` package, and the root-level `predict.py` /
`finetune.py` / `download_models.py` scripts if you prefer to run them in place (each script and
its `hidra-*` command are the same program). The torch wheels ship their own CUDA runtime, so no
system CUDA toolkit is needed. CPU-only works but is much slower.

### Backends

The models were originally written in JAX and have been ported to PyTorch, which is now the
default. The two agree to float32 round-off — every thresholded call and bout boundary is
identical on the published weights — and the PyTorch path is about 8× faster and uses less than
half the memory. Full numbers, and how the equivalence was established:
**[docs/pytorch-port.md](docs/pytorch-port.md)**.

```bash
python predict.py tracking/ --out results/ --pix-per-cm 16 --fps 30                  # torch
python predict.py tracking/ --out results/ --pix-per-cm 16 --fps 30 --backend jax    # original
```

`--backend jax` needs the original backend too — `uv sync --extra jax --extra torch`, which is
also what the equivalence tests and [training from scratch](docs/training.md) require. Inference
and fine-tuning on the default PyTorch path need no JAX at all.

### Model weights

The weights are **not in this repo** — at ~660 MB (individual files up to 156 MB) they exceed
GitHub's file-size limit. They are hosted on the Hugging Face Hub instead, at
**[Neuroethology/HiDRA](https://huggingface.co/Neuroethology/HiDRA)** (public, no login needed).
Download them once into `models/` (needs ~700 MB free):

```bash
python download_models.py
```

The download resumes and skips files already present, so it is safe to re-run.

```bash
python download_models.py --check                  # report what's missing, download nothing
python download_models.py --repo ORG/NAME          # pull from a different Hub repo
python download_models.py --revision v1.0          # pin a branch, tag, or commit
```

That fetches 11 files as **safetensors**: 5 trunk/backbone embeddings, 5 per-lab classifier
heads, and `thresholds.json`. safetensors executes no code on load and is read with numpy
alone, so opening the weights needs neither JAX nor PyTorch.

`--repo` / `--revision` also read from `$HIDRA_HF_REPO` / `$HIDRA_HF_REVISION`. `predict.py`
refuses to start with weights missing and points you back here. Set `$HIDRA_MODELS_DIR` to
keep the weights somewhere other than `models/`.

The original pickled JAX checkpoints hold the identical values and are still readable, but
are no longer on the Hub's `main`; see [models/README.md](models/README.md) to pin them, to
convert your own fine-tuned checkpoints (`hidra-convert-weights`), or to publish a new set
(`hidra-publish-weights`).

## Input

A folder of tracking parquets (`.parquet` or `.pkt`), **long format**, one row per (frame, mouse,
bodypart):

| video_frame | mouse_id | bodypart | x | y |
|---|---|---|---|---|
| 0 | mouse1 | nose | 512.3 | 288.1 |

- `mouse_id` values: `mouse1`, `mouse2`, … (up to `mouse4`).
- **Bodypart names must be *recognized* by the model.** It knows 29 names but actually **uses 7**:
  `tail_base, ear_right, ear_left, nose, neck, body_center, tail_tip`. So:
    - a recognized name that isn't one of those 7 (e.g. `hip_left`, `forepaw_left`) is **loaded but
      ignored** — no error;
    - a **missing** one of the 7 is treated as unobserved (that config just uses fewer keypoints,
      still runs);
    - an **unrecognized** name (not one of the 29) is a **hard error** — rename it to a schema name
      first. The other 22 recognized names are: `head, spine_1, spine_2, hip_left, hip_right,
      lateral_left, lateral_right, forepaw_left, forepaw_right, hindpaw_left, hindpaw_right,
      tail_midpoint, tail_middle_1, tail_middle_2`, and 8 `headpiece_*` markers.

## Metadata (required)

Every recording needs a **pixel scale** (`pix_per_cm`) and **frame rate** (`fps`). A missing pixel
scale silently makes *all* predictions zero, so the tool refuses to run without it. Provide either:

- a `metadata.csv` in the folder:
  ```
  file,pix_per_cm,fps
  *,16.0,30.0                 # a '*' row = default for ALL files
  mouseA_day1.parquet,18.3,30 # per-file rows override the default
  ```
- or one value for all files on the command line: `--pix-per-cm 16.0 --fps 30`.

## Usage

```bash
# activate the environment created in Setup (uv sync writes .venv/)
source .venv/bin/activate

# What can I run? 82 (lab, behaviour) heads across 15 labs:
python predict.py --list-heads

# The quick path: one lab, a couple of behaviours
python predict.py /path/to/folder --out results/ --pix-per-cm 16 --fps 30 \
    --labs GroovyShrew --actions rear,sniffall

# Default with no selection: ALL classifiers x ALL actions x ALL mouse pairs on every parquet
python predict.py /path/to/folder --out results/ --pix-per-cm 16 --fps 30

# Per-pair control — generate an editable job sheet, edit, pass back:
python predict.py /path/to/folder --dump-jobs jobs.csv      # writes all 82 heads
#   edit jobs.csv in a spreadsheet, then:
python predict.py /path/to/folder --jobs jobs.csv --out results/
```

From Python:

```python
import hidra
hidra.heads(action="attack")                      # which labs have an attack classifier
hidra.predict("tracking/", out="results/", labs=["LyricalHare"], actions=["attack"],
              pix_per_cm=16, fps=30)
bouts = hidra.bouts("results/")                   # ethogram DataFrame across all videos
```

Choosing between the labs' classifiers, and what to do when the calls look
systematically off, are covered in **[docs/zero-shot.md](docs/zero-shot.md)**.

### The job sheet (`jobs.csv`)

Columns `run,lab,action,subject,target`:

| column | meaning |
|---|---|
| `run` | `1` = include this row, `0` = skip |
| `lab` | which lab's classifier (AdaptableSnail, LyricalHare, …) |
| `action` | behaviour (attack, sniff, mount, sniffall, …) |
| `subject` | the acting mouse: `mouse1`…`mouse4`, `self`, or `*` (all) |
| `target` | the recipient mouse: `mouse1`…`mouse4`, `self`, or `*` (all) |

`--dump-jobs` pre-fills one row per available head with `run=1, subject=*, target=*`, so customising
is just deleting or zeroing rows. Only the labs that appear in enabled rows are actually run (faster).

Split-lab sniffing appears as the merged **`sniffall`** head; the other labs use **`sniff`**.

## Output (in `--out`, one set per input parquet)

- `<stem>.bouts.csv` — compact ethogram from the thresholded calls:
  `subject,target,lab,action,start_frame,stop_frame,n_frames,mean_prob,threshold`
- `<stem>.frames.parquet` — per-frame `prob` + binary `call` for every kept
  (subject,target,lab,action).

Control with `--output {both,calls,probs}` (default `both`). Thresholds are the leakage-free
train-calibrated values in `derived_thresholds_train.csv` (`prob >= threshold` → call), falling back
to a pooled value then 0.30.

## Options

```
--list-heads       print every (lab, action) classifier head and exit
--labs L1,L2       run only these labs' classifiers
--actions A1,A2    run only these behaviours
--subject M        acting mouse for --labs/--actions: mouse1..mouse4, self, * (default *)
--target M         recipient mouse, same values
--jobs FILE        job sheet, for per-head pair control (default: all heads x all pairs)
--dump-jobs FILE   write a template job sheet and exit
--out DIR          output folder (default: doom_predictions)
--pix-per-cm N     pixels-per-cm for all files (metadata.csv rows override)
--fps N            frame rate for all files
--gpu N            CUDA device index (default 0)
--output MODE      both | calls | probs
--weights TEMPLATE fine-tuned per-lab checkpoints, e.g. 'ft_models/{config}__mytag.pkl'
                   (or .safetensors -- either container, either backend)
--thresholds FILE  thresholds CSV (default: derived_thresholds_train.csv)
--threshold P      one constant threshold for every head
--configs C1,C2    run only these ensemble configs (default all 5; faster, lower quality)
--backend B        torch (default) | jax -- model backend, see docs/pytorch-port.md
--keep-work        keep the scratch inference directory
```

## Fine-tuning on your own annotations

```bash
python finetune.py prepare   --tracking parquets/ --annotations bouts.csv --lab GroovyShrew --out ft_data/
python finetune.py train     --data ft_data/ --lab GroovyShrew --actions rear,sniffall \
                             --videos mouseA_day1,mouseB_day1 --out ft_models/ --tag myrig
python predict.py  held_out/ --labs GroovyShrew --actions rear,sniffall --out ft_preds/ \
                             --weights 'ft_models/{config}__myrig.pkl' --pix-per-cm 16 --fps 30
python finetune.py calibrate --frames ft_preds/ --annotations bouts.csv --out ft_thresholds.csv
```

`prepare` stages your parquets plus a bout CSV into the layout the trainer reads, `train`
warm-starts the adopted lab's head from the published checkpoint and adapts it to your data (5
configs, one GPU; `--videos` keeps a recording out for the next step), and `calibrate` re-fits the
decision thresholds against your annotations — useful on its own, with no training, if the
zero-shot calls are right but the threshold is not. Label bouts with the head names
`--list-heads` prints; for the five sniff-splitting labs that is `sniffall`, which `prepare`
translates into the labels the trainer derives it from. `train` takes `--backend {torch,jax}`
(default `torch`); either writes the checkpoint in the same layout, `predict.py --weights` loads
it on either backend, and `hidra-convert-weights --one` turns it into safetensors. You adopt an
existing (lab, behaviour) head; for anything else see [docs/new-behaviours.md](docs/new-behaviours.md).
Add `--cache-features` (and, while iterating, `--ddi-steps 0 --configs 15fps_5bp`) to turn a
`--mode head` run from an hour per config into seconds. Full walkthrough and caveats:
**[docs/fine-tuning.md](docs/fine-tuning.md)**.

## Training from scratch

The two stages behind the published weights — the self-supervised trunk and the supervised
per-lab tail with its 82-column head — can be re-run with the research code in the package
(`solution.pretrain`, `train_perlab_heads.py`). It needs the JAX extra, a dataset in the trainer's
layout, and GPU-days rather than GPU-minutes; the head table (which (lab, behaviour) pairs get a
column) is declared in `thresholds.json`. **[docs/training.md](docs/training.md)** walks through
it, including the smoke runs to do before committing a GPU.

## Notes / gotchas

- Runs one full 5-config forward pass **per lab** (each lab has its own trunk embedding), so "all
  labs" = 15 passes — a few minutes per folder. Restrict via the job sheet to speed up.
- `video_id`s are derived from filenames; keep filenames unique within a folder.
- The tool sets `SNIFFALL=1` and `PREDICT_BATCH=64` (long-video memory) automatically, plus
  `XLA_FLAGS=--xla_gpu_autotune_level=0` (6-bodypart configs hang otherwise) and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, which only matter to `--backend jax`.
- Annotations for fine-tuning should cover **at least two mice**. The trainer pairs each
  annotated agent with a target, and a video annotating only one mouse leaves it with no
  target to choose (it fails inside numpy with `a cannot be empty unless no samples are taken`).
- A fine-tuned checkpoint belongs to the lab you adopted: always predict with it under
  `--labs <that lab>`, and keep its calibrated thresholds CSV next to it.

## Contents

```
pyproject.toml                   # the package (uv sync / pip install)
predict.py finetune.py download_models.py    # thin shims -> the hidra package

src/hidra/
  __init__.py                    # the Python API (hidra.heads/predict/bouts/...)
  cli.py                         # `predict.py`: inference, zero-shot or fine-tuned
  finetune.py                    # prepare / train / calibrate on your own annotations
  schema.py                      # label vocabularies, ensemble configs, head table
  paths.py                       # where weights/thresholds/scratch live
  solution.py train_perlab_heads.py pm_rule.py          # JAX models, data pipeline, trainers
  run_test_probs_perlab.py run_allbehaviors_perlab.py   # inference engine
  torch/                         # the PyTorch backend (default)
    layers.py models.py          #   ported model
    checkpoint.py convert.py     #   read JAX pickles without JAX; write safetensors
    infer.py                     #   inference loop
    train.py train_perlab.py     #   optimizer/EMA/loop + fine-tuning driver
  checkpoints.py                 # read/write weights in either format, no backend needed
  publish.py                     # hidra-publish-weights: upload a weight set to the Hub
  assets/derived_thresholds_train.csv    # per-(lab,action) thresholds (incl. sniffall)
  assets/model_card.md           # the Hugging Face model card, published by publish.py

reference/                       # the pre-port JAX implementation, byte-identical, runnable
tests/                           # JAX-vs-PyTorch exactness suite + CLI tests
docs/reuse.md                    # every way to reuse the model, with measured costs
docs/zero-shot.md docs/fine-tuning.md docs/new-behaviours.md docs/training.md docs/pytorch-port.md
models/                          # 5x backbone + 5x per-lab head + thresholds.json (~660 MB,
                                 #   NOT in git -- see "Model weights" above)
```
