# HiDRA — the High-Dimensional Rodent Annotator

Apply the trained **per-lab-head behaviour-classifier ensemble** (5-config: 11fps_4bp, 15fps_5bp,
19fps_6bp, 23fps_7bp, 27fps_6bp) to your own pose-tracking parquets. By default HiDRA runs **every
lab's classifiers, for every behaviour, on every mouse pair**; you can narrow this with `--labs` /
`--actions` or a job sheet.

Two ways to use it:

- **[Zero-shot inference](docs/zero-shot.md)** — no labels of your own. Pick one of the 82 trained
  (lab, behaviour) classifier heads and run it on your videos: `predict.py`.
- **[Fine-tuning](docs/fine-tuning.md)** — you have some annotations. Adapt the head you picked to
  your arena, pose rig and annotation style, then re-calibrate its threshold: `finetune.py`.

There is also a small Python API in [`hidra.py`](hidra.py) for notebook use.

## Setup

Python 3.12 + an NVIDIA GPU (driver ≥ 525). Install the deps with the bundled `requirements.txt`:

```bash
python3.12 -m venv hidra-env
source hidra-env/bin/activate
pip install -r requirements.txt        # JAX(cuda12) + numpy + pandas + pyarrow + huggingface_hub
```

The `jax[cuda12]` extra ships its own CUDA wheels, so no system CUDA toolkit is needed. CPU-only works
but is much slower (see the note in `requirements.txt`).

### Model weights

The weights are **not in this repo** — at ~660 MB (individual files up to 149 MB) they exceed
GitHub's file-size limit. They are hosted on the Hugging Face Hub instead, at
**[talmolab/HiDRA](https://huggingface.co/talmolab/HiDRA)** (public, no login needed).
Download them once into `models/` (needs ~700 MB free):

```bash
python download_models.py
```

That fetches 11 files: 5 trunk/backbone embeddings, 5 per-lab classifier heads, and
`thresholds.pkl`. The download resumes and skips files already present, so it is safe to re-run.

```bash
python download_models.py --check                  # report what's missing, download nothing
python download_models.py --repo ORG/NAME          # pull from a different Hub repo
python download_models.py --revision v1.0          # pin a branch, tag, or commit
```

`--repo` / `--revision` also read from `$HIDRA_HF_REPO` / `$HIDRA_HF_REVISION`. `predict.py`
refuses to start with weights missing and points you back here.

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
# activate the environment created in Setup (or use any interpreter with requirements.txt installed)
source hidra-env/bin/activate

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
--thresholds FILE  thresholds CSV (default: derived_thresholds_train.csv)
--threshold P      one constant threshold for every head
--configs C1,C2    run only these ensemble configs (default all 5; faster, lower quality)
--keep-work        keep the scratch inference directory
```

## Fine-tuning on your own annotations

```bash
python finetune.py prepare  --tracking parquets/ --annotations bouts.csv --lab GroovyShrew --out ft_data/
python finetune.py train    --data ft_data/ --lab GroovyShrew --actions rear --out ft_models/ --tag myrig
python finetune.py calibrate --frames ft_preds/ --annotations heldout_bouts.csv --out ft_thresholds.csv
```

`prepare` stages your parquets plus a bout CSV into the layout the trainer reads, `train`
warm-starts the adopted lab's head from the published checkpoint and adapts it to your data (5
configs, one GPU), and `calibrate` re-fits the decision thresholds against your annotations —
useful on its own, with no training, if the zero-shot calls are right but the threshold is not.
You adopt an existing (lab, behaviour) head; new behaviour names and new labs are not possible.
Full walkthrough and caveats: **[docs/fine-tuning.md](docs/fine-tuning.md)**.

## Notes / gotchas

- Runs one full 5-config forward pass **per lab** (each lab has its own trunk embedding), so "all
  labs" = 15 passes — a few minutes per folder. Restrict via the job sheet to speed up.
- `video_id`s are derived from filenames; keep filenames unique within a folder.
- The tool sets `SNIFFALL=1`, `XLA_FLAGS=--xla_gpu_autotune_level=0` (6-bodypart configs hang
  otherwise), `PREDICT_BATCH=64` (long-video memory), and `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  automatically.

## Contents

```
predict.py                       # inference (zero-shot or with fine-tuned weights)
finetune.py                      # prepare / train / calibrate on your own annotations
hidra.py                         # Python API over predict.py + the outputs
docs/zero-shot.md docs/fine-tuning.md
solution.py train_perlab_heads.py pm_rule.py
run_test_probs_perlab.py run_allbehaviors_perlab.py   # inference engine
derived_thresholds_train.csv     # per-(lab,action) thresholds (incl. sniffall)
download_models.py               # fetch the weights from Hugging Face into models/
models/                          # 5x backbone + 5x per-lab head + thresholds.pkl (~660 MB,
                                 #   NOT in git -- see "Model weights" above)
```
