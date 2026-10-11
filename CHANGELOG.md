# Changelog

All notable changes to HiDRA — the code in this repository and the weights it downloads — are
recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-10-10 — the no-shared-head classifier becomes the canonical model

Status at the time of writing: **the code changes below are committed on the `no-shared-head`
branch; the new weights are staged and converted but not yet published.** The remaining steps are
listed under [Still to do](#still-to-do) at the end.

### Why

The HiDRA classifier was retrained in October 2026 on retrained self-supervised trunks, three
ways, to measure what the shared cross-lab head and the external datasets contribute
(`CLASSIFIER_3WAY_HANDOFF.md`, `HANDOFF_NO_SHARING_CANONICAL_TO_A.md` in the retrain workspace).
Held-out DooM test (215 videos), mean frame-level F1 over the 82 (lab, behaviour) heads with
train-calibrated thresholds:

| arm | shared 38-way head | stage-2 data | mean F1 |
|---|---|---|---|
| shared + CalMS21 (the previous architecture) | yes | DooM + CalMS21 + CRIM13 | 0.618 |
| shared − CalMS21 | yes | DooM + CRIM13 | 0.616 |
| **no sharing (this release)** | **no** | DooM (+ external labs, a no-op without a shared head) | **0.613** |

The shared head and the external labs buy nothing measurable, so the simpler model — the
published architecture minus the `out-proj` layer and its `nll37` loss term — is now canonical.
Per-head test F1 / precision / recall for the new model are in the weight package
(`hidra_no_sharing_canonical/eval/test_f1_master_no_sharing.csv`, 82 rows; `f1_joint` averages
to 0.6128).

### The weights (2026-10 set)

All 11 Hub files change; the head table does not.

| | previous (≤ 2026-09) | 2026-10 |
|---|---|---|
| trunks `{cfg}_unsupervised` | original | retrained 2026-10-06/07 (`solution.pretrain`, 125k steps, 9 008 videos / 21 labs) |
| classifiers `{cfg}_supervised_perlab_sniffall` | multi-task: per-lab head + shared 38-way `out-proj` | per-lab head only, 177 tensors; trained 2026-10-07/08 with the default `train_perlab_heads` recipe on the 368 annotated DooM training videos of the 15 labs (early-stopped on validation F1, EMA at best step) |
| head table `thresholds.json` | 80 keys → 82 columns | **unchanged** (md5 `a4d3ba12…`) |
| bundled `derived_thresholds_train.csv` | 82 heads + 17 `pooled__*` rows, calibrated on the previous model | 82 heads, calibrated on this model's own DooM-train ensemble output; **no pooled rows** |
| size on the Hub | ~660 MB | 695 MB (decimal; 663 MiB) |

Source package: `/media/storage/USERS/akennedy/code/2026-10-05-HiDRA_test/hidra_no_sharing_canonical/`
(all MD5 sums verified). Staged for publication on this machine at
`/home/ann/hidra_models_canonical_2026-10/`: the 10 release pickles, their safetensors, `thresholds.json`,
and both thresholds CSVs (`derived_thresholds_train.csv` is what ships; `derived_thresholds_calms21train.csv`
re-tunes attack/sniff/mount/intromit on CalMS21 and is kept for reference only).

The release *pickles* carry an inert copy of the previous model's `out-proj`, spliced in on machine B
so that pre-0.2.0 HiDRA could still open them (`add_dummy_outproj.py`). The safetensors were
converted with the new `hidra-convert-weights --strip-shared-head` and are **byte-identical to the
trainer's raw output** (`weights/raw/{cfg}_v1_noshared_1layer.pkl`), verified for all five configs.

### Changed — model and loaders

- **`train_perlab_heads.py`** (JAX, stage-2 trainer and the model definition both backends mirror):
  `MultiTaskPerLabModel` pops the `out-proj` layer `SupervisedModel` builds, and `compute_loss`
  is the routed per-lab masked BCE alone (`metrics = {"nll", "nll88"}`). The default trainer run
  is therefore exactly the `phaseC_variant1_train.py` recipe that produced the release. New env
  switch **`SHARED_HEAD=1`** restores the pre-2026-10 multi-task architecture and objective
  (`out-proj` + `nll37`), for reproducing the earlier release. New helpers `model_layer_weights`
  (bind exactly the model's layers from a checkpoint, drop an inert `out-proj`, clear error for a
  missing layer) and `restore_input_stats`. Module docstring rewritten to describe the canonical
  architecture and its history.
- **`run_test_probs_perlab.py`** (JAX inference engine): binds `model_layer_weights(...)` instead of
  the whole checkpoint tree, so a checkpoint with an inert `out-proj` loads into the canonical model
  and a checkpoint missing a layer fails with a message naming it.
- **`torch/models.py`**: `MultiTaskPerLabModel(..., shared_head=False)`, `build_perlab(...,
  shared_head=False)`, `load_perlab(..., shared_head=False)`. The canonical head has no `out-proj`;
  a checkpoint carrying one loads with those eight tensors in `load_report["unexpected"]` (next to
  the trunk's dead `norm` leaf). `shared_head=True` builds the legacy multi-task head and then
  requires them.
- **`torch/train.py`**: `ddi_forward` applies `out-proj` only when the head has one;
  `perlab_loss(include_shared=True)` raises a clear `ValueError` on a head without it. Docstrings
  updated.
- **`head_table.py`**: `seed_embedding_row` docstring — a head-free slot's embedding row is now
  *untrained* (no shared head → no gradient), so seeding it from a donor lab matters more.

### Changed — fine-tuning (both backends)

Machine B found that head-only fine-tuning (`--mode head`) plateaued at F1 ≈ 0.32 on a new lab,
below the zero-shot donor head (0.47): `Trainer.initial_state` runs a 256-batch data-dependent
init that re-estimates the input-normalisation statistics (`m1, m2, n, mean, std`) of **every**
`normalize_input` Linear, frozen layers included, so the frozen tail read its inputs through
statistics it was never trained with. Restoring the foundation's statistics after the pass gave
0.36 → 0.51 over 3 → 240 positive seconds (`train_labtail_keepstats.py`). The torch port had the
same behaviour. Both backends now keep the statistics in head mode:

- **`train_perlab_heads.py`**: new `LABTAIL_KEEP_STATS` (default **on** under `LABTAIL_HEAD_ONLY=1`,
  off otherwise; `0`/`1` overrides). It wraps `trainer.initial_state`: the stock pass runs, then
  every layer's statistics are put back from the warm-start checkpoint and the EMA is re-seeded.
  This is the keepstats script ported into the trainer.
- **`torch/train_perlab.py`**: `LABTAIL_DDI_STEPS` defaults to **0 in head mode** (256 in tail and
  embedding mode, as before); the run logs why it skipped the pass.
- **`finetune.py`**: `--ddi-steps N` is passed to both backends (`LABTAIL_DDI_STEPS=N`;
  `LABTAIL_KEEP_STATS=1` iff N == 0), must be ≥ 0, and is shown by `--dry-run`. In head mode without
  the flag a note explains the new default; `--ddi-steps 256` is the pre-2026-10 behaviour.
  `SHARED_HEAD` is stripped from the trainer's environment like the other research switches.

Consequence: `finetune.py train --mode head` now leaves every other lab's head bit-identical and is
the fastest variant; results differ from 0.1.0 unless `--ddi-steps 256` is passed. `tail` and
`embedding` modes are unchanged.

### Changed — conversion and publishing

- **`torch/convert.py`**: new `--strip-shared-head` (and `convert_one(..., drop_layers=...)`),
  applied only to the per-lab kind (the trunk's own `out-proj` is its forecasting output); refuses a
  file without `out-proj-perlab`; records `dropped_layers` in the safetensors metadata; everything
  kept is still verified bit for bit.
- **`publish.py`**: the commit description is now generic (it used to describe the safetensors
  switch) and records the previous revision's sha so users can pin it; default commit message
  updated.
- **`download_models.py`**: comment at `DEFAULT_REVISION` on pinning across the weight change
  (left at `main`; the new upload's sha does not exist yet).
- `pyproject.toml`: version 0.1.0 → **0.2.0**.

### Changed — bundled assets

- `src/hidra/assets/derived_thresholds_train.csv` ← the package's `weights/derived_thresholds_train.csv`
  (md5 `018200cf…`, 82 rows, same key set as before minus the 17 `pooled__*` rows). The CLI's
  pooled fallback code path is untouched; a head without a row now falls back straight to 0.30
  until `finetune.py calibrate` writes it one, and the CLI already says so.
- `src/hidra/assets/model_card.md` (what `hidra-publish-weights` uploads as the Hub README):
  rewritten for the 2026-10 set — model, training, evaluation, changes from the previous release,
  how to pin the previous weights (`6b9770c99f9e6e20a89f7c2b68169a72b14631e5`, the Hub's `main`
  at the time of the dry run; pickles `4147d4fce1c46f7d56fdbf5dcc3e1e3b2888ba00`). One placeholder
  remains for the sha of the new upload.

### Changed — documentation

- `README.md`: weight-change notice with pointer here; backend-parity wording now says which weight
  set it was measured on; fine-tuning and training-from-scratch paragraphs; `CHANGELOG.md` in the
  contents tree.
- `docs/training.md` §3: the stage-2 objective is the per-lab loss alone; a new "The shared head is
  gone" section (the comparison, DooM-only stage-2 data, untrained head-free embedding rows, what the
  checkpoints contain, `SHARED_HEAD=1`); `SHARED_HEAD` and `LABTAIL_KEEP_STATS` in the research-lever
  table.
- `docs/pytorch-port.md`: states that the headline parity numbers are from the pre-2026-10 weights and
  that the GPU comparison on the new set is pending; `shared_head`; the data-dependent-init paragraph;
  `--strip-shared-head`.
- `docs/fine-tuning.md`: head mode keeps the statistics by default (§3, §3b, troubleshooting row);
  parity caveat.
- `docs/zero-shot.md` §5: how the bundled thresholds were derived for this model; no pooled rows.
- `docs/new-behaviours.md`: head-free slots' embedding rows are untrained → use `--seed-from`;
  vocabulary note no longer presumes a shared head.
- `docs/reuse.md`: "both heads" → "the per-lab head". `models/README.md`: a releases table with the
  revisions to pin, and the strip note.

### Tests

- New `tests/test_no_shared_head.py` (10 tests, CPU, random weights, no downloaded weights needed):
  canonical torch head has no `out-proj`; legacy head is opt-in; a checkpoint with an inert
  `out-proj` loads into the canonical head and predicts identically to the stripped one; the legacy
  head refuses a stripped checkpoint; DDI and loss run without the shared head, and
  `include_shared=True` fails clearly; fine-tune checkpoint layout has 177 tensors and round-trips;
  head-mode DDI default; JAX model has no `out-proj`, `model_layer_weights` / `restore_input_stats`
  behave; `SHARED_HEAD=1` rebuilds the legacy JAX model (subprocess); **JAX and torch agree to
  < 1e-4 relative on the per-lab logits of the no-shared-head model** (random weights, synthetic
  batch).
- `tests/test_checkpoint_conversion.py`: tensor counts accept 177 or 185; the bit-identity test
  allows exactly the stripped `out-proj/*`; new `test_strip_shared_head_drops_only_out_proj`;
  the `thresholds.pkl` → JSON round-trip skips when the models dir has only `thresholds.json`.
- `tests/test_training_exactness.py`: the DDI parity check no longer lists `out-proj`.
- `tests/test_train_options.py`: new `test_head_mode_keeps_the_input_statistics_by_default`.
- `tests/test_new_head.py`: widened-checkpoint conversion accepts 177 or 185 tensors.
- `tests/test_packaging.py`, `tests/test_reference_pristine.py`: the two "default models dir" tests
  now unset `HIDRA_MODELS_DIR` so they test the default rather than the session's override.

### Verification performed (machine A, 2026-10-10)

This machine's NVIDIA driver was broken at the time (`nvidia-smi`: driver/library version mismatch),
so everything below ran on **CPU**; GPU-marked tests skipped.

- `md5sum -c MD5SUMS` on the package: every file OK.
- `hidra-convert-weights --models-dir … --strip-shared-head`: 10 safetensors, `--check` clean; the
  five per-lab safetensors are byte-identical to `weights/raw/*_v1_noshared_1layer.pkl` (177 tensors
  each); the five trunks keep their 421 tensors.
- Both loaders on the staged 15fps_5bp release pickle: torch `load_perlab` → 177 tensors loaded,
  `missing == []`, `unexpected == out-proj/*`; JAX `model_layer_weights` binds 17 layers and drops
  `out-proj`; the raw pickle loads into both with nothing unexpected; the legacy torch head refuses
  the raw pickle with a clear error.
- **Real-weights backend agreement on CPU** (synthetic 4-window batch, float32, `tests/exactness`
  helpers): per-lab logits max |Δ| 1.8e-5 (1.0e-6 relative) for 15fps_5bp and 1.5e-5 (8.2e-7) for
  11fps_4bp; probabilities within 1e-6; 12 non-zero action columns for the GroovyShrew embedding,
  all in [0, 1].
- `pytest -m "not slow"` with `HIDRA_MODELS_DIR` pointing at the staged set, in a fresh venv with
  torch 2.14.1+cpu and jax 0.9.2 (CPU): **144 passed, 1 failed, 41 skipped** (GPU-only), 25
  deselected (slow). The one failure,
  `test_reference_pristine.py::test_reference_file_unmodified[download_models.py]`, also fails on
  the untouched checkout at commit 6dabdd9 (its recorded hash predates the repository-ownership
  rename in that file) and is unrelated to this change.
- `hidra-publish-weights --models-dir … --dry-run`: plan is 12 files, 695.2 MB (10 safetensors +
  `thresholds.json` + the model card); Hub `main` was `6b9770c99f9e6e20a89f7c2b68169a72b14631e5`.
- `ruff check src tests`: clean. `hidra-predict --list-heads` with the new CSV: 82 heads, 15 labs.

### Still to do

In the order of `HANDOFF_NO_SHARING_CANONICAL_TO_A.md`:

1. **GPU parity and regression (handoff §3.3)** — not run here (no working CUDA driver, no DooM test
   videos on this machine). On a GPU machine: `pytest tests/test_checkpoint_conversion.py
   tests/test_layer_exactness.py tests/test_endtoend_exactness.py` with
   `HIDRA_MODELS_DIR=/home/ann/hidra_models_canonical_2026-10`, then `predict.py` on a few DooM
   test videos with `--backend jax` and `torch` and diff the ethograms; re-quote the numbers in
   `docs/pytorch-port.md` and the model card. Reproducing the 0.613 needs machine B's eval tree
   and the DooM-repo tools (`eval_arm.sh`).
2. **Publish (§3.5)** — `hidra-publish-weights --models-dir /home/ann/hidra_models_canonical_2026-10
   --message "No-sharing per-lab classifier, retrained trunks (2026-10)"` with a write token (the
   dry run above is the plan it will execute; it removes any `.pkl` left on `main`). Then fill the
   new upload's sha into `src/hidra/assets/model_card.md` (one `TODO` marker) and, if reproducible
   installs are wanted, into `download_models.DEFAULT_REVISION`; tag the git repo (e.g.
   `v0.2.0-noshared`).
3. **Commit** — nothing has been committed; `git status` lists the files. The pre-existing
   `reference/models` symlink change (`../models` → `../models/`) was already in the tree.

### Compatibility

- **Old code (< 0.2.0) with the new Hub weights** fails to load the per-lab safetensors (it expects
  the `out-proj` subtree): upgrade, or pin `--revision 6b9770c99f9e6e20a89f7c2b68169a72b14631e5`.
- **New code with the old weights** works (the shared head is read and ignored), but the bundled
  thresholds are calibrated for the new model — pass the previous release's CSV via `--thresholds`.
- Probabilities and thresholds are not comparable across the two weight sets; a study should stay
  on one revision.
- Fine-tuned checkpoints written by 0.2.0 have no `out-proj` (177 tensors); they load on either
  backend in 0.2.0 and later, not in older versions.
- The head table, head names, `.heads.json` sidecars, `--new-head` and the thresholds CSV format are
  unchanged.

## [0.1.0] — 2026-09

The uv-installable package with the PyTorch port, safetensors weights, fine-tuning (`finetune.py`
with `--new-head`, `--cache-features`, `--cosine-steps`, `--patience`) and the example workflow
documentation; see the git history (commits c3e5f92 … 6dabdd9).
