# Model weights

This directory is **empty in git** — the weights (~660 MB, files up to 156 MB) exceed
GitHub's 100 MB per-file limit, so they are hosted on the Hugging Face Hub at
[Neuroethology/HiDRA](https://huggingface.co/Neuroethology/HiDRA) (public, no login needed).

Fetch them:

```bash
hidra-download-models            # or, from a checkout: python download_models.py
```

That writes the 11 files HiDRA loads at inference time:

| file | count | what it is |
|---|---|---|
| `{config}_unsupervised.safetensors` | 5 | self-supervised trunk (frozen forecaster) |
| `{config}_supervised_perlab_sniffall.safetensors` | 5 | per-lab classifier head, 82 columns |
| `thresholds.json` | 1 | per-(lab, action) decision thresholds |

where `{config}` is one of `11fps_4bp`, `15fps_5bp`, `19fps_6bp`, `23fps_7bp`, `27fps_6bp`.
Inference averages all five configs, so all five are needed.

Set `$HIDRA_MODELS_DIR` to keep them somewhere else — on a scratch filesystem, or shared
between checkouts.

## Releases

| weights | what | pin with |
|---|---|---|
| **2026-10** (current `main`) | retrained self-supervised trunks (2026-10-06/07) and a per-lab classifier trained **without the shared cross-lab head** (2026-10-07/08); same 82 heads, re-calibrated thresholds; held-out mean F1 0.613 | `--revision b5b56d316117de5900832de2dea6f2b7fea6f75f` |
| 2026-09 and earlier | the multi-task model (per-lab head + shared 38-way head), trained on the original trunks | `--revision 6b9770c99f9e6e20a89f7c2b68169a72b14631e5` (safetensors); its pickles: `--format pkl --revision 4147d4fce1c46f7d56fdbf5dcc3e1e3b2888ba00` |

The model card on the Hub (`README.md` there, `src/hidra/assets/model_card.md` here) carries the
exact commit shas. [CHANGELOG.md](../CHANGELOG.md) has the full account of what changed and why.
Both sets load with the current code; only the current set's thresholds are bundled, so a pinned
older set needs the thresholds CSV of its own release (`--thresholds`).

## Formats

safetensors is the default: it executes no code on load and is read with numpy alone, so
opening the weights needs neither JAX nor PyTorch. The 2026-10 per-lab files were converted with
`hidra-convert-weights --strip-shared-head`, which leaves out the inert copy of the previous
model's shared head that the release pickles carry for older HiDRA versions; what remains is
byte-identical to the trainer's output.

The original `.pkl` checkpoints — pickles of JAX device arrays — hold the identical float32
values, verified byte-for-byte, and running the JAX backend from safetensors reproduces a
pickle-based run bit-identically. They are no longer on `main`; if you need them, pin the
last revision that carried them:

```bash
hidra-download-models --format pkl --revision 4147d4fce1c46f7d56fdbf5dcc3e1e3b2888ba00
```

`hidra-download-models` treats a checkpoint as present if *either* container is on disk, so
an existing `models/` full of `.pkl` files is not re-downloaded.

```bash
hidra-download-models --check                  # report what's missing, download nothing
hidra-download-models --repo ORG/NAME          # pull from a different Hub repo
hidra-download-models --revision v1.0          # pin a branch, tag, or commit
```

`--repo` / `--revision` also read from `$HIDRA_HF_REPO` / `$HIDRA_HF_REVISION`.

## Converting and publishing

To convert your own checkpoints — a fine-tune, say — to safetensors:

```bash
hidra-convert-weights                       # every checkpoint in models/
hidra-convert-weights --one ft_models/15fps_5bp__myrig.pkl
```

Conversion verifies byte equality on read-back, and needs neither JAX nor PyTorch: the
pickles reference exactly one JAX symbol, whose only job is to rebuild a numpy array.

Training a new set of weights -- either stage, on your own data or the consortium's -- is
described in [docs/training.md](../docs/training.md); both stages write into `HIDRA_MODELS_DIR`.

To publish a set of weights to the Hub (needs a write token):

```bash
hidra-publish-weights --dry-run             # show exactly what would change
hidra-publish-weights                       # one atomic commit
```

One commit matters: adding the new files and removing the old ones separately would leave
the repo in a state where `hidra-download-models` finds neither container complete.
