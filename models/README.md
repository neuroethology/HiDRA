# Model weights

This directory is **empty in git** — the weights (~660 MB, files up to 149 MB) exceed GitHub's
100 MB per-file limit, so they are hosted on the Hugging Face Hub instead.

Fetch them from the repo root:

```bash
pip install huggingface_hub
python download_models.py
```

That writes the 11 files HiDRA loads at inference time:

| file | count | what it is |
|---|---|---|
| `{config}_unsupervised.pkl` | 5 | trunk / backbone embedding, one per config |
| `{config}_supervised_perlab_sniffall.pkl` | 5 | per-lab classifier heads, one per config |
| `thresholds.pkl` | 1 | ensemble thresholds |

where `{config}` is one of `11fps_4bp`, `15fps_5bp`, `19fps_6bp`, `23fps_7bp`, `27fps_6bp`.

Verify what's present without downloading:

```bash
python download_models.py --check
```

Point at a different Hub repo or pin a revision with `--repo` / `--revision` (or the
`HIDRA_HF_REPO` / `HIDRA_HF_REVISION` environment variables).
