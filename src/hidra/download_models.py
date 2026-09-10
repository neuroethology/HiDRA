#!/usr/bin/env python3
"""
Fetch the HiDRA model weights from the Hugging Face Hub into models/.

The weights (~660 MB) are too large to ship in git, so they live on the Hub and are
downloaded once:

    pip install huggingface_hub
    python download_models.py

Override the source repo with --repo or the HIDRA_HF_REPO environment variable, and pin a
particular upload with --revision (a branch, tag, or commit sha).
"""
import os, sys, argparse

from . import paths

# The Hub repo holding the weights. Override with --repo / $HIDRA_HF_REPO.
DEFAULT_REPO = os.environ.get("HIDRA_HF_REPO", "talmolab/HiDRA")
DEFAULT_REVISION = os.environ.get("HIDRA_HF_REVISION", "main")

CONFIGS = ["11fps_4bp", "15fps_5bp", "19fps_6bp", "23fps_7bp", "27fps_6bp"]

# Every file solution.py / train_perlab_heads.py load out of persist_dir: one unsupervised
# backbone and one per-lab supervised head per config, plus the ensemble thresholds.
REQUIRED = ([f"{c}_unsupervised.pkl" for c in CONFIGS]
            + [f"{c}_supervised_perlab_sniffall.pkl" for c in CONFIGS]
            + ["thresholds.pkl"])


def missing(models_dir=None):
    """Required weight files that are not present on disk."""
    models_dir = str(paths.models_dir()) if models_dir is None else models_dir
    return [f for f in REQUIRED if not os.path.isfile(os.path.join(models_dir, f))]


def require_weights(models_dir=None):
    """Exit with actionable instructions if any weights are absent."""
    models_dir = str(paths.models_dir()) if models_dir is None else models_dir
    absent = missing(models_dir)
    if absent:
        sys.exit(
            f"ERROR: {len(absent)} of {len(REQUIRED)} model weight file(s) missing from {models_dir}\n"
            f"  missing: {', '.join(absent[:4])}{' ...' if len(absent) > 4 else ''}\n\n"
            f"The weights are hosted on Hugging Face, not in the git repo. Fetch them with:\n"
            f"  hidra-download-models\n"
            f"(or, from a checkout: python download_models.py)"
        )


def download(repo=DEFAULT_REPO, revision=DEFAULT_REVISION, models_dir=None, force=False):
    models_dir = str(paths.models_dir()) if models_dir is None else models_dir
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("ERROR: huggingface_hub is not installed.\n  pip install huggingface_hub")

    os.makedirs(models_dir, exist_ok=True)
    todo = REQUIRED if force else missing(models_dir)
    if not todo:
        print(f"All {len(REQUIRED)} weight files already present in {models_dir}; nothing to do.")
        return

    print(f"Fetching {len(todo)} file(s) from {repo} (revision {revision}) into {models_dir}")
    for i, fname in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {fname}", flush=True)
        hf_hub_download(repo_id=repo, filename=fname, revision=revision,
                        local_dir=models_dir)

    still = missing(models_dir)
    if still:
        sys.exit(f"ERROR: still missing after download: {still}")
    print(f"Done. {len(REQUIRED)} weight files in {models_dir}")


def main():
    ap = argparse.ArgumentParser(
        prog="download_models.py",
        description="Download the HiDRA model weights from the Hugging Face Hub into models/.")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"Hugging Face repo id holding the weights (default: {DEFAULT_REPO})")
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help="branch, tag, or commit sha to pull (default: main)")
    ap.add_argument("--models-dir", default=None,
                    help=f"destination directory (default: {paths.models_dir()})")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--check", action="store_true",
                    help="report which weights are missing and exit (downloads nothing)")
    args = ap.parse_args()

    models_dir = args.models_dir or str(paths.models_dir())
    if args.check:
        absent = missing(models_dir)
        if absent:
            print(f"{len(absent)} of {len(REQUIRED)} weight file(s) missing from {models_dir}:")
            for f in absent:
                print(f"  {f}")
            sys.exit(1)
        print(f"All {len(REQUIRED)} weight files present in {models_dir}")
        return

    download(args.repo, args.revision, models_dir, args.force)


if __name__ == "__main__":
    main()
