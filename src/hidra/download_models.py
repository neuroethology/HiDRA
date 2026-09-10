#!/usr/bin/env python3
"""
Fetch the HiDRA model weights from the Hugging Face Hub into models/.

The weights (~660 MB) are too large to ship in git, so they live on the Hub and are
downloaded once:

    pip install huggingface_hub
    python download_models.py

Override the source repo with --repo or the HIDRA_HF_REPO environment variable, and pin a
particular upload with --revision (a branch, tag, or commit sha).

Two containers hold the same weights. `--format safetensors` (the default) executes no code
on load and is read with numpy alone; `--format pkl` is the original format, a pickle of JAX
device arrays. A checkpoint already present in *either* container counts as downloaded, so
an install predating the conversion is never asked to re-fetch 660 MB.
"""
import argparse
import os
import sys

from . import paths

# The Hub repo holding the weights. Override with --repo / $HIDRA_HF_REPO.
DEFAULT_REPO = os.environ.get("HIDRA_HF_REPO", "talmolab/HiDRA")
DEFAULT_REVISION = os.environ.get("HIDRA_HF_REVISION", "main")

CONFIGS = ["11fps_4bp", "15fps_5bp", "19fps_6bp", "23fps_7bp", "27fps_6bp"]

# One self-supervised backbone and one per-lab classifier head per config.
STEMS = ([f"{c}_unsupervised" for c in CONFIGS]
         + [f"{c}_supervised_perlab_sniffall" for c in CONFIGS])

# Both formats hold identical values, so only one of each is needed. safetensors is the
# default: it executes no code on load and is read with numpy alone, whereas the .pkl files
# are pickles of JAX device arrays. The pickles remain on the Hub for anyone pinned to an
# older revision, and `--format pkl` still fetches them.
FORMATS = {
    "safetensors": ([f"{s}.safetensors" for s in STEMS], "thresholds.json"),
    "pkl": ([f"{s}.pkl" for s in STEMS], "thresholds.pkl"),
}
DEFAULT_FORMAT = os.environ.get("HIDRA_WEIGHTS_FORMAT", "safetensors")

# Kept for callers that import it; the default-format file list.
REQUIRED = FORMATS[DEFAULT_FORMAT][0] + [FORMATS[DEFAULT_FORMAT][1]]


def required_files(fmt=None):
    """The files one format needs: the checkpoints plus the thresholds table."""
    fmt = fmt or DEFAULT_FORMAT
    if fmt not in FORMATS:
        sys.exit(f"ERROR: unknown weights format {fmt!r}; choose from {sorted(FORMATS)}")
    checkpoints, thresholds = FORMATS[fmt]
    return list(checkpoints) + [thresholds]


def missing(models_dir=None, fmt=None):
    """Weight files that are not present on disk, in whichever format is satisfied.

    A checkpoint counts as present if *either* container is there, so an install that
    predates the safetensors conversion is not asked to re-download 660 MB.
    """
    models_dir = str(paths.models_dir()) if models_dir is None else models_dir

    def present(name):
        return os.path.isfile(os.path.join(models_dir, name))

    if fmt is not None:
        return [f for f in required_files(fmt) if not present(f)]

    absent = []
    for stem in STEMS:
        if not any(present(f"{stem}{suffix}") for suffix in (".safetensors", ".pkl")):
            absent.append(f"{stem}.safetensors")
    if not any(present(n) for n in ("thresholds.json", "thresholds.pkl")):
        absent.append("thresholds.json")
    return absent


def require_weights(models_dir=None):
    """Exit with actionable instructions if any weights are absent."""
    models_dir = str(paths.models_dir()) if models_dir is None else models_dir
    absent = missing(models_dir)
    if absent:
        sys.exit(
            f"ERROR: {len(absent)} model weight file(s) missing from {models_dir}\n"
            f"  missing: {', '.join(absent[:4])}{' ...' if len(absent) > 4 else ''}\n\n"
            f"The weights are hosted on Hugging Face, not in the git repo. Fetch them with:\n"
            f"  hidra-download-models\n"
            f"(or, from a checkout: python download_models.py)"
        )


def _is_missing_file(exc):
    """True when a Hub download failed because the file is absent (404), not for any other
    reason -- a network error or a permissions problem must not be papered over."""
    from huggingface_hub.errors import EntryNotFoundError

    if isinstance(exc, EntryNotFoundError):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def _other_format_name(fname):
    """The same checkpoint in the other container, or None if there is no counterpart."""
    swaps = {".safetensors": ".pkl", ".pkl": ".safetensors",
             "thresholds.json": "thresholds.pkl", "thresholds.pkl": "thresholds.json"}
    if fname in swaps:
        return swaps[fname]
    for suffix, other in ((".safetensors", ".pkl"), (".pkl", ".safetensors")):
        if fname.endswith(suffix):
            return fname[: -len(suffix)] + other
    return None


def download(repo=DEFAULT_REPO, revision=DEFAULT_REVISION, models_dir=None, force=False,
             fmt=None):
    models_dir = str(paths.models_dir()) if models_dir is None else models_dir
    fmt = fmt or DEFAULT_FORMAT
    wanted = required_files(fmt)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("ERROR: huggingface_hub is not installed.\n  pip install huggingface_hub")

    os.makedirs(models_dir, exist_ok=True)
    todo = wanted if force else missing(models_dir)
    if not todo:
        print(f"All {len(wanted)} weight files already present in {models_dir}; nothing to do.")
        return

    print(f"Fetching {len(todo)} {fmt} file(s) from {repo} (revision {revision}) "
          f"into {models_dir}")
    for i, fname in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {fname}", flush=True)
        try:
            hf_hub_download(repo_id=repo, filename=fname, revision=revision,
                            local_dir=models_dir)
        except Exception as exc:
            # Fall back to the other container rather than failing. Which formats a given
            # repo or revision carries is not something this code should assume: older
            # revisions are pickle-only, newer ones safetensors-only, and a fork may differ.
            alt = _other_format_name(fname)
            if alt is None or not _is_missing_file(exc):
                raise
            print(f"      {fname} is not in this revision; using {alt}", flush=True)
            hf_hub_download(repo_id=repo, filename=alt, revision=revision,
                            local_dir=models_dir)

    still = missing(models_dir)
    if still:
        sys.exit(f"ERROR: still missing after download: {still}")
    print(f"Done. {len(wanted)} weight files in {models_dir}")


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
    ap.add_argument("--format", dest="fmt", default=DEFAULT_FORMAT, choices=sorted(FORMATS),
                    help=f"which container to fetch (default: {DEFAULT_FORMAT}). safetensors "
                         f"loads without a pickle and needs neither JAX nor PyTorch; pkl is "
                         f"the original format. Both hold identical values.")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--check", action="store_true",
                    help="report which weights are missing and exit (downloads nothing)")
    args = ap.parse_args()

    models_dir = args.models_dir or str(paths.models_dir())
    if args.check:
        absent = missing(models_dir)
        if absent:
            print(f"{len(absent)} weight file(s) missing from {models_dir}:")
            for f in absent:
                print(f"  {f}")
            sys.exit(1)
        have = [f for f in required_files("safetensors")
                if os.path.isfile(os.path.join(models_dir, f))]
        print(f"All weight files present in {models_dir} "
              f"({len(have)}/{len(required_files('safetensors'))} as safetensors)")
        return

    download(args.repo, args.revision, models_dir, args.force, args.fmt)


if __name__ == "__main__":
    main()
