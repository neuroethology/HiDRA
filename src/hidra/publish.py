"""Publish the model weights to the Hugging Face Hub.

    hidra-publish-weights --dry-run     # show exactly what would change
    hidra-publish-weights               # upload (needs a WRITE token)

Publishing is scripted rather than done by hand so that the next set of weights lands the
same way: the safetensors checkpoints, the thresholds table, and the model card, in a single
atomic commit. One commit matters -- adding the new files and removing the old ones
separately would leave the repo in a state where `hidra-download-models` finds neither
container complete.

Needs a write token: `hf auth login` with a token that has write access to the target repo,
or `HF_TOKEN=...`. A read token fails with `403 Forbidden: you must use a write token`.
"""
import argparse
import os
import sys
from pathlib import Path

from . import paths
from .download_models import CONFIGS, DEFAULT_REPO

MODEL_CARD = paths.ASSETS_DIR / "model_card.md"

# What a complete safetensors-format publication contains.
CHECKPOINT_STEMS = ([f"{c}_unsupervised" for c in CONFIGS]
                    + [f"{c}_supervised_perlab_sniffall" for c in CONFIGS])


def plan(models_dir=None, drop_pickles=True, repo=DEFAULT_REPO, revision="main"):
    """Work out what to add and what to remove. Returns (adds, deletes, current_sha)."""
    from huggingface_hub import HfApi

    models_dir = Path(models_dir or paths.models_dir())
    adds = []
    for stem in CHECKPOINT_STEMS:
        path = models_dir / f"{stem}.safetensors"
        if not path.is_file():
            sys.exit(f"ERROR: {path} is missing.\n"
                     f"Convert the weights first: hidra-convert-weights")
        adds.append((stem + ".safetensors", path))

    thresholds = models_dir / "thresholds.json"
    if not thresholds.is_file():
        sys.exit(f"ERROR: {thresholds} is missing.\n"
                 f"Convert the weights first: hidra-convert-weights")
    adds.append(("thresholds.json", thresholds))

    if not MODEL_CARD.is_file():
        sys.exit(f"ERROR: model card not found at {MODEL_CARD}")
    adds.append(("README.md", MODEL_CARD))

    api = HfApi()
    info = api.model_info(repo, revision=revision)
    present = {f.rfilename for f in info.siblings}
    deletes = sorted(f for f in present if f.endswith(".pkl")) if drop_pickles else []
    return adds, deletes, info.sha


def publish(models_dir=None, repo=DEFAULT_REPO, revision="main", drop_pickles=True,
            dry_run=False, message=None):
    """Upload in a single commit. Returns the commit URL, or None for a dry run."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    adds, deletes, current_sha = plan(models_dir, drop_pickles, repo, revision)

    total = sum(p.stat().st_size for _, p in adds)
    print(f"target: {repo} @ {revision}  (currently {current_sha})")
    print(f"\nadd {len(adds)} file(s), {total / 1e6:.1f} MB:")
    for name, path in adds:
        print(f"  + {name:<52} {path.stat().st_size / 1e6:>8.2f} MB")
    if deletes:
        print(f"\ndelete {len(deletes)} file(s):")
        for name in deletes:
            print(f"  - {name}")
    if dry_run:
        print("\n(dry run -- nothing uploaded)")
        return None

    api = HfApi()
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                  for name, path in adds]
    operations += [CommitOperationDelete(path_in_repo=name) for name in deletes]

    description = (
        "safetensors holding the identical float32 values as the pickled JAX checkpoints, "
        "verified byte-for-byte. safetensors executes no code on load and is read with "
        "numpy alone, so opening the weights needs neither JAX nor PyTorch. thresholds.pkl "
        "becomes thresholds.json, removing the last pickle from the load path.\n\n"
        "Running the JAX backend from these files reproduces a pickle-based run "
        "bit-identically.\n"
    )
    if deletes:
        description += (
            f"\nThe .pkl files are removed. HiDRA versions predating this change resolve "
            f"them by name; upgrade, or pin the last revision that carried them:\n\n"
            f"    hidra-download-models --format pkl --revision {current_sha}\n")

    result = api.create_commit(
        repo_id=repo, operations=operations,
        commit_message=message or "Switch to safetensors weights; add model card",
        commit_description=description)
    print(f"\ncommitted: {result.commit_url}")
    print(f"the last revision carrying .pkl was {current_sha}")
    return result.commit_url


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="hidra-publish-weights", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"target repo (default: {DEFAULT_REPO})")
    ap.add_argument("--revision", default="main", help="branch to commit to (default: main)")
    ap.add_argument("--models-dir", default=None,
                    help=f"where the converted weights are (default: {paths.models_dir()})")
    ap.add_argument("--keep-pickles", action="store_true",
                    help="leave the existing .pkl files in place instead of removing them")
    ap.add_argument("--message", help="commit message")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without uploading")
    args = ap.parse_args(argv)

    if not args.dry_run:
        from huggingface_hub import get_token
        if not (os.environ.get("HF_TOKEN") or get_token()):
            sys.exit("ERROR: no Hugging Face token found. Log in with a WRITE token:\n"
                     "  hf auth login\n"
                     "or set HF_TOKEN=...")

    publish(models_dir=args.models_dir, repo=args.repo, revision=args.revision,
            drop_pickles=not args.keep_pickles, dry_run=args.dry_run, message=args.message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
