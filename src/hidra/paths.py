"""Filesystem locations, resolved the same way whether HiDRA is run from a git checkout
or from an installed wheel.

A source checkout keeps the weights in `<repo>/models` (that is what `download_models.py`
has always written, and what the README documents). An installed copy has no repo, so the
weights go to a user cache directory instead. Every path is overridable by an environment
variable so a cluster job can point them at scratch storage.

    HIDRA_MODELS_DIR     where the .pkl / .safetensors weights live
    HIDRA_THRESHOLDS     the per-(lab, action) decision-threshold CSV
    HIDRA_WORKDIR        scratch space for the memory-mapped tracking cache
    HIDRA_DATASET_DIR    the researcher-private training/eval dataset root
"""
import os
import tempfile
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PKG_DIR / "assets"


def repo_root():
    """The git checkout containing this package, or None when installed as a wheel.

    Detected by the layout `<root>/src/hidra/paths.py` plus a pyproject.toml at the root,
    so a `pip install`ed copy under site-packages never matches.
    """
    root = PKG_DIR.parent.parent
    if PKG_DIR.parent.name == "src" and (root / "pyproject.toml").is_file():
        return root
    return None


def _cache_home():
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hidra"


def models_dir():
    """Directory holding the downloaded model weights."""
    env = os.environ.get("HIDRA_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    root = repo_root()
    if root is not None:
        return root / "models"
    return _cache_home() / "models"


def thresholds_csv():
    """The bundled leakage-free train-calibrated decision thresholds."""
    env = os.environ.get("HIDRA_THRESHOLDS")
    if env:
        return Path(env).expanduser()
    return ASSETS_DIR / "derived_thresholds_train.csv"


def work_root():
    """Scratch directory for the memory-mapped tracking cache.

    Prefers /dev/shm (the tracking cache is read at random offsets every batch, so it
    wants to be in RAM) and falls back to the system temp dir where /dev/shm is absent
    or not writable.
    """
    env = os.environ.get("HIDRA_WORKDIR")
    if env:
        return Path(env).expanduser()
    # os.getuid is POSIX-only; on Windows fall back to the username (the temp dir is already
    # per-user there), so the default scratch path resolves without HIDRA_WORKDIR being set.
    uid = os.getuid() if hasattr(os, "getuid") else (os.environ.get("USERNAME") or "user")
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return shm / f"hidra-{uid}"
    return Path(tempfile.gettempdir()) / f"hidra-{uid}"


def dataset_dir():
    """Root of the researcher-private dataset tree (train/test parquets + CSVs).

    Only the training and evaluation entry points use this; `hidra predict` stages the
    user's own parquets into a scratch dataset instead.
    """
    env = os.environ.get("HIDRA_DATASET_DIR")
    if env:
        return Path(env).expanduser()
    root = repo_root()
    return (root / "data") if root is not None else (Path.cwd() / "data")
