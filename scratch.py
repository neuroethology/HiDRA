"""Where to put transient files: checkpoints, per-job tracking caches, staged models.

This exists because `/dev/shm` was hardcoded in a dozen places and `/dev/shm` does not exist
outside Linux, so every one of them was an unconditional failure on macOS.

The original reason for `/dev/shm` (train_perlab_heads.py) is worth keeping intact: checkpoints
must live on a filesystem that supports **symlinks**, because orbax writes a `best.pkl` symlink and
the samba mounts this pipeline reads its data from do not support them — the symlink raises
FileNotFoundError there. `/dev/shm` satisfied that and is also a tmpfs, so it is fast and
self-cleaning. The platform temp directory satisfies both requirements too; it is simply not
usually a tmpfs, which costs some speed and nothing in correctness.
"""

import os
import tempfile

SHM = "/dev/shm"


def root() -> str:
    """The scratch root: /dev/shm where usable, else the platform temp directory.

    Checked rather than assumed per-platform: /dev/shm is absent on macOS, and it can also be
    missing or read-only inside a hardened container, which used to fail the same way.
    """
    if os.path.isdir(SHM) and os.access(SHM, os.W_OK):
        return SHM
    return tempfile.gettempdir()


def path(*parts: str) -> str:
    """A path under the scratch root, e.g. path("doom_labtail", "models")."""
    return os.path.join(root(), *parts)
