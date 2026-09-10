"""Fixtures for the exactness suite.

The expensive objects -- synthetic pose data, the staged dataset, and the loaded
checkpoints for both backends -- are session-scoped, because building them costs far more
than any individual comparison.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Must precede any hidra import: SNIFFALL adds the 38th action, which the published
# checkpoints were trained with, and the XLA flags keep the Triton autotuner from
# spamming test output at ERROR level.
os.environ.setdefault("SNIFFALL", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import exactness as X  # noqa: E402


def _have_weights():
    from hidra.download_models import missing
    return not missing()


def _have_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _have_jax():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


requires_weights = pytest.mark.skipif(
    not _have_weights(), reason="model weights absent; run `hidra-download-models`")
requires_cuda = pytest.mark.skipif(not _have_cuda(), reason="no CUDA device")
requires_jax = pytest.mark.skipif(
    not _have_jax(), reason="JAX backend not installed (`uv sync --extra jax`)")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line("markers", "jax: requires the JAX backend installed")
    config.addinivalue_line("markers", "slow: full-ensemble / end-to-end runs")


@pytest.fixture(scope="session")
def device():
    return "cuda" if _have_cuda() else "cpu"


@pytest.fixture(scope="session")
def track_dir(tmp_path_factory):
    """A folder of synthetic tracking parquets plus metadata.csv."""
    from synth import write_dataset

    out = tmp_path_factory.mktemp("tracking")
    write_dataset(str(out), n_videos=2, n_frames=900, seed=0)
    return out


@pytest.fixture(scope="session")
def videos(track_dir, tmp_path_factory):
    """`solution.Video` objects for the synthetic parquets, staged as the CLI stages them.

    Note this also forces `solution.LABS.encode` to a constant, exactly as the inference
    driver does for --embedding-lab; several tests depend on that being in effect.
    """
    work = tmp_path_factory.mktemp("staged")
    return X.stage_videos(track_dir, work)


@pytest.fixture(scope="session")
def config_name():
    return X.DEFAULT_CONFIG


@pytest.fixture(scope="session")
def trunk_checkpoint(config_name):
    return X.load_checkpoints(config_name)[0]


@pytest.fixture(scope="session")
def head_checkpoint(config_name):
    return X.load_checkpoints(config_name)[1]


@pytest.fixture(scope="session")
def jax_head(config_name):
    """The JAX per-lab head with published weights (its `.module.unsupervised_model` is
    the bound trunk)."""
    return X.jax_models(config_name)[0]


@pytest.fixture(scope="session")
def torch_head(config_name, device):
    return X.torch_models(config_name, device=device)[1]


@pytest.fixture(scope="session")
def torch_trunk(torch_head):
    return torch_head.unsupervised_model


@pytest.fixture(scope="session")
def config(config_name):
    from hidra import schema
    return schema.get_configs()[config_name]


@pytest.fixture(scope="session")
def batch(config, videos):
    """One deterministic batch of augmented clips, shared by both backends."""
    return X.make_batch(config, videos, batch_size=4)
