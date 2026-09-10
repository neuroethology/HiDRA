"""The package installs and its entry points work without a source checkout on sys.path."""
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_import_and_api_surface():
    import hidra

    assert len(hidra.heads()) == 82
    assert len(hidra.labs()) == 15
    assert "sniffall" in hidra.actions("GroovyShrew")
    # `heads(lab=..)` and `heads(action=..)` must filter consistently.
    assert set(hidra.heads(lab="BoisterousParrot")) == {("BoisterousParrot", "shepherd")}
    assert all(a == "attack" for _, a in hidra.heads(action="attack"))


def test_thresholds_frame():
    import hidra

    t = hidra.thresholds()
    assert list(t.columns) == ["lab", "action", "threshold"]
    assert t.threshold.between(0, 1).all()
    assert len(t) >= 82


def test_paths_resolve_inside_checkout():
    from hidra import paths

    root = paths.repo_root()
    assert root is not None, "running from a checkout, so repo_root() should be found"
    assert paths.models_dir() == root / "models"
    assert paths.thresholds_csv().is_file()


def test_models_dir_env_override(tmp_path, monkeypatch):
    from hidra import paths

    monkeypatch.setenv("HIDRA_MODELS_DIR", str(tmp_path / "elsewhere"))
    assert paths.models_dir() == tmp_path / "elsewhere"


def test_packaged_assets_are_installed():
    """The thresholds CSV must ship inside the wheel, not only in the repo root."""
    from hidra import paths

    assert (paths.ASSETS_DIR / "derived_thresholds_train.csv").is_file()
    assert (paths.ASSETS_DIR / "metadata.csv.example").is_file()


@pytest.mark.parametrize("entry", ["hidra-predict", "hidra-download-models"])
def test_console_scripts_exist(entry):
    exe = pathlib.Path(sys.executable).parent / entry
    assert exe.is_file(), f"{entry} console script not installed"


def test_list_heads_via_module():
    """`python -m hidra.cli` is how the predict subprocess is reached; it must not need cwd."""
    out = subprocess.run([sys.executable, "-m", "hidra.cli", "--list-heads"],
                         capture_output=True, text=True, cwd="/", timeout=180)
    assert out.returncode == 0, out.stderr
    assert "82 classifier heads across 15 labs" in out.stdout


def test_root_shims_still_work():
    for shim in ("predict.py", "download_models.py"):
        assert (REPO / shim).is_file(), f"{shim} shim missing -- documented commands would break"
    out = subprocess.run([sys.executable, str(REPO / "predict.py"), "--list-heads"],
                         capture_output=True, text=True, cwd=str(REPO), timeout=180,
                         env={**os.environ, "PYTHONPATH": ""})
    assert out.returncode == 0, out.stderr
    assert "classifier heads" in out.stdout
