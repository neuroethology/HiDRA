"""Guard: reference/ must stay byte-identical to the pre-port originals.

The whole exactness argument rests on reference/ being the *unmodified* JAX implementation.
If someone "fixes" a file in there to make a test pass, every comparison downstream becomes
circular. These hashes come from commit 9aed891, the last commit before the port.
"""
import hashlib
import pathlib

import pytest

REFERENCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "reference"
ORIGIN_COMMIT = "9aed8910f297e5f9828f5c01121f0d76681199cd"

REFERENCE_SHA256 = {
    "solution.py": "d5e0122d03477cada190488ee121baadf4af255d2793a00711f6ef9fb988ac83",
    "train_perlab_heads.py": "44688817c1477f5a10532543c58f5aecd13b3b19ab7279c0a682c08561d9245e",
    "predict.py": "7e978460d3d07db7daea5b8111765e9f9b3bfaf4fb31be86ac849e7773012ce8",
    "finetune.py": "5dd772eaf32e19bcec06f375d6189b38a74491a3aa076414d000225efa40e33f",
    "hidra.py": "0e00d7a49d20eeea9fe08d054a897c1fe360d516cbd0a06b79b61d5c72faf1f7",
    "download_models.py": "b9a25eb0aefad027158e97bb2bd45f4793de33857e896d20635de261048cb155",
    "pm_rule.py": "e880992fcd2e9e3eaa0c3d724947baadf835277a9d8fd3dcff1410a578b23ef8",
    "run_test_probs_perlab.py": "1fbc036ee3a2a8783daf146b06cf37c9060d9d53dd8e1ed13efc449781cd3f99",
    "run_allbehaviors_perlab.py": "93c754dc24a9fce2f5c104774387cafb8628aadd7378845803162ca6a6ffbac7",
    "derived_thresholds_train.csv": "009bb51c2acecd44c10eef531db687539ff691758325034cd7b01295f734b14a",
    "metadata.csv.example": "0aec1f56e633f88e601ca24e8584b2b5dbdca0a902f0ff06cc35e4c5c6255adb",
}


@pytest.mark.parametrize("name", sorted(REFERENCE_SHA256))
def test_reference_file_unmodified(name):
    path = REFERENCE_DIR / name
    assert path.is_file(), f"{path} is missing -- reference/ must stay complete"
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == REFERENCE_SHA256[name], (
        f"reference/{name} was modified.\n"
        f"  expected {REFERENCE_SHA256[name]} (commit {ORIGIN_COMMIT})\n"
        f"  got      {got}\n"
        f"Restore it with: git show {ORIGIN_COMMIT}:{name} > reference/{name}"
    )


def test_reference_models_symlink():
    """reference/models must resolve to the same weights the package uses."""
    from hidra import paths
    link = REFERENCE_DIR / "models"
    assert link.is_symlink(), "reference/models should be a symlink to the repo-root models/"
    assert link.resolve() == paths.models_dir().resolve()
