"""Standing data rule: PleasantMeerkat contributes ONLY its 'follow' annotations.

PleasantMeerkat's attack/chase/escape bouts are punctate 0.2s single-bin (5 Hz Solomon-coder)
artifacts, not real bout structure; only its `follow` has a genuine duration distribution. So
from 2026-07-07 on, every DATASET/CLASSIFIER analysis (GT bout stats, Jaccard, ethograms, F1,
category pooling, cross-lab agreement) drops PleasantMeerkat rows whose action != 'follow', and
drops PleasantMeerkat's non-follow classifier heads.

This supersedes the old blanket `DROP={"PleasantMeerkat"}`: for `follow`, INCLUDE PleasantMeerkat.
NOTE: this rule is for ANALYSIS scripts only -- it does NOT touch the inference/submission
pipeline (solution.py, run_submission/*, run_*_predictions.py) or the build_ethograms library.
See memory: pleasantmeerkat-follow-only.
"""
PM = "PleasantMeerkat"
PM_KEEP = "follow"


def drop_pm_nonfollow(df, lab):
    """Annotations DataFrame (needs an 'action' column) for a single `lab`:
    drop PleasantMeerkat rows whose action != 'follow'. No-op for every other lab."""
    return df[df["action"] == PM_KEEP] if lab == PM else df


def pm_action_ok(lab, action):
    """False for a PleasantMeerkat non-follow (lab, action) pair; True otherwise.
    Use to gate CSV behaviors_labeled rows and per-(lab,behavior) classifier heads."""
    return not (lab == PM and action != PM_KEEP)
