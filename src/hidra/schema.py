"""Static model schema: the label vocabularies, the ensemble configs, and the per-lab
head table.

Everything here is plain Python and numpy. It is split out of `solution.py` -- which
imports JAX at module scope -- so the PyTorch backend can describe the model without
pulling in the JAX runtime. `solution.py` re-exports these names, so
`solution.ACTIONS is schema.ACTIONS` and there is exactly one vocabulary in the process.

The `sniffall` action is appended at import time when the SNIFFALL environment variable is
set, matching the original behaviour: default 37-action runs and checkpoints stay
byte-identical, and the merged sniff-family head only exists when asked for. `hidra.cli`
sets SNIFFALL=1 before importing anything, so the published 38-action checkpoints load.
"""
import os

import numpy as np

class Enum:
    def __init__(self, *values):
        self.values = list(values)
        self.value_to_idx = {value: idx for idx, value in enumerate(self.values)}

    def encode(self, value):
        return self.value_to_idx[value]

    def decode(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


MOUSE_IDS = Enum(1, 2, 3, 4)


def parse_mouse_id(m):
    """Convert 'mouse1' -> 1, or pass through if already int. 'self' passes through."""
    if isinstance(m, str) and m.startswith("mouse"):
        return int(m.replace("mouse", ""))
    return m


BODYPARTS = Enum(
    "tail_base",
    "ear_right",
    "ear_left",
    "nose",
    "neck",
    "body_center",
    "tail_tip",
    "tail_midpoint",
    "forepaw_left",
    "forepaw_right",
    "hindpaw_left",
    "hindpaw_right",
    "hip_right",
    "hip_left",
    "lateral_right",
    "lateral_left",
    "head",
    "spine_1",
    "spine_2",
    "tail_middle_1",
    "tail_middle_2",
    "headpiece_topfrontright",
    "headpiece_topbackright",
    "headpiece_topfrontleft",
    "headpiece_topbackleft",
    "headpiece_bottomfrontright",
    "headpiece_bottombackright",
    "headpiece_bottombackleft",
    "headpiece_bottomfrontleft",
)

ACTIONS = Enum(
    "none",
    "sniff",
    "sniffgenital",
    "attack",
    "rear",
    "sniffbody",
    "approach",
    "sniffface",
    "mount",
    "escape",
    "reciprocalsniff",
    "defend",
    "selfgroom",
    "dig",
    "climb",
    "chase",
    "intromit",
    "avoid",
    "dominancemount",
    "dominance",
    "huddle",
    "disengage",
    "follow",
    "rest",
    "attemptmount",
    "shepherd",
    "flinch",
    "chaseattack",
    "tussle",
    "freeze",
    "exploreobject",
    "submit",
    "run",
    "dominancegroom",
    "genitalgroom",
    "allogroom",
    "biteobject",
    # "ejaculate",
)

# SNIFFALL experiment: a merged "sniff-family" behavior (sniff + all sniff subtypes).
# Appended as a NEW action id (37) ONLY when env SNIFFALL is set, so default 37-action
# runs/checkpoints are byte-for-byte unchanged. Its per-frame label is synthesized as
# the OR of the sniff-family channels in MultiTaskPerLabModel._labels37 (train_perlab_heads.py);
# routed to its own head + its own readout slot, so no existing action is corrupted.
if os.environ.get("SNIFFALL"):
    ACTIONS.values.append("sniffall")
    ACTIONS.value_to_idx["sniffall"] = len(ACTIONS.values) - 1

LABS = Enum(
    "AdaptableSnail",
    "BoisterousParrot",
    "CRIM13",
    "CalMS21_supplemental",
    "CalMS21_task1",
    "CalMS21_task2",
    "CautiousGiraffe",
    "DeliriousFly",
    "ElegantMink",
    "GroovyShrew",
    "InvincibleJellyfish",
    "JovialSwallow",
    "LyricalHare",
    "MABe22_keypoints",
    "MABe22_movies",
    "NiftyGoldfinch",
    "PleasantMeerkat",
    "ReflectiveManatee",
    "SparklingTapir",
    "TranquilPanther",
    "UppityFerret",
)

# The five labs that annotated plain `sniff` AND at least one subtype, so a merged
# "sniffall" head is well defined for them. UppityFerret is deliberately excluded: it
# annotated only sniffgenital/reciprocalsniff and never plain sniff, so merging would turn
# genuine un-annotated sniff frames into negatives.
SNIFFALL_LABS = ["CautiousGiraffe", "GroovyShrew", "InvincibleJellyfish",
                 "NiftyGoldfinch", "TranquilPanther"]
SNIFF_FAMILY = ["sniff", "sniffface", "sniffbody", "sniffgenital", "reciprocalsniff"]

# The 10 self-directed behaviours (agent == target); everything else is cross-directed.
SELF_DIRECTED = {"selfgroom", "dig", "climb", "rear", "rest", "run",
                 "biteobject", "exploreobject", "freeze", "huddle"}

TRAIN_ONLY_LABS = [
    "CRIM13",
    "CalMS21_supplemental",
    "CalMS21_task1",
    "CalMS21_task2",
    "MABe22_keypoints",
    "MABe22_movies",
]


def get_configs():
    global_seed = 123456789
    configs = [
        {
            "name": "11fps_4bp",
            "sample_rate": 11,
            "num_bodyparts": 4,
            "max_time_dilation": 1.8,
            "max_scale": 1.6,
            "noise_scale": 3.5,
            "aggregation_radius": 125,
        },
        {
            "name": "15fps_5bp",
            "sample_rate": 15,
            "num_bodyparts": 5,
            "max_time_dilation": 2.0,
            "max_scale": 2.0,
            "noise_scale": 3.0,
            "aggregation_radius": 150,
        },
        {
            "name": "19fps_6bp",
            "sample_rate": 19,
            "num_bodyparts": 6,
            "max_time_dilation": 1.7,
            "max_scale": 1.5,
            "noise_scale": 2.5,
            "aggregation_radius": 175,
        },
        {
            "name": "23fps_7bp",
            "sample_rate": 23,
            "num_bodyparts": 7,
            "max_time_dilation": 1.5,
            "max_scale": 1.7,
            "noise_scale": 2.0,
            "aggregation_radius": 100,
        },
        {
            "name": "27fps_6bp",
            "sample_rate": 27,
            "num_bodyparts": 6,
            "max_time_dilation": 1.4,
            "max_scale": 1.9,
            "noise_scale": 1.5,
            "aggregation_radius": 125,
        },
    ]
    for config_idx, config in enumerate(configs):
        config["split_seed"] = [0, config_idx, global_seed]
        config["pretrain_seed"] = [1, config_idx, global_seed]
        config["train_seed"] = [2, config_idx, global_seed]
        config["eval_seed"] = [3, config_idx, global_seed]

    return {config["name"]: config for config in configs}


# ------------------------------------------------------------------ per-lab head table

def lab_action_table(thresholds_pkl=None, sniffall=None):
    """The ordered (lab, action) list the per-lab head's output columns correspond to.

    Mirrors `train_perlab_heads.LAB_ACTION`: the keys of the trained-threshold pickle,
    minus PleasantMeerkat's non-`follow` heads (punctate annotation artifacts -- see
    pm_rule.py), plus a merged `sniffall` head for the five labs that split sniff into
    subtypes. Sorted, because the column order is what the checkpoint was trained with.
    """
    import pickle

    from .pm_rule import pm_action_ok

    if thresholds_pkl is None:
        from . import paths
        thresholds_pkl = paths.models_dir() / "thresholds.pkl"
    with open(thresholds_pkl, "rb") as f:
        th = pickle.load(f)
    keys = [k for k in th.keys() if pm_action_ok(*k)]
    if sniffall is None:
        sniffall = bool(os.environ.get("SNIFFALL"))
    if sniffall:
        keys = set(keys) | {(lab, "sniffall") for lab in SNIFFALL_LABS}
    return sorted(keys)


def build_P(lab_action=None, thresholds_pkl=None):
    """P[lab, action, head] = 1 iff `head` is the column for that (lab, action) pair.

    Used to project the per-lab head outputs back into action space for one lab, which is
    how a single readout serves every lab's own behaviour definitions.
    """
    lab_action = lab_action_table(thresholds_pkl) if lab_action is None else lab_action
    P = np.zeros((len(LABS), len(ACTIONS), len(lab_action)), dtype="float32")
    for j, (lab, action) in enumerate(lab_action):
        # Index through value_to_idx, NOT LABS.encode(). The inference driver forces every
        # video onto one lab's embedding by replacing `LABS.encode` with a constant
        # (run_allbehaviors_perlab.main). Building P through that patched encoder would put
        # every lab's head in a single row, and predict()'s einsum would then SUM all 82
        # heads into each action instead of selecting one -- yielding probabilities above 1.
        # value_to_idx is a plain dict and is never patched.
        P[LABS.value_to_idx[lab], ACTIONS.value_to_idx[action], j] = 1.0
    return P
