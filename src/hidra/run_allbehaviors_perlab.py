"""All-classifiers x ALL-behaviors inference with the per-lab-HEAD ensemble.

For a given classifier lab L (embedding + its per-lab heads), produce probability
tracks for EVERY behavior L has a head for -- self-directed AND cross-directed --
on EVERY (agent,target) pair of EVERY video, regardless of what that video's data
lab annotated. This is what the existing real-label inference could NOT give: it
only stored behaviors the data lab annotated.

Fix vs autism_predict_per_lab.py's CrossOnlyAllActionsLabels (which dropped the 10
self-directed behaviors): AllBehaviorLabels enumerates BOTH self pairs (a,a) with
the self-directed actions and cross pairs (a,t) with the cross-directed actions,
so Predictions.update routes each action to the correct (agent,target) key.

Model: MultiTaskPerLabModel + {config}_supervised_perlab.pkl (the Y experiment),
run with lab_id overridden to L (selects L's embedding AND L's heads via P). Only
L's heads are non-zero; output is filtered to L's heads at save time.

Datasets (--dataset): test | autism | multiday. Writes NEW stores (does not touch
the shared-model per_lab pkls used by the HMM/bout analyses).

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_allbehaviors_perlab.py --dataset autism --embedding-lab GroovyShrew
  CUDA_VISIBLE_DEVICES=0 python run_allbehaviors_perlab.py --dataset test --embedding-lab GroovyShrew --shard 0/2
  ... --smoke N  limits to the first N videos (validation).
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd


from . import paths, solution
from .run_test_probs_perlab import predict_into   # MultiTaskPerLabModel + perlab ckpt


def _predict_into_torch(config, predictions, num_epochs, dtype):
    """The PyTorch backend's equivalent of run_test_probs_perlab.predict_into.

    Same dataset, same Predictions accumulator, same per-config averaging -- only the
    model changes. HIDRA_PERLAB_CKPT is honoured by load_perlab, so fine-tuned weights
    thread through exactly as on the JAX path.
    """
    import torch

    from .torch import load_perlab, load_unsupervised
    from .torch.infer import predict_into as torch_predict_into

    tdtype = {"float32": torch.float32, "float64": torch.float64,
              "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trunk = load_unsupervised(config["name"], dtype=tdtype, device=device, config=config)
    head = load_perlab(config["name"], trunk, dtype=tdtype, device=device, config=config)
    torch_predict_into(config, predictions, head, num_epochs=num_epochs, device=device)


BACKENDS = {"jax": predict_into, "torch": _predict_into_torch}

# Researcher-private analysis trees. Only meaningful in a source checkout with the
# original dataset layout; `hidra predict` uses the "custom" dataset entry instead.
PROJECT_DIR = str(paths.dataset_dir().parent)

# 10 self-directed behaviors (agent==target); everything else is cross-directed.
# Derived empirically from the existing per-lab probs (clean split, no ambiguity).
SELF_DIRECTED = {"selfgroom", "dig", "climb", "rear", "rest", "run",
                 "biteobject", "exploreobject", "freeze", "huddle"}

from .pm_rule import pm_action_ok
TH = pickle.load(open(os.path.join(str(paths.models_dir()), "thresholds.pkl"), "rb"))
LAB_HEADS = {}
for (lab, act) in TH:
    if not pm_action_ok(lab, act):             # drop PleasantMeerkat attack/chase/escape heads
        continue
    LAB_HEADS.setdefault(lab, set()).add(act)
COMPETITION_LABS = sorted(LAB_HEADS)            # 15 labs

# SNIFFALL experiment: the 5 labs that split sniff into subtypes each get a merged
# "sniffall" head (from the namespaced *_supervised_perlab_sniffall.pkl model). Only
# this head is saved (to {lab}_sniffall.pkl); the existing perlab_allbeh stores are
# left untouched. UppityFerret is intentionally NOT here (see train_perlab_heads.py).
SNIFFALL = bool(os.environ.get("SNIFFALL"))
SNIFFALL_LABS = ["CautiousGiraffe", "GroovyShrew", "InvincibleJellyfish",
                 "NiftyGoldfinch", "TranquilPanther"]
if SNIFFALL:
    for L in SNIFFALL_LABS:
        LAB_HEADS[L].add("sniffall")

# Cross-lab matrices: behaviors annotated by >=5 labs, and the data (column) labs.
import collections as _c
_by_act = _c.defaultdict(set)
for (lab, act) in TH:
    _by_act[act].add(lab)
COMMON_BEHAVIORS = sorted(a for a, l in _by_act.items() if len(l) >= 5)   # attack,chase,escape,rear,sniff,sniffgenital
TRAIN_DATA_LABS = sorted({l for a in COMMON_BEHAVIORS for l in _by_act[a]}) + \
    ["BoisterousParrot", "CRIM13", "CalMS21_task1", "CalMS21_supplemental", "CalMS21_task2"]
# CalMS21_task2 added 2026-07-17 so its videos get cross-lab inference too (it was previously excluded
# from the inference set; its tracking+annotation parquets are present under data/{train_tracking,
# train_annotation}/CalMS21_task2/). Its sniff/attack videos pass the _has_common_behavior filter.


def _has_common_behavior(bl):
    import json as _j
    try:
        acts = {t.split(",")[-1].strip() for t in _j.loads(bl)}
    except Exception:
        return False
    return bool(acts & set(COMMON_BEHAVIORS))


# CalMS21_supplemental videos already scored under the multiday packaging (identical
# tracking); their per-lab-head predictions are recycled, so skip them in fresh inference.
_RECYCLED_TRAIN_IDS = set()
_rj = os.path.join(PROJECT_DIR, "analysis_outputs", "train_predictions", "multiday_to_train_supp.json")
if os.path.isfile(_rj):
    import json as _jj
    _RECYCLED_TRAIN_IDS = {int(v) for v in _jj.load(open(_rj)).values()}


DATASETS = {
    "test":     dict(dir=os.path.join(PROJECT_DIR, "data"), csv="TEST.csv", mode="test",
                     out=os.path.join(PROJECT_DIR, "analysis_outputs", "test_predictions", "perlab_allbeh")),
    "train":    dict(dir=os.path.join(PROJECT_DIR, "data"), csv="TRAIN.csv", mode="train",
                     out=os.path.join(PROJECT_DIR, "analysis_outputs", "train_predictions", "perlab_allbeh")),
    "autism":   dict(dir=os.path.expanduser("~/data/doom-autism/dataset"), csv="autism.csv", mode="autism",
                     out=os.path.expanduser("~/data/doom-autism/predictions/perlab_allbeh")),
    "multiday": dict(dir=os.path.expanduser("~/data/doom-multiday/dataset"), csv="multiday.csv", mode="multiday",
                     out=os.path.expanduser("~/data/doom-multiday/predictions/perlab_allbeh")),
    "sdannce":  dict(dir=os.path.expanduser("~/data/doom-sdannce/dataset"), csv="sdannce.csv", mode="sdannce",
                     out=os.path.expanduser("~/data/doom-sdannce/predictions/perlab_allbeh")),
    "sebastian": dict(dir=os.path.join(PROJECT_DIR, "data"), csv="sebastian.csv", mode="sebastian",
                      out=os.path.join(PROJECT_DIR, "analysis_outputs", "sebastian_predictions", "perlab_allbeh")),
    # MARS multi-annotator: tomomi's 10 physical videos (pose is shared across annotator copies),
    # consolidated from the mars_ft TEST/TRAIN split. Used to score canonical heads vs human annotators.
    "mars":     dict(dir="/dev/shm/mars_canon/data", csv="mars.csv", mode="mars",
                     out="/dev/shm/mars_canon/predictions/perlab_allbeh"),
    # SLEAP-training-budget pose-degradation sweep on CalMS21_task1 (same 101 videos/GT; clean vs
    # pose from SLEAP trained on 100/1000/10000 frames). All share GT data/train_annotation/CalMS21_task1.
    "calms_orig":    dict(dir=os.path.join(PROJECT_DIR, "data"), csv="calms_orig.csv", mode="train",
                          out="/dev/shm/calms_orig/predictions/perlab_allbeh"),
    "calms_n100":    dict(dir="/dev/shm/calms_n100", csv="calms.csv", mode="deg",
                          out="/dev/shm/calms_n100/predictions/perlab_allbeh"),
    "calms_n1000":   dict(dir="/dev/shm/calms_n1000", csv="calms.csv", mode="deg",
                          out="/dev/shm/calms_n1000/predictions/perlab_allbeh"),
    "calms_n10000":  dict(dir="/dev/shm/calms_n10000", csv="calms.csv", mode="deg",
                          out="/dev/shm/calms_n10000/predictions/perlab_allbeh"),
    "calms_n10":     dict(dir="/dev/shm/calms_n10", csv="calms.csv", mode="deg",
                          out="/dev/shm/calms_n10/predictions/perlab_allbeh"),
    "calms_n50":     dict(dir="/dev/shm/calms_n50", csv="calms.csv", mode="deg",
                          out="/dev/shm/calms_n50/predictions/perlab_allbeh"),
    "calms_n500":    dict(dir="/dev/shm/calms_n500", csv="calms.csv", mode="deg",
                          out="/dev/shm/calms_n500/predictions/perlab_allbeh"),
    # generic user dataset for predict.py (paths supplied via env)
    "custom":        dict(dir=os.environ.get("CUSTOM_DIR", ""), csv=os.environ.get("CUSTOM_CSV", "manifest.csv"),
                          mode=os.environ.get("CUSTOM_MODE", "custom"), out=os.environ.get("CUSTOM_OUT", "")),
}


class AllBehaviorLabels:
    """Self pairs (a,a) carry self-directed actions; cross pairs (a,t) carry
    cross-directed actions. Zeros label values (inference ignores them).
    Restricting each pair's action set to its directional type makes
    Predictions.update key every action correctly (self->(a,a), cross->(a,t))."""

    def __init__(self, num_frames, mouse_ids):
        self_acts = [solution.ACTIONS.encode(a) for a in SELF_DIRECTED]
        cross_acts = [i for i in range(1, len(solution.ACTIONS))
                      if i not in self_acts]
        self.label_index, self.label_masks, self.labeled_behaviors = [], [], []
        for a in mouse_ids:                                   # self pairs
            self.label_index.append((a, a))
            self.label_masks.append(self_acts)
            self.labeled_behaviors += [(a, a, aid) for aid in self_acts]
        for a in mouse_ids:                                   # cross pairs
            for t in mouse_ids:
                if t == a:
                    continue
                self.label_index.append((a, t))
                self.label_masks.append(cross_acts)
                self.labeled_behaviors += [(a, t, aid) for aid in cross_acts]
        self.num_labels = len(self.label_index)
        self.label_to_idx = {mp: i for i, mp in enumerate(self.label_index)}
        self._num_frames = num_frames

    def __getitem__(self, slices):
        assert isinstance(slices, tuple)
        fs = slices[0]
        n = 1 if isinstance(fs, int) else fs.stop - fs.start
        res = np.zeros([n, self.num_labels], dtype="int8")
        if len(slices) > 1 and slices[1] != slice(None, None, None):
            ls = slices[1]
            if isinstance(ls, tuple) and len(ls) == 3:
                ls = [ls]
            idx = [self.label_to_idx[i] if isinstance(i, tuple) else i for i in ls]
            res = res[:, idx]
        return res.squeeze(0) if isinstance(fs, int) else res


def load_videos(ds, shard, smoke):
    cfg = DATASETS[ds]
    csv = os.path.join(cfg["dir"], cfg["csv"])
    if not os.path.isfile(csv):                # tolerate TEST.csv vs test.csv
        alt = os.path.join(cfg["dir"], cfg["csv"].lower())
        csv = alt if os.path.isfile(alt) else csv
    df = pd.read_csv(csv).copy()
    df["mode"] = cfg["mode"]
    # (AdaptableSnail@25fps videos are now trained + predicted normally -- skip removed.)
    if ds == "train":                           # cross-lab data labs, videos with >=1 common behavior
        df = df.drop_duplicates("video_id")
        df = df[df["lab_id"].isin(TRAIN_DATA_LABS)]
        # BoisterousParrot annotates only 'shepherd' (no common behavior) but is a DooM
        # lab we want fully covered -> keep ALL its videos; other labs stay common-filtered.
        df = df[(df["lab_id"] == "BoisterousParrot") | df["behaviors_labeled"].apply(_has_common_behavior)]
        if not os.environ.get("NO_RECYCLE"):                 # NO_RECYCLE=1 -> infer CalMS21_supplemental
            df = df[~df["video_id"].isin(_RECYCLED_TRAIN_IDS)]   # FRESH (else recycled from multiday packaging)
    df = df.reset_index(drop=True)
    i, n = shard
    df = df.iloc[i::n]
    if smoke:
        df = df.iloc[:smoke]
    vids = []
    for idx, row in df.iterrows():
        tp = os.path.join(cfg["dir"], f"{cfg['mode']}_tracking", str(row["lab_id"]),
                          f"{int(row['video_id'])}.parquet")
        if not os.path.isfile(tp):
            continue
        vids.append(solution.create_video(idx, row))
    return vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--embedding-lab", required=True, choices=COMPETITION_LABS)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--backend", default=os.environ.get("HIDRA_BACKEND", "jax"),
                    choices=sorted(BACKENDS),
                    help="model backend: 'jax' (the original) or 'torch' (the port). "
                         "Also settable with $HIDRA_BACKEND.")
    args = ap.parse_args()
    run_config = BACKENDS[args.backend]

    ds = DATASETS[args.dataset]
    solution.dataset_dir = ds["dir"]
    # PERLAB_WORKDIR lets each parallel job own a private tracking-cache dir so
    # concurrent labs don't race-write the shared mode-keyed {working_dir}/{mode}/*.bin.
    solution.working_dir = os.environ.get("PERLAB_WORKDIR") or (
        os.path.join(os.path.dirname(ds["dir"]), "working")
        if args.dataset != "test" else solution.working_dir)
    # test.csv shim for create_video paths that expect lowercase
    if args.dataset == "test" and not os.path.isfile(f"{ds['dir']}/test.csv"):
        import shutil; shutil.copy(f"{ds['dir']}/TEST.csv", f"{ds['dir']}/test.csv")
    os.makedirs(ds["out"], exist_ok=True)
    os.makedirs(solution.working_dir, exist_ok=True)
    # The _sniffall checkpoints are now the canonical 90-head model: SNIFFALL runs
    # emit the FULL per-lab head set for EVERY lab (not just the 5 split labs'
    # sniffall head), into the primary {lab}.pkl store.
    i, n = map(int, args.shard.split("/"))
    tag = f"_{i}of{n}" if n > 1 else ""
    suff = f"_smoke{args.smoke}" if args.smoke else ""
    out_path = os.path.join(ds["out"], f"{args.embedding_lab}{tag}{suff}.pkl")
    if os.path.isfile(out_path) and not args.smoke:
        print(f"{out_path} exists, skipping"); return

    L = args.embedding_lab
    emb_idx = int(solution.LABS.encode(L))
    solution.LABS.encode = lambda v, _i=emb_idx: _i     # embedding+heads -> L; keep real lab_name
    heads = LAB_HEADS[L]   # ALL of lab L's heads (incl. sniffall for the 5 split labs)
    print(f"[{args.dataset}/{L}] {len(heads)} heads: {sorted(heads)}", flush=True)

    vids = load_videos(args.dataset, (i, n), args.smoke)
    for v in vids:
        v.labels = AllBehaviorLabels(num_frames=v.num_frames, mouse_ids=v.tracking_data.mouse_index)
    print(f"[{args.dataset}/{L}] {len(vids)} videos (all-pairs all-behaviors labels)", flush=True)

    preds = solution.Predictions(vids)
    configs = solution.get_configs()
    # HIDRA_CONFIGS: run a SUBSET of the 5-config ensemble (comma-separated config names).
    # Cheaper and lower quality -- for iterating on a single fine-tuned config, not for results.
    if os.environ.get("HIDRA_CONFIGS"):
        want = [c.strip() for c in os.environ["HIDRA_CONFIGS"].split(",") if c.strip()]
        bad = [c for c in want if c not in configs]
        assert not bad, f"HIDRA_CONFIGS: unknown config(s) {bad}; available: {list(configs)}"
        configs = {k: v for k, v in configs.items() if k in want}
        print(f"  WARNING: partial ensemble -- {len(configs)}/5 configs ({want})", flush=True)
    t0 = time.time()
    for ci, (cfg_name, cfg) in enumerate(configs.items()):
        tc = time.time()
        run_config(cfg, preds, args.epochs, args.dtype)
        print(f"  [{ci+1}/{len(configs)}] {cfg_name} {time.time()-tc:.0f}s "
              f"[{args.backend}]", flush=True)

    probs = preds.average_probs()
    keep = {k: v.astype("float16") for k, v in probs.items()
            if solution.ACTIONS.decode(k[3]) in heads}
    with open(out_path, "wb") as f:
        pickle.dump(keep, f)
    acts = sorted({solution.ACTIONS.decode(k[3]) for k in keep})
    self_present = sorted(set(acts) & SELF_DIRECTED)
    print(f"[{args.dataset}/{L}] saved {len(keep)} tracks, actions={acts}\n"
          f"   self-directed present: {self_present}\n"
          f"   {os.path.getsize(out_path)/1e6:.1f} MB in {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
