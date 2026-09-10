"""Experiment: per-(lab,behavior) classifier heads, MULTI-TASK with the shared head.

The shared model has one Linear(256 -> 37) head where lab info enters only via a
256-d embedding, so all labs' (e.g.) sniff annotations are averaged into one
shared sniff weight vector. This experiment adds 88 per-(lab,action) heads so
inter-lab annotation differences are NOT averaged out -- while keeping the
shared 37-head so the trunk still trains on ALL data (incl. the 6 train-only
labs that have no scored head).

Architecture (per config):
  frozen forecaster -> merge head -> + 0.1*lab_embedding -> 3x(BiLSTM+FFN) -> x
  x -> Linear(256->37)   shared head        (trained on ALL 8,790 train videos)
  x -> Linear(256->88)   per-lab heads       (trained on the 15 competition labs)
  loss = mean-BCE(shared, 37-space) + mean-BCE(per-lab, 88-space)
  INFERENCE / val-F1 / final predictions use the 88 per-lab heads.

Label routing: constant P (n_LABS, 37, 88), P[lab,action,j]=1 iff j==idx(lab,action).
  per-lab labels88 = labels37 @ P[lab];  mask88 = mask37 @ P[lab]
  predict prob37   = sigmoid(logits88) @ P[lab]^T  (reuses existing Predictions/F1)

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_perlab_heads.py --config 15fps_5bp
  python train_perlab_heads.py --config 15fps_5bp --smoke
"""
import argparse
import os
import pickle

import numpy as np

import jax
import jax.numpy as jnp
import pandas as pd

from . import paths, solution
from .pm_rule import pm_action_ok

MODELS_DIR = str(paths.models_dir())

TH = pickle.load(open(os.path.join(MODELS_DIR, "thresholds.pkl"), "rb"))
# Drop PleasantMeerkat attack/chase/escape heads (punctate artifacts; PM keeps only 'follow').
TH_KEYS = [k for k in TH.keys() if pm_action_ok(*k)]     # 85 (lab, action) after PM cleanup
LAB_ACTION = sorted(TH_KEYS)            # (lab, action), fixed order
N_HEADS = len(LAB_ACTION)

# --- SNIFFALL experiment (env SNIFFALL=1) --------------------------------------
# Add a merged "sniffall" head (union of sniff + all sniff subtypes) for the labs
# that split sniff into subtypes. UppityFerret is EXCLUDED (it annotated only
# sniffgenital/reciprocalsniff, never plain sniff -> merging would treat genuine
# un-annotated sniff frames as negatives). The 5 labs below annotated plain sniff
# AND >=1 subtype. The 5 non-splitting sniff labs (DeliriousFly, ElegantMink,
# JovialSwallow, LyricalHare, ReflectiveManatee) reuse their existing sniff head.
SNIFFALL = bool(os.environ.get("SNIFFALL"))
SNIFFALL_LABS = ["CautiousGiraffe", "GroovyShrew", "InvincibleJellyfish",
                 "NiftyGoldfinch", "TranquilPanther"]
SNIFF_FAMILY = ["sniff", "sniffface", "sniffbody", "sniffgenital", "reciprocalsniff"]
if SNIFFALL:
    assert "sniffall" in solution.ACTIONS.value_to_idx, "set env SNIFFALL before importing solution"
    LAB_ACTION = sorted(set(TH_KEYS) | {(l, "sniffall") for l in SNIFFALL_LABS})   # 85 + 5 = 90
    N_HEADS = len(LAB_ACTION)
    SNIFFALL_COL = np.array([1.0 if a == "sniffall" else 0.0 for (l, a) in LAB_ACTION], "float32")
    OLD_LAB_ACTION = sorted(TH_KEYS)        # existing-model column order (no sniffall)

# FROZEN_TRUNK=1: warm-start the whole model from the EXISTING trained perlab model,
# freeze everything except the per-lab head projection (out-proj-perlab), and fit only
# the new sniffall columns on the existing (frozen) trunk features. This keeps sniffall
# in the SAME feature space as every existing head, at ~10x less compute than a joint
# retrain. Fit/inference share the identical Linear.apply path (guaranteed consistent).
FROZEN_TRUNK = bool(os.environ.get("FROZEN_TRUNK"))
FT_STEPS = int(os.environ.get("FT_STEPS", "6000"))    # undiluted linear fit on frozen 256-d feats converges fast
# Optimizer schedule (stabilisation ablation): FT_LR sets the peak Adam LR (default 0.004, the historical
# constant). LR_COSINE_T>0 switches to a cosine decay FT_LR -> LR_FLOOR reached at step LR_COSINE_T (then
# held at the floor) -- the "decay the LR through the transition" lever for the 0.49<->0.25 bounce.
FT_LR = float(os.environ.get("FT_LR", "0.004"))
LR_COSINE_T = int(os.environ.get("LR_COSINE_T", "0"))
LR_FLOOR = float(os.environ.get("LR_FLOOR", "0.0"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))   # >0 -> decoupled AdamW-style decay
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "0.0"))         # >0 -> global-norm gradient clip


class SchedAdam(solution.Adam):
    """Adam with an optional cosine LR schedule, decoupled weight decay (AdamW), and global-norm grad
    clipping -- the stabilisation levers, all off by default (identical to solution.Adam then). lr(t) =
    floor + (lr0-floor)*0.5*(1+cos(pi*min(t,T)/T)); t is opt_state['t'] (per gradient step, from 1)."""
    def __init__(self, lr0, total=0, floor=0.0, weight_decay=0.0, grad_clip=0.0, **kw):
        super().__init__(lr0, **kw)
        self.lr0 = lr0; self.total = total; self.floor = floor
        self.weight_decay = weight_decay; self.grad_clip = grad_clip

    def _lr(self, t):
        if not self.total:
            return self.lr0
        frac = jnp.clip(t / self.total, 0.0, 1.0)
        return self.floor + (self.lr0 - self.floor) * 0.5 * (1.0 + jnp.cos(jnp.pi * frac))

    def update(self, params, opt_state, grads):
        t = opt_state["t"]
        lr = self._lr(t)
        if self.grad_clip:
            sq = sum(jnp.sum(g * g) for g in jax.tree.leaves(grads))
            gnorm = jnp.sqrt(sq) + 1e-12
            scale = jnp.minimum(1.0, self.grad_clip / gnorm)
            grads = jax.tree.map(lambda g: g * scale, grads)
        b1c = 1.0 / (1 - self.beta1 ** t); b2c = 1.0 / (1 - self.beta2 ** t)

        def upd_m(g, m):
            m["m1"] = m["m1"] + (g - m["m1"]) * (1 - self.beta1)
            m["m2"] = m["m2"] + (g ** 2 - m["m2"]) * (1 - self.beta2)
            return m

        def upd_p(p, m):
            m1 = m["m1"] * b1c; m2 = m["m2"] * b2c
            step = m1 / (jnp.sqrt(m2) + self.eps)
            if self.weight_decay:
                step = step + self.weight_decay * p
            return p - lr * step

        moments = jax.tree.map(upd_m, grads, opt_state["moments"])
        params = jax.tree.map(upd_p, params, moments)
        return params, {"t": t + 1, "moments": moments}

# SNIFFALL_SCRATCH=1 (requires SNIFFALL=1): train a FRESH model -- frozen self-supervised
# forecaster + a TRAINABLE supervised trunk + heads, trained from scratch -- whose only
# per-lab heads are the 5 sniffall columns. This mirrors how the sDANNCE cluster classifiers
# were trained (train_clusters.py): only the GNN forecaster is frozen; the whole supervised
# trunk (LSTM tail etc.) learns from scratch. FROZEN_TRUNK instead froze the entire supervised
# trunk and fit a single LINEAR readout of the 37-class features -- which cannot represent the
# sniff-family union (a fresh trainable trunk can). The shared out-proj(37) head is retained to
# give the trunk rich multi-behaviour signal. Trained on the 5 split labs' videos only.
SNIFFALL_SCRATCH = bool(os.environ.get("SNIFFALL_SCRATCH"))
SCRATCH_STEPS = int(os.environ.get("SCRATCH_STEPS", "20000"))
if SNIFFALL_SCRATCH:
    assert SNIFFALL, "SNIFFALL_SCRATCH requires SNIFFALL=1 (registers action 'sniffall')"
    LAB_ACTION = sorted((l, "sniffall") for l in SNIFFALL_LABS)   # 5 sniffall heads ONLY
    N_HEADS = len(LAB_ACTION)
    SNIFFALL_COL = np.ones(N_HEADS, "float32")

# REFIT="Lab,action": in-place refit of ONE existing head column on the frozen trunk, using
# only REFIT_VIDS videos (e.g. drop JovialSwallow zero-chase videos from the chase head).
# Warm-starts from the canonical _perlab_sniffall model (all 90 cols copied), freezes all but
# out-proj-perlab, masks loss to just this column, trains only on REFIT_VIDS. Spliced back
# afterward so every other head stays bit-identical.
REFIT = os.environ.get("REFIT")
if REFIT:
    assert SNIFFALL and not (FROZEN_TRUNK or SNIFFALL_SCRATCH), "REFIT needs SNIFFALL=1 only"
    _rl, _ra = [x.strip() for x in REFIT.split(",")]
    REFIT_IDX = LAB_ACTION.index((_rl, _ra))
    REFIT_COL = np.zeros(N_HEADS, "float32"); REFIT_COL[REFIT_IDX] = 1.0
    REFIT_VIDS = set(int(v) for v in os.environ["REFIT_VIDS"].split(",") if v.strip())
    FT_STEPS = int(os.environ.get("FT_STEPS", "3000"))

# SCALE="Lab,action": training-set-size scaling experiment. Like REFIT (frozen canonical
# trunk, loss masked to ONE head column, train on only SCALE_VIDS videos) but the target
# head column is RE-INITIALISED FRESH (not warm-started) so it must learn purely from the
# SCALE_VIDS subset -- this is what makes the F1-vs-data curve meaningful (a warm-started
# head would already know the behaviour from the full data and the curve would be flat).
# The frozen supervised trunk is a fixed feature extractor; SCALE measures how the linear
# readout's F1 scales with the number of labelled videos/frames it is fit on.
# SCALE_TAG gives each (pair,size,rep) job unique ckpt + output paths so many run in parallel.
SCALE = os.environ.get("SCALE")
if SCALE:
    assert SNIFFALL and not (FROZEN_TRUNK or SNIFFALL_SCRATCH or REFIT), "SCALE needs SNIFFALL=1 only"
    _sl, _sa = [x.strip() for x in SCALE.split(",")]
    REFIT = SCALE                                    # reuse every REFIT code path below
    REFIT_IDX = LAB_ACTION.index((_sl, _sa))
    REFIT_COL = np.zeros(N_HEADS, "float32"); REFIT_COL[REFIT_IDX] = 1.0
    REFIT_VIDS = set(int(v) for v in os.environ["SCALE_VIDS"].split(",") if v.strip())
    FT_STEPS = int(os.environ.get("FT_STEPS", "4000"))
    SCALE_TAG = os.environ.get("SCALE_TAG", f"{_sl}_{_sa}_n{len(REFIT_VIDS)}")

# LOLO_EXCLUDE="LabName": leave-one-lab-out FOUNDATION. Ordinary full supervised training
# (all layers trainable, incl. the shared feature-merge), but with ALL of the named lab's videos
# removed from train+val -> the shared merge (x0) never sees that lab. Output goes to a distinct
# _LOLO-{lab} file (never the canonical). Used to simulate the model a brand-new lab downloads,
# on top of which we then train that lab's own tail+head (new-lab adaptation).
LOLO_EXCLUDE = os.environ.get("LOLO_EXCLUDE")
if LOLO_EXCLUDE:
    assert SNIFFALL and not (FROZEN_TRUNK or SNIFFALL_SCRATCH or REFIT or SCALE), "LOLO is a normal-training variant; SNIFFALL=1 only"

# LABTAIL="Lab": new-lab ADAPTATION. Freeze the SSL backbone + feature-merge (x0); warm-start
# every layer from LABTAIL_SRC (default the canonical 90-head model, or a LOLO foundation), then
# train ONLY the per-lab TAIL (lab-embedding + the 3 BiLSTM/FF blocks) + out-proj-perlab, on this
# lab's videos, supervising just this lab's head columns (nll88 masked to them). Early-stopped.
#   LABTAIL_VIDS  = optional subset of the lab's train videos (size sweep); default = all.
#   CURATE_ACTION,CURATE_VIDS = supervise that one action ONLY on those videos (e.g. drop the
#                    zero-chase JovialSwallow videos from the chase column).
#   LABTAIL_FRESH=1 = re-initialise the tail+head instead of warm-starting (cold-start / Exp 1).
#   LABTAIL_SRC   = model file to warm-start from (default canonical _perlab_sniffall).
#   LABTAIL_TAG   = unique suffix for ckpt/output.
LABTAIL = os.environ.get("LABTAIL")
# SKIP_PATH=1: give the tail a trainable low-rank view of the PRE-merge SSL features (skip path #3b).
# The shared supervised feature-merge deletes lab information (it never saw the new lab). A fresh
# low-rank projection skip-proj-in (2*feat_dim -> SKIP_RANK, per-node) -> sum over nodes -> skip-proj-out
# (SKIP_RANK -> d_res) is ADDED to x0, letting the per-lab tail recover pre-merge information at transfer
# time WITHOUT a foundation retrain. Requires the LIVE SSL path (no CACHED_X0_DIR). The two skip layers
# are NEW (fresh, trainable); SSL + merge stay frozen. Tests whether the merge information-loss is
# rescuable downstream. SKIP_RANK sets the bottleneck rank r (default 64).
SKIP_PATH = os.environ.get("SKIP_PATH") == "1"
SKIP_RANK = int(os.environ.get("SKIP_RANK", "64"))
SKIP_LAYERS = {"skip-proj-in", "skip-proj-out"}
# FILM=1 (strategy #1): per-dataset input adapter in front of the merger. A per-lab affine (gamma,beta,
# zero-init -> identity) modulates the PRE-merge SSL features before ff-merge-in, trained JOINTLY with the
# foundation. Hypothesis: the shared merge learns to expect per-dataset-normalized input, so a brand-new
# lab needs only to fit its cheap affine (+ head/tail) while the merge stays frozen -> good x0. Needs the
# live SSL path. Active in both foundation (LOLO) training and adaptation (LABTAIL, new lab's row fresh).
FILM = os.environ.get("FILM") == "1"
FILM_LAYERS = {"film-gamma", "film-beta"}
# CORAL=1 (strategy #2): dataset-invariance pressure on the merger OUTPUT (x0). Adds deep-CORAL (+mean)
# alignment of each training lab's x0 first/second moments to the pooled moments, as a foundation-training
# regularizer (loss += CORAL_LAMBDA * coral). Hypothesis: an invariant merge doesn't overfit per-lab
# idiosyncrasies, so a held-out lab's x0 is already in-distribution. Only affects foundation training.
CORAL = os.environ.get("CORAL") == "1"
CORAL_LAMBDA = float(os.environ.get("CORAL_LAMBDA", "1.0"))
CORAL_MIN_FRAMES = int(os.environ.get("CORAL_MIN_FRAMES", "128"))
# FND_TAG: suffix on the LOLO foundation filename so film/coral/control variants don't overwrite the vanilla.
FND_TAG = os.environ.get("FND_TAG", "")
TAIL_TRAINABLE = {"lab-embedding", "lstm-0", "lstm-1", "lstm-2", "out-proj-0", "out-proj-1",
                  "out-proj-2", "ff-0-in", "ff-0-out", "ff-1-in", "ff-1-out", "ff-2-in", "ff-2-out",
                  "out-proj-perlab"}
if LABTAIL:
    assert SNIFFALL and not (FROZEN_TRUNK or SNIFFALL_SCRATCH or REFIT or SCALE or LOLO_EXCLUDE), "LABTAIL: SNIFFALL=1 only"
    # LABTAIL_ACTIONS="a,b" restricts supervision to those behaviours of the lab (Exp 1 = a lab's
    # FIRST classifier -> a single behaviour). Empty => all of the lab's heads (Exp 2 full head).
    LABTAIL_ACTIONS = [a for a in os.environ.get("LABTAIL_ACTIONS", "").split(",") if a.strip()]
    if LABTAIL_ACTIONS:
        LABTAIL_COLS = np.array([1.0 if (l == LABTAIL and a in LABTAIL_ACTIONS) else 0.0
                                 for (l, a) in LAB_ACTION], "float32")
    else:
        LABTAIL_COLS = np.array([1.0 if l == LABTAIL else 0.0 for (l, a) in LAB_ACTION], "float32")
    LABTAIL_VIDS = set(int(v) for v in os.environ.get("LABTAIL_VIDS", "").split(",") if v.strip())
    # LABTAIL_TRIM="vid:T": sub-video training set = the CONTIGUOUS chunk [0,T) frames of that one video
    # (for size points below one video's worth of positives). Implemented by truncating the Video's
    # num_frames to T so window generation tiles ONLY [0,T) -- no masked-window dilution, and it mirrors
    # scale_probe's [0,T) contiguous-chunk semantics. The manifest picks T so [0,T) holds ~target positives.
    _trim = os.environ.get("LABTAIL_TRIM", "").strip()
    if _trim:
        _tv, _tn = _trim.split(":"); LABTAIL_TRIM_VID, LABTAIL_TRIM_N = int(_tv), int(_tn)
    else:
        LABTAIL_TRIM_VID, LABTAIL_TRIM_N = -1, 0
    LABTAIL_FRESH = os.environ.get("LABTAIL_FRESH") == "1"
    # LABTAIL_HEAD_ONLY=1: freeze the (warm-started) tail and train ONLY out-proj-perlab -- Exp-2
    # "linear-only" add of a new behaviour on top of a lab's already-trained tail (cannot degrade the
    # lab's existing heads). Default (0) trains the whole tail+head.
    LABTAIL_HEAD_ONLY = os.environ.get("LABTAIL_HEAD_ONLY") == "1"
    # LABTAIL_EMB_ONLY=1: keep the SHARED tail blocks (LSTM/FF) frozen at the LOLO foundation's values
    # and train ONLY a new lab-embedding + the head -- tests whether lab-specificity via the embedding
    # alone (the canonical design) suffices for a new lab, without retraining the shared tail.
    LABTAIL_EMB_ONLY = os.environ.get("LABTAIL_EMB_ONLY") == "1"
    # layers trained under LABTAIL (fresh re-init also applies to exactly this set):
    if LABTAIL_HEAD_ONLY:
        LABTAIL_TRAIN = {"out-proj-perlab"}
    elif LABTAIL_EMB_ONLY:
        LABTAIL_TRAIN = {"lab-embedding", "out-proj-perlab"}
    else:
        LABTAIL_TRAIN = set(TAIL_TRAINABLE)
    # LABTAIL_TUNE_MERGE=1: ALSO fine-tune the supervised feature-merge (ff-merge-in/out, feat-flat-proj
    # -> x0) alongside the tail+head, on this lab's data. The LOLO merge never saw this lab, so its x0
    # may be a poor fit; adapting the merge lets x0 itself specialise. Requires the LIVE SSL+merge path
    # (no CACHED_X0_DIR -- x0 changes every step). The merge is WARM-STARTED from the foundation and
    # fine-tuned; only the tail+head are fresh-init'd under LABTAIL_FRESH.
    MERGE_LAYERS = {"ff-merge-in", "ff-merge-out", "feat-flat-proj"}
    LABTAIL_TUNE_MERGE = os.environ.get("LABTAIL_TUNE_MERGE") == "1"
    if LABTAIL_TUNE_MERGE:
        assert not os.environ.get("CACHED_X0_DIR"), "LABTAIL_TUNE_MERGE needs the live SSL+merge path; unset CACHED_X0_DIR"
        LABTAIL_TRAIN = set(LABTAIL_TRAIN) | MERGE_LAYERS
    if SKIP_PATH:
        assert not os.environ.get("CACHED_X0_DIR"), "SKIP_PATH needs the live SSL path (pre-merge feats); unset CACHED_X0_DIR"
        LABTAIL_TRAIN = set(LABTAIL_TRAIN) | SKIP_LAYERS
    if FILM:                                                  # adapt the new lab's affine (warm-copied identity row)
        assert not os.environ.get("CACHED_X0_DIR"), "FILM needs the live SSL path (modulates pre-merge feats); unset CACHED_X0_DIR"
        LABTAIL_TRAIN = set(LABTAIL_TRAIN) | FILM_LAYERS
    # FILM layers are warm-copied from the foundation (the new lab's row = identity, never fresh-reinit'd):
    LABTAIL_FRESH_LAYERS = set(LABTAIL_TRAIN) - MERGE_LAYERS - FILM_LAYERS   # never re-init warm merge/film
    # DONOR_LAB="Lab": seed this new lab's (untrained) embedding row + head column(s) from a pose-similar
    # DONOR lab already trained in the LOLO foundation, instead of a fresh/empty init. Reduces what must
    # be learned from the new lab's few videos. Applied AFTER any fresh re-init, on the trainable slots.
    DONOR_LAB = os.environ.get("DONOR_LAB")
    LABTAIL_TAG = os.environ.get("LABTAIL_TAG", LABTAIL + ("_fresh" if LABTAIL_FRESH else "_warm"))
    CURATE_ACTION = os.environ.get("CURATE_ACTION")
    CURATE_COL = LAB_ACTION.index((LABTAIL, CURATE_ACTION)) if CURATE_ACTION else -1
    CURATE_VIDS = np.array(sorted(int(v) for v in os.environ.get("CURATE_VIDS", "").split(",") if v.strip()), "int64")
    FT_STEPS = int(os.environ.get("FT_STEPS", "8000"))
    # CACHED_X0_DIR: skip the SSL backbone + feature-merge during training by reading a precomputed
    # LOLO-merge x0 cache ({lab}_{config}.pkl of {(vid,ag,tg):(nf,256)}); the tail+head train on
    # cached x0 (Exp 1 fast path). Geometric augmentation is a no-op on precomputed x0.
    CACHED_X0_DIR = os.environ.get("CACHED_X0_DIR")
    # CACHED_PREMERGE_DIR: skip ONLY the SSL backbone (not the merge) by reading precomputed PRE-merge
    # feats (regen_premerge_perlab.py), per-video shards {dir}/{lab}_{config}/{vid}.pkl of
    # {(vid,ag,tg):(nf, n_bp*Din) fp16}. The trainable merge is applied LIVE on the cached feats, so this
    # is the fast path for the MERGE-INCLUSIVE size sweep (needs LABTAIL_TUNE_MERGE; only the lab's
    # LABTAIL_VIDS shards are loaded, keeping RAM bounded). Geometric augmentation is a no-op on cached feats.
    CACHED_PREMERGE_DIR = os.environ.get("CACHED_PREMERGE_DIR")
    if CACHED_PREMERGE_DIR:
        assert LABTAIL_TUNE_MERGE, "CACHED_PREMERGE_DIR requires LABTAIL_TUNE_MERGE=1 (merge applied live on cached feats)"
        assert not CACHED_X0_DIR, "CACHED_PREMERGE_DIR and CACHED_X0_DIR are mutually exclusive"
    LABTAIL_OUTDIR = os.environ.get("LABTAIL_OUTDIR")   # durable output dir (default /dev/shm)
else:
    CACHED_X0_DIR = None
    CACHED_PREMERGE_DIR = None

# DISENTANGLE=1 (+ DISENTANGLE_AE=path to models/{config}_disentangle_ae.pkl): route the classifier
# trunk through the FROZEN per-config disentangling encoder -> the per-lab tail sees z_dyn (lab-invariant)
# instead of x0. Freeze SSL + feature-merge + enc_dyn; train the tail (TAIL_TRAINABLE) + a FRESH 128->256
# lift (disent-proj) + heads on ALL competition labs, warm-started from the canonical sniffall model.
# Measures how much classifier performance lives in the setting-entangled subspace that z_dyn discards.
DISENTANGLE = bool(os.environ.get("DISENTANGLE"))
D_DYN = 128                                              # z_dyn width (constant across configs)
if DISENTANGLE:
    assert SNIFFALL and not (FROZEN_TRUNK or SNIFFALL_SCRATCH or REFIT or SCALE or LOLO_EXCLUDE or LABTAIL), \
        "DISENTANGLE: SNIFFALL=1 only"
    DISENT_TRAINABLE = set(TAIL_TRAINABLE) | {"disent-proj"}
    FT_STEPS = int(os.environ.get("FT_STEPS", "20000"))


def _gather_x0(element, cache, pad=32, dim=256):
    """Interpolate native-resolution cached x0 onto this window's in_len input-frame timestamps
    (mirrors scale_cache_validate.extract_cached); returns (in_len, dim). Cache miss -> zeros
    (masked out via batch_mask on padded elements anyway)."""
    in_len = int(element["agent"].shape[0])
    vid = int(element["video_id"]); ag = int(element["agent_id"]); tg = int(element["target_id"])
    nat = cache.get((vid, ag, tg))
    if nat is None:
        return np.zeros((in_len, dim), np.float32)
    t_start = float(element["t_start"]); t_end = float(element["t_end"]); fps = float(element["video_fps"])
    out_len = in_len - 2 * pad
    in_dur = (t_end - t_start) * (in_len / out_len); t_mid = (t_start + t_end) / 2
    input_ts = np.linspace(t_mid - in_dur / 2, t_mid + in_dur / 2, in_len)
    nat_ts = np.arange(nat.shape[0]) / fps
    j = np.clip(np.searchsorted(nat_ts, input_ts), 1, nat.shape[0] - 1); k = j - 1
    den = nat_ts[j] - nat_ts[k]; den[den == 0] = 1e-9
    fr = ((input_ts - nat_ts[k]) / den)[:, None]
    return (nat[k].astype(np.float32) * (1 - fr) + nat[j].astype(np.float32) * fr).astype(np.float32)


class _X0Dataset:
    """Wraps a solution.Dataset so each yielded element carries a precomputed x0 (gathered from the
    LOLO-merge cache), letting the trainer skip the SSL backbone + feature-merge. Delegates all other
    attributes to the base dataset so the Trainer treats it identically."""
    def __init__(self, base, cache):
        self._base = base; self._cache = cache

    def __getattr__(self, k):
        return getattr(self._base, k)

    def element_iterator(self):
        for el in self._base.element_iterator():
            el = dict(el)
            el["x0"] = _gather_x0(el, self._cache)
            yield el


class _PremergeDataset:
    """Like _X0Dataset but carries precomputed SSL PRE-merge feats (flattened n_bp*Din per frame),
    gathered onto each window; the trainable merge is applied downstream in _trunk_feats. Skips the SSL
    backbone only (merge stays live/trainable)."""
    def __init__(self, base, cache, dim):
        self._base = base; self._cache = cache; self._dim = dim

    def __getattr__(self, k):
        return getattr(self._base, k)

    def element_iterator(self):
        for el in self._base.element_iterator():
            el = dict(el)
            el["premerge"] = _gather_x0(el, self._cache, dim=self._dim)
            yield el


class _TruncVideo:
    """Exposes only the first `n` frames of a Video: reports num_frames=n (so the Dataset's window
    tiling covers ONLY [0,n)) while delegating tracking_data/labels/fps/video_id to the base video
    (whose arrays have >= n frames, so windowed reads within [0,n) are unaffected). Used by LABTAIL_TRIM
    to build a sub-video contiguous-chunk training set without loss-masking dilution."""
    def __init__(self, base, n):
        self._base = base
        self.num_frames = int(n)
        self.duration = self.num_frames / base.fps

    def __getattr__(self, k):
        return getattr(self._base, k)


def _load_disent_ae(config_name):
    """Per-config frozen disentangling encoder: models/{config}_disentangle_ae.pkl -> (mu, sd, enc, D_DYN).
    Instance-scoped (not module-global) so one process can hold different configs' encoders (fused eval)."""
    d = pickle.load(open(f"{MODELS_DIR}/{config_name}_disentangle_ae.pkl", "rb"))
    mu = jnp.asarray(np.asarray(d["mu"], np.float32)); sd = jnp.asarray(np.asarray(d["sd"], np.float32))
    enc = [(jnp.asarray(np.asarray(W, np.float32)), jnp.asarray(np.asarray(b, np.float32))) for (W, b) in d["enc_dyn"]]
    return mu, sd, enc, int(d.get("D_DYN", 128))
# ------------------------------------------------------------------------------

# run_test_probs_perlab.py / run_allbehaviors_perlab.py force one lab's embedding by
# REPLACING LABS.encode with a constant. Built through that patched encoder, P would assign
# every lab's head to a single row and predict()'s einsum would SUM all 82 heads into each
# action instead of selecting one -- probabilities above 1 and a meaningless cross-lab
# readout.
#
# The original guarded against this by capturing the encoders at import time, which works
# only while this module is imported *before* the patch is applied. That is a live hazard:
# the PyTorch backend imports the JAX engine lazily, so this module can now be imported
# after main() has already patched, and the capture would grab the patched function.
# Indexing value_to_idx instead removes the ordering dependency altogether -- the patch
# replaces the `encode` attribute and never touches the dict.
_LABS_ENCODE_ORIG = solution.LABS.value_to_idx.__getitem__
_ACTIONS_ENCODE_ORIG = solution.ACTIONS.value_to_idx.__getitem__


def build_P():
    P = np.zeros((len(solution.LABS), len(solution.ACTIONS), N_HEADS), dtype="float32")
    for j, (lab, act) in enumerate(LAB_ACTION):
        P[solution.LABS.value_to_idx[lab], solution.ACTIONS.value_to_idx[act], j] = 1.0
    return jnp.asarray(P)


class MultiTaskPerLabModel(solution.SupervisedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers["out-proj-perlab"] = solution.Linear(self.d_res, N_HEADS, self.dtype)
        if SKIP_PATH:                                            # low-rank view of PRE-merge feats -> add to x0
            raw_dim = self.layers["ff-merge-in"].input_dim       # 2*feat_dim (per-node)
            # normalize_input=False: the fresh running-stat calibration on few adapt batches is numerically
            # fragile (sqrt of a running variance can go negative -> NaN, defeating even the zero-gate since
            # 0*NaN=NaN). We standardize `raw` with a stateless per-token LayerNorm in _trunk_feats instead.
            self.layers["skip-proj-in"] = solution.Linear(raw_dim, SKIP_RANK, self.dtype,
                                                          batch_dims=[self.n_bp], normalize_input=False)
            self.layers["skip-proj-out"] = solution.Linear(SKIP_RANK, self.d_res, self.dtype, normalize_input=False)
        if FILM:                                                 # per-lab affine on pre-merge feats (strategy #1)
            raw_dim = self.layers["ff-merge-in"].input_dim       # gamma/beta zero-init -> identity (see _film_zero)
            self.layers["film-gamma"] = solution.Embedding(raw_dim, len(solution.LABS), self.dtype)
            self.layers["film-beta"] = solution.Embedding(raw_dim, len(solution.LABS), self.dtype)
        if DISENTANGLE:                                          # 128 -> d_res lift of z_dyn (fresh, trainable)
            self.layers["disent-proj"] = solution.Linear(D_DYN, self.d_res, self.dtype)
            self._dz = None                                      # (mu, sd, enc) set via set_disentangle()
        self.P = build_P()

    def set_disentangle(self, *args):
        """Load this config's frozen enc_dyn into the instance. config_name is the LAST arg: the
        set_variables-bound inference object injects `layers` as args[0], the raw model does not."""
        config_name = args[-1]
        mu, sd, enc, dd = _load_disent_ae(config_name)
        assert dd == D_DYN, f"z_dyn dim {dd} != {D_DYN}"
        self._dz = (mu, sd, enc)
        return self

    def _apply_disentangle(self, layers, x):
        """If DISENTANGLE: replace x0 with disent-proj(enc_dyn(x0)) so the tail sees z_dyn, not x0."""
        if not DISENTANGLE:
            return x
        assert self._dz is not None, "call model.set_disentangle(config_name) first"
        mu, sd, enc = self._dz
        z = (x.astype("float32") - mu) / sd
        for i, (W, b) in enumerate(enc):
            z = z @ W + b
            if i < len(enc) - 1:
                z = jax.nn.silu(z)
        return layers["disent-proj"].apply(z.astype(x.dtype))

    def _trunk_feats(self, layers, batch, key, return_x0=False):
        """Everything up to (not incl.) the output head -> (B, T, d_res). return_x0=True also returns the
        merge output x0 (in_len, B, d_res), lab-INDEPENDENT, for the CORAL invariance loss."""
        if CACHED_X0_DIR:                                       # fast path: read precomputed x0, skip SSL+merge
            x = jnp.transpose(batch["x0"].astype("float32"), (1, 0, 2))   # (B,in_len,256) -> (in_len,B,256)
        else:
            if CACHED_PREMERGE_DIR:                             # skip ONLY SSL: read cached PRE-merge feats,
                pm = batch["premerge"].astype("float32")       # apply the (trainable) merge live below.
                Bn, L, _ = pm.shape                            # (B, in_len, n_bp*Din)
                raw = jnp.transpose(pm.reshape(Bn, L, self.n_bp, -1), (1, 0, 2, 3))  # (in_len,B,n_bp,Din)
            else:
                xs = self.unsupervised_model.extract_features(batch, key)
                x = jnp.concat(xs, axis=-1)
                x_agent, x_target = jnp.split(x, 2, axis=1)
                raw = jnp.concat([x_agent, x_target], axis=-1)   # PRE-merge feats (..., n_bp, 2*feat_dim)
            if FILM:                                             # per-lab affine adapter in front of merge (#1)
                g = layers["film-gamma"].apply(batch["lab_id"])  # (B, raw_dim), zero-init -> identity
                b = layers["film-beta"].apply(batch["lab_id"])
                raw = raw * (1.0 + g[None, :, None, :]) + b[None, :, None, :]
            x = layers["ff-merge-in"].apply(raw)
            x = jax.nn.silu(x)
            x = layers["ff-merge-out"].apply(x)
            x = jnp.sum(x, axis=2)
            x = layers["feat-flat-proj"].apply(x)
            if SKIP_PATH:                                        # low-rank skip of pre-merge feats -> x0 (skip #3b)
                rn = raw.astype("float32")                       # stateless per-token LayerNorm (robust, no stats)
                rn = (rn - rn.mean(-1, keepdims=True)) / (rn.std(-1, keepdims=True) + 1e-3)
                s = layers["skip-proj-in"].apply(rn.astype(raw.dtype))   # (..., n_bp, r)
                s = jnp.sum(s, axis=2)                           # pool over nodes, mirroring the merge sum
                x = x + layers["skip-proj-out"].apply(s)         # (..., d_res), zero-gated at init, added to x0
        x0 = x                                                   # merge output (lab-independent) for CORAL
        x = self._apply_disentangle(layers, x)                  # x0 -> z_dyn lift (no-op unless DISENTANGLE)
        x += 0.1 * layers["lab-embedding"].apply(batch["lab_id"])
        for l in range(self.n_layers):
            y = layers[f"lstm-{l}"].apply(x)
            y = layers[f"out-proj-{l}"].apply(y)
            x += y
            y = layers[f"ff-{l}-in"].apply(x)
            y = jax.nn.silu(y)
            y = layers[f"ff-{l}-out"].apply(y)
            x += y
        x = jnp.transpose(x, (1, 0, 2))
        x = x[:, self.padding: x.shape[1] - self.padding]
        if return_x0:
            return x.astype("float32"), x0
        return x.astype("float32")

    def _coral_loss(self, x0, batch):
        """Deep-CORAL (+mean) invariance pressure on the merge output. Align each training lab's x0
        first/second moments to the pooled moments over valid frames. x0: (in_len, B, d)."""
        in_len, B, d = x0.shape
        feats = x0.astype("float32").reshape(in_len * B, d)     # (N, d)
        valid = (batch["batch_mask"] == 1).astype("float32")    # (B,)
        vf = jnp.broadcast_to(valid[None, :], (in_len, B)).reshape(-1)   # (N,)
        oh = jax.nn.one_hot(batch["lab_id"], len(solution.LABS))         # (B, L)
        ohf = jnp.broadcast_to(oh[None], (in_len, B, oh.shape[1])).reshape(-1, oh.shape[1]) * vf[:, None]  # (N,L)
        stride = max(1, (in_len * B) // 4096)                   # cap the (L,d,d) einsum cost; regularizer only
        if stride > 1:
            feats = feats[::stride]; vf = vf[::stride]; ohf = ohf[::stride]
        nL = ohf.sum(0)                                         # (L,) valid frames per lab
        # pooled moments over all valid frames
        wv = vf.sum()
        mu_p = (feats * vf[:, None]).sum(0) / jnp.maximum(wv, 1.0)
        cov_p = (feats * vf[:, None]).T @ feats / jnp.maximum(wv, 1.0) - jnp.outer(mu_p, mu_p)
        # per-lab moments via an explicit (N,L,d) weighting -> batched matmul (memory-safe: no (N,d,d))
        fw = feats[:, None, :] * ohf[:, :, None]               # (N, L, d), rows zeroed for other labs/invalid
        sum_l = fw.sum(0)                                       # (L, d) == einsum nl,ni->li
        mu_l = sum_l / jnp.maximum(nL[:, None], 1.0)
        S_l = jnp.einsum("nli,nj->lij", fw, feats)             # (L, d, d) uncentered 2nd moment (per-lab)
        cov_l = S_l / jnp.maximum(nL[:, None, None], 1.0) - jnp.einsum("li,lj->lij", mu_l, mu_l)
        w = (nL >= CORAL_MIN_FRAMES).astype("float32")         # only labs with enough frames this batch
        cov_term = ((cov_l - cov_p[None]) ** 2).sum((1, 2)) / (d * d)
        mean_term = ((mu_l - mu_p[None]) ** 2).sum(1) / d
        per_lab = cov_term + mean_term
        return (per_lab * w).sum() / jnp.maximum(w.sum(), 1.0)

    def _labels37(self, batch):
        self_labels = jax.nn.one_hot(batch["self_labels"], len(solution.ACTIONS))
        cross_labels = jax.nn.one_hot(batch["cross_labels"], len(solution.ACTIONS))
        labels = self_labels + cross_labels
        lm = batch["self_label_mask"] + batch["cross_label_mask"]
        mask = ((lm == 1) & ((batch["batch_mask"] == 1)[:, None])).astype("float32")
        if SNIFFALL:
            # sniffall channel = OR of the sniff-family channels (subtypes are mutually
            # exclusive under the argmax label, so the sum is in {0,1}; clip guards ties).
            sid = _ACTIONS_ENCODE_ORIG("sniffall")
            fam = [_ACTIONS_ENCODE_ORIG(a) for a in SNIFF_FAMILY]
            labels = labels.at[..., sid].set(jnp.clip(labels[..., fam].sum(-1), 0.0, 1.0))
            mask = mask.at[..., sid].set(jnp.clip(mask[..., fam].sum(-1), 0.0, 1.0))
        return labels, mask

    @staticmethod
    def _masked_bce(logits, labels, mask):
        lp = jnp.where(labels == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))
        lp = (lp * mask[:, None]).sum(axis=-1)
        w = mask.sum()
        nll = -lp.mean(axis=1).sum() / jnp.maximum(w, 1.0)
        return nll, w

    def compute_loss(self, layers, batch, key):
        if CORAL:
            x, x0 = self._trunk_feats(layers, batch, key, return_x0=True)
        else:
            x = self._trunk_feats(layers, batch, key)
        logits37 = layers["out-proj"].apply(x)
        logits88 = layers["out-proj-perlab"].apply(x)
        labels37, mask37 = self._labels37(batch)
        Pb = self.P[batch["lab_id"]]
        labels88 = jnp.einsum("bta,baj->btj", labels37, Pb)
        mask88 = jnp.einsum("ba,baj->bj", mask37, Pb)
        if FROZEN_TRUNK:
            # Fit ONLY the sniffall columns. Restricting mask88 to those columns removes the
            # ~90x gradient dilution of a single head inside the joint masked-BCE (the other
            # 88 heads are warm-started + frozen, so they need no gradient). Drop nll37 too
            # (the shared head is frozen and irrelevant here).
            mask88 = mask88 * jnp.asarray(SNIFFALL_COL)
        if REFIT:
            mask88 = mask88 * jnp.asarray(REFIT_COL)
        if LABTAIL:
            mask88 = mask88 * jnp.asarray(LABTAIL_COLS)           # supervise only this lab's heads
            if CURATE_ACTION:                                    # e.g. chase only on chase-present videos
                assert "video_id" in batch, "CURATE needs batch['video_id']"
                vid = batch["video_id"].astype("int64")
                ok = jnp.any(vid[:, None] == jnp.asarray(CURATE_VIDS)[None, :], axis=1).astype(mask88.dtype)
                mask88 = mask88.at[:, CURATE_COL].multiply(ok)
        nll37, w37 = self._masked_bce(logits37, labels37, mask37)   # all labs
        nll88, w88 = self._masked_bce(logits88, labels88, mask88)   # competition labs
        nll = nll88 if (FROZEN_TRUNK or REFIT or LABTAIL) else (nll37 + nll88)
        metrics = {"nll": (nll, w37 + w88), "nll37": (nll37, w37), "nll88": (nll88, w88)}
        if CORAL:                                                   # strategy #2: dataset-invariance pressure
            coral = self._coral_loss(x0, batch)
            nll = nll + CORAL_LAMBDA * coral
            metrics["coral"] = (coral, jnp.asarray(1.0))
            metrics["nll"] = (nll, w37 + w88)
        return (nll, metrics)

    def predict(self, layers, batch, key):
        x = self._trunk_feats(layers, batch, key)
        probs88 = jax.nn.sigmoid(layers["out-proj-perlab"].apply(x))
        Pb = self.P[batch["lab_id"]]
        return jnp.einsum("btj,baj->bta", probs88, Pb)             # 37-space for Predictions


def load_train_videos_robust():
    """Like solution.load_videos('train') but skips videos whose tracking
    parquet is missing from disk (1 known SparklingTapir file). Caches the
    built video list to working_dir so later configs reuse it."""
    cache = f"{solution.working_dir}/train/videos.pkl"
    if os.path.isfile(cache):
        return pickle.load(open(cache, "rb"))
    csv = os.path.join(solution.dataset_dir, "train.csv")
    if not os.path.isfile(csv):
        import shutil
        shutil.copy(os.path.join(solution.dataset_dir, "TRAIN.csv"), csv)
    df = pd.read_csv(csv); df["mode"] = "train"
    videos, skipped = [], 0
    for i, row in df.iterrows():
        p = os.path.join(solution.dataset_dir, "train_tracking",
                         str(row["lab_id"]), f"{int(row['video_id'])}.parquet")
        if not os.path.isfile(p):
            skipped += 1; continue
        videos.append(solution.create_video(i, row))
    print(f"built {len(videos)} train videos ({skipped} skipped: missing tracking)", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(videos, f)
    return videos


def train_perlab(config, smoke=False):
    # HIDRA_DATA_DIR: train from a dataset staged somewhere else (finetune.py writes
    # {dir}/TRAIN.csv + {dir}/train_{tracking,annotation}/{lab}/{video_id}.parquet).
    # Unset -> the in-repo data/ dir, i.e. the original competition layout.
    if os.environ.get("HIDRA_DATA_DIR"):
        solution.dataset_dir = os.path.abspath(os.environ["HIDRA_DATA_DIR"])
        print(f"[DATA] dataset_dir -> {solution.dataset_dir}", flush=True)
    # SCALE sweep: honor a private tracking-cache dir so parallel jobs (one per GPU lane) don't
    # race the shared mode-keyed {working_dir}/train/*.bin cache. Assign whole labs to a lane so
    # a lane rebuilds each video's .bin once and reuses it across that lab's size jobs.
    if os.environ.get("PERLAB_WORKDIR"):     # private tracking-cache dir (parallel jobs must not race the shared .bin)
        solution.working_dir = os.environ["PERLAB_WORKDIR"]
        os.makedirs(solution.working_dir, exist_ok=True)
    if os.environ.get("LABTAIL_SEED"):       # per-run seed: drives train data order + rotate/flip/noise aug
        config = {**config, "train_seed": [int(os.environ["LABTAIL_SEED"])]}   # reps = independent trajectories
        print(f"[SEED] train_seed overridden -> {config['train_seed']}", flush=True)
    videos = load_train_videos_robust()
    train_videos, val_videos = solution.split_videos(videos, validation_frac=0.15,
                                                      random_seed=config["split_seed"])
    val_videos = [v for v in val_videos if v.lab_name not in solution.TRAIN_ONLY_LABS]
    if LOLO_EXCLUDE:                                      # foundation that never sees the held-out lab
        n0 = len(train_videos)
        train_videos = [v for v in train_videos if v.lab_name != LOLO_EXCLUDE]
        val_videos = [v for v in val_videos if v.lab_name != LOLO_EXCLUDE]
        print(f"[LOLO] excluding {LOLO_EXCLUDE}: dropped {n0 - len(train_videos)} train videos "
              f"-> {len(train_videos)} train, {len(val_videos)} val", flush=True)
    if SNIFFALL_SCRATCH:                                  # fresh sniffall model: 5 split labs only
        train_videos = [v for v in train_videos if v.lab_name in SNIFFALL_LABS]
        val_videos = [v for v in val_videos if v.lab_name in SNIFFALL_LABS]
    if REFIT:
        train_videos = [v for v in videos if int(v.video_id) in REFIT_VIDS]
        val_videos = train_videos
        print(f"[REFIT {REFIT}] {len(train_videos)} videos: {sorted(int(v.video_id) for v in train_videos)}", flush=True)
    if LABTAIL:
        tr_ids = {int(v.video_id) for v in train_videos}; va_ids = {int(v.video_id) for v in val_videos}
        pool = [v for v in videos if v.lab_name == LABTAIL]
        train_videos = [v for v in pool if int(v.video_id) in tr_ids]
        val_videos = [v for v in pool if int(v.video_id) in va_ids]
        if LABTAIL_VIDS:                                  # LABTAIL_VIDS is authoritative for the train set
            train_videos = [v for v in pool if int(v.video_id) in LABTAIL_VIDS]   # may include globally-val vids
        if not val_videos:                                # guarantee an eval signal (may overlap train)
            val_videos = train_videos[-max(1, len(train_videos) // 6):]
        if LABTAIL_TRIM_VID >= 0:                             # sub-video: keep only [0,T) of the trim video
            def _trunc(vs):
                return [_TruncVideo(v, LABTAIL_TRIM_N) if int(v.video_id) == LABTAIL_TRIM_VID else v for v in vs]
            train_videos, val_videos = _trunc(train_videos), _trunc(val_videos)
            got = [v for v in train_videos if int(v.video_id) == LABTAIL_TRIM_VID]
            assert got, f"LABTAIL_TRIM video {LABTAIL_TRIM_VID} not in train set {[int(v.video_id) for v in train_videos]}"
            print(f"[LABTAIL_TRIM] video {LABTAIL_TRIM_VID}: using frames [0,{LABTAIL_TRIM_N}) "
                  f"({LABTAIL_TRIM_N / got[0].fps:.1f}s)", flush=True)
        print(f"[LABTAIL {LABTAIL}] {len(train_videos)} train, {len(val_videos)} val "
              f"(fresh={LABTAIL_FRESH}, curate={CURATE_ACTION} on {len(CURATE_VIDS)} vids)", flush=True)
    print(f"{len(train_videos)} train (all labs), {len(val_videos)} val (competition labs)", flush=True)

    common = dict(seq_len=64, sample_rate=config["sample_rate"], padding=32,
                  num_bodyparts=config["num_bodyparts"], unsupervised=False)
    # cached-x0 training skips SSL+merge, so geometric augmentation (rotate/flip/scale) and keypoint
    # noise are no-ops -> use deterministic eval-style windows (fixed in_len for clean batching).
    _tr_aug = (dict(max_scale=1, max_time_dilation=1, rotate=False, flip=False, noise_scale=0.0)
               if (CACHED_X0_DIR or CACHED_PREMERGE_DIR) else
               dict(max_scale=config["max_scale"], max_time_dilation=config["max_time_dilation"],
                    rotate=True, flip=True, noise_scale=config["noise_scale"]))
    train_dataset = solution.Dataset(
        videos=train_videos, num_epochs=100000, num_workers=8,
        seed=[0] + config["train_seed"], **_tr_aug, **common)
    val_dataset = solution.Dataset(
        videos=val_videos, num_epochs=1, max_scale=1, max_time_dilation=1,
        rotate=False, flip=False, noise_scale=config["noise_scale"],
        num_workers=8, seed=[1] + config["train_seed"], **common)
    if CACHED_X0_DIR:
        _cpath = f"{CACHED_X0_DIR}/{LABTAIL}_{config['name']}.pkl"
        _cache = pickle.load(open(_cpath, "rb"))
        print(f"[CACHED_X0] loaded {len(_cache)} x0 tracks from {_cpath}", flush=True)
        train_dataset = _X0Dataset(train_dataset, _cache)
        val_dataset = _X0Dataset(val_dataset, _cache)
    elif CACHED_PREMERGE_DIR:
        # load ONLY the per-video pre-merge shards this job needs (train+val vids) -> bounded RAM
        _pmdir = f"{CACHED_PREMERGE_DIR}/{LABTAIL}_{config['name']}"
        _need = {int(v.video_id) for v in train_videos} | {int(v.video_id) for v in val_videos}
        _cache = {}
        for _vid in sorted(_need):
            _p = f"{_pmdir}/{_vid}.pkl"
            if os.path.isfile(_p):
                _cache.update(pickle.load(open(_p, "rb")))
            else:
                print(f"[CACHED_PREMERGE] WARN missing shard {_p}", flush=True)
        assert _cache, f"no pre-merge shards found in {_pmdir} for {len(_need)} vids"
        _dim = int(next(iter(_cache.values())).shape[1])       # n_bp*Din (flattened)
        _gb = sum(a.nbytes for a in _cache.values()) / 1e9
        print(f"[CACHED_PREMERGE] loaded {len(_cache)} pre-merge tracks (dim={_dim}, {_gb:.1f} GB) "
              f"from {len(_need)} vids in {_pmdir}", flush=True)
        train_dataset = _PremergeDataset(train_dataset, _cache, _dim)
        val_dataset = _PremergeDataset(val_dataset, _cache, _dim)

    unsupervised_model = solution.UnsupervisedModel(
        d_res=192, d_lstm=192, d_ff=384, d_edge=96, n_layers=4,
        n_bp=config["num_bodyparts"], sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"], dtype="bfloat16")
    unsupervised_path = f"{solution.persist_dir}/{config['name']}_unsupervised.pkl"
    model = MultiTaskPerLabModel(
        d_res=256, d_ff=768, d_lstm=256, n_layers=3,
        n_bp=config["num_bodyparts"], padding=32, dtype="bfloat16",
        unsupervised_model=(unsupervised_model, unsupervised_path))
    if DISENTANGLE:
        model.set_disentangle(config["name"])
    if FILM and not (FROZEN_TRUNK or REFIT or LABTAIL or DISENTANGLE):
        # From-scratch foundation (LOLO) path: zero-init the film embeddings so the adapter starts at
        # identity (raw * (1+0) + 0). The warm-start path instead copies trained rows from the source.
        _cv0 = model.create_variables
        def _film_zero_cv(key, _cv0=_cv0):
            v = _cv0(key)
            for ln in ("film-gamma", "film-beta"):
                w = v[ln]["w"]
                v[ln]["w"] = solution.Variable(jnp.zeros_like(w.value), w.trainable)
            print("[FILM] zero-init film-gamma/beta (identity adapter) for from-scratch foundation", flush=True)
            return v
        model.create_variables = _film_zero_cv

    if FROZEN_TRUNK or REFIT or LABTAIL or DISENTANGLE:
        # Warm-start every layer from an EXISTING model and freeze a subset. FROZEN_TRUNK/REFIT
        # freeze everything except out-proj-perlab. LABTAIL freezes SSL + feature-merge and trains
        # the per-lab TAIL (TAIL_TRAINABLE) + head -- the new-lab adaptation on top of frozen x0.
        # DISENTANGLE freezes SSL + feature-merge + enc_dyn; trains TAIL_TRAINABLE + disent-proj + heads.
        assert SNIFFALL, "warm-start modes are sniffall-only; set SNIFFALL=1 too"
        if LABTAIL:
            _srcpath = os.environ.get("LABTAIL_SRC") or f"{solution.persist_dir}/{config['name']}_supervised_perlab_sniffall.pkl"
        elif DISENTANGLE:
            _srcpath = f"{solution.persist_dir}/{config['name']}_supervised_perlab_sniffall.pkl"
        else:
            _src = "_supervised_perlab_sniffall" if REFIT else "_supervised_perlab"
            _srcpath = f"{solution.persist_dir}/{config['name']}{_src}.pkl"
        existing = pickle.load(open(_srcpath, "rb"))
        _orig_cv = model.create_variables
        _stats = {"copied": 0, "trained": 0, "frozen": 0}

        def _reinit_weights(warm, fresh):
            # fresh-init the trainable weights but KEEP the normalize_input running stats
            # (m1/m2/n/mean/std), which only update in stage='init' -- a fresh 0/1 would break norm.
            if solution.is_variable(warm):
                fv = fresh.value if solution.is_variable(fresh) else warm.value
                return solution.Variable(fv, warm.trainable)
            return {k: (warm[k] if k in ("m1", "m2", "n", "mean", "std")
                        else _reinit_weights(warm[k], fresh[k]) if k in fresh else warm[k]) for k in warm}

        def _merge(vnode, enode, train_layer):
            # Recurse the variable tree (layers can nest, e.g. LSTM sub-layers). Copy the
            # existing tensor where shapes match; set trainable only for out-proj-perlab.
            if solution.is_variable(vnode):
                val = vnode.value
                if enode is not None and tuple(np.shape(enode)) == tuple(np.shape(val)):
                    val = jnp.asarray(enode); _stats["copied"] += 1
                keep = bool(vnode.trainable and train_layer)
                _stats["trained" if keep else "frozen"] += 1
                return solution.Variable(val, keep)
            return {k: _merge(vnode[k], enode.get(k) if isinstance(enode, dict) else None, train_layer)
                    for k in vnode}

        def _warmstart(key, _orig_cv=_orig_cv, _existing=existing):
            v = _orig_cv(key)
            if LABTAIL:
                _is_train = lambda ln: ln in LABTAIL_TRAIN
            elif DISENTANGLE:
                _is_train = lambda ln: ln in DISENT_TRAINABLE
            else:
                _is_train = lambda ln: ln == "out-proj-perlab"
            out = {lname: _merge(sub, _existing.get(lname), _is_train(lname))
                   for lname, sub in v.items()}
            if DISENTANGLE:
                # tail warm-started; disent-proj is a NEW layer (no match in _existing -> left fresh &
                # trainable by _merge). enc_dyn is applied functionally (not a variable) -> frozen.
                print(f"[DISENTANGLE] warm-start copied {_stats['copied']}; training tail+disent-proj+heads "
                      f"({_stats['trained']} tensors), SSL+merge+enc_dyn frozen ({_stats['frozen']})", flush=True)
                return out
            if LABTAIL:
                if LABTAIL_FRESH:                            # fresh-init the tail+head (NOT the merge)
                    for ln in LABTAIL_FRESH_LAYERS:
                        if ln in out and ln in v:
                            out[ln] = _reinit_weights(out[ln], v[ln])
                if SKIP_PATH:                                # zero-GATE the skip at init: s=0 -> skip contributes
                    # nothing, model == no-skip baseline (stable). w stays unit-norm (get_weights divides by
                    # ||w|| -> can't zero w). s is trainable, so the gate opens only as gradients warrant.
                    so = out["skip-proj-out"]
                    so["s"] = solution.Variable(jnp.zeros_like(so["s"].value), so["s"].trainable)
                    print("[SKIP_PATH] zero-gated skip-proj-out (s=0); skip opens during training", flush=True)
                if DONOR_LAB:                                # seed target's empty embedding row + head col(s) from a donor lab
                    si = int(solution.LABS.encode(DONOR_LAB)); di = int(solution.LABS.encode(LABTAIL))
                    emb = out["lab-embedding"]; ew = np.asarray(emb["w"].value).copy()
                    ew[di] = np.asarray(_existing["lab-embedding"]["w"])[si]
                    emb["w"] = solution.Variable(jnp.asarray(ew), emb["w"].trainable)
                    op = out["out-proj-perlab"]; exop = _existing["out-proj-perlab"]
                    w = np.asarray(op["w"].value).copy(); s = np.asarray(op["s"].value).copy(); b = np.asarray(op["b"].value).copy()
                    ew2 = np.asarray(exop["w"]); es = np.asarray(exop["s"]); eb = np.asarray(exop["b"])
                    seeded = []
                    _acts = LABTAIL_ACTIONS or [a for (l, a) in LAB_ACTION if l == LABTAIL]
                    for a in _acts:
                        try:
                            sj = LAB_ACTION.index((DONOR_LAB, a)); dj = LAB_ACTION.index((LABTAIL, a))
                        except ValueError:
                            continue
                        w[:, dj] = ew2[:, sj]; s[dj] = es[sj]; b[dj] = eb[sj]; seeded.append(a)
                    op["w"] = solution.Variable(jnp.asarray(w), op["w"].trainable)
                    op["s"] = solution.Variable(jnp.asarray(s), op["s"].trainable)
                    op["b"] = solution.Variable(jnp.asarray(b), op["b"].trainable)
                    print(f"[DONOR {DONOR_LAB}->{LABTAIL}] seeded embedding row {si}->{di} + head cols {seeded}", flush=True)
                _fbits = "SSL frozen, merge FINE-TUNED" if LABTAIL_TUNE_MERGE else "SSL+merge frozen"
                print(f"[LABTAIL {LABTAIL}] warm-start copied {_stats['copied']}; training "
                      f"({_stats['trained']} tensors), {_fbits} ({_stats['frozen']}); "
                      f"fresh={LABTAIL_FRESH} tune_merge={LABTAIL_TUNE_MERGE}", flush=True)
                return out
            # Warm-start the 88 EXISTING head columns into their positions in the 93-head
            # layout (out-proj-perlab; shapes 88->93 differ so _merge left them fresh). The 5
            # sniffall columns stay freshly-initialised and are the only ones that get gradient
            # (compute_loss masks the loss to them), so the 88 heads keep their trained quality.
            if REFIT:   # all 90 cols already copied by _merge (shapes match); only col REFIT_IDX gets gradient
                if SCALE:
                    # RE-INIT the target column with the model's native fresh init (from the
                    # un-merged variable tree `v`) so the head learns purely from SCALE_VIDS.
                    op, fresh = out["out-proj-perlab"], v["out-proj-perlab"]
                    for pk in ("w", "s", "b"):
                        cur = np.asarray(op[pk].value).copy(); fv = np.asarray(fresh[pk].value)
                        if pk == "w":
                            cur[:, REFIT_IDX] = fv[:, REFIT_IDX]
                        else:
                            cur[REFIT_IDX] = fv[REFIT_IDX]
                        op[pk] = solution.Variable(jnp.asarray(cur), op[pk].trainable)
                    print(f"[SCALE {SCALE}] warm-start trunk (copied {_stats['copied']}); "
                          f"col {REFIT_IDX} RE-INIT fresh; training on {len(REFIT_VIDS)} vids", flush=True)
                else:
                    print(f"[REFIT] warm-start from sniffall model: copied {_stats['copied']} tensors; "
                          f"fitting only col {REFIT_IDX} ({REFIT})", flush=True)
                return out
            op, ex = out["out-proj-perlab"], _existing["out-proj-perlab"]
            w = np.asarray(op["w"].value).copy(); s = np.asarray(op["s"].value).copy(); b = np.asarray(op["b"].value).copy()
            ew, es, eb = np.asarray(ex["w"]), np.asarray(ex["s"]), np.asarray(ex["b"])
            for oi, key_ in enumerate(OLD_LAB_ACTION):
                nj = LAB_ACTION.index(key_)
                w[:, nj] = ew[:, oi]; s[nj] = es[oi]; b[nj] = eb[oi]
            # SEED each sniffall column from that lab's already-trained SNIFF column. A fresh
            # linear head on the frozen trunk does NOT converge from random init (Adam plateaus
            # at a mediocre basin ~0.16 BCE), but the frozen sniff direction already scores AP
            # ~0.76 on the union GT -> start the sniffall head there so training only refines it
            # (e.g. adds reciprocalsniff sensitivity), rather than searching from scratch.
            for lab in SNIFFALL_LABS:
                sj = LAB_ACTION.index((lab, "sniff")); aj = LAB_ACTION.index((lab, "sniffall"))
                w[:, aj] = w[:, sj]; s[aj] = s[sj]; b[aj] = b[sj]
            op["w"] = solution.Variable(jnp.asarray(w), op["w"].trainable)
            op["s"] = solution.Variable(jnp.asarray(s), op["s"].trainable)
            op["b"] = solution.Variable(jnp.asarray(b), op["b"].trainable)
            print(f"[FROZEN_TRUNK] warm-start: copied {_stats['copied']} tensors + remapped 88 head cols; "
                  f"training only {int(SNIFFALL_COL.sum())} sniffall cols of out-proj-perlab", flush=True)
            return out
        model.create_variables = _warmstart

    trainer = solution.Trainer(
        experiment_name=f"{config['name']}/train_perlab",
        model=model, optimizer=(
            SchedAdam(FT_LR, total=LR_COSINE_T, floor=LR_FLOOR, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP)
            if (LR_COSINE_T or WEIGHT_DECAY or GRAD_CLIP or FT_LR != 0.004) else solution.Adam(0.004)),
        train_dataset=train_dataset, eval_dataset=val_dataset,
        train_batch_size=128, seed=[2] + config["train_seed"],
        train_log_interval=int(os.environ["LABTAIL_LOG_INTERVAL"]) if os.environ.get("LABTAIL_LOG_INTERVAL")
        else (100 if smoke else 500),
        eval_interval=int(os.environ["LABTAIL_EVAL_INTERVAL"]) if os.environ.get("LABTAIL_EVAL_INTERVAL")
        else (300 if smoke else (500 if LABTAIL else (1000 if (FROZEN_TRUNK or REFIT or DISENTANGLE) else 2000))),
        # FROZEN_TRUNK: near-off EMA so the saved weights track the RAW trained head. With the
        # default 0.9993 decay (~1400-step time constant) the sniffall column, which drifts across
        # training, is saved as a blurred trajectory-average -> looks undertrained. Tracking raw
        # weights tests whether the poor AP is EMA blur vs a genuine optimization failure.
        ema_decay=float(os.environ["LABTAIL_EMA"]) if os.environ.get("LABTAIL_EMA")
        else (0.9 if (FROZEN_TRUNK or REFIT) else 0.9993),
        # SNIFFALL_SCRATCH: predict() only fills the sniffall action, which is NOT in the val
        # videos' scored behaviours -> logged val-f1 is ~0 and flat. Disable early stop (huge
        # patience) and save EMA-last instead of best.pkl (below), like the FROZEN_TRUNK path.
        early_stopping_config={"metric_name": "f1", "lower_is_better": False,
                               "patience": 10**9 if (SNIFFALL_SCRATCH or REFIT) else 10000},
        max_training_steps=600 if smoke else (FT_STEPS if (FROZEN_TRUNK or REFIT or LABTAIL or DISENTANGLE) else (SCRATCH_STEPS if SNIFFALL_SCRATCH else 50000)))
    # Run UNSHARDED (single GPU per process). The Trainer sets up a 1-device
    # batch mesh, but under this JAX version that triggers a sharding-broadcast
    # error in the frozen forecaster's LSTM scatter (hs unsharded vs h @batch).
    # Disabling the mesh matches the working inference path.
    jax.set_mesh(jax.make_mesh((), ()))   # clear Trainer's 1-device mesh (jax 0.9.2: set_mesh(None) is rejected;
    trainer.sharding = None               # the empty mesh is the "no mesh" state that matches the inference path)
    # Checkpoints must live OFF the samba mount (no symlink support there -> the
    # best.pkl symlink raises FileNotFoundError). Use /dev/shm; we copy the final
    # weights into models/ (a plain file write, which samba does support).
    _mtag = "_refit" if REFIT else ("_sniffall_scratch" if SNIFFALL_SCRATCH else ("_sniffall" if SNIFFALL else ""))
    ckpt = (f"/dev/shm/doom_scale/{config['name']}__{SCALE_TAG}/checkpoints" if SCALE
            else f"/dev/shm/doom_lolo/{config['name']}_LOLO-{LOLO_EXCLUDE}{FND_TAG}/checkpoints" if LOLO_EXCLUDE
            else f"/dev/shm/doom_labtail/{config['name']}__{LABTAIL_TAG}/checkpoints" if LABTAIL
            else f"/dev/shm/doom_disent/{config['name']}/checkpoints" if DISENTANGLE
            else f"/dev/shm/doom_exp/{config['name']}_train_perlab{_mtag}/checkpoints")
    os.makedirs(ckpt, exist_ok=True)
    trainer.checkpoint_dir = ckpt
    trainer.checkpoint_manager.checkpoint_dir = ckpt
    # TRAJ_KEEP=N: keep EVERY eval checkpoint (raise max_checkpoints so free -inf slots absorb all writes,
    # never evicting) and disable early stopping -- to record a full train-length trajectory for scoring.
    if os.environ.get("TRAJ_KEEP"):
        n = int(os.environ["TRAJ_KEEP"])
        trainer.checkpoint_manager.max_checkpoints = n
        trainer.checkpoint_manager.checkpoints = [(None, float("-inf"))] * n
        trainer.checkpoint_manager.patience = 10**9
        print(f"[TRAJ] keeping up to {n} checkpoints, early-stop disabled", flush=True)
    trainer.custom_eval_loop = solution.get_custom_eval_loop(trainer)
    # FROZEN_TRUNK: the val-f1 metric only measures the 88 (warm-started, frozen) heads, so it
    # is flat and best.pkl would be the EARLIEST checkpoint (undertrained sniffall). Capture the
    # EMA weights at every eval and save the LAST (fully-converged sniffall) instead.
    _ema_holder = {}
    if FROZEN_TRUNK or SNIFFALL_SCRATCH or REFIT or LABTAIL:
        _base_eval = trainer.custom_eval_loop
        def _capturing_eval(ema_values, _b=_base_eval, _h=_ema_holder):
            _h["ema"] = ema_values
            return _b(ema_values)
        trainer.custom_eval_loop = _capturing_eval
    trainer.train()

    if not smoke:
        if SCALE:
            # Full model (canonical trunk + the one freshly-trained column) to /dev/shm so the
            # existing single-config inference can load it directly. Transient, unique per job.
            _sdir = "/dev/shm/doom_scale/models"; os.makedirs(_sdir, exist_ok=True)
            out = f"{_sdir}/{config['name']}__{SCALE_TAG}.pkl"
        elif LOLO_EXCLUDE:
            os.makedirs(solution.persist_dir, exist_ok=True)
            out = f"{solution.persist_dir}/{config['name']}_supervised_perlab_sniffall_LOLO-{LOLO_EXCLUDE}{FND_TAG}.pkl"
        elif LABTAIL:
            _ldir = LABTAIL_OUTDIR or "/dev/shm/doom_labtail/models"; os.makedirs(_ldir, exist_ok=True)
            out = f"{_ldir}/{config['name']}__{LABTAIL_TAG}.pkl"
        elif DISENTANGLE:
            os.makedirs(solution.persist_dir, exist_ok=True)
            out = f"{solution.persist_dir}/{config['name']}_supervised_perlab_sniffall_disentangle.pkl"
        else:
            os.makedirs(solution.persist_dir, exist_ok=True)
            out = (f"{solution.persist_dir}/{config['name']}_supervised_perlab_sniffall_refit.pkl" if REFIT else
                   f"{solution.persist_dir}/{config['name']}_supervised{'_sniffall_scratch' if SNIFFALL_SCRATCH else ('_perlab_sniffall' if SNIFFALL else '_perlab')}.pkl")
        if FROZEN_TRUNK or SNIFFALL_SCRATCH or REFIT or LABTAIL:
            cpu = jax.devices("cpu")[0]
            params = jax.tree.map(lambda x: jax.device_put(x, cpu), _ema_holder["ema"])
            with open(out, "wb") as f:
                pickle.dump(params, f)
        else:
            with open(f"{trainer.checkpoint_dir}/best.pkl", "rb") as f1, open(out, "wb") as f2:
                f2.write(f1.read())
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(solution.get_configs()))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_perlab(solution.get_configs()[args.config], smoke=args.smoke)
