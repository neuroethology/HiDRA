"""Run the per-(lab,behavior)-HEAD experiment model on the TEST set and save
per-frame PROBABILITIES, mirroring run_test_probs.py exactly but swapping the
shared SupervisedModel for the MultiTaskPerLabModel.

The per-lab model's predict() emits 37-space probs computed from the 88 per-lab
heads via the P^T projection -> for a video whose lab is L, the prob track for
action A is nonzero ONLY if L has a trained (L,A) head (exactly the classifiers
we want to evaluate). Predictions.update keys are identical to the shared model
(vid, agent_enc, target_enc, action_id), so merge_test_probs logic is reused.

Each (config, shard) is one process. Output raw sums+weights per shard so the
5-config ensemble average reconstructs exactly.

Usage:
  CUDA_VISIBLE_DEVICES=1 python run_test_probs_perlab.py --config 15fps_5bp --shard 0/1 --epochs 5
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
import jax
import solution
from train_perlab_heads import MultiTaskPerLabModel

OUT_DIR = os.path.join(PROJECT_DIR, "analysis_outputs", "test_predictions", "perlab")
SNIFFALL = bool(os.environ.get("SNIFFALL"))          # load the namespaced 93-head sniffall model + its heads
_PERLAB_SUFFIX = "_sniffall" if SNIFFALL else ""


def predict_into(config, predictions, num_epochs, dtype):
    ds = solution.Dataset(
        videos=predictions.videos, seq_len=64, sample_rate=config["sample_rate"],
        padding=32, num_bodyparts=config["num_bodyparts"], num_epochs=num_epochs,
        unsupervised=False, max_scale=0.95 * config["max_scale"],
        max_time_dilation=0.95 * config["max_time_dilation"], rotate=True, flip=True,
        noise_scale=config["noise_scale"], num_workers=3, seed=[1] + config["eval_seed"])
    um = solution.UnsupervisedModel(d_res=192, d_lstm=192, d_ff=384, d_edge=96,
        n_layers=4, n_bp=config["num_bodyparts"], sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"], dtype=dtype)
    upath = f"{solution.persist_dir}/{config['name']}_unsupervised.pkl"
    sm = MultiTaskPerLabModel(d_res=256, d_ff=768, d_lstm=256, n_layers=3,
        n_bp=config["num_bodyparts"], padding=32, dtype=dtype,
        unsupervised_model=(um, upath))
    sm.set_context({"stage": "eval"})
    weights = pickle.load(open(f"{solution.persist_dir}/{config['name']}_supervised_perlab{_PERLAB_SUFFIX}.pkl", "rb"))
    sm = sm.set_variables(weights)

    @jax.jit
    def step(batch, key):
        key, sub = jax.random.split(key)
        return sm.predict(batch, sub), key
    key = jax.random.key(0)

    def outs(batch):
        nonlocal key
        p, key = step(batch, key)
        return batch, p
    elements = solution.Pipeline(ds.element_iterator())
    _bn = int(os.environ.get("PREDICT_BATCH", "256"))       # lower (e.g. 64) to fit long videos at high-fps configs
    elements.batch(_bn).to_device(None).map(outs).to_host().unbatch().map(
        lambda x: predictions.update(*x)).last()


def load_videos(shard_i, n_shard):
    df = pd.read_csv(f"{solution.dataset_dir}/test.csv").copy()
    df["mode"] = "test"
    # AdaptableSnail@25fps videos are now trained + predicted normally (skip removed).
    df = df.reset_index(drop=True)
    df = df.iloc[shard_i::n_shard]
    return [solution.create_video(0, row) for _, row in df.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--embedding-lab", default=None,
                    help="override lab_id for ALL videos -> the per-lab model's "
                         "trunk uses this lab's embedding AND its predict() P-"
                         "projection selects this lab's (lab,action) heads. So the "
                         "saved 37-space probs ARE this lab's per-lab classifier, "
                         "evaluated on every data-lab's videos (cross-lab matrices).")
    args = ap.parse_args()
    if not os.path.isfile(f"{solution.dataset_dir}/test.csv"):
        import shutil; shutil.copy(f"{solution.dataset_dir}/TEST.csv", f"{solution.dataset_dir}/test.csv")
    os.makedirs(args.out, exist_ok=True)
    i, n = map(int, args.shard.split("/"))
    tag = f"emb-{args.embedding_lab}_" if args.embedding_lab else ""
    outpath = os.path.join(args.out, f"raw_{tag}{args.config}_{i}of{n}.pkl")
    if os.path.isfile(outpath):
        print(f"{outpath} exists, skipping"); return

    cfg = solution.get_configs()[args.config]
    vids = load_videos(i, n)
    if args.embedding_lab:
        # Force every video's lab_id to embedding_lab (monkeypatch LABS.encode,
        # used only by Dataset.get_element for batch["lab_id"]). video.lab_name
        # stays the true source lab so the AdaptableSnail@25fps skip is unaffected.
        emb_idx = int(solution.LABS.encode(args.embedding_lab))
        solution.LABS.encode = lambda v, _i=emb_idx: _i
        print(f"  embedding+head overridden -> {args.embedding_lab} (idx {emb_idx})", flush=True)
    print(f"[perlab {tag}{args.config} shard {i}/{n}] {len(vids)} videos, epochs={args.epochs}", flush=True)
    preds = solution.Predictions(vids)
    t0 = time.time()
    predict_into(cfg, preds, args.epochs, args.dtype)
    sums = {k: v.astype("float16") for k, v in preds.sums.items()}
    weights = {k: v.astype("float16") for k, v in preds.weights.items()}
    with open(outpath, "wb") as f:
        pickle.dump({"sums": sums, "weights": weights}, f)
    print(f"[perlab {args.config} shard {i}/{n}] done {len(vids)} vids in "
          f"{(time.time()-t0)/60:.1f} min -> {outpath}", flush=True)


if __name__ == "__main__":
    main()
