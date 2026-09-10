"""PyTorch fine-tuning driver -- the torch counterpart of `train_perlab_heads.py`.

Invoked the same way and configured by the same environment variables, so
`finetune.py train` only has to choose which module to run:

    python -m hidra.torch.train_perlab --config 15fps_5bp

Scope: this ports the **LABTAIL** flow, which is the only path `finetune.py` can reach.
`train_perlab_heads.py` also carries FROZEN_TRUNK, REFIT, SCALE, LOLO, DISENTANGLE, CORAL,
FILM, SKIP_PATH and the cached-feature fast paths -- research scaffolding that `cmd_train`
explicitly strips from the environment before invoking the trainer. Porting them unused
would multiply the surface with no way to exercise it; if one is ever needed it belongs
here as a deliberate addition.

What LABTAIL does: warm-start every layer from the published per-lab checkpoint, freeze
the self-supervised trunk and the shared feature merge, and train the adopted lab's tail
(or just its embedding, or just its head column) on the user's annotations, supervising
only that lab's selected behaviours.

The saved checkpoint is written in the **JAX variable layout**, so a fine-tune produced
here loads under either backend via `predict.py --weights`.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from .. import paths, schema
from . import train as T
from .models import load_perlab, load_unsupervised

# ------------------------------------------------------------------ configuration

def env_flag(name):
    return os.environ.get(name) == "1"


def resolve_mode():
    """The three modes `finetune.py --mode` exposes, as env flags."""
    if env_flag("LABTAIL_HEAD_ONLY"):
        return "head"
    if env_flag("LABTAIL_EMB_ONLY"):
        return "embedding"
    return "tail"


def env_settings(config_name):
    """Everything the run needs, read from the environment `finetune.py` sets."""
    lab = os.environ.get("LABTAIL")
    if not lab:
        sys.exit("ERROR: LABTAIL must name the lab being fine-tuned "
                 "(finetune.py sets it; see docs/fine-tuning.md)")
    if lab not in schema.LABS.value_to_idx:
        sys.exit(f"ERROR: LABTAIL={lab!r} is not one of the {len(schema.LABS)} known labs")
    actions = [a.strip() for a in os.environ.get("LABTAIL_ACTIONS", "").split(",") if a.strip()]
    return {
        "config_name": config_name,
        "lab": lab,
        "actions": actions or None,
        "mode": resolve_mode(),
        "steps": int(os.environ.get("FT_STEPS", "8000")),
        "lr": float(os.environ.get("FT_LR", "0.004")),
        "cosine_total": int(os.environ.get("LR_COSINE_T", "0")),
        "lr_floor": float(os.environ.get("LR_FLOOR", "0.0")),
        "weight_decay": float(os.environ.get("WEIGHT_DECAY", "0.0")),
        "grad_clip": float(os.environ.get("GRAD_CLIP", "0.0")),
        "tag": os.environ.get("LABTAIL_TAG", f"{lab}_warm"),
        "out_dir": os.environ.get("LABTAIL_OUTDIR") or str(paths.work_root() / "labtail_models"),
        "video_ids": {int(v) for v in os.environ.get("LABTAIL_VIDS", "").split(",") if v.strip()},
        "seed": int(os.environ["LABTAIL_SEED"]) if os.environ.get("LABTAIL_SEED") else None,
        "eval_interval": int(os.environ.get("LABTAIL_EVAL_INTERVAL", "500")),
        "log_interval": int(os.environ.get("LABTAIL_LOG_INTERVAL", "500")),
        "ema_decay": float(os.environ.get("LABTAIL_EMA", "0.9993")),
        "batch_size": int(os.environ.get("LABTAIL_BATCH", "128")),
        "dtype": os.environ.get("LABTAIL_DTYPE", "bfloat16"),
        "ddi_steps": int(os.environ.get("LABTAIL_DDI_STEPS", "256")),
    }


DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float64": torch.float64}


# ------------------------------------------------------------------ data

def load_finetune_videos():
    """Videos from the staged fine-tuning dataset, mirroring `load_train_videos_robust`.

    Videos whose tracking parquet is absent are skipped rather than raising, matching the
    original (the competition data has one such file).
    """
    import pandas as pd

    from .. import data

    if os.environ.get("HIDRA_DATA_DIR"):
        data.dataset_dir = os.path.abspath(os.environ["HIDRA_DATA_DIR"])
        print(f"[DATA] dataset_dir -> {data.dataset_dir}", flush=True)
    if os.environ.get("PERLAB_WORKDIR"):
        data.working_dir = os.environ["PERLAB_WORKDIR"]
        os.makedirs(data.working_dir, exist_ok=True)

    csv = os.path.join(data.dataset_dir, "train.csv")
    if not os.path.isfile(csv):
        upper = os.path.join(data.dataset_dir, "TRAIN.csv")
        if not os.path.isfile(upper):
            sys.exit(f"ERROR: no TRAIN.csv in {data.dataset_dir}; run "
                     f"`finetune.py prepare` first")
        import shutil
        shutil.copy(upper, csv)

    df = pd.read_csv(csv)
    df["mode"] = "train"
    videos, skipped = [], 0
    for i, row in df.iterrows():
        p = os.path.join(data.dataset_dir, "train_tracking", str(row["lab_id"]),
                         f"{int(row['video_id'])}.parquet")
        if not os.path.isfile(p):
            skipped += 1
            continue
        videos.append(data.create_video(i, row))
    print(f"built {len(videos)} videos ({skipped} skipped: missing tracking)", flush=True)
    return videos


def split_labtail_videos(videos, config, lab, video_ids=None):
    """The LABTAIL train/val split.

    Uses the *global* split seed so a video that was held out for the published model stays
    held out here -- otherwise the validation metric would be measured on frames the
    original training already saw. `video_ids` (LABTAIL_VIDS) overrides the train side
    outright, which is how the size sweeps pick a subset.
    """
    from .. import data

    train_videos, val_videos = data.split_videos(
        videos, validation_frac=0.15, random_seed=config["split_seed"])
    val_videos = [v for v in val_videos if v.lab_name not in schema.TRAIN_ONLY_LABS]

    tr_ids = {int(v.video_id) for v in train_videos}
    va_ids = {int(v.video_id) for v in val_videos}
    pool = [v for v in videos if v.lab_name == lab]
    train_videos = [v for v in pool if int(v.video_id) in tr_ids]
    val_videos = [v for v in pool if int(v.video_id) in va_ids]
    if video_ids:
        train_videos = [v for v in pool if int(v.video_id) in video_ids]
    if not val_videos:
        # Guarantee an eval signal even for a handful of videos; may overlap the train set,
        # as in the original. The metric is then optimistic -- calibrate on held-out data.
        val_videos = train_videos[-max(1, len(train_videos) // 6):]
        print("[LABTAIL] no held-out videos for this lab; validating on a slice of the "
              "training videos (metric is optimistic)", flush=True)
    if not train_videos:
        sys.exit(f"ERROR: no staged videos for lab {lab!r}")
    return train_videos, val_videos


def build_datasets(config, train_videos, val_videos):
    """Train and validation datasets, with the original's augmentation settings."""
    from .. import data

    common = dict(seq_len=64, sample_rate=config["sample_rate"], padding=32,
                  num_bodyparts=config["num_bodyparts"], unsupervised=False)
    train_dataset = data.Dataset(
        videos=train_videos, num_epochs=100000, num_workers=8,
        seed=[0] + config["train_seed"],
        max_scale=config["max_scale"], max_time_dilation=config["max_time_dilation"],
        rotate=True, flip=True, noise_scale=config["noise_scale"], **common)
    val_dataset = data.Dataset(
        videos=val_videos, num_epochs=1, max_scale=1, max_time_dilation=1,
        rotate=False, flip=False, noise_scale=config["noise_scale"],
        num_workers=8, seed=[1] + config["train_seed"], **common)
    return train_dataset, val_dataset


def torch_batches(dataset, batch_size, device):
    """An endless stream of device-resident batches, labels included."""
    from .. import data
    from .infer import to_torch_batch

    label_keys = ("self_labels", "cross_labels", "self_label_mask", "cross_label_mask",
                  "batch_mask")
    for batch in data.batch(dataset.element_iterator(), batch_size):
        tb = to_torch_batch(batch, device)
        for key in label_keys:
            tb[key] = torch.as_tensor(np.asarray(batch[key]), device=device)
        yield tb


# ------------------------------------------------------------------ evaluation

def make_eval_fn(config, val_videos, device, batch_size):
    """Validation F1, via the same `Predictions.score()` the JAX path uses.

    Scoring goes through the accumulator rather than the raw loss because that is the
    metric the checkpoint selection and early stopping are defined on: per-frame
    probabilities are resampled back onto video time, averaged over augmented passes, and
    swept over thresholds. A per-batch loss would rank checkpoints differently.
    """
    from .. import data

    def eval_fn(head):
        predictions = data.Predictions(val_videos)
        dataset = data.Dataset(
            videos=val_videos, seq_len=64, sample_rate=config["sample_rate"], padding=32,
            num_bodyparts=config["num_bodyparts"], num_epochs=1, unsupervised=False,
            max_scale=1, max_time_dilation=1, rotate=False, flip=False,
            noise_scale=config["noise_scale"], num_workers=8,
            seed=[1] + config["train_seed"])
        from .infer import to_torch_batch, unbatch_numpy
        with torch.no_grad():
            for batch in data.batch(dataset.element_iterator(), batch_size):
                probs = head.predict(to_torch_batch(batch, device))
                probs = probs.detach().to("cpu", torch.float32).numpy()
                for i in range(probs.shape[0]):
                    predictions.update(unbatch_numpy(batch, i), probs[i])
        metrics, _ = predictions.score()
        return {k: float(v) for k, v in metrics.items()}

    return eval_fn


# ------------------------------------------------------------------ checkpoint writing

def to_jax_layout(state):
    """Flat torch tensor names -> the nested dict the JAX code unpickles.

    Written in the JAX layout on purpose: a fine-tune done on the torch backend then loads
    under either backend through `predict.py --weights`, so switching backends never
    invalidates someone's trained checkpoint.
    """
    from .models import _jax_to_torch_path  # noqa: F401  (documents the inverse mapping)

    tree = {}
    for name, tensor in state.items():
        if name == "P":
            continue  # derived from thresholds.pkl, not a weight
        path = name
        if path.startswith("layers."):
            path = path[len("layers."):]
        path = path.replace("lstm_fw", "lstm-fw").replace("lstm_bw", "lstm-bw")
        parts = path.split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        value = tensor.detach().to("cpu", torch.float32).numpy()
        node[parts[-1]] = value
    return tree


def save_checkpoint(path, state):
    import pickle

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(to_jax_layout(state), f)
    return path


# ------------------------------------------------------------------ the run

def finetune_config(settings, smoke=False, device=None):
    """Fine-tune one ensemble config. Returns the path of the saved checkpoint."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = schema.get_configs()[settings["config_name"]]
    if settings["seed"] is not None:
        config = {**config, "train_seed": [settings["seed"]]}
        print(f"[SEED] train_seed -> {config['train_seed']}", flush=True)

    dtype = DTYPES[settings["dtype"]]
    lab, actions, mode = settings["lab"], settings["actions"], settings["mode"]

    videos = load_finetune_videos()
    train_videos, val_videos = split_labtail_videos(videos, config, lab,
                                                    settings["video_ids"])
    print(f"[LABTAIL {lab}] {len(train_videos)} train, {len(val_videos)} val video(s); "
          f"mode={mode} steps={settings['steps']} lr={settings['lr']}", flush=True)

    train_dataset, _ = build_datasets(config, train_videos, val_videos)

    # Warm start: every layer comes from the published checkpoint.
    trunk = load_unsupervised(settings["config_name"], dtype=dtype, device=device, config=config)
    head = load_perlab(settings["config_name"], trunk, dtype=dtype, device=device, config=config)
    n_train, n_frozen = T.set_trainable_layers(head, T.MODE_LAYERS[mode])
    print(f"[LABTAIL {lab}] warm-started; training {n_train} tensor(s) across "
          f"{len(T.MODE_LAYERS[mode])} layer(s), {n_frozen} frozen "
          f"(trunk + feature merge stay frozen)", flush=True)

    columns = T.head_column_selector(head.lab_action, lab, actions, device=device)
    print(f"[LABTAIL {lab}] supervising {int(columns.sum())} head column(s): "
          f"{[a for (l, a) in head.lab_action if l == lab and (not actions or a in actions)]}",
          flush=True)

    batch_size = settings["batch_size"]
    steps = 100 if smoke else settings["steps"]
    ddi_steps = 8 if smoke else settings["ddi_steps"]

    # Data-dependent init: recalibrate the head's input standardization on this data.
    t0 = time.time()
    n_ddi = 0
    if ddi_steps:
        head.set_stage("init", init_decay=0.99)
        with torch.no_grad():
            for i, batch in enumerate(torch_batches(train_dataset, batch_size, device)):
                T.ddi_forward(head, batch)
                n_ddi = i + 1
                if n_ddi >= ddi_steps:
                    break
        head.set_stage("train")
        print(f"[DDI] recalibrated input statistics over {n_ddi} batch(es) "
              f"({time.time() - t0:.0f}s)", flush=True)

    trainable = [p for p in head.parameters() if p.requires_grad]
    optimizer = T.SchedAdam(trainable, settings["lr"], total=settings["cosine_total"],
                            floor=settings["lr_floor"],
                            weight_decay=settings["weight_decay"],
                            grad_clip=settings["grad_clip"])

    eval_fn = None if smoke else make_eval_fn(config, val_videos, device, batch_size)
    ckpt_dir = os.path.join(str(paths.work_root()), "labtail",
                            f"{settings['config_name']}__{settings['tag']}", "checkpoints")
    tuner = T.FineTuner(
        head, optimizer, torch_batches(train_dataset, batch_size, device),
        eval_fn=eval_fn, ema_decay=settings["ema_decay"], checkpoint_dir=ckpt_dir,
        max_training_steps=steps, log_interval=min(settings["log_interval"], max(steps // 4, 1)),
        eval_interval=settings["eval_interval"], metric_name="f1", lower_is_better=False,
        patience=10000, head_columns=columns, ddi_steps=0)
    # The EMA was seeded before DDI moved the statistics; re-seed so it starts from the
    # calibrated state rather than averaging across the recalibration.
    tuner.tracked = tuner._tracked_tensors()
    tuner.ema = T.EMA(tuner.tracked, settings["ema_decay"])

    done = tuner.train()
    print(f"[LABTAIL {lab}] {done} step(s) in {(time.time() - t0) / 60:.1f} min", flush=True)

    if smoke:
        print("(--smoke writes no checkpoint; it proves the data + env are wired up)")
        return None

    # LABTAIL saves the LAST EMA snapshot, not the best-scoring one: the validation F1 here
    # measures the adopted lab's heads on few videos and is noisy, so the earliest
    # checkpoint often wins on chance. The original does the same.
    state = tuner.last_ema_values or tuner.ema.values()
    out = os.path.join(settings["out_dir"], f"{settings['config_name']}__{settings['tag']}.pkl")
    save_checkpoint(out, state)
    print(f"saved {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m hidra.torch.train_perlab",
        description="Fine-tune one ensemble config's per-lab head (PyTorch backend).")
    ap.add_argument("--config", required=True, choices=list(schema.get_configs()))
    ap.add_argument("--smoke", action="store_true",
                    help="100 steps, no eval, no checkpoint -- just proves the wiring")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    finetune_config(env_settings(args.config), smoke=args.smoke, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
