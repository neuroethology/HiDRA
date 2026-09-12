#!/usr/bin/env python3
"""
HiDRA as a Python API -- the same thing predict.py does, callable from a notebook.

    import hidra

    hidra.heads()                       # [(lab, action), ...] every classifier head
    hidra.heads(lab="GroovyShrew")      # just that lab's behaviours

    # zero-shot inference (writes bouts.csv + frames.parquet into out/, like predict.py)
    hidra.predict("tracking/", out="results/", labs=["GroovyShrew"], actions=["rear"],
                  pix_per_cm=16, fps=30)

    # ... with fine-tuned weights + re-calibrated thresholds (see docs/fine-tuning.md)
    hidra.predict("tracking/", out="results/", labs=["GroovyShrew"], actions=["rear"],
                  pix_per_cm=16, fps=30, weights="ft_models/{config}__myrig.pkl",
                  thresholds="ft_thresholds.csv")

    bouts = hidra.bouts("results/")      # ethogram DataFrame, one row per bout, + a `video` column
    probs = hidra.frames("results/")     # per-frame probabilities + calls, + a `video` column

predict() needs a GPU and the downloaded weights; everything else is pure pandas. Reading the
outputs back is just pandas, so `bouts`/`frames` work anywhere.
"""
import glob
import os

import pandas as pd

from . import cli as _p

THRESHOLDS = _p.THR_CSV
run = _p.run                 # the full-control entrypoint (same kwargs as predict.py's flags)


def heads(lab=None, action=None, thresholds=THRESHOLDS):
    """Every trained (lab, action) classifier head, optionally filtered."""
    hs = _p.enum_heads(thresholds)
    return [(l, a) for l, a in hs if (lab in (None, l)) and (action in (None, a))]


def labs(thresholds=THRESHOLDS):
    """Labs that have at least one classifier head."""
    return sorted({l for l, _ in _p.enum_heads(thresholds)})


def actions(lab=None, thresholds=THRESHOLDS):
    """Behaviours with a head, for one lab or across all of them."""
    return sorted({a for l, a in _p.enum_heads(thresholds) if lab in (None, l)})


def thresholds(path=THRESHOLDS):
    """The decision thresholds as a DataFrame[lab, action, threshold]."""
    d = pd.read_csv(path)
    keys = [str(k) for k in d[d.columns[0]]]
    out = pd.DataFrame(dict(head=keys, threshold=d[d.columns[1]].astype(float)))
    out[["lab", "action"]] = out["head"].str.split("__", n=1, expand=True)
    return out[["lab", "action", "threshold"]]


def predict(folder, out="doom_predictions", labs=None, actions=None, subject="*", target="*",
            pix_per_cm=None, fps=None, jobs=None, gpu="0", output="both", weights=None,
            thresholds=THRESHOLDS, threshold=None, keep_work=False, backend="torch",
            configs=None):
    """Run the ensemble over `folder` and return the bouts DataFrame (see run() for the
    full argument list; this is the same call predict.py's CLI makes). `configs` runs a
    subset of the five ensemble members -- faster and lower quality, for iterating."""
    _p.run(folder, jobs=jobs, labs=labs, actions=actions, subject=subject, target=target,
           out=out, pix_per_cm=pix_per_cm, fps=fps, gpu=gpu, output=output,
           keep_work=keep_work, weights=weights, thresholds=thresholds, threshold=threshold,
           backend=backend, configs=configs)
    return bouts(out) if output in ("both", "calls") else frames(out)


def _read_all(pattern, folder, suffix):
    fs = sorted(glob.glob(os.path.join(folder, pattern)))
    if not fs:
        raise FileNotFoundError(f"no {pattern} in {folder}")
    ds = []
    for f in fs:
        d = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        d.insert(0, "video", os.path.basename(f)[: -len(suffix)])
        ds.append(d)
    return pd.concat(ds, ignore_index=True)


def bouts(folder="doom_predictions"):
    """Every <stem>.bouts.csv in `folder`, concatenated, with a `video` column."""
    return _read_all("*.bouts.csv", folder, ".bouts.csv")


def frames(folder="doom_predictions"):
    """Every <stem>.frames.parquet in `folder`, concatenated, with a `video` column.
    One row per (frame, subject, target, lab, action) -- large; filter as you go."""
    return _read_all("*.frames.parquet", folder, ".frames.parquet")


def ethogram(folder="doom_predictions", lab=None, action=None, subject=None, target=None):
    """bouts() with the common filters applied, sorted for reading top-to-bottom."""
    d = bouts(folder)
    for col, val in (("lab", lab), ("action", action), ("subject", subject), ("target", target)):
        if val is not None:
            d = d[d[col].isin([val] if isinstance(val, str) else list(val))]
    return d.sort_values(["video", "subject", "target", "lab", "action", "start_frame"], ignore_index=True)
