"""Convert the published JAX checkpoints to torch-native safetensors.

    hidra-convert-weights                 # all 5 configs, both checkpoints each
    hidra-convert-weights --check         # report what is converted, write nothing
    hidra-convert-weights --one ft_models/15fps_5bp__myrig.pkl

The point of converting is not speed -- `hidra.torch` reads the .pkl files directly -- it
is dependency removal. A .pkl of JAX device arrays can only be unpickled where JAX's
`_reconstruct_array` resolves; safetensors needs nothing but numpy. (This tool itself does
not need JAX either: see `checkpoint.load_jax_checkpoint`.)

Conversion is a pure re-container: the float32 values are copied through unchanged, and
`--verify` re-reads the result and asserts bit equality against the source.

A checkpoint whose head is wider than the published table (`finetune.py train --new-head`)
comes with a `.heads.json` sidecar; `--one` carries that table into the safetensors metadata
(`lab_action`), so the converted file still describes its own columns (see `hidra.head_table`).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

CONFIGS = ["11fps_4bp", "15fps_5bp", "19fps_6bp", "23fps_7bp", "27fps_6bp"]
KINDS = ["unsupervised", "supervised_perlab_sniffall"]


def source_paths(models_dir):
    """The published .pkl checkpoints, as (config, kind, path)."""
    return [(c, k, Path(models_dir) / f"{c}_{k}.pkl") for c in CONFIGS for k in KINDS]


def target_path(pkl_path):
    return Path(pkl_path).with_suffix(".safetensors")


def convert_thresholds(models_dir, verify=True):
    """Rewrite thresholds.pkl as thresholds.json -- the last pickle in the load path."""
    from ..checkpoints import load_thresholds, save_thresholds

    models_dir = Path(models_dir)
    src = models_dir / "thresholds.pkl"
    if not src.is_file():
        return None
    original = load_thresholds(src)
    out = save_thresholds(models_dir / "thresholds.json", original)
    if verify and load_thresholds(out) != original:
        raise RuntimeError(f"{out}: thresholds changed during conversion")
    return out


def convert_one(pkl_path, out_path=None, verify=True, metadata=None):
    """Convert one checkpoint. Returns (out_path, n_tensors)."""
    from ..checkpoints import load_flat_checkpoint, save_safetensors
    from ..head_table import METADATA_KEY, load_head_table, metadata_value

    pkl_path = Path(pkl_path)
    out_path = target_path(pkl_path) if out_path is None else Path(out_path)
    flat = load_flat_checkpoint(pkl_path)

    meta = {"source": pkl_path.name, "format": "hidra-jax-port-v1", "tensors": len(flat)}
    # A widened head's column list rides along in the metadata; the published checkpoints
    # carry none and keep being described by thresholds.json.
    table = load_head_table(pkl_path)
    if table is not None:
        meta[METADATA_KEY] = metadata_value(table)
        meta["n_heads"] = len(table)
    meta.update(metadata or {})
    save_safetensors(out_path, flat, metadata=meta)

    if verify:
        back = load_flat_checkpoint(out_path)
        if set(back) != set(flat):
            raise RuntimeError(f"{out_path}: tensor set changed during conversion")
        for key, want in flat.items():
            got = np.asarray(back[key])
            want = np.asarray(want)
            if got.shape != want.shape or got.dtype != want.dtype:
                raise RuntimeError(f"{out_path}: {key} changed shape/dtype "
                                   f"{want.shape}/{want.dtype} -> {got.shape}/{got.dtype}")
            # Bit equality, not approximate: conversion re-containers, it does not
            # compute. Compare raw bytes rather than values, so a NaN or a -0.0 anywhere in
            # the weights still has to survive intact. (`.view(uint8)` cannot be used --
            # several variables, like the running-stat counter `n`, are 0-d.)
            if np.ascontiguousarray(got).tobytes() != np.ascontiguousarray(want).tobytes():
                raise RuntimeError(f"{out_path}: {key} values changed during conversion")
        if table is not None and load_head_table(out_path) != table:
            raise RuntimeError(f"{out_path}: the head table did not survive conversion")
    return out_path, len(flat)


def main(argv=None):
    from .. import paths

    ap = argparse.ArgumentParser(
        prog="hidra-convert-weights", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--models-dir", default=None,
                    help=f"where the checkpoints live (default: {paths.models_dir()})")
    ap.add_argument("--one", metavar="PKL", help="convert a single .pkl (e.g. a fine-tuned head)")
    ap.add_argument("--out", metavar="PATH", help="output path, only with --one")
    ap.add_argument("--force", action="store_true", help="re-convert files that already exist")
    ap.add_argument("--check", action="store_true", help="report status and exit")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the bit-equality read-back (not recommended)")
    args = ap.parse_args(argv)

    models_dir = Path(args.models_dir or paths.models_dir())
    verify = not args.no_verify

    if args.one:
        from ..head_table import load_head_table

        out, n = convert_one(args.one, args.out, verify=verify)
        table = load_head_table(out)
        heads = f", {len(table)}-column head table in metadata" if table is not None else ""
        print(f"{args.one} -> {out} ({n} tensors{heads})")
        return 0

    todo = source_paths(models_dir)
    absent = [p for _, _, p in todo if not p.is_file()]
    if absent:
        sys.exit(f"ERROR: {len(absent)} source checkpoint(s) missing from {models_dir}, "
                 f"e.g. {absent[0].name}\nFetch them with: hidra-download-models")

    if args.check:
        done = [p for _, _, p in todo if target_path(p).is_file()]
        print(f"{len(done)}/{len(todo)} checkpoints converted in {models_dir}")
        for _, _, p in todo:
            print(f"  {'ok     ' if target_path(p).is_file() else 'missing'} "
                  f"{target_path(p).name}")
        th_ok = (models_dir / "thresholds.json").is_file()
        print(f"  {'ok     ' if th_ok else 'missing'} thresholds.json")
        return 0 if (len(done) == len(todo) and th_ok) else 1

    th = convert_thresholds(models_dir, verify=verify)
    if th is not None:
        print(f"  thresholds.pkl -> {th.name}")

    total = 0
    for i, (_config, _kind, pkl) in enumerate(todo, 1):
        out = target_path(pkl)
        if out.is_file() and not args.force:
            print(f"  [{i}/{len(todo)}] {out.name} exists, skipping")
            continue
        print(f"  [{i}/{len(todo)}] {pkl.name} -> {out.name}", flush=True)
        _, n = convert_one(pkl, out, verify=verify)
        total += n
    print(f"Done. {total} tensors written to {models_dir}")
    print("hidra.torch prefers the .safetensors files automatically when present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
