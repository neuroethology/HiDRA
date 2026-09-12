"""The per-lab head table, and how a checkpoint carries its own copy of it.

`out-proj-perlab` has one column per (lab, action) pair. Which pairs, in which order, is the
*head table*. For the published checkpoints it is `schema.lab_action_table()` -- the keys of
`models/thresholds.json`, minus PleasantMeerkat's non-`follow` heads, plus `sniffall` for the
five sniff-splitting labs: 82 columns.

`finetune.py train --new-head Lab,action` builds a checkpoint one column wider. Such a
checkpoint can no longer be described by thresholds.json, so it carries its table with it:

* a `.pkl` checkpoint gets a sidecar `<stem>.heads.json` next to it;
* a `.safetensors` checkpoint carries the same list in its metadata header (key
  `lab_action`), which `hidra-convert-weights --one` writes from the sidecar.

`load_head_table` reads whichever is present, and both backends' loaders ask it before
falling back to the published table, so `predict.py --weights` works without editing
thresholds.json. `remap_head_columns` is the column copy that widens a checkpoint: the
generalisation of what `train_perlab_heads.FROZEN_TRUNK` hard-wires for sniffall.

Everything here is numpy and JSON; neither backend is imported. (The module is not called
`heads` because `hidra.heads` is the public API function in `hidra/__init__.py`, and a
submodule of that name would shadow it.)
"""
import json
from pathlib import Path

import numpy as np

HEAD_LAYER = "out-proj-perlab"
SIDECAR_SUFFIX = ".heads.json"
METADATA_KEY = "lab_action"
TABLE_FORMAT = "hidra-head-table-v1"


# ------------------------------------------------------------------ the table itself

def as_pairs(table):
    """Normalise a table (lists from JSON, tuples from schema) to a list of (lab, action)."""
    return [(str(lab), str(action)) for lab, action in table]


def parse_head_spec(spec):
    """'Lab,action' -> ('Lab', 'action'); anything else is a ValueError."""
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"expected 'Lab,action', got {spec!r}")
    return parts[0], parts[1]


def extend_table(table, new_pairs):
    """`table` plus the new (lab, action) pairs, sorted like `schema.lab_action_table()`.

    Sorted rather than appended so the convention "the table is sorted" that the published
    checkpoint and the sniffall extension both follow keeps holding. Column positions are
    never assumed anywhere -- the loaders go through the persisted table -- so the order is
    a convention, not a contract.
    """
    table = as_pairs(table)
    new = as_pairs(new_pairs)
    present = set(table)
    dup = [p for p in new if p in present]
    if dup:
        raise ValueError(f"{dup} already have a head column")
    return sorted(present | set(new))


def validate_new_head(lab, action, table, seed_from=None):
    """The checks behind `--new-head` (and `--seed-from`), worded for the command line.

    The action must be an existing vocabulary name: adding a *name* changes the label space
    every checkpoint was trained in and is out of scope (docs/new-behaviours.md §5). The lab
    must already have heads, because everything downstream -- the inference driver's lab list,
    `--labs` -- enumerates labs from the head table.
    """
    from . import schema

    table = as_pairs(table)
    if lab not in schema.LABS.value_to_idx:
        raise ValueError(f"unknown lab {lab!r}; the labs are {schema.LABS.values}")
    labs_with_heads = sorted({l for l, _ in table})
    if lab not in labs_with_heads:
        raise ValueError(f"{lab} has no published head, so a column cannot be added under it; "
                         f"labs with heads: {labs_with_heads}")
    if action == "none" or action not in schema.ACTIONS.value_to_idx:
        raise ValueError(f"{action!r} is not a behaviour in the vocabulary "
                         f"({', '.join(a for a in schema.ACTIONS.values if a != 'none')}). A new head "
                         f"needs an existing name; adding a name is not supported -- see "
                         f"docs/new-behaviours.md")
    if (lab, action) in set(table):
        raise ValueError(f"({lab}, {action}) already has a head column; fine-tune it with "
                         f"--actions {action} instead of --new-head")
    if seed_from is not None:
        seed_from = tuple(seed_from)
        if seed_from not in set(table):
            who = sorted(l for l, a in table if a == seed_from[1])
            raise ValueError(f"--seed-from {seed_from[0]},{seed_from[1]} is not an existing head"
                             + (f"; labs with a {seed_from[1]!r} head: {who}" if who else ""))
    return lab, action


# ------------------------------------------------------------------ persistence

def sidecar_path(ckpt_path):
    """`ft_models/15fps_5bp__tag.pkl` -> `ft_models/15fps_5bp__tag.heads.json`."""
    return Path(ckpt_path).with_suffix(SIDECAR_SUFFIX)


def save_head_table(ckpt_path, table):
    """Write the sidecar describing `ckpt_path`'s columns. Returns the sidecar path."""
    table = as_pairs(table)
    path = sidecar_path(ckpt_path)
    doc = {"format": TABLE_FORMAT, "checkpoint": Path(ckpt_path).name, "n_heads": len(table),
           METADATA_KEY: [[lab, action] for lab, action in table]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return path


def metadata_value(table):
    """The head table as the string stored under `lab_action` in safetensors metadata."""
    return json.dumps([[lab, action] for lab, action in as_pairs(table)])


def _parse_table(obj, source):
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict):
        obj = obj.get(METADATA_KEY)
    ok = isinstance(obj, list) and all(
        isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, str) for x in p)
        for p in obj)
    if not ok:
        raise ValueError(f"{source}: malformed head table; expected a list of [lab, action] pairs")
    table = [tuple(p) for p in obj]
    if len(set(table)) != len(table):
        raise ValueError(f"{source}: the head table lists a (lab, action) twice")
    return table


def load_head_table(ckpt_path):
    """The (lab, action) column list a checkpoint carries, or None when it carries none --
    in which case the published table (`schema.lab_action_table()`) describes it.

    safetensors metadata first, then the `.heads.json` sidecar. The checkpoint itself is not
    opened for a `.pkl`, so this is cheap enough to call from a CLI before any model loads.
    """
    path = Path(ckpt_path)
    if path.suffix == ".safetensors" and path.is_file():
        from .checkpoints import read_metadata

        meta = read_metadata(path)
        if METADATA_KEY in meta:
            return _parse_table(meta[METADATA_KEY], f"{path} metadata")
    side = sidecar_path(path)
    if side.is_file():
        with open(side) as f:
            return _parse_table(json.load(f), str(side))
    return None


def head_table_for(ckpt_path):
    """`load_head_table`, falling back to the published table."""
    table = load_head_table(ckpt_path)
    if table is None:
        from . import schema

        table = schema.lab_action_table()
    return as_pairs(table)


def head_table_for_template(template, config_names):
    """The table shared by the checkpoints of a `{config}` template, over the configs that
    exist on disk. None when none of them carries a table (the published table applies);
    ValueError when they disagree, since inference averages them and needs one width.
    """
    tables, plain = {}, []
    for name in config_names:
        path = Path(template.format(config=name))
        if not path.is_file():
            continue
        table = load_head_table(path)
        if table is None:
            plain.append(name)
        else:
            tables[name] = table
    if not tables:
        return None
    if plain:
        raise ValueError(f"{sorted(tables)} carry a head table but {plain} do not (template "
                         f"{template}); a fine-tune must be trained the same way for every config")
    distinct = {tuple(t) for t in tables.values()}
    if len(distinct) > 1:
        raise ValueError(f"the checkpoints of {template} disagree on their head tables: "
                         + "; ".join(f"{k}: {len(v)} columns" for k, v in sorted(tables.items())))
    return as_pairs(next(iter(tables.values())))


# ------------------------------------------------------------------ the column remap

def check_head_width(flat, table, source="the checkpoint"):
    """Refuse a checkpoint whose head width does not match `table`, with the fix spelled out.

    Without this the failure is a bare shape-mismatch from the loader; the user's actual
    mistake is almost always a widened checkpoint separated from its sidecar.
    """
    w = flat.get(f"{HEAD_LAYER}/w")
    if w is None:
        return
    n = int(np.asarray(w).shape[-1])
    if n != len(table):
        raise ValueError(
            f"{source} has {n} head columns but the head table has {len(table)} entries. A "
            f"checkpoint with extra (lab, action) columns must travel with its own table: the "
            f"`{SIDECAR_SUFFIX}` sidecar next to a .pkl, or the `{METADATA_KEY}` metadata of a "
            f".safetensors (finetune.py train writes the sidecar; hidra-convert-weights --one "
            f"carries it into the metadata).")


def remap_head_columns(flat, old_table, new_table, seed_from=None, seed=0):
    """Widen `out-proj-perlab` from `old_table`'s columns to `new_table`'s.

    Returns `(new_flat, new_columns)`: a shallow copy of `flat` with the head's `w`
    (in, n), `s` (n,) and `b` (n,) replaced, and `{(lab, action): column}` for the columns
    that did not exist before. Every old column is copied *by name* to its new position, bit
    for bit, so the order of the two tables need not be related. New columns start from
    `seed_from`'s column when given -- an existing (lab, action) whose behaviour resembles the
    new one, the trick that made the sniffall heads converge -- and otherwise from the init
    `Linear.create_variables` would draw: w ~ N(0, 1) / (sqrt(max(in, out)) / 8), s = 1,
    b = 0. The input-standardization statistics (mean/std/m1/m2/n) are per *input* feature
    and are not touched.
    """
    old_table, new_table = as_pairs(old_table), as_pairs(new_table)
    keys = {k: f"{HEAD_LAYER}/{k}" for k in ("w", "s", "b")}
    w, s, b = (np.asarray(flat[keys[k]]) for k in ("w", "s", "b"))
    if w.shape[-1] != len(old_table):
        raise ValueError(f"{keys['w']} has {w.shape[-1]} columns but old_table lists "
                         f"{len(old_table)} pairs")
    if len(set(new_table)) != len(new_table):
        raise ValueError("new_table lists a (lab, action) twice")
    position = {pair: j for j, pair in enumerate(new_table)}
    dropped = [pair for pair in old_table if pair not in position]
    if dropped:
        raise ValueError(f"new_table drops existing columns {dropped}; only widening is supported")

    n_in, n_new = w.shape[-2], len(new_table)
    rng = np.random.default_rng(seed)
    lrmul = np.sqrt(max(n_in, n_new)) / np.sqrt(64)
    new_w = (rng.standard_normal(w.shape[:-1] + (n_new,)) / lrmul).astype(w.dtype)
    new_s = np.ones(s.shape[:-1] + (n_new,), dtype=s.dtype)
    new_b = np.zeros(b.shape[:-1] + (n_new,), dtype=b.dtype)
    for i, pair in enumerate(old_table):
        j = position[pair]
        new_w[..., j] = w[..., i]
        new_s[..., j] = s[..., i]
        new_b[..., j] = b[..., i]

    new_columns = {pair: position[pair] for pair in new_table if pair not in set(old_table)}
    if seed_from is not None:
        seed_from = tuple(seed_from)
        if seed_from not in set(old_table):
            raise ValueError(f"seed_from {seed_from} is not a column of old_table")
        i = old_table.index(seed_from)
        for j in new_columns.values():
            new_w[..., j] = w[..., i]
            new_s[..., j] = s[..., i]
            new_b[..., j] = b[..., i]

    out = dict(flat)
    out[keys["w"]], out[keys["s"]], out[keys["b"]] = new_w, new_s, new_b
    return out, new_columns
