# Pristine JAX reference implementation

These are the original files, byte-for-byte as of commit `9aed891` (before the `src/hidra`
restructure and the PyTorch port). **Do not edit them.** They exist so the port has a
reference that can be *run*, not just read:

- `tests/` compares PyTorch layer/module/end-to-end outputs against numbers produced by
  this code, on the same weights and the same inputs.
- When a discrepancy shows up, this is the ground truth to bisect against.

`models` is a symlink to the repo-root `models/`, so the reference resolves its weights out
of the same download the package uses (`solution.persist_dir = <this dir>/models`).

Running it needs the JAX backend and the original flat-module import layout, so invoke it
with `reference/` as the working directory:

```bash
uv sync --extra jax
cd reference && ../.venv/bin/python predict.py --list-heads
```

Verify the files still match the original commit:

```bash
git show 9aed891 --stat            # the commit these came from
python tests/test_reference_pristine.py   # hashes reference/ against git
```
