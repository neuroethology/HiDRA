# Changelog

## 0.2.0 — 2026-10-10

HiDRA 0.2.0 replaces the model with a simpler one, trained without the shared cross-lab head.
This entry describes what is different from the previous release.

### The model

The previous classifier was multi-task. Behind the frozen trunk, the per-lab tail fed two
output layers: the 82-column per-lab head that inference reads, and a shared 38-way head
(`out-proj`) trained on every video of every lab, including labs that have no scored head.
The new classifier drops the shared head and its loss term. Everything else about the
architecture is unchanged: trunk, feature merge, lab embedding, the 3 × (BiLSTM + FFN) tail,
and the per-lab head with the same 82 columns in the same order.

Because the shared head was the only thing that saw the labs without a scored head, those labs
no longer influence training, and the embedding rows of their lab slots are left untrained.

### The weights

All eleven files on the Hub change; the head table does not.

- The five self-supervised trunks are retrained.
- The five per-lab classifiers are new, trained on the retrained trunks with the architecture
  above. Each checkpoint has 177 tensors instead of 185: the eight `out-proj` tensors are gone.
- `thresholds.json` is unchanged.
- The bundled decision thresholds (`derived_thresholds_train.csv`) are re-calibrated on the new
  model's output. The previous file also carried pooled fallback rows; the new one has exactly one
  row per head.

Held-out mean F1 over the 82 heads is 0.613, which is within noise of the previous model.

### What this means if you use HiDRA

- Probabilities and thresholds are not comparable across the two releases. Keep a study on one
  set. To stay on the previous weights, pin them:

  ```bash
  hidra-download-models --revision 6b9770c99f9e6e20a89f7c2b68169a72b14631e5
  ```

  and pair them with that release's thresholds CSV (`--thresholds`).
- Do not use the previous thresholds with the new weights or vice versa.
- HiDRA versions before 0.2.0 cannot load the new weights, because they expect the `out-proj`
  subtree. 0.2.0 loads both the new weights and the old ones.

### The code

- Both backends build the model without the shared head by default. The JAX trainer trains on
  the per-lab loss alone; the PyTorch port takes `shared_head=False`. Setting `SHARED_HEAD=1`
  restores the previous multi-task model for reproducing the old release.
- Both loaders accept checkpoints with or without `out-proj`, so old checkpoints still load.
- `hidra-convert-weights --strip-shared-head` drops an inert `out-proj` from a per-lab checkpoint
  on conversion to safetensors.
- Head-only fine-tuning (`finetune.py train --mode head`) now keeps the warm-start checkpoint's
  input-normalisation statistics instead of re-estimating them on your videos. Re-estimating them
  shifted the inputs of the frozen tail and made head-only fine-tunes worse than the zero-shot
  head they started from. `--ddi-steps 256` gives the previous behaviour. Tail and embedding
  modes are unchanged.
- Model card, README and the docs describe the new model. The model card lists the revision of
  the previous weights.
- Version 0.1.0 → 0.2.0.

## 0.1.0

The multi-task model: 82-column per-lab head plus a shared 38-way head, on the original trunks.
