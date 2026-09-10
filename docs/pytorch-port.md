# The PyTorch port

HiDRA's models were written in a hand-rolled JAX framework. They now also run on PyTorch,
which is the default backend. This document records what was ported, how the two backends
were shown to agree, what genuinely differs, and what is left before the JAX dependency can
be removed.

```bash
python predict.py tracking/ --out results/ --pix-per-cm 16 --fps 30   # torch (default)
python predict.py tracking/ --out results/ --pix-per-cm 16 --fps 30 --backend jax
```

Both read the same published weights. `--backend jax` needs the `[jax]` extra installed.

## Headline result

Measured on an RTX A6000 with the published checkpoints:

| level | comparison | result |
|---|---|---|
| layer | 12 `Linear` shapes, `LSTM`, `BiLSTM`, `Constant`, `Embedding` | **~5e-7** relative |
| model | 4 trunk feature tensors, 82 head logits, 38 probabilities, all 5 configs | **~5e-7** relative |
| end-to-end | 43,200 thresholded binary calls | **identical** |
| end-to-end | bout boundaries | **identical** |
| end-to-end | per-frame probability | max 1.2e-3, correlation 0.9999993 |

Full five-config ensemble, one lab (12 heads x 4 mouse pairs x 2 videos, 43,200 frame-rows):
every binary call identical, every bout boundary identical, worst per-frame probability
difference 7.3e-4, correlation 0.9999998.

It is also faster and lighter, which was not a goal but is worth having:

| | wall time | peak RSS |
|---|---|---|
| JAX | 49.1 s | 3.83 GB |
| PyTorch | 6.2 s | 1.62 GB |

Same workload, same GPU. Most of the gap is start-up: the JAX path pays XLA compilation for
each of the five configs, and needs `--xla_gpu_autotune_level=0` set for the 6-bodypart
configs or they hang. The torch path needs neither.

## Why "bit-exact" was the wrong target

The obvious test — diff the two backends and demand agreement to the last bit — is
unachievable here, and chasing it would have hidden the real finding.

**JAX has been computing the model's float32 matmuls in TF32.** That is the default float32
matmul precision on Ampere and later, and TF32 keeps 10 mantissa bits instead of 24.
Measured against a float64 reference on a single `Linear` layer:

| | error vs float64 |
|---|---|
| PyTorch, float32 (default) | 2.3e-07 |
| JAX, `precision="default"` | 2.6e-04 |
| JAX, `precision="tensorfloat32"` | 2.6e-04 |
| JAX, `precision="float32"` | 5.7e-07 |

So the *reference* implementation is the imprecise one, by about three orders of magnitude,
and the port is a more accurate evaluation of the same model. A backend diff would have
measured JAX's TF32 error and reported the port as broken.

The tests therefore compare three ways:

- **port vs JAX pinned to `precision="float32"`** — tight (1e-5 at layer level, 1e-4 whole
  model). This is the question "did we port the same math?"
- **port vs a float64 reference** — how accurate the port is, and an assertion that it is
  never *less* accurate than default-precision JAX.
- **port vs JAX at its default precision** — reported, not asserted tightly. This is the
  drift a user actually sees, and it is dominated by TF32, not by the port.

`tests/exactness.jax_matmul_precision` is the switch that separates "the port is wrong" from
"the GPU used fewer mantissa bits".

This does not mean the published results are wrong. TF32 error at 1e-3 on probabilities is
far below the noise floor of behaviour classification, and every binary call and bout
boundary is unchanged. It does mean the port is the better numerical path going forward.

## What was ported

```
hidra/torch/layers.py       Linear, Constant, Embedding, LSTM, BidirectionalLSTM
hidra/torch/models.py       UnsupervisedModel (trunk), MultiTaskPerLabModel (head)
hidra/torch/checkpoint.py   read the published pickles with numpy alone; write safetensors
hidra/torch/convert.py      hidra-convert-weights
hidra/torch/infer.py        the inference loop
hidra/torch/train.py        Adam, SchedAdam, EMA, CheckpointManager, the loss, FineTuner
hidra/torch/train_perlab.py the fine-tuning driver
hidra/schema.py             label vocabularies, ensemble configs, per-lab head table
```

Nothing maps onto a stock `torch.nn` layer, which is why they are written out:

**`Linear` is three operations.** A frozen per-feature input standardization
(`x = (x - mean) / std`, from running moments accumulated during a data-dependent-init pass
and then held fixed — not a LayerNorm, the statistics are global); weight normalization
(the stored `w` is L2-normalized over the input axis and rescaled by a learned per-output
gain `s`); and the matmul as `einsum("...i,...io->...o")`, so `batch_dims` give independent
weights per bodypart or per bodypart *pair* rather than one shared projection.

**`LSTM`** has learned initial states (`h0` through `tanh`, `c0` raw), gates ordered
`i, f, c, o` as contiguous quarters, one bias vector, and optional per-bodypart weight sets.
`torch.nn.LSTM` matches none of that.

Only the inference and LABTAIL fine-tuning paths are ported. `train_perlab_heads.py` also
carries FROZEN_TRUNK, REFIT, SCALE, LOLO, DISENTANGLE, CORAL, FILM, SKIP_PATH and cached-
feature fast paths, all gated on environment variables that the CLI never sets — `cmd_train`
explicitly strips them. They are absent rather than ported unused; adding one back is a
deliberate act, not an accident.

## Real behavioural differences between JAX and PyTorch

These are places where the two frameworks genuinely disagree and the port had to choose.

**Out-of-bounds gathers.** `solution.batch` pads a short final batch with `np.empty_like` —
uninitialized memory — and flags it `batch_mask=0`. So `batch["lab_id"]` legitimately holds
garbage in padding rows (1072902963 was observed). JAX's gather clamps such indices and
produces a value that `Predictions.update` then discards; PyTorch raises a device-side
assert and kills the run. `jax_gather_index` reproduces JAX's rule exactly — negatives wrap
once by adding the size, then everything clamps — so padding rows agree across backends and
whole-tensor comparisons stay clean. (The clamping is XLA's, so it only appears under `jit`;
eagerly the layer holds numpy weights and numpy raises. The gather test traces the JAX side
accordingly, which is how the real pipeline runs it.)

**In-place addition.** Every JAX `x += y` became `x = x + y`. In JAX that statement is
functional and broadcasts freely, and the trunk depends on it: `x` starts at shape
`(1, 1, n_bp, d_res)` and the first `+=` broadcasts it out to `(T, B, n_bp, d_res)`. An
in-place torch add would raise instead.

**Type promotion.** `jnp.einsum` promotes a bfloat16-by-float32 product to float32; torch
raises. Fine-tuning runs the model in bfloat16 while the head-routing matrix `P` is always
float32, so `predict()` promotes explicitly.

**NaN is load-bearing.** Missing keypoints arrive as NaN and the model routes them
deliberately: a learned `nan-emb` substitutes for absent lag differences, and
`where(isnan(...), 0, ...)` zeroes edges to unobserved bodyparts. The order matters — the
divide happens *before* the where, so the NaN it produces is what gets replaced.

## Training

A training run cannot be compared the way inference is: float32 round-off in the first
gradient changes every subsequent batch's parameters, so the two backends diverge from step
one. The argument is assembled from pieces that *are* exactly checkable:

| piece | result |
|---|---|
| Adam / SchedAdam, 25 steps (cosine, weight decay, grad clip, all three) | within 1e-5 |
| EMA, 30 updates | 1e-6; step counters exact |
| loss vs `compute_loss` under real LABTAIL env | 1.1e-7 |
| analytic gradients vs float64 central differences | 7 significant figures, eps 1e-3…1e-7 |
| data-dependent-init statistics | mean/std match; counters exact |

The gradient check is what pins the backward pass, and it needs no JAX autodiff: if the
forward matches JAX's forward, and the backward is the verified derivative of that forward,
then the backward matches JAX's too.

Two things worth knowing about the training path:

**The EMA is what gets saved.** `Trainer.train_step` folds parameters into the EMA *before*
applying the optimizer update, so the average trails the live weights by one step, and it is
the EMA — not the raw weights — that is evaluated and checkpointed. The published checkpoints
are averages. A consequence: a fine-tuned checkpoint's *frozen* layers are not bit-identical
to the published ones. `values() = sums / (1 - decay**t)` recovers a constant weight exactly
in real arithmetic but not in float32, and the EMA tracks every variable, trainable or not.
The drift is ~8e-7 while a trained head moves 13–19%. The JAX original does the identical
round-trip.

**Data-dependent init runs even on a warm start.** `Trainer.initial_state` runs 256 batches
at `stage="init"` after building the variables, so the classifier head's input
standardization is re-calibrated on the new data even when every weight was copied from a
published checkpoint. The frozen trunk is excluded — it is held outside the head's layer dict
and pinned to `stage="eval"`, so `set_stage` never reaches it. The port keeps the trunk out
of the module tree for the same reason. Both output projections *are* included, because
`compute_loss` applies them.

## Checkpoints

`hidra-convert-weights` rewrites the 10 published pickles as safetensors, verifying byte
equality on read-back. `hidra.torch` prefers `.safetensors` when present and falls back to
`.pkl`, and a test asserts inference is bit-identical from either container.

The conversion needs no JAX. The pickles reference exactly one JAX symbol —
`jax._src.array._reconstruct_array`, whose only job is to rebuild a numpy array and
`device_put` it — so intercepting it lets the weights be read with numpy alone. That also
means users can convert their own fine-tuned checkpoints without installing JAX.

A fine-tune done on either backend is written in the **JAX variable layout**, so the
resulting checkpoint loads under either backend through `predict.py --weights`. Fine-tuning
costs five configs of GPU time; a checkpoint should not be stranded by a backend choice.

## JAX is no longer required

The numpy data pipeline now lives in `hidra/data.py`, which imports numpy and nothing else.
Inference and fine-tuning run with JAX absent:

```bash
uv sync --extra torch     # no jax
python predict.py tracking/ --out results/ --pix-per-cm 16 --fps 30
python finetune.py train --data ft_data/ --lab GroovyShrew --actions rear --out ft_models/
```

`tests/test_endtoend_exactness.py::test_full_torch_inference_needs_no_jax` asserts it by
running the real CLI with `jax` blocked in `sys.meta_path` — via a `sitecustomize.py` on
`PYTHONPATH`, not just in the test process, because `cli.run` shells out to a subprocess that
would otherwise be free to import it.

`hidra.data` holds `TrackingData`, `Labels`, `Video`, `Dataset`, `Predictions`, `F1` and the
stream helpers. `solution.py` re-exports every name — the same objects, so
`solution.Dataset is data.Dataset` and a driver monkeypatching one affects both backends —
and forwards reads *and writes* of the four path globals (`dataset_dir`, `working_dir`,
`persist_dir`, `project_dir`) onto `data`. Forwarding writes matters: entry points have
always set them as `solution.working_dir = ...`, and a plain re-import would let such an
assignment shadow the real value, so the tracking cache would be built where no reader looks
and nothing would complain.

The `jax` extra is now needed only for `--backend jax` and for the exactness tests that
compare the backends against each other.

### Two hazards the split exposed

Both were caught by the end-to-end comparison, and both now have regression tests in
`tests/test_data_pipeline.py`.

**The stream helpers are not flat-dict-only.** `predict_into` maps the stream through a
function returning `(batch, probs)`, so everything downstream of it — `to_host`, `unbatch`,
`get_batch_size` — carries a *tuple* whose first element is the batch dict. `jax.tree.*`
handled that structure for free. A dict-only replacement raised `'tuple' object has no
attribute 'items'` the moment the JAX engine ran, so `data.py` carries real `tree_map` /
`tree_leaves` / `tree_stack` over dicts, lists and tuples.

**`build_P` must not be built through a patched `LABS.encode`.** The inference drivers force
one lab's embedding by *replacing* that attribute with a constant. A P built through it
routes every lab's head into a single row, and the readout einsum then sums all 82 heads
instead of selecting one. The original guarded against this by capturing the encoder at
import time — which only works while `train_perlab_heads` is imported *before* the patch is
applied. Making the JAX engine import lazily (so the torch path does not drag it in) broke
that ordering, and probabilities came out above 3.0. Both `build_P` implementations now index
`value_to_idx`, a plain dict the patch never touches, which removes the ordering dependency
rather than depending on it. The end-to-end test additionally asserts that every probability
lies in [0, 1] on both backends, which localizes this class of bug instantly.

## Reproducing the numbers

```bash
uv sync --extra jax --extra torch      # both backends in one environment
hidra-download-models                  # ~660 MB
python -m pytest -q                    # 91 tests, ~5 min on one A6000
python -m pytest -q -m "not slow"      # ~1.5 min
```

The suite needs a CUDA device and the JAX extra; tests skip cleanly without them. `reference/`
holds the pre-port implementation byte-identically (guarded by
`tests/test_reference_pristine.py`) and is the ground truth every comparison is made against.

Test data is synthetic and deterministic (`tests/synth.py`): two mice on a correlated random
walk with a rotating body frame, dropped keypoints, and scripted approach-and-hold episodes.
The episodes matter — without them every social head sits at its ~1e-4 floor and an
end-to-end probability comparison is nearly insensitive.
