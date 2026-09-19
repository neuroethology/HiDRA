"""PyTorch ports of HiDRA's optimizer, EMA, checkpointing and fine-tuning loop.

These mirror `solution.Adam`, `solution.EMA`, `solution.CheckpointManager`,
`solution.Trainer` and `train_perlab_heads.SchedAdam`. The optimizer is written out
explicitly rather than delegating to `torch.optim.Adam`: the two are mathematically the
same update, but `solution.Adam` computes its bias corrections from `t` *before*
incrementing it and applies `eps` inside the square root's scaling, and writing it out
makes the correspondence auditable and lets a test assert step-for-step agreement.

Three details of the original are easy to miss and are reproduced deliberately:

* **The EMA lags one step.** `Trainer.train_step` folds the parameters into the EMA
  *before* applying the optimizer update, so the average trails the live weights by a step.
  It is the EMA weights -- not the raw ones -- that get evaluated and checkpointed, which is
  why the published checkpoints are averages.
* **Data-dependent init runs even on a warm start.** `Trainer.initial_state` runs 256
  batches at `stage="init"` after `create_variables`, so the classifier head's input
  standardization statistics are re-calibrated on the new data even when every weight was
  copied from a published checkpoint. Skipping it would leave the head reading its inputs
  through the source lab's statistics.
* **The frozen trunk is excluded from that.** The trunk is held outside the head's layer
  dict and pinned to `stage="eval"` at construction, so `set_stage` never reaches it and
  its statistics stay put. The port keeps the trunk out of the module tree for the same
  reason.
"""
import math
import os
import pickle
import time

import numpy as np
import torch

from .layers import STAGE_EVAL, STAGE_INIT, STAGE_TRAIN, jax_gather_index

# ------------------------------------------------------------------ optimizers

class Adam:
    """`solution.Adam`, operating on a list of parameters that carry `.grad`.

    Bias corrections are read off `t` *before* it is incremented, matching the original.
    At the first step that gives t=1, correction 1/(1-beta) -- the standard step-1
    correction -- so this agrees with `torch.optim.Adam` mathematically while keeping the
    original's exact expression order.
    """

    def __init__(self, params, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 1.0
        self.m1 = [torch.zeros_like(p) for p in self.params]
        self.m2 = [torch.zeros_like(p) for p in self.params]

    def lr_at(self, t):
        return self.learning_rate

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def _transform_grads(self, grads):
        """Hook for subclasses (clipping). Returns the gradient list to use."""
        return grads

    def _param_step(self, param, m1, m2, lr):
        step = m1 / (torch.sqrt(m2) + self.eps)
        return param - lr * step

    @torch.no_grad()
    def step(self):
        b1_correction = 1.0 / (1 - self.beta1 ** self.t)
        b2_correction = 1.0 / (1 - self.beta2 ** self.t)
        lr = self.lr_at(self.t)

        grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in self.params]
        grads = self._transform_grads(grads)

        for i, (param, grad) in enumerate(zip(self.params, grads, strict=True)):
            self.m1[i] += (grad - self.m1[i]) * (1 - self.beta1)
            self.m2[i] += (grad ** 2 - self.m2[i]) * (1 - self.beta2)
            m1 = self.m1[i] * b1_correction
            m2 = self.m2[i] * b2_correction
            param.copy_(self._param_step(param, m1, m2, lr))
        self.t += 1

    def state_dict(self):
        return {"t": self.t, "m1": [m.clone() for m in self.m1],
                "m2": [m.clone() for m in self.m2]}

    def load_state_dict(self, state):
        self.t = state["t"]
        for dst, src in zip(self.m1, state["m1"], strict=True):
            dst.copy_(src)
        for dst, src in zip(self.m2, state["m2"], strict=True):
            dst.copy_(src)


class SchedAdam(Adam):
    """`train_perlab_heads.SchedAdam`: cosine LR decay, decoupled weight decay, grad clip.

    All three are off by default, in which case this is identical to `Adam`.
    """

    def __init__(self, params, learning_rate, total=0, floor=0.0, weight_decay=0.0,
                 grad_clip=0.0, **kw):
        super().__init__(params, learning_rate, **kw)
        self.total = total
        self.floor = floor
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def lr_at(self, t):
        if not self.total:
            return self.learning_rate
        frac = min(max(t / self.total, 0.0), 1.0)
        return self.floor + (self.learning_rate - self.floor) * 0.5 * (1.0 + math.cos(math.pi * frac))

    def _transform_grads(self, grads):
        if not self.grad_clip:
            return grads
        # Global-norm clip over all gradients at once, as in the original.
        sq = sum(float(torch.sum(g * g)) for g in grads)
        gnorm = math.sqrt(sq) + 1e-12
        scale = min(1.0, self.grad_clip / gnorm)
        return [g * scale for g in grads]

    def _param_step(self, param, m1, m2, lr):
        step = m1 / (torch.sqrt(m2) + self.eps)
        if self.weight_decay:
            step = step + self.weight_decay * param
        return param - lr * step


# ------------------------------------------------------------------ EMA

class EMA:
    """`solution.EMA`: a debiased exponential moving average over every variable.

    Tracks *all* tensors, trainable or not, so a frozen layer's statistics are carried
    through unchanged. `values()` divides by `1 - decay**t`, which makes the average
    unbiased from the very first step.
    """

    def __init__(self, named_tensors, decay):
        self.decay = decay
        self.t = 1.0
        self.sums = {k: v.detach().clone() * (1.0 - decay) for k, v in named_tensors.items()}

    @torch.no_grad()
    def update(self, named_tensors):
        for key, value in named_tensors.items():
            self.sums[key] += (value.detach() - self.sums[key]) * (1 - self.decay)
        self.t += 1

    @torch.no_grad()
    def values(self):
        norm = 1 - self.decay ** self.t
        return {k: v / norm for k, v in self.sums.items()}


# ------------------------------------------------------------------ checkpointing

class CheckpointManager:
    """`solution.CheckpointManager`: keep the best `max_checkpoints`, stop on patience.

    Scores are negated when lower is better, so the list is always "higher is better".
    `checkpoints[0]` is the worst kept slot and `checkpoints[-1]` the best.
    """

    def __init__(self, checkpoint_dir, max_checkpoints=3, metric_name="obj",
                 lower_is_better=True, patience=None):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.checkpoints = [(None, float("-inf"))] * max_checkpoints

    def path(self, step):
        return os.path.join(self.checkpoint_dir, f"{step}.pkl")

    def write_file(self, state, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = {k: v.detach().to("cpu", torch.float32).numpy() for k, v in state.items()}
        with open(path, "wb") as f:
            pickle.dump(arrays, f)

    def update(self, state, step, metrics):
        score = metrics[self.metric_name]
        if np.isnan(score):
            raise ValueError(f"{self.metric_name} is NaN at step {step}; training diverged")
        if self.lower_is_better:
            score = -score

        if score > self.checkpoints[0][1]:
            stale = self.checkpoints[0][0]
            if stale is not None and os.path.isfile(self.path(stale)):
                os.remove(self.path(stale))
            self.checkpoints[0] = (step, score)
            self.checkpoints.sort(key=lambda x: x[1])
            self.write_file(state, self.path(step))

        best_step = self.checkpoints[-1][0]
        if self.patience is not None and best_step is not None:
            return (step - best_step) > self.patience
        return False


# ------------------------------------------------------------------ losses

def masked_bce(logits, labels, mask):
    """`MultiTaskPerLabModel._masked_bce`, term for term.

    `mask` is per-(sample, action) -- an action a video did not annotate contributes
    nothing -- and the normalizer is the total unmasked action count, not the element
    count, so a batch with few annotated behaviours is not up-weighted.
    """
    lp = torch.where(labels == 1,
                     torch.nn.functional.logsigmoid(logits),
                     torch.nn.functional.logsigmoid(-logits))
    lp = (lp * mask[:, None]).sum(dim=-1)
    w = mask.sum()
    nll = -lp.mean(dim=1).sum() / torch.clamp(w, min=1.0)
    return nll, w


def labels_in_action_space(batch, n_actions, sniffall_id=None, sniff_family_ids=(),
                           dtype=torch.float32):
    """`MultiTaskPerLabModel._labels37`: one-hot labels and the per-action supervision mask.

    Self-directed and cross-directed labels are stored as separate action-id tracks for the
    same (agent, target) pair and are summed here. `mask == 1` keeps only actions supervised
    by exactly one of the two tracks -- an action claimed by both would be ambiguous.

    With SNIFFALL, the merged `sniffall` channel is synthesized as the OR over the sniff
    family. The subtypes are mutually exclusive under an argmax label so the sum is already
    in {0, 1}; the clamp guards ties.
    """
    self_labels = torch.nn.functional.one_hot(batch["self_labels"].long(), n_actions).to(dtype)
    cross_labels = torch.nn.functional.one_hot(batch["cross_labels"].long(), n_actions).to(dtype)
    labels = self_labels + cross_labels

    lm = batch["self_label_mask"].to(dtype) + batch["cross_label_mask"].to(dtype)
    mask = ((lm == 1) & (batch["batch_mask"] == 1)[:, None]).to(dtype)

    if sniffall_id is not None:
        fam = list(sniff_family_ids)
        labels[..., sniffall_id] = labels[..., fam].sum(-1).clamp(0.0, 1.0)
        mask[..., sniffall_id] = mask[..., fam].sum(-1).clamp(0.0, 1.0)
    return labels, mask


def ddi_forward(head, batch):
    """One data-dependent-init pass: exactly the layers `compute_loss` applies.

    It is not enough to run `trunk_feats`. `Trainer.get_ddi_loop` calls `compute_loss`,
    which also applies BOTH output projections -- the shared 38-way head and the 82-way
    per-lab head -- so their input statistics get calibrated too. Under LABTAIL the shared
    head's *loss* is discarded, but the layer is still evaluated, so it still sees data.
    Calibrating only the trunk would leave `out-proj-perlab` reading its inputs through the
    source lab's statistics -- with the head layer being the one thing every mode trains.
    """
    x = head.trunk_feats(batch)
    head.layers["out-proj"](x)
    head.layers["out-proj-perlab"](x)
    return x


def perlab_loss(head, batch, head_columns=None, include_shared=False):
    """The fine-tuning objective: masked BCE on the per-lab head columns.

    `head_columns` is the 0/1 column selector (`LABTAIL_COLS` in the original) restricting
    supervision to the adopted lab's chosen behaviours. Without it, a single head inside a
    joint 82-way masked BCE receives roughly 1/82 of the gradient signal.

    `include_shared` adds the 38-way shared head's loss, which the original does for
    foundation training but *not* for LABTAIL fine-tuning (the shared head is frozen there
    and its loss would only add noise).
    """
    return perlab_loss_from_feats(head, head.trunk_feats(batch), batch, head_columns,
                                  include_shared)


def perlab_loss_from_feats(head, x, batch, head_columns=None, include_shared=False):
    """`perlab_loss` from the penultimate features `x` (`trunk_feats`' output, or a cache of
    it). `batch` supplies the labels, masks and lab ids only."""
    from .. import schema

    logits_perlab = head.layers["out-proj-perlab"](x)

    sniffall_id = schema.ACTIONS.value_to_idx.get("sniffall")
    fam = [schema.ACTIONS.value_to_idx[a] for a in schema.SNIFF_FAMILY
           if a in schema.ACTIONS.value_to_idx] if sniffall_id is not None else ()
    labels, mask = labels_in_action_space(batch, len(schema.ACTIONS), sniffall_id, fam,
                                          dtype=logits_perlab.dtype)

    Pb = head.P[jax_gather_index(batch["lab_id"], head.P.shape[0])].to(logits_perlab.dtype)
    labels_perlab = torch.einsum("bta,baj->btj", labels, Pb)
    mask_perlab = torch.einsum("ba,baj->bj", mask, Pb)
    if head_columns is not None:
        mask_perlab = mask_perlab * head_columns.to(mask_perlab.dtype)

    nll, w = masked_bce(logits_perlab, labels_perlab, mask_perlab)
    metrics = {"nll_perlab": (float(nll.detach()), float(w.detach()))}

    if include_shared:
        logits_shared = head.layers["out-proj"](x)
        nll_shared, w_shared = masked_bce(logits_shared, labels, mask)
        metrics["nll_shared"] = (float(nll_shared.detach()), float(w_shared.detach()))
        nll = nll + nll_shared
    metrics["nll"] = (float(nll.detach()), float(w.detach()))
    return nll, metrics


# ------------------------------------------------------------------ trainable-layer selection

# `train_perlab_heads.TAIL_TRAINABLE`: the per-lab tail, i.e. everything after the shared
# feature merge. The SSL trunk and the merge stay frozen during fine-tuning.
TAIL_TRAINABLE = {
    "lab-embedding", "lstm-0", "lstm-1", "lstm-2",
    "out-proj-0", "out-proj-1", "out-proj-2",
    "ff-0-in", "ff-0-out", "ff-1-in", "ff-1-out", "ff-2-in", "ff-2-out",
    "out-proj-perlab",
}
MERGE_LAYERS = {"ff-merge-in", "ff-merge-out", "feat-flat-proj"}

# The three modes finetune.py exposes.
MODE_LAYERS = {
    "head": {"out-proj-perlab"},
    "embedding": {"lab-embedding", "out-proj-perlab"},
    "tail": set(TAIL_TRAINABLE),
}


def set_trainable_layers(head, layer_names):
    """Freeze every parameter except those of the named layers.

    Returns (n_trainable_tensors, n_frozen_tensors). Freezing is per *layer*, matching the
    original's `_merge`, which flips `trainable` on every variable of a layer at once.
    """
    trainable, frozen = 0, 0
    for name, module in head.layers.items():
        want = name in layer_names
        for p in module.parameters():
            p.requires_grad_(want)
            if want:
                trainable += 1
            else:
                frozen += 1
    return trainable, frozen


def head_column_selector(lab_action, lab, actions=None, device=None):
    """`LABTAIL_COLS`: 1.0 on the columns for (lab, action), 0.0 elsewhere."""
    wanted = set(actions) if actions else None
    col = [1.0 if (l == lab and (wanted is None or a in wanted)) else 0.0
           for (l, a) in lab_action]
    out = torch.tensor(col, dtype=torch.float32, device=device)
    if float(out.sum()) == 0:
        raise ValueError(f"no head column matches lab={lab!r} actions={actions!r}")
    return out


# ------------------------------------------------------------------ the fine-tuning loop

class FineTuner:
    """`solution.Trainer` restricted to the fine-tuning path `finetune.py` actually drives.

    The generic Trainer also supports pretraining the trunk and foundation training over
    every lab; neither is reachable from the CLI, so neither is ported. What is here is the
    LABTAIL flow: warm-start from a published checkpoint, freeze all but the selected
    layers, re-calibrate the head's input statistics, then train with EMA and early
    stopping on a validation metric.
    """

    def __init__(self, head, optimizer, train_batches, eval_fn=None, ema_decay=0.9993,
                 checkpoint_dir=None, max_training_steps=8000, log_interval=500,
                 eval_interval=500, metric_name="f1", lower_is_better=False,
                 patience=10000, head_columns=None, ddi_steps=256, ddi_decay=0.99,
                 loss_fn=None):
        self.head = head
        self.optimizer = optimizer
        self.train_batches = train_batches
        self.eval_fn = eval_fn
        self.max_training_steps = max_training_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.head_columns = head_columns
        self.ddi_steps = ddi_steps
        self.ddi_decay = ddi_decay
        # `loss_fn(head, batch) -> (loss, metrics)`; default the live `perlab_loss`. The
        # cached-feature path (`FeatureCache`) supplies one that skips the frozen layers.
        self.loss_fn = loss_fn

        self.tracked = self._tracked_tensors()
        self.ema = EMA(self.tracked, ema_decay)
        self.checkpoints = CheckpointManager(
            checkpoint_dir or ".", metric_name=metric_name,
            lower_is_better=lower_is_better, patience=patience)
        self.last_ema_values = None

    def _tracked_tensors(self):
        """Every tensor of the head, params and buffers, keyed by name.

        Buffers are included because the input-standardization statistics are part of the
        model's state; the original's EMA tracks all variables, not only trainable ones.
        """
        out = dict(self.head.named_parameters())
        out.update({k: v for k, v in self.head.named_buffers() if k != "P"})
        return out

    def run_ddi(self, batches):
        """Re-calibrate the head's input standardization on the fine-tuning data.

        `solution.Trainer.initial_state` does this for 256 batches after building the
        variables, warm start or not. The statistics start from the checkpoint's values, so
        this shifts them toward the new data rather than rebuilding them from zero.
        """
        if not self.ddi_steps:
            return 0
        self.head.set_stage(STAGE_INIT, init_decay=self.ddi_decay)
        n = 0
        with torch.no_grad():
            for batch in batches:
                ddi_forward(self.head, batch)
                n += 1
                if n >= self.ddi_steps:
                    break
        self.head.set_stage(STAGE_TRAIN)
        # The EMA was seeded before calibration, so re-seed it from the calibrated state.
        self.ema = EMA(self._tracked_tensors(), self.ema.decay)
        return n

    def train_step(self, batch):
        self.head.set_stage(STAGE_TRAIN)
        if self.loss_fn is not None:
            loss, metrics = self.loss_fn(self.head, batch)
        else:
            loss, metrics = perlab_loss(self.head, batch, head_columns=self.head_columns)
        self.optimizer.zero_grad()
        loss.backward()
        # EMA folds in the weights BEFORE the update, matching Trainer.train_step, so the
        # average trails the live parameters by exactly one step.
        self.ema.update(self.tracked)
        self.optimizer.step()
        return metrics

    def evaluate(self):
        if self.eval_fn is None:
            return None
        ema_values = self.ema.values()
        self.last_ema_values = ema_values
        with swapped_tensors(self.head, ema_values):
            self.head.set_stage(STAGE_EVAL)
            metrics = self.eval_fn(self.head)
        self.head.set_stage(STAGE_TRAIN)
        return metrics

    def train(self, log=print):
        step = 0
        t_prev = time.time()
        running = {}
        while step < self.max_training_steps:
            if self.eval_fn is not None and step % self.eval_interval == 0:
                metrics = self.evaluate()
                if metrics is not None:
                    log(f"[EVAL step {step:>7}] "
                        + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                        + f"  ({time.time() - t_prev:.1f}s)")
                    if self.checkpoints.update(self.last_ema_values, step, metrics):
                        log("early stopping: validation metric stopped improving")
                        return step
                    t_prev = time.time()

            try:
                batch = next(self.train_batches)
            except StopIteration:
                log(f"training data exhausted after {step} steps")
                break

            metrics = self.train_step(batch)
            for key, (value, weight) in metrics.items():
                acc = running.setdefault(key, [0.0, 0.0])
                acc[0] += value * weight
                acc[1] += weight
            step += 1

            if step % self.log_interval == 0:
                summary = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in running.items()}
                log(f"[step {step:>7}] "
                    + "  ".join(f"{k}: {v:.4f}" for k, v in summary.items())
                    + f"  lr: {self.optimizer.lr_at(self.optimizer.t):.5f}"
                    + f"  ({time.time() - t_prev:.1f}s)")
                running = {}
                t_prev = time.time()

        # A final evaluation, so the returned EMA reflects the end of training. The original
        # saves the LAST captured EMA for LABTAIL runs rather than the best-scoring one.
        if self.eval_fn is not None:
            self.evaluate()
        elif self.last_ema_values is None:
            self.last_ema_values = self.ema.values()
        return step


class swapped_tensors:
    """Temporarily install a set of tensors (typically the EMA average) into a module.

    Evaluation and checkpointing use the averaged weights while training continues from the
    live ones, so the swap has to be reversible and must not disturb the optimizer state.
    """

    def __init__(self, module, values):
        self.module = module
        self.values = values
        self.saved = None

    def __enter__(self):
        tensors = dict(self.module.named_parameters())
        tensors.update(dict(self.module.named_buffers()))
        self.saved = {}
        with torch.no_grad():
            for key, value in self.values.items():
                if key in tensors:
                    self.saved[key] = tensors[key].detach().clone()
                    tensors[key].copy_(value)
        self.tensors = tensors
        return self.module

    def __exit__(self, *exc):
        with torch.no_grad():
            for key, value in self.saved.items():
                self.tensors[key].copy_(value)
        return False


# ------------------------------------------------------------------ cached frozen features

class FeatureCache:
    """Outputs of the frozen layers for a fixed set of augmented windows, so the trainable
    part of the model can be fit on them without re-running the trunk every step.

    The fine-tuning modes leave everything up to some point frozen: `head` trains the
    linear head on the tail's output, `embedding` and `tail` train the tail on the merge
    output `x0`. Whatever is frozen is a deterministic function of the augmented window, so
    it can be computed once per window and reused for as many optimizer steps as wanted.
    That turns the ~0.5-1 s live step -- bound by the numpy data pipeline and the trunk --
    into a millisecond one, at the price of a fixed set of augmentation draws: the cache
    holds `passes` epochs of the training windows, each with its own augmentation, and
    training cycles through them. This is the `CACHED_X0_DIR` fast path of the research
    code, generalised to the head's own features and computed in-process.

    `level` is "feats" (the tail's output, `trunk_feats`; for mode=head) or "x0" (the merge
    output, `merge_feats`; for mode=embedding/tail). Features are stored on the CPU in the
    dtype the model produced them (bfloat16 `x0` under the default compute dtype, float32
    `feats`), labels alongside; `train_batches` re-shuffles windows every epoch and moves
    one batch at a time to the device. Windows from the padding of a short final batch
    (`batch_mask == 0`) are dropped at build time, so every cached window is a real one.
    """

    LEVELS = {"head": "feats", "embedding": "x0", "tail": "x0"}
    LABEL_KEYS = ("self_labels", "cross_labels", "self_label_mask", "cross_label_mask", "lab_id")
    # What `data.Predictions.update` reads from an element, besides the probabilities.
    ELEMENT_KEYS = ("video_id", "agent_id", "target_id", "t_start", "t_end", "video_fps",
                    "self_label_mask", "cross_label_mask")

    def __init__(self, level):
        if level not in ("feats", "x0"):
            raise ValueError(f"level must be 'feats' or 'x0', got {level!r}")
        self.level = level
        self.feats = None
        self.labels = {}
        self.elements = None
        self.n = 0
        self.n_batches = 0

    @torch.no_grad()
    def build(self, head, batches, device, keep_elements=False):
        """Run the frozen layers over `batches` -- raw numpy batch dicts from `data.batch`,
        consumed to exhaustion -- and keep the results. `keep_elements` also keeps the
        per-window fields `evaluate` needs. Returns self."""
        from .infer import to_torch_batch

        head.set_stage(STAGE_EVAL)
        feats, labels, elements = [], {k: [] for k in self.LABEL_KEYS}, []
        for batch in batches:
            keep = np.asarray(batch["batch_mask"]) == 1
            self.n_batches += 1
            if not keep.any():
                continue
            tb = to_torch_batch(batch, device)
            if self.level == "x0":
                f = head.merge_feats(tb).permute(1, 0, 2)          # (B, tsteps, d_res)
            else:
                f = head.trunk_feats(tb)                           # (B, seq_len, d_res)
            idx = torch.as_tensor(np.flatnonzero(keep), device=device)
            feats.append(f.index_select(0, idx).to("cpu"))
            for k in self.LABEL_KEYS:
                labels[k].append(np.asarray(batch[k])[keep])
            if keep_elements:
                in_len = int(np.asarray(batch["agent"]).shape[1])
                for i in np.flatnonzero(keep):
                    el = {k: np.asarray(batch[k])[i].copy() for k in self.ELEMENT_KEYS}
                    # Predictions.update reads only the length of "agent" and the mask flag.
                    el["agent"] = np.zeros((in_len,), np.int8)
                    el["batch_mask"] = np.int8(1)
                    elements.append(el)
        if not feats:
            raise ValueError("no windows to cache: the dataset produced no valid elements")
        self.feats = torch.cat(feats, dim=0)
        self.labels = {k: torch.as_tensor(np.concatenate(v, axis=0)) for k, v in labels.items()}
        # `to_torch_batch` hands the model a long lab_id; keep the cache's identical.
        self.labels["lab_id"] = self.labels["lab_id"].long()
        self.elements = elements if keep_elements else None
        self.n = int(self.feats.shape[0])
        return self

    @property
    def nbytes(self):
        n = self.feats.numel() * self.feats.element_size()
        return n + sum(v.numel() * v.element_size() for v in self.labels.values())

    def _batch(self, idx, device):
        idx = torch.as_tensor(idx, dtype=torch.long)
        out = {"feats" if self.level == "feats" else "x0": self.feats[idx].to(device)}
        for k, v in self.labels.items():
            out[k] = v[idx].to(device)
        # Every cached window is a real one: the padding of a short final batch was dropped
        # at build time, so `batch_mask` is all ones and nothing is masked out of the loss.
        out["batch_mask"] = torch.ones(len(idx), dtype=torch.int8, device=device)
        return out

    def train_batches(self, batch_size, device, seed=0):
        """An endless stream of device-resident batches, re-shuffling every epoch. A final
        short epoch batch is dropped rather than padded, unless it is the only one."""
        g = torch.Generator().manual_seed(int(seed))
        while True:
            perm = torch.randperm(self.n, generator=g)
            n_full = max(self.n // batch_size, 1)
            for b in range(n_full):
                yield self._batch(perm[b * batch_size:(b + 1) * batch_size], device)

    def features(self, head, batch):
        """The penultimate features for a cached batch: the cache itself at level "feats",
        or the tail run live on cached `x0` (whose tail layers are what is being trained)."""
        if self.level == "feats":
            return batch["feats"]
        x0 = batch["x0"].permute(1, 0, 2)                          # (tsteps, B, d_res)
        return head.tail_feats(x0, batch["lab_id"])

    def loss_fn(self, head_columns=None):
        """A `FineTuner.loss_fn` computing `perlab_loss` from the cache."""
        def fn(head, batch):
            return perlab_loss_from_feats(head, self.features(head, batch), batch, head_columns)
        return fn

    @torch.no_grad()
    def evaluate(self, head, videos, batch_size, device):
        """Validation metrics from the cached windows (built with `keep_elements=True`), via
        the same `data.Predictions.score()` the live evaluation uses."""
        from .. import data

        if self.elements is None:
            raise ValueError("evaluate needs a cache built with keep_elements=True")
        predictions = data.Predictions(videos)
        for start in range(0, self.n, batch_size):
            idx = torch.arange(start, min(start + batch_size, self.n))
            batch = self._batch(idx, device)
            probs = head.predict_from_feats(self.features(head, batch), batch["lab_id"])
            probs = probs.detach().to("cpu", torch.float32).numpy()
            for i, j in enumerate(idx.tolist()):
                predictions.update(self.elements[j], probs[i])
        metrics, _ = predictions.score()
        return {k: float(v) for k, v in metrics.items()}
