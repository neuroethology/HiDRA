"""Torch inference loop -- the PyTorch counterpart of
`run_test_probs_perlab.predict_into`.

The data pipeline is *deliberately shared* with the JAX path: this calls the very same
`hidra.data.Dataset` and `hidra.data.Predictions`, which are pure numpy and seeded
deterministically. Only the model is swapped. That is what makes a backend comparison
meaningful -- identical augmented clips in, identical accumulation out, so any difference
in the probability tracks is attributable to the model arithmetic and nothing else.

`hidra.data` imports numpy and nothing else, so this module -- and the whole PyTorch
inference path -- needs no JAX runtime.
"""
import os

import numpy as np
import torch

# Batch keys the model consumes, and the dtype each must arrive in.
_MODEL_INPUTS = {
    "agent": torch.float32,
    "target": torch.float32,
    "augmentation_params": torch.float32,
    "lab_id": torch.long,
}


def default_batch_size():
    """PREDICT_BATCH, matching the env var the JAX path honours (cli.py sets it to 64)."""
    return int(os.environ.get("PREDICT_BATCH", "256"))


def make_dataset(config, videos, num_epochs=1, seed=None):
    """The eval dataset, with exactly the arguments `predict_into` uses in the JAX path.

    The 0.95 factors on scale/time-dilation and the rotate/flip flags are not incidental:
    inference averages several augmented passes per video, so the augmentation
    distribution is part of the model's definition, not a training-time detail.
    """
    from .. import data

    return data.Dataset(
        videos=videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=num_epochs,
        unsupervised=False,
        max_scale=0.95 * config["max_scale"],
        max_time_dilation=0.95 * config["max_time_dilation"],
        rotate=True,
        flip=True,
        noise_scale=config["noise_scale"],
        num_workers=3,
        seed=[1] + config["eval_seed"] if seed is None else seed,
    )


def to_torch_batch(batch, device):
    """The numpy batch dict -> the tensors the ported model wants, on `device`."""
    out = {}
    for key, dtype in _MODEL_INPUTS.items():
        out[key] = torch.as_tensor(np.asarray(batch[key]), dtype=dtype, device=device)
    return out


def unbatch_numpy(batch, index):
    """One element out of a batched numpy dict, mirroring `data.unbatch`."""
    return {k: v[index].copy() for k, v in batch.items()}


def iter_batches(dataset, batch_size):
    """Batched elements from the dataset, using the JAX path's own batching helper.

    `data.batch` pads a short final batch and flags it with `batch_mask=0`, which
    `Predictions.update` then skips. Reusing it keeps the element-to-batch assignment --
    and therefore the padding behaviour -- identical across backends.
    """
    from .. import data

    yield from data.batch(dataset.element_iterator(), batch_size)


@torch.no_grad()
def predict_into(config, predictions, head, num_epochs=1, device=None, batch_size=None,
                 progress=None):
    """Run the ported model over `predictions.videos` and accumulate into `predictions`.

    `head` is a loaded `MultiTaskPerLabModel`. Returns the number of elements consumed.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = default_batch_size() if batch_size is None else batch_size
    dataset = make_dataset(config, predictions.videos, num_epochs=num_epochs)

    n = 0
    for batch in iter_batches(dataset, batch_size):
        probs = head.predict(to_torch_batch(batch, device))
        probs = probs.detach().to("cpu", torch.float32).numpy()
        for i in range(probs.shape[0]):
            predictions.update(unbatch_numpy(batch, i), probs[i])
            n += 1
        if progress is not None:
            progress(n)
    return n


@torch.no_grad()
def logits_for_batch(head, batch, device=None):
    """Per-lab head logits for one already-built numpy batch. Used by the exactness tests,
    where comparing pre-sigmoid values is far more sensitive than comparing probabilities
    that mostly sit near zero."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return head.logits_perlab(to_torch_batch(batch, device))


@torch.no_grad()
def trunk_features_for_batch(trunk, batch, device=None):
    """The frozen trunk's per-layer features for one numpy batch."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tb = to_torch_batch(batch, device)
    return trunk.extract_features(tb)
