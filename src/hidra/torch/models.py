"""PyTorch ports of the HiDRA trunk and per-lab classifier head.

`UnsupervisedModel` mirrors `hidra.solution.UnsupervisedModel` and `MultiTaskPerLabModel`
mirrors `hidra.train_perlab_heads.MultiTaskPerLabModel`. Each forward pass is transcribed
operation-for-operation from the JAX original so the two can be diffed line by line.

Two transcription rules worth knowing when comparing against the reference:

* Every JAX `x += y` becomes `x = x + y`. In JAX that statement is functional and
  broadcasts freely; the trunk relies on this, starting `x` at shape (1, 1, n_bp, d_res)
  and letting the first `+=` broadcast it out to (T, B, n_bp, d_res). An in-place torch
  `+=` would raise instead of broadcasting.
* NaN is load-bearing, not an error state. Missing keypoints arrive as NaN and the model
  routes them deliberately: `dt_mask` swaps a learned `nan-emb` in for absent lag
  differences, and `torch.where(isnan(...), 0, ...)` zeroes edges to unobserved bodyparts.
  Order matters -- the divide happens before the where, so the NaN it produces is what
  gets replaced.
"""
import numpy as np
import torch
import torch.nn as nn

from .layers import (
    LSTM,
    STAGE_EVAL,
    BidirectionalLSTM,
    Constant,
    Embedding,
    HidraModule,
    Linear,
    jax_gather_index,
    silu,
)


class UnsupervisedModel(HidraModule):
    """The frozen self-supervised trunk: `solution.UnsupervisedModel`."""

    def __init__(self, d_res, d_lstm, d_ff, d_edge, n_layers, n_bp, sample_rate,
                 aggregation_radius, dtype=torch.float32, device=None):
        super().__init__()
        self.d_res = d_res
        self.d_lstm = d_lstm
        self.d_ff = d_ff
        self.d_edge = d_edge
        self.n_layers = n_layers
        self.n_bp = n_bp
        self.output_bins = 16
        self.sample_rate = sample_rate
        self.compute_dtype = dtype

        self.max_y_norm_30 = 8
        self.norm_rescale = np.sqrt(30 / self.sample_rate)
        self.max_y_norm = self.norm_rescale * self.max_y_norm_30
        self.aggregation_radius = aggregation_radius

        self.lags = [1, 2, 3, 4]
        max_norms_30 = {1: 19, 2: 34, 3: 46, 4: 57}
        self.lag_max_norms = {k: self.norm_rescale * v for k, v in max_norms_30.items()}

        def lin(i, o, bd=()):
            return Linear(i, o, batch_dims=bd, dtype=dtype, device=device)

        L = nn.ModuleDict()
        L["x-emb"] = Constant((n_bp, d_res), dtype=dtype, device=device)
        for lag in self.lags:
            L[f"nan-emb-{lag}"] = Constant((d_res,), dtype=dtype, device=device)
            L[f"dt-proj-{lag}"] = lin(2, d_res, (n_bp,))
        L["aug-proj"] = lin(6, d_res)
        for l in range(n_layers):
            pair = (n_bp, n_bp)
            L[f"self-in-{l}"] = lin(2 * d_res, d_edge, pair)
            L[f"self-dx-{l}"] = lin(2, d_edge, pair)
            L[f"self-out-{l}"] = lin(d_edge, d_res, pair)
            L[f"cross-in-{l}"] = lin(2 * d_res, d_edge, pair)
            L[f"cross-dx-{l}"] = lin(2, d_edge, pair)
            L[f"cross-out-{l}"] = lin(d_edge, d_res, pair)
            L[f"merge-{l}"] = lin(2 * d_res, d_res, (n_bp,))
            L[f"ff-in-{l}"] = lin(d_res, d_ff, (n_bp,))
            L[f"ff-out-{l}"] = lin(d_ff, d_res, (n_bp,))
            L[f"lstm-{l}"] = LSTM(d_res, d_lstm, batch_dims=(n_bp,), dtype=dtype, device=device)
            L[f"lstm-res-{l}"] = lin(d_lstm, d_res, (n_bp,))
        L["out-proj"] = lin(d_res, self.output_bins ** 2, (n_bp,))
        self.layers = L

    # ---------------------------------------------------------------- edge geometry
    def _edge_terms(self, x_t):
        """The pairwise-displacement inputs, computed once from raw coordinates.

        `dx / ||dx|| * sqrt(||dx||)` is a distance-attenuated unit vector; the division
        makes self-pairs (zero displacement) NaN, which the following `where` zeroes.
        `adj` gates edges to bodyparts further apart than the aggregation radius.
        """
        self_dx = x_t[:, :, :, None] - x_t[:, :, None]
        self_norms = torch.linalg.vector_norm(self_dx, dim=-1, keepdim=True)
        self_adj = self_norms < self.aggregation_radius
        self_dx = (self_dx / self_norms) * torch.sqrt(self_norms)
        self_dx = torch.where(torch.isnan(self_dx), torch.zeros_like(self_dx), self_dx)

        # The other mouse: swap the two halves of the batch axis.
        x_cross = torch.cat(torch.chunk(x_t, 2, dim=1)[::-1], dim=1)
        cross_dx = x_cross[:, :, :, None] - x_t[:, :, None]
        cross_norms = torch.linalg.vector_norm(cross_dx, dim=-1, keepdim=True)
        cross_adj = cross_norms < self.aggregation_radius
        cross_dx = (cross_dx / cross_norms) * torch.sqrt(cross_norms)
        cross_dx = torch.where(torch.isnan(cross_dx), torch.zeros_like(cross_dx), cross_dx)

        def cast(t):
            return t.to(self.compute_dtype)

        return cast(self_dx), cast(self_adj), cast(cross_dx), cast(cross_adj)

    # ---------------------------------------------------------------- node embedding
    def _node_embedding(self, x_t, augmentation_params):
        """Per-bodypart token from the learned embedding plus four lag differences."""
        L = self.layers
        x = L["x-emb"]()[None, None]
        for lag in self.lags:
            # Shift by `lag` frames, padding the head with NaN (= unobserved).
            pad = torch.full_like(x_t[:lag], float("nan"))
            x_lag = torch.cat([pad, x_t[:-lag]], dim=0)
            dt = x_t - x_lag
            dt_mask = ~torch.any(torch.isnan(dt), dim=-1, keepdim=True)

            max_norm = self.lag_max_norms[lag]
            norms = torch.linalg.vector_norm(dt, dim=-1, keepdim=True)
            dt = torch.where(norms > max_norm, max_norm * (dt / norms), dt)
            dt = torch.where(dt_mask, dt, torch.zeros_like(dt)).to(self.compute_dtype)

            dt_proj = L[f"dt-proj-{lag}"](dt)
            nan_emb = L[f"nan-emb-{lag}"]()
            dt_proj = torch.where(dt_mask, dt_proj, nan_emb[None, None, None])
            x = x + dt_proj
        x = x / ((len(self.lags) + 1) ** 0.5)

        # Tell the model how the clip was augmented (agent and target share the params).
        ap = torch.cat([augmentation_params, augmentation_params], dim=0).to(self.compute_dtype)
        b = L["aug-proj"](ap)[None, :, None]
        x = (x + b) * (0.5 ** 0.5)
        return silu(x)

    # ---------------------------------------------------------------- forward
    def forward(self, x, augmentation_params, extract_features=False):
        """x: (tsteps, batch, n_bp, 2) raw coordinates in mm, NaN where unobserved."""
        L = self.layers
        x_t = x
        self_dx, self_adj, cross_dx, cross_adj = self._edge_terms(x_t)
        x = self._node_embedding(x_t, augmentation_params)

        feats = []
        for l in range(self.n_layers):
            # --- message passing within the mouse
            src = x[:, :, :, None].expand(*x.shape[:3], self.n_bp, x.shape[-1])
            dst = x[:, :, None].expand(*x.shape[:2], self.n_bp, *x.shape[2:])
            self_feats = torch.cat([src, dst], dim=-1)
            self_feats = L[f"self-in-{l}"](self_feats)
            self_feats = self_feats + L[f"self-dx-{l}"](self_dx)
            self_feats = silu(self_feats)
            self_feats = L[f"self-out-{l}"](self_feats)
            self_feats = (self_feats * self_adj).sum(dim=-2) / 2.0

            # --- message passing to the other mouse
            x_cross = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
            cross_src = x[:, :, :, None].expand(*x.shape[:3], self.n_bp, x.shape[-1])
            cross_dst = x_cross[:, :, None].expand(*x_cross.shape[:2], self.n_bp, *x_cross.shape[2:])
            cross_feats = torch.cat([cross_src, cross_dst], dim=-1)
            cross_feats = L[f"cross-in-{l}"](cross_feats)
            cross_feats = cross_feats + L[f"cross-dx-{l}"](cross_dx)
            cross_feats = silu(cross_feats)
            cross_feats = L[f"cross-out-{l}"](cross_feats)
            cross_feats = (cross_feats * cross_adj).sum(dim=-2) / 2.0

            y = torch.cat([self_feats, cross_feats], dim=-1)
            x = x + L[f"merge-{l}"](y)

            y = L[f"ff-in-{l}"](x)
            y = silu(y)
            x = x + L[f"ff-out-{l}"](y)

            y = L[f"lstm-{l}"](x)
            x = x + L[f"lstm-res-{l}"](y)

            feats.append(x)

        if extract_features:
            return feats
        return L["out-proj"](x)

    def extract_features(self, batch):
        """The trunk features the classifier head consumes: `n_layers` tensors of
        (tsteps, 2*batch, n_bp, d_res). Agent and target are stacked along the batch axis
        so a single pass computes both, and the cross-mouse edges can find the partner by
        splitting that axis in half."""
        x = torch.cat([batch["agent"], batch["target"]], dim=0)
        x = x.permute(1, 0, 2, 3)
        return self.forward(x, batch["augmentation_params"], extract_features=True)


class MultiTaskPerLabModel(HidraModule):
    """The per-lab classifier head: `train_perlab_heads.MultiTaskPerLabModel`.

    Only the inference path is ported. The JAX class also carries CACHED_X0_DIR /
    CACHED_PREMERGE_DIR / FILM / SKIP_PATH / DISENTANGLE / CORAL branches, all gated on
    environment variables that `hidra.cli` never sets, so they are dead at inference. They
    are intentionally absent here rather than ported unused.
    """

    def __init__(self, d_res, d_ff, d_lstm, n_layers, n_bp, padding, n_actions, n_labs,
                 n_heads, feat_dim, node_dim, unsupervised_model=None,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.d_res = d_res
        self.d_ff = d_ff
        self.d_lstm = d_lstm
        self.n_layers = n_layers
        self.n_bp = n_bp
        self.padding = padding
        self.n_actions = n_actions
        self.n_heads = n_heads
        self.compute_dtype = dtype
        # Held outside the module list so the frozen trunk's parameters are not part of
        # this module's state_dict (they live in their own checkpoint).
        object.__setattr__(self, "unsupervised_model", unsupervised_model)

        def lin(i, o, bd=()):
            return Linear(i, o, batch_dims=bd, dtype=dtype, device=device)

        L = nn.ModuleDict()
        L["ff-merge-in"] = lin(2 * feat_dim, node_dim, (n_bp,))
        L["ff-merge-out"] = lin(node_dim, node_dim, (n_bp,))
        L["feat-flat-proj"] = lin(node_dim, d_res)
        L["lab-embedding"] = Embedding(d_res, n_labs, dtype=dtype, device=device)
        for l in range(n_layers):
            L[f"lstm-{l}"] = BidirectionalLSTM(d_res, d_lstm, dtype=dtype, device=device)
            L[f"out-proj-{l}"] = lin(2 * d_lstm, d_res)
            L[f"ff-{l}-in"] = lin(d_res, d_ff)
            L[f"ff-{l}-out"] = lin(d_ff, d_res)
        L["out-proj"] = lin(d_res, n_actions)
        L["out-proj-perlab"] = lin(d_res, n_heads)
        self.layers = L

        # P[lab, action, head] = 1 iff `head` is that (lab, action) pair's column. Routes
        # the 82 per-lab head outputs back into action space for a given lab.
        self.register_buffer("P", torch.zeros(n_labs, n_actions, n_heads,
                                              dtype=torch.float32, device=device),
                             persistent=False)

    def set_P(self, P):
        self.P.copy_(torch.as_tensor(np.asarray(P), dtype=torch.float32))
        return self

    def trunk_feats(self, batch):
        """Everything up to the output heads -> (batch, seq_len, d_res)."""
        L = self.layers
        xs = self.unsupervised_model.extract_features(batch)
        x = torch.cat(xs, dim=-1)
        x_agent, x_target = torch.chunk(x, 2, dim=1)
        # Pair the two mice on the feature axis: each node now sees itself and its partner.
        raw = torch.cat([x_agent, x_target], dim=-1)

        x = L["ff-merge-in"](raw)
        x = silu(x)
        x = L["ff-merge-out"](x)
        x = x.sum(dim=2)                      # pool over bodyparts
        x = L["feat-flat-proj"](x)

        x = x + 0.1 * L["lab-embedding"](batch["lab_id"])
        for l in range(self.n_layers):
            y = L[f"lstm-{l}"](x)
            x = x + L[f"out-proj-{l}"](y)
            y = L[f"ff-{l}-in"](x)
            y = silu(y)
            x = x + L[f"ff-{l}-out"](y)

        x = x.permute(1, 0, 2)
        # Drop the context padding the trunk needed but the labels do not cover.
        x = x[:, self.padding: x.shape[1] - self.padding]
        return x.float()

    def logits_perlab(self, batch):
        """Raw per-(lab, action) head logits -> (batch, seq_len, n_heads)."""
        return self.layers["out-proj-perlab"](self.trunk_feats(batch))

    def predict(self, batch):
        """Per-frame probabilities in action space -> (batch, seq_len, n_actions).

        Columns for actions this lab has no head for come out exactly zero, which is what
        makes `run_allbehaviors_perlab` able to filter by head afterwards.
        """
        probs = torch.sigmoid(self.logits_perlab(batch))
        # Same clamping gather as the lab embedding: padding rows carry garbage lab ids.
        Pb = self.P[jax_gather_index(batch["lab_id"], self.P.shape[0])]
        return torch.einsum("btj,baj->bta", probs, Pb)


# ------------------------------------------------------------------ checkpoint loading

def _jax_to_torch_path(torch_name):
    """torch parameter path -> the JAX variable path holding the same tensor.

    The two trees differ only in addressing. `nn.ModuleDict` keeps the JAX layer keys
    verbatim (including the '-' characters), so the mapping is: drop the `layers.` prefix,
    undo the '-'->'_' rename that `BidirectionalLSTM`'s child attributes needed, and swap
    '.' for '/'.
    """
    name = torch_name
    if name.startswith("layers."):
        name = name[len("layers."):]
    name = name.replace("lstm_fw", "lstm-fw").replace("lstm_bw", "lstm-bw")
    return name.replace(".", "/")


def load_state_into(module, flat_state, strict=True):
    """Copy a flat {"jax/path": array} checkpoint into a ported module.

    Returns a report rather than staying silent, because the failure mode that matters here
    is a *quietly* unloaded tensor: a layer left at its random initialization would still
    produce plausible-looking probabilities. `strict=True` refuses that outright.

    The trunk checkpoints carry one variable the model has no home for -- "norm", a dead
    leftover that `solution.Linear.get_weights` never reads -- so it is expected in the
    `unexpected` list.
    """
    tensors = dict(module.named_parameters())
    tensors.update(dict(module.named_buffers()))
    # state_dict() omits non-persistent buffers (P), which no checkpoint provides.
    required = set(module.state_dict().keys())
    mapping = {_jax_to_torch_path(name): name for name in tensors}

    loaded, unexpected = [], []
    with torch.no_grad():
        for jax_path, value in flat_state.items():
            target = mapping.get(jax_path)
            if target is None:
                unexpected.append(jax_path)
                continue
            dst = tensors[target]
            src = torch.as_tensor(np.asarray(value), dtype=dst.dtype, device=dst.device)
            if tuple(src.shape) != tuple(dst.shape):
                raise ValueError(f"shape mismatch for {jax_path}: checkpoint "
                                 f"{tuple(src.shape)} vs model {tuple(dst.shape)}")
            dst.copy_(src)
            loaded.append(target)

    missing = sorted(required - set(loaded))
    if strict and missing:
        raise ValueError(
            f"{len(missing)} model tensor(s) absent from the checkpoint, so they would stay "
            f"at their random initialization: "
            f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
    return {"loaded": sorted(loaded), "missing": missing, "unexpected": sorted(unexpected)}


def _read_checkpoint(path):
    """Read either a published JAX .pkl or a converted .safetensors into a flat dict."""
    from .checkpoint import flatten_tree, load_jax_checkpoint, read_state

    path = str(path)
    if path.endswith(".safetensors"):
        return read_state(path)
    return flatten_tree(load_jax_checkpoint(path))


def build_unsupervised(config, dtype=torch.float32, device=None):
    """The trunk, sized for one ensemble config, with random weights."""
    return UnsupervisedModel(
        d_res=192, d_lstm=192, d_ff=384, d_edge=96, n_layers=4,
        n_bp=config["num_bodyparts"], sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"], dtype=dtype, device=device)


def build_perlab(config, unsupervised_model, n_heads, n_actions, n_labs,
                 dtype=torch.float32, device=None):
    """The per-lab classifier head, sized for one ensemble config, with random weights."""
    feat_dim = unsupervised_model.n_layers * unsupervised_model.d_res
    node_dim = unsupervised_model.d_res
    return MultiTaskPerLabModel(
        d_res=256, d_ff=768, d_lstm=256, n_layers=3, n_bp=config["num_bodyparts"],
        padding=32, n_actions=n_actions, n_labs=n_labs, n_heads=n_heads,
        feat_dim=feat_dim, node_dim=node_dim, unsupervised_model=unsupervised_model,
        dtype=dtype, device=device)


def load_unsupervised(config_name, path=None, dtype=torch.float32, device=None, config=None):
    """Build the trunk for `config_name` and load its published weights.

    `path` defaults to `<models_dir>/{config}_unsupervised.pkl`, and also accepts a
    converted `.safetensors` file.
    """
    from .. import paths, schema

    config = schema.get_configs()[config_name] if config is None else config
    if path is None:
        base = paths.models_dir()
        st = base / f"{config_name}_unsupervised.safetensors"
        path = st if st.is_file() else base / f"{config_name}_unsupervised.pkl"
    model = build_unsupervised(config, dtype=dtype, device=device).to(device)
    report = load_state_into(model, _read_checkpoint(path))
    model.load_report = report
    model.set_stage(STAGE_EVAL).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_perlab(config_name, unsupervised_model=None, path=None, dtype=torch.float32,
                device=None, config=None, lab_action=None):
    """Build the per-lab head for `config_name`, load its weights, and wire up P.

    `unsupervised_model` defaults to loading the matching trunk. `path` defaults to
    `<models_dir>/{config}_supervised_perlab_sniffall.pkl` (or the converted
    `.safetensors`), or to `$HIDRA_PERLAB_CKPT` when set, which is how `hidra.cli`
    threads fine-tuned checkpoints through.
    """
    import os

    from .. import paths, schema

    config = schema.get_configs()[config_name] if config is None else config
    if unsupervised_model is None:
        unsupervised_model = load_unsupervised(config_name, dtype=dtype, device=device,
                                               config=config)
    if path is None:
        override = os.environ.get("HIDRA_PERLAB_CKPT")
        if override:
            path = override.format(config=config_name)
        else:
            base = paths.models_dir()
            st = base / f"{config_name}_supervised_perlab_sniffall.safetensors"
            path = st if st.is_file() else base / f"{config_name}_supervised_perlab_sniffall.pkl"

    lab_action = schema.lab_action_table() if lab_action is None else lab_action
    model = build_perlab(config, unsupervised_model, n_heads=len(lab_action),
                         n_actions=len(schema.ACTIONS), n_labs=len(schema.LABS),
                         dtype=dtype, device=device).to(device)
    report = load_state_into(model, _read_checkpoint(path))
    model.load_report = report
    model.lab_action = lab_action
    model.set_P(schema.build_P(lab_action))
    model.set_stage(STAGE_EVAL).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
