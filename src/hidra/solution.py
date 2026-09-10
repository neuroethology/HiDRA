import functools
import os
import pickle
import sys
import time
import types
from collections import defaultdict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import checkpoints, data


# The label vocabularies live in schema.py, which is jax-free so the PyTorch backend can
# import them too. Re-exported here (same objects, not copies) so every existing
# `solution.ACTIONS` / `solution.LABS` reference keeps working and there is one vocabulary
# per process -- important because SNIFFALL mutates ACTIONS at import time.
from .schema import (  # noqa: F401
    ACTIONS,
    BODYPARTS,
    Enum,
    LABS,
    MOUSE_IDS,
    SELF_DIRECTED,
    SNIFF_FAMILY,
    SNIFFALL_LABS,
    TRAIN_ONLY_LABS,
    get_configs,
    parse_mouse_id,
)

# The data pipeline moved to data.py, which is numpy-only so the PyTorch backend can use it
# without the JAX runtime. Re-exported here (the same objects, not copies) so every existing
# `solution.Dataset` / `solution.Predictions` / `solution.batch` reference keeps working --
# and so monkeypatching one of them from a driver still affects both backends.
from .data import (  # noqa: F401
    BaseDataset,
    Dataset,
    Epoch,
    F1,
    Labels,
    Pipeline,
    Predictions,
    RingBuffer,
    SimpleEpoch,
    TrackingData,
    Video,
    average_metrics,
    batch,
    create_labels,
    create_tracking_data,
    create_video,
    flat_seed,
    get_batch_size,
    load_videos,
    shared_array,
    split_videos,
    take,
    to_device,
    to_host,
    tree_leaves,
    tree_map,
    tree_stack,
    unbatch,
)


@dataclass
class Variable:
    value: None
    trainable: bool

    def set(self, x):
        self.value = x
        return self


def is_variable(x):
    return isinstance(x, Variable)


jax.tree_util.register_dataclass(Variable, data_fields=["value"], meta_fields=["trainable"])


class Layer:
    def create_variables(self, key):
        raise NotImplementedError

    def get_weights(self, variables):
        raise NotImplementedError

    def set_context(self, context):
        self.context = context

    def set_variables(self, variables):
        return ParameterizedLayer(self, variables)


class Linear(Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        dtype,
        use_bias=True,
        batch_dims=[],
        init_bias=0.0,
        normalize_input=True,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_dims = batch_dims
        self.dtype = dtype
        self.use_bias = use_bias
        self.init_bias = init_bias
        self.normalize_input = normalize_input

    def create_variables(self, key):
        dim = max(self.input_dim, self.output_dim)
        default_lrmul = np.sqrt(dim) / np.sqrt(64)

        variables = {}
        w_key, b_key = jax.random.split(key)
        w_shape = self.batch_dims + [self.input_dim, self.output_dim]
        w = jax.random.normal(w_key, shape=w_shape, dtype="float32")
        w = w / default_lrmul
        variables["w"] = Variable(w, trainable=True)

        s_shape = self.batch_dims + [self.output_dim]
        s = jnp.ones(shape=s_shape, dtype="float32")
        variables["s"] = Variable(s, trainable=True)

        if self.use_bias:
            b_shape = self.batch_dims + [self.output_dim]
            init_bias = jnp.array(self.init_bias, dtype="float32")
            init_bias = jnp.broadcast_to(init_bias, b_shape)
            variables["b"] = Variable(init_bias, trainable=True)

        if self.normalize_input:
            input_shape = self.batch_dims + [self.input_dim]
            variables["m1"] = jnp.zeros(input_shape, dtype="float32")
            variables["m2"] = jnp.zeros(input_shape, dtype="float32")
            variables["n"] = jnp.zeros([], dtype="float32")
            variables["mean"] = jnp.zeros(input_shape, dtype="float32")
            variables["std"] = jnp.ones(input_shape, dtype="float32")

            variables["m1"] = Variable(variables["m1"], trainable=False)
            variables["m2"] = Variable(variables["m2"], trainable=False)
            variables["n"] = Variable(variables["n"], trainable=False)
            variables["mean"] = Variable(variables["mean"], trainable=False)
            variables["std"] = Variable(variables["std"], trainable=False)

        return variables

    def get_weights(self, variables):
        weights = {}
        norm = jnp.linalg.norm(variables["w"], axis=-2, keepdims=True)
        w = variables["w"] / norm
        w = w * variables["s"][..., None, :]

        weights["w"] = w.astype(self.dtype)
        if self.use_bias:
            weights["b"] = variables["b"].astype(self.dtype)

        if self.normalize_input:
            weights["m1"] = variables["m1"]
            weights["m2"] = variables["m2"]
            weights["n"] = variables["n"]
            weights["mean"] = variables["mean"]
            weights["std"] = variables["std"]

        return weights

    def apply(self, weights, x, mask=None):
        updates = {}
        if self.normalize_input:
            if self.context["stage"] == "init":
                decay = self.context["init_decay"]
                leading_dims = list(range(len(x.shape) - len(self.batch_dims) - 1))

                m1 = jnp.mean(x, leading_dims)
                m2 = jnp.mean(x**2, leading_dims)
                weights["m1"] = weights["m1"] + (m1 - weights["m1"]) * (1 - decay)
                weights["m2"] = weights["m2"] + (m2 - weights["m2"]) * (1 - decay)
                weights["n"] = weights["n"] + 1

                norm = 1.0 / (1 - decay ** weights["n"])
                m1 = weights["m1"] * norm
                m2 = weights["m2"] * norm
                input_std = jnp.sqrt(m2 - m1**2)

                weights["std"] = input_std + 1e-4
                weights["mean"] = m1

                updates["m1"] = weights["m1"]
                updates["m2"] = weights["m2"]
                updates["n"] = weights["n"]
                updates["std"] = weights["std"]
                updates["mean"] = weights["mean"]

            x = (x - weights["mean"]) / weights["std"]
            x = x.astype(self.dtype)

        y = jnp.einsum("...i,...io->...o", x, weights["w"])
        if self.use_bias:
            y += weights["b"]

        return y, updates


class Constant(Layer):
    def __init__(self, shape, dtype, batch_dims=[]):
        self.shape = shape
        self.dtype = dtype
        self.batch_dims = batch_dims

    def create_variables(self, key):
        shape = self.batch_dims + list(self.shape)
        c = jax.random.normal(key, shape=shape, dtype="float32")
        return {"c": Variable(c, trainable=True)}

    def get_weights(self, variables):
        return {"c": variables["c"].astype(self.dtype)}

    def apply(self, weights):
        updates = {}
        return weights["c"], updates


class Embedding(Layer):
    def __init__(self, input_dim, cardinality, dtype):
        self.input_dim = input_dim
        self.cardinality = cardinality
        self.dtype = dtype

    def create_variables(self, key):
        w = jax.random.normal(key, shape=[self.cardinality, self.input_dim], dtype="float32")
        return {"w": Variable(w, trainable=True)}

    def get_weights(self, variables):
        return {"w": variables["w"].astype(self.dtype)}

    def apply(self, weights, indices):
        updates = {}
        return weights["w"][indices], updates


class ParameterizedLayer:
    def __init__(self, layer, variables):
        self.layer = layer
        self.variables = variables
        self.weights = layer.get_weights(variables)

    def get_variables(self):
        return self.variables

    def set_variables(self, variables):
        self.variables = variables

    def apply(self, *args, **kwargs):
        outputs, updates = self.layer.apply(self.weights, *args, **kwargs)
        updates_key_vals = dict(jax.tree.flatten_with_path(updates)[0])
        self.variables = jax.tree.map_with_path(lambda path, x: updates_key_vals.get(path, x), self.variables)
        return outputs


class Module:
    def create_variables(self, key):
        variables = {}
        for layer_name, layer in self.layers.items():
            key, subkey = jax.random.split(key)
            variables[layer_name] = layer.create_variables(subkey)
        return variables

    def get_variables(self):
        state = {}
        for layer_name, layer in self.layers.items():
            state[layer_name] = layer.get_variables()
        return state

    def set_variables(self, variables):
        return ParameterizedModule(self, variables)

    def set_context(self, context):
        self.context = context
        for layer_name, layer in self.layers.items():
            layer.set_context(context)


class ParameterizedModule:
    def __init__(self, module, variables):
        self.module = module
        self.variables = variables

        self.layers = {}
        for layer_name, layer in module.layers.items():
            self.layers[layer_name] = layer.set_variables(variables[layer_name])

    def get_variables(self):
        variables = {}
        for layer_name, layer in self.layers.items():
            variables[layer_name] = layer.get_variables()
        return variables

    def set_variables(self, variables):
        for layer_name, layer in self.layers.items():
            layer.set_variables(variables[layer_name])

    def parameterized_function(self, name):
        module_fn = getattr(self.module, name)

        def function(*args, **kwargs):
            outputs = module_fn(self.layers, *args, **kwargs)
            return outputs

        return function

    def __getattr__(self, name):
        if hasattr(self.module, name):
            if isinstance(getattr(self.module, name), types.MethodType):
                return self.parameterized_function(name)
            else:
                t = type(getattr(self.module, name))
                assert False, f"only function attributes passed through, received {t}"
        else:
            assert False


class LSTM(Module):
    def __init__(self, input_dim, hidden_dim, dtype, forget_bias=0.0, batch_dims=[]):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_dims = batch_dims
        self.dtype = dtype
        self.forget_bias = forget_bias

        forget_bias = self.forget_bias * np.repeat([0.0, 1.0, 0.0, 0.0], hidden_dim)

        self.layers = {}
        self.layers["is_linear"] = Linear(
            input_dim=self.input_dim,
            output_dim=4 * self.hidden_dim,
            batch_dims=self.batch_dims,
            dtype=self.dtype,
            use_bias=True,
            init_bias=forget_bias,
        )
        self.layers["ss_linear"] = Linear(
            input_dim=self.hidden_dim,
            output_dim=4 * self.hidden_dim,
            batch_dims=self.batch_dims,
            dtype=self.dtype,
            use_bias=False,
            normalize_input=False,
        )
        self.layers["h0"] = Constant(shape=[self.hidden_dim], batch_dims=self.batch_dims, dtype=self.dtype)
        self.layers["c0"] = Constant(shape=[self.hidden_dim], batch_dims=self.batch_dims, dtype=self.dtype)

    def apply(self, layers, x):
        tsteps, bs = x.shape[0], x.shape[1]

        h0 = jnp.tanh(layers["h0"].apply())
        c0 = layers["c0"].apply()

        tiling = [bs] + [1] * (len(self.batch_dims) + 1)
        h0 = jnp.tile(h0[None], tiling)
        c0 = jnp.tile(c0[None], tiling)

        xs = layers["is_linear"].apply(x)
        hs_shape = [tsteps, bs] + self.batch_dims + [self.hidden_dim]
        hs = jnp.empty(hs_shape, dtype=self.dtype)

        def body(t, state):
            x_t = xs[t]
            x_hh = layers["ss_linear"].apply(state["h"])
            gates = x_t + x_hh

            i, f, c, o = jnp.split(gates, 4, axis=-1)
            c = jax.nn.sigmoid(f) * state["c"] + jax.nn.sigmoid(i) * jnp.tanh(c)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            hs = state["hs"].at[t].set(h)
            return {"h": h, "c": c, "hs": hs}

        state = {"h": h0, "c": c0, "hs": hs}
        outputs = jax.lax.fori_loop(0, tsteps, body, state)
        return outputs["hs"]


class BidirectionalLSTM(Module):
    def __init__(self, input_dim, hidden_dim, dtype, forget_bias=0.0, batch_dims=[]):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_dims = batch_dims
        self.dtype = dtype
        self.forget_bias = forget_bias

        self.layers = {}
        self.layers["lstm-fw"] = LSTM(input_dim, hidden_dim, dtype, forget_bias, batch_dims)
        self.layers["lstm-bw"] = LSTM(input_dim, hidden_dim, dtype, forget_bias, batch_dims)

    def apply(self, layers, x):
        h_fw = layers["lstm-fw"].apply(x)
        x_bw = jnp.flip(x, axis=0)
        h_bw = layers["lstm-bw"].apply(x_bw)
        h_bw = jnp.flip(h_bw, axis=0)
        h = jnp.concat([h_fw, h_bw], axis=-1)
        return h


class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def create_variables(self, params):
        def init_moments(param):
            return {
                "m1": jnp.zeros(param.shape, dtype=param.dtype),
                "m2": jnp.zeros(param.shape, dtype=param.dtype),
            }

        moments = jax.tree.map(init_moments, params, is_leaf=is_variable)
        return {"t": jnp.ones([]), "moments": moments}

    def update(self, params, opt_state, grads):
        b1_correction = 1.0 / (1 - self.beta1 ** opt_state["t"])
        b2_correction = 1.0 / (1 - self.beta2 ** opt_state["t"])

        def update_moments(grad, moments):
            moments["m1"] += (grad - moments["m1"]) * (1 - self.beta1)
            moments["m2"] += (grad**2 - moments["m2"]) * (1 - self.beta2)
            return moments

        def update_param(param, moments):
            m1 = moments["m1"] * b1_correction
            m2 = moments["m2"] * b2_correction
            param -= self.learning_rate * (m1 / (jnp.sqrt(m2) + self.eps))
            return param

        moments = jax.tree.map(update_moments, grads, opt_state["moments"])
        params = jax.tree.map(update_param, params, moments)
        opt_state = {"t": opt_state["t"] + 1, "moments": moments}
        return params, opt_state


class EMA:
    def __init__(self, decay):
        self.decay = decay

    def create_variables(self, params):
        return {
            "t": jnp.ones([], dtype="float32"),
            "sums": jax.tree.map(lambda x: x * (1.0 - self.decay), params),
        }

    def update(self, ema_state, variables):
        def update_sums(variable, sum):
            return sum + (variable.value - sum) * (1 - self.decay)

        return {
            "t": ema_state["t"] + 1,
            "sums": jax.tree.map(update_sums, variables, ema_state["sums"], is_leaf=is_variable),
        }

    def values(self, ema_state):
        norm = 1 - self.decay ** ema_state["t"]
        return jax.tree.map(lambda x: x / norm, ema_state["sums"])


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir,
        max_checkpoints,
        metric_name,
        lower_is_better=True,
        patience=None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.lower_is_better = lower_is_better
        self.patience = patience
        self.checkpoints = [(None, float("-inf"))] * self.max_checkpoints

    def write_file(self, params, path):
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        cpu = jax.devices("cpu")[0]
        params = jax.tree.map(lambda x: jax.device_put(x, cpu), params)
        with open(path, "wb") as f:
            pickle.dump(params, f)

    def path(self, step):
        return f"{self.checkpoint_dir}/{step}.pkl"

    def update(self, params, step, metrics):
        score = metrics[self.metric_name]
        if np.isnan(score):
            # todo: nan breaks things
            assert False
        if self.lower_is_better:
            score = -score

        if score > self.checkpoints[0][1]:
            del_path = self.path(self.checkpoints[0][0])
            if os.path.isfile(del_path):
                os.remove(del_path)

            self.checkpoints[0] = (step, score)
            self.checkpoints = sorted(self.checkpoints, key=lambda x: x[1])
            self.write_file(params, self.path(step))

            msg = f"writing {self.path(step)}"
            if step == self.checkpoints[-1][0]:
                if os.path.islink(self.path("best")):
                    os.unlink(self.path("best"))
                os.symlink(os.path.abspath(self.path(step)), self.path("best"))
                msg += " *"
            print(msg)

        terminate_training = False
        if self.patience is not None:
            if step - self.checkpoints[-1][0] > self.patience:
                terminate_training = True

        return terminate_training


class Trainer:
    def __init__(
        self,
        experiment_name,
        model,
        optimizer,
        train_dataset,
        train_batch_size,
        eval_dataset=None,
        eval_batch_size=None,
        custom_eval_loop=None,
        eval_interval=2000,
        skip_eval=False,
        seed=None,
        train_log_interval=500,
        max_training_steps=10**10,
        ema_decay=0.9993,
        early_stopping_config={},
    ):
        self.experiment_name = experiment_name
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size or 4 * train_batch_size
        self.custom_eval_loop = custom_eval_loop
        self.eval_interval = eval_interval
        self.skip_eval = skip_eval
        self.seed = seed
        self.train_log_interval = train_log_interval
        self.max_training_steps = max_training_steps
        self.ema_decay = ema_decay

        self.experiment_dir = f"experiments/{self.experiment_name}"
        self.checkpoint_dir = f"{self.experiment_dir}/checkpoints"
        self.checkpoint_manager = CheckpointManager(
            self.checkpoint_dir,
            max_checkpoints=3,
            metric_name=early_stopping_config.get("metric_name", "obj"),
            lower_is_better=early_stopping_config.get("lower_is_better", True),
            patience=early_stopping_config.get("patience", None),
        )

        self.ema = EMA(self.ema_decay)

        rng = np.random.default_rng(self.seed)
        seeds = rng.integers(low=0, high=2**63, size=[3], dtype="int64")
        self.model_seed, self.train_seed, self.eval_seed = seeds

        n_devices = jax.local_device_count()
        mesh = jax.make_mesh((n_devices,), ("batch",))
        jax.set_mesh(mesh)
        self.sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))

    def get_ddi_loop(self):
        @functools.partial(jax.jit, donate_argnums=[0])
        def ddi_step(state, batch):
            context = {"stage": "init", "init_decay": 0.99}
            self.model.set_context(context)
            key, subkey = jax.random.split(state["key"])

            variable_values = jax.tree.map(
                lambda x: x.value,
                state["variables"],
                is_leaf=is_variable,
            )
            pmodel = self.model.set_variables(variable_values)
            pmodel.compute_loss(batch, subkey)
            updated_variable_values = pmodel.get_variables()
            variables = jax.tree.map(lambda x, y: y.set(x), updated_variable_values, state["variables"])
            next_state = {"variables": variables, "key": key}
            return next_state

        def ddi_loop(ddi_state, n):
            def batch_update(batch):
                nonlocal ddi_state
                ddi_state = ddi_step(ddi_state, batch)
                return ddi_state

            elements = Pipeline(self.train_dataset.element_iterator())
            batches = elements.batch(self.train_batch_size).to_device(self.sharding)
            ddi_state = batches.map(batch_update).take(n).last()
            return ddi_state

        return ddi_loop

    def initial_state(self):
        key = jax.random.key(self.model_seed)

        init_key, ddi_key = jax.random.split(key, 2)
        variables = self.model.create_variables(init_key)

        ddi_loop = self.get_ddi_loop()
        ddi_state = {"variables": variables, "key": ddi_key}
        ddi_state = ddi_loop(ddi_state, 256)
        variables = ddi_state["variables"]

        variables_flat, variables_treedef = jax.tree.flatten(variables, is_leaf=is_variable)

        trainable_indices = []
        static_indices = []
        trainable_values = []
        static_values = []
        for i, variable in enumerate(variables_flat):
            if variable.trainable:
                trainable_values.append(variable.value)
                trainable_indices.append(i)
            else:
                static_values.append(variable.value)
                static_indices.append(i)

        optimizer_state = self.optimizer.create_variables(trainable_values)

        variable_values = jax.tree.map(lambda x: x.value, variables, is_leaf=is_variable)
        ema_state = self.ema.create_variables(variable_values)

        train_state = {
            "variables": variables,
            "optimizer": optimizer_state,
            "ema": ema_state,
            "key": jax.random.key(self.train_seed),
        }
        return train_state

    def get_eval_loop(self):
        @jax.jit
        def eval_step(variables, batch, key):
            context = {"stage": "eval"}
            self.model.set_context(context)
            key, subkey = jax.random.split(key)
            pmodel = self.model.set_variables(variables)
            loss, metrics = pmodel.compute_loss(batch, subkey)
            return metrics, key

        def eval_loop(train_state):
            def get_batch_metrics(batch):
                nonlocal key
                metrics, key = eval_step(train_state, batch, key)
                return metrics

            key = jax.random.key(self.eval_seed)
            elements = Pipeline(self.eval_dataset.element_iterator())
            batches = elements.batch(self.eval_batch_size).to_device(self.sharding)
            metrics = batches.map(get_batch_metrics).to_host().average_metrics()
            return metrics

        return eval_loop

    def train_step(self, state, batch):
        context = {"stage": "train"}
        self.model.set_context(context)
        key, subkey = jax.random.split(state["key"])

        variables = state["variables"]
        variables_flat, variables_treedef = jax.tree.flatten(variables, is_leaf=is_variable)

        trainable_indices = []
        static_indices = []
        trainable_values = []
        static_values = []
        for i, variable in enumerate(variables_flat):
            if variable.trainable:
                trainable_values.append(variable.value)
                trainable_indices.append(i)
            else:
                static_values.append(variable.value)
                static_indices.append(i)

        def compute_loss(trainable_values, static_values, batch, key):
            merged_vars_flat = [None] * (len(variables_flat))
            for i, var in zip(trainable_indices, trainable_values):
                merged_vars_flat[i] = var
            for i, var in zip(static_indices, static_values):
                merged_vars_flat[i] = var
            merged_vars = jax.tree.unflatten(variables_treedef, merged_vars_flat)
            pmodel = self.model.set_variables(merged_vars)
            loss, metrics = pmodel.compute_loss(batch, key)
            updated_variables = pmodel.get_variables()
            return loss, (metrics, updated_variables)

        key, subkey = jax.random.split(key)
        grad_fn = jax.value_and_grad(compute_loss, argnums=0, has_aux=True)
        (loss, aux), grads = grad_fn(trainable_values, static_values, batch, key)
        metrics, updated_variables = aux
        updated_variables_flat, _ = jax.tree.flatten(updated_variables)

        grad_updated_values, next_opt_state = self.optimizer.update(trainable_values, state["optimizer"], grads)
        for i, value in zip(trainable_indices, grad_updated_values):
            updated_variables_flat[i] = value

        updated_variables_values = jax.tree.unflatten(variables_treedef, updated_variables_flat)
        updated_variables = jax.tree.map(lambda x, y: y.set(x), updated_variables_values, variables)
        next_ema_state = self.ema.update(state["ema"], variables)

        next_state = {"variables": updated_variables, "optimizer": next_opt_state, "ema": next_ema_state, "key": key}
        return next_state, metrics

    def get_train_loop(self):
        train_step_fn = jax.jit(self.train_step, donate_argnames=["state"])
        elements = Pipeline(self.train_dataset.element_iterator())
        batches = elements.batch(self.train_batch_size).to_device(self.sharding)

        def train_loop(train_state, n):
            def get_batch_metrics(batch):
                nonlocal train_state
                train_state, metrics = train_step_fn(train_state, batch)
                return metrics

            metrics = batches.take(n).map(get_batch_metrics).to_host()
            metrics = metrics.average_metrics()
            return train_state, metrics

        return train_loop

    def log_metrics(self, metrics, dt=None, tag=None):
        strings = []
        if tag is not None:
            strings.append(tag)
        for k, v in metrics.items():
            strings.append(f"{k}: {v:>8.3f}")
        if dt is not None:
            strings.append(f"({dt: >3.1f}s)")
        print("     ".join(strings))

    def train(self, path=None):
        state = self.initial_state()

        eval_loop = None
        if not self.skip_eval:
            if self.custom_eval_loop is not None:
                eval_loop = self.custom_eval_loop
            elif self.eval_dataset is not None:
                eval_loop = self.get_eval_loop()

        train_loop = self.get_train_loop()
        get_ema_values = jax.jit(self.ema.values)

        t_prev = time.time()
        step = 0
        while step < self.max_training_steps:
            if eval_loop is not None and step % self.eval_interval == 0:
                if eval_loop is not None:
                    eval_t0 = time.time()
                    ema_values = get_ema_values(state["ema"])
                    metrics = eval_loop(ema_values)
                    eval_dt = time.time() - eval_t0
                    self.log_metrics(metrics, dt=eval_dt, tag="[EVAL]")
                    terminate_training = self.checkpoint_manager.update(ema_values, step, metrics)
                    t_prev = time.time()
                    if terminate_training:
                        print("training finished")
                        return

            state, train_metrics = train_loop(state, self.train_log_interval)
            step += self.train_log_interval

            dt = time.time() - t_prev
            self.log_metrics(train_metrics, dt=dt, tag=f"[step {step:>8}]")
            t_prev = time.time()
        print("training finished")


class UnsupervisedModel(Module):
    def __init__(
        self,
        d_res,
        d_lstm,
        d_ff,
        d_edge,
        n_layers,
        n_bp,
        sample_rate,
        aggregation_radius,
        dtype,
    ):
        self.d_res = d_res
        self.d_lstm = d_lstm
        self.d_ff = d_ff
        self.d_edge = d_edge

        self.n_layers = n_layers
        self.dtype = dtype
        self.n_bp = n_bp
        self.output_bins = 16
        self.sample_rate = sample_rate

        self.max_y_norm_30 = 8
        self.norm_rescale = np.sqrt(30 / self.sample_rate)
        self.max_y_norm = self.norm_rescale * self.max_y_norm_30
        self.aggregation_radius = aggregation_radius

        self.lags = [1, 2, 3, 4]
        max_norms_30 = {1: 19, 2: 34, 3: 46, 4: 57}
        self.lag_max_norms = {k: self.norm_rescale * v for k, v in max_norms_30.items()}

        self.layers = {}
        self.layers["x-emb"] = Constant([n_bp, d_res], dtype)
        for lag in self.lags:
            self.layers[f"nan-emb-{lag}"] = Constant([d_res], dtype)
            self.layers[f"dt-proj-{lag}"] = Linear(2, d_res, dtype, batch_dims=[n_bp])

        self.layers["aug-proj"] = Linear(6, d_res, dtype)
        for l in range(self.n_layers):
            kwargs = dict(dtype=dtype, batch_dims=[n_bp, n_bp])
            self.layers[f"self-in-{l}"] = Linear(2 * d_res, d_edge, **kwargs)
            self.layers[f"self-dx-{l}"] = Linear(2, d_edge, **kwargs)
            self.layers[f"self-out-{l}"] = Linear(d_edge, d_res, **kwargs)

            self.layers[f"cross-in-{l}"] = Linear(2 * d_res, d_edge, **kwargs)
            self.layers[f"cross-dx-{l}"] = Linear(2, d_edge, **kwargs)
            self.layers[f"cross-out-{l}"] = Linear(d_edge, d_res, **kwargs)

            self.layers[f"merge-{l}"] = Linear(2 * d_res, d_res, dtype, batch_dims=[n_bp])

            self.layers[f"ff-in-{l}"] = Linear(d_res, d_ff, dtype, batch_dims=[n_bp])
            self.layers[f"ff-out-{l}"] = Linear(d_ff, d_res, dtype, batch_dims=[n_bp])

            self.layers[f"lstm-{l}"] = LSTM(d_res, d_lstm, dtype, batch_dims=[n_bp])
            self.layers[f"lstm-res-{l}"] = Linear(d_lstm, d_res, dtype, batch_dims=[n_bp])

        self.layers["out-proj"] = Linear(d_res, self.output_bins**2, dtype, batch_dims=[n_bp])

    def get_labels(self, x, batch):
        x_tp1 = jnp.pad(x[1:], [(0, 1), (0, 0), (0, 0), (0, 0)], constant_values=np.nan)
        y = x_tp1 - x

        y_mask = jnp.all(~jnp.isnan(y), axis=-1)
        y = jnp.where(y_mask[..., None], y, 1.0)

        y_r = jnp.linalg.norm(y, axis=-1)
        y_r = jnp.minimum(y_r, self.max_y_norm)
        y_r = (y_r / self.max_y_norm) * (self.output_bins - 1)
        y_r = jnp.round(y_r).astype("int32")

        y_theta = jnp.arctan2(y[..., 1], y[..., 0])
        y_theta = (y_theta + np.pi) / (2 * np.pi)
        y_theta = y_theta * (self.output_bins - 1)
        y_theta = jnp.round(y_theta).astype("int32")

        y = y_r * self.output_bins + y_theta

        batch_mask = jnp.concat([batch["batch_mask"]] * 2, axis=0)
        y_mask = y_mask & ((batch_mask == 1)[None, :, None])
        return y, y_mask

    def forward(self, layers, x, batch, key, extract_features=False):
        tsteps, bs, num_bodyparts, channels = x.shape
        x_t = x

        x_emb = layers["x-emb"].apply()
        x = x_emb[None, None]
        for lag in self.lags:
            x_lag = jnp.pad(x_t[:-lag], [(lag, 0), (0, 0), (0, 0), (0, 0)], constant_values=np.nan)
            dt = x_t - x_lag
            dt_mask = ~jnp.any(jnp.isnan(dt), axis=-1, keepdims=True)
            max_norm = self.lag_max_norms[lag]
            norms = jnp.linalg.norm(dt, axis=-1, keepdims=True)
            dt = jnp.where(norms > max_norm, max_norm * (dt / norms), dt)
            dt = jnp.where(dt_mask, dt, 0.0)
            dt = dt.astype(self.dtype)

            dt_proj = layers[f"dt-proj-{lag}"].apply(dt)
            nan_emb = layers[f"nan-emb-{lag}"].apply()
            dt_proj = jnp.where(dt_mask, dt_proj, nan_emb[None, None, None])

            x += dt_proj

        x = x / ((len(self.lags) + 1) ** 0.5)

        ap = jnp.concat([batch["augmentation_params"], batch["augmentation_params"]], axis=0)
        ap = ap.astype(self.dtype)
        b = layers["aug-proj"].apply(ap)[None, :, None]
        x = (x + b) * (0.5**0.5)

        x = jax.nn.silu(x)

        self_dx = x_t[:, :, :, None] - x_t[:, :, None]
        self_norms = jnp.linalg.norm(self_dx, axis=-1, keepdims=True)
        self_adj = self_norms < self.aggregation_radius
        self_dx = (self_dx / self_norms) * jnp.sqrt(self_norms)
        self_dx = jnp.where(jnp.isnan(self_dx), 0.0, self_dx)
        self_dx, self_adj = self_dx.astype(self.dtype), self_adj.astype(self.dtype)

        x_cross = jnp.concat(jnp.split(x_t, 2, axis=1)[::-1], axis=1)
        cross_dx = x_cross[:, :, :, None] - x_t[:, :, None]
        cross_norms = jnp.linalg.norm(cross_dx, axis=-1, keepdims=True)
        cross_adj = cross_norms < self.aggregation_radius
        cross_dx = (cross_dx / cross_norms) * jnp.sqrt(cross_norms)
        cross_dx = jnp.where(jnp.isnan(cross_dx), 0.0, cross_dx)
        cross_dx, cross_adj = cross_dx.astype(self.dtype), cross_adj.astype(self.dtype)

        feats = []
        for l in range(self.n_layers):
            # self
            src_feats = jnp.tile(x[:, :, :, None], (1, 1, 1, self.n_bp, 1))
            dst_feats = jnp.tile(x[:, :, None], (1, 1, self.n_bp, 1, 1))
            self_feats = jnp.concat([src_feats, dst_feats], axis=-1)
            self_feats = layers[f"self-in-{l}"].apply(self_feats)
            self_feats = self_feats + layers[f"self-dx-{l}"].apply(self_dx)
            self_feats = jax.nn.silu(self_feats)
            self_feats = layers[f"self-out-{l}"].apply(self_feats)
            self_feats = (self_feats * self_adj).sum(axis=-2) / 2.0

            # cross
            x_cross = jnp.concat(jnp.split(x, 2, axis=1)[::-1], axis=1)
            cross_src = jnp.tile(x[:, :, :, None], (1, 1, 1, self.n_bp, 1))
            cross_dst = jnp.tile(x_cross[:, :, None], (1, 1, self.n_bp, 1, 1))
            cross_feats = jnp.concat([cross_src, cross_dst], axis=-1)
            cross_feats = layers[f"cross-in-{l}"].apply(cross_feats)
            cross_feats = cross_feats + layers[f"cross-dx-{l}"].apply(cross_dx)
            cross_feats = jax.nn.silu(cross_feats)
            cross_feats = layers[f"cross-out-{l}"].apply(cross_feats)
            cross_feats = (cross_feats * cross_adj).sum(axis=-2) / 2.0

            # merge
            y = jnp.concat([self_feats, cross_feats], axis=-1)
            y = layers[f"merge-{l}"].apply(y)
            x += y

            # ff
            y = layers[f"ff-in-{l}"].apply(x)
            y = jax.nn.silu(y)
            y = layers[f"ff-out-{l}"].apply(y)
            x += y

            # lstm
            y = layers[f"lstm-{l}"].apply(x)
            y = layers[f"lstm-res-{l}"].apply(y)
            x += y

            feats.append(x)

        if extract_features:
            return feats

        logits = layers["out-proj"].apply(x)
        return logits

    def compute_loss(self, layers, batch, key):
        x = jnp.concat([batch["agent"], batch["target"]], axis=0)
        x = x.transpose(1, 0, 2, 3)

        logits = self.forward(layers, x, batch, key).astype("float32")
        y, y_mask = self.get_labels(x, batch)

        labels = jax.nn.one_hot(y, self.output_bins**2)
        logprobs = (labels * jax.nn.log_softmax(logits)).sum(axis=-1)
        y_mask = y_mask.astype(logprobs.dtype)
        nll_weight = y_mask.sum()
        nll = -(logprobs * y_mask).sum() / nll_weight

        metrics = {"nll": (nll, nll_weight)}
        return (nll, metrics)

    def extract_features(self, layers, batch, key):
        x = jnp.concat([batch["agent"], batch["target"]], axis=0)
        x = x.transpose(1, 0, 2, 3)
        feats = self.forward(layers, x, batch, key, extract_features=True)
        return feats


def pretrain(config):
    videos = load_videos(mode="train", use_cached=True)
    train_videos, val_videos = split_videos(videos, validation_frac=0.15, random_seed=config["split_seed"])

    train_dataset = Dataset(
        videos=train_videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=100000,
        unsupervised=True,
        max_scale=config["max_scale"],
        max_time_dilation=config["max_time_dilation"],
        rotate=True,
        flip=True,
        noise_scale=config["noise_scale"],
        num_workers=4,
        seed=[0] + config["pretrain_seed"],
    )
    val_dataset = Dataset(
        videos=val_videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=1,
        unsupervised=True,
        max_scale=1,
        max_time_dilation=1,
        rotate=False,
        flip=False,
        noise_scale=config["noise_scale"],
        num_workers=8,
        seed=[1] + config["pretrain_seed"],
    )
    model = UnsupervisedModel(
        d_res=192,
        d_lstm=192,
        d_ff=192 * 2,
        d_edge=96,
        n_layers=4,
        n_bp=config["num_bodyparts"],
        sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"],
        dtype="bfloat16",
    )
    trainer = Trainer(
        experiment_name=f"{config['name']}/pretrain",
        model=model,
        optimizer=Adam(0.02),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_batch_size=128,
        seed=[2] + config["pretrain_seed"],
        early_stopping_config={
            "metric_name": "nll",
            "lower_is_better": True,
            "patience": 15000,
        },
        max_training_steps=125000,
    )
    trainer.train()

    os.makedirs(data.persist_dir, exist_ok=True)
    with open(f"{trainer.checkpoint_dir}/best.pkl", "rb") as f1:
        with open(f"{data.persist_dir}/{config['name']}_unsupervised.pkl", "wb") as f2:
            f2.write(f1.read())


class SupervisedModel(Module):
    def __init__(self, d_res, d_ff, d_lstm, n_layers, n_bp, padding, dtype, unsupervised_model):
        self.d_res = d_res
        self.d_ff = d_ff
        self.d_lstm = d_lstm
        self.n_layers = n_layers
        self.dtype = dtype
        self.n_bp = n_bp
        self.padding = padding

        unsupervised_model, unsupervised_path = unsupervised_model
        # Either container. The values are identical; safetensors just needs no pickle.
        unsupervised_params = checkpoints.load_checkpoint(unsupervised_path)
        feat_dim = unsupervised_model.n_layers * unsupervised_model.d_res
        node_dim = unsupervised_model.d_res
        unsupervised_model.set_context({"stage": "eval"})
        self.unsupervised_model = unsupervised_model.set_variables(unsupervised_params)

        self.layers = {}
        self.layers["ff-merge-in"] = Linear(2 * feat_dim, node_dim, dtype, batch_dims=[n_bp])
        self.layers["ff-merge-out"] = Linear(node_dim, node_dim, dtype, batch_dims=[n_bp])
        self.layers["feat-flat-proj"] = Linear(node_dim, d_res, dtype)

        self.layers["lab-embedding"] = Embedding(d_res, len(LABS), dtype)
        for l in range(self.n_layers):
            self.layers[f"lstm-{l}"] = BidirectionalLSTM(d_res, d_lstm, dtype)
            self.layers[f"out-proj-{l}"] = Linear(2 * d_lstm, d_res, dtype)

            self.layers[f"ff-{l}-in"] = Linear(d_res, d_ff, dtype)
            self.layers[f"ff-{l}-out"] = Linear(d_ff, d_res, dtype)

        self.layers["out-proj"] = Linear(d_res, len(ACTIONS), dtype)

    def compute_logits(self, layers, batch, key):
        xs = self.unsupervised_model.extract_features(batch, key)
        x = jnp.concat(xs, axis=-1)
        x_agent, x_target = jnp.split(x, 2, axis=1)
        x = jnp.concat([x_agent, x_target], axis=-1)

        x = layers["ff-merge-in"].apply(x)
        x = jax.nn.silu(x)
        x = layers["ff-merge-out"].apply(x)
        x = jnp.sum(x, axis=2)
        x = layers["feat-flat-proj"].apply(x)

        x += 0.1 * layers["lab-embedding"].apply(batch["lab_id"])
        for l in range(self.n_layers):
            y = layers[f"lstm-{l}"].apply(x)
            y = layers[f"out-proj-{l}"].apply(y)
            x += y

            y = layers[f"ff-{l}-in"].apply(x)
            y = jax.nn.silu(y)
            y = layers[f"ff-{l}-out"].apply(y)
            x += y

        logits = layers["out-proj"].apply(x)
        logits = jnp.transpose(logits, (1, 0, 2))

        logits = logits[:, self.padding : logits.shape[1] - self.padding]
        logits = logits.astype("float32")
        return logits

    def get_labels(self, batch):
        self_labels = jax.nn.one_hot(batch["self_labels"], len(ACTIONS))
        cross_labels = jax.nn.one_hot(batch["cross_labels"], len(ACTIONS))
        labels = self_labels + cross_labels
        label_mask = batch["self_label_mask"] + batch["cross_label_mask"]
        mask = (label_mask == 1) & ((batch["batch_mask"] == 1)[:, None])
        mask = mask.astype("float32")
        return labels, mask

    def compute_loss(self, layers, batch, key):
        logits = self.compute_logits(layers, batch, key)
        labels, mask = self.get_labels(batch)

        log_prob = jnp.where(labels == 1, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits))
        log_prob = (log_prob * mask[:, None]).sum(axis=-1)
        nll_weight = mask.sum()
        nll = -log_prob.mean(axis=1).sum() / nll_weight
        metrics = {"nll": (nll, nll_weight)}
        return (nll, metrics)

    def predict(self, layers, batch, key):
        logits = self.compute_logits(layers, batch, key)
        probs = jax.nn.sigmoid(logits)
        return probs


def get_custom_eval_loop(trainer):
    @jax.jit
    def eval_step(variables, batch, key):
        key, subkey = jax.random.split(key)
        pmodel = trainer.model.set_variables(variables)
        probs = pmodel.predict(batch, subkey)
        return probs, key

    def custom_eval_loop(variables):
        def get_batch_outputs(batch):
            nonlocal key
            probs, key = eval_step(variables, batch, key)
            return batch, probs

        key = jax.random.key(trainer.eval_seed)
        predictions = Predictions(trainer.eval_dataset.videos)

        elements = Pipeline(trainer.eval_dataset.element_iterator())
        batches = elements.batch(trainer.eval_batch_size).to_device(trainer.sharding)
        batch_outputs = batches.map(get_batch_outputs).to_host()
        batch_outputs.unbatch().map(lambda x: predictions.update(*x)).last()
        metrics, _ = predictions.score()
        return metrics

    return custom_eval_loop


def train(config):
    videos = load_videos(mode="train", use_cached=True)
    train_videos, val_videos = split_videos(videos, validation_frac=0.15, random_seed=config["split_seed"])
    val_videos = [v for v in val_videos if v.lab_name not in TRAIN_ONLY_LABS]

    train_dataset = Dataset(
        videos=train_videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=100000,
        unsupervised=False,
        max_scale=config["max_scale"],
        max_time_dilation=config["max_time_dilation"],
        rotate=True,
        flip=True,
        noise_scale=config["noise_scale"],
        num_workers=4,
        seed=[0] + config["train_seed"],
    )
    val_dataset = Dataset(
        videos=val_videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=1,
        unsupervised=False,
        max_scale=1,
        max_time_dilation=1,
        rotate=False,
        flip=False,
        noise_scale=config["noise_scale"],
        num_workers=8,
        seed=[1] + config["train_seed"],
    )
    unsupervised_model = UnsupervisedModel(
        d_res=192,
        d_lstm=192,
        d_ff=192 * 2,
        d_edge=96,
        n_layers=4,
        n_bp=config["num_bodyparts"],
        sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"],
        dtype="bfloat16",
    )
    unsupervised_path = f"{data.persist_dir}/{config['name']}_unsupervised.pkl"
    supervised_model = SupervisedModel(
        d_res=256,
        d_ff=768,
        d_lstm=256,
        n_layers=3,
        n_bp=config["num_bodyparts"],
        padding=32,
        dtype="bfloat16",
        unsupervised_model=(unsupervised_model, unsupervised_path),
    )
    trainer = Trainer(
        experiment_name=f"{config['name']}/train",
        model=supervised_model,
        optimizer=Adam(0.004),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_batch_size=128,
        seed=[2] + config["train_seed"],
        early_stopping_config={
            "metric_name": "f1",
            "lower_is_better": False,
            "patience": 10000,
        },
        max_training_steps=50000,
    )
    trainer.custom_eval_loop = get_custom_eval_loop(trainer)
    trainer.train()

    os.makedirs(data.persist_dir, exist_ok=True)
    with open(f"{trainer.checkpoint_dir}/best.pkl", "rb") as f1:
        with open(f"{data.persist_dir}/{config['name']}_supervised.pkl", "wb") as f2:
            f2.write(f1.read())


def compute_ensemble_thresholds(configs):
    def get_action_counts(videos):
        counts = defaultdict(int)
        videos_filtered = [v for v in videos if v.lab_name not in TRAIN_ONLY_LABS]
        for video in videos_filtered:
            annotation_path = f"{data.dataset_dir}/train_annotation/{video.lab_name}/{video.video_id}.parquet"
            if os.path.isfile(annotation_path):
                df = pd.read_parquet(annotation_path)
                label_counts = df["action"].value_counts()
                for action, v in dict(label_counts).items():
                    if action != "ejaculate":
                        action_id = ACTIONS.encode(action)
                        counts[(video.lab_name, action_id)] += v
        return counts

    rows = []
    for config_name, config in configs.items():
        thresholds = pickle.load(open(f"{data.persist_dir}/{config_name}_thresholds.pkl", "rb"))

        videos = load_videos(mode="train", use_cached=True)
        train_videos, val_videos = split_videos(videos, validation_frac=0.15, random_seed=config["split_seed"])
        train_counts = get_action_counts(train_videos)
        val_counts = get_action_counts(val_videos)

        all_keys = set(train_counts.keys()) | set(val_counts.keys())
        for lab, action in all_keys:
            weight = min(train_counts[(lab, action)], val_counts[(lab, action)])
            threshold = thresholds[lab].get(action, np.nan)
            row = {"config": config_name, "lab": lab, "action": action, "weight": weight, "threshold": threshold}
            rows.append(row)

    ensemble_thresholds = {}
    df = pd.DataFrame.from_records(rows)
    for (lab_name, action_id), action_df in df.groupby(["lab", "action"]):
        b = action_df[~pd.isnull(action_df["threshold"])]
        if len(b) > 0:
            thresholds = []
            for weight, threshold in zip(b["weight"], b["threshold"]):
                thresholds.extend([threshold] * int(round(weight)))

            prior_mean, prior_var = 30, 15**2
            obs_mean, obs_var = np.mean(thresholds), np.var(thresholds)
            pos_mean = (obs_var / (prior_var + obs_var)) * prior_mean + (prior_var / (prior_var + obs_var)) * obs_mean
            ensemble_thresholds[(lab_name, ACTIONS.decode(action_id))] = int(round(pos_mean))

    os.makedirs(data.persist_dir, exist_ok=True)
    pickle.dump(ensemble_thresholds, open(f"{data.persist_dir}/thresholds.pkl", "wb"))


def predict(config, predictions=None, is_val=False):
    if predictions is None:
        is_val = True
        videos = load_videos(mode="train", use_cached=True)
        train_videos, val_videos = split_videos(videos, validation_frac=0.15, random_seed=config["split_seed"])
        val_videos = [v for v in val_videos if v.lab_name not in TRAIN_ONLY_LABS]
        predictions = Predictions(val_videos)

    sharding = None

    dataset = Dataset(
        videos=predictions.videos,
        seq_len=64,
        sample_rate=config["sample_rate"],
        padding=32,
        num_bodyparts=config["num_bodyparts"],
        num_epochs=5,
        unsupervised=False,
        max_scale=0.95 * config["max_scale"],
        max_time_dilation=0.95 * config["max_time_dilation"],
        rotate=True,
        flip=True,
        noise_scale=config["noise_scale"],
        num_workers=8,
        seed=[1] + config["eval_seed"],
    )
    unsupervised_model = UnsupervisedModel(
        d_res=192,
        d_lstm=192,
        d_ff=192 * 2,
        d_edge=96,
        n_layers=4,
        n_bp=config["num_bodyparts"],
        sample_rate=config["sample_rate"],
        aggregation_radius=config["aggregation_radius"],
        dtype="bfloat16",
    )
    unsupervised_path = f"{data.persist_dir}/{config['name']}_unsupervised.pkl"
    supervised_model = SupervisedModel(
        d_res=256,
        d_ff=768,
        d_lstm=256,
        n_layers=3,
        n_bp=config["num_bodyparts"],
        padding=32,
        dtype="bfloat16",
        unsupervised_model=(unsupervised_model, unsupervised_path),
    )

    supervised_path = f"{data.persist_dir}/{config['name']}_supervised.pkl"
    supervised_params = pickle.load(open(supervised_path, "rb"))
    supervised_model.set_context({"stage": "eval"})
    supervised_model = supervised_model.set_variables(supervised_params)

    @jax.jit
    def prediction_step(batch, key):
        key, subkey = jax.random.split(key)
        probs = supervised_model.predict(batch, subkey)
        return probs, key

    def get_batch_outputs(batch):
        nonlocal key
        probs, key = prediction_step(batch, key)
        return batch, probs

    key = jax.random.key(0)
    elements = Pipeline(dataset.element_iterator())
    batches = elements.batch(256).to_device(sharding)
    batch_outputs = batches.map(get_batch_outputs).to_host()
    batch_outputs.unbatch().map(lambda x: predictions.update(*x)).last()

    if is_val:
        metrics, thresholds = predictions.score()
        os.makedirs(data.persist_dir, exist_ok=True)
        pickle.dump(thresholds, open(f"{data.persist_dir}/{config['name']}_thresholds.pkl", "wb"))
        print(metrics)


def train_ensemble():
    configs = get_configs()
    for config_name, config in configs.items():
        pretrain(config)
        train(config)
        predict(config)
    compute_ensemble_thresholds(configs)


def test_ensemble():
    configs = get_configs()
    print("Loading test videos...", flush=True)
    test_videos = load_videos(mode="test", use_cached=True)
    print(f"Loaded {len(test_videos)} test videos", flush=True)
    test_predictions = Predictions(test_videos)
    for i, (config_name, config) in enumerate(configs.items()):
        t0 = time.time()
        print(f"[{i+1}/{len(configs)}] Running config: {config_name}...", flush=True)
        predict(config, predictions=test_predictions)
        print(f"[{i+1}/{len(configs)}] {config_name} done in {time.time()-t0:.1f}s", flush=True)
    print("Generating submission...", flush=True)
    submission_df = test_predictions.to_submission_df()
    submission_path = os.path.join(data.project_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}", flush=True)


# The path globals now live in data.py, next to the code that reads them. Entry points have
# always set them through this module (`solution.working_dir = ...`), so both reads and
# writes are forwarded there. Forwarding *writes* is the point: a plain `from .data import
# data.working_dir` would let an assignment here shadow the real value, the tracking cache would
# be built somewhere the reader never looks, and nothing would complain.
_FORWARDED_PATHS = ("project_dir", "dataset_dir", "working_dir", "persist_dir")


class _SolutionModule(types.ModuleType):
    def __getattr__(self, name):
        if name in _FORWARDED_PATHS:
            return getattr(data, name)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in _FORWARDED_PATHS:
            setattr(data, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SolutionModule

if __name__ == "__main__":
    mode = "test"  # 'train' requires TPU v5e-8, 'test' requires P100

    if mode == "train":
        train_ensemble()

    if mode == "test":
        test_ensemble()