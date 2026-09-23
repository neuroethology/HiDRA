"""Equivalence of the training machinery: optimizer, EMA, loss, gradients, and DDI.

Training cannot be compared the way inference is -- two runs diverge from step one, because
float32 round-off in the first gradient changes every later batch's parameters. So the
argument is assembled from pieces that *are* exactly checkable:

  * **Optimizer** -- given identical parameters and gradients, N steps of the port must
    land where `solution.Adam` / `SchedAdam` land. Pure arithmetic, no model involved.
  * **EMA** -- likewise, over a sequence of updates.
  * **Loss** -- the port's objective must equal `MultiTaskPerLabModel.compute_loss` on the
    same batch, running under the real LABTAIL environment.
  * **Gradients** -- the port's analytic gradients must equal float64 central differences
    *of the port's own loss*. Combined with the loss check, that pins the gradient to the
    JAX forward pass without needing JAX's autodiff or a re-implementation of the loss:
    if forward_torch == forward_jax and backward_torch == d(forward_torch), then
    backward_torch == d(forward_jax).
  * **Data-dependent init** -- the recalibrated input statistics must match.

The remaining gap is the optimization *trajectory*, which is inherently chaotic and is
covered by an end-to-end smoke run rather than a numerical assertion.
"""
import os
import pickle
import subprocess
import sys

import exactness as X
import numpy as np
import pytest
import torch
from conftest import requires_cuda, requires_jax, requires_weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Two independent float32 implementations of the same update, over many steps.
TOL_OPT = 1e-5


# ------------------------------------------------------------------ optimizer / EMA

def _grad_sequence(shapes, n, seed=0):
    rng = np.random.default_rng(seed)
    p0 = [rng.standard_normal(s).astype("float32") for s in shapes]
    grads = [[rng.standard_normal(s).astype("float32") * 0.1 for s in shapes] for _ in range(n)]
    return p0, grads


SHAPES = [(5, 8), (8,), (3, 4, 6), ()]

OPT_CASES = [
    ("adam-default-lr", dict(learning_rate=0.004), {}),
    ("adam-pretrain-lr", dict(learning_rate=0.02), {}),
    ("cosine", dict(learning_rate=0.004, total=20, floor=1e-4), dict(total=20, floor=1e-4)),
    ("weight-decay", dict(learning_rate=0.004, weight_decay=0.01), dict(weight_decay=0.01)),
    ("grad-clip", dict(learning_rate=0.004, grad_clip=0.05), dict(grad_clip=0.05)),
    ("all-three", dict(learning_rate=0.004, total=15, floor=5e-4, weight_decay=0.01,
                       grad_clip=0.05),
     dict(total=15, floor=5e-4, weight_decay=0.01, grad_clip=0.05)),
]


@pytest.mark.jax
@requires_jax
@pytest.mark.parametrize("name,torch_kw,jax_kw", OPT_CASES, ids=[c[0] for c in OPT_CASES])
def test_optimizer_matches(name, torch_kw, jax_kw):
    import jax.numpy as jnp

    from hidra.torch.train import SchedAdam as TSchedAdam
    from hidra.train_perlab_heads import SchedAdam as JSchedAdam

    p0, grads = _grad_sequence(SHAPES, 25, seed=0)

    jopt = JSchedAdam(torch_kw["learning_rate"], **jax_kw)
    jparams = [jnp.asarray(p) for p in p0]
    jstate = jopt.create_variables(jparams)
    for g in grads:
        jparams, jstate = jopt.update(jparams, jstate, [jnp.asarray(x) for x in g])

    tparams = [torch.nn.Parameter(torch.as_tensor(p.copy())) for p in p0]
    topt = TSchedAdam(tparams, **torch_kw)
    for g in grads:
        for p, gi in zip(tparams, g, strict=True):
            p.grad = torch.as_tensor(gi.copy())
        topt.step()

    for i, (jp, tp) in enumerate(zip(jparams, tparams, strict=True)):
        a = np.asarray(jp, np.float64)
        b = tp.detach().numpy().astype(np.float64)
        err = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
        assert err < TOL_OPT, f"{name}: tensor {i} diverged by {err:.2e} after 25 steps"


@pytest.mark.jax
@requires_jax
def test_ema_matches():
    import jax.numpy as jnp

    from hidra import solution
    from hidra.torch.train import EMA as TEMA

    p0, updates = _grad_sequence(SHAPES, 30, seed=1)
    jema = solution.EMA(0.9993)
    jstate = jema.create_variables([jnp.asarray(p) for p in p0])
    tema = TEMA({str(i): torch.as_tensor(p.copy()) for i, p in enumerate(p0)}, 0.9993)

    for step in updates:
        jstate = jema.update(jstate, [solution.Variable(jnp.asarray(x), True) for x in step])
        tema.update({str(i): torch.as_tensor(x.copy()) for i, x in enumerate(step)})

    jvals = jema.values(jstate)
    tvals = tema.values()
    assert tema.t == float(jstate["t"]), "EMA step counters diverged"
    for i in range(len(SHAPES)):
        a = np.asarray(jvals[i], np.float64)
        b = tvals[str(i)].numpy().astype(np.float64)
        err = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
        assert err < TOL_OPT, f"EMA tensor {i} diverged by {err:.2e}"


def test_ema_is_unbiased_from_the_first_step():
    """`values()` divides by 1 - decay**t, so the average equals the input immediately.

    Without the correction a 0.9993-decay average would start at 0.07% of the weights and
    take thousands of steps to catch up -- and it is the EMA that gets checkpointed.
    """
    from hidra.torch.train import EMA

    w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    ema = EMA({"w": w}, 0.9993)
    np.testing.assert_allclose(ema.values()["w"].numpy(), w.numpy(), rtol=1e-6)


def test_checkpoint_manager_keeps_best_and_stops_on_patience(tmp_path):
    from hidra.torch.train import CheckpointManager

    cm = CheckpointManager(str(tmp_path), max_checkpoints=3, metric_name="f1",
                           lower_is_better=False, patience=200)
    state = {"w": torch.zeros(2)}
    for step, f1 in [(0, 0.1), (100, 0.5), (200, 0.3), (300, 0.4)]:
        assert cm.update(state, step, {"f1": f1}) is False
    # best is step 100 (f1 0.5); at step 400 we are 300 past it, beyond patience 200
    assert cm.update(state, 400, {"f1": 0.2}) is True
    kept = sorted(int(p.stem) for p in tmp_path.glob("*.pkl"))
    assert len(kept) <= 3
    assert 100 in kept, "the best-scoring checkpoint was evicted"


def test_patience_zero_turns_early_stopping_off(tmp_path, monkeypatch):
    """`finetune.py --patience 0` sets LABTAIL_PATIENCE=0, and the fine-tuner then gets no
    patience at all: a config runs its full --steps however flat validation goes."""
    from hidra.torch.train import CheckpointManager
    from hidra.torch.train_perlab import env_settings

    monkeypatch.setenv("LABTAIL", "NiftyGoldfinch")
    monkeypatch.delenv("LABTAIL_PATIENCE", raising=False)
    assert env_settings("15fps_5bp")["patience"] == 10000
    monkeypatch.setenv("LABTAIL_PATIENCE", "0")
    patience = env_settings("15fps_5bp")["patience"] or None   # as finetune_config passes it
    cm = CheckpointManager(str(tmp_path), metric_name="f1", lower_is_better=False,
                           patience=patience)
    state = {"w": torch.zeros(2)}
    assert cm.update(state, 0, {"f1": 0.9}) is False
    assert cm.update(state, 10**6, {"f1": 0.1}) is False


def test_checkpoint_manager_rejects_nan(tmp_path):
    """A NaN metric means the run diverged; silently keeping going wastes hours."""
    from hidra.torch.train import CheckpointManager

    cm = CheckpointManager(str(tmp_path), metric_name="f1", lower_is_better=False)
    with pytest.raises(ValueError, match="NaN"):
        cm.update({"w": torch.zeros(1)}, 0, {"f1": float("nan")})


# ------------------------------------------------------------------ trainable-layer selection

@requires_weights
@requires_cuda
@pytest.mark.parametrize("mode,expected", [
    ("head", {"out-proj-perlab"}),
    ("embedding", {"lab-embedding", "out-proj-perlab"}),
    ("tail", None),
])
def test_mode_freezes_the_right_layers(mode, expected, torch_head):
    """The trunk and the shared feature merge must stay frozen in every mode.

    The merge maps trunk features to the 256-d stream every lab shares; fine-tuning it on
    one user's few videos would move all 15 labs' heads off their calibration.
    """
    from hidra.torch import train as T

    want = T.MODE_LAYERS[mode] if expected is None else expected
    assert T.MODE_LAYERS[mode] == want
    T.set_trainable_layers(torch_head, T.MODE_LAYERS[mode])

    for name, module in torch_head.layers.items():
        params = list(module.parameters())
        if not params:
            continue
        trainable = {p.requires_grad for p in params}
        assert len(trainable) == 1, f"{name}: mixed requires_grad within one layer"
        assert trainable.pop() is (name in want), f"{name}: wrong requires_grad for mode={mode}"

    for layer in T.MERGE_LAYERS:
        assert not any(p.requires_grad for p in torch_head.layers[layer].parameters()), \
            f"the shared feature merge layer {layer} must never be trained"
    assert not any(p.requires_grad for p in torch_head.unsupervised_model.parameters()), \
        "the self-supervised trunk must stay frozen"

    # Restore, so later tests in the session see a clean model.
    T.set_trainable_layers(torch_head, set())


@requires_weights
def test_head_column_selector(config_name):
    from hidra import schema
    from hidra.torch.train import head_column_selector

    lab_action = schema.lab_action_table()
    col = head_column_selector(lab_action, "GroovyShrew", ["rear"])
    assert col.sum() == 1
    j = int(col.argmax())
    assert lab_action[j] == ("GroovyShrew", "rear")

    # No action filter -> every one of that lab's heads.
    allcol = head_column_selector(lab_action, "GroovyShrew")
    assert int(allcol.sum()) == sum(1 for l, _ in lab_action if l == "GroovyShrew")

    with pytest.raises(ValueError, match="no head column"):
        head_column_selector(lab_action, "GroovyShrew", ["nonexistent"])
    with pytest.raises(ValueError, match="no head column"):
        head_column_selector(lab_action, "BoisterousParrot", ["rear"])  # has only shepherd


# ------------------------------------------------------------------ loss and gradients

@pytest.mark.jax
@requires_jax
@requires_weights
@requires_cuda
def test_loss_matches_jax_compute_loss(track_dir):
    """The port's objective vs `MultiTaskPerLabModel.compute_loss` under real LABTAIL env.

    Runs in a subprocess: LABTAIL is read at *import* time by train_perlab_heads, and the
    column mask and loss-term selection are decided from it, so it cannot be set after the
    module is already imported in the test session.
    """
    script = f'''
import os, sys
os.environ.update(SNIFFALL="1", LABTAIL="GroovyShrew", LABTAIL_ACTIONS="rear",
                  LABTAIL_TAG="test", TF_CPP_MIN_LOG_LEVEL="3",
                  XLA_FLAGS="--xla_gpu_autotune_level=0",
                  XLA_PYTHON_CLIENT_PREALLOCATE="false")
sys.path.insert(0, {os.path.join(REPO, "tests")!r})
import numpy as np, torch, jax, jax.numpy as jnp
import exactness as X
from hidra import schema
from hidra import train_perlab_heads as JT
from hidra.torch import train as TT
from hidra.torch.infer import to_torch_batch

assert JT.LABTAIL == "GroovyShrew"
assert int(JT.LABTAIL_COLS.sum()) == 1, JT.LABTAIL_COLS.sum()

videos = X.stage_videos({str(track_dir)!r}, os.environ["PYTEST_TMP"])
jhead, config = X.jax_models()
_, thead, _ = X.torch_models(device="cuda")
batch = dict(X.make_batch(config, videos, batch_size=4))

# Synthetic annotations: the loss arithmetic is indifferent to whether labels are real,
# and the staged synthetic videos carry none.
rng = np.random.default_rng(7)
n_act = len(schema.ACTIONS); B, Tn = batch["self_labels"].shape
rear = schema.ACTIONS.value_to_idx["rear"]
batch["self_labels"] = np.where(rng.random((B, Tn)) < 0.3, rear, 0).astype("int16")
batch["cross_labels"] = np.zeros((B, Tn), dtype="int16")
slm = np.zeros((B, n_act), dtype="int8"); slm[:, rear] = 1
batch["self_label_mask"] = slm
batch["cross_label_mask"] = np.zeros((B, n_act), dtype="int8")

with X.jax_matmul_precision("float32"):
    jloss, jmetrics = jhead.compute_loss({{k: jnp.asarray(v) for k, v in batch.items()}},
                                         jax.random.key(0))
jloss = float(jloss)

cols = TT.head_column_selector(thead.lab_action, "GroovyShrew", ["rear"], device="cuda")
tb = to_torch_batch(batch, "cuda")
for k in ("self_labels", "cross_labels", "self_label_mask", "cross_label_mask", "batch_mask"):
    tb[k] = torch.as_tensor(np.asarray(batch[k]), device="cuda")
thead.set_stage("train")
with torch.no_grad():
    tloss, tmet = TT.perlab_loss(thead, tb, head_columns=cols)
tloss = float(tloss)

rel = abs(jloss - tloss) / abs(jloss)
print(f"jax {{jloss:.10f}} torch {{tloss:.10f}} rel {{rel:.3e}}")
assert jloss > 0.01, f"loss {{jloss}} is degenerate; the comparison would be vacuous"
assert rel < 1e-5, f"loss disagrees by {{rel:.3e}}"
assert abs(float(jmetrics["nll88"][1]) - tmet["nll_perlab"][1]) < 1e-6, "mask weights differ"
print("OK")
'''
    env = {**os.environ, "PYTEST_TMP": os.path.join(str(track_dir), "_lossmatch")}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=1800, cwd=REPO, env=env)
    assert out.returncode == 0, f"stdout={out.stdout[-3000:]}\nstderr={out.stderr[-3000:]}"
    assert "OK" in out.stdout, out.stdout
    print("\n" + "\n".join(l for l in out.stdout.splitlines() if l.startswith("jax ")))


@requires_weights
@requires_cuda
def test_gradients_match_float64_finite_differences(config, videos):
    """Analytic gradients vs float64 central differences of the same loss.

    Checked on one parameter per representative layer -- the output head, the lab
    embedding, a feed-forward weight, and both directions of two different BiLSTM blocks --
    perturbing each tensor's largest-gradient entry, which is the most informative
    coordinate and the one least swamped by round-off.
    """
    from hidra import schema
    from hidra.torch import load_perlab, load_unsupervised
    from hidra.torch import train as T
    from hidra.torch.infer import to_torch_batch

    batch = dict(X.make_batch(config, videos, batch_size=2))
    rng = np.random.default_rng(7)
    n_act = len(schema.ACTIONS)
    B, Tn = batch["self_labels"].shape
    rear = schema.ACTIONS.value_to_idx["rear"]
    batch["self_labels"] = np.where(rng.random((B, Tn)) < 0.3, rear, 0).astype("int16")
    batch["cross_labels"] = np.zeros((B, Tn), dtype="int16")
    slm = np.zeros((B, n_act), dtype="int8")
    slm[:, rear] = 1
    batch["self_label_mask"] = slm
    batch["cross_label_mask"] = np.zeros((B, n_act), dtype="int8")

    name = config["name"]
    trunk = load_unsupervised(name, dtype=torch.float64, device="cuda", config=config)
    head = load_perlab(name, trunk, dtype=torch.float64, device="cuda", config=config)
    T.set_trainable_layers(head, T.MODE_LAYERS["tail"])
    head.set_stage("train")
    cols = T.head_column_selector(head.lab_action, "GroovyShrew", ["rear"],
                                  device="cuda").double()

    tb = to_torch_batch(batch, "cuda")
    for k in ("self_labels", "cross_labels", "self_label_mask", "cross_label_mask",
              "batch_mask"):
        tb[k] = torch.as_tensor(np.asarray(batch[k]), device="cuda")
    tb = {k: (v.double() if v.dtype.is_floating_point else v) for k, v in tb.items()}

    def loss_value():
        with torch.no_grad():
            value, _ = T.perlab_loss(head, tb, head_columns=cols)
        return float(value)

    # Determinism is a precondition: a nondeterministic forward makes the whole check noise.
    assert loss_value() == loss_value(), "the float64 forward pass is not deterministic"

    loss, _ = T.perlab_loss(head, tb, head_columns=cols)
    head.zero_grad(set_to_none=True)
    loss.backward()
    assert float(loss.detach()) > 0.01, "degenerate loss; the gradient check would be vacuous"

    named = dict(head.named_parameters())
    targets = ["layers.out-proj-perlab.w", "layers.out-proj-perlab.b",
               "layers.out-proj-perlab.s", "layers.lab-embedding.w",
               "layers.ff-0-in.w", "layers.ff-2-out.w",
               "layers.lstm-0.lstm_fw.is_linear.w",
               "layers.lstm-2.lstm_bw.ss_linear.w", "layers.out-proj-1.b"]
    eps = 1e-5
    worst = 0.0
    for target in targets:
        p = named[target]
        assert p.grad is not None, f"{target} received no gradient"
        idx = np.unravel_index(int(torch.argmax(p.grad.abs())), tuple(p.grad.shape))
        analytic = float(p.grad[idx])
        assert analytic != 0.0, f"{target} has an all-zero gradient"

        with torch.no_grad():
            p[idx] += eps
        plus = loss_value()
        with torch.no_grad():
            p[idx] -= 2 * eps
        minus = loss_value()
        with torch.no_grad():
            p[idx] += eps
        numeric = (plus - minus) / (2 * eps)

        rel = abs(analytic - numeric) / max(abs(numeric), 1e-30)
        worst = max(worst, rel)
        assert rel < 1e-4, (f"{target}: analytic {analytic:.6e} vs finite-difference "
                            f"{numeric:.6e} (rel {rel:.2e})")
    print(f"\nworst analytic-vs-finite-difference error over {len(targets)} tensors: {worst:.2e}")


@pytest.mark.jax
@requires_jax
@requires_weights
@requires_cuda
def test_ddi_recalibration_matches(config, videos, jax_head, device):
    """The data-dependent-init pass must move the same statistics the same way.

    Both output projections have to be included: `Trainer.get_ddi_loop` calls
    `compute_loss`, which applies them, so their input statistics get calibrated too.
    Running only the trunk would leave `out-proj-perlab` -- the one layer every fine-tune
    mode trains -- reading its inputs through the source lab's statistics.
    """
    import jax
    import jax.numpy as jnp

    from hidra.torch import train as T
    from hidra.torch.infer import iter_batches, make_dataset, to_torch_batch

    n_batches = 16
    dataset = make_dataset(config, videos, num_epochs=3)
    batches = []
    for i, b in enumerate(iter_batches(dataset, 4)):
        batches.append(b)
        if i + 1 >= n_batches:
            break
    assert len(batches) == n_batches

    jax_head.module.set_context({"stage": "init", "init_decay": 0.99})
    with X.jax_matmul_precision("float32"):
        for b in batches:
            jax_head.compute_loss({k: jnp.asarray(v) for k, v in b.items()},
                                  jax.random.key(0))
    jax_head.module.set_context({"stage": "eval"})

    _, head, _ = X.torch_models(config["name"], device=device)
    head.set_stage("init", init_decay=0.99)
    with torch.no_grad():
        for b in batches:
            T.ddi_forward(head, to_torch_batch(b, device))
    head.set_stage("eval")

    checked = ["feat-flat-proj", "ff-0-in", "ff-2-out", "out-proj", "out-proj-perlab"]
    for layer in checked:
        jvars = jax_head.layers[layer].variables
        tl = head.layers[layer]
        # The counter is integer-valued arithmetic and must agree exactly.
        assert float(jvars["n"]) == pytest.approx(float(tl.n), abs=1e-4), \
            f"{layer}: DDI step counter differs ({float(jvars['n'])} vs {float(tl.n)})"
        assert float(tl.n) > 256, f"{layer} was not visited by the DDI pass"
        for stat in ("mean", "std"):
            a = np.asarray(jvars[stat], np.float64)
            b = getattr(tl, stat).double().cpu().numpy()
            err = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
            # Looser than float32 round-off on purpose: the statistics are accumulated from
            # activations, and the JAX activations carry its TF32 matmul error (~1e-3).
            assert err < 1e-2, f"{layer}.{stat} differs by {err:.2e}"


# ------------------------------------------------------------------ checkpoint interop

@requires_weights
def test_saved_checkpoint_uses_the_jax_variable_layout(config_name, tmp_path):
    """A torch-trained checkpoint must load on either backend.

    `predict.py --weights` feeds the file to whichever backend is selected, so a fine-tune
    would be stranded if the layout were torch-specific.
    """
    from hidra import paths
    from hidra.torch.checkpoint import flatten_tree, load_jax_checkpoint
    from hidra.torch.train_perlab import to_jax_layout

    published = load_jax_checkpoint(
        paths.models_dir() / f"{config_name}_supervised_perlab_sniffall.pkl")
    # Build a state dict shaped like the trainer's, from the published weights.
    state = {}
    for path, value in flatten_tree(published).items():
        torch_name = "layers." + path.replace("lstm-fw", "lstm_fw").replace("lstm-bw", "lstm_bw").replace("/", ".")
        state[torch_name] = torch.as_tensor(np.asarray(value))
    state["P"] = torch.zeros(2, 2, 2)  # must be dropped, it is derived not trained

    tree = to_jax_layout(state)
    assert "P" not in tree
    got = set(flatten_tree(tree))
    want = set(flatten_tree(published))
    assert got == want, f"layout differs: missing {sorted(want - got)[:5]}, extra {sorted(got - want)[:5]}"
    for key in want:
        np.testing.assert_array_equal(flatten_tree(tree)[key], np.asarray(flatten_tree(published)[key]))


# ------------------------------------------------------------------ end-to-end fine-tuning

@pytest.fixture(scope="module")
def annotated_dataset(track_dir, tmp_path_factory):
    """Stage the synthetic videos with synthetic `rear` annotations via `finetune.py prepare`.

    Both mice are annotated on purpose. `solution.epochs_supervised` forms an (agent, target)
    pair per annotated agent and falls back to `rng.choice` over the other mice when an agent
    has no cross-directed labels -- with only one mouse annotated that choice is over an empty
    list and the trainer dies inside numpy. Real annotation sets normally cover both.
    """
    import pandas as pd

    out = tmp_path_factory.mktemp("annotated")
    bouts_csv = out / "bouts.csv"
    rows = []
    rng = np.random.default_rng(3)
    for pq in sorted(track_dir.glob("*.parquet")):
        n = int(pd.read_parquet(pq, columns=["video_frame"]).video_frame.max() + 1)
        for mouse in ("mouse1", "mouse2"):
            t = int(rng.integers(30, 90))
            while t < n - 60:
                dur = int(rng.integers(15, 45))
                rows.append(dict(file=pq.name, agent=mouse, target=mouse, action="rear",
                                 start_frame=t, stop_frame=t + dur))
                t += dur + int(rng.integers(70, 180))
    pd.DataFrame(rows).to_csv(bouts_csv, index=False)

    staged = out / "ft_data"
    res = subprocess.run(
        [sys.executable, os.path.join(REPO, "finetune.py"), "prepare",
         "--tracking", str(track_dir), "--annotations", str(bouts_csv),
         "--lab", "GroovyShrew", "--out", str(staged),
         "--pix-per-cm", "16", "--fps", "30"],
        capture_output=True, text=True, timeout=900, cwd=REPO)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    return staged


@pytest.mark.slow
@requires_weights
@requires_cuda
def test_finetune_runs_and_improves_the_head(annotated_dataset, tmp_path):
    """A real fine-tune: the objective must fall and the head must move the right way.

    Asserting on the loss trajectory rather than exact numbers, because a training run is
    chaotic -- float32 round-off in the first gradient changes every later batch. What is
    checkable is the direction: the supervised behaviour's probability should rise sharply
    on data annotated for it.
    """
    out_dir = tmp_path / "ft_models"
    env = {**os.environ, "LABTAIL_DDI_STEPS": "16"}
    res = subprocess.run(
        [sys.executable, os.path.join(REPO, "finetune.py"), "train",
         "--data", str(annotated_dataset), "--lab", "GroovyShrew", "--actions", "rear",
         "--out", str(out_dir), "--tag", "pytest", "--mode", "head", "--steps", "150",
         "--configs", "15fps_5bp", "--eval-interval", "150", "--backend", "torch"],
        capture_output=True, text=True, timeout=3600, cwd=REPO, env=env)
    assert res.returncode == 0, f"{res.stdout[-4000:]}\n{res.stderr[-4000:]}"
    assert "trained 1/1 config" in res.stdout, res.stdout[-2000:]

    ckpt = out_dir / "15fps_5bp__pytest.pkl"
    assert ckpt.is_file(), "no checkpoint written"

    # Only the trained layer may differ from the published weights.
    from hidra import paths
    from hidra.torch.checkpoint import flatten_tree, load_jax_checkpoint

    published = flatten_tree(load_jax_checkpoint(
        paths.models_dir() / "15fps_5bp_supervised_perlab_sniffall.pkl"))
    with open(ckpt, "rb") as f:
        trained = flatten_tree(pickle.load(f))

    # The published file carries no "norm" leaf for the head, so the key sets should match.
    assert set(trained) == set(published), (
        f"layout drift: missing {sorted(set(published) - set(trained))[:5]}, "
        f"extra {sorted(set(trained) - set(published))[:5]}")

    def rel_change(key):
        a = np.asarray(published[key], np.float64)
        b = np.asarray(trained[key], np.float64)
        return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30))

    weight_keys = [k for k in trained if k.rsplit("/", 1)[-1] in ("w", "s", "b", "c")]
    head_keys = [k for k in weight_keys if k.startswith("out-proj-perlab/")]
    frozen_keys = [k for k in weight_keys if not k.startswith("out-proj-perlab/")]
    assert head_keys and frozen_keys

    head_change = max(rel_change(k) for k in head_keys)
    frozen_change = max(rel_change(k) for k in frozen_keys)
    assert head_change > 1e-3, f"the trained head barely moved ({head_change:.2e})"
    # Frozen layers are *not* bit-identical, and should not be expected to be: the EMA
    # tracks every variable, trainable or not, and `values()` returns `sums / (1 -
    # decay**t)`, which recovers a constant weight exactly in real arithmetic but not in
    # float32. The JAX original does the identical round-trip. What matters is that no
    # actual gradient leaked, i.e. the drift stays at round-off.
    assert frozen_change < 1e-5, (
        f"frozen layers moved by {frozen_change:.2e}, far above the EMA round-trip's "
        f"~1e-7 -- mode=head leaked gradient outside out-proj-perlab")
    assert head_change > 100 * frozen_change, (
        f"head change {head_change:.2e} is not clearly above the frozen-layer floor "
        f"{frozen_change:.2e}")

    # And the loss came down over training.
    losses = [float(line.split("nll_perlab:")[1].split()[0])
              for line in res.stdout.splitlines() if "nll_perlab:" in line]
    assert len(losses) >= 2, f"expected several logged losses, got {losses}"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


@pytest.mark.slow
@requires_weights
@requires_cuda
def test_finetuned_checkpoint_loads_on_both_backends(annotated_dataset, track_dir, tmp_path):
    """A torch-trained checkpoint must be usable from the JAX backend, and vice versa.

    Fine-tuning is expensive -- five configs, one per ensemble member -- so a checkpoint
    must not be tied to the backend that produced it. This is also the test that caught the
    JAX loader assuming its weights were already JAX arrays: a numpy-valued checkpoint made
    `Embedding.apply` hand a traced index to numpy and crash inside jit.
    """
    out_dir = tmp_path / "ft_models"
    env = {**os.environ, "LABTAIL_DDI_STEPS": "8"}
    res = subprocess.run(
        [sys.executable, os.path.join(REPO, "finetune.py"), "train",
         "--data", str(annotated_dataset), "--lab", "GroovyShrew", "--actions", "rear",
         "--out", str(out_dir), "--tag", "interop", "--mode", "head", "--steps", "60",
         "--configs", "15fps_5bp", "--eval-interval", "1000", "--backend", "torch"],
        capture_output=True, text=True, timeout=3600, cwd=REPO, env=env)
    assert res.returncode == 0, f"{res.stdout[-3000:]}\n{res.stderr[-3000:]}"

    template = str(out_dir / "{config}__interop.pkl")
    outputs = {}
    for backend in ("torch", "jax"):
        dest = tmp_path / f"pred_{backend}"
        res = subprocess.run(
            [sys.executable, os.path.join(REPO, "predict.py"), str(track_dir),
             "--out", str(dest), "--labs", "GroovyShrew", "--actions", "rear",
             "--configs", "15fps_5bp", "--gpu", "0", "--backend", backend,
             "--weights", template],
            capture_output=True, text=True, timeout=3600, cwd=REPO)
        assert res.returncode == 0, f"{backend}: {res.stdout[-3000:]}\n{res.stderr[-3000:]}"
        assert "inference exited" not in res.stdout, \
            f"{backend} could not load the checkpoint:\n{res.stdout[-3000:]}"
        import pandas as pd
        frames = dest / "synth00.frames.parquet"
        assert frames.is_file(), f"{backend} wrote no probability track"
        outputs[backend] = pd.read_parquet(frames).sort_values(
            ["subject", "target", "lab", "action", "frame"], ignore_index=True)

    a, b = outputs["jax"], outputs["torch"]
    assert len(a) == len(b) > 0
    diff = np.abs(a.prob.to_numpy(np.float64) - b.prob.to_numpy(np.float64)).max()
    assert diff < 2e-3, f"the same checkpoint gave different answers per backend: {diff:.3e}"
    assert (a.call.to_numpy() == b.call.to_numpy()).all(), "calls differ across backends"
    print(f"\ntorch-trained checkpoint on both backends: max|prob diff| {diff:.2e}, "
          f"calls identical")


@pytest.mark.slow
@pytest.mark.jax
@requires_jax
@requires_weights
@requires_cuda
def test_jax_finetune_runs_from_safetensors_only(annotated_dataset, tmp_path):
    """`finetune.py train --backend jax` must work on a fresh install, i.e. a models
    directory holding only safetensors.

    The JAX trainer used to hardcode `{config}_unsupervised.pkl` and
    `{config}_supervised_perlab_sniffall.pkl` for its warm start -- the only two load sites
    that did not go through `hidra.checkpoints` -- so it died with FileNotFoundError the
    moment the Hub switched to safetensors, while the torch backend and inference on both
    backends kept working. A smoke run is enough: the failure was at model construction.
    """
    from hidra import paths

    models = paths.models_dir()
    safetensors = sorted(models.glob("*.safetensors"))
    if len(safetensors) < 10 or not (models / "thresholds.json").is_file():
        pytest.skip("weights not converted; run hidra-convert-weights")
    only = tmp_path / "models"
    only.mkdir()
    for f in safetensors:
        (only / f.name).symlink_to(f)
    (only / "thresholds.json").symlink_to(models / "thresholds.json")

    res = subprocess.run(
        [sys.executable, os.path.join(REPO, "finetune.py"), "train",
         "--data", str(annotated_dataset), "--lab", "GroovyShrew", "--actions", "rear",
         "--out", str(tmp_path / "ft_models"), "--tag", "jaxsmoke", "--mode", "head",
         "--configs", "15fps_5bp", "--backend", "jax", "--smoke",
         "--workdir", str(tmp_path / "work")],
        capture_output=True, text=True, timeout=3600, cwd=REPO,
        env={**os.environ, "HIDRA_MODELS_DIR": str(only)})
    assert res.returncode == 0, f"{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    assert "trained 1/1 config" in res.stdout, res.stdout[-2000:]
    assert "warm-start copied" in res.stdout, "the JAX trainer did not warm-start from the published head"
