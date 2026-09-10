"""The numpy data pipeline, and the two hazards found when it was split out of solution.py.

`hidra.data` holds the pipeline both backends share. Two things about it are easy to get
wrong, and both were got wrong once:

1. The stream helpers are **not** flat-dict-only. After `predict_into` maps the stream
   through a function returning `(batch, probs)`, everything downstream carries a tuple.
   `jax.tree.*` handled that structure for free; the numpy replacements have to as well.
2. `build_P` must not be built through a *patched* `LABS.encode`. The inference drivers
   replace that attribute with a constant to force one lab's embedding; a P built through it
   routes every lab's head into one row, and the readout einsum then sums all 82 heads.
"""
import numpy as np
import pytest
from conftest import requires_jax, requires_weights

# ------------------------------------------------------------------ pytree helpers

def test_tree_helpers_handle_the_tuple_the_pipeline_actually_carries():
    """The regression test for hazard 1.

    A dict-only `tree_map` raises `AttributeError: 'tuple' object has no attribute 'items'`
    the moment the JAX engine's `(batch, probs)` pair reaches `to_host`.
    """
    from hidra.data import tree_leaves, tree_map

    batch = {"agent": np.zeros((2, 3)), "lab_id": np.array([9, 9])}
    probs = np.ones((2, 4))
    item = (batch, probs)

    doubled = tree_map(lambda x: x * 2, item)
    assert isinstance(doubled, tuple) and len(doubled) == 2
    assert isinstance(doubled[0], dict) and set(doubled[0]) == {"agent", "lab_id"}
    np.testing.assert_array_equal(doubled[1], 2.0)

    assert len(list(tree_leaves(item))) == 3
    # Nested lists too, since jax.tree treated those as structure as well.
    assert tree_map(lambda x: x + 1, {"a": [np.zeros(1), np.zeros(1)]})["a"][1][0] == 1


def test_tree_stack_matches_per_key_stacking():
    from hidra.data import tree_stack

    elems = [{"x": np.array([i, i]), "n": np.int8(i)} for i in range(3)]
    out = tree_stack(elems)
    assert set(out) == {"x", "n"}
    np.testing.assert_array_equal(out["x"], np.array([[0, 0], [1, 1], [2, 2]]))
    np.testing.assert_array_equal(out["n"], np.array([0, 1, 2], dtype=np.int8))

    pairs = [({"x": np.array([i])}, np.array([[i, i]])) for i in range(2)]
    stacked = tree_stack(pairs)
    assert isinstance(stacked, tuple)
    assert stacked[0]["x"].shape == (2, 1) and stacked[1].shape == (2, 1, 2)


def test_get_batch_size_and_unbatch_roundtrip():
    from hidra.data import get_batch_size, unbatch

    batch = ({"agent": np.arange(6).reshape(3, 2), "lab_id": np.array([9, 9, 9])},
             np.arange(12).reshape(3, 4))
    assert get_batch_size(batch) == 3
    elements = list(unbatch(iter([batch])))
    assert len(elements) == 3
    assert isinstance(elements[0], tuple)
    np.testing.assert_array_equal(elements[1][0]["agent"], np.array([2, 3]))
    np.testing.assert_array_equal(elements[2][1], np.arange(8, 12))


def test_batch_pads_a_short_batch_and_flags_it():
    from hidra.data import batch

    def stream(n):
        for i in range(n):
            yield {"x": np.full(2, i, dtype="float32")}

    batches = list(batch(stream(5), 4))
    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0]["batch_mask"], np.ones(4, dtype=np.int8))
    # The short final batch is padded to 4 with batch_mask=0 rows.
    assert batches[1]["x"].shape == (4, 2)
    np.testing.assert_array_equal(batches[1]["batch_mask"], np.array([1, 0, 0, 0], dtype=np.int8))


# ------------------------------------------------------------------ jax-free import

def test_data_module_imports_without_jax():
    """`hidra.data` is the module the whole jax-free claim rests on."""
    import subprocess
    import sys

    script = """
import sys
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("blocked")
        return None
sys.meta_path.insert(0, Blocker())
from hidra import data
assert "jax" not in sys.modules
assert data.Dataset and data.Predictions and data.batch and data.tree_map
print("OK")
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, f"{out.stdout}\n{out.stderr}"
    assert "OK" in out.stdout


def test_solution_forwards_the_path_globals():
    """`solution.working_dir = ...` must still reach the code that reads it.

    The globals moved to data.py; solution forwards reads *and writes*. Forwarding writes is
    the point -- a plain re-import would let an assignment shadow the real value, the
    tracking cache would be built where no reader looks, and nothing would complain.
    """
    from hidra import data, solution

    original = data.working_dir
    try:
        solution.working_dir = "/tmp/hidra-forward-check"
        assert data.working_dir == "/tmp/hidra-forward-check"
        assert solution.working_dir == "/tmp/hidra-forward-check"
    finally:
        data.working_dir = original
    assert solution.working_dir == original

    # Re-exports must be the same objects, so a driver patching one affects both backends.
    assert solution.Dataset is data.Dataset
    assert solution.Predictions is data.Predictions
    assert solution.batch is data.batch


# ------------------------------------------------------------------ P routing

@requires_weights
def test_schema_build_P_survives_the_encode_patch():
    import numpy as np

    from hidra import schema

    before = schema.build_P()
    saved = schema.LABS.encode
    try:
        schema.LABS.encode = lambda value, _i=9: 9      # what the driver does
        after = schema.build_P()
    finally:
        schema.LABS.encode = saved
    np.testing.assert_array_equal(before, after)
    assert after.sum(axis=2).max() == 1.0, "a (lab, action) pair maps to more than one head"


@pytest.mark.jax
@requires_jax
@requires_weights
def test_jax_build_P_survives_the_encode_patch():
    """The regression test for hazard 2, on the JAX side.

    The original captured `LABS.encode` at import time, which only protects it while this
    module is imported *before* the patch. The PyTorch backend imports the JAX engine
    lazily, so that ordering no longer holds -- and with the capture in place, every
    probability came out summed over all 82 heads (values above 3.0 were observed
    end-to-end). Indexing `value_to_idx` removes the ordering dependency.
    """
    import numpy as np

    from hidra import solution, train_perlab_heads

    before = np.asarray(train_perlab_heads.build_P())
    saved = solution.LABS.encode
    try:
        solution.LABS.encode = lambda value, _i=9: 9
        after = np.asarray(train_perlab_heads.build_P())
    finally:
        solution.LABS.encode = saved
    np.testing.assert_array_equal(before, after)
    assert after.sum(axis=2).max() == 1.0
    # And it agrees with the schema-side table.
    from hidra import schema
    np.testing.assert_array_equal(before, schema.build_P())
