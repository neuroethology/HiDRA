"""The seed-flattening fix must not move the random stream.

`reference/solution.py` seeds its epoch shuffle with `np.random.default_rng([epoch_idx,
dataset.seed])`, where `dataset.seed` is itself a list -- a nested sequence that numpy
coerced silently up to 2.4 and rejects from 2.5 on. `hidra.solution.flat_seed` flattens it
explicitly, which is only a safe fix if numpy's nested coercion *was* a flatten.

Element ordering is not cosmetic here: `SimpleEpoch` pairs the shuffled `row_idx` with the
sequential `element_idx`, and `element_idx` seeds the per-element augmentation RNG. A
different permutation would therefore augment different chunks differently and change every
prediction.

The GOLDEN values below were captured on numpy 2.4.6 from the *nested* form, so this test
keeps working on numpy >= 2.5 where the nested form can no longer even be constructed.
"""
import numpy as np
import pytest

from hidra.solution import flat_seed

# (epoch_idx, dataset_seed) -> what default_rng([epoch_idx, dataset_seed]) produced on
# numpy 2.4.6: rng.bit_generator.jumped() then permutation(24), plus a plain uniform(4).
GOLDEN = [
    ((0, [1, 3, 0, 123456789]),
     [20, 3, 6, 19, 5, 7, 11, 18, 17, 8, 0, 14, 13, 12, 9, 2, 22, 1, 21, 16, 15, 23, 4, 10],
     [0.22747230645745276, 0.8250004520138394, 0.8176111968612919, 0.04690652507019488]),
    ((2, [1, 3, 4, 123456789]),
     [4, 11, 7, 8, 22, 17, 10, 0, 19, 21, 9, 16, 3, 2, 15, 23, 13, 18, 6, 1, 20, 12, 14, 5],
     [0.809009161530799, 0.954635871920077, 0.29131148275176877, 0.19701699015027263]),
    ((0, [0, 1, 2, 3]),
     [5, 22, 23, 2, 4, 7, 3, 21, 6, 18, 10, 16, 15, 9, 8, 1, 17, 13, 0, 11, 19, 20, 14, 12],
     [0.0022758928290496083, 0.7776582695502101, 0.6045843319473571, 0.09258104442326276]),
    # A bare int seed, the other shape Dataset.seed can take (secrets.randbits fallback).
    ((7, 42),
     [7, 2, 9, 23, 10, 16, 4, 5, 20, 18, 0, 22, 13, 8, 14, 12, 15, 19, 3, 6, 17, 1, 21, 11],
     [0.6311650112196174, 0.12063973429477726, 0.0009510417971345664, 0.5853008800307393]),
]


@pytest.mark.parametrize("spec,want_perm,want_uniform", GOLDEN)
def test_flat_seed_reproduces_nested_stream(spec, want_perm, want_uniform):
    epoch_idx, seed = spec
    flat = flat_seed(epoch_idx, seed)

    rng = np.random.default_rng(flat)
    rng.bit_generator.state = rng.bit_generator.jumped().state
    assert rng.permutation(24).tolist() == want_perm

    got = np.random.default_rng(flat).uniform(size=4)
    assert got.tolist() == pytest.approx(want_uniform, abs=0.0)


@pytest.mark.skipif(np.lib.NumpyVersion(np.__version__) >= "2.5.0",
                    reason="numpy >= 2.5 rejects nested SeedSequence input outright")
@pytest.mark.parametrize("spec,_p,_u", GOLDEN)
def test_flat_matches_nested_directly(spec, _p, _u):
    """On numpy < 2.5, compare against the nested form itself rather than the goldens."""
    epoch_idx, seed = spec
    nested = np.random.default_rng([epoch_idx, seed]).permutation(64)
    flat = np.random.default_rng(flat_seed(epoch_idx, seed)).permutation(64)
    np.testing.assert_array_equal(nested, flat)


def test_flat_seed_shape():
    assert flat_seed(0, [1, 3, 0, 7]) == [0, 1, 3, 0, 7]
    assert flat_seed(5, 9) == [5, 9]
    assert flat_seed(1, 2, [3, [4, 5]]) == [1, 2, 3, 4, 5]
    assert all(isinstance(v, int) for v in flat_seed(np.int32(3), [np.int64(4)]))
