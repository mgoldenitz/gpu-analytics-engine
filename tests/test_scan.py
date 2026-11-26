"""Check our exclusive scan against NumPy."""
import cupy as cp
import numpy as np
import pytest

from engine.scan import exclusive_scan


# 511-513 sit on the 512-element block edge; 262_145 = 512 * 512 + 1 needs three scan levels.
@pytest.mark.parametrize("n", [1, 255, 256, 257, 511, 512, 513, 262_145, 1_000_000, 100_000_000])
def test_exclusive_scan_matches_numpy(n):
    rng = np.random.default_rng(7)
    flags = rng.integers(0, 2, size=n, dtype=np.int32)
    expected = np.cumsum(flags) - flags  # inclusive minus self = exclusive

    result = cp.asnumpy(exclusive_scan(cp.asarray(flags)))
    np.testing.assert_array_equal(result, expected)
