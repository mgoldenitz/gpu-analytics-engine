"""Check the GPU filter against pandas."""
import cupy as cp
import numpy as np
import pandas as pd
import pytest

from engine.filter import filter_less_than

VALUE_RANGE = 1_000_000  # values are uniform in [0, 1M), so threshold = pct * 10_000


@pytest.mark.parametrize("scan", ["own", "cub"])
@pytest.mark.parametrize("pct", [0, 1, 10, 50, 90, 100])
@pytest.mark.parametrize("n", [1, 255, 256, 257, 1_000_000])  # sizes around the 256-thread block
def test_filter_matches_pandas(n, pct, scan):
    rng = np.random.default_rng(42)
    values = rng.integers(0, VALUE_RANGE, size=n, dtype=np.int32)
    threshold = pct * VALUE_RANGE // 100

    s = pd.Series(values)
    expected = s[s < threshold].to_numpy()
    if n == 1_000_000:  # only a large sample hits the target selectivity
        assert abs(len(expected) / n * 100 - pct) < 0.5

    result = cp.asnumpy(filter_less_than(cp.asarray(values), threshold, scan=scan))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("threshold, error", [(2.5, TypeError), (2**31, ValueError)])
def test_bad_threshold_rejected(threshold, error):
    with pytest.raises(error):
        filter_less_than(cp.arange(4, dtype=cp.int32), threshold)


def test_numpy_int_threshold_accepted():
    result = filter_less_than(cp.arange(4, dtype=cp.int32), np.int64(2))
    np.testing.assert_array_equal(cp.asnumpy(result), [0, 1])
