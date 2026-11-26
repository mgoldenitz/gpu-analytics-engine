"""Check the GPU hash group-by against pandas."""
import cupy as cp
import numpy as np
import pandas as pd
import pytest

from engine.groupby import group_by_sum_count

INT_MIN = np.iinfo(np.int32).min


def make_keys(rng, n, distinct, dist):
    # Key values spread over the whole int32 range (not 0..d-1), so the hash and probing get real work.
    pool = rng.integers(INT_MIN + 1, 2**31 - 1, size=distinct, dtype=np.int32)
    if dist == "uniform":
        idx = rng.integers(0, distinct, size=n)
    else:  # Zipf: key 0 is very common, the tail is rare
        idx = (rng.zipf(1.2, size=n) - 1) % distinct
    return pool[idx]


@pytest.mark.parametrize("dist", ["uniform", "zipf"])
@pytest.mark.parametrize("distinct", [1, 10, 1_000, 1_000_000])
@pytest.mark.parametrize("n", [1, 255, 256, 257, 1_000_000])  # sizes around the 256-thread block
def test_groupby_matches_pandas(n, distinct, dist):
    distinct = min(distinct, n)
    rng = np.random.default_rng(42)
    keys = make_keys(rng, n, distinct, dist)
    # Full int32 range: sums pass the int32 limit, and negative values check the sign handling.
    values = rng.integers(INT_MIN, 2**31 - 1, size=n, dtype=np.int32)

    df = pd.DataFrame({"k": keys, "v": values.astype(np.int64)})
    expected = df.groupby("k")["v"].agg(["sum", "count"])  # sorted by key

    k, s, c = group_by_sum_count(cp.asarray(keys), cp.asarray(values), expected_groups=distinct)
    order = cp.argsort(k)
    np.testing.assert_array_equal(cp.asnumpy(k[order]), expected.index.to_numpy())
    np.testing.assert_array_equal(cp.asnumpy(s[order]), expected["sum"].to_numpy())
    np.testing.assert_array_equal(cp.asnumpy(c[order]), expected["count"].to_numpy())


def test_overfull_table_raises():
    keys = cp.arange(10, dtype=cp.int32)  # 10 groups, but capacity is 2
    with pytest.raises(RuntimeError, match="full"):
        group_by_sum_count(keys, cp.ones(10, dtype=cp.int32), expected_groups=1)


def test_int_min_key_rejected():
    keys = cp.asarray([1, INT_MIN], dtype=cp.int32)
    with pytest.raises(ValueError):
        group_by_sum_count(keys, cp.ones(2, dtype=cp.int32), expected_groups=2)
