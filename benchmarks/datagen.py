"""Seeded synthetic columns for the benchmarks. Made in memory; nothing goes to disk."""
import numpy as np

SIZES = [1_000_000, 10_000_000, 50_000_000, 100_000_000, 200_000_000]
SELECTIVITIES = [1, 10, 50, 90]
VALUE_RANGE = 1_000_000_000  # wide range, so few ties and the quantile hits the target closely


def uniform_int32(n, seed=42):
    """n uniform int32 values in [0, VALUE_RANGE)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, VALUE_RANGE, size=n, dtype=np.int32)


GROUP_SIZES = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
GROUP_COUNTS = [10, 1_000, 1_000_000]


def group_keys(n, groups, dist, seed=42):
    """n int32 keys drawn from `groups` distinct values in [0, 2^31 - 1), uniform or Zipf (a=1.2)."""
    rng = np.random.default_rng(seed)
    pool = rng.choice(2**31 - 1, size=groups, replace=False).astype(np.int32)  # never INT_MIN
    if dist == "uniform":
        idx = rng.integers(0, groups, size=n)
    else:  # Zipf: one key is very common, the tail is rare; the tail folds back into range
        idx = (rng.zipf(1.2, size=n) - 1) % groups
    return pool[idx]


def thresholds(x, pcts=SELECTIVITIES):
    """{pct: t} so that about pct% of x is below t."""
    qs = np.quantile(x, [p / 100 for p in pcts])  # one sort for all four
    return {p: int(q) for p, q in zip(pcts, qs)}
