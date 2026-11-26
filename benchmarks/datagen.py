"""Seeded synthetic columns for the benchmarks. Made in memory; nothing goes to disk."""
import numpy as np

SIZES = [1_000_000, 10_000_000, 50_000_000, 100_000_000, 200_000_000]
SELECTIVITIES = [1, 10, 50, 90]
VALUE_RANGE = 1_000_000_000  # wide range, so few ties and the quantile hits the target closely


def uniform_int32(n, seed=42):
    """n uniform int32 values in [0, VALUE_RANGE)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, VALUE_RANGE, size=n, dtype=np.int32)


def thresholds(x, pcts=SELECTIVITIES):
    """{pct: t} so that about pct% of x is below t."""
    qs = np.quantile(x, [p / 100 for p in pcts])  # one sort for all four
    return {p: int(q) for p, q in zip(pcts, qs)}
