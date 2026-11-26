"""Time `value < threshold` on CPU and GPU engines; write benchmarks/results/filter.csv.

Run from the project root:  python -m benchmarks.bench_filter
"""
import csv
import statistics
import time
from pathlib import Path

import cupy as cp
import duckdb
import numpy as np
import pandas as pd
import polars as pl

from benchmarks.datagen import SIZES, thresholds, uniform_int32
from benchmarks.summary import print_summary
from engine.filter import filter_less_than

WARMUP, RUNS = 3, 10
GPU_BYTES = 8 * 1024**3
OUT = Path(__file__).parent / "results" / "filter.csv"

# GPU engines: (device array, threshold) -> device array of the passing values.
# bytes_per_row is a high estimate of peak GPU memory per input row, for the 8 GB check.
GPU_ENGINES = {
    "cupy_builtin": (lambda xd, t: xd[xd < t], 24),  # x, bool mask, int64 indices, out
    "filter_cub": (lambda xd, t: filter_less_than(xd, t, scan="cub"), 24),  # x, flags, cumsum, pos, out
    "filter_own": (lambda xd, t: filter_less_than(xd, t, scan="own"), 20),  # x, flags, pos, out
}


def median_ms(fn):
    """3 warmup runs, then the median of 10. The GPU is synchronized around each run."""
    dev = cp.cuda.Device()
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(RUNS):
        dev.synchronize()
        t0 = time.perf_counter()
        fn()
        dev.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def cpu_engines(x):
    """Each engine gets its frame or table once; only the filter is timed."""
    s = pd.Series(x, copy=False)
    df = pl.DataFrame({"v": x})
    con = duckdb.connect()
    con.register("t", {"v": x})  # DuckDB scans the numpy array in place
    return {
        "pandas": lambda t: s[s < t].to_numpy(),
        "polars": lambda t: df.filter(pl.col("v") < t)["v"].to_numpy(),
        "duckdb": lambda t: con.execute("SELECT v FROM t WHERE v < ?", [t]).fetchnumpy()["v"],
    }


def check(engine, got, expected):
    if not np.array_equal(got, expected):
        raise AssertionError(f"{engine} result differs from pandas")


def bench_size(n, rows, skipped):
    x = uniform_int32(n)
    ts = thresholds(x)
    cpu = cpu_engines(x)
    xd = cp.asarray(x)
    for pct, t in ts.items():
        expected = cpu["pandas"](t)
        for name, f in cpu.items():
            check(name, f(t), expected)
            rows.append((name, n, pct, "cpu", median_ms(lambda: f(t))))
        for name, (f, bytes_per_row) in GPU_ENGINES.items():
            if n * bytes_per_row > GPU_BYTES:
                skipped.append(f"{name} n={n:,}: needs about {n * bytes_per_row / 1e9:.1f} GB")
                continue
            check(name, cp.asnumpy(f(xd, t)), expected)
            rows.append((name, n, pct, "kernel", median_ms(lambda: f(xd, t))))
            rows.append((name, n, pct, "end_to_end",
                         median_ms(lambda: cp.asnumpy(f(cp.asarray(x), t)))))
        print(f"n={n:,} sel={pct}% done", flush=True)
    del xd
    cp.get_default_memory_pool().free_all_blocks()


def main():
    rows, skipped = [], []
    for n in SIZES:
        bench_size(n, rows, skipped)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["engine", "n", "selectivity_pct", "mode", "median_ms"])
        w.writerows((e, n, p, m, f"{ms:.3f}") for e, n, p, m, ms in rows)
    for s in skipped:
        print("skipped:", s)
    print_summary(OUT)


if __name__ == "__main__":
    main()
