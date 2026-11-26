"""Time SELECT key, SUM(value), COUNT(*) GROUP BY key on CPU and GPU engines.

Writes benchmarks/results/groupby.csv.  Run from the project root:  python -m benchmarks.bench_groupby
"""
import csv
from pathlib import Path

import cupy as cp
import duckdb
import numpy as np
import pandas as pd
import polars as pl

from benchmarks.bench_filter import median_ms, pinned_empty, to_host_pinned
from benchmarks.datagen import GROUP_COUNTS, GROUP_SIZES, group_keys, uniform_int32
from engine.groupby import group_by_sum_count

OUT = Path(__file__).parent / "results" / "groupby.csv"
DISTS = ["uniform", "zipf"]
GPU_MODES = ["kernel_only", "e2e_pageable", "e2e_pinned"]
STRATEGIES = ["global", "block"]


def cpu_engines(k, v):
    """Each engine gets its frame or table once; only the group-by is timed. Sums are int64."""
    df = pd.DataFrame({"k": k, "v": v}, copy=False)
    pdf = pl.DataFrame({"k": k, "v": v})
    con = duckdb.connect()
    con.register("t", {"k": k, "v": v})  # DuckDB scans the numpy arrays in place
    return {
        # sort=False: the query has no ORDER BY, and Polars and DuckDB do not sort either.
        "pandas": lambda: df.groupby("k", sort=False)["v"].agg(["sum", "count"]),
        # Polars sums Int32 as Int32 and overflows, so cast first (SQL SUM gives BIGINT).
        "polars": lambda: pdf.group_by("k").agg(pl.col("v").cast(pl.Int64).sum().alias("s"),
                                                pl.len().alias("c")),
        "duckdb": lambda: con.execute(
            "SELECT k, SUM(v)::BIGINT AS s, COUNT(*) AS c FROM t GROUP BY k").fetchnumpy(),
    }


def columns(engine, r):
    """(keys, sums, counts) as numpy arrays, from each engine's result type."""
    if engine == "pandas":
        return r.index.to_numpy(), r["sum"].to_numpy(), r["count"].to_numpy()
    if engine == "polars":
        return r["k"].to_numpy(), r["s"].to_numpy(), r["c"].to_numpy()
    if engine == "duckdb":
        return r["k"], r["s"], r["c"]
    return tuple(a if isinstance(a, np.ndarray) else cp.asnumpy(a) for a in r)  # GPU


def sort_by_key(k, s, c):
    o = np.argsort(k)
    return k[o], s[o], c[o]


def check(engine, got, expected):
    if not all(np.array_equal(a, b) for a, b in zip(sort_by_key(*got), expected)):
        raise AssertionError(f"{engine} result differs from pandas")


def gpu_modes(k, v, kd, vd, k_pin, v_pin, groups, strategy):
    f = lambda a, b: group_by_sum_count(a, b, expected_groups=groups, strategy=strategy)
    outs = [pinned_empty(groups, t) for t in (np.int32, np.int64, np.int64)]
    return {
        "kernel_only": lambda: f(kd, vd),
        "e2e_pageable": lambda: tuple(cp.asnumpy(a) for a in f(cp.asarray(k), cp.asarray(v))),
        "e2e_pinned": lambda: tuple(to_host_pinned(d, o) for d, o in
                                    zip(f(cp.asarray(k_pin), cp.asarray(v_pin)), outs)),
    }


def bench_size(n, rows):
    v = uniform_int32(n, seed=7)
    vd = cp.asarray(v)
    v_pin = pinned_empty(n)  # inputs already in pinned memory, as a pinned data loader would give them
    v_pin[:] = v
    k_pin = pinned_empty(n)
    for groups in sorted({min(g, n) for g in GROUP_COUNTS}):
        for dist in DISTS:
            k = group_keys(n, groups, dist)
            k_pin[:] = k
            kd = cp.asarray(k)
            cpu = cpu_engines(k, v)
            expected = sort_by_key(*columns("pandas", cpu["pandas"]()))
            true_groups = expected[0].size  # Zipf leaves some keys unused
            for name, f in cpu.items():
                check(name, columns(name, f()), expected)
                rows.append((name, "-", n, groups, dist, "cpu", median_ms(f)))
            for st in STRATEGIES:
                for mode, f in gpu_modes(k, v, kd, vd, k_pin, v_pin, true_groups, st).items():
                    check(f"hash_groupby {st} {mode}", columns("gpu", f()), expected)
                    rows.append(("hash_groupby", st, n, groups, dist, mode, median_ms(f)))
            print(f"n={n:,} groups={groups:,} {dist} ({true_groups:,} used) done", flush=True)
            del kd, cpu
    del vd, v_pin, k_pin
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def print_rows(rows):
    d = {r[:-1]: r[-1] for r in rows}
    cols = [("pandas", "-", "cpu"), ("polars", "-", "cpu"), ("duckdb", "-", "cpu")] + \
           [("hash_groupby", st, m) for st in STRATEGIES for m in GPU_MODES]
    for n in (10_000_000, 100_000_000):
        print(f"\nn = {n:,}  (median ms)")
        print(f"{'groups':>9} {'dist':>8}" +
              "".join(f"{(e if st == '-' else st)[:6] + '/' + m[:8]:>16}" for e, st, m in cols))
        for g in sorted({k[3] for k in d if k[2] == n}):
            for dist in DISTS:
                print(f"{g:>9,} {dist:>8}" + "".join(
                    f"{d.get((e, st, n, g, dist, m), float('nan')):>16.2f}" for e, st, m in cols))


def main():
    rows = []
    for n in GROUP_SIZES:
        bench_size(n, rows)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["engine", "strategy", "n", "groups", "dist", "mode", "median_ms"])
        w.writerows((*r[:-1], f"{r[-1]:.3f}") for r in rows)
    print_rows(rows)


if __name__ == "__main__":
    main()
