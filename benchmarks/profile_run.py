"""Workload for Nsight Systems and Nsight Compute: the published cases, with NVTX ranges.

Cases (100M int32 rows, pinned host memory, same data and seeds as the benchmarks):
  - filter, 10% selectivity: H2D copy, flag, scan, count sync, scatter, D2H copy
  - group-by, strategy block and global, 10 and 1M uniform groups: H2D, group-by, D2H
Each case runs 3 warmup iterations, then 3 profiled iterations. Every result is
checked against pandas. Kernel code is not changed: the filter phases call the same
kernels that engine.filter.filter_less_than(scan="own") calls, in the same order.

Run from the project root (normally under nsys or ncu, see docs/profiling.md):
  python -m benchmarks.profile_run
"""
from contextlib import contextmanager

import cupy as cp
import numpy as np
import pandas as pd
from cupy.cuda import nvtx

from benchmarks.bench_filter import pinned_empty, to_host_pinned
from benchmarks.bench_groupby import columns, sort_by_key
from benchmarks.datagen import group_keys, thresholds, uniform_int32
from engine import filter as gfilter
from engine.filter import filter_less_than
from engine.groupby import group_by_sum_count
from engine.scan import _scan

N = 100_000_000
WARMUP, PROFILED = 3, 3
SELECTIVITY = 10
GROUP_CASES = [(st, g) for g in (10, 1_000_000) for st in ("block", "global")]


@contextmanager
def rng(name):
    """One NVTX range. The GPU is synchronized at the end, so the range covers its kernels."""
    nvtx.RangePush(name)
    try:
        yield
        cp.cuda.Device().synchronize()
    finally:
        nvtx.RangePop()


def filter_phases(x_pin, t, out_pin, tag):
    """filter_less_than(scan="own") split into its phases, one NVTX range each."""
    with rng(f"{tag} H2D"):
        xd = cp.asarray(x_pin)
    n = xd.size
    grid = ((n + gfilter.THREADS - 1) // gfilter.THREADS,)
    with rng(f"{tag} flag"):
        flags = cp.empty(n, dtype=cp.int32)
        gfilter._flag(grid, (gfilter.THREADS,), (xd, flags, cp.int32(t), cp.int32(n)))
    with rng(f"{tag} scan"):
        pos = _scan(flags)
    with rng(f"{tag} count sync"):
        count = int(pos[-1] + flags[-1])  # the same device-to-host read as engine/filter.py
    with rng(f"{tag} scatter"):
        out = cp.empty(count, dtype=cp.int32)
        gfilter._scatter(grid, (gfilter.THREADS,), (xd, flags, pos, out, cp.int32(0), cp.int32(n)))
    with rng(f"{tag} D2H"):
        return to_host_pinned(out, out_pin)


def check(name, ok):
    if not ok:
        raise AssertionError(f"{name} result differs from pandas")


def run_filter():
    x = uniform_int32(N)
    t = thresholds(x)[SELECTIVITY]
    s = pd.Series(x, copy=False)
    expected = s[s < t].to_numpy()
    x_pin = pinned_empty(N)
    x_pin[:] = x
    out_pin = pinned_empty(N)
    # The phase split must give the same answer as the engine function itself.
    check("filter_less_than", np.array_equal(cp.asnumpy(filter_less_than(cp.asarray(x_pin), t)),
                                             expected))
    for i in range(WARMUP + PROFILED):
        label = "warmup" if i < WARMUP else f"iter{i - WARMUP + 1}"
        with rng(f"filter {label}"):
            got = filter_phases(x_pin, t, out_pin, "filter")
        check(f"filter {label}", np.array_equal(got, expected))
    del x_pin, out_pin


def run_groupby():
    v = uniform_int32(N, seed=7)
    v_pin = pinned_empty(N)
    v_pin[:] = v
    k_pin = pinned_empty(N)
    for groups in sorted({g for _, g in GROUP_CASES}):
        k = group_keys(N, groups, "uniform")
        k_pin[:] = k
        df = pd.DataFrame({"k": k, "v": v}, copy=False)
        r = df.groupby("k", sort=False)["v"].agg(["sum", "count"])  # same query as the benchmark
        expected = sort_by_key(*columns("pandas", r))
        true_groups = expected[0].size
        outs = [pinned_empty(true_groups, d) for d in (np.int32, np.int64, np.int64)]
        for st in [s for s, g in GROUP_CASES if g == groups]:
            tag = f"groupby {st} {groups}"
            for i in range(WARMUP + PROFILED):
                label = "warmup" if i < WARMUP else f"iter{i - WARMUP + 1}"
                with rng(f"{tag} {label}"):
                    with rng(f"{tag} H2D"):
                        kd, vd = cp.asarray(k_pin), cp.asarray(v_pin)
                    with rng(f"{tag} kernels"):
                        res = group_by_sum_count(kd, vd, expected_groups=true_groups, strategy=st)
                    with rng(f"{tag} D2H"):
                        got = tuple(to_host_pinned(d, o) for d, o in zip(res, outs))
                got = sort_by_key(*(a.copy() for a in got))
                check(f"{tag} {label}", all(np.array_equal(a, b) for a, b in zip(got, expected)))
                del kd, vd, res
    del v_pin, k_pin


def main():
    print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    run_filter()
    cp.get_default_memory_pool().free_all_blocks()
    run_groupby()
    print("all results match pandas")


if __name__ == "__main__":
    main()
