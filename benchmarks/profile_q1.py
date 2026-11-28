"""Workload for Nsight Systems and Nsight Compute: TPC-H Q1 at SF10, the e2e_pinned path.

Each iteration: H2D copy of the 6 encoded columns from pinned memory, q1_sums (fill + q1_block),
D2H copy of the 288-byte result. 3 warmup and 3 profiled iterations, one NVTX range each.
Every result is checked against the DuckDB Q1 reference (benchmarks/bench_q1.check).

Run from the project root (normally under nsys or ncu, see docs/profiling.md):
  python -m benchmarks.profile_q1
"""
import csv

import cupy as cp
import numpy as np

from benchmarks import tpch_data
from benchmarks.bench_filter import pinned_empty, to_host_pinned
from benchmarks.bench_q1 import check, rows_of
from benchmarks.profile_run import rng
from engine.q1 import AGGS, GROUPS, q1_rows, q1_sums

SF = "10"
WARMUP, PROFILED = 3, 3


def main():
    _, cutoff = tpch_data.cutoff_days()
    c = tpch_data.encode(SF)
    with open(tpch_data.expected_path(SF), newline="") as fh:
        expected = list(csv.DictReader(fh))
    pin = {}
    for k, v in c.items():
        pin[k] = pinned_empty(v.size, v.dtype)
        pin[k][:] = v
    out_pin = pinned_empty(GROUPS * AGGS, np.uint64)
    for i in range(WARMUP + PROFILED):
        label = "warmup" if i < WARMUP else f"iter{i - WARMUP + 1}"
        with rng(f"q1 {label}"):
            with rng("q1 H2D"):
                d = {k: cp.asarray(v) for k, v in pin.items()}
            with rng("q1 kernel"):
                sums = q1_sums(**d, cutoff=cutoff)
            with rng("q1 D2H"):
                got = q1_rows(to_host_pinned(sums.ravel(), out_pin).reshape(GROUPS, AGGS))
        check(f"q1 {label}", rows_of(got), expected)
        del d, sums
    print("all Q1 results match the DuckDB reference")


if __name__ == "__main__":
    main()
