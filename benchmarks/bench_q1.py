"""Time TPC-H Q1 at SF1 and SF10 on pandas, Polars, DuckDB and the fused GPU kernel.

All engines get the same encoded integer columns (benchmarks/tpch_data.encode), built once
before any timing. Each engine computes the products, sums and averages inside its timed
query. Every result is checked against DuckDB's Q1 answer on the original table first.

Writes benchmarks/results/tpch_q1.csv.  Run from the project root:  python -m benchmarks.bench_q1
"""
import csv
import math
import os
from pathlib import Path

import cupy as cp
import duckdb
import numpy as np
import pandas as pd
import polars as pl

from benchmarks import tpch_data
from benchmarks.bench_filter import median_ms, pinned_empty, warm_gpu, to_host_pinned
from engine.q1 import AGGS, GROUPS, NAMES, q1_rows, q1_sums

OUT = Path(__file__).parent / "results" / "tpch_q1.csv"
BENCH_SCALES = ["1", "10"]
SUMS = ["count_order", "sum_qty", "sum_base_price", "sum_disc_price", "sum_charge"]
SCALED = {"sum_qty": 2, "sum_base_price": 2, "sum_disc_price": 4, "sum_charge": 6}
AVGS = ["avg_qty", "avg_price", "avg_disc"]

DUCKDB_SQL = """
SELECT gid, count(*) AS count_order, sum(qty)::BIGINT AS sum_qty,
       sum(price)::BIGINT AS sum_base_price,
       sum(price * (100 - disc))::BIGINT AS sum_disc_price,
       sum(price * (100 - disc) * (100 + tax))::BIGINT AS sum_charge,
       avg(qty) / 100 AS avg_qty, avg(price) / 100 AS avg_price, avg(disc) / 100 AS avg_disc
FROM t WHERE ship <= ? GROUP BY gid ORDER BY gid
"""


def cpu_engines(c, cutoff):
    """Each engine gets its frame or table once; only the query is timed. Sums are int64."""
    df = pd.DataFrame(c, copy=False)
    pdf = pl.DataFrame(c)
    con = duckdb.connect()
    con.register("t", c)  # DuckDB scans the numpy arrays in place
    i64 = lambda name: pl.col(name).cast(pl.Int64)
    dp = i64("price") * (100 - i64("disc"))
    lazy = (pdf.lazy().filter(pl.col("ship") <= cutoff).group_by("gid").agg(
        pl.len().alias("count_order"), i64("qty").sum().alias("sum_qty"),
        i64("price").sum().alias("sum_base_price"), dp.sum().alias("sum_disc_price"),
        (dp * (100 + i64("tax"))).sum().alias("sum_charge"),
        (pl.col("qty").mean() / 100).alias("avg_qty"),
        (pl.col("price").mean() / 100).alias("avg_price"),
        (pl.col("disc").mean() / 100).alias("avg_disc")).sort("gid"))

    def pandas_q1():
        m = df[df["ship"] <= cutoff]
        dp = m["price"] * (100 - m["disc"].astype(np.int64))
        t = pd.DataFrame({"gid": m["gid"], "qty": m["qty"].astype(np.int64), "price": m["price"],
                          "dp": dp, "ch": dp * (100 + m["tax"].astype(np.int64)),
                          "disc": m["disc"].astype(np.int64)})
        r = t.groupby("gid").agg(count_order=("qty", "size"), sum_qty=("qty", "sum"),
                                 sum_base_price=("price", "sum"), sum_disc_price=("dp", "sum"),
                                 sum_charge=("ch", "sum"), avg_qty=("qty", "mean"),
                                 avg_price=("price", "mean"), avg_disc=("disc", "mean"))
        r[AVGS] /= 100
        return r.reset_index()

    threads = {"pandas": 1, "polars": pl.thread_pool_size(),
               "duckdb": int(con.execute("SELECT current_setting('threads')").fetchone()[0])}
    engines = {"pandas": pandas_q1, "polars": lazy.collect,
               "duckdb": lambda: con.execute(DUCKDB_SQL, [cutoff]).fetchdf()}
    return engines, threads


def rows_of(r):
    """Any engine's result as [(returnflag, linestatus, 5 sums, 3 averages)] in gid order."""
    if isinstance(r, list):  # GPU: q1_rows output
        return [(x["l_returnflag"], x["l_linestatus"], *(x[k] for k in SUMS),
                 *(x[k] for k in AVGS)) for x in r]
    if isinstance(r, pl.DataFrame):
        r = r.to_pandas()
    return [(*NAMES[int(x.gid)], *(int(getattr(x, k)) for k in SUMS),
             *(float(getattr(x, k)) for k in AVGS)) for x in r.itertuples()]


def as_int(text, decimals):
    whole, _, frac = text.partition(".")
    assert len(frac) == decimals
    return int(whole + frac)


def check(engine, got, expected):
    """Sums and counts exact; averages within 1e-15 relative.

    The CPU engines take mean(int)/100 and land 1 ulp from DuckDB's DECIMAL avg in some
    groups (seen at SF1); the GPU's sum/100/count is bit-exact (tests/test_q1.py).
    """
    ok = len(got) == len(expected)
    for g, e in zip(got, expected):
        want = (e["l_returnflag"], e["l_linestatus"], int(e["count_order"]),
                *(as_int(e[k], d) for k, d in SCALED.items()))
        ok &= tuple(g[:7]) == want
        ok &= all(math.isclose(v, float(e[k]), rel_tol=1e-15) for v, k in zip(g[7:], AVGS))
    if not ok:
        raise AssertionError(f"{engine} result differs from the DuckDB Q1 reference")


def gpu_modes(c, cutoff):
    d = {k: cp.asarray(v) for k, v in c.items()}
    pin = {}
    for k, v in c.items():  # inputs already in pinned memory, as in bench_groupby
        pin[k] = pinned_empty(v.size, v.dtype)
        pin[k][:] = v
    out_d = cp.zeros(GROUPS * AGGS, dtype=cp.uint64)
    out_pin = pinned_empty(GROUPS * AGGS, np.uint64)
    modes = {
        "kernel_only": lambda: q1_sums(**d, cutoff=cutoff, out=out_d),
        "e2e_pinned": lambda: q1_rows(to_host_pinned(
            q1_sums(**{k: cp.asarray(v) for k, v in pin.items()}, cutoff=cutoff).ravel(),
            out_pin).reshape(GROUPS, AGGS)),
    }
    return modes, d, pin


def event_ms(fn, runs=10):
    """Median GPU time of fn by CUDA events: a cross-check that the harness time is the kernel."""
    a, b = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(runs):
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(cp.cuda.get_elapsed_time(a, b))
    return float(np.median(times))


def bench_sf(sf, cutoff, rows, threads_seen):
    c = tpch_data.encode(sf)
    with open(tpch_data.expected_path(sf), newline="") as fh:
        expected = list(csv.DictReader(fh))
    n = c["price"].size
    engines, threads = cpu_engines(c, cutoff)
    threads_seen.update(threads)
    for name, f in engines.items():
        check(f"{name} SF{sf}", rows_of(f()), expected)
        rows.append((sf, n, name, "cpu", threads[name], median_ms(f)))
        print(f"SF{sf} {name}: {rows[-1][-1]:.2f} ms", flush=True)
    del engines
    modes, d, pin = gpu_modes(c, cutoff)
    for mode, f in modes.items():
        r = f()
        check(f"gpu {mode} SF{sf}", rows_of(r if mode != "kernel_only" else
                                            q1_rows(cp.asnumpy(r))), expected)
        warm_gpu(f)
        rows.append((sf, n, "gpu_q1", mode, "-", median_ms(f)))
        print(f"SF{sf} gpu {mode}: {rows[-1][-1]:.2f} ms", flush=True)
        if mode == "kernel_only":  # a second guard against a slow GPU clock state
            ev = event_ms(f)
            print(f"SF{sf} gpu kernel by CUDA events: {ev:.2f} ms", flush=True)
            if abs(rows[-1][-1] - ev) > 0.10 * ev:
                raise SystemExit("harness and CUDA-event kernel times differ by more than 10%")
    del modes, d, pin, c
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def main():
    _, cutoff = tpch_data.cutoff_days()
    rows, threads = [], {}
    for sf in BENCH_SCALES:
        bench_sf(sf, cutoff, rows, threads)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sf", "rows", "engine", "mode", "threads", "median_ms"])
        w.writerows((*r[:-1], f"{r[-1]:.3f}") for r in rows)
    print("wrote", OUT, "| threads:", threads, "| logical CPUs:", os.cpu_count())


if __name__ == "__main__":
    main()
