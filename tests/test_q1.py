"""Check the fused TPC-H Q1 kernel against DuckDB's own Q1 answer, and two edge cases against pandas."""
import csv

import cupy as cp
import numpy as np
import pandas as pd
import pytest

from benchmarks import tpch_data
from engine.q1 import AGGS, q1

# Each integer column of the DuckDB CSV and how many decimals it carries.
SCALED = {"sum_qty": 2, "sum_base_price": 2, "sum_disc_price": 4, "sum_charge": 6}
AVGS = ["avg_qty", "avg_price", "avg_disc"]


def to_device(cols):
    return {k: cp.asarray(v) for k, v in cols.items()}


def as_int(text, decimals):
    """'123.45' with 2 decimals -> 12345, with no float on the way."""
    whole, _, frac = text.partition(".")
    assert len(frac) == decimals, f"{text} does not have {decimals} decimals"
    return int(whole + frac)


@pytest.mark.parametrize("sf", tpch_data.REPO_SCALES)
def test_q1_matches_duckdb(sf):
    if not tpch_data.parquet_path(sf).exists():
        pytest.skip(f"run python -m benchmarks.tpch_data {sf} first")
    with open(tpch_data.expected_path(sf), newline="") as fh:
        expected = list(csv.DictReader(fh))
    _, cutoff = tpch_data.cutoff_days()
    got = q1(**to_device(tpch_data.encode(sf)), cutoff=cutoff)

    assert [(r["l_returnflag"], r["l_linestatus"]) for r in got] == \
           [(e["l_returnflag"], e["l_linestatus"]) for e in expected]
    for r, e in zip(got, expected):
        assert r["count_order"] == int(e["count_order"])
        for name, decimals in SCALED.items():  # exact integer match
            assert r[name] == as_int(e[name], decimals), name
        for name in AVGS:  # equal when rounded to the decimals DuckDB prints
            places = len(e[name].partition(".")[2])
            assert round(r[name], places) == round(float(e[name]), places), name


def small_table(gid, ship):
    rng = np.random.default_rng(7)
    n = gid.size
    return {"price": rng.integers(90_000, 10_500_000, n, dtype=np.int64),
            "qty": rng.integers(100, 5_001, n).astype(np.int16),
            "disc": rng.integers(0, 11, n).astype(np.int16),
            "tax": rng.integers(0, 9, n).astype(np.int16),
            "ship": ship.astype(np.int32), "gid": gid.astype(np.uint8)}


def test_q1_no_rows_pass():
    cols = small_table(np.zeros(1000), np.full(1000, 10_000))
    assert q1(**to_device(cols), cutoff=9_999) == []


@pytest.mark.parametrize("n", [1, 4095, 4096, 4097, 100_000])  # sizes around one block's rows
def test_q1_one_group(n):
    rng = np.random.default_rng(3)
    ship = rng.integers(9_000, 11_000, n)
    cols = small_table(np.full(n, 3), ship)  # group 3 = N,O
    cutoff = 10_471
    got = q1(**to_device(cols), cutoff=cutoff)

    df = pd.DataFrame(cols)[cols["ship"] <= cutoff].astype("int64")
    if df.empty:
        assert got == []
        return
    disc_price = df["price"] * (100 - df["disc"])
    assert len(got) == 1
    r = got[0]
    assert (r["l_returnflag"], r["l_linestatus"]) == ("N", "O")
    assert r["count_order"] == len(df)
    assert r["sum_qty"] == df["qty"].sum()
    assert r["sum_base_price"] == df["price"].sum()
    assert r["sum_disc_price"] == disc_price.sum()
    assert r["sum_charge"] == (disc_price * (100 + df["tax"])).sum()
    assert r["avg_disc"] == pytest.approx(df["disc"].mean() / 100)
    assert AGGS == 6
