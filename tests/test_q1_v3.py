"""Check q1_v3 (narrow columns, vector loads) against DuckDB's Q1 answer and against pandas."""
import csv

import cupy as cp
import numpy as np
import pandas as pd
import pytest

from benchmarks import tpch_data
from engine.q1 import q1
from engine.q1_narrow import DTYPES, VEC

SCALED = {"sum_qty": 2, "sum_base_price": 2, "sum_disc_price": 4, "sum_charge": 6}
AVGS = ["avg_qty", "avg_price", "avg_disc"]


def to_device(cols):
    return {k: cp.asarray(v) for k, v in cols.items()}


def as_int(text, decimals):
    whole, _, frac = text.partition(".")
    assert len(frac) == decimals
    return int(whole + frac)


@pytest.mark.parametrize("sf", tpch_data.REPO_SCALES)
def test_q1_v3_matches_duckdb(sf):
    if not tpch_data.parquet_path(sf).exists():
        pytest.skip(f"run python -m benchmarks.tpch_data {sf} first")
    with open(tpch_data.expected_path(sf), newline="") as fh:
        expected = list(csv.DictReader(fh))
    _, cutoff = tpch_data.cutoff_days()
    c = tpch_data.narrow(tpch_data.encode(sf))
    got = q1(**to_device(c), cutoff=tpch_data.narrow_cutoff(cutoff), kernel="v3")

    assert [(r["l_returnflag"], r["l_linestatus"]) for r in got] == \
           [(e["l_returnflag"], e["l_linestatus"]) for e in expected]
    for r, e in zip(got, expected):
        assert r["count_order"] == int(e["count_order"])
        for name, decimals in SCALED.items():
            assert r[name] == as_int(e[name], decimals), name
        for name in AVGS:
            places = len(e[name].partition(".")[2])
            assert round(r[name], places) == round(float(e[name]), places), name


def small_table(n, seed=7):
    """Narrow columns in the measured TPC-H ranges, all 6 groups mixed."""
    rng = np.random.default_rng(seed)
    return {"price": rng.integers(90_000, 10_500_000, n).astype(np.int32),
            "qty": rng.integers(1, 51, n).astype(np.uint8),
            "disc": rng.integers(0, 11, n).astype(np.uint8),
            "tax": rng.integers(0, 9, n).astype(np.uint8),
            "ship": rng.integers(0, 2526, n).astype(np.uint16),
            "gid": rng.integers(0, 6, n).astype(np.uint8)}


def pandas_q1(cols, cutoff):
    df = pd.DataFrame(cols)[cols["ship"] <= cutoff].astype("int64")
    df["dp"] = df["price"] * (100 - df["disc"])
    df["ch"] = df["dp"] * (100 + df["tax"])
    r = df.groupby("gid").agg(count_order=("qty", "size"), sum_qty=("qty", "sum"),
                              sum_base_price=("price", "sum"), sum_disc_price=("dp", "sum"),
                              sum_charge=("ch", "sum"), disc=("disc", "sum"))
    return r


def test_q1_v3_no_rows_pass():
    cols = small_table(1000)
    cols["ship"][:] = 2000
    assert q1(**to_device(cols), cutoff=1999, kernel="v3") == []


# n not a multiple of VEC, and sizes around one block's worth of vector loads
@pytest.mark.parametrize("n", [1, 2, 3, VEC, VEC + 1, 1023, 1024, 1025, 100_003])
def test_q1_v3_six_groups(n):
    assert VEC == 4
    cols = small_table(n, seed=n)
    cutoff = 2435
    got = q1(**to_device(cols), cutoff=cutoff, kernel="v3")
    want = pandas_q1(cols, cutoff)
    assert [r["count_order"] for r in got] == want["count_order"].tolist()
    for r, (g, w) in zip(got, want.iterrows()):
        assert r["sum_qty"] == w["sum_qty"] * 100  # v3 sums whole units, reports hundredths
        assert r["sum_base_price"] == w["sum_base_price"]
        assert r["sum_disc_price"] == w["sum_disc_price"]
        assert r["sum_charge"] == w["sum_charge"]
        assert r["avg_disc"] == w["disc"] / 100 / w["count_order"]


def test_q1_v3_only_tail_rows_pass():
    """Only the last n % VEC rows pass the filter, so only the leftover loop adds."""
    n = 4 * 1000 + 3
    cols = small_table(n)
    cols["ship"][:] = 2500
    cols["ship"][-3:] = 0
    got = q1(**to_device(cols), cutoff=10, kernel="v3")
    assert sum(r["count_order"] for r in got) == 3


def test_q1_v3_rejects_wide_dtypes():
    cols = small_table(8)
    cols["price"] = cols["price"].astype(np.int64)
    with pytest.raises(TypeError):
        q1(**to_device(cols), cutoff=10, kernel="v3")
    assert DTYPES["price"] == np.int32


def test_q1_v3_rejects_misaligned_input():
    d = to_device(small_table(9))
    d = {k: v[1:] for k, v in d.items()}  # one element in: not 16-byte aligned for int4
    with pytest.raises(ValueError, match="aligned"):
        q1(**d, cutoff=10, kernel="v3")


@pytest.mark.parametrize("col, value", [("price", 2**31), ("qty", 150), ("disc", 300),
                                        ("ship", tpch_data.SHIP_BASE - 1),
                                        ("ship", tpch_data.SHIP_BASE + 65536)])
def test_narrow_raises_out_of_range(col, value):
    wide = {"price": np.array([100, 200], np.int64), "qty": np.array([100, 200], np.int16),
            "disc": np.array([5, 6], np.int16), "tax": np.array([1, 2], np.int16),
            "ship": np.array([tpch_data.SHIP_BASE, tpch_data.SHIP_BASE + 9], np.int32),
            "gid": np.array([0, 5], np.uint8)}
    wide[col] = wide[col].astype(np.int64)
    wide[col][1] = value
    with pytest.raises(ValueError):
        tpch_data.narrow(wide)


def test_narrow_keeps_values():
    wide = {"price": np.array([90_091, 10_494_950], np.int64),
            "qty": np.array([100, 5000], np.int16), "disc": np.array([0, 10], np.int16),
            "tax": np.array([0, 8], np.int16),
            "ship": np.array([tpch_data.SHIP_BASE, tpch_data.SHIP_BASE + 2525], np.int32),
            "gid": np.array([0, 5], np.uint8)}
    n = tpch_data.narrow(wide)
    assert {k: v.dtype for k, v in n.items()} == {k: np.dtype(t) for k, t in DTYPES.items()}
    assert n["qty"].tolist() == [1, 50] and n["ship"].tolist() == [0, 2525]
    assert n["price"].tolist() == [90_091, 10_494_950]
