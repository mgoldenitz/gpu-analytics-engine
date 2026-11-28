"""TPC-H data for Q1: generate lineitem, save the DuckDB reference answer, encode the columns.

Step A: DuckDB's tpch extension writes lineitem into a file-backed database
        C:\\data\\gpu-analytics-engine\\tpch\\sf<N>.duckdb, then only the 7 Q1 columns go to
        C:\\data\\gpu-analytics-engine\\tpch\\sf<N>\\lineitem_q1.parquet.
Step B: the standard Q1 text runs in DuckDB on the full lineitem table. Its 4 rows are the
        ground truth (DuckDB's own CSV writer, so the decimals are exact text).
Step C: encode() turns the parquet columns into exact integers, once, before any timing.

Run from the project root:  python -m benchmarks.tpch_data
"""
import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(r"C:\data\gpu-analytics-engine\tpch")
RESULTS = Path(__file__).parent / "results"
SCALES = ["0.01", "1", "10"]
REPO_SCALES = ["0.01", "1"]  # the SF10 reference stays in C:\data
COLUMNS = "l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate"

# TPC-H Q1 as the specification writes it (delta = 90 days).
Q1_SQL = """
SELECT l_returnflag, l_linestatus,
       sum(l_quantity) AS sum_qty,
       sum(l_extendedprice) AS sum_base_price,
       sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
       avg(l_quantity) AS avg_qty,
       avg(l_extendedprice) AS avg_price,
       avg(l_discount) AS avg_disc,
       count(*) AS count_order
FROM lineitem
WHERE l_shipdate <= date '1998-12-01' - interval '90' day
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
"""
CUTOFF_SQL = "SELECT (date '1998-12-01' - interval '90' day)::DATE"

# Group id = 2 * returnflag index + linestatus index. Sorting by id is Q1's ORDER BY.
RETURNFLAGS = ["A", "N", "R"]
LINESTATUSES = ["F", "O"]
GROUPS = len(RETURNFLAGS) * len(LINESTATUSES)  # 6 slots; TPC-H data uses 4 of them


def db_path(sf):
    return ROOT / f"sf{sf}.duckdb"


def parquet_path(sf):
    return ROOT / f"sf{sf}" / "lineitem_q1.parquet"


def expected_path(sf):
    if sf in REPO_SCALES:
        return RESULTS / f"tpch_q1_expected_sf{sf}.csv"
    return ROOT / f"sf{sf}" / "tpch_q1_expected.csv"


def generate(sf):
    """Steps A and B for one scale factor. Returns the lineitem row count."""
    ROOT.mkdir(parents=True, exist_ok=True)
    parquet_path(sf).parent.mkdir(exist_ok=True)
    con = duckdb.connect(str(db_path(sf)))
    con.execute(f"SET temp_directory = '{(ROOT / 'tmp').as_posix()}'")
    con.execute("INSTALL tpch; LOAD tpch")
    if not con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'lineitem'"
                       ).fetchone()[0]:
        con.execute(f"CALL dbgen(sf = {sf})")
    rows = con.execute("SELECT count(*) FROM lineitem").fetchone()[0]
    con.execute(f"COPY (SELECT {COLUMNS} FROM lineitem) TO '{parquet_path(sf).as_posix()}' "
                "(FORMAT parquet)")
    con.execute(f"COPY ({Q1_SQL}) TO '{expected_path(sf).as_posix()}' (HEADER, DELIMITER ',')")
    con.close()
    return rows


def cutoff_days():
    """The Q1 cutoff as int32 days since 1970-01-01, computed by DuckDB."""
    con = duckdb.connect()
    day = con.execute(CUTOFF_SQL).fetchone()[0]
    days = con.execute(f"SELECT (DATE '{day}' - DATE '1970-01-01')::INTEGER").fetchone()[0]
    return day, days


def encode(sf):
    """Step C: the Q1 columns as exact integers (numpy arrays). Raises if a value does not fit.

    price in cents (int64); quantity, discount, tax in hundredths (int16);
    shipdate in days since 1970-01-01 (int32); returnflag and linestatus as one group id (uint8).
    """
    con = duckdb.connect()
    gid = " ".join(f"WHEN l_returnflag = '{r}' AND l_linestatus = '{s}' THEN {2 * i + j}"
                   for i, r in enumerate(RETURNFLAGS) for j, s in enumerate(LINESTATUSES))
    cols = con.execute(f"""
        SELECT CAST(l_extendedprice * 100 AS BIGINT) AS price,
               CAST(l_quantity * 100 AS INTEGER) AS qty,
               CAST(l_discount * 100 AS INTEGER) AS disc,
               CAST(l_tax * 100 AS INTEGER) AS tax,
               (l_shipdate - DATE '1970-01-01')::INTEGER AS ship,
               CASE {gid} ELSE 255 END AS gid
        FROM '{parquet_path(sf).as_posix()}'""").fetchnumpy()
    if (cols["gid"] == 255).any():
        raise ValueError("unknown returnflag or linestatus value")
    for name in ("qty", "disc", "tax"):  # int16 halves the bytes; check that no value is cut
        if cols[name].min() < 0 or cols[name].max() > np.iinfo(np.int16).max:
            raise ValueError(f"{name} does not fit in int16")
    return {"price": cols["price"].astype(np.int64), "qty": cols["qty"].astype(np.int16),
            "disc": cols["disc"].astype(np.int16), "tax": cols["tax"].astype(np.int16),
            "ship": cols["ship"].astype(np.int32), "gid": cols["gid"].astype(np.uint8)}


def sum_bounds(c):
    """Largest possible sum of each aggregate: every row in one group, every row at the maximum."""
    n = c["price"].size
    p, q, d, t = (int(c[k].max()) for k in ("price", "qty", "disc", "tax"))
    dmin = int(c["disc"].min())
    return {"count": n, "sum_qty": n * q, "sum_base_price": n * p,
            "sum_disc_price": n * p * (100 - dmin),
            "sum_charge": n * p * (100 - dmin) * (100 + t), "sum_disc": n * d}


def main(scales):
    for sf in scales:
        print(f"SF {sf}: lineitem rows = {generate(sf):,}", flush=True)
    day, days = cutoff_days()
    print(f"cutoff = {day} = {days} days since 1970-01-01")
    if "10" in scales:
        b = sum_bounds(encode("10"))
        for k, v in b.items():
            print(f"SF10 bound {k:15s} {v:>26,}  {'< 2^63' if v < 2**63 else 'OVER 2^63'}")
        if max(b.values()) >= 2**63:
            raise SystemExit("an aggregate can overflow int64")


if __name__ == "__main__":
    main(sys.argv[1:] or SCALES)
