"""Make docs/figures/tpch_q1.png from results/tpch_q1.csv, in the style of plot.py.

Two panels (SF1, SF10) of horizontal bars, one bar per engine, fastest at the top.
CPU engines use the narrow columns, like q1_v3; the wide CPU times stay in the README text.

Run from the project root:  python -m benchmarks.plot_q1
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from benchmarks.plot import COLOR, LABEL, load, plain_fmt, save

# (engine, mode, label, colour, hatched). Hatched = kernel only (no copies), as in plot.py
# where texture, not a second hue, separates two results of the same engine.
# q1_v2 gets a lighter green so the two solid end-to-end GPU bars differ by more than their label.
GPU_LIGHT = "#b9dd85"
BARS = [
    ("gpu_q1_v3", "kernel_only", "q1_v3 kernel only", COLOR["gpu"], True),
    ("gpu_q1_v3", "e2e_pinned", "q1_v3 end to end", COLOR["gpu"], False),
    ("gpu_q1_v2", "e2e_pinned", "q1_v2 end to end (wide)", GPU_LIGHT, False),
    ("polars_narrow", "cpu", LABEL["polars"], COLOR["polars"], False),
    ("duckdb_narrow", "cpu", LABEL["duckdb"], COLOR["duckdb"], False),
    ("pandas_narrow", "cpu", LABEL["pandas"], COLOR["pandas"], False),
]


def ms_fmt(t):
    return f"{t:,.1f} ms"  # one decimal, as in the README


def tpch_q1(q):
    sfs = sorted({int(k[0]) for k in q})
    rows = {int(k[0]): int(k[1]) for k in q}
    get = lambda sf, e, m: q[next(k for k in q if int(k[0]) == sf and k[2] == e and k[3] == m)]
    # one engine order for both panels: fastest at the top, by the largest scale factor
    order = sorted(BARS, key=lambda b: get(sfs[-1], b[0], b[1]))
    fig, axes = plt.subplots(1, len(sfs), figsize=(9, 3.9), sharey=True)
    hi = max(get(sf, b[0], b[1]) for sf in sfs for b in BARS)
    lo = min(get(sf, b[0], b[1]) for sf in sfs for b in BARS)
    for ax, sf in zip(axes, sfs):
        ys = range(len(order))
        ts = [get(sf, b[0], b[1]) for b in order]
        for y, t, b in zip(ys, ts, order):
            ax.barh(y, t, height=0.62, color=b[3], hatch="//" if b[4] else None,
                    edgecolor="white", linewidth=1.5)
            ax.annotate(ms_fmt(t), (t, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(lo / 2, hi * 6)  # room for the value labels
        ax.xaxis.set_major_formatter(FuncFormatter(plain_fmt))
        ax.minorticks_off()
        ax.grid(axis="y", visible=False)
        ax.set_title(f"SF{sf} ({rows[sf] / 1e6:.1f}M rows)", fontsize=9.5, loc="left")
        ax.set_xlabel("Median time, ms (log scale)")
    axes[0].set_yticks(list(range(len(order))), [b[2] for b in order])
    axes[0].invert_yaxis()  # fastest at the top
    axes[0].tick_params(axis="y", length=0)
    save(fig, "tpch_q1.png", "TPC-H Q1, narrow columns: median time by engine")


if __name__ == "__main__":
    tpch_q1(load("tpch_q1.csv"))
