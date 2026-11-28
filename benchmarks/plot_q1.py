"""Make docs/figures/tpch_q1.png from results/tpch_q1.csv, in the style of plot.py.

Run from the project root:  python -m benchmarks.plot_q1
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from benchmarks.plot import LABEL, line, load, plain_fmt, save


def tpch_q1(q):
    sfs = sorted({int(k[0]) for k in q})
    rows = {int(k[0]): int(k[1]) for k in q}
    ms = lambda e, m: [q[next(k for k in q if int(k[0]) == sf and k[2] == e and k[3] == m)]
                       for sf in sfs]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ends = [line(ax, sfs, ms(e, "cpu"), e, LABEL[e]) for e in ("pandas", "polars", "duckdb")]
    ends.append(line(ax, sfs, ms("gpu_q1", "e2e_pinned"), "gpu", "GPU end to end (pinned)"))
    ends.append(line(ax, sfs, ms("gpu_q1", "kernel_only"), "gpu", "GPU kernel only", hollow=True))
    ax.set(xscale="log", yscale="log", xlabel="TPC-H scale factor (lineitem rows)",
           ylabel="Median time, ms (log scale)")
    ax.set_xticks(sfs, [f"SF{sf}\n{rows[sf] / 1e6:.1f}M rows" for sf in sfs])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.set_xlim(sfs[0] / 1.6, sfs[-1] * 3.2)  # room for the end labels
    best = [min(a, b) for a, b in zip(ms("polars", "cpu"), ms("duckdb", "cpu"))]
    sp = [b / g for b, g in zip(best, ms("gpu_q1", "e2e_pinned"))]
    save(fig, "tpch_q1.png", f"TPC-H Q1: GPU end to end (pinned) beats the fastest CPU engine "
         f"by {sp[0]:.1f}x at SF{sfs[0]} and {sp[-1]:.1f}x at SF{sfs[-1]}", (ax, ends))


if __name__ == "__main__":
    tpch_q1(load("tpch_q1.csv"))
