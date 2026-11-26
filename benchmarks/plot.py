"""Make the result charts from results/filter.csv and results/groupby.csv.

Saves PNGs (200 dpi) to docs/figures/.  Run from the project root:  python -m benchmarks.plot
"""
import csv
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no window; files only
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RESULTS = Path(__file__).parent / "results"
FIGS = Path(__file__).parent.parent / "docs" / "figures"

# One color per engine in every chart. Slots 1-6 of the dataviz reference palette,
# in its validated order (colorblind-safe on adjacent pairs, light surface).
COLOR = {
    "pandas": "#257de0", "polars": "#f47230", "duckdb": "#00b894",
    "filter_own": "#f2a600", "global": "#ee7fac", "block": "#008a17",
}
COPY_GRAY = "#9c9c9c"  # not an engine: the copy part of a GPU bar
TEXT, TEXT2, GRID = "#191919", "#666666", "#e5e5e5"
LABEL = {"pandas": "pandas", "polars": "Polars", "duckdb": "DuckDB",
         "global": "GPU global (pinned)", "block": "GPU block (pinned)"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
})


def load(name):
    """{(column values except median_ms): median_ms} from one results CSV."""
    with (RESULTS / name).open() as fh:
        rows = list(csv.DictReader(fh))
    return {tuple(v for k, v in r.items() if k != "median_ms"): float(r["median_ms"]) for r in rows}


def rows_fmt(x, _):
    return f"{x / 1e6:g}M"


def plain_fmt(x, _):
    return f"{x:,.0f}" if x >= 1 else f"{x:g}"


def line(ax, xs, ys, key, label, dashed=False, dy=0):
    """2px line with markers and a direct label at its right end (text stays in text ink).
    dy moves the label up or down in points when two lines end at about the same value."""
    ax.plot(xs, ys, color=COLOR[key], lw=2, ls="--" if dashed else "-", marker="o", ms=5,
            mec="white", mew=1, label=label)
    ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=8, color=TEXT)


def save(fig, name, title):
    fig.suptitle(textwrap.fill(title, 75), x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", FIGS / name)


def filter_crossover(f):
    sizes = sorted({int(k[1]) for k in f})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    # Polars and GPU pageable end close together: split their labels.
    for e, dy in (("pandas", 0), ("polars", -5), ("duckdb", 0)):
        line(ax, sizes, [f[(e, str(n), "10", "cpu")] for n in sizes], e, LABEL[e], dy=dy)
    for mode, dashed, dy in (("e2e_pageable", True, 5), ("e2e_pinned", False, 0)):
        ys = [f[("filter_own", str(n), "10", mode)] for n in sizes]
        line(ax, sizes, ys, "filter_own", f"GPU filter {mode[4:]}", dashed, dy)
    ax.set(xscale="log", yscale="log", xlabel="Rows (log scale)",
           ylabel="Median time, ms (log scale)")
    ax.xaxis.set_major_formatter(FuncFormatter(rows_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.set_xticks(sizes)
    ax.set_xlim(right=sizes[-1] * 3.2)  # room for the end labels
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "filter_crossover.png", "Filter, 10% selectivity: GPU beats pandas and DuckDB at "
         "every size. Pinned beats Polars from 10M rows; pageable beats it only at 10M")


def filter_time_split(f):
    n = "100000000"
    kernel = f[("filter_own", n, "10", "kernel_only")]
    modes = ["kernel_only", "e2e_pageable", "e2e_pinned"]
    totals = [f[("filter_own", n, "10", m)] for m in modes]
    copies = [t - kernel for t in totals]
    copies[0] = 0.0  # kernel_only has no copy by definition
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(modes))
    ax.bar(x, [kernel] * 3, width=0.55, color=COLOR["filter_own"], label="Kernel",
           edgecolor="white", linewidth=2)
    ax.bar(x, copies, bottom=kernel, width=0.55, color=COPY_GRAY,
           label="Copy and transfer overhead", edgecolor="white", linewidth=2)
    for i, (t, c) in enumerate(zip(totals, copies)):
        note = f"{t:.1f} ms" + (f"\ncopy {c / t:.0%}" if c else "")
        ax.annotate(note, (i, t), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x), ["kernel only", "end to end\npageable", "end to end\npinned"])
    ax.set(ylabel="Median time, ms", ylim=(0, max(totals) * 1.25))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    pg, pn = copies[1] / totals[1], copies[2] / totals[2]
    save(fig, "filter_time_split.png", f"Filter, 100M rows, 10%: the copy is the bottleneck "
         f"({pg:.0%} of pageable time, {pn:.0%} of pinned)")


def groupby_by_groups(g):
    n, groups = "100000000", [10, 1_000, 1_000_000]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    # pandas and DuckDB end at about the same time, and so do global and block: split the labels.
    for e, dy in (("pandas", -7), ("polars", 0), ("duckdb", 7)):
        ys = [g[(e, "-", n, str(k), "uniform", "cpu")] for k in groups]
        line(ax, groups, ys, e, LABEL[e], dy=dy)
    for st, dy in (("global", -7), ("block", 7)):
        ys = [g[("hash_groupby", st, n, str(k), "uniform", "e2e_pinned")] for k in groups]
        line(ax, groups, ys, st, LABEL[st], dy=dy)
    ax.set(xscale="log", yscale="log", xlabel="Number of groups (log scale)",
           ylabel="Median time, ms (log scale)")
    ax.set_xticks(groups)
    ax.xaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.set_xlim(5, 2e7)  # room for the end labels
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "groupby_by_groups.png", "Group-by, 100M rows, uniform keys: GPU block beats every "
         "CPU engine at all group counts; global loses to Polars at 10")


def groupby_block_speedup(g):
    n, groups = "100000000", ["10", "1000", "1000000"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.36
    for j, dist in enumerate(("uniform", "zipf")):
        sp = [g[("hash_groupby", "global", n, k, dist, "kernel_only")] /
              g[("hash_groupby", "block", n, k, dist, "kernel_only")] for k in groups]
        xs = [i + (j - 0.5) * w for i in range(len(groups))]
        # Same engine (block) both times: texture, not a second hue, tells the keys apart.
        ax.bar(xs, sp, width=w, color=COLOR["block"], hatch="//" if dist == "zipf" else None,
               edgecolor="white", linewidth=2, label=f"{dist.capitalize()} keys")
        for x, s in zip(xs, sp):
            ax.annotate(f"{s:.2f}x", (x, s), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.axhline(1, color=TEXT2, lw=1, ls=":")
    ax.set_xticks(range(len(groups)), ["10", "1,000", "1,000,000"])
    ax.set(xlabel="Number of groups (dotted line: same speed)",
           ylabel="Speedup of block over global, x (kernel only)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "groupby_block_speedup.png", "Group-by, 100M rows: block is 6-7x faster at 10 "
         "groups, but 5% slower than global at 1M uniform groups")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    f, g = load("filter.csv"), load("groupby.csv")
    filter_crossover(f)
    filter_time_split(f)
    groupby_by_groups(g)
    groupby_block_speedup(g)


if __name__ == "__main__":
    main()
