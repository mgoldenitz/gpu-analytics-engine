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

# One color per engine in every chart: pandas blue and Polars orange (colorblind-safe pair),
# DuckDB red, GPU in NVIDIA green. Each engine also gets its own marker shape.
# Line style: GPU lines solid and 2.5 wide, CPU engine lines dashed and 1.5 wide.
COLOR = {"pandas": "#0072B2", "polars": "#E69F00", "duckdb": "#D62728", "gpu": "#76b900"}
MARKER = {"pandas": "s", "polars": "^", "duckdb": "D", "gpu": "o"}
COPY_GRAY = "#d4d4d4"  # not an engine: the copy part of a GPU bar, light grey, so it does not look like an engine
TEXT, TEXT2, GRID = "#191919", "#666666", "#e5e5e5"
LABEL = {"pandas": "pandas", "polars": "Polars", "duckdb": "DuckDB"}
LABEL_GAP_PT = 12  # smallest vertical distance between two end labels

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


def line(ax, xs, ys, key, label):
    """Line with markers: GPU solid 2.5, CPU dashed 1.5. Returns its end point and label for end_labels()."""
    gpu = key == "gpu"
    ax.plot(xs, ys, color=COLOR[key], lw=2.5 if gpu else 1.5, ls="-" if gpu else "--", marker=MARKER[key], ms=5,
            mec="white", mew=1)
    return xs[-1], ys[-1], label


def end_labels(ax, ends):
    """Write each line's label at its right end, pushed apart so no two overlap.
    Works in screen points, so it is right on log and linear axes. Text stays in text ink."""
    ax.set_ylim(ax.get_ylim())  # freeze the axis so the positions below stay true
    pt = 72 / ax.figure.dpi
    ends = sorted(ends, key=lambda e: ax.transData.transform(e[:2])[1])
    want = [ax.transData.transform(e[:2])[1] * pt for e in ends]
    placed = []
    for y in want:  # bottom to top: each label at least LABEL_GAP_PT above the one below
        placed.append(max(y, placed[-1] + LABEL_GAP_PT) if placed else y)
    shift = (sum(want) - sum(placed)) / len(placed)  # center the group on its lines again
    for (x, y, text), w, p in zip(ends, want, placed):
        ax.annotate(text, (x, y), xytext=(6, p + shift - w), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT)


def save(fig, name, title, ends=None):
    """ends = (ax, end points): labels go on after the layout is final, so they are placed right."""
    fig.suptitle(textwrap.fill(title, 75), x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    if ends:
        end_labels(*ends)
    fig.savefig(FIGS / name, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", FIGS / name)


def filter_crossover(f):
    sizes = sorted({int(k[1]) for k in f})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ends = [line(ax, sizes, [f[(e, str(n), "10", "cpu")] for n in sizes], e, LABEL[e])
            for e in ("pandas", "polars", "duckdb")]
    for mode in ("e2e_pageable", "e2e_pinned"):
        ys = [f[("filter_own", str(n), "10", mode)] for n in sizes]
        ends.append(line(ax, sizes, ys, "gpu", f"GPU filter {mode[4:]}"))
    ax.set(xscale="log", yscale="log", xlabel="Rows (log scale)",
           ylabel="Median time, ms (log scale)")
    ax.xaxis.set_major_formatter(FuncFormatter(rows_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.set_xticks(sizes)
    ax.set_xlim(right=sizes[-1] * 3.2)  # room for the end labels
    save(fig, "filter_crossover.png", "Filter, 10% selectivity: GPU beats pandas and DuckDB at "
         "every size. Pinned beats Polars only at 1M rows; pageable never does",
         (ax, ends))


def filter_time_split(f):
    n = "100000000"
    kernel = f[("filter_own", n, "10", "kernel_only")]
    modes = ["kernel_only", "e2e_pageable", "e2e_pinned"]
    totals = [f[("filter_own", n, "10", m)] for m in modes]
    copies = [t - kernel for t in totals]
    copies[0] = 0.0  # kernel_only has no copy by definition
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(modes))
    ax.bar(x, [kernel] * 3, width=0.55, color=COLOR["gpu"], label="Kernel",
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
    ends = [line(ax, groups, [g[(e, "-", n, str(k), "uniform", "cpu")] for k in groups],
                 e, LABEL[e]) for e in ("pandas", "polars", "duckdb")]
    for st in ("global", "block"):
        ys = [g[("hash_groupby", st, n, str(k), "uniform", "e2e_pinned")] for k in groups]
        ends.append(line(ax, groups, ys, "gpu", f"GPU {st} (pinned)"))
    ax.set(xscale="log", yscale="log", xlabel="Number of groups (log scale)",
           ylabel="Median time, ms (log scale)")
    ax.set_xticks(groups)
    ax.xaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.set_xlim(5, 2e7)  # room for the end labels
    save(fig, "groupby_by_groups.png", "Group-by, 100M rows, uniform keys: GPU block beats every "
         "CPU engine at all group counts; global loses to Polars at 10", (ax, ends))


def groupby_block_speedup(g):
    n, groups = "100000000", ["10", "1000", "1000000"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.36
    for j, dist in enumerate(("uniform", "zipf")):
        sp = [g[("hash_groupby", "global", n, k, dist, "kernel_only")] /
              g[("hash_groupby", "block", n, k, dist, "kernel_only")] for k in groups]
        xs = [i + (j - 0.5) * w for i in range(len(groups))]
        # Same engine (block) both times: texture, not a second hue, tells the keys apart.
        ax.bar(xs, sp, width=w, color=COLOR["gpu"], hatch="//" if dist == "zipf" else None,
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
         "groups, but about 5% slower than global at 1M uniform groups")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    f, g = load("filter.csv"), load("groupby.csv")
    filter_crossover(f)
    filter_time_split(f)
    groupby_by_groups(g)
    groupby_block_speedup(g)


if __name__ == "__main__":
    main()
