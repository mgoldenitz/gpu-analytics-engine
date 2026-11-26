"""Print the filter benchmark summary from the CSV.  python -m benchmarks.summary"""
import csv
from pathlib import Path

CPU = ["pandas", "polars", "duckdb"]
GPU = ["cupy_builtin", "filter_cub", "filter_own"]
GPU_MODES = ["kernel_only", "e2e_pageable", "e2e_pinned"]
SHORT = {"kernel_only": "ker", "e2e_pageable": "pag", "e2e_pinned": "pin"}
CSV = Path(__file__).parent / "results" / "filter.csv"


def load(path):
    with open(path, newline="") as fh:
        return {(r["engine"], int(r["n"]), int(r["selectivity_pct"]), r["mode"]): float(r["median_ms"])
                for r in csv.DictReader(fh)}


def print_summary(path=CSV):
    d = load(path)
    sizes = sorted({k[1] for k in d})
    pcts = sorted({k[2] for k in d})
    cols = [(e, "cpu") for e in CPU] + [(e, m) for e in GPU for m in GPU_MODES]
    for n in (10_000_000, 100_000_000):
        print(f"\nn = {n:,}  (median ms; ker = kernel_only, pag = e2e_pageable, pin = e2e_pinned)")
        print("sel%" + "".join(f"{e[:8] + ('' if m == 'cpu' else '/' + SHORT[m]):>13}" for e, m in cols))
        for p in pcts:
            print(f"{p:>4}" + "".join(f"{d.get((e, n, p, m), float('nan')):>13.2f}" for e, m in cols))

    for mode in ("e2e_pageable", "e2e_pinned"):
        print(f"\n{mode}: smallest n from which the GPU beats the CPU engine at that size and every")
        print("larger one (- = it does not stay ahead up to the largest tested size):")
        for g in GPU:
            for c in CPU:
                wins = []
                for p in pcts:
                    beats = [(g, n, p, mode) in d and d[(g, n, p, mode)] < d[(c, n, p, "cpu")]
                             for n in sizes]
                    first = next((n for i, n in enumerate(sizes) if all(beats[i:])), None)
                    wins.append(f"{p}%: {first:,}" if first else f"{p}%: -")
                print(f"  {g:>12} vs {c:<6}  " + "  ".join(wins))


if __name__ == "__main__":
    print_summary()
