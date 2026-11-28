"""Turn the Nsight Compute CSV exports into ncu_summary.csv, one row per profiled kernel.

Reads <dir>/ncu_<case>.csv (`ncu --import <report> --page details --csv`) and
<dir>/ncu_<case>_raw.csv (`--page raw --csv`). Every number is copied from those exports.
The one product is the DRAM peak: ncu's dram__bytes.sum.peak_sustained (32 byte per DRAM
cycle on this GPU, from the extra run in docs/profiling.md) times the DRAM clock that ncu
reports for the same kernel, because the DRAM clock is not the same in every report.

Run from the project root:
  python -m benchmarks.ncu_summary C:\\data\\gpu-analytics-engine\\profiles
"""
import csv
import sys
from pathlib import Path

OUT = Path(__file__).parent / "results" / "profiling" / "ncu_summary.csv"
DRAM_BYTES_PER_CYCLE = 32.0  # dram__bytes.sum.peak_sustained
DETAILS = {  # column -> (section, metric name) in the details page
    "duration": ("GPU Speed Of Light Throughput", "Duration"),
    "dram_pct": ("GPU Speed Of Light Throughput", "DRAM Throughput"),
    "sm_pct": ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"),
    "l2_pct": ("GPU Speed Of Light Throughput", "L2 Cache Throughput"),
    "l2_hit_pct": ("Memory Workload Analysis", "L2 Hit Rate"),
    "theoretical_occ_pct": ("Occupancy", "Theoretical Occupancy"),
    "achieved_occ_pct": ("Occupancy", "Achieved Occupancy"),
    "registers_per_thread": ("Launch Statistics", "Registers Per Thread"),
    "smem_static": ("Launch Statistics", "Static Shared Memory Per Block"),
    "smem_dynamic": ("Launch Statistics", "Dynamic Shared Memory Per Block"),
    "grid_size": ("Launch Statistics", "Grid Size"),
}
RAW = {  # column -> raw metric
    "sm_ghz": "gpc__cycles_elapsed.avg.per_second",
    "dram_ghz": "dram__cycles_elapsed.avg.per_second",
    "dram_gb_s": "dram__bytes.sum.per_second",
    "global_atom": "l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum",
    "global_red": "l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum",
    "l2_atom_sectors": "lts__t_sectors_op_atom.sum",
    "l2_red_sectors": "lts__t_sectors_op_red.sum",
    "l2_atomic_unit_busy_pct": "lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l2_atomic_unit_busy_max_slice_pct":
        "lts__d_atomic_input_cycles_active.max.pct_of_peak_sustained_elapsed",
    "l2_throughput_max_slice_pct": "lts__throughput.max.pct_of_peak_sustained_elapsed",
    "shared_atom_inst": "smsp__inst_executed_op_shared_atom.sum",
}
TO_MS = {"ms": 1.0, "us": 1e-3, "ns": 1e-6, "s": 1e3}


def num(text):
    return float(text.replace(",", ""))


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def summarize(details_path, raw_path):
    rows = read(details_path)
    raw = read(raw_path)[1]  # row 0 holds the units
    by = {(r["Section Name"], r["Metric Name"]): r for r in rows if r["Metric Name"]}
    out = {"kernel": rows[0]["Kernel Name"]}
    for col, key in DETAILS.items():
        out[col] = by[key]["Metric Value"].replace(",", "")
    out["duration_ms"] = f"{num(out.pop('duration')) * TO_MS[by[DETAILS['duration']]['Metric Unit']]:.3f}"
    for col in ("smem_static", "smem_dynamic"):  # as ncu prints it; Kbyte has only 2 decimals
        out[col] = f"{out[col]} {by[DETAILS[col]]['Metric Unit']}"
    out.update({col: raw[m].replace(",", "") for col, m in RAW.items()})
    out["dram_peak_gb_s"] = f"{DRAM_BYTES_PER_CYCLE * num(out['dram_ghz']):.1f}"
    sol = [r for r in rows if r["Rule Name"] == "SOLBottleneck"]
    out["bottleneck"] = " ".join(sol[0]["Rule Description"].split()) if sol else ""
    return out


def main(folder):
    folder = Path(folder)
    rows = []
    for d in sorted(folder.glob("ncu_*.csv")):
        if d.stem.endswith("_raw") or d.stem.startswith("ncu_peak"):
            continue
        rows.append({"case": d.stem[4:], **summarize(d, d.with_name(d.stem + "_raw.csv"))})
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main(sys.argv[1])
