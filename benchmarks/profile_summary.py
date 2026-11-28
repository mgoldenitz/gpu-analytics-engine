"""Turn the nsys stats CSV exports of benchmarks/profile_run.py into nsys_summary.csv.

Reads <prefix>_nvtx_gpu_proj_trace.csv and <prefix>_cuda_gpu_trace.csv (from `nsys stats
--format csv --output <prefix>`). Only the 3 profiled iterations count; warmups are left out.
For each case and phase it writes the median over those 3 of: the CPU range time, the GPU
span (NVTX range projected onto the GPU), kernel time, copy time and GB/s, and GPU idle
time inside the span (launch overhead and syncs). It also prints every kernel launch in
order, with the --launch-skip value that lands ncu on the first profiled iteration.

Run from the project root:
  python -m benchmarks.profile_summary C:\\data\\gpu-analytics-engine\\profiles\\stats
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "results" / "profiling" / "nsys_summary.csv"
ITER = re.compile(r"^(.*) (iter\d+|warmup)$")
FIELDS = ["cpu_range_ms", "gpu_span_ms", "kernel_ms", "h2d_ms", "h2d_mb", "d2h_ms", "d2h_mb",
          "gpu_idle_ms", "gpu_ops"]


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def ops_in(ops, start, dur):
    return [o for o in ops if start <= o["start"] < start + dur]


def measure(ops, r):
    """One range: CPU time, GPU span, and how the GPU spent that span."""
    span = int(r["Projected Duration (ns)"])
    inside = ops_in(ops, int(r["Projected Start (ns)"]), span)
    kern = [o for o in inside if not o["name"].startswith("[")]
    h2d = [o for o in inside if o["name"] == "[CUDA memcpy Host-to-Device]"]
    d2h = [o for o in inside if o["name"] == "[CUDA memcpy Device-to-Host]"]
    ms = lambda xs: sum(o["dur"] for o in xs) / 1e6
    return {"cpu_range_ms": int(r["Orig Duration (ns)"]) / 1e6, "gpu_span_ms": span / 1e6,
            "kernel_ms": ms(kern), "h2d_ms": ms(h2d), "h2d_mb": sum(o["mb"] for o in h2d),
            "d2h_ms": ms(d2h), "d2h_mb": sum(o["mb"] for o in d2h),
            "gpu_idle_ms": (span - sum(o["dur"] for o in inside)) / 1e6, "gpu_ops": len(inside),
            "kernels": kern}


def main(prefix):
    ranges = read(f"{prefix}_nvtx_gpu_proj_trace.csv")
    ops = [{"start": int(o["Start (ns)"]), "dur": int(o["Duration (ns)"]), "name": o["Name"],
            "mb": float(o["Bytes (MB)"] or 0)} for o in read(f"{prefix}_cuda_gpu_trace.csv")]
    by_id = {r["RangeId"]: r for r in ranges}
    rows = defaultdict(list)  # (case, phase) -> one dict per profiled iteration
    for r in ranges:
        top = by_id[r["ParentId"]] if r["ParentId"] else r
        m = ITER.match(top["Name"].lstrip(":"))
        if not m or m.group(2) == "warmup":
            continue
        case = m.group(1)
        phase = "iteration total" if r is top else r["Name"].lstrip(":")[len(case) + 1:]
        got = measure(ops, r)
        rows[(case, phase)].append(got)
        if r is top:  # one row per kernel name: its total time inside the iteration
            per = defaultdict(int)
            for k in got["kernels"]:
                per[k["name"]] += k["dur"]
            for name, ns in per.items():
                rows[(case, "kernel " + name)].append({"kernel_ms": ns / 1e6})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "phase", "iterations"] + FIELDS + ["h2d_gb_s", "d2h_gb_s"])
        for (case, phase), its in rows.items():
            med = {f: statistics.median(i[f] for i in its) for f in FIELDS if f in its[0]}
            gbs = [med[f"{d}_mb"] / med[f"{d}_ms"] if med.get(f"{d}_ms") else ""
                   for d in ("h2d", "d2h")]  # 1 MB per ms = 1 GB/s
            w.writerow([case, phase, len(its)] +
                       [f"{med[f]:.3f}" if f in med else "" for f in FIELDS] +
                       [f"{g:.2f}" if g != "" else "" for g in gbs])
    print("wrote", OUT)
    launch_list(ranges, ops)


def launch_list(ranges, ops):
    """Every kernel launch in order; the skip for each kernel's first launch in each iter1."""
    tops = [r for r in ranges if not r["ParentId"]]
    seen = defaultdict(int)
    first = {}
    print("\n#  launch  kernel (index among launches of the same name)  range")
    for i, o in enumerate(x for x in ops if not x["name"].startswith("[")):
        where = next((t["Name"].lstrip(":") for t in tops
                      if ops_in([o], int(t["Projected Start (ns)"]),
                                int(t["Projected Duration (ns)"]))), "(no range)")
        print(f"{i:5d}  {o['name']} [{seen[o['name']]}]  {where}")
        if where.endswith("iter1"):
            first.setdefault((where, o["name"]), seen[o["name"]])
        seen[o["name"]] += 1
    print("\n--launch-skip for the first launch of each kernel in iter1 (count by that name):")
    for (where, name), skip in first.items():
        print(f"  {where:32s} {name:30s} {skip}")


if __name__ == "__main__":
    main(sys.argv[1])
