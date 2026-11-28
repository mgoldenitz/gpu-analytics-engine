# Profiling

Session 1 (filter and group-by, stock clocks) is first. Session 2 (TPC-H Q1, SM clock locked at
945 MHz) is in "Session 2: TPC-H Q1" below.

## Session 1: filter and group-by

Nsight Systems and Nsight Compute runs of the published cases. Workload: `benchmarks/profile_run.py`
(100M int32 rows, pinned host memory, 3 warmups and 3 profiled iterations per case, every result
checked against pandas). No kernel code was changed.

Numbers here come only from `benchmarks/results/profiling/nsys_summary.csv` and
`benchmarks/results/profiling/ncu_summary.csv`. The raw reports stay in
`C:\data\gpu-analytics-engine\profiles` (not in the repo).

Run date: 2026-11-27, at HEAD 62e2e04. Checks before the runs: free RAM 10.03 GB, AC power,
Windows power mode Best performance (overlay ded574b5-45a0-4f42-8737-46345c09c238), highest other
process 2.2% of the CPU.

## Read this first

- nsys times are at normal clocks and are near the benchmark times. Compare them with the README.
- ncu locks the SM clock to base clock (`--clock-control base`, the default) and runs each kernel
  alone with counters on. ncu durations are longer than nsys and benchmark times
  (for example global insert at 10 groups: 150.710 ms in ncu, 88.970 ms in nsys).
  Never put an ncu duration next to a benchmark time. Use ncu only for ratios and percentages.
- The DRAM clock that ncu reports is not the same in every report: 8.99 GHz in 4 reports and
  11.99 GHz in 6. So each DRAM % is against the peak at that kernel's own clock
  (`dram_peak_gb_s` = ncu's `dram__bytes.sum.peak_sustained`, 32 byte per cycle, times the DRAM
  clock in the same report: 287.8 or 383.7 GB/s).

## Nsight Systems: one iteration per case (median of 3)

Filter, 10% selectivity:

- Iteration 30.886 ms (CPU range). GPU span 30.675 ms, kernels 12.630 ms, GPU idle 1.263 ms.
- H2D 15.147 ms for 400 MB (26.41 GB/s). D2H 1.518 ms for 40 MB (26.35 GB/s).
- flag_lt 2.385 ms. Scan range 7.486 ms: scan_blocks 4.944 ms, add_offsets 2.350 ms,
  0.190 ms idle. Count sync 0.088 ms on the GPU (0.194 ms CPU range). scatter 2.948 ms.
- Copies are 16.665 of 30.675 ms. Kernels are 12.630 ms.

Group-by (iteration, H2D, kernels range, insert kernel, D2H):

- block 10: 48.267 ms; H2D 36.003 ms for 800 MB (22.22 GB/s); kernels range 12.057 ms;
  insert_block 8.585 ms; D2H 0.095 ms (CPU range).
- global 10: 128.823 ms; H2D 35.829 ms (22.33 GB/s); kernels range 92.740 ms; insert 88.970 ms;
  D2H 0.096 ms.
- block 1M: 128.695 ms; H2D 35.963 ms (22.25 GB/s); kernels range 90.694 ms;
  insert_block 85.829 ms; D2H 0.789 ms for 20 MB (25.33 GB/s).
- global 1M: 127.056 ms; H2D 35.912 ms (22.28 GB/s); kernels range 89.196 ms; insert 84.354 ms;
  D2H 0.768 ms for 20 MB (26.04 GB/s).
- In every group-by case the kernels range also has CuPy library kernels from the input check
  `(keys == EMPTY).any()`: cupy_equal (1.787 to 2.132 ms) and cub_any_pass1 (0.663 to 0.793 ms).
  They are not our kernels.

Against the published numbers at that time (the stock-clock CSVs at commit 46deed6, not the
945 MHz CSVs that the README now uses):

- Filter: kernels 12.630 against 12.8 ms (-1.3%). Iteration 30.886 against e2e_pinned 29.7 ms (+4.0%).
- block 10: 12.057 against kernel_only 12.419 ms (-2.9%); 48.267 against e2e_pinned 45.837 ms (+5.3%).
- global 10: 92.740 against 89.554 ms (+3.6%); 128.823 against 124.587 ms (+3.4%).
- block 1M: 90.694 against 91.302 ms (-0.7%); 128.695 against 119.261 ms (+7.9%).
- global 1M: 89.196 against 86.598 ms (+3.0%); 127.056 against 118.060 ms (+7.6%).
- No difference is above 10%. The e2e numbers are all higher. Each NVTX range ends with a device
  sync (3 or 6 per iteration), and the benchmark has only one. This is a probable cause; it is not
  measured.
- Open: H2D is 26.41 GB/s in the filter and 22.2 to 22.3 GB/s in the group-by (16% lower).
  The cause is not known.

## Nsight Compute: per kernel (base SM clock, 1 launch each, first profiled iteration)

flag_lt (filter): 3.090 ms, DRAM 88.54% (254.8 of 287.8 GB/s), SM 28.76%, occupancy 78.90%
achieved / 100% theoretical, 16 registers, 0 shared memory. ncu: DRAM-bound ("DRAM (88.5% of peak)").

scan_blocks (filter): 7.490 ms, DRAM 28.83% (105.5 of 383.7 GB/s), SM 79.76%, occupancy 96.53% /
100%, 21 registers, 2.05 Kbyte static shared memory. ncu: limited by L1 ("L1 (83.1% of peak)").
This is the only filter kernel that is not DRAM-bound, and it is the longest one.

add_offsets (filter, level 0): 3.000 ms, DRAM 90.67% (260.9 of 287.8 GB/s), SM 17.89%, occupancy
82.08% / 100%, 16 registers, 0 shared memory. ncu: DRAM-bound ("DRAM (90.7% of peak)").

scatter (filter): 3.530 ms, DRAM 69.56% (254.2 of 383.7 GB/s), SM 33.93%, occupancy 81.31% / 100%,
16 registers, 0 shared memory. ncu: "Memory is more heavily utilized than Compute, dominated by
DRAM (69.6% of peak)".

insert_block (group-by, 10 groups): 14.080 ms, DRAM 15.71% (57.0 of 383.7 GB/s), SM 68.57%,
occupancy 82.21% / 83.33%, 28 registers, 16.38 Kbyte static shared memory. ncu: limited by L1
("L1 (99.5% of peak)"), which is where shared memory atomics run.

insert (group-by global, 10 groups): 150.710 ms, DRAM 1.48% (5.4 of 383.7 GB/s), SM 2.48%,
occupancy 81.55% / 100%, 16 registers, 0 shared memory. ncu: "low compute throughput and memory
bandwidth utilization ... typically indicate latency issues".

insert_block (group-by, 1M groups): 91.830 ms, DRAM 29.48% (107.0 of 383.7 GB/s), SM 31.71%,
occupancy 80.87% / 83.33%, 28 registers, 16.38 Kbyte static shared memory. ncu: latency issues.

insert (group-by global, 1M groups): 87.940 ms, DRAM 30.14% (109.4 of 383.7 GB/s), SM 6.58%,
occupancy 77.60% / 100%, 16 registers, 0 shared memory. ncu: latency issues.

flag_used (group-by block, 1M): 0.049 ms, DRAM 59.90%, occupancy 71.52% / 100%. ncu: latency issues.
scatter (group-by block, 1M): 0.217 ms, DRAM 93.90% (270.1 of 287.7 GB/s). ncu: DRAM-bound.
Both are less than 0.3% of the group-by kernel time.

## Why block is 7.21x faster than global at 10 groups

7.21x is the stock-clock kernel_only ratio (89.554 / 12.419 ms, commit 46deed6). The 945 MHz
CSV gives 5.90x. In nsys the insert kernels alone
differ by 10.4x (88.970 / 8.585 ms); the CuPy check kernels take about the same time in both
(2.579 ms block, 2.466 ms global).

- Global insert sends every row to global memory atomics: 52,844,313 atom and 52,248,712 red
  global accesses (L1 set accesses). In L2 that is 80,832,701 atomic sectors and 124,881,455
  reduction sectors.
- Block insert does the same work in shared memory: 59,413,149 shared atomic instructions, and
  only 268,565 atom and 415,055 red global accesses (one flush per block). That is 154x fewer
  global atomic accesses.
- With only 10 keys, the global atomics go to very few L2 slices. ncu shows one L2 slice at
  99.98% throughput while the average over all slices is 7.99%. The L2 atomic unit is 70.74% busy
  on its busiest slice and 5.99% on average. DRAM is 1.48% and SM 2.48%. So the global kernel waits
  on a few L2 slices that serialize the atomics; it is not limited by bandwidth or compute.
- In block insert the busiest L2 slice is at 7.85% and the busiest atomic unit at 2.69%. Its limit
  moves to L1 (99.5%), where the shared atomics run.

At 1M groups the two strategies are close (ncu 91.830 against 87.940 ms; nsys 85.829 against
84.354 ms). Block insert still does about the same number of global atomics as global insert
(145,007,083 atom and 199,687,365 red, against 145,203,311 and 199,969,859), and it also does
215,368,455 shared atomic instructions. The shared table does not hold 1M keys, so the shared work
is extra. Busiest L2 slice: 19.56% (block) and 19.25% (global); the contention at 10 groups is gone.

## Commands

All commands run from the project root in PowerShell, with these paths:

```
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.2\target-windows-x64\nsys.exe'
$ncu  = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe'
$py   = 'C:\venvs\gpu-analytics-engine\Scripts\python.exe'
$p    = 'C:\data\gpu-analytics-engine\profiles'
```

Nsight Systems timeline and stats:

```
& $nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true -o "$p\engine_timeline" $py -m benchmarks.profile_run
& $nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum,nvtx_gpu_proj_trace,cuda_gpu_trace --format csv --output "$p\stats" "$p\engine_timeline.nsys-rep"
& $py -m benchmarks.profile_summary "$p\stats"
```

`profile_summary` also prints every kernel launch and the `--launch-skip` value for each kernel's
first launch in iter1. Nsight Compute, one run per kernel (name, regex, skip):
filter_flag_lt flag_lt 4, filter_scan_blocks scan_blocks 12, filter_add_offsets add_offsets 9,
filter_scatter scatter 4, groupby_block_10_insert_block insert_block 3, groupby_global_10_insert
insert 3, groupby_block_1M_insert_block insert_block 9, groupby_global_1M_insert insert 9,
groupby_block_1M_flag_used flag_used 15, groupby_block_1M_scatter scatter 22.

```
& $ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --metrics "l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,lts__t_requests_srcunit_l1_op_atom.sum,lts__t_requests_srcunit_l1_op_red.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed_op_shared_atom.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum" -k "regex:^<kernel>$" --launch-skip <skip> --launch-count 1 -f -o "$p\ncu_<name>" $py -m benchmarks.profile_run
& $ncu --import "$p\ncu_<name>.ncu-repz" --page details --csv > "$p\ncu_<name>.csv"
& $ncu --import "$p\ncu_<name>.ncu-repz" --page raw --csv > "$p\ncu_<name>_raw.csv"
```

DRAM peak (one extra run), then the summary:

```
& $ncu --metrics "dram__bytes.sum.peak_sustained,dram__cycles_elapsed.avg.per_second,dram__bytes.sum.per_second,dram__bytes.sum.pct_of_peak_sustained_elapsed" -k "regex:^flag_lt$" --launch-skip 4 --launch-count 1 -f -o "$p\ncu_peak_flag_lt" $py -m benchmarks.profile_run
& $py -m benchmarks.ncu_summary $p
```

## Differences from the session plan

- nsys `--trace=osrt` is Linux only, so the trace is `cuda,nvtx`.
- The workload runs with `-m benchmarks.profile_run`, not the file path, because it imports
  from `benchmarks` and `engine`.
- ncu 2026.4.0 writes `.ncu-repz`, not `.ncu-rep`. Both are in `.gitignore`.
- `lts__t_requests_srcunit_l1_op_atom.sum` and `..._op_red.sum` are 0 for every kernel on sm_120,
  so the L2 atomic figures use `lts__t_sectors_op_atom.sum` and `lts__t_sectors_op_red.sum`.
- The `lts__throughput` and `lts__d_atomic_input_cycles_active` maximum over slices come from the
  raw page that the sections already collect.

## Session 2: TPC-H Q1 (SM clock locked at 945 MHz)

Workload: `benchmarks/profile_q1.py` (SF10, 59,986,052 rows, the e2e_pinned path, 3 warmups and
3 profiled iterations, every result checked against the DuckDB Q1 answer). Run date 2026-11-28.
Clocks locked with nvidia-smi: SM 945 MHz, memory 12001 MHz. Raw reports in
`C:\data\gpu-analytics-engine\profiles` (`q1_timeline.nsys-rep`, `q1_stats_*.csv`,
`ncu_q1_block_lock945.*`).

Checks: before nsys, free RAM 9.79 GB, Best performance overlay, AC power, highest other process
sqlservr 4.35%. For the ncu run the check ran just after the run: free RAM 9.76 GB, same overlay,
AC power; one 10 s sample had sqlservr at 5.21%, the others were below 5%.

Nsight Systems, per iteration:

- 1 kernel launch (q1_block), 1 memset (the output fill), 6 H2D copies (1139.7 MB in total),
  1 D2H copy of 288 bytes. No hash table and no compaction kernels.
- q1_block 65.917 ms (median). H2D 42.85 ms on average (26.6 GB/s); NVTX "q1 H2D" median 43.13 ms.
- Iteration about 109.6 ms, against 109.342 ms e2e_pinned in `tpch_q1.csv`.

Nsight Compute, q1_block (iter1, `--launch-skip 3`, `--clock-control none` so the 945 MHz lock
stays; ncu reports SM frequency 945.00 MHz):

- Duration 65.87 ms (nsys 65.917 ms), so at this clock ncu and nsys agree.
- DRAM 4.90% of peak: 17.39 GB/s of 383.7 GB/s (32 byte per cycle at the reported 11.99 GHz DRAM
  clock). 1.145 GB read, about 19 bytes per row.
- L1/TEX 81.67%, Compute (SM) 74.73%, L2 2.44%.
- 296,781,817 shared atomic instructions; 102,522 L2 reduction sectors (the one flush per block).
- Occupancy 66.47% achieved / 66.67% theoretical, limited by registers (64 per thread).
  Grid 14,646 blocks of 256 threads, 288 bytes static shared memory per block.
- Bottleneck: the L1/shared-memory pipe, where the shared atomics run, and SM issue. DRAM is at
  4.9%, so the kernel is not limited by memory bandwidth. A faster kernel needs fewer shared
  atomics per row (for example a warp-level reduce first), not fewer bytes.
- A first ncu run used the default `--clock-control base`. It reported SM 1.44 GHz, 43.30 ms and
  DRAM 7.21%. It is not used, because it was not at the lock.

Commands (same paths as session 1):

```
& $nsys profile --trace=cuda,nvtx --sample=none --force-overwrite=true -o "$p\q1_timeline" $py -m benchmarks.profile_q1
& $nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum,cuda_gpu_trace --format csv --output "$p\q1_stats" "$p\q1_timeline.nsys-rep"
& $ncu --clock-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --metrics "smsp__inst_executed_op_shared_atom.sum,lts__t_sectors_op_red.sum,dram__bytes.sum,dram__bytes.sum.peak_sustained,dram__cycles_elapsed.avg.per_second,dram__bytes.sum.per_second" -k "regex:^q1_block$" --launch-skip 3 --launch-count 1 -f -o "$p\ncu_q1_block_lock945" $py -m benchmarks.profile_q1
& $ncu --import "$p\ncu_q1_block_lock945.ncu-repz" --page details --csv > "$p\ncu_q1_block_lock945.csv"
& $ncu --import "$p\ncu_q1_block_lock945.ncu-repz" --page raw --csv > "$p\ncu_q1_block_lock945_raw.csv"
```

The kernel `q1_block` is named `q1_v1` from v1.1.0. The file names above keep the old name.

## Q1 kernel fix, v1.1.0 (2026-11-28, under the lock)

`q1_v2` keeps the 36 sums in registers per thread (grid-stride loop, masked adds), adds them
across each warp with 5 `__shfl_down_sync` steps, then does one shared atomic add per non-zero sum
per warp and one global atomic add per non-zero sum per block. Grid 52 blocks of 256 threads
(2 per SM from the occupancy API, 26 SMs). Files: `C:\data\gpu-analytics-engine\profiles\ncu_q1_v2_lock945.*`.

Nsight Compute, q1_v2 against q1_v1 (SF10, `--launch-skip 3`, `--clock-control none`, SM 945 MHz):

- Duration 11.59 ms (q1_v1 65.87 ms).
- Shared atomic instructions 9,994 (296,781,817). L2 reduction sectors 364 (102,522).
- L1/TEX 12.94% (81.67%), Compute (SM) 34.35% (74.73%), L2 13.89% (2.44%).
- DRAM 98.54 GB/s (17.39 GB/s), 1.14 GB read. ncu gives 34.24% of peak, but at a DRAM clock of
  8.99 GHz (about 287.8 GB/s peak). The q1_v1 run had 11.99 GHz. Compare GB/s, not percent.
- 96 registers per thread, 0 local memory spill requests, 288 bytes static shared memory.
- Occupancy 33.14% achieved / 33.33% theoretical, limited by registers (2 blocks per SM).
- ncu: "low compute throughput and memory bandwidth utilization relative to the peak", which
  "typically indicate[s] latency issues". The next limit is latency at low occupancy, not atomics.

Command (the same as above, with the kernel name, the output name, 3 more metrics and the
argument `v2` for `profile_q1`):

```
& $ncu --clock-control none --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --metrics "smsp__inst_executed_op_shared_atom.sum,lts__t_sectors_op_red.sum,dram__bytes.sum,dram__bytes.sum.peak_sustained,dram__cycles_elapsed.avg.per_second,dram__bytes.sum.per_second,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,launch__registers_per_thread,smsp__cycles_active.avg.per_second" -k "regex:^q1_v2$" --launch-skip 3 --launch-count 1 -f -o "$p\ncu_q1_v2_lock945" $py -m benchmarks.profile_q1 v2
& $ncu --import "$p\ncu_q1_v2_lock945.ncu-repz" --page details --csv > "$p\ncu_q1_v2_lock945_details.csv"
& $ncu --import "$p\ncu_q1_v2_lock945.ncu-repz" --page raw --csv > "$p\ncu_q1_v2_lock945_raw.csv"
```

## Q1 narrow columns, v1.2.0 (2026-11-28, under the lock)

`q1_v3` is `q1_v2` on narrow columns (10 bytes per row, not 19) with 4-row vector loads, and
`__launch_bounds__(256, 2)`. Files: `C:\data\gpu-analytics-engine\q1_narrow\` (`q1_v2_timeline`,
`q1_v3_timeline`, `q1_*_stats_*.csv`, `ncu_q1_v3_lock945.*`). Each run started after the benchmark
checks passed (free RAM 8.34 and 9.12 GB, Best performance overlay, AC power, no process above 5%).

Nsight Systems, SF10, iterations 1 to 3, median:

- q1_v2 wide: H2D 42.82 ms for 1,139.7 MB (26.6 GB/s), kernel 11.40 ms, iteration about 55.1 ms.
- q1_v3 narrow: H2D 22.67 ms for 599.9 MB (26.5 GB/s), kernel 4.44 ms, iteration about 27.7 ms.
- The PCIe speed is the same. Half the bytes give half the copy time.

Nsight Compute, q1_v3 against q1_v2 (`--launch-skip 3`, `--clock-control none`, SM 945 MHz,
DRAM 8.99 GHz in both reports, so about 287.8 GB/s peak for both):

- Duration 4.46 ms (11.59 ms).
- DRAM 136.03 GB/s (98.54 GB/s), 0.606 GB read (1.14 GB). 47.27% of peak (34.24%).
- L1/TEX 15.75% (12.94%), Compute (SM) 83.06% (34.35%).
- 128 registers per thread (96). The compiler reports 776 bytes of local memory; ncu counts
  0 local memory spill requests, so the spill code does not run in the loop.
- Occupancy 26.94% achieved / 33.33% theoretical (33.14% / 33.33%), limited by registers,
  2 blocks per SM, grid 52 blocks of 256 threads.
- Shared atomic instructions 10,016, L2 reduction sectors 364.
- ncu: "utilizing greater than 80.0% of the available compute or memory performance". The
  limit is now SM compute (the masked adds over 6 groups, 6 sums each), not memory or latency.

Commands (the same as for q1_v2, with `v3` in place of `v2`; nsys with `-o "$p\q1_v3_timeline"`).
