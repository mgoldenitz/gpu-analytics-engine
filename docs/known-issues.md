# Known issues

All numbers are from the locked CSVs (SM 945 MHz, memory 12001 MHz), unless a line says otherwise. Older Balanced-mode and stock-clock notes are in git history.

## Results

- Filter: no crossover against Polars at 10%. GPU pinned wins only at 1M rows (0.533 against 0.848 ms). At 1% Polars wins at every size. At 100M and 1% the copies (40.2 - 20.8 = 19.4 ms) cost more than the whole Polars filter (10.7 ms).
- Pageable end to end never beats Polars. At 100M and 10%, the copies are 67% of the pageable time (66.0 ms total, 21.6 ms kernel).
- Group-by `global` at 10 groups: all threads atomicAdd to a few slots (159.4 ms kernel_only at 100M). `block` gives 27.0 ms.
- Group-by `block` at 1M uniform groups: most rows miss the shared table, so it is 4% slower than `global` (98.0 against 94.1 ms). The 2M-slot table (32 MB) is about the L2 size, so the atomics go to DRAM.
- The locked times are slower than stock. Example: block at 10 groups, 27.0 ms locked against 12.4 ms at stock.
- Filling the pinned buffer takes 21.2 ms at 100M rows (stock-clock run, outside the CSV). The pinned times leave it out.

## Lock and noise

- 3 filter cells were still off-lock after a rerun: filter_cub 10M 50% e2e_pageable, filter_cub 10M 90% e2e_pageable, filter_cub 100M 50% kernel_only. The CSV keeps them. 10 of 13 off-lock cells were fixed by the rerun.
- Power cap: some off-lock samples show SW power cap (0x4) with the SM at 397 to 892 MHz at 34 to 43 W.
- At stock clocks the GPU dropped to P8 or P4 between runs, and group-by times changed by 60 to 77%. This is why the clocks are locked.
- Polars filter at 100M and 10% went from 23.8 ms (stock run) to 15.5 ms (locked run), -35%. Polars does not use the GPU. Cause not measured.
- Polars 1K Zipf groups at 100M is 731.4 ms, about 6x its uniform time. Stock run: the uniform point had two modes (114.2, 137.4, 544.7 ms).
- A browser was open at about 00:50 on 2026-11-27, just after the locked group-by sweep started (00:50:22). I stopped it at 00:53 and ran group-by and Q1 again. The filter cell rerun had ended at 00:46:15, before the browser opened.
- Some gate outputs (free RAM, power overlay, CPU use) were not saved to a file.
- The gate for the Q1 ncu run was checked after the run. One sample showed sqlservr at 5.21% CPU, above the 5% limit. ncu counts are not timing, so the run stays.
- The v1.1.0 Q1 rerun: Windows logged a power source change at 02:33:04 and 02:33:09 (about 5 s on battery), during the CPU-use check of the gate. The power check at 02:33:27 showed AC, and the timed runs were 02:33:28 to 02:33:47, so no run overlapped. Clock log: 8 windows, 0 off lock; 2 windows had no samples after the 1 s skip.
- Q1 H2D at SF10 (nsys, v1.2.0, under the lock): 42.821 ms for 1,139.7 MB (`q1_v2` wide), 22.672 ms for 599.9 MB (`q1_v3` narrow), both about 26.5 GB/s.
- The v1.2.0 sweep gave `q1_v2` SF10 e2e pinned 54.336 ms, against 64.675 ms in v1.1.0. The kernel did not change (11.377 against 11.380 ms), so the copy changed. Cause not measured.
- v1.2.0 sweep: 12 windows, 0 off lock; 3 windows had no samples after the 1 s skip.
- The gate for the v1.2.0 ncu run had one sample with msedgewebview2 at 10.45% CPU. The average check passed. ncu counts are not timing, so the run stays.
- `filter_noisy_run1.csv` was made with 3.4 GB RAM free. Do not use it.

## Profiling

- H2D copy speed: 26.4 GB/s for the filter and 22.2 GB/s for the group-by (nsys, session 1, stock clocks). Cause of the gap not measured.
- ncu reports the DRAM clock as 8.99 GHz (session 1, stock) and 11.99 GHz (locked). The DRAM % values of the two sessions are not directly comparable.
- In the `q1_v2` ncu run the DRAM clock was 8.99 GHz, not 11.99 GHz as in the `q1_v1` run. So its 34.24% DRAM uses a lower peak. Compare GB/s (98.54 against 17.39). Cause not measured; maybe the 11.6 ms kernel is too short to keep the memory at P0 under ncu replay.
- `q1_v2` occupancy is 33%, limited by 96 registers. ncu says the kernel is latency bound now.
- `q1_v3` uses 128 registers and 776 bytes of local memory per thread (launch bounds). Occupancy 26.94% achieved, 33.33% theoretical. ncu says Compute (SM) is 83.06%, so more bytes saved would not help much now.
- ncu changes the clocks unless `--clock-control none` is set. Its default base clock gave 1.44 GHz (superseded run in docs/profiling.md).

## Limits

- Group-by needs `expected_groups`. Too low raises an error, too high wastes memory (2x too high: 1.93x slower, stock run). Output is in hash order.
- Q1: group ids are fixed to the 6 TPC-H pairs. The largest int64 sum is at 0.737 of 2^63 at SF10.
- Q1 narrow: qty must be whole units 0 to 255, disc and tax 0 to 255 hundredths, ship 1992-01-02 plus 0 to 65,535 days, price below 2^31 cents. TPC-H dbgen keeps these at every SF; other data can fail `narrow()`.
- Sanitizers for Q1: memcheck ran on the full suite (200 tests) for v1.2.0. racecheck and synccheck ran on the 37 Q1 tests only.
- racecheck does not find races between shared atomics and plain shared reads (barrier 2 removed: 3947 of 4096 counted, 0 hazards). The pandas tests are the real check.
- `exclusive_scan` checks only the total, not partial sums with negative values.
- The memory check counts the table, flags and scan positions, not inputs or outputs. About 2^27 groups fit in 8 GB.
- Polars #E69F00 against GPU #76b900 fails the protan check (delta E 2.5); line style and marker are the second encoding.
- Chart titles in `plot.py` are fixed text. Check them after a new sweep.
