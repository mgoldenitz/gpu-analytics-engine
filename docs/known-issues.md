# Known issues

- Polars beats GPU end to end at 1% selectivity from 50M rows, also with pinned memory. The host-to-device copy (about 22 ms at 100M rows) costs more than the Polars filter (12 ms).
- Pageable end to end never beats Polars. The copies take about 90% of the time.
- Some benchmark points are still not consistent, for example `cupy_builtin` kernel_only 13.77 ms at 10M and 90%, but about 1 ms at other selectivities.
- The time to fill the pinned buffer from pageable memory is not measured.
- The 8 GB GPU check uses estimates of bytes per row, not measured memory.
- Many 1M-row wins are very close (about 0.4 ms for both sides at 1%).
- `filter_noisy_run1.csv` was made with only 3.4 GB RAM free. Do not use it for the chart.- Group-by with 1 or 10 groups: all threads atomicAdd to the same few slots, so they run one after another. Not benchmarked yet. A fix is per-block pre-aggregation in shared memory.
- Group-by needs the caller to give expected_groups. Too low gives an error, too high wastes memory.
- Group-by output is in slot (hash) order, not key order.
- Group-by run is noisy in some points. Polars 1K groups: 261 ms at 50M but 109 ms at 100M uniform, and 717 ms at 100M Zipf. GPU kernel_only at 10M, 1K groups: 7.7 ms, but 0.5 ms at 1M and 10.9 ms at 50M, and slower than e2e_pinned (5.4 ms). Rerun with nothing else open before the chart.
- Group-by with 1M groups: the table is 2M slots x 16 bytes = 32 MB, about the size of the L2 cache, so the random atomics go to DRAM. That is why 1M groups is slower than 1K groups.
