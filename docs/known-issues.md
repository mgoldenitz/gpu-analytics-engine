# Known issues

- Polars beats GPU end to end at 1% selectivity from 50M rows, also with pinned memory. The host-to-device copy (about 22 ms at 100M rows) costs more than the Polars filter (12 ms).
- Pageable end to end never beats Polars. The copies take about 90% of the time.
- Some benchmark points are still not consistent, for example `cupy_builtin` kernel_only 13.77 ms at 10M and 90%, but about 1 ms at other selectivities.
- The time to fill the pinned buffer from pageable memory is not measured.
- The 8 GB GPU check uses estimates of bytes per row, not measured memory.
- Many 1M-row wins are very close (about 0.4 ms for both sides at 1%).
- `filter_noisy_run1.csv` was made with only 3.4 GB RAM free. Do not use it for the chart.
- `CLAUDE.md` line 9 still says that only `check_gpu.py` exists.
