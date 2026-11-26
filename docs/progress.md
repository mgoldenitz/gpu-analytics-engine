# Progress

## Done

- 2026-11-25: project setup, `check_gpu.py` prints RTX 5060 Laptop GPU, sm_120, True.
- Venv at `C:\venvs\gpu-analytics-engine` (Python 3.14.7), outside OneDrive.
- `engine/filter.py`: filter `value < threshold` (flag, prefix scan, scatter), `scan="own"` or `scan="cub"`.
- `engine/scan.py`: own Blelloch exclusive scan, 512 elements per block, recursive over block totals.
- Tests: 70 pass (60 filter, 10 scan). memcheck with the pool off: 0 errors. racecheck on scan: 0 hazards.
- 2026-11-26: `benchmarks/bench_filter.py`, `datagen.py`, `summary.py`. Modes kernel_only, e2e_pageable, e2e_pinned.
- Results in `benchmarks/results/filter.csv` (clean run, 5.8 GB RAM free) and `filter_noisy_run1.csv` (old run).
- Pinned: GPU end to end beats pandas and DuckDB from 1M rows, and Polars at 10, 50 and 90% selectivity.

## Next

- Crossover chart from `filter.csv`.
- Hash group-by kernel with pandas tests.
- README: where and why the GPU wins or loses.
