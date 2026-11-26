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
- `engine/groupby.py`: hash group-by, SUM (int64) and COUNT per int32 key. Linear probing, atomicCAS, compact with own scan. 42 tests vs pandas. memcheck 0 errors, racecheck 0 hazards.
- 2026-11-26: `benchmarks/bench_groupby.py` writes `results/groupby.csv` (1M to 100M rows; 10, 1K, 1M groups; uniform, Zipf). Every engine is checked against pandas first. 5.7 GB RAM free.
- Group-by `strategy="block"`: shared-memory table per block (1024 slots, 4096 rows per block), flush once per key. 85 group-by tests (both strategies, plus a fallback case). memcheck and synccheck 0 errors, racecheck 0 hazards.
- Group-by e2e pinned beats Polars at 1K and 1M groups at every size (about 10x at 1M groups, 100M rows). It loses at 10 groups at every size (123 vs 89 ms at 100M).
- 2026-11-26: `bench_groupby` runs both strategies; `groupby.csv` has a `strategy` column (`-` for CPU rows). 11.9 GB RAM free.
- Block vs global kernel_only: 10 groups 5.4 to 7.4x faster, 1K groups 1.3 to 1.8x faster, 1M groups uniform about equal (5.6% slower at 100M), 1M groups Zipf 2.8 to 3.1x faster.
- Block e2e pinned now beats Polars at 10 groups: 5.1 vs 10.4 ms at 10M, 43.1 vs 88.4 ms at 100M (uniform), about 2x.
- `benchmarks/plot.py` writes 4 PNGs to `docs/figures/` (filter crossover, filter time split, group-by by groups, block speedup).
- Filter crossover points (10%, 50M, 100M pandas, 200M) rerun 3x; medians in `filter.csv`. Charts: new GPU color, no legends on line charts, end labels never overlap.
- Reran suspect CPU points 3x each: none changed more than 20%, so `groupby.csv` stays.
- /reflect: 4 new rules in CLAUDE.md (int64 sums in every engine, check cited paths with gh api, run new tests red first, racecheck and synccheck for kernels with barriers).

## Next

- Choose the strategy automatically from expected_groups, or keep it a caller option.
- README: where and why the GPU wins or loses.
