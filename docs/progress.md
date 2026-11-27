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
- Filter crossover points (10%, 50M, 100M pandas, 200M) rerun 3x; medians in `filter.csv`. Charts: GPU NVIDIA green, CPU greys, no legends on line charts, end labels never overlap.
- Reran suspect CPU points 3x each: none changed more than 20%, so `groupby.csv` stays.
- /reflect: 4 new rules in CLAUDE.md (int64 sums in every engine, check cited paths with gh api, run new tests red first, racecheck and synccheck for kernels with barriers).

- 2026-11-26: fixed all 5 code-review findings in `engine/` (input checks, GPU memory check, constants set once, scan overflow check), each with a test that failed first. 163 tests pass; memcheck 0 errors.

- 2026-11-26: README.md and MIT LICENSE added. Both sweeps rerun in Best performance on AC (11.4 GB RAM free), commit e95f180. Polars now wins the filter end to end from 10M rows; group-by results hold.
- /reflect: 3 new rules in CLAUDE.md (power mode and busy-program check, CPU-use command, check chart titles and README after a sweep).

- 2026-11-26: found that Balanced mode ran the CPU at about 800 to 900 MHz and made Polars up to 2x slower. Both sweeps rerun in Best performance on AC; the Balanced CSVs are in git history.
- Stress tests: Polars 1K uniform 100M rerun 3x (114 to 545 ms, two modes); pinned fill 21.2 ms at 100M; first group-by call 3.7 s (cold start); 200M check used 3.17 GiB of GPU memory.
- README rewritten with the Best performance numbers; two stale plot.py titles fixed (commit 03117d0). known-issues updated to the new CSVs (commit 02bf138).
- 2026-11-27 (Best performance on AC, 10.9 GB RAM free): reran global 10M 1K uniform 3x. kernel_only 2.00 ms, e2e_pinned 5.26 ms (median run) replace the 8.29 / 5.54 ms outlier in `groupby.csv`. Charts use 100M only, so no PNG changed.
- CLAUDE.md: new rule, check power mode and 5% CPU per process before any benchmark run, and record the power mode.
- 2026-11-27: charts use pandas #0072B2, Polars #E69F00, DuckDB red #D62728, GPU #76b900. GPU lines solid 2.5, CPU dashed 1.5; hollow markers for GPU pageable and global. README References adds NVIDIA/cudf and anilshanbhag/crystal.
- /reflect: 2 new chart rules in CLAUDE.md (update decisions.md with style changes, check same-colour lines still differ); PROJECT_NOTES cudf path now NVIDIA/cudf.

## Next

- Choose the strategy automatically from expected_groups, or keep it a caller option.
