# Decisions

- Stream compaction (flag, exclusive scan, scatter) for the filter. Output stays in input order, so it compares directly with pandas.
- Own scan is Blelloch (GPU Gems 3 ch. 39), 256 threads with 512 elements per block. No bank-conflict avoidance, to keep it readable.
- Keep `scan="cub"` (`cp.cumsum`) next to `scan="own"`, so the benchmarks compare our scan with NVIDIA's.
- The `cub` path uses the inclusive cumsum directly. The scatter subtracts 1 (`shift`), so there is no extra pass over memory.
- int32 positions: limit of 2^31 - 1 rows. 200M rows fits.
- memcheck runs with the CuPy pool off (`set_allocator(None)`), because the pool rounds allocations and hides small overruns.
- pytest cache is off (`-p no:cacheprovider`), because OneDrive locks the cache folder.
- Benchmark data is made in memory with a seed. Nothing goes to disk or into the repo.
- Timing: 3 warmup runs, then the median of 10, with `Device.synchronize` before and after each run.
- Pinned mode fills the pinned input buffer before the timer starts, as a pinned data loader would give it.
- Crossover = smallest n from which the GPU wins at that size and at all larger sizes.
