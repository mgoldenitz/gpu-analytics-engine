# Known issues

- Polars beats GPU end to end at 1% selectivity from 50M rows, also with pinned memory. The host-to-device copy (about 22 ms at 100M rows) costs more than the Polars filter (12 ms).
- Pageable end to end never beats Polars. The copies take about 90% of the time.
- Some benchmark points are still not consistent, for example `cupy_builtin` kernel_only 13.77 ms at 10M and 90%, but about 1 ms at other selectivities.
- The time to fill the pinned buffer from pageable memory is not measured.
- The 8 GB GPU check uses estimates of bytes per row, not measured memory.
- Many 1M-row wins are very close (about 0.4 ms for both sides at 1%).
- `filter_noisy_run1.csv` was made with only 3.4 GB RAM free. Do not use it for the chart.
- Group-by `global` with 10 groups: all threads atomicAdd to the same few slots, so they run one after another (89 ms kernel_only at 100M). `block` fixes it (12 ms).
- Group-by needs the caller to give expected_groups. Too low gives an error, too high wastes memory.
- Group-by output is in slot (hash) order, not key order.
- CPU rerun, 3x each (2026-11-26): Polars 1K Zipf 100M 701/718/711 ms (repeats); Polars 1K 10M 49-55 ms uniform, 63-65 ms Zipf (repeats); DuckDB 1M uniform 100M 4175/6473/8673 ms (noisy, 2x spread; median of runs is within 5% of the CSV).
- Filter rerun at 10%, 3x each (2026-11-26): pandas 100M gave 1676/1501/413 ms (two modes, fast and slow; cause unknown). The CSV now has 1501 ms (was 401). pandas 200M (793 ms) is now the suspect point, as it is faster than 100M. DuckDB 100M repeats (186/186/197 vs 181 ms). DuckDB 50M is noisy (262/363/293 ms), so its dip at 100M is probably 50M noise.
- Chart titles are fixed text. Rerun the benchmarks and the titles can be wrong.
- Group-by rerun (2026-11-26, both strategies): the old GPU 10M 1K-groups point (7.7 ms) is now 1.99 ms, below e2e_pinned. Still suspect: Polars 1K Zipf at 100M is 710 ms vs 106 ms uniform (also 717 ms in the first run, so it repeats; cause unknown). Polars 1K groups at 10M is 48 ms, 4.8 ms per 1M rows, vs 1.1 ms per 1M rows at 100M. DuckDB 1M uniform groups at 100M is 6815 ms, 4x its Zipf time.
- Group-by with 1M groups: the table is 2M slots x 16 bytes = 32 MB, about the size of the L2 cache, so the random atomics go to DRAM. That is why 1M groups is slower than 1K groups.
- racecheck does not find races between shared atomics and plain shared reads. Test: with barrier 2 removed, the count was wrong (3947 of 4096) and racecheck still gave 0 hazards, also with `--racecheck-detect-level info`. The pandas tests are the real check for the block kernel.
- Block strategy with many distinct keys per block (1M uniform groups): most rows miss the shared table and go to global. Measured: 87.7 vs 83.0 ms kernel_only at 100M (5.6% slower). At 10M it is 0.13 ms faster, which is noise. With 1M Zipf groups the hot keys hit the shared table, so block is 3x faster.
