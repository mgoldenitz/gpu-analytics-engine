# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A GPU analytics engine: hand-written CUDA kernels (filter, hash group-by, hash join as a stretch goal) run through CuPy `RawKernel`, benchmarked against pandas, Polars and DuckDB. Target hardware is an RTX 5060 (sm_120) on Windows 11. Read `PROJECT_NOTES.md` for scope, stop rule, datasets and reference repos before starting new work.

Layout:

- `engine/`: the kernels. `filter.py` (flag, scan, scatter; `scan="own"` or `"cub"`) and `scan.py` (own Blelloch exclusive scan).
- `tests/`: pytest checks of each kernel against pandas or NumPy.
- `benchmarks/`: `bench_filter.py` (timing runs), `datagen.py` (seeded data in memory), `summary.py` (prints the summary from the CSV), and `results/` (CSV output).
- `docs/`: `progress.md`, `decisions.md`, `known-issues.md`.

## Rules

- Data lives in `C:\data\gpu-analytics-engine` and never in the repo. Do not write `.parquet` or generated datasets inside the project folder (it is in OneDrive).
- Every kernel gets a pytest that checks its results against pandas. Write the test first and run it once before the kernel exists. It must fail.
- Benchmarks report kernel_only, e2e_pageable and e2e_pinned times, against pandas, Polars and DuckDB. Every engine must sum int32 as int64: Polars needs `.cast(pl.Int64)`, DuckDB needs `::BIGINT`.
- Before a full benchmark run, check free RAM (Get-CimInstance Win32_OperatingSystem), that the power overlay is Best performance on AC (ActiveOverlayAcPowerScheme = ded574b5-45a0-4f42-8737-46345c09c238 under HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes), and that no other program (for example a video render) is busy. If RAM is below 5 GB or a check fails, stop and ask the owner.
- Before any benchmark run, check that Windows is in Best performance on AC power and that no other process uses more than 5% CPU. Stop if either check fails. Record the power mode in the run notes.
- To check CPU use: (Get-Counter '\Process(*)\% Processor Time' -SampleInterval 3).CounterSamples, divide by 20 (logical processors), and ignore Idle, _Total and the WMI host. The limit is 5% of the whole CPU.
- After a new sweep, run plot.py and compare each fixed chart title and each README number with the new CSV.
- Crossover point = the smallest n from which the GPU wins at that size and at all larger sizes, not the first n where it wins once.
- Explain every kernel line by line before moving on.
- Write our own kernels. Read and cite the reference repos in `PROJECT_NOTES.md` (cuda-samples, CCCL/CUB, cuCollections, cudf), do not copy them. Before you write a cited path, make sure that it exists: `gh api repos/<owner>/<repo>/contents/<path>`.
- cuDF/RAPIDS is last and runs only in WSL2 with a CUDA 13.4 toolkit. Do not add it to the Windows venv.
- Minimum shippable version: filter and group-by kernels, the three CPU baselines, a crossover chart (data size where the GPU starts to win), and a README that explains where and why the GPU wins or loses.

## Commands

Run from the project root in PowerShell.

```
C:\venvs\gpu-analytics-engine\Scripts\Activate.ps1
python check_gpu.py        # must print the GPU name, sm_120 and True before any other work
pytest                     # all tests
pytest path\to\test_file.py::test_name   # one test
```

The venv is at `C:\venvs\gpu-analytics-engine`, outside OneDrive. Installed: cupy-cuda13x 14.2.0, numpy, pandas, polars, duckdb, pyarrow, pytest, matplotlib. The venv is Python 3.14.7.

Memory check for kernels. compute-sanitizer is not on PATH, so use the full path. CuPy's memory pool rounds each allocation up to 512 bytes, and memcheck cannot see a small overrun inside that rounding, so turn the pool off:

```
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\compute-sanitizer\compute-sanitizer.exe" --tool memcheck python -c "import cupy, pytest, sys; cupy.cuda.set_allocator(None); sys.exit(pytest.main(['tests', '-q']))"
```

The last line must be `ERROR SUMMARY: 0 errors`.

For kernels with `__syncthreads()`, also run `--tool racecheck` and `--tool synccheck` with the same command. racecheck does not find races between shared atomics and plain shared reads (removing a barrier gave 0 hazards but wrong counts), so the pandas tests are the real check.

TPC-H data comes from DuckDB: `CALL dbgen(sf=1)` (about 6M lineitem rows). Load only the columns a query needs.

## End of session

Update the three docs in `docs/`, as short bullet points (no more than a screen each):

- `docs/progress.md`: what is done and what is next.
- `docs/decisions.md`: each design choice and its reason.
- `docs/known-issues.md`: open problems and limits of the results.

## Repo hooks

`.claude/settings.json` runs two hooks on every Edit/Write: `check-line-limit.mjs` (300-line limit per file) and `check-secrets.mjs` (blocks keys in changes). `git push` asks first.
