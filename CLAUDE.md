# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A GPU analytics engine: hand-written CUDA kernels (filter, hash group-by, hash join as a stretch goal) run through CuPy `RawKernel`, benchmarked against pandas, Polars and DuckDB. Target hardware is an RTX 5060 (sm_120) on Windows 11. Read `PROJECT_NOTES.md` for scope, stop rule, datasets and reference repos before starting new work.

The project is at its start. Only `check_gpu.py` exists as code. No package layout, tests or benchmarks are written yet.

## Rules

- Data lives in `C:\data\gpu-analytics-engine` and never in the repo. Do not write `.parquet` or generated datasets inside the project folder (it is in OneDrive).
- Every kernel gets a pytest that checks its results against pandas.
- Benchmarks report times with and without host-to-device transfer, against pandas, Polars and DuckDB.
- Explain every kernel line by line before moving on.
- Write our own kernels. Read and cite the reference repos in `PROJECT_NOTES.md` (cuda-samples, CCCL/CUB, cuCollections, cudf), do not copy them.
- cuDF/RAPIDS is last and runs only in WSL2 with a CUDA 13.4 toolkit. Do not add it to the Windows venv.
- Minimum shippable version: filter and group-by kernels, the three CPU baselines, a crossover chart (data size where the GPU starts to win), and a README that explains where and why the GPU wins or loses.

## Commands

Run from the project root in PowerShell.

```
.venv\Scripts\Activate.ps1
python check_gpu.py        # must print the GPU name, sm_120 and True before any other work
pytest                     # all tests
pytest path\to\test_file.py::test_name   # one test
```

Installed in `.venv`: cupy-cuda13x 14.2.0, numpy, pandas, polars, duckdb, pyarrow, pytest, matplotlib. The venv is Python 3.14.7.

TPC-H data comes from DuckDB: `CALL dbgen(sf=1)` (about 6M lineitem rows). Load only the columns a query needs.

## Repo hooks

`.claude/settings.json` runs two hooks on every Edit/Write: `check-line-limit.mjs` (300-line limit per file) and `check-secrets.mjs` (blocks keys in changes). `git push` asks first.
