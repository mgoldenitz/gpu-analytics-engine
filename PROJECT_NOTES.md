# GPU analytics engine: project notes

Saved 2026-11-25 from the project planning thread. Read this before starting.

## Why this project

A council review ranked this first of three CUDA projects (analytics engine, Monte Carlo, MRI pipeline) for overall tech employability. It targets data engineer, analytics engineer, data platform engineer and data scientist roles, and it extends existing SQL work. CUDA here is a tiebreaker, not the main hiring signal: 0 of 15 Toronto data engineer listings checked mentioned CUDA or GPU.

## Scope and stop rule

- Minimum shippable version: filter and hash group-by kernels; benchmarks against pandas, Polars and DuckDB with host-to-device transfer counted; a crossover chart showing the data size where the GPU starts to win; a README explaining where and why the GPU wins or loses.
- Hash join is a stretch goal. cuDF goes last, in WSL2 with a CUDA 13.4 toolkit (RAPIDS lists 13.0 to 13.4; this machine has 13.5).
- Publish the minimum version by week 4. Stop at week 8 whatever its state, ship what exists, then spend the next block on dbt and Airflow for the ClinicalTrials.gov pipeline.
- Possible second CUDA project afterwards: a PyTorch custom CUDA operator.

## Setup (Windows 11, VS Code terminal, local Claude Code CLI)

1. VS Code extensions: Python, C/C++, Nsight Visual Studio Code Edition, Claude Code.
2. Visual Studio 2022 Build Tools with "Desktop development with C++" (needed by nvcc for .cu files later).
3. Repo here; data in C:\data\gpu-analytics-engine (outside OneDrive).
4. Python 3.14 venv:
   ```
   py -3.14 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install cupy-cuda13x numpy pandas polars duckdb pyarrow pytest matplotlib
   ```
5. Run check_gpu.py (CuPy RawKernel add-one test) first. It must print the RTX 5060 and True before anything else.
6. .gitignore: `.venv/`, `__pycache__/`, `*.parquet`.
7. Start `claude`, run `/init`, and add the rules below to CLAUDE.md.

## CLAUDE.md rules

- Data lives in C:\data\gpu-analytics-engine and never in the repo.
- Every kernel gets a pytest that checks its results against pandas.
- Benchmarks report times with and without host-to-device transfer, against pandas, Polars and DuckDB.
- Explain every kernel line by line before moving on.

## Skills and tools

Superpowers (brainstorming, writing-plans, test-driven-development, systematic-debugging, verification-before-completion), /code-review, /simplify, engineering:testing-strategy, engineering:documentation, dataviz or data:create-viz, github-portfolio-builder, context7, GitHub connector. Skip treg, agent swarms and the GitHub topic lists.

## Datasets

1. Seeded synthetic NumPy columns, 1M to 200M rows. Control filter selectivity (1%, 10%, 50%, 90%), group-by key cardinality (10, 1,000, 1M) and key skew. 100M int32 is about 400 MB.
2. TPC-H from DuckDB: `CALL dbgen(sf=1)`. Q6 (filter + sum) and Q1 (filter + group-by). SF1 is about 6M lineitem rows, SF10 about 60M. Load only needed columns.
3. FAERS SQLite (10.8M cases) as the real-data case, e.g. reaction counts by drug. Expect CPU engines to win at this size and say so.

## Reference repos (read and cite; write your own kernels)

- NVIDIA/cuda-samples: reductions, prefix scan, stream compaction (the filter kernel).
- NVIDIA/cccl (CUB, Thrust): device-wide reduce and scan; a "my kernel vs NVIDIA's" baseline.
- NVIDIA/cuCollections: GPU hash maps and collision handling for group-by and join.
- rapidsai/cudf: production group-by and join code, read after yours works.
- anilshanbhag/crystal: SIGMOD 2020 GPU vs CPU database analytics engine and paper. Not verified; confirm the repo exists.

## Known facts checked

- cupy-cuda13x 14.2.0 ships native Windows wheels; sm_120 not named on the pages read, so the GPU check comes first.
- cuDF/RAPIDS runs on Windows only through WSL.
- A council report and transcript are in the project files under council/.
