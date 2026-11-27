"""Keep the int32 values below a threshold, on the GPU (stream compaction)."""
import operator

import cupy as cp

from engine.scan import exclusive_scan

THREADS = 256

# Step 1, predicate: flags[i] = 1 if row i passes, else 0.
_flag = cp.RawKernel(r'''
extern "C" __global__ void flag_lt(const int* x, int* flags, int threshold, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flags[i] = x[i] < threshold;
}''', 'flag_lt')

# Step 3, scatter: each passing row writes itself to its slot in the output.
# pos is an exclusive scan (shift = 0) or an inclusive scan (shift = 1, slot = pos[i] - 1).
_scatter = cp.RawKernel(r'''
extern "C" __global__ void scatter(const int* x, const int* flags, const int* pos,
                                   int* out, int shift, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && flags[i]) out[pos[i] - shift] = x[i];
}''', 'scatter')


def filter_less_than(x, threshold, scan="own"):
    """Return the values of x below threshold, in their original order.

    scan="own" uses our scan kernel (engine/scan.py); scan="cub" uses CuPy's
    cumsum, which runs NVIDIA CUB. Both stay so the benchmarks can compare them.
    """
    if not isinstance(x, cp.ndarray) or x.dtype != cp.int32 or x.ndim != 1:
        raise TypeError("x must be a 1-D cupy int32 array")
    if scan not in ("own", "cub"):
        raise ValueError('scan must be "own" or "cub"')
    threshold = operator.index(threshold)  # TypeError for 2.5: cp.int32(2.5) would cut it to 2
    if not -2**31 <= threshold < 2**31:
        raise ValueError("threshold must fit in int32")
    x = cp.ascontiguousarray(x)
    n = x.size
    if n == 0:
        return cp.empty(0, dtype=cp.int32)

    grid = ((n + THREADS - 1) // THREADS,)
    flags = cp.empty(n, dtype=cp.int32)
    _flag(grid, (THREADS,), (x, flags, cp.int32(threshold), cp.int32(n)))

    # Step 2, prefix scan. int32 positions limit n to 2^31 - 1 rows; 200M rows fits.
    if scan == "own":
        pos = exclusive_scan(flags)  # passing rows before i
        shift = 0
        count = int(pos[-1] + flags[-1])
    else:
        pos = cp.cumsum(flags, dtype=cp.int32)  # passing rows up to and including i
        shift = 1  # no extra subtraction pass; the scatter subtracts 1
        count = int(pos[-1])

    out = cp.empty(count, dtype=cp.int32)
    _scatter(grid, (THREADS,), (x, flags, pos, out, cp.int32(shift), cp.int32(n)))
    return out
