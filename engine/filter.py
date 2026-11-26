"""Keep the int32 values below a threshold, on the GPU (stream compaction)."""
import cupy as cp

THREADS = 256

# Step 1, predicate: flags[i] = 1 if row i passes, else 0.
_flag = cp.RawKernel(r'''
extern "C" __global__ void flag_lt(const int* x, int* flags, int threshold, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flags[i] = x[i] < threshold;
}''', 'flag_lt')

# Step 3, scatter: each passing row writes itself to its slot in the output.
_scatter = cp.RawKernel(r'''
extern "C" __global__ void scatter(const int* x, const int* flags, const int* pos,
                                   int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && flags[i]) out[pos[i]] = x[i];
}''', 'scatter')


def filter_less_than(x, threshold):
    """Return the values of x below threshold, in their original order."""
    if not isinstance(x, cp.ndarray) or x.dtype != cp.int32 or x.ndim != 1:
        raise TypeError("x must be a 1-D cupy int32 array")
    x = cp.ascontiguousarray(x)
    n = x.size
    if n == 0:
        return cp.empty(0, dtype=cp.int32)

    grid = ((n + THREADS - 1) // THREADS,)
    flags = cp.empty(n, dtype=cp.int32)
    _flag(grid, (THREADS,), (x, flags, cp.int32(threshold), cp.int32(n)))

    # Step 2, exclusive prefix scan: pos[i] = number of passing rows before i.
    # shortcut: the scan is CuPy's cumsum (CUB); replace with our own scan kernel later.
    # int32 positions limit n to 2^31 - 1 rows; 200M rows fits.
    pos = cp.cumsum(flags, dtype=cp.int32) - flags
    count = int(pos[-1] + flags[-1])

    out = cp.empty(count, dtype=cp.int32)
    _scatter(grid, (THREADS,), (x, flags, pos, out, cp.int32(n)))
    return out
