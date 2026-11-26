"""Exclusive prefix scan (sum) of an int32 array, our own kernels.

Work-efficient (Blelloch) scan, as in GPU Gems 3, chapter 39, "Parallel Prefix
Sum (Scan) with CUDA" (Harris, Sengupta, Owens), Example 39-2, and the scan
sample in NVIDIA/cuda-samples. Bank-conflict avoidance (section 39.2.3) is left
out to keep the code readable.
"""
import cupy as cp

THREADS = 256
ELEMS = 2 * THREADS  # each thread loads and writes two elements

_SOURCE = r'''
#define ELEMS 512

// Step 1: scan one block of 512 elements in shared memory.
extern "C" __global__ void scan_blocks(const int* in, int* out, int* block_sums, int n) {
  __shared__ int temp[ELEMS];
  int tid = threadIdx.x;
  int a = blockIdx.x * ELEMS + 2 * tid;
  int b = a + 1;
  temp[2 * tid]     = (a < n) ? in[a] : 0;
  temp[2 * tid + 1] = (b < n) ? in[b] : 0;

  // Up-sweep: build partial sums in a tree; temp[ELEMS - 1] ends as the block total.
  int offset = 1;
  for (int d = ELEMS >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset <<= 1;
  }

  // Save the block total for step 2, then clear it to start the down-sweep.
  if (tid == 0) {
    block_sums[blockIdx.x] = temp[ELEMS - 1];
    temp[ELEMS - 1] = 0;
  }

  // Down-sweep: walk the tree back down; each node passes its sum to its right child.
  for (int d = 1; d < ELEMS; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  // Not needed today: the last down-sweep step touches only this thread's own pair.
  // Kept (as in GPU Gems 3) so a change to how threads map to elements stays safe.
  __syncthreads();

  if (a < n) out[a] = temp[2 * tid];
  if (b < n) out[b] = temp[2 * tid + 1];
}

// Step 3: add the scanned total of all earlier blocks to every element of this block.
extern "C" __global__ void add_offsets(int* out, const int* block_offsets, int n) {
  int a = blockIdx.x * ELEMS + 2 * threadIdx.x;
  int add = block_offsets[blockIdx.x];
  if (a < n) out[a] += add;
  if (a + 1 < n) out[a + 1] += add;
}
'''

_module = cp.RawModule(code=_SOURCE)
_scan_blocks = _module.get_function('scan_blocks')
_add_offsets = _module.get_function('add_offsets')


def exclusive_scan(x):
    """Return out where out[i] = x[0] + ... + x[i-1], and out[0] = 0."""
    if not isinstance(x, cp.ndarray) or x.dtype != cp.int32 or x.ndim != 1:
        raise TypeError("x must be a 1-D cupy int32 array")
    x = cp.ascontiguousarray(x)
    n = x.size
    out = cp.empty(n, dtype=cp.int32)
    if n == 0:
        return out

    blocks = (n + ELEMS - 1) // ELEMS
    block_sums = cp.empty(blocks, dtype=cp.int32)
    _scan_blocks((blocks,), (THREADS,), (x, out, block_sums, cp.int32(n)))

    if blocks > 1:
        # Step 2: scan the block totals with this same function (recursion).
        block_offsets = exclusive_scan(block_sums)
        _add_offsets((blocks,), (THREADS,), (out, block_offsets, cp.int32(n)))
    return out
