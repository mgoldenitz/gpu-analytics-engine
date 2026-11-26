"""Check that CuPy can compile and run a CUDA kernel on this GPU."""
import cupy as cp

k = cp.RawKernel(r'''
extern "C" __global__ void add1(const int* x, int* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1;
}''', 'add1')

n = 10_000_000
x = cp.arange(n, dtype=cp.int32)
y = cp.empty_like(x)
k(((n + 255) // 256,), (256,), (x, y, cp.int32(n)))
props = cp.cuda.runtime.getDeviceProperties(0)
print(props['name'].decode() if isinstance(props['name'], bytes) else props['name'],
      f"sm_{props['major']}{props['minor']}", bool((y == x + 1).all()))
