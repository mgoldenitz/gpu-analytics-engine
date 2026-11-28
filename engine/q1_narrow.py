"""TPC-H Q1 kernel q1_v3: narrow columns (10 bytes per row) and 4-row vector loads.

The same plan as q1_v2 in engine/q1.py (36 register totals per thread, warp shuffle, one
shared add per warp, one global add per block). Two changes:
  1. Narrow types (benchmarks/tpch_data.narrow): price int32 cents, qty uint8 whole units,
     disc and tax uint8 hundredths, ship uint16 days since SHIP_BASE, gid uint8.
  2. Each thread loads 4 consecutive rows at once: int4 (16 bytes) for price, uint2
     (8 bytes) for ship, one u32 (4 bytes) for each 1-byte column. The last n % 4 rows
     go through a plain one-row loop.
sum_qty leaves the kernel in hundredths (x 100 in the block flush), so engine.q1.q1_rows
reads the result the same way as for v1 and v2.
"""
import cupy as cp
import numpy as np

from engine.q1 import AGGS, GROUPS, THREADS

VEC = 4  # rows per vector load
DTYPES = {"price": np.int32, "qty": np.uint8, "disc": np.uint8, "tax": np.uint8,
          "ship": np.uint16, "gid": np.uint8}
ALIGN = {"price": 16, "ship": 8, "qty": 4, "disc": 4, "tax": 4, "gid": 4}  # bytes per load

_SOURCE = r'''
#define SLOTS 36
#define GROUPS 6
#define AGGS 6

__device__ __forceinline__
void add_row(unsigned long long (&r)[SLOTS], int price, unsigned int qty, unsigned int disc,
             unsigned int tax, unsigned int ship, unsigned int g, int cutoff)
{
    if ((int)ship > cutoff) return;
    unsigned long long p = (unsigned long long)price;
    unsigned long long dp = p * (100u - disc);
    unsigned long long v[AGGS] = {1ULL, qty, p, dp, dp * (100u + tax), disc};
    #pragma unroll
    for (int k = 0; k < GROUPS; k++) {
        unsigned long long m = (g == k) ? ~0ULL : 0ULL;
        #pragma unroll
        for (int a = 0; a < AGGS; a++) r[k * AGGS + a] += v[a] & m;
    }
}

// 2 blocks per SM, as q1_v2. Without it: 156 registers, 1 block per SM, 5.71 ms at SF10.
// With it: 128 registers and 776 B spill, but 4.54 ms (scratch timing, 2026-11-28).
extern "C" __global__
__launch_bounds__(256, 2)
void q1_v3(const int* price, const unsigned char* qty, const unsigned char* disc,
           const unsigned char* tax, const unsigned short* ship, const unsigned char* gid,
           long long n, int cutoff, unsigned long long* out)
{
    __shared__ unsigned long long acc[SLOTS];
    for (int s = threadIdx.x; s < SLOTS; s += blockDim.x) acc[s] = 0;
    __syncthreads();

    unsigned long long r[SLOTS];
    #pragma unroll
    for (int s = 0; s < SLOTS; s++) r[s] = 0;

    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    long long n4 = n / 4;  // whole groups of 4 rows
    for (long long i = tid; i < n4; i += stride) {
        int4 p = reinterpret_cast<const int4*>(price)[i];         // 16 bytes: 4 prices
        unsigned int q = reinterpret_cast<const unsigned int*>(qty)[i];   // 4 bytes: 4 rows
        unsigned int d = reinterpret_cast<const unsigned int*>(disc)[i];
        unsigned int t = reinterpret_cast<const unsigned int*>(tax)[i];
        unsigned int g = reinterpret_cast<const unsigned int*>(gid)[i];
        uint2 s = reinterpret_cast<const uint2*>(ship)[i];          // 8 bytes: 4 ship days
        #pragma unroll 1  // one row at a time, so the 4 rows do not each keep their values live
        for (int j = 0; j < 4; j++) {
            int pj = j == 0 ? p.x : j == 1 ? p.y : j == 2 ? p.z : p.w;
            unsigned int sj = ((j < 2 ? s.x : s.y) >> (16 * (j & 1))) & 0xffffu;
            add_row(r, pj, (q >> (8 * j)) & 0xffu, (d >> (8 * j)) & 0xffu,
                    (t >> (8 * j)) & 0xffu, sj, (g >> (8 * j)) & 0xffu, cutoff);
        }
    }
    for (long long i = n4 * 4 + tid; i < n; i += stride)  // the last n % 4 rows
        add_row(r, price[i], qty[i], disc[i], tax[i], ship[i], gid[i], cutoff);

    #pragma unroll
    for (int s = 0; s < SLOTS; s++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) r[s] += __shfl_down_sync(0xffffffffu, r[s], off);
    }
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (int s = 0; s < SLOTS; s++) if (r[s] != 0) atomicAdd(acc + s, r[s]);
    }
    __syncthreads();

    for (int s = threadIdx.x; s < SLOTS; s += blockDim.x)  // slot 1 of each group is sum_qty
        if (acc[s] != 0) atomicAdd(out + s, (s % AGGS == 1) ? acc[s] * 100ULL : acc[s]);
}
'''

_kernel = cp.RawModule(code=_SOURCE).get_function("q1_v3")
_SMS = cp.cuda.Device().attributes["MultiProcessorCount"]


def v3_blocks(n):
    """SMs x blocks per SM (occupancy API), never more blocks than 256-thread groups of 4 rows."""
    per_sm = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(_kernel.kernel.ptr,
                                                                      THREADS, 0)
    return max(1, min(_SMS * per_sm, (n // VEC + THREADS - 1) // THREADS))


def q1_v3_sums(price, qty, disc, tax, ship, gid, cutoff, out=None):
    """Launch q1_v3. cutoff is in narrow ship units. Returns the (6 x 6) uint64 sums on the GPU."""
    cols = {"price": price, "qty": qty, "disc": disc, "tax": tax, "ship": ship, "gid": gid}
    n = price.size
    for name, a in cols.items():
        if not isinstance(a, cp.ndarray) or a.ndim != 1 or a.dtype != DTYPES[name]:
            raise TypeError(f"{name} must be a 1-D cupy {np.dtype(DTYPES[name]).name} array")
        if a.size != n:
            raise ValueError("all columns must have the same length")
        if a.data.ptr % ALIGN[name] or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous and {ALIGN[name]}-byte aligned")
    if out is None:
        out = cp.zeros(GROUPS * AGGS, dtype=cp.uint64)
    else:
        out.fill(0)
    if n:
        _kernel((v3_blocks(n),), (THREADS,), (price, qty, disc, tax, ship, gid,
                                              np.int64(n), np.int32(cutoff), out))
    return out.reshape(GROUPS, AGGS)


def kernel_attributes():
    """Registers, local (spill) bytes and blocks per SM of q1_v3, for the docs."""
    a = _kernel.attributes
    per_sm = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(_kernel.kernel.ptr,
                                                                      THREADS, 0)
    return {"num_regs": a["num_regs"], "local_size_bytes": a["local_size_bytes"],
            "blocks_per_sm": per_sm}
